"""BatchNorm / fp16 진단 (교수님 2026-10-25 요청: N=전체 lr 3e-4 에서 '학습은 정상, 평가에서 붕괴'의 원인 확인).

query 는 전혀 쓰지 않는다. 학습 결과/가중치/RNG 에 영향을 주지 않도록 hook 은 읽기만 한다.
cfg["bn_diag"]=True 일 때만 adapter.finetune() 이 사용한다. 진단 오류가 실험을 죽이지 않도록 finish() 는 예외를 삼킨다.

epoch 마다 기록: head BatchNorm running_mean/var 의 NaN/inf/최대/최소, BatchNorm 입력과 encoder 각 block 출력의
NaN/inf 개수와 절댓값 최대(fp16 overflow 점검), 손실이 비정상인 step 수, GradScaler scale 과 건너뛴 step 수.
학습 후: 학습 데이터 일부(최대 2,000개)를 아래 조건으로 평가하고 BatchNorm 입력 분포를 running 통계와 비교.
  eval_fp32      : predict() 와 같은 조건(eval 모드, fp32)
  eval_fp16      : eval 모드, 학습과 같은 정밀도(autocast)
  trainlike_fp16 : 학습처럼 dropout 켠 특징(BatchNorm 만 eval 로 둬서 running 통계는 건드리지 않음)
  recal_*        : BatchNorm running 통계를 eval 특징으로 다시 계산한 뒤 평가(원래 통계는 복원)
"""
import copy
import json
import logging
import math
import traceback
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger("adapters.bn_diag")


def _clean(t):
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


class BNDiag:
    def __init__(self, model, cfg, scaler):
        self.model, self.cfg, self.scaler = model, cfg, scaler
        bn = getattr(model, "norm", None)
        self.bn = bn if isinstance(bn, nn.BatchNorm1d) else None
        try:
            self.blocks = list(model.encoder.transformer_encoder)
        except Exception:
            self.blocks = []
        self.records = []
        self.capture = None
        self.force = False
        self.enabled = True
        self.handles = []
        self._reset_epoch()
        if self.bn is not None:
            self.handles.append(self.bn.register_forward_hook(self._bn_hook))
        for i, blk in enumerate(self.blocks):
            self.handles.append(blk.register_forward_hook(self._blk_hook(i)))
        logger.info(f"[BNDIAG] 활성화: BatchNorm={'있음' if self.bn is not None else '없음'}, encoder block hook {len(self.blocks)}개, "
                    f"tag={cfg.get('bn_diag_tag')}, out={cfg.get('bn_diag_out')}")

    # ---- 누적 ----
    def _reset_epoch(self):
        self.stats = {}
        self.nonfinite_loss = 0
        self.skipped = 0

    def _acc(self, key, val, mode):
        cur = self.stats.get(key)
        if cur is None:
            self.stats[key] = val
        else:
            self.stats[key] = cur + val if mode == "sum" else torch.maximum(cur, val)

    def _active(self):
        return self.force or (self.enabled and self.model.training)

    def _bn_hook(self, mod, inp, out):
        x = inp[0].detach()
        if self.capture is not None:
            self.capture.append(x.float().cpu())
        if self._active():
            xf = x.float()
            fin = torch.isfinite(xf)
            self._acc("bn_in_nonfinite", (~fin).sum(), "sum")
            self._acc("bn_in_absmax", torch.where(fin, xf.abs(), torch.zeros_like(xf)).max(), "max")

    def _blk_hook(self, i):
        def hook(mod, inp, out):
            if not self._active():
                return
            o = out[0] if isinstance(out, (tuple, list)) else out
            of = o.detach().float()
            fin = torch.isfinite(of)
            self._acc(f"blk{i}_nonfinite", (~fin).sum(), "sum")
            self._acc(f"blk{i}_absmax", torch.where(fin, of.abs(), torch.zeros_like(of)).max(), "max")
        return hook

    # ---- finetune() 에서 호출 ----
    def after_step(self, scaler, scale_before):
        try:
            if scale_before is not None and scaler.get_scale() < scale_before:
                self.skipped += 1          # GradScaler 가 inf/NaN gradient 로 step 을 건너뜀
        except Exception:
            pass

    def after_loss(self, loss_value):
        if not math.isfinite(loss_value):
            self.nonfinite_loss += 1

    def _bn_buffers(self):
        rm = self.bn.running_mean.detach().double().cpu()
        rv = self.bn.running_var.detach().double().cpu()
        return {"rm_nan": int(torch.isnan(rm).sum()), "rm_inf": int(torch.isinf(rm).sum()),
                "rm_absmax": float(_clean(rm).abs().max()),
                "rv_nan": int(torch.isnan(rv).sum()), "rv_inf": int(torch.isinf(rv).sum()),
                "rv_min": float(_clean(rv).min()), "rv_max": float(_clean(rv).max()),
                "num_batches": int(self.bn.num_batches_tracked)}

    def end_epoch(self, epoch, mean_loss):
        rec = {"epoch": epoch, "loss": float(mean_loss), "nonfinite_loss_steps": self.nonfinite_loss,
               "skipped_steps": self.skipped, "scaler_scale": float(self.scaler.get_scale()) if self.scaler is not None else None}
        for k, v in self.stats.items():
            rec[k] = float(v)
        if self.bn is not None:
            rec.update(self._bn_buffers())
        nb = len(self.blocks)
        rec["blk_absmax_max"] = max([rec.get(f"blk{i}_absmax", 0.0) for i in range(nb)] or [0.0])
        rec["blk_nonfinite_sum"] = sum(rec.get(f"blk{i}_nonfinite", 0.0) for i in range(nb))
        self.records.append(rec)
        logger.info(
            f"[BNDIAG] epoch {epoch}: loss={rec['loss']:.4f} 비정상loss={rec['nonfinite_loss_steps']} 건너뛴step={rec['skipped_steps']} "
            f"scale={rec['scaler_scale']} | BN입력 nonfinite={rec.get('bn_in_nonfinite', 0):.0f} absmax={rec.get('bn_in_absmax', 0):.3g} "
            f"| block출력 nonfinite합={rec['blk_nonfinite_sum']:.0f} absmax최대={rec['blk_absmax_max']:.3g} "
            f"| running_mean nan/inf={rec.get('rm_nan', '-')}/{rec.get('rm_inf', '-')} absmax={rec.get('rm_absmax', 0):.3g} "
            f"running_var nan/inf={rec.get('rv_nan', '-')}/{rec.get('rv_inf', '-')} min={rec.get('rv_min', 0):.3g} max={rec.get('rv_max', 0):.3g}")
        self._reset_epoch()

    # ---- 학습 후 진단 ----
    def finish(self, train_loader, enc_eval, n_unfrozen, amp, autocast_dtype, device, max_batches=250):
        try:
            self._finish(train_loader, enc_eval, n_unfrozen, amp, autocast_dtype, device, max_batches)
        except Exception as e:
            logger.warning(f"[BNDIAG] 진단 실패(실험 결과에는 영향 없음): {e!r}\n{traceback.format_exc()}")
        finally:
            self.force, self.capture, self.enabled = False, None, False
            for h in self.handles:
                h.remove()
            self.handles = []
            self.model.eval()

    def _finish(self, train_loader, enc_eval, n_unfrozen, amp, autocast_dtype, device, max_batches):
        model, bn = self.model, self.bn
        xs, ys = [], []
        for i, batch in enumerate(train_loader):
            if i >= max_batches:
                break
            xs.append(batch["x"])
            ys.append(batch["celltype_labels"])
        n_cells = sum(len(y) for y in ys)
        nb = len(self.blocks)

        def set_mode(name):
            model.eval()
            if name == "trainlike":
                model.train()
                if enc_eval:
                    if n_unfrozen <= 0:
                        model.encoder.eval()
                    else:
                        for blk in model.encoder.transformer_encoder[:-n_unfrozen]:
                            blk.eval()
                if bn is not None:
                    bn.eval()          # running 통계가 바뀌지 않도록

        def run(name, use_amp):
            set_mode(name)
            self.capture, self.stats, self.force = [], {}, True
            correct = total = 0
            with torch.no_grad():
                for x, y in zip(xs, ys):
                    x, y = x.to(device), y.to(device)
                    with torch.cuda.amp.autocast(enabled=(amp and use_amp), dtype=autocast_dtype):
                        logits = model(x)
                    correct += (logits.argmax(1) == y).sum().item()
                    total += len(y)
            feats = torch.cat(self.capture).double() if self.capture else None
            blk_absmax = max([float(self.stats.get(f"blk{i}_absmax", 0.0)) for i in range(nb)] or [0.0])
            blk_nonfinite = sum(float(self.stats.get(f"blk{i}_nonfinite", 0.0)) for i in range(nb))
            self.capture, self.force = None, False
            out = {"acc": correct / max(total, 1), "n": total, "blk_absmax_max": blk_absmax, "blk_nonfinite_sum": blk_nonfinite}
            if feats is not None:
                fin = torch.isfinite(feats)
                out["bn_in_nonfinite"] = int((~fin).sum())
                f2 = torch.where(fin, feats, torch.zeros_like(feats))
                m, v = f2.mean(0), f2.var(0, unbiased=False)
                out["bn_in_absmax"] = float(f2.abs().max())
                if bn is not None:
                    rm, rv = bn.running_mean.detach().double().cpu(), bn.running_var.detach().double().cpu()
                    z = (m - rm).abs() / torch.sqrt(rv.clamp_min(0) + bn.eps)
                    ratio = v / (rv.clamp_min(0) + bn.eps)
                    out.update(z_mean=float(z.mean()), z_p95=float(torch.quantile(z, 0.95)), z_max=float(z.max()),
                               var_ratio_median=float(ratio.median()), var_ratio_min=float(ratio.min()), var_ratio_max=float(ratio.max()))
            return out

        post = {}
        post["eval_fp32"] = run("eval", False)
        if amp:
            post["eval_fp16"] = run("eval", True)
            post["trainlike_fp16"] = run("trainlike", True)
        else:
            post["trainlike_fp32"] = run("trainlike", False)

        if bn is not None:
            backup, old_mom = copy.deepcopy(bn.state_dict()), bn.momentum
            old_rm, old_rv = bn.running_mean.detach().double().cpu().clone(), bn.running_var.detach().double().cpu().clone()
            try:
                bn.reset_running_stats()
                bn.momentum = None                 # 누적 평균
                model.eval()
                bn.train()                         # BatchNorm 만 배치 통계로 갱신, 나머지는 eval(dropout 끔)
                with torch.no_grad():
                    for x in xs:
                        model(x.to(device))        # fp32 (predict 와 같은 조건)
                bn.eval()
                bn.momentum = old_mom
                new_rm, new_rv = bn.running_mean.detach().double().cpu(), bn.running_var.detach().double().cpu()
                dz = (new_rm - old_rm).abs() / torch.sqrt(old_rv.clamp_min(0) + bn.eps)
                vr = new_rv / (old_rv.clamp_min(0) + bn.eps)
                post["recal_vs_orig"] = {"mean_shift_z_mean": float(dz.mean()), "mean_shift_z_p95": float(torch.quantile(dz, 0.95)),
                                         "mean_shift_z_max": float(dz.max()), "var_ratio_median": float(vr.median()),
                                         "var_ratio_min": float(vr.min()), "var_ratio_max": float(vr.max())}
                post["recal_eval_fp32"] = run("eval", False)
                if amp:
                    post["recal_eval_fp16"] = run("eval", True)
            finally:
                bn.load_state_dict(backup)         # 원래 running 통계 복원 (이후 predict() 결과가 바뀌지 않도록)
                bn.momentum = old_mom
                bn.eval()
            assert torch.equal(bn.running_mean.detach().double().cpu(), old_rm), "BatchNorm running_mean 복원 실패"

        result = {"tag": self.cfg.get("bn_diag_tag"), "lr": self.cfg.get("lr"), "seed": self.cfg.get("seed"),
                  "unfrozen_layers": self.cfg.get("scfoundation_unfrozen_layers"), "n_per_type": self.cfg.get("train_n_per_type"),
                  "precision": self.cfg.get("precision"), "n_cells_sampled": n_cells,
                  "eval_train_acc_full": getattr(model, "_eval_train_acc", None), "epochs": self.records, "post": post}
        model._bn_diag = result
        for k, d in post.items():
            logger.info(f"[BNDIAG] post {k}: " + " ".join(f"{a}={b:.4g}" if isinstance(b, float) else f"{a}={b}" for a, b in d.items()))
        out_path = self.cfg.get("bn_diag_out")
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
            logger.info(f"[BNDIAG] 저장: {out_path}")
