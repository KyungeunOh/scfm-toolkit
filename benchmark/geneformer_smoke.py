#!/usr/bin/env python3
"""Geneformer 어댑터 시험 실행(G2): 작은 데이터로 어댑터 전 단계를 run.py 와 같은 순서로 돌려,
어느 단계까지 되는지 확인한다. 성능을 재는 실험이 아니다(에폭 1, 세포 수백 개).

순서(src/run.py 의 finetune_predict 와 동일):
  load_data -> load_vocab_full -> (부분 표본) -> preprocess -> prepare_inputs -> load_model -> finetune -> predict

사용 (컨테이너 안):
  python /workspace/benchmark/geneformer_smoke.py --config /workspace/config_geneformer.yaml --n-per-type 10 --n-query 100

결과는 <output_dir>/smoke_<시각>/ 아래에 쓰고(체크포인트 포함), 단계별 성공/실패와 소요 시간은
smoke_result.json 에 남긴다. 실패하면 어느 단계에서 났는지 출력하고 종료 코드 1 로 끝난다.
"""
import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def subsample(adata, celltype_col, n_per_type, n_query, seed):
    """reference 는 세포 유형마다 최대 n_per_type 개, query 는 무작위 n_query 개를 남긴다(순서 유지)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    batch = adata.obs["str_batch"].astype(str).values
    labels = adata.obs[celltype_col].astype(str).values
    ref_idx = np.where(batch == "0")[0]
    qry_idx = np.where(batch == "1")[0]
    keep = []
    for lab in sorted(set(labels[ref_idx])):
        idx = ref_idx[labels[ref_idx] == lab]
        keep.extend(rng.choice(idx, size=min(n_per_type, len(idx)), replace=False))
    keep.extend(rng.choice(qry_idx, size=min(n_query, len(qry_idx)), replace=False))
    return adata[np.sort(np.array(keep))].copy()


def score(result, celltype_col):
    """query 예측 점수. macro F1 은 query 에 실제로 있는 유형만 대상으로 한다(시험용 지표, 본 실험 지표와 다름)."""
    from sklearn.metrics import accuracy_score, f1_score

    y_true = result.obs[celltype_col].astype(str).values
    y_pred = result.obs["predictions"].astype(str).values
    labels = sorted(set(y_true))
    return {
        "n_query": int(len(y_true)),
        "n_missing_pred": int(result.obs["predictions"].isna().sum()),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1_default": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),  # sweep 과 같은 정의
        "macro_f1_query_labels": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-per-type", type=int, default=10)
    ap.add_argument("--n-query", type=int, default=100)
    # 2026-11-19 batch: 설정값 덮어쓰기(주지 않으면 config 값 그대로)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    import yaml

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for _k in ("epochs", "lr", "seed"):
        if getattr(args, _k) is not None:
            cfg[_k] = getattr(args, _k)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = Path(cfg["output_dir"]) / f"smoke_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(out_dir)  # 어댑터의 작업 폴더(geneformer_work)와 체크포인트가 여기에 생긴다

    sys.path.insert(0, os.environ.get("SCFM_SRC", "/workspace/src"))
    from adapters.geneformer_adapter import GeneformerAdapter

    result = {
        "started_utc": stamp, "config": args.config, "n_per_type": args.n_per_type, "n_query": args.n_query,
        "seed": cfg.get("seed", 42), "epochs": cfg.get("epochs"), "lr": cfg.get("lr"), "batch_size": cfg.get("batch_size"),
        "model_dir": cfg["model_dir"],
        "git_commit": os.environ.get("SCFM_GIT_COMMIT", ""), "geneformer_commit": os.environ.get("GENEFORMER_COMMIT", ""),
        "stages": [], "ok": False,
    }

    def stage(name, fn):
        t0 = time.time()
        print(f"\n=== [{name}] 시작 ===", flush=True)
        try:
            out = fn()
            sec = round(time.time() - t0, 1)
            result["stages"].append({"stage": name, "status": "ok", "seconds": sec})
            print(f"=== [{name}] 성공 ({sec}s) ===", flush=True)
            return out
        except Exception as e:
            result["stages"].append({"stage": name, "status": "error", "error": f"{type(e).__name__}: {e}"})
            traceback.print_exc()
            print(f"=== [{name}] 실패: {type(e).__name__}: {e} ===", flush=True)
            raise

    try:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        result["device"] = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        print("device:", result["device"], flush=True)
    except Exception as e:  # 샌드박스 시험용
        torch, device = None, "cpu"
        result["device"] = f"no torch ({e})"

    adapter = GeneformerAdapter()
    try:
        adata, _raw, id2type, num_types = stage("load_data", lambda: adapter.load_data(cfg))
        print(f"  세포 {adata.n_obs} (reference {(adata.obs['str_batch'] == '0').sum()}, query {(adata.obs['str_batch'] == '1').sum()}), "
              f"유전자 {adata.n_vars}, 유형 {num_types}", flush=True)
        n_before = adata.n_vars
        adata, vocab, model_configs = stage("load_vocab_full", lambda: adapter.load_vocab_full(adata, cfg["model_dir"]))
        print(f"  유전자 {n_before} -> {adata.n_vars} (토큰화 가능한 유전자만 남김)", flush=True)
        result["n_genes_before"], result["n_genes_after"] = int(n_before), int(adata.n_vars)

        adata = stage("subsample", lambda: subsample(adata, cfg["celltype_col"], args.n_per_type, args.n_query, cfg.get("seed", 42)))
        print(f"  표본 세포 {adata.n_obs} (reference {(adata.obs['str_batch'] == '0').sum()}, query {(adata.obs['str_batch'] == '1').sum()})", flush=True)

        adata = stage("preprocess", lambda: adapter.preprocess(adata, cfg))
        n0 = int((adata.obs["n_counts"] <= 0).sum())
        print(f"  n_counts<=0 인 세포: {n0}, X 형식: {type(adata.X).__name__}", flush=True)
        result["n_zero_count_cells"] = n0

        prepared = stage("prepare_inputs", lambda: adapter.prepare_inputs(adata, cfg, vocab=vocab))
        model = stage("load_model", lambda: adapter.load_model(cfg, num_types, device, vocab=vocab, model_configs=model_configs))
        model = stage("finetune", lambda: adapter.finetune(model, prepared, cfg, device))
        pred = stage("predict", lambda: adapter.predict(model, adata, prepared, id2type, cfg, device))
        _cols = [c for c in ("_scfm_cell_id", cfg["celltype_col"], "predictions", "pred_score") if c in pred.obs.columns]
        pred.obs[_cols].to_csv(out_dir / "predictions.csv")
        result["score"] = stage("score", lambda: score(pred, cfg["celltype_col"]))
        print("  점수:", result["score"], flush=True)
        result["ok"] = True
    except Exception:
        pass
    finally:
        if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            result["peak_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            result["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
        with open(out_dir / "smoke_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장: {out_dir / 'smoke_result.json'}  ok={result['ok']}", flush=True)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
