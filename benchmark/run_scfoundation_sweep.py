"""
benchmark/run_scfoundation_sweep.py
scFoundation용 sweep 러너. run_scgpt_sweep.py와 구조는 같지만 run.py를 거치지
않고 adapter 메서드를 직접 호출한다 - 이유(중요):

adapters/scfoundation_adapter.py의 load_vocab_full()은 일부러 NotImplementedError를
낸다. run.py의 오케스트레이션이 모든 adapter에 대해
`adapter.load_vocab_full(adata, cfg["model_dir"])`을 호출하도록 고정돼 있는데,
scFoundation은 경로가 하나(model_dir)가 아니라 세 개(scfoundation_repo_dir,
scfoundation_ckpt_path, scfoundation_gene_list_path)가 필요해서 이 시그니처에
맞지 않는다(scfoundation_adapter.py의 load_vocab_full docstring 참고). 즉
**scFoundation은 아직 run.py CLI(mode: finetune_predict 등)로 실행할 수 없다** -
지금은 이 스크립트처럼 어댑터를 직접 호출하는 경로로만 쓴다. run.py 완전 통합
(base.py 인터페이스를 cfg 전체를 받도록 넓히는 것)은 이 브랜치 범위 밖의 후속
작업으로 남겨둔다(scGPT/Geneformer에도 영향을 주는 변경이라 신중하게 별도로
검토해야 함).

사용법 (서버에서, scFoundation 체크포인트/repo를 준비한 뒤):
  python benchmark/run_scfoundation_sweep.py \\
      --config config/config_scfoundation.yaml \\
      --grid smoke --out benchmark_results/scfoundation_sweep.csv

주의: 이 adapter 자체가 아직 GPU에서 한 번도 실행 검증되지 않았다
(scfoundation_adapter.py 모듈 docstring의 "검증 상태" 참고). 이 스크립트를 처음
돌릴 때는 --grid smoke의 첫 번째 조합 하나만이라도 성공하는지부터 확인할 것 -
전체 sweep을 바로 돌리지 말 것.
"""

import argparse
import copy
import json
import logging
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from adapters.scfoundation_adapter import ScFoundationAdapter  # noqa: E402
from pipeline.config import load_config  # noqa: E402

from benchmark import grid as grid_mod  # noqa: E402
from benchmark import run_log as blog  # noqa: E402
from benchmark.memory_probe import measure, set_memory_budget_gb  # noqa: E402

_OVERRIDE_KEY_MAP = {"micro_batch_size": "batch_size", "max_seq_len": "n_hvg_genes"}
# max_seq_len -> n_hvg_genes: scGPT sweep과 같은 CSV 컬럼(max_seq_len)에 기록하되,
# scFoundation 쪽 config 키는 n_hvg_genes다(scfoundation_adapter.preprocess() 참고,
# scGPT처럼 시퀀스를 자르는 게 아니라 HVG로 유전자 자체를 미리 줄이는 방식이라
# 메커니즘은 다르지만 "gene 수 축"이라는 의미는 같아서 같은 grid.py를 재사용한다).


def apply_overrides(base_cfg: dict, override: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for k, v in override.items():
        cfg[_OVERRIDE_KEY_MAP.get(k, k)] = v
    return cfg


def run_one(adapter, ctx: dict, override: dict, device, csv_path: Path, memory_budget_gb=None) -> None:
    cfg = apply_overrides(ctx["cfg_base"], override)
    run_id = blog.new_run_id()

    row_kwargs = dict(
        run_id=run_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        model="scfoundation",
        mode="finetune_predict",
        checkpoint=cfg.get("scfoundation_ckpt_path"),
        dataset=cfg.get("reference_path"),
        n_cells=ctx["adata_raw"].n_obs,
        n_genes_input=override.get("max_seq_len", cfg.get("n_hvg_genes") or 19264),
        max_seq_len=override.get("max_seq_len", cfg.get("n_hvg_genes", "")),
        finetune_method="full_ft",
        precision=override.get("precision", cfg.get("precision", "")),
        micro_batch_size=override.get("micro_batch_size", cfg.get("batch_size")),
        grad_accum_steps=override.get("grad_accum_steps", cfg.get("grad_accum_steps", 1)),
        effective_batch_size=grid_mod.effective_batch_size(override),
        activation_checkpointing=override.get("activation_checkpointing", False),
        notes=("activation_checkpointing=True가 요청됐지만 scfoundation_adapter는 아직 미지원 "
               "(finetune()이 경고 후 무시함)" if override.get("activation_checkpointing") else ""),
        lr=override.get("lr", cfg.get("lr", 0.0001)),
        seed=cfg.get("seed", 42),
        **blog.base_environment_fields(),
        **blog.gpu_fields(),
    )

    if memory_budget_gb is not None:
        try:
            simulated = set_memory_budget_gb(memory_budget_gb, device)
        except Exception as e:
            blog.log_run(csv_path, status="error", error_message=f"memory_budget 설정 실패: {e}", **row_kwargs)
            return
        row_kwargs["memory_budget_gb"] = memory_budget_gb
        row_kwargs["memory_budget_is_simulated"] = simulated

    try:
        # n_hvg_genes(gene 수 축)는 preprocess() 단계에 영향을 주므로 override마다 다시 실행.
        adata = adapter.preprocess(ctx["adata_raw"].copy(), cfg)
        prepared = adapter.prepare_inputs(adata, cfg)
        model = adapter.load_model(cfg, ctx["num_types"], device)
    except Exception as e:
        blog.log_run(
            csv_path, status="error",
            error_message=f"preprocess/prepare_inputs/load_model 실패: {e}\n{traceback.format_exc()[-500:]}",
            **row_kwargs,
        )
        return

    finetuned_holder = {}
    with measure(device) as mem_result:
        finetuned_holder["model"] = adapter.finetune(model, prepared, cfg, device)

    if mem_result.status == "oom":
        blog.log_run(
            csv_path, status="oom", error_message=mem_result.error_message,
            peak_allocated_mb=mem_result.peak_allocated_mb, peak_reserved_mb=mem_result.peak_reserved_mb,
            total_seconds=round(mem_result.seconds, 2),
            **row_kwargs,
        )
        return

    macro_f1, accuracy = "", ""
    try:
        adata_result = adapter.predict(finetuned_holder["model"], adata, prepared, ctx["id2type"], cfg, device)
        celltype_col = cfg.get("celltype_col")
        y_true = adata_result.obs[celltype_col].astype(str)
        y_pred = adata_result.obs["predictions"].astype(str)
        macro_f1 = round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4)
        accuracy = round(float((y_true == y_pred).mean()), 4)
    except Exception as e:
        row_kwargs["notes"] = (row_kwargs.get("notes", "") + f" | predict/f1 계산 실패: {e}").strip(" |")

    blog.log_run(
        csv_path, status="success",
        peak_allocated_mb=mem_result.peak_allocated_mb, peak_reserved_mb=mem_result.peak_reserved_mb,
        seconds_per_epoch=round(mem_result.seconds / max(cfg.get("epochs", 1), 1), 2),
        total_seconds=round(mem_result.seconds, 2),
        macro_f1=macro_f1, accuracy=accuracy,
        **row_kwargs,
    )


def _setup_logging():
    """run_scgpt_sweep.py의 _setup_logging()과 동일한 이유 - src/run.py가 하는
    logging.basicConfig()/setLevel() 없이는 scfoundation_adapter.py의 logger.info()
    (Fine-tuning 시작/Epoch 진행 로그 포함)가 전부 조용히 버려진다."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s",
                         handlers=[logging.StreamHandler(sys.stdout)])
    logging.getLogger("adapters.scfoundation_adapter").setLevel(logging.INFO)


def _build_ctx(cfg_base: dict, adapter, quiet: bool = False) -> dict:
    if not quiet:
        print("Step: load_data (sweep 축과 무관, 1회만 실행)")
    adata_raw, adata_test_raw, id2type, num_types = adapter.load_data(cfg_base)
    if not quiet:
        print(f"  준비 완료: {adata_raw.n_obs}개 세포, cell type {num_types}종")
    return {"adata_raw": adata_raw, "id2type": id2type, "num_types": num_types, "cfg_base": cfg_base}


def _run_single_combo(args) -> None:
    """run_scgpt_sweep.py의 _run_single_combo()와 동일한 목적 - 조합마다 별도
    프로세스로 격리해서 peak_reserved_mb가 이전 조합의 GPU 캐시에 오염되지 않게
    한다(2026-09-16 scGPT smoke test에서 실측으로 확인된 문제, run_scgpt_sweep.py
    모듈 docstring 참고). scFoundation은 아직 GPU 검증 전이라 이 경로 자체가
    미검증이지만, 구조는 미리 맞춰둔다."""
    override = json.loads(args.override_json)
    cfg_base = load_config(args.config)
    adapter = ScFoundationAdapter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = _build_ctx(cfg_base, adapter, quiet=True)
    run_one(adapter, ctx, override, device, Path(args.out), memory_budget_gb=args.memory_budget_gb)


def _run_all_in_one_process(args, combos, out_path: Path) -> None:
    cfg_base = load_config(args.config)
    adapter = ScFoundationAdapter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = _build_ctx(cfg_base, adapter, quiet=False)

    print(f"\n{len(combos)}개 조합 실행 (grid={args.grid}, --no-isolate: 프로세스 격리 없음) -> {out_path}")
    print("주의: 이 adapter는 아직 GPU 검증 전입니다 - 첫 조합 결과를 반드시 직접 확인하세요.")
    for i, override in enumerate(combos, 1):
        print(f"\n[{i}/{len(combos)}] {override}")
        t0 = time.time()
        run_one(adapter, ctx, override, device, out_path, memory_budget_gb=args.memory_budget_gb)
        print(f"  기록 완료 ({time.time() - t0:.1f}초 소요) -> {out_path}")

    print(f"\n전체 sweep 완료. 결과: {out_path}")


def _run_isolated(args, combos, out_path: Path) -> None:
    this_script = str(Path(__file__).resolve())
    print(f"\n{len(combos)}개 조합 실행 (grid={args.grid}, 조합별 프로세스 격리) -> {out_path}")
    print("주의: 이 adapter는 아직 GPU 검증 전입니다 - 첫 조합 결과를 반드시 직접 확인하세요.")

    for i, override in enumerate(combos, 1):
        print(f"\n[{i}/{len(combos)}] {override}")
        t0 = time.time()
        cmd = [
            sys.executable, this_script,
            "--config", args.config,
            "--out", str(out_path),
            "--override-json", json.dumps(override),
        ]
        if args.memory_budget_gb is not None:
            cmd += ["--memory-budget-gb", str(args.memory_budget_gb)]
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print(f"  완료 ({time.time() - t0:.1f}초 소요) -> {out_path}")
        else:
            print(f"  !! 비정상 종료(returncode={result.returncode}, {time.time() - t0:.1f}초 소요) - "
                  f"이 조합은 CSV에 안 남았을 수 있음. 위 stderr를 확인할 것.")

    print(f"\n전체 sweep 완료. 결과: {out_path}")


_GRID_FUNCS = {
    "smoke": grid_mod.default_smoke_grid,
    "coarse": grid_mod.coarse_grid,
    "gene_length": grid_mod.scfoundation_gene_length_grid,
    "priority": grid_mod.scfoundation_priority_grid,
}
# 2026-09-22 추가: scGPT의 run_scgpt_sweep.py와 같은 패턴(_GRID_FUNCS 딕셔너리)으로
# --grid 선택지를 확장 - gene_length_grid로 gene-count 단독 효과를 본다.


def main():
    _setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="scFoundation용 config.yaml 경로 "
                                                          "(scfoundation_repo_dir/ckpt_path/gene_list_path 포함)")
    parser.add_argument("--grid", choices=list(_GRID_FUNCS.keys()), default="smoke")
    parser.add_argument("--out", required=True)
    parser.add_argument("--memory-budget-gb", type=float, default=None)
    parser.add_argument("--no-isolate", action="store_true",
                         help="조합별 프로세스 격리를 끄고 예전처럼 한 프로세스 안에서 전부 실행")
    parser.add_argument("--override-json", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.override_json is not None:
        _run_single_combo(args)
        return

    combos = _GRID_FUNCS[args.grid]()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.no_isolate:
        _run_all_in_one_process(args, combos, out_path)
    else:
        _run_isolated(args, combos, out_path)


if __name__ == "__main__":
    main()
