"""
benchmark/run_scgpt_sweep.py
scGPT에 대해 precision / micro_batch_size / grad_accum_steps /
activation_checkpointing / max_seq_len(gene 수) 조합을 돌며 GPU 메모리·시간·
macro-F1을 측정해서 benchmark/run_log.py로 CSV 한 곳에 기록하는 sweep 러너.

기존 src/run.py의 오케스트레이션(Step 3~9)을 재사용한다. 다만 load_data/
load_vocab_full/preprocess(Step 3~5)는 sweep 축(precision 등)과 전혀 무관하므로
main()에서 딱 한 번만 실행하고, prepare_inputs(Step 6 - max_seq_len/batch_size에
의존)과 load_model+finetune(Step 7~8 - precision/checkpointing에 의존)만 조합마다
새로 실행한다. LoRA/scPEFT 축은 여기서 다루지 않는다(benchmark/grid.py 모듈
docstring 참고 - scPEFT 공식 코드를 외부에서 별도로 돌리는 방향으로 이미 정리됨).

사용법 (서버에서, 예):
  python benchmark/run_scgpt_sweep.py \\
      --config config/config.yaml \\
      --grid smoke \\
      --out benchmark_results/scgpt_sweep.csv

  # coarse grid까지 확인했고 batch_size=16(성공)~64(OOM) 사이 경계를 더 보고 싶다면:
  python - <<'PY'
  from benchmark.grid import refine_between
  print(refine_between("micro_batch_size", 16, 64, n_points=4))
  PY
  # 위 출력을 참고해 grid.py에 조합을 추가하거나, 이 스크립트를 직접 override 리스트를
  # 받는 형태로 확장해서 사용.

해결됨 (2026-09-16): 조합마다 이 스크립트를 별도 프로세스로 재호출해서(--override-json)
완전히 새 CUDA 컨텍스트에서 시작하도록 바꿨다 - 실제 smoke test에서 checkpointing=True
조합의 peak_reserved_mb(21.8GB)가 바로 직전 OOM 조합의 reserved(23.5GB)와 비슷하게 높게
나온 걸 보고 위 우려가 실제로 나타난 것으로 판단, 기본 동작으로 격리를 켰다. 디버깅/속도가
더 중요할 땐 --no-isolate로 예전 방식(한 프로세스 안에서 grid 전체, Step 3~5는 1회만)을
그대로 쓸 수 있다 - 다만 그 경우 peak_reserved_mb 비교는 다시 신뢰할 수 없어진다.
프로세스 격리의 비용은 Step 3~5(데이터 로드/vocab/전처리)를 조합마다 다시 실행하는 것 -
이 데이터셋(21k 세포) 기준 수십 초 수준이라 감수할 만하다고 판단.

주의(아직 미검증):
  - seconds_per_epoch은 fine-tuning 전체 시간을 epoch 수로 나눈 평균값이다(실제
    epoch별 타이밍을 재려면 adapters/scgpt_adapter.py의 finetune() 내부에 별도
    계측이 필요 - 지금은 안 함).
  - macro_f1/accuracy는 query h5ad에 celltype_col 라벨이 있을 때만 계산된다.
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

from adapters import get_adapter  # noqa: E402
from pipeline.config import load_config  # noqa: E402

from benchmark import grid as grid_mod  # noqa: E402
from benchmark import run_log as blog  # noqa: E402
from benchmark.memory_probe import measure, set_memory_budget_gb  # noqa: E402

# benchmark/grid.py의 축 이름("micro_batch_size")과 scgpt_adapter.py가 실제로 읽는
# config.yaml 키("batch_size")가 다르다 - 여기서 번역한다. 새 축을 추가할 때 이름이
# 안 맞으면 여기 추가할 것(조용히 무시되지 않고, config.yaml에 없는 키는 adapter가
# cfg["batch_size"] 같은 필수 키 조회에서 KeyError로 바로 드러나므로 안전).
_OVERRIDE_KEY_MAP = {"micro_batch_size": "batch_size"}


def apply_overrides(base_cfg: dict, override: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for k, v in override.items():
        cfg[_OVERRIDE_KEY_MAP.get(k, k)] = v
    return cfg


def run_one(adapter, ctx: dict, override: dict, device, csv_path: Path, memory_budget_gb=None) -> None:
    """ctx: main()에서 미리 만들어둔 {"adata", "id2type", "num_types", "vocab",
    "model_configs", "cfg_base"} - Step 3~5까지 이미 끝낸 공유 상태."""
    cfg = apply_overrides(ctx["cfg_base"], override)
    run_id = blog.new_run_id()

    row_kwargs = dict(
        run_id=run_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        model="scgpt",
        mode="finetune_predict",
        checkpoint=cfg.get("model_dir"),
        dataset=cfg.get("reference_path"),
        n_cells=ctx["adata"].n_obs,
        n_genes_input=ctx["adata"].n_vars,
        max_seq_len=override.get("max_seq_len", cfg.get("max_seq_len")),
        finetune_method="full_ft",
        precision=override.get("precision", cfg.get("precision", "")),
        micro_batch_size=override.get("micro_batch_size", cfg.get("batch_size")),
        grad_accum_steps=override.get("grad_accum_steps", cfg.get("grad_accum_steps", 1)),
        effective_batch_size=grid_mod.effective_batch_size(override),
        activation_checkpointing=override.get("activation_checkpointing", cfg.get("activation_checkpointing", False)),
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
        prepared = adapter.prepare_inputs(ctx["adata"], cfg, vocab=ctx["vocab"])
        model = adapter.load_model(
            cfg, ctx["num_types"], device, vocab=ctx["vocab"], model_configs=ctx["model_configs"],
        )
    except Exception as e:
        blog.log_run(
            csv_path, status="error",
            error_message=f"prepare_inputs/load_model 실패: {e}\n{traceback.format_exc()[-500:]}",
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

    macro_f1, accuracy, notes = "", "", ""
    try:
        adata_result = adapter.predict(finetuned_holder["model"], ctx["adata"], prepared, ctx["id2type"], cfg, device)
        celltype_col = cfg.get("celltype_col")
        if celltype_col and celltype_col in adata_result.obs.columns:
            y_true = adata_result.obs[celltype_col].astype(str)
            y_pred = adata_result.obs["predictions"].astype(str)
            macro_f1 = round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4)
            accuracy = round(float((y_true == y_pred).mean()), 4)
    except Exception as e:
        notes = f"predict/f1 계산 실패(fine-tuning 자체는 성공): {e}"

    blog.log_run(
        csv_path, status="success",
        peak_allocated_mb=mem_result.peak_allocated_mb, peak_reserved_mb=mem_result.peak_reserved_mb,
        seconds_per_epoch=round(mem_result.seconds / max(cfg.get("epochs", 1), 1), 2),
        total_seconds=round(mem_result.seconds, 2),
        macro_f1=macro_f1, accuracy=accuracy, notes=notes,
        **row_kwargs,
    )


def _setup_logging():
    """src/run.py(Step 3~9 오케스트레이션의 원래 진입점)는 이 두 줄로 adapter의
    logger.info() 호출(예: "Fine-tuning 시작...", "Epoch N/M ...")을 stdout으로
    내보낸다. 이 sweep 스크립트는 run.py를 거치지 않고 adapter를 직접 호출하다 보니
    이 설정이 빠져 있었다 - 그 결과 root logger가 기본값(WARNING)에 머물러
    adapter 내부의 모든 INFO 로그(특히 epoch별 진행 상황)가 handler 자체가 없어
    "지연되는 것"이 아니라 아예 조용히 버려지고 있었다(2026-09-15 서버에서
    smoke grid 1번째 조합이 1시간 넘게 아무 출력 없이 실행 중인 상태를 보고 발견 -
    실제로는 멈춘 게 아니라 batch_size=1 x max_seq_len=3001 x epochs=10이라는
    smoke grid 최악 조합이 원래 오래 걸리는 것 + 이 로깅 버그가 겹쳐서 "멈춘 것처럼
    보인" 것). run.py의 설정과 완전히 동일하게 맞춘다."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s",
                         handlers=[logging.StreamHandler(sys.stdout)])
    logging.getLogger("adapters.scgpt_adapter").setLevel(logging.INFO)


def _build_ctx(cfg_base: dict, adapter, quiet: bool = False) -> dict:
    """Step 3~5(load_data/load_vocab_full/preprocess) - sweep 축과 무관해서 한 번만
    실행하면 되는 부분. --no-isolate(한 프로세스)에서는 main()이 1회만 부르고,
    격리 모드(기본값)에서는 조합마다 새 프로세스에서 _run_single_combo()가 부른다."""
    if not quiet:
        print("Step 3~5 (load_data / load_vocab_full / preprocess) - sweep 축과 무관, 1회만 실행")
    adata, adata_test_raw, id2type, num_types = adapter.load_data(cfg_base)
    adata, vocab, model_configs = adapter.load_vocab_full(adata, cfg_base["model_dir"])
    adata = adapter.preprocess(adata, cfg_base)
    if not quiet:
        print(f"  준비 완료: {adata.n_obs}개 세포, {adata.n_vars}개 유전자(vocab 교집합 후), "
              f"cell type {num_types}종")
    return {
        "adata": adata, "adata_test_raw": adata_test_raw, "id2type": id2type,
        "num_types": num_types, "vocab": vocab, "model_configs": model_configs,
        "cfg_base": cfg_base,
    }


def _run_single_combo(args) -> None:
    """--override-json으로 넘어온 조합 딱 하나만 실행하고 끝낸다. main()이 격리
    모드에서 이 스크립트 자신을 이 인자로 재호출하는 용도 - 조합마다 완전히 새
    프로세스/CUDA 컨텍스트에서 시작되게 만들어서, peak_reserved_mb가 이전 조합의
    GPU 메모리 캐시에 영향받지 않도록 한다(2026-09-16 smoke test에서 실제로 문제가
    확인됨: checkpointing=True 조합의 peak_reserved가 직전 OOM 조합과 비슷하게
    21.8GB로 높게 나왔음). 사람이 직접 쓸 일은 거의 없다 - 보통은 --grid만 넘기면
    main()이 알아서 이 인자를 붙여 재호출한다."""
    override = json.loads(args.override_json)
    cfg_base = load_config(args.config)
    adapter = get_adapter("scgpt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = _build_ctx(cfg_base, adapter, quiet=True)
    run_one(adapter, ctx, override, device, Path(args.out), memory_budget_gb=args.memory_budget_gb)


def _run_all_in_one_process(args, combos, out_path: Path) -> None:
    """예전 방식(--no-isolate). Step 3~5를 1회만 실행하고 모든 조합을 같은
    프로세스/CUDA 컨텍스트 안에서 연달아 돈다 - 빠르지만 peak_reserved_mb가 조합
    간에 섞일 수 있다(README "memory-benchmark 브랜치 상태" 참고). 디버깅이나
    빠른 반복 확인용으로만 쓸 것 - 최종 수치를 낼 때는 기본(격리) 모드를 쓸 것."""
    cfg_base = load_config(args.config)
    adapter = get_adapter("scgpt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("경고: CUDA를 찾지 못했습니다 - peak_allocated_mb/reserved는 의미 없는 값이 됩니다 "
              "(코드 골격 검증용으로만 쓸 것).")

    ctx = _build_ctx(cfg_base, adapter, quiet=False)

    print(f"\n{len(combos)}개 조합 실행 (grid={args.grid}, --no-isolate: 프로세스 격리 없음) -> {out_path}")
    for i, override in enumerate(combos, 1):
        print(f"\n[{i}/{len(combos)}] {override}")
        t0 = time.time()
        run_one(adapter, ctx, override, device, out_path, memory_budget_gb=args.memory_budget_gb)
        print(f"  기록 완료 ({time.time() - t0:.1f}초 소요) -> {out_path}")

    print(f"\n전체 sweep 완료. 결과: {out_path}")


def _run_isolated(args, combos, out_path: Path) -> None:
    """기본 모드. 조합마다 이 스크립트를 --override-json 하나로 재호출해서 별도
    프로세스로 실행한다 - Step 3~5를 조합마다 다시 실행하는 비용(수십 초 수준,
    21k 세포 기준)을 대가로 peak_reserved_mb를 조합 간 오염 없이 신뢰할 수 있게
    만든다."""
    this_script = str(Path(__file__).resolve())
    print(f"\n{len(combos)}개 조합 실행 (grid={args.grid}, 조합별 프로세스 격리) -> {out_path}")

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


def main():
    _setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="base config.yaml 경로")
    parser.add_argument("--grid", choices=["smoke", "coarse"], default="smoke",
                         help="smoke: 5개 조합(처음 실행 권장), coarse: grid.py의 전체 조합(144개 기본값)")
    parser.add_argument("--out", required=True, help="결과 CSV 저장 경로 (이미 있으면 append)")
    parser.add_argument("--memory-budget-gb", type=float, default=None,
                         help="예: 12 -> 실제 GPU 위에서 '12GB memory budget'을 흉내냄 "
                              "(memory_probe.set_memory_budget_gb 참고, 실제 GPU 총량보다 작아야 함)")
    parser.add_argument("--no-isolate", action="store_true",
                         help="조합별 프로세스 격리를 끄고 예전처럼 한 프로세스 안에서 전부 실행 "
                              "(빠르지만 peak_reserved_mb가 조합 간에 섞일 수 있음 - 디버깅용)")
    parser.add_argument("--override-json", default=None,
                         help=argparse.SUPPRESS)  # 내부용: 격리 모드에서 자기 자신을 재호출할 때만 쓰임
    args = parser.parse_args()

    if args.override_json is not None:
        _run_single_combo(args)
        return

    combos = grid_mod.default_smoke_grid() if args.grid == "smoke" else grid_mod.coarse_grid()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.no_isolate:
        _run_all_in_one_process(args, combos, out_path)
    else:
        _run_isolated(args, combos, out_path)


if __name__ == "__main__":
    main()
