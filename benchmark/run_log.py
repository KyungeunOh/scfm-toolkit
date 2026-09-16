"""
benchmark/run_log.py
교수님이 요청하신 "모든 실험 결과를 하나의 CSV/DB로 합칠 수 있어야 한다"는
요구사항을 위한 로깅 스키마 + append 함수.

설계 원칙(기존 pipeline/report.py와 동일):
- 사람이 실행 후 수동으로 표에 옮겨 적는 방식은 쓰지 않는다 - run_*_sweep.py가
  config 조합 하나를 실행할 때마다 이 모듈로 CSV에 한 줄씩 바로 append한다.
- 실행 도중 어디선가 죽어도(OOM 등) 그 전까지 기록된 행은 CSV에 이미 저장돼
  있어야 한다 - pipeline/grn.py의 save_metagene_scores()와 같은 이유
  ("그림이 안 예뻐도 데이터는 안전하게" 원칙, Phase 13 참고).
- "12GB 조건"을 A6000 등에서 memory_fraction으로 흉내낸 것과 실제 12GB GPU를
  절대 혼동하지 않도록 gpu_name(실제 GPU)과 memory_budget_gb(있다면 흉내낸 예산)
  컬럼을 분리해서 둔다 - 교수님이 명시적으로 요청한 부분.
"""

import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# 교수님이 요청하신 항목을 그대로 컬럼으로 옮긴 것. 순서는 "무엇을 돌렸는지 ->
# 어떤 설정이었는지 -> 결과가 어땠는지" 순으로 사람이 CSV를 열었을 때 읽기 쉽게
# 배치했다. 새 축이 필요해지면 이 리스트 끝에 추가하고, 기존에 쌓인 CSV는
# 헤더가 안 맞으면 append_row()가 명확히 에러를 내도록 한다(조용히 컬럼이
# 밀리는 것 방지).
RUN_LOG_COLUMNS = [
    # 실행 식별
    "run_id", "timestamp_utc", "model", "mode",
    # 모델/체크포인트/데이터
    "checkpoint", "dataset", "n_cells",
    # 입력 축 (생물학적 정보량 - batch/checkpointing과 반드시 구분해서 본다)
    "n_genes_input", "max_seq_len",
    # fine-tuning 방식
    "finetune_method",       # full_ft / lora / scpeft_official / scpeft_manual
    "lora_rank",             # finetune_method가 lora일 때만 의미 있음, 아니면 빈 값
    "lora_target_layers",
    # 시스템 설정 축
    "precision",             # fp32 / fp16 / bf16
    "micro_batch_size",
    "grad_accum_steps",
    "effective_batch_size",  # micro_batch_size * grad_accum_steps (편의 컬럼)
    "activation_checkpointing",
    "lr",                     # learning rate (2026-09-16 추가 - batch_size=1에서
                              # majority-class collapse가 lr/batch 불일치 때문으로
                              # 확인된 뒤, priority_grid()에서 배치별로 lr을 바꿔가며
                              # 테스트하기 위해 필요해짐. 기존 CSV에 이 컬럼을 추가하려면
                              # README.md의 "스키마 마이그레이션" 절 참고)
    # GPU 조건 (실제 GPU와 흉내낸 예산을 절대 혼동하지 않도록 분리)
    "gpu_name",               # torch.cuda.get_device_name() 실측값
    "gpu_total_memory_gb",    # 그 GPU의 실제 전체 메모리
    "memory_budget_gb",       # set_per_process_memory_fraction으로 흉내낸 예산.
                              # 비어있으면 "예산 제한 없음(GPU 전체 사용)"을 의미.
    "memory_budget_is_simulated",  # True면 memory_budget_gb는 "실제 GPU"가 아니라
                                    # "memory budget"라고 반드시 표기해야 함(교수님 지적).
    # 결과
    "status",                 # success / oom / error
    "error_message",
    "peak_allocated_mb",
    "peak_reserved_mb",
    "seconds_per_epoch",
    "total_seconds",
    "macro_f1",
    "accuracy",
    # 재현성/환경
    "seed",
    "python_version",
    "torch_version",
    "git_commit",
    "notes",
]


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")


def _get_package_version(pkg_name: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(pkg_name)
    except Exception:
        return None


def _get_git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def base_environment_fields() -> Dict[str, Any]:
    """python/torch 버전, git commit 등 - pipeline/report.py의 save_environment_report()와
    같은 정보를 이 CSV 행에도 그대로 채워 넣기 위한 헬퍼(중복이지만, environment.json은
    실행 하나당 파일 하나라 여러 실행을 한 CSV로 비교하려면 행에도 있어야 함)."""
    fields = {
        "python_version": sys.version.split()[0],
        "torch_version": _get_package_version("torch"),
        "git_commit": _get_git_commit(),
    }
    return fields


def gpu_fields() -> Dict[str, Any]:
    """실제 GPU 이름/전체 메모리. GPU가 없는 환경(코드 골격만 검증할 때)에서도
    에러 없이 빈 값을 반환한다."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
            }
    except Exception:
        pass
    return {"gpu_name": None, "gpu_total_memory_gb": None}


def build_row(**kwargs) -> Dict[str, Any]:
    """RUN_LOG_COLUMNS에 정의된 필드만 받아 한 행(dict)을 만든다.
    정의에 없는 키를 넘기면 즉시 에러 - 스키마에 없는 필드가 조용히 무시되는 것을
    방지한다(나중에 "분명히 기록했는데 CSV엔 없다"는 혼란을 막기 위함)."""
    unknown = set(kwargs) - set(RUN_LOG_COLUMNS)
    if unknown:
        raise ValueError(
            f"RUN_LOG_COLUMNS에 없는 필드: {sorted(unknown)}. "
            f"새 축이면 benchmark/run_log.py의 RUN_LOG_COLUMNS에 먼저 추가하세요."
        )
    row = {col: kwargs.get(col, "") for col in RUN_LOG_COLUMNS}
    return row


def append_row(row: Dict[str, Any], csv_path: Path) -> None:
    """CSV에 한 행을 append한다. 파일이 없으면 헤더부터 만든다.
    이미 있는 CSV의 헤더가 RUN_LOG_COLUMNS와 다르면(예: 스키마를 나중에 바꿨는데
    옛날 CSV에 이어붙이려는 경우) 컬럼이 밀려서 조용히 잘못된 값이 들어가는 걸
    막기 위해 명확히 에러를 낸다."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()
    if file_exists:
        with open(csv_path, "r", newline="") as f:
            existing_header = next(csv.reader(f), None)
        if existing_header is not None and existing_header != RUN_LOG_COLUMNS:
            raise RuntimeError(
                f"{csv_path}의 기존 헤더가 현재 RUN_LOG_COLUMNS와 다릅니다.\n"
                f"  기존: {existing_header}\n  현재: {RUN_LOG_COLUMNS}\n"
                f"스키마를 바꿨다면 새 CSV 경로를 쓰거나, 기존 파일을 마이그레이션하세요."
            )

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_LOG_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def log_run(csv_path: Path, **kwargs) -> Dict[str, Any]:
    """build_row + append_row를 한 번에. run_*_sweep.py에서 이 함수 하나만 부르면 된다."""
    row = build_row(**kwargs)
    append_row(row, csv_path)
    return row
