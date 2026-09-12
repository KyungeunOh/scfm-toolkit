"""
benchmark/grid.py
"모든 조합을 전부 실행할 필요는 없다 - 먼저 간격이 큰 설정으로 대략적인 경향을
확인하고, OOM이 발생하는 경계 근처만 자세히 측정하면 된다"는 교수님 지시를
코드로 옮긴 것.

사용 흐름:
  1. coarse_grid()로 성긴 조합을 만들어 실행 -> 각 config가 success/oom 중 뭐였는지 확인
  2. 성공/실패가 갈리는 축(예: micro_batch_size)을 골라 refine_between()으로
     그 경계 사이만 촘촘한 조합을 새로 만들어 실행

두 그리드 모두 dict 리스트를 반환한다 - 각 dict는 config.yaml에 덮어씌울 override
필드들이다(run_scgpt_sweep.py의 apply_overrides() 참고).

LoRA/rank 축은 여기서 다루지 않는다 - scfm-toolkit 자체 fine-tuning 루프에 LoRA를
새로 구현하지 않고 scPEFT 공식 코드(github.com/laolintou/scPEFT)를 scGPT/scFoundation에
그대로 적용해 비교군으로 쓰기로 한 방향(2026-09-10 교수님께 보낸 메일)과 일치시키기
위함 - 이 grid.py는 "Full FT" 축의 시스템 설정(precision/batch/grad_accum/
checkpointing/gene 수)만 다룬다.
"""

import itertools
from typing import Any, Dict, List, Optional


def coarse_grid(
    precisions: List[str] = ("fp32", "fp16", "bf16"),
    micro_batch_sizes: List[int] = (1, 4, 16, 64),
    grad_accum_steps: List[int] = (1, 8),
    activation_checkpointing: List[bool] = (False, True),
    max_seq_lens: List[int] = (500, 1500, 3001),
) -> List[Dict[str, Any]]:
    """
    성긴 간격의 전체 조합(full factorial). 기본값 예시:
    3 precision x 4 batch x 2 grad_accum x 2 checkpointing x 3 gene 수 = 144개.
    실제로는 이 중 상당수가 "당연히 성공"(예: batch=1+checkpointing=True는 거의
    항상 됨)이거나 "당연히 실패"할 것이므로, 처음엔 이보다 더 줄인 부분집합으로
    시작하는 것을 권장한다(아래 default_smoke_grid 참고) - 이 함수는 축을 조합하는
    로직 자체를 재사용 가능하게 만든 것이지, 기본값 그대로 다 돌리라는 뜻이 아니다.
    """
    combos = itertools.product(
        precisions, micro_batch_sizes, grad_accum_steps, activation_checkpointing, max_seq_lens,
    )
    return [
        {
            "precision": p,
            "micro_batch_size": b,
            "grad_accum_steps": g,
            "activation_checkpointing": c,
            "max_seq_len": m,
        }
        for p, b, g, c, m in combos
    ]


def default_smoke_grid() -> List[Dict[str, Any]]:
    """
    coarse_grid()보다도 더 작은, "일단 파이프라인이 도는지 + 대략적인 OOM 위치
    감을 잡기 위한" 최소 조합. 서버에서 이 sweep 코드를 처음 돌릴 때 여기서부터
    시작하는 것을 권장한다(전체 coarse_grid는 그 다음).
    """
    return [
        {"precision": "fp16", "micro_batch_size": 1, "grad_accum_steps": 1,
         "activation_checkpointing": False, "max_seq_len": 3001},
        {"precision": "fp16", "micro_batch_size": 8, "grad_accum_steps": 1,
         "activation_checkpointing": False, "max_seq_len": 3001},
        {"precision": "fp16", "micro_batch_size": 32, "grad_accum_steps": 1,
         "activation_checkpointing": False, "max_seq_len": 3001},
        {"precision": "fp16", "micro_batch_size": 32, "grad_accum_steps": 1,
         "activation_checkpointing": True, "max_seq_len": 3001},
        {"precision": "bf16", "micro_batch_size": 32, "grad_accum_steps": 1,
         "activation_checkpointing": True, "max_seq_len": 3001},
    ]


def refine_between(
    axis: str,
    known_ok_value,
    known_oom_value,
    n_points: int = 4,
    base_override: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    coarse_grid에서 axis(예: "micro_batch_size")가 known_ok_value에서는 성공,
    known_oom_value에서는 OOM이었다는 걸 확인했다면, 그 사이를 n_points개로
    나눠서 정확한 경계를 좁히기 위한 override 리스트를 만든다.

    현재는 micro_batch_size/max_seq_len처럼 정수 축만 지원한다(선형 보간 후
    정수로 반올림, 중복 제거). precision/activation_checkpointing처럼 값이
    2~3개뿐인 범주형 축은애초에 "경계"라는 개념이 없어 refine 대상이 아니다.

    base_override: 다른 축은 고정해둔 채로 이 축만 바꾸고 싶을 때 그 고정값들.
    예: base_override={"precision": "bf16", "activation_checkpointing": True}
    """
    if not isinstance(known_ok_value, int) or not isinstance(known_oom_value, int):
        raise TypeError(
            f"refine_between()은 지금은 정수 축만 지원합니다 (axis={axis}). "
            f"micro_batch_size 또는 max_seq_len에 사용하세요."
        )
    lo, hi = sorted([known_ok_value, known_oom_value])
    step = max(1, (hi - lo) // (n_points + 1))
    candidates = sorted(set(range(lo + step, hi, step)))
    base_override = dict(base_override or {})
    return [dict(base_override, **{axis: v}) for v in candidates]


def effective_batch_size(override: Dict[str, Any]) -> Optional[int]:
    if "micro_batch_size" in override and "grad_accum_steps" in override:
        return override["micro_batch_size"] * override["grad_accum_steps"]
    return None
