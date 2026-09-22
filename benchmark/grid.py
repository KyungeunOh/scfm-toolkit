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


def priority_grid(
    batch_boundary_lo: int = 8,
    batch_boundary_hi: int = 32,
) -> List[Dict[str, Any]]:
    """
    144개 coarse_grid() 전체 대신, 2026-09-16 smoke test 결과에서 나온 두 가지
    구체적인 질문에 답하기 위해 축소한 조합("다음 단계" 4번). smoke 5개 조합
    기준 실측 소요시간(2시간 반, batch=1이 대부분)을 감안해 이미 있는 조합은
    반복하지 않는다 - 아래 각 항목에서 "smoke에 이미 있음"이라고 표시한 값은
    빠져 있다.

    포함하는 것:
    1. batch=1 + 낮은 lr 2종 (labmeeting_report 3.1절 가설 검증 - lr을 낮추면
       majority-class collapse 없이 정상 학습되는지). 기존 lr=1e-4 결과는
       smoke CSV에 이미 있음.
    2. checkpointing 없이 micro_batch_size=batch_boundary_lo(성공)~
       batch_boundary_hi(OOM) 사이 OOM 경계를 refine_between()으로 촘촘하게.
    3. checkpointing 켜고 같은 batch 지점들 - checkpointing이 OOM 경계 자체를
       얼마나 뒤로 미루는지 2번과 짝지어 비교.
    4. OOM 경계 중간값(batch_boundary_lo/hi 사이 refine 지점 중 하나) 기준
       precision별 비교 - fp16은 2/3번에 이미 포함되므로 fp32/bf16만 추가.
    5. grad_accum_steps 효과 - micro_batch_size=batch_boundary_lo를 고정하고
       grad_accum=8로 늘렸을 때(유효 배치 8배) peak memory가 실제로 그대로인지.
    6. max_seq_len(gene 수) 효과 - micro_batch_size=batch_boundary_lo 고정,
       500/1500에서 memory가 얼마나 줄어드는지(3001은 smoke에 이미 있음).

    override dict에 "lr" 키를 직접 넣으면 apply_overrides()가 cfg["lr"]로
    그대로 덮어쓴다(run_scgpt_sweep.py의 _OVERRIDE_KEY_MAP에 없는 키는 이름
    그대로 쓰임) - scgpt_adapter.finetune()이 cfg.get("lr", 1e-4)로 읽으므로
    코드 수정 없이 바로 동작한다. run_log.py의 lr 컬럼에 실제 쓰인 값이
    기록되므로, 이 grid로 실행한 뒤 CSV에서 바로 확인 가능하다.
    """
    combos: List[Dict[str, Any]] = []

    # 1. batch=1 + 낮은 lr (majority-class collapse 가설 검증)
    for lr in (3e-5, 1e-5):
        combos.append({
            "precision": "fp16", "micro_batch_size": 1, "grad_accum_steps": 1,
            "activation_checkpointing": False, "max_seq_len": 3001, "lr": lr,
        })

    # 2~3. OOM 경계 refine, checkpointing 유/무 각각
    boundary_points = refine_between(
        "micro_batch_size", batch_boundary_lo, batch_boundary_hi, n_points=5,
    )
    mid_batch = None
    for point in boundary_points:
        b = point["micro_batch_size"]
        if mid_batch is None or abs(b - (batch_boundary_lo + batch_boundary_hi) / 2) < \
                abs(mid_batch - (batch_boundary_lo + batch_boundary_hi) / 2):
            mid_batch = b
        combos.append({
            "precision": "fp16", "micro_batch_size": b, "grad_accum_steps": 1,
            "activation_checkpointing": False, "max_seq_len": 3001,
        })
        combos.append({
            "precision": "fp16", "micro_batch_size": b, "grad_accum_steps": 1,
            "activation_checkpointing": True, "max_seq_len": 3001,
        })

    # 4. OOM 경계 중간 배치 기준 precision 비교 (fp16은 2~3번에 이미 있음)
    for precision in ("fp32", "bf16"):
        combos.append({
            "precision": precision, "micro_batch_size": mid_batch, "grad_accum_steps": 1,
            "activation_checkpointing": False, "max_seq_len": 3001,
        })

    # 5. grad_accum_steps 효과 (batch_boundary_lo 고정, 유효 배치만 8배로)
    combos.append({
        "precision": "fp16", "micro_batch_size": batch_boundary_lo, "grad_accum_steps": 8,
        "activation_checkpointing": False, "max_seq_len": 3001,
    })

    # 6. max_seq_len(gene 수) 효과 (batch_boundary_lo 고정, 3001은 smoke에 이미 있음)
    for seq_len in (500, 1500):
        combos.append({
            "precision": "fp16", "micro_batch_size": batch_boundary_lo, "grad_accum_steps": 1,
            "activation_checkpointing": False, "max_seq_len": seq_len,
        })

    return combos


def gene_length_grid() -> List[Dict[str, Any]]:
    """
    2026-09-17: priority_grid()의 gene 수(max_seq_len) 축 결과, 1500과 3001이
    peak_allocated_mb를 거의 완전히 동일하게 냈다(9746.87MB vs ~9740MB) - 원인을
    diagnose_seq_lengths.py로 확인해보니, tokenize_and_pad_batch(max_len=max_seq_len,
    include_zero_gene=False)가 "세포당 발현 유전자 수만큼만 채우고 모자라면
    패딩"하는 방식인데, 이 데이터셋(reference 7,844세포)의 세포당 발현 유전자 수가
    min=52, median=224, p95=658, **max=1337**로 이미 1500보다 작다. 즉
    max_seq_len >= 1337인 값은 전부 실질적으로 같은 조건("자를 필요 없음")이라
    500/1500/3001 세 점만으로는 진짜 "gene 수 -> 메모리" 관계를 볼 수 없었다
    (500만 실제로 잘라내는 값이었고 1500/3001은 둘 다 자연스러운 상한(1337)에서
    이미 정체됨).

    이 함수는 그 상한(1337) 아래쪽을 촘촘하게 훑어서 실제 스케일링 곡선을 그리기
    위한 grid: min(52)~max(1337) 사이에 대략 균등 분포된 값들 + 정확히 max(1337)
    자체(여기서부터 더 늘려도 효과 없어야 정상 - 1500 결과와 거의 같은 값이
    나오는지 재확인하는 용도). batch_size/precision/checkpointing은
    priority_grid()의 max_seq_len=500 조합과 맞춰서 고정(직접 비교 가능하게) -
    500 자체는 이미 priority_grid 결과에 있으므로 여기서 반복하지 않는다.
    """
    # min=52, median=224, p95=658, max=1337 (2026-09-17 diagnose_seq_lengths.py 실측)
    # 사이를 대략 균등하게 나눈 지점들 + 정확히 max(1337) 지점.
    seq_lens = [150, 300, 700, 900, 1100, 1337]
    return [
        {
            "precision": "fp16", "micro_batch_size": 8, "grad_accum_steps": 1,
            "activation_checkpointing": False, "max_seq_len": s,
        }
        for s in seq_lens
    ]


def effective_batch_size(override: Dict[str, Any]) -> Optional[int]:
    if "micro_batch_size" in override and "grad_accum_steps" in override:
        return override["micro_batch_size"] * override["grad_accum_steps"]
    return None


def scfoundation_gene_length_grid():
    """scFoundation 전용 gene-count sweep (2026-09-22 추가) - scGPT의 gene_length_grid()와
    같은 목적이지만 메커니즘은 다르다. scFoundation은 main_gene_selection()으로 항상
    고정 19264-길이 dense 벡터(0-padding)로 재배열되므로(scfoundation_adapter.py 참고),
    n_hvg_genes를 줄여도 모델에 들어가는 seq_len 자체는 안 줄어들 수 있다 -
    gatherData()가 실제로 0이 아닌 유전자만 골라 시퀀스를 줄이는지가 어댑터 docstring의
    미검증 가정 #4였는데, 이 grid 결과로 scGPT처럼 quadratic인지 거의 평평한지 확인한다.
    batch_size=8/precision=fp16/activation_checkpointing=False로 고정
    (checkpointing은 scfoundation_adapter 미지원이라 여기선 의미 없어 뺐다)."""
    base = dict(precision="fp16", micro_batch_size=8, grad_accum_steps=1,
                activation_checkpointing=False)
    points = [100, 300, 700, 1200, 2000, 3000]
    return [dict(base, max_seq_len=p) for p in points]
