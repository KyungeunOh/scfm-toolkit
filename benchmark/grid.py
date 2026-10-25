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


def scfoundation_priority_grid():
    """2026-09-22 gene_length_grid() 후속 실험 - '평평한 구간(300~1200) + 급격한
    quadratic 구간(1200~3000)' 전환점(elbow)을 정밀하게 잡고, scGPT처럼 batch에 대한
    OOM 경계도 확인한다 (교수님의 "config 자동 선택" 연구 질문에 직접 답하는 실험).

    1) elbow 정밀화: batch=8 고정, n_hvg_genes를 1200~2000 사이 100 단위로 촘촘히.
    2) batch-불변성 체크: 같은 elbow 구간을 batch=32에서도 반복 - floor가 모델
       가중치(배치 무관)에서 오는 거라면 elbow 위치가 batch와 무관해야 한다.
    3) OOM/batch 경계: gene=3000(=데이터셋 전체 유전자) 고정, batch=8→3552MB /
       batch=32→10898MB 실측을 선형 외삽하면 batch~64에서 20.7GB(카드 23.6GB의 88%),
       batch~80에서 25.6GB(초과)로 추정 - 48/64/80/96으로 브래킷을 잡는다.
    """
    combos = []
    elbow_points = [1300, 1400, 1500, 1600, 1700, 1800, 1900]
    for g in elbow_points:
        combos.append(dict(precision="fp16", micro_batch_size=8, grad_accum_steps=1,
                            activation_checkpointing=False, max_seq_len=g))
    for g in elbow_points:
        combos.append(dict(precision="fp16", micro_batch_size=32, grad_accum_steps=1,
                            activation_checkpointing=False, max_seq_len=g))
    for b in [48, 64, 80, 96]:
        combos.append(dict(precision="fp16", micro_batch_size=b, grad_accum_steps=1,
                            activation_checkpointing=False, max_seq_len=3000))
    return combos


def scgpt_checkpointing_oom_grid():
    """checkpointing=True 상태에서 scGPT의 실제 OOM 경계를 찾기 위한 coarse 탐색.
    기존 priority_grid는 batch<=28까지만(전부 성공, 5.5GB 이하) 테스트해서 진짜 경계를
    확인한 적이 없음. ckpt=False 쪽 batch당 증가율(~1.11GB/batch) 대비 ckpt=True는
    ~0.17GB/batch(1/6.5 수준)로 관찰됐으므로, 24GB 도달 예상 지점(batch~130 근방)을
    coarse하게 감싸는 5개 지점으로 탐색한다.
    """
    base = dict(precision="fp16", max_seq_len=3001, grad_accum_steps=1,
                activation_checkpointing=True)
    points = [64, 96, 128, 160, 192]
    return [dict(base, micro_batch_size=p) for p in points]


def scgpt_checkpointing_oom_grid():
    """checkpointing=True 상태에서 scGPT의 실제 OOM 경계를 찾기 위한 coarse 탐색.
    기존 priority_grid는 batch<=28까지만(전부 성공, 5.5GB 이하) 테스트해서 진짜 경계를
    확인한 적이 없음. ckpt=False 쪽 batch당 증가율(~1.11GB/batch) 대비 ckpt=True는
    ~0.17GB/batch(1/6.5 수준)로 관찰됐으므로, 24GB 도달 예상 지점(batch~130 근방)을
    coarse하게 감싸는 5개 지점으로 탐색한다.
    """
    base = dict(precision="fp16", max_seq_len=3001, grad_accum_steps=1,
                activation_checkpointing=True)
    points = [64, 96, 128, 160, 192]
    return [dict(base, micro_batch_size=p) for p in points]


def scfoundation_scope_lr_grid():
    """2026-10 교수님 후속 요청: scFoundation 학습 범위(unfrozen_layers)와 lr 변화 효과.
    batch=32 / max_seq_len=3001 / fp16 고정 (기존 2블록 베이스라인 batch=32 -> 10.6GB와
    직접 비교 가능). 순서: 메모리가 작은 조합부터 (full FT가 OOM이어도 앞 결과는 보존).
    2블록+lr=1e-4는 기존 베이스라인이 있어 제외하되, 2블록+lr=3e-5는 'lr만 바꾼 대조군'으로 포함.
    """
    base = dict(precision="fp16", micro_batch_size=32, grad_accum_steps=1,
                activation_checkpointing=False, max_seq_len=3001)
    scopes = [
        dict(scfoundation_unfrozen_layers=2, scfoundation_frozen_embeddings=True),
        dict(scfoundation_unfrozen_layers=4, scfoundation_frozen_embeddings=True),
        dict(scfoundation_unfrozen_layers=12, scfoundation_frozen_embeddings=True),
        dict(scfoundation_unfrozen_layers=12, scfoundation_frozen_embeddings=False),
    ]
    combos = []
    for scope in scopes:
        for lr in [1e-4, 3e-5]:
            if scope["scfoundation_unfrozen_layers"] == 2 and lr == 1e-4:
                continue
            combos.append(dict(base, lr=lr, **scope))
    return combos


def scfoundation_scope_repeat_grid():
    """scope_lr 결과(단일 실행)가 노이즈(F1 범위 ~0.05)와 구분되는지 확인하기 위한 반복.
    같은 GPU(A6000)에서 batch=32/lr=1e-4: 2블록 3회, 4블록 2회."""
    base = dict(precision="fp16", micro_batch_size=32, grad_accum_steps=1,
                activation_checkpointing=False, max_seq_len=3001, lr=1e-4)
    combos = []
    for _ in range(3):
        combos.append(dict(base, scfoundation_unfrozen_layers=2, scfoundation_frozen_embeddings=True))
    for _ in range(2):
        combos.append(dict(base, scfoundation_unfrozen_layers=4, scfoundation_frozen_embeddings=True))
    return combos


def scfoundation_scope_fit_grid():
    """교수님 질문4: 4블록/12블록/full FT를 12GiB 안에 넣을 수 있는 batch로 재측정.
    선형 외삽 예측: 4블록 b16 ~10.5GiB, 12블록 b4 ~8.7GiB, full b4 ~9.0GiB (검증 대상)."""
    base = dict(precision="fp16", grad_accum_steps=1, activation_checkpointing=False,
                max_seq_len=3001, lr=1e-4)
    return [
        dict(base, micro_batch_size=16, scfoundation_unfrozen_layers=4, scfoundation_frozen_embeddings=True),
        dict(base, micro_batch_size=4, scfoundation_unfrozen_layers=12, scfoundation_frozen_embeddings=True),
        dict(base, micro_batch_size=4, scfoundation_unfrozen_layers=12, scfoundation_frozen_embeddings=False),
    ]


def scfoundation_scope_mem_grid():
    """교수님 질문4 - 12GiB(12288MB) 경계 탐색용 메모리 전용 그리드 (epochs=1).
    1 epoch의 F1/accuracy는 의미 없음 - peak_allocated_mb만 본다.
    선형 외삽 예측(검증 대상): 4블록 b12~8.9 / b16~10.5 / b20~12.8GiB,
    12블록 b4~8.7 / b6~12.0 / b8~15.4GiB, full(emb 학습) b4~9.0 / b6~12.2GiB."""
    base = dict(precision="fp16", grad_accum_steps=1, activation_checkpointing=False,
                max_seq_len=3001, lr=1e-4, epochs=1)
    combos = []
    for b in (12, 16, 20):
        combos.append(dict(base, micro_batch_size=b, scfoundation_unfrozen_layers=4,
                           scfoundation_frozen_embeddings=True))
    for b in (4, 6, 8):
        combos.append(dict(base, micro_batch_size=b, scfoundation_unfrozen_layers=12,
                           scfoundation_frozen_embeddings=True))
    for b in (4, 6):
        combos.append(dict(base, micro_batch_size=b, scfoundation_unfrozen_layers=12,
                           scfoundation_frozen_embeddings=False))
    return combos


def scfoundation_scope_perf_grid():
    """12GiB 상한 안에 들어가는 조합의 10 epoch 성능 확인 (scope_mem 결과 기반).
    우선순위 순: full FT b4 -> 12블록(emb freeze) b4 -> 4블록 b16. 모두 lr=1e-4, epochs는 config(10).
    scope_mem의 1 epoch peak: full b4 8.20 / 12블록 b4 8.08 / 4블록 b16 10.01 GiB
    (10 epoch peak가 이와 같은지가 '1 epoch peak 가정'의 검증)."""
    base = dict(precision="fp16", grad_accum_steps=1, activation_checkpointing=False,
                max_seq_len=3001, lr=1e-4)
    return [
        dict(base, micro_batch_size=4, scfoundation_unfrozen_layers=12, scfoundation_frozen_embeddings=False),
        dict(base, micro_batch_size=4, scfoundation_unfrozen_layers=12, scfoundation_frozen_embeddings=True),
        dict(base, micro_batch_size=16, scfoundation_unfrozen_layers=4, scfoundation_frozen_embeddings=True),
    ]


def scfoundation_scope_accum_grid():
    """full FT를 12GiB 안에 넣되(micro-batch 4 -> ~8.6GiB) effective batch를 32로 유지.
    scope_perf(batch 4, accum 1)에서 F1이 0.63대로 낮았던 원인이 '범위'인지 'step 수/lr'인지 분리한다.
    micro_batch 4 x grad_accum 8 = effective 32 (베이스라인과 동일한 step 수)."""
    base = dict(precision="fp16", micro_batch_size=4, grad_accum_steps=8,
                activation_checkpointing=False, max_seq_len=3001,
                scfoundation_unfrozen_layers=12, scfoundation_frozen_embeddings=False)
    return [dict(base, lr=1e-4), dict(base, lr=3e-5)]


def scfoundation_scope_accum_repeat_grid():
    """scope_accum의 full FT(micro 4 x accum 8 = effective 32, lr=1e-4)를 2회 더 반복해
    n=3으로 만든다 (2블록 n=6, 4블록 n=3과 비교하기 위함)."""
    base = dict(precision="fp16", micro_batch_size=4, grad_accum_steps=8,
                activation_checkpointing=False, max_seq_len=3001,
                scfoundation_unfrozen_layers=12, scfoundation_frozen_embeddings=False, lr=1e-4)
    return [dict(base), dict(base)]


def scfoundation_scope_seed_grid():
    """scope_repeat의 2블록(batch 32, lr 1e-4) 설정을 그대로 두고 seed만 43/44/45로 바꾼다.
    (seed는 train/valid 분할을 바꾸며, 학습 난수는 별도로 고정되지 않는다)"""
    base = [c for c in scfoundation_scope_repeat_grid()
            if c.get("scfoundation_unfrozen_layers") == 2][0]
    return [dict(base, seed=s) for s in (43, 44, 45)]


# ---------------------------------------------------------------------------
# 교수님 2026-10-08 요청: reference를 type별 N개로 줄이고, scope별로 비교
# ---------------------------------------------------------------------------
_SUBSAMPLE_SCOPES = (
    # (이름, scfoundation_unfrozen_layers, scfoundation_frozen_embeddings)
    ("freeze", 0, True),    # encoder 전체 freeze + 분류 헤드만 학습
    ("last2", 2, True),     # 마지막 2 block + 헤드 (기존 기본값)
    ("last4", 4, True),     # 마지막 4 block + 헤드
    ("full", 12, False),    # 12 block + embedding 전부 (full fine-tuning)
)


def scfoundation_subsample_grid(ns=(10, 50), seeds=(42, 43, 44, 45, 46), epochs=10, freeze_eval=True):
    """scope 4 x N x seed. 같은 (N, seed)는 scope와 무관하게 같은 subsample을 쓴다.
    micro batch 8(config 기본값), accumulation 없음, 메모리 cap 없음(이번 실험은 성능 비교 목적)."""
    base = [c for c in scfoundation_scope_repeat_grid()
            if c.get("scfoundation_unfrozen_layers") == 2][0]
    out = []
    for s in seeds:
        for n in ns:
            for _name, u, fe in _SUBSAMPLE_SCOPES:
                out.append(dict(base, micro_batch_size=8, grad_accum_steps=1, epochs=epochs,
                                scfoundation_unfrozen_layers=u, scfoundation_frozen_embeddings=fe,
                                train_n_per_type=n, seed=s, seed_torch=True))
            if freeze_eval:
                # 추가 조건: 전체 freeze + encoder eval(dropout 끔). encoder dropout 0.1 효과를 분리하기 위함.
                out.append(dict(base, micro_batch_size=8, grad_accum_steps=1, epochs=epochs,
                                scfoundation_unfrozen_layers=0, scfoundation_frozen_embeddings=True,
                                scfoundation_encoder_eval=True,
                                train_n_per_type=n, seed=s, seed_torch=True))
    return out


def scfoundation_subsample_pilot_grid():
    """파일럿: N=10, seed=42, 전체 freeze / full FT / 전체 freeze+encoder eval 3개만."""
    return [c for c in scfoundation_subsample_grid(ns=(10,), seeds=(42,))
            if c["scfoundation_unfrozen_layers"] in (0, 12)]


def scfoundation_subsample_pilot2_grid():
    """파일럿2(수렴 점검, N=10, seed=42): 전체 freeze(dropout 켬/끔) x lr {3e-4, 1e-3, 3e-3} = 6개
    + 마지막 2/4 block (lr 1e-4) = 2개. 학습 곡선(train loss/acc)을 보고 freeze의 lr을 정하기 위함.
    (lr 선택은 train loss 기준으로만 하고 query F1은 선택에 쓰지 않는다.)"""
    out = []
    for c in scfoundation_subsample_grid(ns=(10,), seeds=(42,)):
        u = c["scfoundation_unfrozen_layers"]
        if u == 0:
            for lr in (3e-4, 1e-3, 3e-3):
                out.append(dict(c, lr=lr))
        elif u in (2, 4):
            out.append(c)
    return out


def scfoundation_subsample_pilot3_grid():
    """파일럿3(N=10, seed=42, 16개): 부분 fine-tuning scope의 학습이 수렴했는지 + frozen 구간 eval 효과.
    (a) 현재 프로토콜(frozen 구간도 train 모드): last2/last4 x lr {3e-4,1e-3,3e-3}, full x {3e-4,1e-3}
    (b) frozen 구간 eval: last2/last4 x lr {1e-4,3e-4,1e-3,3e-3}
    (lr 선택은 train loss 기준으로만 한다. query F1은 선택에 쓰지 않는다.)"""
    by_u = {c["scfoundation_unfrozen_layers"]: c
            for c in scfoundation_subsample_grid(ns=(10,), seeds=(42,), freeze_eval=False)}
    out = []
    for u, lrs in ((2, (3e-4, 1e-3, 3e-3)), (4, (3e-4, 1e-3, 3e-3)), (12, (3e-4, 1e-3))):
        for lr in lrs:
            out.append(dict(by_u[u], lr=lr))
    for u in (2, 4):
        for lr in (1e-4, 3e-4, 1e-3, 3e-3):
            out.append(dict(by_u[u], lr=lr, scfoundation_encoder_eval=True))
    return out


def scfoundation_subsample_diag_grid():
    """진단(N=10, seed=42, 3개): pilot3에서 query로 붕괴했던 설정이 '학습 데이터의 eval 모드 정확도'로
    미리 드러나는지 확인한다 (정상 설정 1개 포함)."""
    by_u = {c["scfoundation_unfrozen_layers"]: c
            for c in scfoundation_subsample_grid(ns=(10,), seeds=(42,), freeze_eval=False)}
    return [dict(by_u[2], lr=1e-3, scfoundation_encoder_eval=True),   # pilot3: F1 0.309 (붕괴)
            dict(by_u[2], lr=1e-3),                                   # pilot3: F1 0.079 (붕괴)
            dict(by_u[2], lr=3e-4, scfoundation_encoder_eval=True)]   # pilot3: F1 0.701 (정상)


# 본 실험 조건. lr은 pilot2/3(N=10, seed 42)에서 'train loss'만으로 정했다 (query F1은 쓰지 않음).
# 규칙: 발산하지 않은 run 중 마지막 train loss가 가장 낮은 lr. 단, 최저 loss와 10% 이내면 더 작은 lr.
_MAIN_CONDITIONS = (
    # (이름, unfrozen_layers, frozen_embeddings, encoder_eval(frozen 구간 eval), lr)
    ("freeze_eval", 0, True, True, 3e-3),
    ("last2_eval", 2, True, True, 3e-4),
    ("last4_eval", 4, True, True, 3e-4),
    ("full", 12, False, False, 1e-4),
    ("freeze_train", 0, True, False, 3e-3),    # 기존 프로토콜(frozen 구간도 train 모드) 민감도 확인용
    ("last2_train", 2, True, False, 3e-4),
    ("last4_train", 4, True, False, 3e-4),
)


def scfoundation_subsample_main_grid(ns=(10, 50), seeds=(42, 43, 44, 45, 46), epochs=10):
    """본 실험: 7개 조건 x N x seed = 70회. 같은 (N, seed)는 조건과 무관하게 같은 subsample."""
    base = [c for c in scfoundation_scope_repeat_grid()
            if c.get("scfoundation_unfrozen_layers") == 2][0]
    out = []
    for s in seeds:
        for n in ns:
            for _name, u, fe, ev, lr in _MAIN_CONDITIONS:
                c = dict(base, micro_batch_size=8, grad_accum_steps=1, epochs=epochs,
                         scfoundation_unfrozen_layers=u, scfoundation_frozen_embeddings=fe,
                         train_n_per_type=n, seed=s, seed_torch=True, lr=lr)
                if ev:
                    c["scfoundation_encoder_eval"] = True
                out.append(c)
    return out


def scfoundation_subsample_all_grid(seeds=(42, 43, 44), epochs=10):
    """N=전체 기준선: reference 7,844개를 전부 학습에 사용(validation 없음, 마지막 epoch), 본 실험과 같은 프로토콜.
    train_n_per_type=100000 은 'type당 최대 100000개' = 전부라는 뜻. 조건은 frozen eval 3개 + full FT = 4개."""
    return [c for c in scfoundation_subsample_main_grid(ns=(100000,), seeds=seeds, epochs=epochs)
            if c.get("scfoundation_encoder_eval") or c["scfoundation_unfrozen_layers"] == 12]


def scfoundation_subsample_all_lr1e4_grid(seeds=(42, 43, 44), epochs=10):
    """N=전체에서 lr=3e-4 의 부분 fine-tuning(2-block, 4-block, frozen eval)이 불안정했던 것의 재실행: lr=1e-4.
    사전 규칙(train 모드 acc - eval 모드 train acc > 0.2 이면 lr/3 재실행) 적용. 같은 조건 3 seed 전부 다시 돌린다."""
    return [dict(c, lr=1e-4) for c in scfoundation_subsample_all_grid(seeds=seeds, epochs=epochs)
            if c["scfoundation_unfrozen_layers"] in (2, 4)]


def scfoundation_subsample_n50_lr1e4_grid(seeds=(42, 43, 44, 45, 46), epochs=10):
    """민감도 분석: N=50에서 2-block, 4-block(frozen eval)을 lr=1e-4로. 본 실험(lr 3e-4)과 같은 세포 subsample(paired).
    목적: N=50의 'full 최고' 순위가 부분 scope의 lr 선택에 의존하는지 확인. 두 lr 결과를 모두 보고하며 lr 선택에는 쓰지 않는다."""
    return [dict(c, lr=1e-4) for c in scfoundation_subsample_main_grid(ns=(50,), seeds=seeds, epochs=epochs)
            if c.get("scfoundation_encoder_eval") and c["scfoundation_unfrozen_layers"] in (2, 4)]
