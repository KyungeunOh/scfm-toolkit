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


def scfoundation_bn_diag_grid(seeds=(42,), epochs=10):
    """N=전체에서 2/4-block lr 3e-4 의 '학습은 정상, 평가에서 붕괴' 원인 진단 (교수님 2026-10-25 요청).
    같은 설정(본 실험 N=전체 anchor, frozen eval)을 bn_diag=True 로 다시 돌린다: 4-block 3e-4(가장 크게 붕괴), 2-block 3e-4,
    대조군 4-block 1e-4. query 는 진단에 쓰지 않는다. seed_torch 라도 GPU 비결정성 때문에 같은 붕괴가 재현되는지는 보장되지 않는다."""
    base = scfoundation_subsample_all_grid(seeds=seeds, epochs=epochs)
    out = []
    for s in seeds:
        for u, lr in ((4, 3e-4), (2, 3e-4), (4, 1e-4)):
            c = [c for c in base if c["seed"] == s and c["scfoundation_unfrozen_layers"] == u and c.get("scfoundation_encoder_eval")][0]
            tag = "nall_last%d_lr%g_s%d" % (u, lr, s)
            out.append(dict(c, lr=lr, bn_diag=True, bn_diag_tag=tag, bn_diag_out="benchmark_results/bn_diag/%s.json" % tag))
    return out


def scfoundation_bn_recal_grid(seeds=(42,), epochs=10):
    """BN running 통계 재계산 평가 (bn_diag 결과의 후속, 2026-10-25). bn_diag grid 와 같은 3개 조합(N=전체, seed 42):
    4-block 3e-4, 2-block 3e-4, 4-block 1e-4(대조). bn_recalibrate=True 이면 macro_f1/accuracy 열은 재계산 통계,
    원래 통계 결과는 notes(원래통계 f1=, acc=)와 로그([BNRECAL])에 같이 기록된다. query 는 통계 계산에 쓰지 않는다."""
    base = scfoundation_subsample_all_grid(seeds=seeds, epochs=epochs)
    out = []
    for s in seeds:
        for u, lr in ((4, 3e-4), (2, 3e-4), (4, 1e-4)):
            c = [c for c in base if c["seed"] == s and c["scfoundation_unfrozen_layers"] == u and c.get("scfoundation_encoder_eval")][0]
            out.append(dict(c, lr=lr, bn_recalibrate=True))
    return out


# ---------------------------------------------------------------------------
# 교수님 2026-10-18 요청 1번: scope 별 lr 변화 (N=10/50). 2026-10-25 BN 재계산 평가 도입 이후 설계.
# ---------------------------------------------------------------------------
def scfoundation_lr_grid(ns, seeds=(42, 43, 44, 45, 46), epochs=10, holdout_n=2000):
    """scope 별 lr 격자. 조건은 본 실험(main grid)과 같은 프로토콜(frozen 구간 eval, batch 8, seed_torch, 마지막 epoch).
    학습 block 이 있는 scope(last2/last4/full)는 bn_recalibrate=True (원래/재계산 평가를 한 번에 기록),
    freeze 는 encoder 전체가 eval 이라 BN 통계가 일치하므로 끈다. 모든 run 에 holdout 평가(학습에 쓰지 않은 reference 최대 holdout_n 개).
    각 scope 의 lr: (새 격자점 + 재계산 평가를 위한 기존 격자점 재실행)
      freeze : 3e-4, 1e-3, 1e-2  (3e-3 은 기존)
      last2/last4 : 3e-5 (모든 N), 1e-4 (N=10 만 새로, N=50 은 기존 재실행), 3e-4 (기존 재실행)
      full   : 3e-5, 3e-4 (새로), 1e-4 (기존 재실행)
    같은 (N, seed)는 조건과 무관하게 같은 학습 subsample, 같은 holdout."""
    base = [c for c in scfoundation_scope_repeat_grid() if c.get("scfoundation_unfrozen_layers") == 2][0]
    specs = (
        # (unfrozen_layers, frozen_embeddings, encoder_eval, bn_recalibrate, lrs)
        (0, True, True, False, (3e-4, 1e-3, 1e-2)),
        (2, True, True, True, (3e-5, 1e-4, 3e-4)),
        (4, True, True, True, (3e-5, 1e-4, 3e-4)),
        (12, False, False, True, (3e-5, 1e-4, 3e-4)),
    )
    out = []
    for s in seeds:
        for n in ns:
            for u, fe, ev, recal, lrs in specs:
                for lr in lrs:
                    c = dict(base, micro_batch_size=8, grad_accum_steps=1, epochs=epochs,
                             scfoundation_unfrozen_layers=u, scfoundation_frozen_embeddings=fe,
                             train_n_per_type=n, seed=s, seed_torch=True, lr=lr,
                             holdout_n=holdout_n, holdout_exclude_n=50)
                    if ev:
                        c["scfoundation_encoder_eval"] = True
                    if recal:
                        c["bn_recalibrate"] = True
                    out.append(c)
    return out


def scfoundation_lr_n50_grid(seeds=(42, 43, 44, 45, 46), epochs=10):
    """N=50 lr sweep: 5 seed x (freeze 3 + last2 3 + last4 3 + full 3) = 60회."""
    return scfoundation_lr_grid(ns=(50,), seeds=seeds, epochs=epochs)


def scfoundation_lr_n10_grid(seeds=(42, 43, 44, 45, 46), epochs=10):
    """N=10 lr sweep: 5 seed x 12 = 60회."""
    return scfoundation_lr_grid(ns=(10,), seeds=seeds, epochs=epochs)


# ---------------------------------------------------------------------------
# 2026-10-27: Pancreas 확인, 2-block lr 확장, 같은 seed 반복(요청 3번) 통제 실험.
# 모두 scfoundation_lr_grid 에서 조건만 골라 쓰므로 프로토콜은 lr sweep 과 같다.
# ---------------------------------------------------------------------------
_PANCREAS_RULE_LR = {0: 1e-3, 2: 1e-4, 4: 1e-4, 12: 1e-4}   # brain atlas lr sweep 에서 정한 사전 규칙(같은 sweep 에서 만든 것이라 낙관적)


def _pick_lr_runs(us, ns, seeds, epochs):
    out = []
    for c in scfoundation_lr_grid(ns=ns, seeds=seeds, epochs=epochs):
        u = c["scfoundation_unfrozen_layers"]
        if u in us and c["lr"] == _PANCREAS_RULE_LR[u]:
            out.append(c)
    return out


def scfoundation_pancreas_smoke_grid(epochs=10):
    """Pancreas 경로/시간/메모리 확인용 2회: N=50, seed 42, freeze lr 1e-3 + 2-block lr 1e-4."""
    out = []
    for c in scfoundation_lr_grid(ns=(50,), seeds=(42,), epochs=epochs):
        u, lr = c["scfoundation_unfrozen_layers"], c["lr"]
        if (u == 0 and lr == 1e-3) or (u == 2 and lr == 1e-4):
            out.append(c)
    return out


def scfoundation_pancreas_rule_grid(seeds=(42, 43, 44, 45, 46), epochs=10):
    """사전 규칙 검증: freeze(1e-3), 2-block(1e-4), 4-block(1e-4) x N=10,50 x 5 seed = 30회."""
    return _pick_lr_runs((0, 2, 4), (10, 50), seeds, epochs)


def scfoundation_pancreas_rule_full_grid(seeds=(42, 43, 44, 45, 46), epochs=10):
    """사전 규칙 검증 full scope(1e-4) x N=10,50 x 5 seed = 10회. brain atlas 에서 N=50 full 이 약 23.7GB 라 GPU 가 비어 있을 때만."""
    return _pick_lr_runs((12,), (10, 50), seeds, epochs)


def _last2_lr_variants(lrs, ns, seeds, epochs):
    out = []
    for c in scfoundation_lr_grid(ns=ns, seeds=seeds, epochs=epochs):
        if c["scfoundation_unfrozen_layers"] == 2 and c["lr"] == 3e-4:
            for lr in lrs:
                out.append(dict(c, lr=lr))
    return out


def scfoundation_last2_hi_grid(seeds=(42, 43, 44, 45, 46), epochs=10):
    """brain atlas 2-block lr 1e-3: N=10,50 x 5 seed = 10회 (lr sweep 의 최고점이 3e-4 격자 끝이라 위쪽 확인)."""
    return _last2_lr_variants((1e-3,), (10, 50), seeds, epochs)


def scfoundation_last2_hi3_grid(seeds=(42, 43, 44, 45, 46), epochs=10):
    """brain atlas 2-block lr 3e-3: 10회 (1e-3 결과를 본 뒤 필요할 때)."""
    return _last2_lr_variants((3e-3,), (10, 50), seeds, epochs)


def scfoundation_seed_repeat_grid(repeats=3, epochs=10):
    """교수님 요청 3번 통제 실험: 같은 설정(2-block lr 1e-4, N=50, seed 42)을 seed_torch True 로 repeats 회, False 로 repeats 회."""
    base = [c for c in scfoundation_lr_grid(ns=(50,), seeds=(42,), epochs=epochs)
            if c["scfoundation_unfrozen_layers"] == 2 and c["lr"] == 1e-4][0]
    out = []
    for st in (True, False):
        for _ in range(repeats):
            out.append(dict(base, seed_torch=st))
    return out


def scfoundation_last2_anchor_grid(seeds=(42, 43, 44, 45, 46), epochs=10):
    """brain atlas 2-block lr 3e-4 재실행(기준점): lr_last2_hi(1e-3)와 같은 GPU 에서 비교하기 위해 N=10,50 x 5 seed = 10회.
    설정은 lr sweep 의 2-block 3e-4 와 동일(같은 subsample, holdout, seed_torch). 기존 결과는 RTX 3090, 이 grid 는 실행한 GPU 에서 새로 얻는다."""
    return _last2_lr_variants((3e-4,), (10, 50), seeds, epochs)


def scfoundation_pancreas_last2_3e4_grid(seeds=(42, 43, 44, 45, 46), epochs=10):
    """Pancreas 민감도 확인: 2-block lr 3e-4 x N=10,50 x 5 seed = 10회. pancreas_rule 의 2-block(1e-4)이 brain atlas 최고 lr(3e-4)보다
    낮았던 점이 N=10 결과에 영향을 주었는지 보기 위한 것. 새 최적 lr 탐색이 아니다(다른 scope 는 이 lr 로 다시 돌리지 않음)."""
    out = []
    for c in scfoundation_lr_grid(ns=(10, 50), seeds=seeds, epochs=epochs):
        if c["scfoundation_unfrozen_layers"] == 2 and c["lr"] == 3e-4:
            out.append(c)
    return out


def scfoundation_seed_repeat_dropout_grid(epochs=10):
    """교수님 요청 3번 보강(고정 block eval 모드 단독 효과): brain atlas 2-block lr 1e-4, N=50, seed 42, 난수 고정 없음(seed_torch=False).
    eval 모드 켬 2회(기존 seed_repeat 의 3회와 합쳐 5회) + eval 모드 끔(frozen 구간도 dropout 켬, 10/6~10/14 기존 방식) 5회 = 7회.
    앞 2개가 eval 켬, 뒤 5개가 eval 끔. 구분은 CSV 의 finetune_method(EncEval 포함 여부)로 한다."""
    base = [c for c in scfoundation_lr_grid(ns=(50,), seeds=(42,), epochs=epochs)
            if c["scfoundation_unfrozen_layers"] == 2 and c["lr"] == 1e-4][0]
    out = [dict(base, seed_torch=False) for _ in range(2)]
    out += [dict(base, seed_torch=False, scfoundation_encoder_eval=False) for _ in range(5)]
    return out


_PANCREAS_LR_DONE = {(0, 1e-3), (2, 1e-4), (2, 3e-4), (4, 1e-4), (12, 1e-4)}   # pancreas_rule, pancreas_rule_full, pancreas_last2_3e4 로 이미 돌린 (scope, lr)


def scfoundation_pancreas_lr_rest_grid(seeds=(42, 43, 44, 45, 46), epochs=10):
    """Pancreas 자체 lr 격자의 나머지: brain atlas lr sweep 과 같은 격자(freeze 3e-4/1e-3/1e-2, 2/4-block/full 3e-5/1e-4/3e-4)
    x N=10,50 x 5 seed = 120회 중 이미 돌린 5개 (scope, lr) 조합(50회)을 뺀 70회."""
    return [c for c in scfoundation_lr_grid(ns=(10, 50), seeds=seeds, epochs=epochs)
            if (c["scfoundation_unfrozen_layers"], c["lr"]) not in _PANCREAS_LR_DONE]


_PANCREAS_LR_EDGE = (
    # (N, unfrozen_layers, 새 lr): Pancreas lr 격자(120회)에서 평균 C 최고 lr 이 격자 끝에 걸린 설정을 바깥쪽으로 확장한다.
    (10, 2, (1e-3,)), (10, 4, (1e-3,)), (10, 12, (1e-3,)),   # N=10: 최고 3e-4 (격자 위쪽 끝)
    (50, 0, (1e-4, 3e-5)),                                    # N=50 freeze: 최고 3e-4 (아래쪽 끝)
    (50, 12, (1e-5,)),                                        # N=50 full: 최고 3e-5 (아래쪽 끝)
)


def scfoundation_pancreas_lr_edge_grid(seeds=(42, 43, 44, 45, 46), epochs=10):
    """Pancreas lr 격자 가장자리 확장: _PANCREAS_LR_EDGE 의 (N, scope, lr) x 5 seed = 30회.
    같은 (N, scope, seed)의 scfoundation_lr_grid 설정을 복사하고 lr 만 바꾼다
    (같은 (N, seed)는 같은 학습 subsample, frozen eval, BN 재계산, holdout 유지)."""
    base = scfoundation_lr_grid(ns=(10, 50), seeds=seeds, epochs=epochs)
    existing = {(c["train_n_per_type"], c["scfoundation_unfrozen_layers"], c["lr"]) for c in base}
    out = []
    for s in seeds:
        for n, u, lrs in _PANCREAS_LR_EDGE:
            tmpl = [c for c in base if c["train_n_per_type"] == n
                    and c["scfoundation_unfrozen_layers"] == u and c["seed"] == s][0]
            for lr in lrs:
                assert (n, u, lr) not in existing, (n, u, lr)   # 이미 돌린 조합이 섞이지 않게
                out.append(dict(tmpl, lr=lr))
    return out


def scfoundation_pancreas_ckpt_grid(seeds=(42, 43, 44, 45, 46), epochs=10):
    """Pancreas N=50 full lr 1e-5 + activation checkpointing (2026-11-04).
    pancreas_lr_edge 의 같은 설정(N=50, full, lr 1e-5, 5 seed)을 그대로 가져와 activation_checkpointing=True 만 추가:
    같은 GPU(A6000)의 기존 비ckpt 행과 seed 별 동등성을 비교하고, 2080 Ti(11GB)에서는 완주 여부를 본다."""
    edge = [c for c in scfoundation_pancreas_lr_edge_grid(seeds=seeds, epochs=epochs)
            if c["train_n_per_type"] == 50 and c["scfoundation_unfrozen_layers"] == 12]
    assert len(edge) == len(seeds), len(edge)
    return [dict(c, activation_checkpointing=True) for c in edge]


def scfoundation_pancreas_ckpt_11gb_grid(seeds=(42, 43, 44, 45, 46), epochs=10):
    """pancreas_ckpt 와 같은 설정, 별도 CSV 파일용 alias (2080 Ti)."""
    return scfoundation_pancreas_ckpt_grid(seeds=seeds, epochs=epochs)


_LR_ALL_SPECS = (
    # (unfrozen_layers, lrs): N=10/50 lr sweep(scfoundation_lr_grid)과 같은 격자
    (0, (3e-4, 1e-3, 1e-2)),
    (2, (3e-5, 1e-4, 3e-4)),
    (4, (3e-5, 1e-4, 3e-4)),
    (12, (3e-5, 1e-4, 3e-4)),
)


def scfoundation_lr_all_grid(seeds=(42, 43, 44), epochs=10):
    """N=전체(reference 7,844개 전부, validation 없음, 마지막 epoch) lr 격자 (교수님 2026-11-01 우선 실험 1번).
    scfoundation_subsample_all_grid 의 scope 별 설정에 lr 을 덮어쓰고, 학습 block 이 있는 scope 는 bn_recalibrate=True
    (원래/재계산 평가를 한 번에 기록). freeze 는 encoder 전체가 eval 이라 재계산을 끈다. 12조합 x seed 수."""
    base = scfoundation_subsample_all_grid(seeds=seeds, epochs=epochs)
    out = []
    for s in seeds:
        for u, lrs in _LR_ALL_SPECS:
            cs = [c for c in base if c["seed"] == s and c["scfoundation_unfrozen_layers"] == u]
            assert len(cs) == 1, (s, u, len(cs))
            for lr in lrs:
                c = dict(cs[0], lr=lr)
                if u > 0:
                    c["bn_recalibrate"] = True
                out.append(c)
    return out


def scfoundation_lr_all_s42_grid(epochs=10):
    return scfoundation_lr_all_grid(seeds=(42,), epochs=epochs)


def scfoundation_lr_all_s43_grid(epochs=10):
    return scfoundation_lr_all_grid(seeds=(43,), epochs=epochs)


def scfoundation_lr_all_s44_grid(epochs=10):
    return scfoundation_lr_all_grid(seeds=(44,), epochs=epochs)


def scfoundation_lr_all_low_grid(seeds=(42, 43, 44), epochs=10):
    """Brain N=전체 full 의 격자 끝(3e-5) 바깥 확인 (2026-11-03): lr 1e-5, bn_recalibrate 켬."""
    base = scfoundation_subsample_all_grid(seeds=seeds, epochs=epochs)
    out = []
    for s in seeds:
        cs = [c for c in base if c["seed"] == s and c["scfoundation_unfrozen_layers"] == 12]
        assert len(cs) == 1, (s, len(cs))
        out.append(dict(cs[0], lr=1e-5, bn_recalibrate=True))
    return out


def _fit_full_nall(shapes, seed=42, epochs=10):
    base = scfoundation_subsample_all_grid(seeds=(seed,), epochs=epochs)
    cs = [c for c in base if c["scfoundation_unfrozen_layers"] == 12]
    assert len(cs) == 1, len(cs)
    return [dict(cs[0], lr=3e-5, bn_recalibrate=True, micro_batch_size=mb, grad_accum_steps=ga) for mb, ga in shapes]


def scfoundation_fit_real24_grid(epochs=10):
    """실제 24GB 카드(RTX 3090)에서 full N=전체 lr 3e-5 가 기본 micro-batch 8 로 완주하는지 (2026-11-03)."""
    return _fit_full_nall(((8, 1),), epochs=epochs)


def scfoundation_fit_real11_grid(epochs=10):
    """실제 11GB 카드(RTX 2080 Ti)에서 full N=전체 lr 3e-5: effective batch 8 유지, micro-batch 4x2, 2x4 (2026-11-03)."""
    return _fit_full_nall(((4, 2), (2, 4)), epochs=epochs)


def scfoundation_bn_recal_rerun_grid(seeds=(42,), epochs=10):
    """10/25 bn_recal 과 같은 3개 설정을 별도 CSV 로 다시 (GPU 가설 확인, 2026-11-03)."""
    return scfoundation_bn_recal_grid(seeds=seeds, epochs=epochs)


def scfoundation_micro_effect_grid(seeds=(42, 43, 44), epochs=10):
    """full N=전체 lr 3e-5, effective batch 8 고정, micro-batch 4x2 / 2x4 만 바꿔 같은 GPU(A6000)에서 반복 (2026-11-03).
    8x1 기준은 lr_all_s42/43/44 에 이미 있음. seed 우선 순서(중간에 끊겨도 seed 단위로 쌍 비교 가능)."""
    out = []
    for s in seeds:
        out += _fit_full_nall(((4, 2), (2, 4)), seed=s, epochs=epochs)
    return out


def scfoundation_ckpt_check_grid(seeds=(42,), epochs=10):
    """activation checkpointing 을 켠 full N=전체 lr 3e-5, micro 8x1(기본 batch). 같은 GPU 의 lr_all_s4x 결과와 비교해
    수치 동등성, 메모리(peak_allocated), 시간을 확인한다. 11GB 카드에서도 이 그리드로 8x1 완주를 시험한다."""
    out = []
    for s in seeds:
        out += [dict(c, activation_checkpointing=True) for c in _fit_full_nall(((8, 1),), seed=s, epochs=epochs)]
    return out


def scfoundation_ckpt_check_24gb_grid():
    """3090(24GB)에서 ckpt 8x1 seed 42: fit_real24 (같은 GPU, 비ckpt)와 동등성 비교 (2026-11-04)."""
    return scfoundation_ckpt_check_grid(seeds=(42,))


def scfoundation_ckpt_more_11gb_grid():
    """2080 Ti(11GB)에서 ckpt 8x1 seed 43, 44 (CSV 분리용 alias, 2026-11-04)."""
    return scfoundation_ckpt_more_grid()


def scfoundation_ckpt_scope_grid(seeds=(42, 43, 44), epochs=10):
    """일부 block만 학습하는 scope(2-block, 4-block, lr 3e-5)의 activation checkpointing 동등성.
    lr_all_s42/43/44 의 같은 (seed, scope, lr) 설정에 activation_checkpointing=True 만 추가 (A6000 기준 비교용)."""
    base = scfoundation_lr_all_grid(seeds=seeds, epochs=epochs)
    out = []
    for c in base:
        if c["scfoundation_unfrozen_layers"] in (2, 4) and c["lr"] == 3e-5:
            out.append(dict(c, activation_checkpointing=True))
    assert len(out) == 2 * len(seeds), len(out)
    return out


def scfoundation_lr_all_lower_grid(seeds=(42, 43, 44), epochs=10):
    """Brain N=전체 full: lr 3e-6 (lr_all_low 1e-5 보다 한 칸 더 아래, 2026-11-04), bn_recalibrate 켬."""
    base = scfoundation_subsample_all_grid(seeds=seeds, epochs=epochs)
    out = []
    for s in seeds:
        cs = [c for c in base if c["seed"] == s and c["scfoundation_unfrozen_layers"] == 12]
        assert len(cs) == 1, (s, len(cs))
        out.append(dict(cs[0], lr=3e-6, bn_recalibrate=True))
    return out


def scfoundation_ckpt_more_grid(epochs=10):
    """ckpt_check(seed 42) 통과 후 seed 43, 44 추가."""
    return scfoundation_ckpt_check_grid(seeds=(43, 44), epochs=epochs)
