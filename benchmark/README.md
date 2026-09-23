# memory-benchmark 브랜치

2026-09 이제근 교수님 피드백(BioLLM/scPEFT 대비 차별화 필요, "모델+데이터+GPU
메모리를 주면 OOM 없는 설정을 자동으로 고를 수 있는가") 대응용 실험 코드.
`main` 브랜치(4개 mode GPU 검증 완료 상태)는 건드리지 않고 이 브랜치에서만
작업한다.

## 지금 이 브랜치가 하는 일 / 안 하는 일

**한다**: scGPT/scFoundation에 대해 precision(fp32/fp16/bf16), micro-batch size,
gradient accumulation, activation checkpointing, 입력 gene 수를 조합해 돌리면서
GPU peak memory·시간·macro-F1을 CSV 하나에 기록하는 "Full fine-tuning" 축의
측정 인프라.

**안 한다** (의도적으로 범위 밖): LoRA/scPEFT 구현. scfm-toolkit 안에 새로
구현하지 않고 scPEFT 공식 코드([laolintou/scPEFT](https://github.com/laolintou/scPEFT))를
scGPT/scFoundation에 그대로 적용해 비교군(②공식 설정 ③수동 조정 설정)으로 쓰는
방향으로 2026-09-10 교수님께 이미 보고함 - `benchmark/grid.py`의 축에도
lora_rank가 없다.

## 구조

```
benchmark/
  run_log.py                CSV 로깅 스키마(RUN_LOG_COLUMNS) + append 함수
                             (구 이름 logging.py -> 표준 logging 모듈과 이름 충돌해서 rename)
  memory_probe.py           GPU peak memory 측정 + OOM을 "실패"가 아니라 데이터
                             포인트로 다루는 컨텍스트 매니저, memory budget 흉내
  grid.py                   성긴 그리드(coarse) -> OOM 경계 근처 정밀 그리드(refine).
                             priority_grid()는 144개 coarse 대신 smoke 결과 기반
                             축소 조합(~17개, 2026-09-16 추가)
  run_scgpt_sweep.py         scGPT sweep 러너 (src/run.py의 Step 3~9 재사용).
                             기본적으로 조합마다 별도 프로세스로 격리해서 실행
                             (아래 "조합별 프로세스 격리" 참고)
  run_scfoundation_sweep.py  scFoundation sweep 러너 (run.py 미사용 - 이유는
                             src/adapters/scfoundation_adapter.py 참고). 프로세스
                             격리 구조는 scGPT와 동일하게 맞춰둠(아직 GPU 미검증)
  summarize_results.py      CSV를 표로 요약 + status별 개수 + checkpointing on/off
                             peak_allocated_mb 비교 (pandas 필요 - 컨테이너 안에서
                             --entrypoint python으로 실행할 것, 호스트에 pandas 없음)
  labmeeting_report_*.md    랩미팅 발표용 정리 (배경/시도/결과/다음 단계)

src/adapters/scfoundation_adapter.py   새 모델 adapter (아래 "검증 상태" 참고)
src/adapters/scgpt_adapter.py          기존 파일에 precision/activation_checkpointing
                                        옵션 + nan/inf loss 진단 카운터 추가 (기본값은
                                        기존 동작과 100% 동일 - cfg에 새 키를 안 넣으면
                                        기존 4개 GPU-validated mode 동작이 바뀌지 않음)
```

## 검증 상태 (2026-09-16 기준)

**scGPT는 GPU에서 첫 smoke test(5개 조합)를 완료했다** — 결과와 해석은
`labmeeting_report_2026-09-16.md`, 원본 CSV는 서버의
`benchmark_results/scgpt_sweep_smoke.csv` 참고. 핵심: batch=32는 activation
checkpointing 없이 OOM(22.65GB, RTX 3090 24GB), checkpointing 켜면 6.17GB로
성공 — "batch/checkpointing 조정만으로 OOM 경계가 달라지는가"에 대한 1차 증거 확보.

**아직 원인 미확인인 이상치 2건** (다음 실행에서 확인할 것):
1. batch=1 조합의 accuracy(2.3%)가 18종 랜덤 기대값(≈5.5%)보다 낮음. batch=1
   특유의 학습 불안정 또는 fp16 loss가 nan/inf로 발산했을 가능성 — 이번에
   `scgpt_adapter.py`의 `finetune()`에 nan/inf loss 스텝 카운터를 추가했으니
   (`Epoch N/M: loss가 nan/inf였던 스텝 X/Y개` 경고 로그), 다음 실행에서 이 로그가
   찍히는지로 원인을 좁힐 수 있다.
2. checkpointing=True 조합들의 `peak_reserved_mb`(21.8GB)가 바로 직전 OOM
   조합의 reserved(23.5GB)와 비슷하게 높게 나옴 — allocated(6.2GB)와 큰 격차.
   5개 조합을 한 프로세스 안에서 연달아 돌린 탓에 GPU 메모리 캐시가 조합 간
   완전히 반납되지 않았을 가능성 → **조합별 프로세스 격리를 기본 동작으로
   변경**해서 해결 시도(아래 참고). 다음 실행에서 `peak_reserved_mb`가
   `peak_allocated_mb`에 더 가깝게 나오는지 확인할 것.

scFoundation adapter는 **아직 GPU에서 한 번도 실행 검증되지 않았다**(체크포인트/
repo 준비 전). 서버에서 처음 실행할 때 확인 순서:

1. scGPT는 검증 완료 — 이제 `--grid coarse`(축소판) 또는 `refine_between()`으로
   경계 정밀화 단계로 넘어갈 것.
2. **scFoundation adapter** - 위험도가 가장 높다(`src/adapters/scfoundation_adapter.py`
   모듈 docstring의 "검증 상태" 절 참고). 체크포인트/repo부터 준비:
   - `git clone https://github.com/biomap-research/scFoundation.git`
   - 체크포인트는 GitHub에 없음 - 원본 repo README의 SharePoint 링크에서 별도 다운로드
   - `pip install local_attention` (requirements-scfoundation.txt) - scGPT
     스택과 충돌하는지 먼저 별도로 확인, 괜찮으면 requirements.txt에 합칠지 결정
   - `config/config_scfoundation.example.yaml`을 복사해서 경로 채운 뒤
     `run_scfoundation_sweep.py --grid smoke`의 **첫 조합 하나만** 먼저 시도

## 조합별 프로세스 격리 (2026-09-16 추가)

`run_scgpt_sweep.py`/`run_scfoundation_sweep.py`는 이제 **기본적으로** 조합마다
자기 자신을 `--override-json`으로 재호출해서 완전히 새 프로세스/CUDA 컨텍스트에서
실행한다 — 위 이상치 2번(peak_reserved 오염 의심) 때문에 추가함. 대가는 Step
3~5(데이터 로드/vocab/전처리)를 조합마다 다시 실행하는 것(이 데이터셋 기준
수십 초 수준, 감수할 만함).

예전처럼 한 프로세스 안에서 grid 전체를 도는 방식은 `--no-isolate`로 여전히
쓸 수 있다(디버깅/빠른 반복 확인용 — 이 경우 `peak_reserved_mb` 비교는 다시
신뢰할 수 없어짐, `peak_allocated_mb`는 두 방식 다 신뢰 가능).

## 실행 예시

```bash
# scGPT (기본: 조합별 프로세스 격리)
python benchmark/run_scgpt_sweep.py \
    --config config/config.yaml \
    --grid smoke \
    --out benchmark_results/scgpt_sweep.csv

# priority: smoke 결과 기반 축소 조합 (144개 coarse 대신, ~17개)
python benchmark/run_scgpt_sweep.py \
    --config config/config.yaml \
    --grid priority \
    --out benchmark_results/scgpt_sweep_priority.csv

# 격리 끄고 예전처럼 한 프로세스 안에서 (디버깅용)
python benchmark/run_scgpt_sweep.py \
    --config config/config.yaml --grid smoke \
    --out benchmark_results/scgpt_sweep.csv --no-isolate

# "12GB memory budget" 흉내 (실제 GPU는 이보다 커야 함, 예: A6000 48GB에서)
python benchmark/run_scgpt_sweep.py \
    --config config/config.yaml --grid smoke \
    --out benchmark_results/scgpt_sweep_12gb_budget.csv \
    --memory-budget-gb 12

# scFoundation (경로 준비 후)
python benchmark/run_scfoundation_sweep.py \
    --config config/config_scfoundation.yaml \
    --grid smoke \
    --out benchmark_results/scfoundation_sweep.csv
```

결과는 지정한 CSV 하나에 계속 append된다(교수님이 요청하신 "여러 실행 결과를
사람이 수동으로 옮기지 않고 하나로 합칠 수 있어야 한다" 요구사항). "12GB
memory budget"처럼 흉내낸 조건은 CSV의 `memory_budget_is_simulated=True`와
실제 GPU 이름(`gpu_name`)이 항상 같이 남아서 실제 GPU와 혼동되지 않는다.

## 스키마 마이그레이션 (2026-09-16: `lr` 컬럼 추가)

`run_log.py`의 `append_row()`는 기존 CSV의 헤더가 지금 `RUN_LOG_COLUMNS`와
한 글자라도 다르면 조용히 넘어가지 않고 **에러를 내고 멈춘다**(컬럼이 밀려서
잘못된 값이 들어가는 걸 막기 위한 의도적 설계). batch=1 majority-class
collapse 원인이 lr/batch 불일치로 확인된 뒤, `priority_grid()`에서 배치별로
lr을 바꿔 테스트하려고 `lr` 컬럼을 스키마에 추가했다 — 즉 **기존에 이미 만든
`scgpt_sweep_*.csv` 파일에 이 시점부터 이어서 append하려면, 그 CSV에도 먼저
`lr` 컬럼을 넣어줘야 한다.**

기존 smoke CSV(5개 조합 모두 `config/config.yaml`의 기본값 `lr: 0.0001`로
실행됨)에 대해 서버에서 한 번만 실행하면 되는 마이그레이션:

```bash
cd benchmark_results   # scgpt_sweep_smoke.csv가 있는 디렉토리
python3 - <<'EOF'
import csv
from pathlib import Path

path = Path("scgpt_sweep_smoke.csv")
rows = list(csv.reader(open(path, newline="")))
header, data = rows[0], rows[1:]

if "lr" in header:
    print("이미 lr 컬럼이 있음 - 마이그레이션 불필요")
else:
    idx = header.index("activation_checkpointing") + 1
    header.insert(idx, "lr")
    for row in data:
        row.insert(idx, "0.0001")  # 이 CSV의 5개 조합은 전부 기본 lr로 실행됨
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data)
    print(f"완료: {path} 에 lr 컬럼 추가(전부 0.0001)")
EOF
```

CSV를 새로 만드는 경우(`--out`에 아직 없는 파일명, 예: `scgpt_sweep_priority.csv`)는
해당 없음 — 처음부터 새 스키마로 헤더가 만들어진다.

## 남은 작업

- [x] scGPT: `--grid smoke` 서버 실행 검증 (2026-09-16 완료)
- [x] scGPT: activation checkpointing이 실제로 peak memory를 줄이는지 확인
      (22.65GB → 6.17GB, batch=32 기준)
- [ ] batch=1 accuracy 이상치(2.3%) 원인 확인 — nan/inf loss 진단 로그 추가함,
      다음 실행에서 재현되는지 확인
- [ ] 조합별 프로세스 격리 후 `peak_reserved_mb`가 신뢰 가능한 값으로 나오는지 확인
- [ ] scFoundation: 체크포인트/repo 준비 + 첫 forward/finetune 성공 확인
- [ ] scFoundation: activation checkpointing 지원 (지금은 encoder 내부 구조
      미확인으로 미지원 - `select_model()`이 고르는 실제 아키텍처(performer/flash
      등) 확인 후 착수)
- [ ] LoRA/scPEFT 비교군: scPEFT 공식 코드를 scGPT/scFoundation에 적용 (이 브랜치
      범위 밖, 별도 작업)
- [x] scGPT `priority_grid()`(~17개, 2026-09-16 추가) 서버 실행 완료 (2026-09-17) —
      batch=1 낮은 lr(3e-5/1e-5) 둘 다 정상 학습 확인(집단 붕괴 해결),
      checkpointing 없이 OOM 경계가 16(성공)~20(OOM) 사이로 좁혀짐, checkpointing
      켜면 28까지 문제없음(거의 선형, batch당 ~0.17GB), grad_accum_steps는
      메모리에 거의 영향 없음(예상대로) 확인. max_seq_len=1500/3001 결과가
      동일하게 나온 원인은 diagnose_seq_lengths.py로 확인(아래).
- [x] `diagnose_seq_lengths.py` 실행 완료 (2026-09-17) — 세포당 발현 gene 수
      min=52/median=224/p95=658/**max=1337**. max_seq_len>=1337은 전부 "자를
      필요 없음"이라 500만 실제로 잘리고 1500/3001은 자연 상한(1337)에서 이미
      정체된 것으로 확인됨(코드 버그 아님, 실험 설계 문제였음).
- [ ] scGPT `gene_length_grid()`(6개, 2026-09-17 추가) 서버 실행 — 위에서 찾은
      자연 상한(1337) 아래를 촘촘히 훑어 진짜 "gene 수 -> 메모리" 곡선 확인.
      `--grid gene_length`로 실행 (`bash benchmark/run_benchmark.sh gene_length`)
- [ ] run.py 완전 통합: scFoundation을 mode: finetune_predict CLI로도 돌릴 수
      있게 base.py의 load_vocab_full 시그니처 확장 (scgpt/geneformer 영향 검토 필요)

## scFoundation GPU 검증 + priority_grid 결과 (2026-09-22~23)

### 첫 GPU 실행에서 발견/수정한 버그 5개 (`src/adapters/scfoundation_adapter.py`)
1. `preprocess()`가 `get_embedding` import 전에 `sys.path`에 repo를 추가 안 함 (순서 버그, `load_model()`에만 있었음)
2. `get_embedding.py` 모듈 최상단 `argparse.parse_args()`가 우리 CLI 인자를 파싱해 죽음 → import 순간만 `sys.argv` 임시 치환
3. 같은 파일의 gene index tsv 상대경로 읽기(`./OS_scRNA_gene_index.19264.tsv`) → import 순간만 cwd를 `repo/model`로 임시 이동
4. `load_data()`에서 reference(gene symbol)/query(Ensembl ID) 유전자 이름 형식이 안 맞아 `concatenate()` 후 유전자 0개로 사라짐 → `gene_name` 컬럼으로 재색인(scgpt_adapter.py와 동일 방식)
5. 분류 헤드 `BatchNorm1d` + `DataLoader`의 `drop_last` 누락으로 마지막 배치가 1개 남으면 학습 크래시 → `drop_last=True`

### smoke grid 결과 (5개 조합 전부 success)
precision/batch/max_seq_len(=n_hvg_genes)에 따라 peak_allocated_mb 1273~10898MB, accuracy 0.58~0.86.
**activation_checkpointing=True는 scfoundation_adapter 미지원이라 효과 없음** (peak_allocated_mb 10898.49 vs 10898.21, notes 컬럼에 명시) — scGPT는 checkpointing이 OOM 경계를 크게 밀어내는 것과 대조적. "모델마다 적절한 설정이 다르다"는 교수님 질문에 대한 가장 직접적인 증거.

### gene_length_grid + priority_grid — gene 수 스케일링 (batch=8 기준, 100~3000)
| n_hvg_genes | peak_allocated_mb |
|---|---|
| 100 | 1056.36 |
| 300~1200 | 1274~1325 (거의 평평 — 모델 가중치 메모리가 floor로 작용) |
| 1300~1900 | 1392~1903 (완만히 가속) |
| 2000 | 1988.04 |
| 3000 | 3550.10 |

1200 이상 9개 지점 quadratic 피팅 R²=0.9996 (linear는 R²=0.965) — **scGPT와 동일하게 quadratic**이지만, 낮은 gene 수에서는 모델 자체 파라미터(1억개, 그중 1400만개 학습)가 만드는 메모리 바닥에 가려 거의 안 보이다가 1200 이후 뚜렷해짐. `n_hvg_genes=100`일 때 accuracy가 0.065(사실상 랜덤, 18-class 기준 5.5%)로 폭락하는데 메모리는 거의 안 줄어(1056 vs 1274MB) — **gene 수 축소가 메모리 절약 대비 생물학적 정보 손실이 훨씬 큰 비효율적 수단**임을 정량적으로 확인.

batch=32에서 같은 구간 반복 결과, gene당 증가율이 batch=8 대비 정확히 ~3.9배 (naive 배치비 32/8=4배와 거의 일치) — **quadratic 곡선의 모양 자체는 batch와 무관**, batch는 전체를 선형 스케일업만 함 (self-attention 메모리의 `batch × L²` 이론과 일치).

### OOM/batch 경계 (gene=3000 고정)
batch 48(15.9GB)/64(20.9GB) 성공, 80/96 OOM — 경계는 64~80 사이, 선형 외삽(311.82MB/batch)이 정확히 들어맞음.
**주의**: batch=80이 OOM난 시점의 peak_allocated(17.79GB)는 오히려 성공한 batch=64(20.87GB)보다 낮음 — PyTorch 캐싱 할당자의 메모리 파편화 때문에 OOM 경계가 `peak_allocated_mb`만으로는 정확히 예측 안 됨. **자동 설정 선택 시 예측 경계값을 그대로 쓰지 말고 안전 마진이 필요**하다는 실증 근거.

### 남은 작업 (업데이트)
- ~~scFoundation 첫 GPU 실행~~ / ~~gene 수 quadratic 재현 여부~~ / ~~OOM 경계~~ — **모두 완료**
- **(미착수, 다음 단계)** scPEFT 공식 저장소(github.com/laolintou/scPEFT)를 scGPT/scFoundation에 적용해 비교군 ②③④ 확보 — scFoundation 지원 범위 미확인 상태로 시작해야 함
