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
  logging.py               CSV 로깅 스키마(RUN_LOG_COLUMNS) + append 함수
  memory_probe.py           GPU peak memory 측정 + OOM을 "실패"가 아니라 데이터
                             포인트로 다루는 컨텍스트 매니저, memory budget 흉내
  grid.py                   성긴 그리드(coarse) -> OOM 경계 근처 정밀 그리드(refine)
  run_scgpt_sweep.py         scGPT sweep 러너 (src/run.py의 Step 3~9 재사용)
  run_scfoundation_sweep.py  scFoundation sweep 러너 (run.py 미사용 - 이유는
                             src/adapters/scfoundation_adapter.py 참고)

src/adapters/scfoundation_adapter.py   새 모델 adapter (아래 "검증 상태" 참고)
src/adapters/scgpt_adapter.py          기존 파일에 precision/activation_checkpointing
                                        옵션 추가 (기본값은 기존 동작과 100% 동일 -
                                        cfg에 새 키를 안 넣으면 기존 4개 GPU-validated
                                        mode 동작이 바뀌지 않음)
```

## 검증 상태 (2026-09-12 기준, 아직 GPU에서 한 번도 안 돌려봄)

이 브랜치 코드 전체는 **아직 실제 GPU/데이터로 실행 검증되지 않았다**. 이 개발
환경에 torch/scGPT/scFoundation 라이브러리와 GPU가 없어서, 지금까지는 (1) 코드가
논리적으로 맞는지 정적으로 검토, (2) `tests/test_benchmark.py`로 GPU/모델
라이브러리 없이도 되는 부분(grid 조합 생성, CSV 로깅 스키마)만 실행 검증했다.
서버에서 처음 실행할 때 확인해야 할 것들을 우선순위 순으로 정리:

1. **scGPT `run_scgpt_sweep.py --grid smoke`** (5개 조합, 가장 먼저 시도할 것) -
   기존에 GPU 검증된 `mode: finetune_predict`과 같은 adapter 메서드를 재사용하므로
   위험도가 가장 낮다. 확인할 것: activation_checkpointing=True일 때
   `enable_activation_checkpointing()`이 실제로 `model.transformer_encoder`를
   찾는지(찾으면 로그에 아무 경고 없음, 못 찾으면 warning 출력됨), bf16이 이
   프로젝트 GPU(gnode01 등)에서 지원되는지(구형 GPU는 bf16 미지원 - RuntimeError로
   바로 드러남).
2. **scFoundation adapter** - 위험도가 가장 높다(`src/adapters/scfoundation_adapter.py`
   모듈 docstring의 "검증 상태" 절 참고). 체크포인트/repo부터 준비:
   - `git clone https://github.com/biomap-research/scFoundation.git`
   - 체크포인트는 GitHub에 없음 - 원본 repo README의 SharePoint 링크에서 별도 다운로드
   - `pip install local_attention` (requirements-scfoundation.txt) - scGPT
     스택과 충돌하는지 먼저 별도로 확인, 괜찮으면 requirements.txt에 합칠지 결정
   - `config/config_scfoundation.example.yaml`을 복사해서 경로 채운 뒤
     `run_scfoundation_sweep.py --grid smoke`의 **첫 조합 하나만** 먼저 시도
3. 전체 `--grid coarse`(scGPT 144개 기본 조합)는 1, 2가 끝난 뒤에.

## 실행 예시

```bash
# scGPT
python benchmark/run_scgpt_sweep.py \
    --config config/config.yaml \
    --grid smoke \
    --out benchmark_results/scgpt_sweep.csv

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

## 남은 작업

- [ ] scGPT: `--grid smoke` 서버 실행 검증 (최우선)
- [ ] scGPT: activation checkpointing이 실제로 peak memory를 줄이는지 확인
- [ ] scFoundation: 체크포인트/repo 준비 + 첫 forward/finetune 성공 확인
- [ ] scFoundation: activation checkpointing 지원 (지금은 encoder 내부 구조
      미확인으로 미지원 - `select_model()`이 고르는 실제 아키텍처(performer/flash
      등) 확인 후 착수)
- [ ] LoRA/scPEFT 비교군: scPEFT 공식 코드를 scGPT/scFoundation에 적용 (이 브랜치
      범위 밖, 별도 작업)
- [ ] scGPT `coarse_grid()` 전체(144개) 실행 + `refine_between()`으로 OOM 경계
      근처 정밀 측정
- [ ] run.py 완전 통합: scFoundation을 mode: finetune_predict CLI로도 돌릴 수
      있게 base.py의 load_vocab_full 시그니처 확장 (scgpt/geneformer 영향 검토 필요)
