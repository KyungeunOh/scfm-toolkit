# scFM Toolkit

여러 single-cell foundation model(scFM)을 비전문가도 h5ad 파일과 `config.yaml`만으로
쉽게 실행할 수 있게 하는 재현 가능한 toolkit. 첫 번째 지원 모델은 scGPT (annotation task).

## 구조

```
config/config.yaml       사용자가 수정하는 유일한 파일 (model/mode/경로/하이퍼파라미터)
model/                   vocab.json, args.json, best_model.pt (사전학습 가중치)
src/
  run.py                 얇은 오케스트레이터. "무엇을 언제 실행할지"만 알고
                         "어떻게 실행할지"는 모른다 (scgpt import 없음).
  pipeline/               모델에 무관한 공통 로직
    config.py             config.yaml validation (필수 키/경로/mode 체크, 친절한 에러 메시지)
    data_io.py             h5ad validation (cell/gene 수, label column, vocab 매칭률)
    report.py               표준 output 저장 (predictions, metrics, 예측 신뢰도 경고,
                            confusion matrix, resolved_config, environment report)
  adapters/               모델별 구현 (scGPT 세부사항은 전부 여기에만 있음)
    base.py                모든 모델이 구현해야 하는 공통 인터페이스 (ModelAdapter)
    scgpt_adapter.py        scGPT 구현체 (Tutorial_Annotation.ipynb 이식)
    __init__.py             config의 model: 값으로 adapter를 선택하는 레지스트리
tests/
  make_synthetic_data.py  torch/scGPT 가중치 없이 pipeline을 검증하기 위한 합성 데이터
  test_pipeline.py         config/h5ad validation + 모델 재사용/신뢰도 로직 단위 테스트
  make_demo_outputs.py     표준 output 구조 데모 생성
```

## 실행

```bash
bash run.sh              # 로컬 Docker
sbatch run_slurm.sh       # HPC(SLURM)
```

`config/config.yaml`만 수정하면 되고, 나머지는 자동으로 검증 후 실행된다.

## Fine-tuned 모델 재사용

실행이 끝나면 fine-tune된 가중치가 `output_dir/finetuned_model.pt`로 자동 저장된다.
같은 reference로 다시 predict만 하고 싶다면(재학습 없이) `config.yaml`의
`finetuned_model_path`에 그 파일 경로를 지정하면 된다 — Step 8(fine-tuning)을 건너뛰고
바로 Step 9(예측)로 넘어간다. 비워두면(기본값) 매번 새로 fine-tune한다.

## 예측 신뢰도 경고

예측 확률(softmax 최대값, `pred_score`)이 `config.yaml`의 `low_confidence_threshold`
(기본 0.5)보다 낮은 셀은 `predictions.csv`/`predictions.h5ad`에 `low_confidence` 컬럼으로
표시되고, `metrics.json`에도 비율이 함께 저장된다. label(정답 celltype)이 없는 순수 예측
실행에서도 동작한다. 끄려면 `low_confidence_threshold: null`로 설정한다.

## 표준 output 구조

실행이 끝나면 `output_dir`에 아래가 자동으로 생성된다.

| 파일 | 내용 |
|---|---|
| `predictions.csv` / `predictions.h5ad` | 셀 단위 예측 결과 (low_confidence 플래그 포함) |
| `metrics.json` | 전체 accuracy 요약(label 있을 때) + 예측 신뢰도 요약(항상) |
| `per_class_metrics.csv` | label이 있을 경우 클래스별 precision/recall/f1 |
| `confusion_matrix.png` | label이 있을 경우 confusion matrix |
| `finetuned_model.pt` | fine-tune된 모델 가중치 (재사용 가능, 위 "Fine-tuned 모델 재사용" 참고) |
| `resolved_config.yaml` | 기본값까지 채워서 실제로 사용된 config 전체 (재현성) |
| `environment.json` | 라이브러리 버전, GPU, git commit (재현성) |

## 새 모델 추가하는 법 (예: Geneformer)

1. `src/adapters/geneformer_adapter.py`에서 `ModelAdapter`(`base.py`)를 구현
2. `src/adapters/__init__.py`의 레지스트리에 한 줄 추가
3. `config.yaml`에서 `model: geneformer`로 지정

`run.py`와 `pipeline/` 코드는 전혀 수정할 필요 없음.

## 로드맵

- [x] config.yaml validation (필수 키/경로, 친절한 에러 메시지)
- [x] h5ad validation (label column, vocab 매칭률 기준 미달 시 중단)
- [x] 표준 output 구조 (predictions, metrics, confusion matrix, resolved_config, environment)
- [x] pipeline(공통) / adapters(모델별) 구조 분리
- [x] config에 `mode` 필드 자리 마련 (`finetune_predict` 구현 완료)
- [x] fine-tuned 모델 저장/재사용 (`finetuned_model_path`)
- [x] 예측 신뢰도 경고 (`low_confidence_threshold`)
- [x] Docker 이미지에서 미사용 flash-attn 빌드 단계 제거 (빌드 시간 단축)
- [x] CI로 `tests/test_pipeline.py` 자동 실행
- [ ] `mode: embed`, `mode: train_head` 구현
- [ ] 두 번째 모델(Geneformer) adapter 추가 (같은 annotation task로, ModelAdapter 인터페이스 검증 목적)
- [ ] 새 task(GRN, Integration, Multiomics, Perturbation, Reference Mapping 등) 지원은 위 모델 축 검증 이후 — ModelAdapter/run.py를 task에 무관하게 다시 일반화해야 함
