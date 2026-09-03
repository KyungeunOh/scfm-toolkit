# scFM Toolkit

여러 single-cell foundation model(scFM)을 비전문가도 h5ad 파일과 `config.yaml`만으로
쉽게 실행할 수 있게 하는 재현 가능한 toolkit. annotation task 기준으로 scGPT, Geneformer
두 모델을 지원하고(`config.yaml`의 `model:` 값만 바꾸면 됨), scGPT는 fine-tuning 없는
zero-shot task도 두 가지 지원한다: reference mapping(`mode: embed`), batch 통합 품질
평가(`mode: integration`).

## 구조

```
config/config.yaml       사용자가 수정하는 유일한 파일 (model/mode/경로/하이퍼파라미터)
model/                   vocab.json, args.json, best_model.pt (사전학습 가중치)
src/
  run.py                 얇은 오케스트레이터. "무엇을 언제 실행할지"만 알고
                         "어떻게 실행할지"는 모른다 (scgpt import 없음).
  pipeline/               모델에 무관한 공통 로직
    config.py             config.yaml validation (필수 키/경로/mode 체크, 친절한 에러 메시지)
    data_io.py             h5ad validation + 전체 로드 (cell/gene 수, label column, vocab 매칭률)
    reference_mapping.py    mode: embed에서 쓰는 k-NN 기반 label 전파 (모델 무관 - 임베딩만
                            주어지면 어떤 adapter의 embed() 결과에도 재사용 가능)
    integration.py           mode: integration에서 쓰는 batch 통합 품질 평가 (UMAP 저장 +
                            scib 기반 정량 지표 - 마찬가지로 모델 무관)
    report.py               표준 output 저장 (predictions, metrics, 예측 신뢰도 경고,
                            confusion matrix, resolved_config, environment report)
  adapters/               모델별 구현 (scGPT/Geneformer 세부사항은 전부 여기에만 있음)
    base.py                모든 모델이 구현해야 하는 공통 인터페이스 (ModelAdapter)
    scgpt_adapter.py        scGPT 구현체 (Tutorial_Annotation.ipynb 이식)
    geneformer_adapter.py   Geneformer 구현체 (아래 "Geneformer 지원" 참고 - 검증 상태 포함)
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

실행이 끝나면 fine-tune 결과가 `output_dir/<adapter.finetuned_model_name>`로 자동
저장된다. scGPT는 단일 파일(`finetuned_model.pt`), Geneformer는 HuggingFace Trainer
체크포인트 디렉터리(`finetuned_model/`)로 저장 형태가 다른데, `run.py`는 이 차이를
전혀 몰라도 되게 `adapter.save_finetuned_model()`/`load_finetuned_model()`에 위임한다.
같은 reference로 다시 predict만 하고 싶다면(재학습 없이) `config.yaml`의
`finetuned_model_path`에 그 경로(파일 또는 디렉터리)를 지정하면 된다 — Step
8(fine-tuning)을 건너뛰고 바로 Step 9(예측)로 넘어간다. 비워두면(기본값) 매번 새로
fine-tune한다.

## Reference Mapping (mode: embed)

fine-tuning 없이, 사전학습 모델로 reference/query를 각각 임베딩한 뒤 임베딩 공간에서
k-NN 다수결로 reference의 `celltype_col` label을 query에 전파한다
(scGPT `Tutorial_Reference_Mapping.ipynb`와 동일한 방식 - zero-shot이라 annotation의
fine-tuning 경로보다 훨씬 빠르다). `config.yaml`에서 `mode: embed`로 지정하면 되고,
`knn_k`(기본 10)/`gene_col`(기본 `gene_name`) 두 값만 이 mode 전용이다. 결과는
finetune_predict와 동일한 표준 output 구조(`predictions.csv`, `metrics.json` 등)로
저장되므로 아래 표를 그대로 참고하면 된다.

**현재 알려진 제약**: `pipeline/data_io.py`의 h5ad 검증 단계가 reference/query 둘 다
`celltype_col`을 요구한다 (finetune_predict 경로와 동일한 검증 함수를 재사용해서).
그래서 지금은 query에도 정답 label이 있어야 accuracy까지 함께 계산되는 형태로 동작하고,
"라벨이 전혀 없는 새 데이터를 매핑"하는 순수한 용도로 쓰려면 이 검증 로직을 mode별로
분리하는 작업이 추가로 필요하다 (로드맵 참고). scGPT만 `embed()`를 구현했고
(`src/adapters/scgpt_adapter.py`, 내부적으로 scGPT의 `scg.tasks.embed_data()`를 그대로
사용), Geneformer는 아직 `mode: embed`를 지원하지 않는다(명확한 에러로 안내됨).

## Zero-shot Batch Integration (mode: integration)

fine-tuning 없이, 사전학습 모델 임베딩만으로 여러 batch(샘플)의 데이터를 통합했을 때
batch 효과는 얼마나 제거되고 cell type은 잘 구분되는지 평가한다 (scGPT
`tutorials/zero-shot/Tutorial_ZeroShot_Integration.ipynb`와 동일한 방식 - GitHub 원본
코드를 직접 대조해서 구현). `config.yaml`에서 `mode: integration`으로 지정하고,
`data_path`(여러 batch가 섞인 h5ad 파일 하나), `batch_key`(배치/샘플 구분 obs 컬럼),
`n_hvg`(기본 3000)를 설정한다. reference/query 구분이 없다는 점이 다른 두 mode와
다르다.

내부적으로 highly variable gene을 선택한 뒤(`scanpy` `flavor=seurat_v3` - raw count
데이터 전제) scGPT로 zero-shot 임베딩을 뽑고, **비교 기준선으로 fine-tuning도 scGPT도
쓰지 않는 HVG+PCA 방식도 똑같이 계산**해서 같이 저장한다 - scGPT 임베딩이 단순 PCA보다
실제로 나은지 바로 비교하기 위함이다. 결과로 다음을 `output_dir`에 저장한다:

| 파일 | 내용 |
|---|---|
| `integration_umap_scgpt.png` | scGPT zero-shot 임베딩 UMAP (cell type/batch 두 패널) |
| `integration_umap_hvg_pca.png` | HVG+PCA 베이스라인 UMAP (같은 두 패널) |
| `integration_metrics.json` | `scgpt_zero_shot`/`hvg_pca_baseline` 각각의 scib 지표 (NMI/ARI/ASW/graph_conn/PCR, avg_bio, avg_batch) |
| `resolved_config.yaml`, `environment.json` | 다른 mode와 동일 |

**현재 알려진 제약**:
- `batch_key`는 데이터마다 실제 obs 컬럼 이름이 다르므로 `config.yaml`의 예시값을
  그대로 쓰면 안 되고, 실행 전에 실제 데이터를 확인해야 한다 (`config.yaml`의 해당
  절 주석에 확인 명령 포함).
- `data_path`의 `adata.X`가 정규화/log1p된 상태면 `seurat_v3` HVG 선택 결과가
  이상해질 수 있다 (raw count 전제).
- `mode: integration`을 위해 `config.yaml`에 필요한 키(`data_path`, `batch_key`)를
  추가로 검증하지만, 다른 mode용 키(`reference_path`, `n_bins` 등)도 여전히 형식상
  필수로 요구된다 (이 프로젝트는 `config.yaml` 하나를 여러 mode가 공유하는 사용
  패턴이라 실제로는 문제되지 않음 - 아래 로드맵 참고).
- `scib==1.1.7`이 `requirements.txt`에 이미 있고 이 값으로 빌드된 Docker 이미지가
  annotation/embed 두 mode 모두 GPU에서 이미 실행 검증됐으므로 별도 설치는 필요
  없을 것으로 강하게 추정되지만, `mode: integration` 자체(HVG 선택 → embed →
  scib 지표 계산 전체 흐름)는 아직 GPU에서 실제로 돌려보지 못했다 (아래 로드맵 참고).

## 예측 신뢰도 경고

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
| `finetuned_model.pt` (scGPT) / `finetuned_model/` (Geneformer) | fine-tune된 모델 (재사용 가능, 위 "Fine-tuned 모델 재사용" 참고) |
| `resolved_config.yaml` | 기본값까지 채워서 실제로 사용된 config 전체 (재현성) |
| `environment.json` | 라이브러리 버전, GPU, git commit (재현성) |

## Geneformer 지원 (검증 필요)

`config.yaml`에서 `model: geneformer`로 지정하면 동작하도록 구현했지만, 이 툴킷을 개발한
환경에는 `geneformer`/`torch` 패키지가 없어 **실제 라이브러리를 대상으로 실행 검증은 아직
못했다** (공식 소스코드를 읽고 시그니처/동작을 근거로 작성 —
`src/adapters/geneformer_adapter.py` 상단 docstring에 어떤 부분이 특히 불확실한지
구체적으로 적어뒀다). GPU가 있는 실제 환경(HPC/Docker)에서 아래 순서로 처음 검증할 것을
권장한다.

### 1) geneformer 패키지 설치

`geneformer`는 PyPI 패키지가 아니라서 `pip install -r requirements.txt`만으로는 설치되지
않는다. HuggingFace 공식 저장소를 git-lfs로 클론해서 직접 설치해야 한다 (이 클론
디렉터리 자체가 사전학습 체크포인트도 포함하고 있어 아래 `model_dir`로 그대로 쓸 수 있다):

```bash
git lfs install
git clone https://huggingface.co/ctheodoris/Geneformer
cd Geneformer && pip install .
```

그다음 `pip install -r requirements.txt`로 나머지 부가 의존성(transformers, peft 등,
requirements.txt 하단 "Geneformer 어댑터 전용" 구간)을 설치한다 — scGPT 스택과 같은
환경에 함께 설치 시 버전 충돌 여부는 미검증이니, 가능하면 scGPT와 별도 가상환경/이미지로
먼저 시도해보는 걸 권장한다.

### 2) 데이터/config 준비

- `reference_path`/`query_path` h5ad가 **raw count**인지 확인 (Geneformer는 세포당 raw
  총 count가 필요 — scGPT용으로 준비된 정규화된 데이터셋을 그대로 재사용하면 안 됨)
- `model_dir`을 위 1)에서 클론한 `Geneformer` 디렉터리로 지정
  (`config.json`, `model.safetensors`, `gene_name_id_dict*.pkl` 등이 그 안에 있어야 함)
- `config.yaml`에서 `model: geneformer`로 바꾸고, `n_bins`/`data_is_raw` 등 scGPT 전용
  파라미터는 무시됨(geneformer_adapter.py가 참조하지 않음)

### 3) 실행 후 확인 포인트

- `src/adapters/geneformer_adapter.py`의 `finetune()`에 있는 체크포인트 디렉터리 이름
  glob 패턴(`*_geneformer_cellClassifier_reference`)이 실제 설치된 geneformer 버전의
  출력과 맞는지 — 안 맞으면 명확한 `RuntimeError` 메시지로 알려주도록 만들어뒀으니, 에러
  메시지에 `work_dir`(=`output_dir/geneformer_work`) 아래 실제 생성된 디렉터리 이름을
  붙여서 알려주면 그 부분만 빠르게 고칠 수 있다
- 에러가 나면 스택 트레이스와 함께 `work_dir` 아래 실제로 뭐가 생성됐는지
  (`ls output_dir/geneformer_work`) 같이 확인하면 원인 파악이 훨씬 빠르다

## 새 모델을 추가하는 법 (일반)

1. `src/adapters/<model>_adapter.py`에서 `ModelAdapter`(`base.py`)를 구현
   (scGPT/Geneformer 둘 다 참고할 수 있는 예시 — 특히 fine-tune 결과가 단일 파일이 아니라
   디렉터리/여러 파일이라면 `finetuned_model_name`과 `save_finetuned_model`/
   `load_finetuned_model`을 그에 맞게 정의)
2. `src/adapters/__init__.py`의 레지스트리에 한 줄 추가
3. `config.yaml`에서 `model: <model>`로 지정

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
- [x] 두 번째 모델(Geneformer) adapter 추가 (같은 annotation task로, ModelAdapter 인터페이스 검증 목적 —
      단, 위 "Geneformer 지원 (검증 필요)" 참고: 실제 geneformer 라이브러리로 실행 검증은 아직 안 됨)
- [ ] Geneformer adapter를 실제 환경(geneformer 설치, raw count h5ad, 사전학습 체크포인트)에서
      end-to-end 실행 검증
- [x] `mode: embed` 구현 (scGPT reference mapping, `Tutorial_Reference_Mapping.ipynb` 기반) —
      위 "Reference Mapping (mode: embed)" 절의 알려진 제약(query도 label 필요) 참고
- [x] `mode: embed`를 GPU 환경에서 실제 h5ad로 end-to-end 실행 검증 — gnode01(RTX 3090)에서
      실행 완료, accuracy 73.89%(9951/13468). fine-tuned annotation(87.6%)보다 낮은 건
      zero-shot의 정상적인 트레이드오프 (같은 데이터로 실측 비교함, 버그 아님 — 공식
      튜토리얼 코드와 한 줄씩 대조해서 구현이 원본과 일치함도 확인)
- [x] 세 번째 task 추가: Zero-shot Batch Integration (`mode: integration`, scGPT
      `Tutorial_ZeroShot_Integration.ipynb` 기반, scib 지표 + HVG/PCA 베이스라인 비교) —
      코드 완료, GPU 실행 검증 전 (위 "Zero-shot Batch Integration" 절 참고)
- [ ] `mode: integration`을 GPU 환경에서 실제 h5ad로 end-to-end 실행 검증 (scib/scanpy가
      없는 개발 환경이라 py_compile + ABC 스텁 검증까지만 함 - embed_data() 관련 로직과
      마찬가지로 실제 실행은 아직 못 해봄)
- [x] (부분) mode별로 다른 config 필수 키 검증 — `ModelAdapter.extra_required_config_keys(mode)`/
      `extra_path_config_keys(mode)` 훅을 추가해서 `mode: integration`의 `data_path`/`batch_key`는
      제대로 필수 항목으로 검증됨. 다만 기존 `required_config_keys`(mode 무관 공통)는 그대로라,
      `embed`/`integration`에서도 여전히 `n_bins`/`max_seq_len`/`epochs` 등 무관한 키가 형식상
      필수로 남아있음 (완전한 mode별 분리는 후속 작업)
- [ ] query에 label이 없어도 되는 순수 매핑 경로 지원 (위 "현재 알려진 제약" 참고)
- [ ] `mode: train_head` 구현
- [ ] 새 task(GRN, Integration, Multiomics, Perturbation 등) 지원은 위 모델 축 검증 이후 — ModelAdapter/run.py를 task에 무관하게 다시 일반화해야 함
