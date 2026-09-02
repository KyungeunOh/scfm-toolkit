"""
adapters/geneformer_adapter.py

Geneformer용 ModelAdapter 구현체. scGPT(scgpt_adapter.py)에 이어 두 번째로 추가하는
모델로, adapters/base.py의 ModelAdapter 인터페이스가 실제로 "다른 모델"에도
일반화되는지 검증하는 목적도 겸한다 (base.py의 load_vocab_full 승격,
save_finetuned_model/load_finetuned_model 추가는 이 작업 과정에서 나온 결과물).

Geneformer는 scGPT와 근본적으로 다른 워크플로우를 쓴다:
  - scGPT: h5ad -> 직접 토큰화 -> torch DataLoader -> 우리가 직접 짠 학습 루프
  - Geneformer: h5ad -> TranscriptomeTokenizer -> 디스크에 HuggingFace Dataset ->
    geneformer.Classifier가 내부적으로 HuggingFace transformers.Trainer로 학습/평가
그래서 이 adapter는 텐서를 직접 다루기보다 "geneformer 라이브러리가 요구하는
디스크 위 파일들을 준비하고, 그 라이브러리를 호출하고, 결과 파일을 다시 읽어서
scfm-toolkit 표준 형식(adata.obs['predictions']/['pred_score'])으로 변환"하는
역할이 크다.

=== 중요: 검증 상태 ===
이 파일은 https://github.com/jkobject/geneformer (Geneformer 공식 배포 미러)의
실제 소스코드(tokenizer.py, classifier.py)를 읽고 API 시그니처/동작을 근거로
작성했다. 하지만 이 개발 환경에는 geneformer/torch가 설치돼 있지 않아 실제로
실행해서 검증하지는 못했다. 특히 아래 부분은 사용자의 실제 환경(geneformer
패키지 버전)에서 직접 확인이 필요하다:
  - TranscriptomeTokenizer 생성자 인자 (model_input_size 등, 버전별로 다를 수 있음)
  - Classifier.train_all_data()가 만드는 체크포인트 디렉터리 이름 규칙
    (아래 glob 패턴이 실제 출력과 다르면 finetune()이 RuntimeError로 명확히 알려줌)
  - evaluate_saved_model()이 저장하는 pred_dict.pkl의 정확한 키 이름
  - Classifier 생성자가 요구하는 training_args 관련 인자 (기본값에 맡겨둠 -
    사용자가 epochs/lr 등을 직접 제어하고 싶으면 이 부분을 확장해야 함)
아래 각 메서드의 docstring에도 관련 caveat을 남겨뒀다.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .base import ModelAdapter

logger = logging.getLogger(__name__)


class _GeneformerModelHandle:
    """
    load_model()이 반환하는 값. scGPT의 TransformerModel과 달리 실제 nn.Module이
    아니라, finetune/save/load/predict 사이에서 필요한 경로 정보만 들고 다니는
    가벼운 핸들이다 - 실제 모델 가중치 로딩/추론은 geneformer.Classifier가
    내부적으로 담당한다.
    """

    def __init__(self, classifier, pretrained_model_dir, id_class_dict_file=None,
                 finetuned_model_dir=None):
        self.classifier = classifier
        self.pretrained_model_dir = str(pretrained_model_dir)
        self.id_class_dict_file = str(id_class_dict_file) if id_class_dict_file else None
        self.finetuned_model_dir = str(finetuned_model_dir) if finetuned_model_dir else None


class GeneformerAdapter(ModelAdapter):
    name = "geneformer"
    required_config_keys = [
        "reference_path", "query_path", "model_dir",
        "celltype_col", "batch_size", "epochs",
    ]
    path_config_keys = ["reference_path", "query_path", "model_dir", "finetuned_model_path"]
    #: finetuned_model_path는 scGPT와 마찬가지로 선택 항목.
    #: Geneformer의 fine-tuned 결과는 단일 파일이 아니라 HuggingFace Trainer
    #: 체크포인트 "디렉터리"라서 파일 확장자가 없다.
    finetuned_model_name = "finetuned_model"

    # ------------------------------------------------------------------
    # vocab (h5ad validation에서 gene overlap 계산용)
    # ------------------------------------------------------------------
    def load_vocab_genes(self, cfg: Dict[str, Any]) -> set:
        """
        pipeline/data_io.py의 overlap 계산은 adata.var의 gene 이름(보통 symbol)과
        직접 비교하는데, Geneformer의 실제 vocabulary는 Ensembl ID 기준이다.
        그래서 여기서는 "Ensembl ID 집합"이 아니라, model_dir에 들어있는
        symbol -> Ensembl ID 매핑 파일(gene_name_id_dict*.pkl)의 symbol(key) 쪽
        집합을 반환해서 overlap 계산이 h5ad의 symbol과 맞게 비교되도록 한다.
        """
        mapping = self._load_symbol_to_ensembl(cfg["model_dir"])
        return set(mapping.keys())

    def _load_symbol_to_ensembl(self, model_dir) -> Dict[str, str]:
        import pickle

        matches = sorted(Path(model_dir).glob("gene_name_id_dict*.pkl"))
        if not matches:
            raise FileNotFoundError(
                f"{model_dir}에서 symbol->Ensembl ID 매핑 파일(gene_name_id_dict*.pkl)을 "
                f"찾을 수 없습니다. Geneformer 사전학습 모델 디렉터리(config.json, "
                f"model.safetensors, gene_name_id_dict*.pkl 등이 있는 곳)가 맞는지 "
                f"확인해주세요."
            )
        with open(matches[0], "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # 데이터 로드
    # ------------------------------------------------------------------
    def load_data(self, cfg: Dict[str, Any]) -> Tuple[Any, Any, Dict, int]:
        import scanpy as sc

        ref_path = cfg["reference_path"]
        query_path = cfg["query_path"]
        celltype_col = cfg["celltype_col"]
        source_col = cfg.get("source_celltype_col", celltype_col)

        logger.info(f"Reference 로드: {ref_path}")
        adata = sc.read(ref_path)
        logger.info(f"Query 로드: {query_path}")
        adata_test = sc.read(query_path)

        adata.obs[celltype_col] = adata.obs[source_col].astype("category")
        adata_test.obs[celltype_col] = adata_test.obs[source_col].astype("category")

        adata.obs["str_batch"] = "0"
        adata_test.obs["str_batch"] = "1"

        adata_test_raw = adata_test.copy()
        adata = adata.concatenate(adata_test, batch_key="str_batch")

        # evaluate_saved_model()의 예측 결과를 원래 셀에 위치가 아니라 id로
        # 안전하게 재정렬하기 위한 고유 id (predict() 참고 - positional zip은
        # 배치 처리 중 일부 행이 드롭되면 셀이 뒤섞이는 조용한 오류가 된다).
        adata.obs["_scfm_cell_id"] = adata.obs_names.astype(str)

        celltype_id_labels = adata.obs[celltype_col].astype("category").cat.codes.values
        num_types = len(np.unique(celltype_id_labels))
        id2type = dict(enumerate(adata.obs[celltype_col].astype("category").cat.categories))

        logger.info(f"  Reference: {(adata.obs['str_batch']=='0').sum()} 세포")
        logger.info(f"  Query:     {(adata.obs['str_batch']=='1').sum()} 세포")
        logger.info(f"  Cell type: {num_types}종 → {list(id2type.values())}")

        return adata, adata_test_raw, id2type, num_types

    def load_vocab_full(self, adata, model_dir: str):
        """scGPT의 동명 메서드와 같은 역할: adata를 모델이 인식하는 유전자
        교집합으로 필터링한다. Ensembl ID로 매핑되지 않는 유전자는 제거."""
        mapping = self._load_symbol_to_ensembl(model_dir)

        gene_name_col = "gene_name" if "gene_name" in adata.var.columns else None
        gene_names = adata.var[gene_name_col] if gene_name_col else adata.var_names

        adata.var["ensembl_id"] = [mapping.get(g) for g in gene_names]
        n_before = adata.n_vars
        adata = adata[:, adata.var["ensembl_id"].notna()].copy()
        logger.info(f"vocab(symbol→Ensembl) 교집합: {n_before} → {adata.n_vars} 유전자")

        # TranscriptomeTokenizer/Classifier는 자기 model_dir(config.json 등)을
        # 직접 읽으므로 scGPT의 args.json 같은 별도 model_configs가 필요 없다.
        return adata, mapping, {}

    # ------------------------------------------------------------------
    # 전처리
    # ------------------------------------------------------------------
    def preprocess(self, adata, cfg: Dict[str, Any]):
        """
        TranscriptomeTokenizer는 adata.obs['n_counts'](세포별 raw 총 count)가
        미리 계산돼 있어야 하고, 자동으로 계산해주지 않는다.

        주의: 이 값은 반드시 raw count(정규화/log1p 이전)여야 한다.
        scfm-toolkit의 scGPT용 예시 config는 data_is_raw: false인 데이터셋을
        가리키는 경우가 있는데, 그 상태로 Geneformer용 config에 그대로 재사용하면
        n_counts가 실제 시퀀싱 depth를 반영하지 못해 잘못된 입력이 된다 -
        Geneformer로 돌릴 때는 reference_path/query_path가 raw count h5ad를
        가리키는지 별도로 확인해야 한다.
        """
        from scipy.sparse import issparse

        X = adata.X
        counts = np.asarray(X.sum(axis=1)).flatten() if issparse(X) else np.asarray(X).sum(axis=1)
        adata.obs["n_counts"] = counts
        return adata

    # ------------------------------------------------------------------
    # 입력 준비 (h5ad -> 토큰화된 HuggingFace Dataset)
    # ------------------------------------------------------------------
    def prepare_inputs(self, adata, cfg: Dict[str, Any], vocab=None):
        """
        scGPT처럼 DataLoader를 직접 만드는 대신, reference/query를 각각 임시 h5ad로
        쪼개 저장한 뒤 TranscriptomeTokenizer로 토큰화된 HuggingFace Dataset을
        디스크에 만들고, Classifier.prepare_data()로 label(celltype) -> id 매핑을
        생성한다. Query는 반드시 reference의 매핑을 재사용해야 label id가 어긋나지
        않는다 (id_class_dict_file을 그대로 넘겨줌).

        base.py 계약상 반환 dict는 len()이 되는 'train_loader'/'valid_loader'를
        포함해야 하는데(run.py 진행 로그용), Geneformer의 Classifier.train_all_data()는
        내부적으로 별도 validation split을 만들지 않는다 - 그래서 두 키 모두 동일한
        reference labeled Dataset을 가리키는 자리채움이다. 실제 학습은
        prepared_train_file 경로 쪽 파일을 참조해서 이뤄진다 (finetune() 참고).
        """
        from datasets import load_from_disk
        from geneformer import Classifier, TranscriptomeTokenizer

        work_dir = Path(cfg["output_dir"]) / "geneformer_work"
        work_dir.mkdir(parents=True, exist_ok=True)

        celltype_col = cfg["celltype_col"]

        ref_mask = adata.obs["str_batch"] == "0"
        query_mask = adata.obs["str_batch"] == "1"

        ref_h5ad_dir = work_dir / "ref_h5ad"
        query_h5ad_dir = work_dir / "query_h5ad"
        ref_h5ad_dir.mkdir(exist_ok=True)
        query_h5ad_dir.mkdir(exist_ok=True)

        adata[ref_mask].copy().write_h5ad(ref_h5ad_dir / "reference.h5ad")
        adata[query_mask].copy().write_h5ad(query_h5ad_dir / "query.h5ad")

        custom_attr_name_dict = {celltype_col: celltype_col, "_scfm_cell_id": "_scfm_cell_id"}

        tokenizer = TranscriptomeTokenizer(
            custom_attr_name_dict=custom_attr_name_dict,
            model_input_size=cfg.get("max_seq_len", 2048),
        )
        logger.info("Reference 토큰화 중 (TranscriptomeTokenizer)...")
        tokenizer.tokenize_data(str(ref_h5ad_dir), str(work_dir), "reference_tokenized", file_format="h5ad")
        logger.info("Query 토큰화 중 (TranscriptomeTokenizer)...")
        tokenizer.tokenize_data(str(query_h5ad_dir), str(work_dir), "query_tokenized", file_format="h5ad")

        classifier = Classifier(
            classifier="cell",
            cell_state_dict={"state_key": celltype_col, "states": "all"},
            forward_batch_size=cfg.get("eval_batch_size", cfg["batch_size"]),
            nproc=cfg.get("num_workers", 1),
        )

        logger.info("Reference prepare_data (label→id 매핑 생성)...")
        classifier.prepare_data(
            input_data_file=str(work_dir / "reference_tokenized.dataset"),
            output_directory=str(work_dir),
            output_prefix="reference",
        )
        id_class_dict_file = work_dir / "reference_id_class_dict.pkl"

        logger.info("Query prepare_data (reference의 label 매핑 재사용)...")
        classifier.prepare_data(
            input_data_file=str(work_dir / "query_tokenized.dataset"),
            output_directory=str(work_dir),
            output_prefix="query",
            id_class_dict_file=str(id_class_dict_file),
        )

        prepared_train_file = work_dir / "reference_labeled.dataset"
        prepared_query_file = work_dir / "query_labeled.dataset"
        train_dataset = load_from_disk(str(prepared_train_file))
        logger.info(f"reference labeled dataset: {len(train_dataset)}개 셀")

        return {
            "train_loader": train_dataset,
            "valid_loader": train_dataset,  # 위 docstring 참고: 별도 valid split 없음
            "classifier": classifier,
            "work_dir": work_dir,
            "id_class_dict_file": id_class_dict_file,
            "prepared_train_file": prepared_train_file,
            "prepared_query_file": prepared_query_file,
        }

    # ------------------------------------------------------------------
    # 모델 로드
    # ------------------------------------------------------------------
    def load_model(self, cfg: Dict[str, Any], num_types: int, device, vocab=None, model_configs=None):
        """
        Geneformer는 scGPT처럼 여기서 즉시 가중치를 텐서로 로드하지 않는다 - 실제
        로딩/학습/평가는 전부 geneformer.Classifier가 담당하므로, 이후 단계에
        필요한 경로 정보만 담은 가벼운 핸들을 반환한다.
        (device는 Classifier/HuggingFace Trainer가 자체적으로 처리해서 여기서는
        쓰지 않는다 - 다른 adapter와 시그니처를 맞추기 위해서만 받는다.)
        """
        return _GeneformerModelHandle(
            classifier=None,
            pretrained_model_dir=cfg["model_dir"],
        )

    # ------------------------------------------------------------------
    # fine-tune
    # ------------------------------------------------------------------
    def finetune(self, model, prepared_inputs, cfg: Dict[str, Any], device):
        """
        Reference 전체로 classification head를 학습한다 (HuggingFace Trainer 경유).

        scGPT adapter와 다른 점(base.py에도 명시): train_all_data()는 자체
        validation split이 없어서 "best validation epoch 가중치"를 고를 수 없다 -
        Trainer가 마지막에 저장한 체크포인트를 그대로 쓴다.

        train_all_data()는 Trainer 객체를 반환할 뿐 체크포인트 경로를 직접
        돌려주지 않아서, 실제 생성된 디렉터리를 glob으로 찾는다. geneformer
        버전에 따라 이름 규칙이 다를 수 있어 못 찾으면 에러 메시지로 명확히 알린다.
        """
        work_dir = prepared_inputs["work_dir"]
        classifier = prepared_inputs["classifier"]

        logger.info("Fine-tuning 시작 (geneformer Classifier.train_all_data, HuggingFace Trainer 경유)...")
        classifier.train_all_data(
            model_directory=model.pretrained_model_dir,
            prepared_input_data_file=str(prepared_inputs["prepared_train_file"]),
            id_class_dict_file=str(prepared_inputs["id_class_dict_file"]),
            output_directory=str(work_dir) + "/",
            output_prefix="reference",
        )

        matches = sorted(work_dir.glob("*_geneformer_cellClassifier_reference"))
        if not matches:
            raise RuntimeError(
                "fine-tuning은 실행됐지만 예상되는 체크포인트 디렉터리"
                f"(work_dir/*_geneformer_cellClassifier_reference)를 찾지 못했습니다. "
                f"geneformer 라이브러리 버전에 따라 출력 디렉터리 이름 규칙이 다를 수 있으니 "
                f"{work_dir} 아래 실제로 생성된 디렉터리 이름을 확인하고 이 glob 패턴을 "
                f"맞춰주세요."
            )
        checkpoint_dir = matches[-1]
        logger.info(f"fine-tuned 체크포인트: {checkpoint_dir}")

        model.classifier = classifier
        model.finetuned_model_dir = str(checkpoint_dir)
        model.id_class_dict_file = str(prepared_inputs["id_class_dict_file"])
        return model

    # ------------------------------------------------------------------
    # fine-tuned 모델 저장/불러오기
    # ------------------------------------------------------------------
    def save_finetuned_model(self, model, path) -> None:
        """
        scGPT는 단일 .pt 파일이지만, Geneformer의 fine-tune 결과는 HuggingFace
        Trainer가 만든 체크포인트 디렉터리(config.json, model.safetensors 등)다.
        id_class_dict.pkl(label<->id 매핑)도 함께 복사해둬야 나중에
        load_finetuned_model()만으로 predict()가 가능하다.
        """
        import shutil

        path = Path(path)
        if path.exists():
            shutil.rmtree(path)
        shutil.copytree(model.finetuned_model_dir, path)
        shutil.copy(model.id_class_dict_file, path / "id_class_dict.pkl")
        logger.info(f"fine-tuned 체크포인트 저장: {path}")

    def load_finetuned_model(self, model, path, device):
        """
        save_finetuned_model()로 저장해둔 디렉터리를 가리키도록 핸들을 갱신한다.
        실제 가중치 로딩은 predict()가 evaluate_saved_model()을 호출할 때
        일어난다 (device는 Classifier가 자체 처리하므로 여기서 쓰지 않음).
        """
        path = Path(path)
        model.finetuned_model_dir = str(path)
        model.id_class_dict_file = str(path / "id_class_dict.pkl")
        return model

    # ------------------------------------------------------------------
    # 예측
    # ------------------------------------------------------------------
    def predict(self, model, adata, prepared_inputs, id2type: Dict, cfg: Dict[str, Any], device):
        """
        id2type(adata의 category 코드 기반 매핑)은 여기서 쓰지 않는다 - Geneformer는
        자기 자신의 label<->id 매핑(id_class_dict.pkl, prepare_data가 생성)을
        따로 갖고 있고, evaluate_saved_model()도 그 매핑 기준으로 결과를 낸다.

        중요한 정합성 이슈: evaluate_saved_model()은 배치 처리 중 마지막 일부
        행을 드롭할 수 있어 반환되는 예측 개수가 원래 셀 수보다 적을 수 있다.
        그래서 위치 기반(zip)으로 이어붙이면 셀이 조용히 뒤섞일 위험이 있어,
        load_data()에서 미리 심어둔 '_scfm_cell_id'로 pandas.Series.reindex을
        이용해 안전하게 재정렬한다.
        """
        import pickle

        import pandas as pd
        from datasets import load_from_disk

        work_dir = prepared_inputs["work_dir"]
        classifier = prepared_inputs["classifier"]

        logger.info("evaluate_saved_model로 query 예측 중...")
        classifier.evaluate_saved_model(
            model_directory=model.finetuned_model_dir,
            id_class_dict_file=model.id_class_dict_file,
            test_data_file=str(prepared_inputs["prepared_query_file"]),
            output_directory=str(work_dir) + "/",
            output_prefix="query",
            predict=True,
        )

        with open(work_dir / "query_pred_dict.pkl", "rb") as f:
            pred_dict = pickle.load(f)
        with open(model.id_class_dict_file, "rb") as f:
            id_class_dict = pickle.load(f)

        # evaluate_saved_model()은 softmax를 적용하지 않은 raw logits을 반환하므로
        # 직접 softmax를 적용해 신뢰도(pred_score)를 계산한다.
        logits = np.array(pred_dict["predictions"])
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        pred_ids = probs.argmax(axis=1)
        pred_scores = probs.max(axis=1)

        query_dataset = load_from_disk(str(prepared_inputs["prepared_query_file"]))
        n = len(pred_ids)
        cell_ids = query_dataset["_scfm_cell_id"][:n]

        pred_labels = [id_class_dict[i] for i in pred_ids]
        pred_series = pd.Series(pred_labels, index=cell_ids)
        score_series = pd.Series(pred_scores, index=cell_ids)

        adata_test = adata[adata.obs["str_batch"] == "1"].copy()
        adata_test.obs["predictions"] = pred_series.reindex(adata_test.obs["_scfm_cell_id"]).values
        adata_test.obs["pred_score"] = score_series.reindex(adata_test.obs["_scfm_cell_id"]).values

        n_missing = int(adata_test.obs["predictions"].isna().sum())
        if n_missing:
            logger.warning(
                f"{n_missing}개 셀에서 예측 결과를 찾지 못했습니다 "
                f"(evaluate_saved_model 처리 중 드롭됐을 가능성) - "
                f"predictions.csv에서 빈 값으로 표시됩니다."
            )

        logger.info("예측 완료")
        return adata_test
