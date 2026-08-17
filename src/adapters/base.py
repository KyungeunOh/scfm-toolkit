"""
adapters/base.py

모든 single-cell foundation model 어댑터가 따라야 하는 공통 인터페이스.
scGPT는 첫 번째 구현체(scgpt_adapter.py)이고, Geneformer 등을 추가할 때는
이 인터페이스를 그대로 구현하는 새 adapter 파일만 추가하면 되도록 설계함.

pipeline/(run.py orchestrator, config validation, h5ad validation, report 생성)은
이 인터페이스에만 의존하고, 모델별 세부사항(scgpt 라이브러리 API 등)은 전혀 모른다.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class ModelAdapter(ABC):
    """single-cell foundation model 하나를 감싸는 어댑터의 공통 인터페이스."""

    #: 이 모델을 돌리기 위해 config.yaml에 반드시 있어야 하는 키
    required_config_keys: List[str] = []
    #: 그중 실제 파일/디렉토리 경로로 존재해야 하는 키
    path_config_keys: List[str] = []
    #: config.yaml의 model: 필드에 들어갈 이름 (예: "scgpt", "geneformer")
    name: str = "base"

    @abstractmethod
    def load_vocab_genes(self, cfg: Dict[str, Any]) -> set:
        """
        h5ad validation(pipeline/data_io.py)에서 gene overlap 비율을 계산하기 위해
        모델 vocabulary의 gene 이름 집합을 반환한다.
        무거운 모델 로딩 전에 vocab만 가볍게 읽을 수 있어야 한다.
        """
        raise NotImplementedError

    @abstractmethod
    def load_data(self, cfg: Dict[str, Any]) -> Tuple[Any, Any, Dict, int]:
        """reference/query h5ad를 로드하고 (adata, adata_test_raw, id2type, num_types) 반환"""
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, adata, cfg: Dict[str, Any]):
        raise NotImplementedError

    @abstractmethod
    def prepare_inputs(self, adata, cfg: Dict[str, Any]):
        """토큰화, DataLoader 구성 등 모델 입력 준비"""
        raise NotImplementedError

    @abstractmethod
    def load_model(self, cfg: Dict[str, Any], num_types: int, device):
        raise NotImplementedError

    @abstractmethod
    def finetune(self, model, prepared_inputs, cfg: Dict[str, Any], device):
        """
        mode: finetune_predict / train_head 에서 사용.
        validation 기준 best epoch의 모델을 반환해야 한다 (마지막 epoch이 아니라).
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, model, adata, prepared_inputs, id2type: Dict, cfg: Dict[str, Any], device):
        """예측 결과가 담긴 adata(obs에 predictions/pred_score 포함)를 반환"""
        raise NotImplementedError

    def embed(self, model, adata, prepared_inputs, cfg: Dict[str, Any], device):
        """
        mode: embed 용 자리. 모든 모델이 embedding 추출을 지원하는 건 아니므로
        기본 구현은 명확한 에러를 내도록 하고, 지원하는 adapter만 override한다.
        """
        raise NotImplementedError(
            f"{self.name} adapter는 아직 embed 모드를 지원하지 않습니다."
        )
