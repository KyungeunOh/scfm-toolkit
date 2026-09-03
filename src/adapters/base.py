"""
adapters/base.py

모든 single-cell foundation model 어댑터가 따라야 하는 공통 인터페이스.
scGPT는 첫 번째 구현체(scgpt_adapter.py)이고, Geneformer(geneformer_adapter.py)를
같은 annotation task에 추가하면서 이 인터페이스를 한 번 검증/정리했다:

- load_vocab_full()을 정식 추상 메서드로 승격시켰다. run.py가 모든 adapter에 대해
  무조건 이 메서드를 호출하고 있었는데(Step 4), 이전에는 이 인터페이스에 선언돼
  있지 않아 "사실상 필수인데 문서화 안 된 계약"이었다 — scGPT 하나만 있을 땐
  드러나지 않던 문제.
- save_finetuned_model()/load_finetuned_model()을 추가했다. scGPT는 fine-tune 결과가
  단일 torch state_dict라 run.py가 직접 torch.save/load를 했었는데, Geneformer의
  fine-tune 결과는 HuggingFace Trainer가 만드는 체크포인트 "디렉터리"라 같은 방식이
  안 맞는다. 저장/불러오기 방식 자체를 adapter가 결정하게 해서 run.py는 다시
  완전히 모델 무관하게 유지했다 (finetuned_model_name으로 파일이 될지 디렉터리가
  될지도 adapter가 정한다).
- load_model/prepare_inputs의 시그니처에 실제로 쓰이던 vocab/model_configs
  키워드 인자를 명시했다 (기존엔 문서화가 실제 호출부와 어긋나 있었음).

pipeline/(run.py orchestrator, config validation, h5ad validation, report 생성)은
이 인터페이스에만 의존하고, 모델별 세부사항(scgpt/geneformer 라이브러리 API 등)은
전혀 모른다.
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
    #: save_finetuned_model()/load_finetuned_model()이 output_dir 아래 사용할 이름.
    #: 파일(예: "finetuned_model.pt")일 수도, 디렉터리(예: "finetuned_model")일 수도
    #: 있다 — 어떤 형태로 저장할지는 전적으로 adapter가 결정한다.
    finetuned_model_name: str = "finetuned_model.pt"

    @abstractmethod
    def load_vocab_genes(self, cfg: Dict[str, Any]) -> set:
        """
        h5ad validation(pipeline/data_io.py)에서 gene overlap 비율을 계산하기 위해
        모델 vocabulary의 gene 이름 집합을 반환한다.
        무거운 모델 로딩 전에 vocab만 가볍게 읽을 수 있어야 한다.

        주의: 여기서 "gene 이름"은 h5ad의 adata.var(gene_name 등)와 직접 비교되는
        표현이어야 한다 — 모델 내부 vocabulary의 키가 다른 형식(예: Ensembl ID)이면
        그 형식으로 변환 가능한 별도의 매핑을 이 메서드가 대신 반환하거나
        load_vocab_full()에서 처리해야 한다 (geneformer_adapter.py 참고: 실제
        vocabulary는 Ensembl ID 기준이라 symbol -> Ensembl 매핑의 symbol 쪽을
        반환한다).
        """
        raise NotImplementedError

    @abstractmethod
    def load_data(self, cfg: Dict[str, Any]) -> Tuple[Any, Any, Dict, int]:
        """reference/query h5ad를 로드하고 (adata, adata_test_raw, id2type, num_types) 반환"""
        raise NotImplementedError

    @abstractmethod
    def load_vocab_full(self, adata, model_dir: str) -> Tuple[Any, Any, Dict]:
        """
        run.py Step 4에서 모든 adapter에 대해 호출된다: 모델 vocabulary를 제대로
        로드하고, adata를 그 vocabulary와 매칭되는 유전자로 필터링한 뒤
        (adata, vocab, model_configs)를 반환한다.

        vocab과 model_configs는 run.py 입장에서는 완전히 불투명한 값이다 — 그 뒤
        preprocess/prepare_inputs/load_model에 그대로 전달만 될 뿐, run.py가
        내용을 들여다보지 않는다. 그러니 adapter마다 원하는 대로 아무 타입이나
        (dict, 커스텀 객체 등) 담아서 반환해도 된다.
        """
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, adata, cfg: Dict[str, Any]):
        raise NotImplementedError

    @abstractmethod
    def prepare_inputs(self, adata, cfg: Dict[str, Any], vocab=None):
        """
        토큰화, 학습/검증 입력 준비 등을 담당. 반환하는 dict에는 최소한
        'train_loader'/'valid_loader' 키가 있어야 한다 — run.py가 진행 상황
        로그에 len(prepared['train_loader'])를 찍기 때문이다. 이 값이 실제
        torch DataLoader일 필요는 없고, len()이 되는 것(HuggingFace Dataset 등)이면
        충분하다. 그 외 키는 adapter가 자기 finetune()/predict()에서만 쓰는
        내부 값이라 자유롭게 추가해도 된다.
        """
        raise NotImplementedError

    @abstractmethod
    def load_model(self, cfg: Dict[str, Any], num_types: int, device, vocab=None, model_configs=None):
        """
        여기서 반환하는 "model"도 run.py 입장에서는 불투명한 값이다 — 실제
        torch.nn.Module일 필요는 없다 (geneformer_adapter.py는 체크포인트 경로를
        담은 가벼운 핸들을 반환한다). finetune/predict/save_finetuned_model/
        load_finetuned_model에 그대로 다시 전달되므로, adapter가 자기 자신에게
        필요한 형태로 아무 값이나 반환해도 된다.
        """
        raise NotImplementedError

    @abstractmethod
    def finetune(self, model, prepared_inputs, cfg: Dict[str, Any], device):
        """
        mode: finetune_predict / train_head 에서 사용.
        가능하면 validation 기준 best epoch의 모델을 반환해야 한다 (마지막 epoch이
        아니라) — 다만 이게 불가능한 학습 방식(예: 내부 validation split이 없는
        geneformer의 train_all_data)이라면 그 사실을 docstring에 명시할 것.
        """
        raise NotImplementedError

    @abstractmethod
    def save_finetuned_model(self, model, path) -> None:
        """
        finetune()이 반환한 model을 path에 저장해서, 다음 실행에서
        load_finetuned_model()로 재학습 없이 재사용할 수 있게 한다. path가
        파일이 될지 디렉터리가 될지는 finetuned_model_name과 함께 adapter가
        정한다 (예: scGPT는 단일 .pt state_dict 파일, geneformer는 HuggingFace
        체크포인트 디렉터리).
        """
        raise NotImplementedError

    @abstractmethod
    def load_finetuned_model(self, model, path, device):
        """
        save_finetuned_model()로 저장해둔 걸 다시 불러와 predict()에 바로 넘길 수
        있는 형태로 반환한다. 인자로 받는 model은 load_model()이 만들어준 값
        (사전학습 가중치 로드까지 끝난 상태)이고, 반환값은 predict()가 받는
        model 인자와 같은 형태여야 한다 — 굳이 nn.Module일 필요는 없다.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, model, adata, prepared_inputs, id2type: Dict, cfg: Dict[str, Any], device):
        """예측 결과가 담긴 adata(obs에 predictions/pred_score 포함)를 반환"""
        raise NotImplementedError

    def embed(self, adata, cfg: Dict[str, Any], device):
        """
        mode: embed(현재는 reference mapping에 사용 - Tutorial_Reference_Mapping.ipynb
        참고)에서 호출된다.

        finetune_predict 경로(load_vocab_full → preprocess → prepare_inputs →
        load_model 순서로 run.py가 단계별로 오케스트레이션)와 시그니처가 다른 이유:
        실제로 scGPT의 embed_data() 같은 고수준 유틸리티는 vocab 매칭/토큰화/모델
        로딩을 전부 자체적으로 처리해서, annotation 경로처럼 여러 단계로 쪼갤 필요가
        없었다 - 처음엔 finetune_predict와 같은 시그니처(model, prepared_inputs를
        미리 받는 형태)로 자리만 마련해뒀었는데, 실제 reference mapping을 구현하면서
        불필요한 걸 확인하고 이렇게 단순화했다.

        run.py는 원본 adata(reference 또는 query 하나)를 그대로 넘긴다 - 이 메서드가
        내부적으로 필요한 모델 로딩/전처리를 전부 알아서 한다. 반환값은
        (n_cells, embed_dim) 모양의 numpy 배열(세포별 임베딩)이어야 한다. 그 이후의
        reference-query 매핑(k-NN 다수결)은 pipeline/reference_mapping.py가 모델
        무관하게 처리하므로, 이 메서드는 "임베딩을 만드는 것"까지만 책임지면 된다.

        기본 구현은 명확한 에러를 낸다 - 모든 모델이 embedding 추출을 지원하는 건
        아니므로, 지원하는 adapter만 override한다.
        """
        raise NotImplementedError(
            f"{self.name} adapter는 아직 embed 모드를 지원하지 않습니다."
        )
