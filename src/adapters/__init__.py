"""
adapters/__init__.py

config.yaml의 `model:` 필드 값으로 어댑터를 선택하기 위한 레지스트리.
Geneformer 등 새 모델을 추가할 때는:
  1. adapters/geneformer_adapter.py 에서 ModelAdapter를 구현하고
  2. 아래 _REGISTRY에 한 줄 추가하면 된다.
run.py나 pipeline/ 쪽 코드는 전혀 건드릴 필요 없음.
"""

from typing import Dict, Type

from .base import ModelAdapter

_REGISTRY: Dict[str, str] = {
    "scgpt": "adapters.scgpt_adapter.ScGPTAdapter",
    "geneformer": "adapters.geneformer_adapter.GeneformerAdapter",
    # 2026-09 memory-benchmark 브랜치에서 추가 - 아직 run.py CLI 경로(mode:
    # finetune_predict 등)로는 실행할 수 없다(scfoundation_adapter.py의
    # load_vocab_full() docstring 참고, 원인: 필요한 경로 설정이 3개라 base.py의
    # model_dir 단일 인자 시그니처와 안 맞음). 지금은 benchmark/run_scfoundation_sweep.py
    # 에서만 어댑터 메서드를 직접 호출해서 쓴다. 레지스트리에는 등록해뒀는데, config.yaml에
    # model: scfoundation으로 run.py를 직접 돌리면 Step 4에서 이 상황을 설명하는
    # NotImplementedError가 명확히 뜬다(조용히 잘못된 값이 나오는 것보다 나음).
    "scfoundation": "adapters.scfoundation_adapter.ScFoundationAdapter",
}


def get_adapter(name: str) -> ModelAdapter:
    if name not in _REGISTRY:
        supported = ", ".join(_REGISTRY.keys())
        raise ValueError(
            f"model='{name}'은 지원하지 않습니다. 현재 지원 모델: {supported}"
        )
    module_path, cls_name = _REGISTRY[name].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    adapter_cls: Type[ModelAdapter] = getattr(module, cls_name)
    return adapter_cls()
