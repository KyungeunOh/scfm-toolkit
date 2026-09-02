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
