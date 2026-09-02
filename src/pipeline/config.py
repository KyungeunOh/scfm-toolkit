"""
pipeline/config.py
config.yaml 로딩과 validation을 담당.

기존 run.py에 있던 validate_config()를 이 모듈로 옮기고,
model / mode 필드를 추가해서 여러 모델·여러 실행 모드를 지원하도록 확장함.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()

# 이 두 필드는 항상 필요. 나머지 필수 키는 모델(adapter)별로 다를 수 있어서
# adapter 쪽에서 required_config_keys()로 선언하게 한다.
CORE_REQUIRED_KEYS = ["output_dir"]

SUPPORTED_MODES = ["finetune_predict", "embed", "train_head"]
DEFAULT_MODE = "finetune_predict"
DEFAULT_MODEL = "scgpt"


class ConfigError(Exception):
    """config validation 실패 시 발생. 사용자가 읽을 수 있는 메시지를 담는다."""


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ConfigError(f"config 파일을 찾을 수 없습니다: {path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("model", DEFAULT_MODEL)
    cfg.setdefault("mode", DEFAULT_MODE)
    return cfg


def validate_config(
    cfg: Dict[str, Any],
    adapter_required_keys: List[str] = None,
    adapter_path_keys: List[str] = None,
) -> None:
    """
    config를 검증하고, 결과를 rich 테이블로 출력한다.
    실패 시 Python traceback이 아니라 ConfigError(사용자 메시지)를 던진다.

    adapter_required_keys: 이 모델(adapter)이 config에서 필수로 요구하는 키 목록
    adapter_path_keys:     그중에서 실제 파일/디렉토리 경로로 존재해야 하는 키 목록.
                            여기 있는 키라도 config에 값이 없거나 null/빈 문자열이면
                            "선택 항목이라 안 씀"으로 보고 건너뛴다 (필수 여부는
                            adapter_required_keys가 따로 결정).
    """
    adapter_required_keys = adapter_required_keys or []
    adapter_path_keys = adapter_path_keys or []
    checks: List[Tuple[str, str, str]] = []

    def fail(msg: str):
        _print_checks(checks)
        raise ConfigError(msg)

    # 1) mode 검증
    mode = cfg.get("mode", DEFAULT_MODE)
    mode_ok = mode in SUPPORTED_MODES
    checks.append((
        "mode 값",
        "✅" if mode_ok else "❌",
        mode if mode_ok else f"'{mode}'는 지원하지 않음 (지원: {', '.join(SUPPORTED_MODES)})",
    ))
    if not mode_ok:
        fail(
            f"config.yaml의 mode='{mode}'는 아직 지원하지 않습니다. "
            f"현재 지원되는 값: {', '.join(SUPPORTED_MODES)}"
        )

    if mode != "finetune_predict":
        # embed / train_head는 구조상 자리는 마련해두었지만 아직 구현 전.
        # 사용자가 혼란스러운 traceback을 보기 전에 여기서 명확히 안내.
        checks.append(("mode 구현 여부", "⚠️", f"'{mode}'는 아직 구현되지 않았습니다 (로드맵)"))
        _print_checks(checks)
        raise ConfigError(
            f"mode='{mode}'는 config 구조상 예약되어 있으나 아직 구현되지 않았습니다. "
            f"지금은 mode: finetune_predict 만 실행할 수 있습니다."
        )

    # 2) 공통 필수 키
    for key in CORE_REQUIRED_KEYS + adapter_required_keys:
        ok = key in cfg and cfg[key] not in (None, "")
        checks.append((f"필수 키: {key}", "✅" if ok else "❌", "" if ok else "config.yaml에 없음"))
        if not ok:
            fail(
                f"config.yaml에 필수 항목 '{key}'가 없습니다. "
                f"config.yaml에 '{key}: <값>' 을 추가해주세요."
            )

    # 3) 경로로 존재해야 하는 키 (값이 비어있으면 "선택 항목 미사용"으로 보고 건너뜀)
    for key in adapter_path_keys:
        if key not in cfg or cfg[key] in (None, ""):
            continue
        p = Path(cfg[key])
        ok = p.exists()
        checks.append((f"경로 존재: {key}", "✅" if ok else "❌", str(p)))
        if not ok:
            fail(
                f"config.yaml의 '{key}' 경로를 찾을 수 없습니다: {p}\n"
                f"  → 경로 오타이거나, 파일이 아직 준비되지 않았을 수 있습니다."
            )

    _print_checks(checks)


def _print_checks(checks: List[Tuple[str, str, str]]) -> None:
    table = Table(title="config.yaml 검증 결과", show_lines=False)
    table.add_column("항목", style="bold")
    table.add_column("결과", justify="center")
    table.add_column("비고", style="dim")
    for name, status, note in checks:
        table.add_row(name, status, note)
    console.print(table)
