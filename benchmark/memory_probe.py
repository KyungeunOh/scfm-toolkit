"""
benchmark/memory_probe.py
GPU peak memory 측정 + OOM을 "실험 실패"가 아니라 "하나의 데이터 포인트"로
다루기 위한 헬퍼. 지금 src/pipeline/report.py의 save_environment_report()는
GPU 이름만 기록하고 peak memory는 아예 측정하지 않는다 - 이 프로젝트에 없던
기능이라 여기서 새로 만든다(2026-09-12 기준, README.md/코드 어디에도
torch.cuda.max_memory_allocated 호출이 없음을 grep으로 확인).

미검증 표시: 아래 로직은 PyTorch 공식 문서(torch.cuda.reset_peak_memory_stats,
max_memory_allocated/reserved, set_per_process_memory_fraction, OutOfMemoryError)
동작 방식을 기준으로 작성했고, 이 프로젝트가 실제로 쓰는 PyTorch 버전(requirements.txt
기준 torch 본체는 베이스 Docker 이미지가 제공 - 정확한 버전 미확인)에서 API가
동일한지는 서버에서 첫 실행 시 확인이 필요하다. 특히 OutOfMemoryError는
PyTorch 1.13+에서 RuntimeError의 서브클래스로 도입되었으므로, 혹시 더 오래된
torch라면 문자열 "CUDA out of memory" 매칭으로 폴백하도록 이미 처리해뒀다.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemoryProbeResult:
    status: str = "success"          # success / oom / error
    error_message: str = ""
    peak_allocated_mb: Optional[float] = None
    peak_reserved_mb: Optional[float] = None
    seconds: float = 0.0


def _is_oom_error(exc: BaseException) -> bool:
    """torch.cuda.OutOfMemoryError(신버전)와 "CUDA out of memory" 문자열이 담긴
    RuntimeError(구버전) 둘 다 OOM으로 인식한다."""
    try:
        import torch
        if hasattr(torch.cuda, "OutOfMemoryError") and isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return "out of memory" in str(exc).lower()


@contextmanager
def measure(device=None):
    """
    사용 예:
        result = MemoryProbeResult()
        with measure(device) as result:
            model = adapter.finetune(model, prepared, cfg, device)
        # result.status가 "oom"이면 model은 무의미한 상태 - 호출부는 이 config를
        # "실패"로 CSV에 기록하고 다음 config로 넘어가야 한다(전체 sweep을 죽이지 않음).

    with 블록 안에서 발생한 예외 중 OOM만 흡수하고 result.status="oom"으로 남긴다.
    OOM이 아닌 다른 예외는 그대로 다시 던진다(코드 버그를 "OOM"으로 오분류하지
    않기 위함) - 호출부(run_*_sweep.py)가 그 예외를 잡아 status="error"로 기록한다.
    """
    import torch

    result = MemoryProbeResult()
    if device is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    try:
        yield result
    except Exception as e:  # noqa: BLE001 - OOM인지 판단 후 재던지기 위해 넓게 잡음
        result.seconds = time.time() - start
        if _is_oom_error(e):
            result.status = "oom"
            result.error_message = str(e)[:500]
            # OOM 이후 CUDA 컨텍스트가 다음 config 실행에 영향을 주지 않도록 캐시 정리.
            # (torch 공식 권장 패턴: OOM catch 후 torch.cuda.empty_cache())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            raise
    else:
        result.seconds = time.time() - start
    finally:
        if device is not None and torch.cuda.is_available():
            result.peak_allocated_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
            result.peak_reserved_mb = round(torch.cuda.max_memory_reserved(device) / (1024 ** 2), 2)


def set_memory_budget_gb(budget_gb: Optional[float], device=None) -> bool:
    """
    "12GB memory budget"처럼 더 큰 실제 GPU(A6000 등)에서 메모리 상한을 흉내낸다.
    torch.cuda.set_per_process_memory_fraction(fraction, device)을 사용 - 이 프로세스가
    쓸 수 있는 GPU 메모리를 실제 전체 용량의 일부로 제한한다.

    반환값 True/False는 실제로 제한을 걸었는지 여부(budget_gb가 None이면 아무것도
    안 하고 False 반환 - "예산 제한 없음"과 "제한을 걸려 했지만 실패"를 구분하기 위함).

    주의(교수님이 명시적으로 지적한 부분): 이렇게 흉내낸 조건은 benchmark/run_log.py의
    memory_budget_gb + memory_budget_is_simulated=True로 반드시 같이 기록해서,
    CSV만 보고 "12GB 짜리 실제 GPU에서 테스트했다"고 오인하지 않게 한다. 실제 GPU
    이름(gpu_name)과 실제 총 메모리(gpu_total_memory_gb)는 항상 별도 컬럼에 그대로 남는다.

    미검증: set_per_process_memory_fraction은 "이미 할당된 메모리"에는 영향을 주지
    않고 앞으로의 할당 상한만 건다 - 따라서 각 sweep config를 별도 프로세스(또는
    최소한 모델을 완전히 새로 만들고 캐시를 비운 상태)에서 실행하는 게 안전하다.
    같은 프로세스 안에서 여러 config를 연달아 돌릴 때 이전 config의 메모리가 완전히
    해제됐는지는 서버에서 첫 실행 시 확인이 필요하다.
    """
    if budget_gb is None:
        return False
    import torch
    if not torch.cuda.is_available():
        return False
    device = device if device is not None else torch.cuda.current_device()
    total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    if budget_gb > total_gb:
        raise ValueError(
            f"memory_budget_gb={budget_gb}가 실제 GPU 전체 메모리({total_gb:.1f}GB)보다 큽니다."
        )
    fraction = budget_gb / total_gb
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    return True
