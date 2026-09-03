"""
pipeline/data_io.py
h5ad 입력 파일에 대한 validation.

기존 코드는 vocab 매칭 후 남은 유전자 수를 '로그'로만 찍었는데,
여기서는 이걸 사용자가 실행 전에 미리 확인할 수 있는 '체크'로 승격시킨다.
- obs에 label column이 실제 있는지
- gene name이 모델 vocabulary와 얼마나 겹치는지 (매칭률이 너무 낮으면 경고/중단)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()

# 매칭률이 이 값보다 낮으면 실행을 막는다 (silent하게 정확도만 낮게 나오는 것 방지)
MIN_VOCAB_OVERLAP_RATIO = 0.3
# 이 구간에서는 실행은 하되 경고를 보여준다
WARN_VOCAB_OVERLAP_RATIO = 0.6


class DataValidationError(Exception):
    """h5ad validation 실패 시 발생. 사용자가 읽을 수 있는 메시지를 담는다."""


def validate_h5ad(
    path: str,
    label: str,
    celltype_col: Optional[str] = None,
    source_celltype_col: Optional[str] = None,
    vocab_genes: Optional[set] = None,
    gene_name_col: str = "gene_name",
) -> Dict:
    """
    h5ad 파일 하나를 검증하고 요약 정보를 dict로 반환한다.
    label: 로그/테이블에 표시할 이름 (예: "Reference", "Query")
    """
    import anndata as ad  # 무거운 import는 함수 안에서 (CLI 시작 속도용)

    p = Path(path)
    if not p.exists():
        raise DataValidationError(f"[{label}] 파일을 찾을 수 없습니다: {path}")

    try:
        adata = ad.read_h5ad(p, backed="r")
    except Exception as e:
        raise DataValidationError(f"[{label}] h5ad 파일을 열 수 없습니다 ({path}): {e}")

    checks = []
    n_cells, n_genes = adata.shape
    checks.append((f"{label}: cell 수", "✅" if n_cells > 0 else "❌", str(n_cells)))
    checks.append((f"{label}: gene 수", "✅" if n_genes > 0 else "❌", str(n_genes)))
    if n_cells == 0 or n_genes == 0:
        _print_checks(checks)
        raise DataValidationError(f"[{label}] {path}에 cell 또는 gene이 없습니다.")

    # label column 존재 확인
    label_col = source_celltype_col or celltype_col
    if label_col is not None:
        ok = label_col in adata.obs.columns
        checks.append((
            f"{label}: label column",
            "✅" if ok else "❌",
            label_col if ok else f"'{label_col}'가 obs에 없음 (있는 컬럼: {list(adata.obs.columns)[:8]})",
        ))
        if not ok:
            _print_checks(checks)
            raise DataValidationError(
                f"[{label}] obs에 '{label_col}' 컬럼이 없습니다. "
                f"config.yaml의 celltype_col / source_celltype_col 값을 확인해주세요."
            )

    overlap_ratio = None
    if vocab_genes is not None:
        gene_names = _get_gene_names(adata, gene_name_col)
        matched = sum(1 for g in gene_names if g in vocab_genes)
        overlap_ratio = matched / max(len(gene_names), 1)
        pct = f"{overlap_ratio:.1%} ({matched}/{len(gene_names)})"

        if overlap_ratio < MIN_VOCAB_OVERLAP_RATIO:
            checks.append((f"{label}: vocab 매칭률", "❌", pct + " — 실행 중단 기준 미달"))
            _print_checks(checks)
            raise DataValidationError(
                f"[{label}] 모델 vocabulary와 겹치는 유전자가 {pct}뿐입니다 "
                f"(최소 기준: {MIN_VOCAB_OVERLAP_RATIO:.0%}). "
                f"gene name 형식(symbol vs Ensembl ID 등)이 맞는지 확인해주세요."
            )
        elif overlap_ratio < WARN_VOCAB_OVERLAP_RATIO:
            checks.append((f"{label}: vocab 매칭률", "⚠️", pct + " — 예측 정확도가 낮아질 수 있음"))
        else:
            checks.append((f"{label}: vocab 매칭률", "✅", pct))

    _print_checks(checks)
    return {
        "n_cells": n_cells,
        "n_genes": n_genes,
        "vocab_overlap_ratio": overlap_ratio,
    }


def load_h5ad_full(path: str):
    """
    validate_h5ad()는 셀/유전자 수·컬럼 존재 여부만 가볍게 확인하려고 backed 모드로
    읽지만(전체를 메모리에 올리지 않음), mode: embed(reference mapping)처럼 실제로
    임베딩을 계산하려면 전체 데이터가 메모리에 있어야 한다 - 그 용도의 일반 h5ad
    로더. h5ad를 다루는 다른 model-agnostic 로직과 마찬가지로 pipeline/ 쪽에 둔다
    (run.py는 anndata를 직접 import하지 않는다는 설계 원칙 유지).
    """
    import anndata as ad

    return ad.read_h5ad(path)


def _get_gene_names(adata, gene_name_col: str) -> List[str]:
    if gene_name_col in adata.var.columns:
        return adata.var[gene_name_col].astype(str).tolist()
    return adata.var_names.astype(str).tolist()


def _print_checks(checks) -> None:
    table = Table(title="h5ad 입력 검증 결과", show_lines=False)
    table.add_column("항목", style="bold")
    table.add_column("결과", justify="center")
    table.add_column("비고", style="dim")
    for name, status, note in checks:
        table.add_row(name, status, note)
    console.print(table)
