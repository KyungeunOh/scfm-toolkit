"""
pipeline/integration.py
mode: integration(zero-shot batch integration) 전용 - 모델 무관 순수 분석 로직.

scGPT의 공식 tutorials/zero-shot/Tutorial_ZeroShot_Integration.ipynb를 그대로 조사해서
(GitHub 원본 코드를 직접 fetch해서 한 줄씩 대조 - 짐작으로 작성하지 않음) 같은 방식으로
구현했다:

  1. reference/query 구분 없이 여러 batch(샘플)가 섞인 h5ad 하나를 임베딩한다
     (adapter.embed() 재사용 - mode: embed용으로 이미 GPU에서 검증된 그 메서드
     그대로, 새 adapter 코드가 필요 없다).
  2. 임베딩 공간에서 UMAP을 그려 batch는 섞이고 cell type은 분리되는지 육안으로 확인.
  3. scib 패키지로 정량 지표를 계산한다 - 튜토리얼의 scib_eval()과 완전히 동일한
     scib.metrics.metrics() 호출/플래그를 그대로 재사용했다 (아래 evaluate_integration
     docstring 참고). requirements.txt에 scib==1.1.7이 이미 있고, 이 프로젝트
     Docker 이미지(scgpt-toolkit:v0.1)는 이미 이 requirements.txt로 빌드되어
     annotation/embed 두 모드 모두 실제로 GPU에서 실행 검증까지 끝난 상태라, scib도
     이미 설치돼 있을 것으로 강하게 추정된다(별도 설치 불필요) - 다만 실제
     `import scib`까지 이 세션에서 확인하지는 못했으므로 처음 실행 시 확인 필요.

reference_mapping.py와 마찬가지로 이 모듈은 어떤 adapter가 만든 임베딩이든(scgpt든
나중에 geneformer든) 그대로 받아서 쓰는 모델 무관 로직이다 - adapter.embed()가
(n_cells, embed_dim) numpy 배열만 반환하면 된다는 계약을 그대로 재사용한다.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def evaluate_integration(
    adata,
    batch_key: str,
    celltype_key: str,
    embed_key: str = "X_scGPT",
) -> Dict[str, float]:
    """
    scib.metrics.metrics()로 batch 통합 품질을 정량 평가한다.

    Tutorial_ZeroShot_Integration.ipynb의 scib_eval() 함수와 완전히 동일한 인자를
    쓴다 (원본 노트북을 GitHub에서 직접 fetch해서 대조 확인함):

      scib.metrics.metrics(
          adata, adata_int=adata, batch_key=batch_key, label_key=celltype_key,
          embed=embed_key, isolated_labels_asw_=False, silhouette_=True,
          hvg_score_=False, graph_conn_=True, pcr_=True,
          isolated_labels_f1_=False, trajectory_=False, nmi_=True, ari_=True,
          cell_cycle_=False, kBET_=False, ilisi_=False, clisi_=False,
      )

    비활성화한 지표(isolated_labels_asw_/f1_, hvg_score_, trajectory_, cell_cycle_,
    kBET_, ilisi_, clisi_)는 원본 튜토리얼도 끈 것들이다 - kBET/iLISI/cLISI는 R
    의존성(rpy2)이 필요해질 수 있어 특히 그대로 꺼둔 채 가져왔다(추가 의존성 위험
    회피). 켜둔 지표: NMI/ARI(label 기준 클러스터링과 실제 label의 일치도 -
    "생물학적 신호가 보존됐는가"), ASW_label(cell type별 임베딩 분리도),
    ASW_label/batch + graph_conn(batch가 얼마나 섞였는가), PCR(주성분이 batch를
    얼마나 설명하는가, 낮을수록 좋음).

    반환값에 avg_bio(NMI/ARI/ASW_label 평균), avg_batch(graph_conn/ASW_label_batch
    평균)를 원본과 동일한 방식으로 추가한다. NaN인 지표는 결과에서 제외한다
    (scib가 일부 조건에서 확인함는 반환하기 때문 - 원본 튜토리얼의
    scib_eval()도 동일하게 필터링한다).
    """
    import numpy as np
    import scib

    results = scib.metrics.metrics(
        adata,
        adata_int=adata,
        batch_key=batch_key,
        label_key=celltype_key,
        embed=embed_key,
        isolated_labels_asw_=False,
        silhouette_=True,
        hvg_score_=False,
        graph_conn_=True,
        pcr_=True,
        isolated_labels_f1_=False,
        trajectory_=False,
        nmi_=True,
        ari_=True,
        cell_cycle_=False,
        kBET_=False,
        ilisi_=False,
        clisi_=False,
    )
    result_dict = results[0].to_dict()

    bio_keys = ["NMI_cluster/label", "ARI_cluster/label", "ASW_label"]
    batch_keys = ["graph_conn", "ASW_label/batch"]
    if all(k in result_dict for k in bio_keys):
        result_dict["avg_bio"] = float(np.mean([result_dict[k] for k in bio_keys]))
    if all(k in result_dict for k in batch_keys):
        result_dict["avg_batch"] = float(np.mean([result_dict[k] for k in batch_keys]))

    result_dict = {k: float(v) for k, v in result_dict.items() if v is not None and not np.isnan(v)}
    logger.info(f"scib 통합 품질 지표: {result_dict}")
    return result_dict


def save_integration_metrics(metrics: Dict, output_dir: Path, filename: str = "integration_metrics.json") -> Path:
    """scib 지표 dict를 JSON으로 저장한다 (pipeline/report.py의 metrics.json 저장
    방식과 동일한 패턴 - run.py는 파일 형식을 몰라도 되게 여기서 캡슐화)."""
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return path


def save_integration_umap(
    adata,
    batch_key: str,
    celltype_key: str,
    output_path: Path,
    embed_key: str = "X_scGPT",
    title_prefix: str = "scGPT zero-shot",
) -> Path:
    """
    embed_key(기본 X_scGPT) 임베딩으로 neighbor graph + UMAP을 계산하고,
    cell type / batch 두 패널로 색칠한 그림을 저장한다.

    Tutorial_ZeroShot_Integration.ipynb의 시각화 코드(sc.pp.neighbors(use_rep=...) →
    sc.tl.umap → sc.pl.umap(color=[celltype_key, batch_key]))를 그대로 따른다.
    matplotlib을 Agg 백엔드로 강제해 GUI 없는 서버 환경(Docker/SLURM)에서도 동작한다
    (pipeline/report.py의 confusion_matrix.png 저장 방식과 동일한 패턴).
    """
    import matplotlib
    matplotlib.use("Agg")
    import scanpy as sc

    sc.pp.neighbors(adata, use_rep=embed_key)
    sc.tl.umap(adata)
    fig = sc.pl.umap(
        adata,
        color=[celltype_key, batch_key],
        frameon=False,
        wspace=0.4,
        title=[f"{title_prefix}: cell type", f"{title_prefix}: batch"],
        show=False,
        return_fig=True,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    logger.info(f"UMAP 저장: {output_path.name}")
    return output_path
