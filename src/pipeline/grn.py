"""
pipeline/grn.py
mode: grn (유전자 임베딩 기반 GRN/모듈 분석) 전용 - 모델 무관 순수 분석 로직.

scGPT 공식 tutorials/Tutorial_GRN.ipynb를 GitHub 원본에서 직접 fetch해서 확인한 방식을
독립적으로 재구현했다. scgpt.tasks.GeneEmbedding 클래스를 그대로 import하지 않는 이유:
run.py의 설계 원칙("scgpt import 한 줄도 없음")을 pipeline/ 쪽에서 지키기 위해서다 -
integration.py가 scib/scanpy 같은 범용 도구만 쓰는 것과 동일한 패턴. GeneEmbedding이
받는 입력(Dict[유전자 이름, 임베딩 벡터])과 하는 일(클러스터링/네트워크/점수화) 자체는
모델과 무관한 범용 로직이라, scanpy/sklearn/networkx 같은 범용 라이브러리만으로 같은
결과를 낼 수 있다 - adapter.extract_gene_embeddings()가 Dict[str, np.ndarray]만
반환하면(scGPT든 나중에 Geneformer든) 이 모듈이 그대로 받아 쓴다.

신뢰도 표기(중요): mode: embed/integration은 튜토리얼 코드를 한 줄씩 대조해서 완전히
동일한 API 호출로 구현했다(README/project 문서에 그 근거를 남김). 이 모듈은 저작권
정책상 scgpt.tasks.grn.GeneEmbedding의 소스 코드 자체를 인용할 수 없어서, WebFetch로
얻은 요약(neighbors+leiden 클러스터링과 resolution 파라미터, leiden 클러스터 = "metagene",
sc.tl.score_genes 기반 점수화 + MinMaxScaler 정규화, cosine similarity 기반 네트워크,
gseapy.enrichr(gene_sets=['Reactome_2022'], organism='Human') pathway enrichment)를
근거로 독립적으로 재구현했다. 알고리즘의 큰 흐름은 원본과 동일하지만, 세부 정규화 축
등 소스 코드 수준까지 100% 동일하다고 보증하지는 않는다 - 실제 GPU 실행 결과를 보고
필요하면 조정할 것 (mode: embed/integration과의 검증 신뢰도 차이를 README에도 명시).

파이프라인 흐름:
  1. adapter.extract_gene_embeddings()로 얻은 Dict[gene, embedding vector]를
     cluster_gene_embeddings()로 Leiden 클러스터링한다 (resolution 파라미터로 클러스터
     개수/크기 조절 - 튜토리얼 예시값 40).
  2. get_metagenes()로 클러스터 id -> 유전자 리스트를 뽑는다 ("metagene": scGPT
     임베딩 공간에서 서로 가깝게 묶인 유전자 그룹 - 실제 조절 관계를 증명하진 않지만,
     함께 발현/기능할 가능성이 있는 유전자 모듈 후보).
  3. score_metagenes()로 각 metagene이 실제 데이터의 어느 cell type에서 발현 점수가
     높은지 계산한다 (sc.tl.score_genes 기반).
  4. enrich_metagenes()로 상위 N개 metagene에 대해 Reactome pathway enrichment를
     구한다 (gseapy, Enrichr 온라인 API 호출 - 이 프로젝트 HPC 환경의 실제 인터넷
     접근 여부는 미확인이므로 클러스터 단위 try/except로 실패해도 나머지는 계속
     진행하고, config의 grn_skip_enrichment로 아예 끌 수도 있게 했다).
  5. save_metagene_network()로 대표 metagene 하나의 유전자 간 cosine similarity
     네트워크를 그려서 저장한다.
"""

import collections
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1) 유전자 임베딩 클러스터링
# ---------------------------------------------------------------------------
def cluster_gene_embeddings(gene_embeddings: Dict[str, np.ndarray], resolution: float = 40.0):
    """
    유전자 임베딩 벡터들을 scanpy 표준 클러스터링 파이프라인(PCA -> neighbors -> Leiden
    -> UMAP)에 그대로 태운다 - 관측 단위가 "세포"가 아니라 "유전자"라는 점만 다르고
    절차는 동일하다 (scGPT GRN 튜토리얼의 GeneEmbedding.get_adata()와 같은 방식).

    resolution이 클수록(튜토리얼 예시값: 40) 더 많고 더 작은 클러스터로 나뉜다. 세포
    클러스터링에 흔히 쓰는 resolution(보통 0.1~2)보다 훨씬 큰 값을 쓰는 이유는, 여기서는
    "세포 타입" 수준이 아니라 훨씬 더 세분화된 "유전자 모듈" 단위로 나누고 싶기 때문 -
    원본 튜토리얼의 선택을 그대로 따랐다.

    반환값: obs_names가 유전자 이름이고 obs['leiden']에 클러스터 id가 담긴 AnnData.
    """
    import anndata as ad
    import scanpy as sc

    genes = list(gene_embeddings.keys())
    if len(genes) < 3:
        raise ValueError(f"클러스터링하기엔 유전자가 너무 적습니다 ({len(genes)}개, 최소 3개 필요)")

    X = np.stack([gene_embeddings[g] for g in genes]).astype(np.float32)
    gdata = ad.AnnData(X=X)
    gdata.obs_names = genes

    n_pcs = max(2, min(50, X.shape[1] - 1, X.shape[0] - 1))
    sc.pp.pca(gdata, n_comps=n_pcs)
    sc.pp.neighbors(gdata)
    sc.tl.leiden(gdata, resolution=resolution)
    sc.tl.umap(gdata)

    n_clusters = gdata.obs["leiden"].nunique()
    logger.info(f"유전자 {gdata.n_obs}개를 {n_clusters}개 metagene으로 클러스터링 (resolution={resolution})")
    return gdata


def get_metagenes(gdata) -> Dict[str, List[str]]:
    """
    클러스터 id -> 유전자 이름 리스트. gdata는 obs_names(유전자 이름 iterable)와
    obs["leiden"](클러스터 id iterable)만 있으면 되므로, 실제 AnnData가 아니어도
    (테스트용 스텁 포함) 이 인터페이스만 맞으면 동작한다.

    유전자 수가 많은 클러스터부터 정렬해서 반환한다 - top-N을 고를 때(enrichment,
    네트워크 시각화) 자연스럽게 "가장 큰/중요할 가능성이 높은" 것부터 순서가 매겨진다.
    """
    metagenes: Dict[str, List[str]] = collections.defaultdict(list)
    for gene, cluster in zip(gdata.obs_names, gdata.obs["leiden"]):
        metagenes[str(cluster)].append(gene)
    return dict(sorted(metagenes.items(), key=lambda kv: -len(kv[1])))


def save_metagene_assignments(metagenes: Dict[str, List[str]], output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "grn_metagenes.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metagene", "n_genes", "genes"])
        for cluster_id, genes in metagenes.items():
            writer.writerow([cluster_id, len(genes), ";".join(genes)])
    logger.info(f"metagene 목록 저장: {out_path.name} ({len(metagenes)}개 metagene)")
    return out_path


# ---------------------------------------------------------------------------
# 2) metagene의 cell type별 발현 점수화
# ---------------------------------------------------------------------------
def score_metagenes(adata, metagenes: Dict[str, List[str]], celltype_col: str, gene_col: Optional[str] = None):
    """
    각 metagene(유전자 리스트)의 발현 점수를 sc.tl.score_genes()로 세포별로 계산한 뒤,
    cell type별 평균을 내서 (cell type x metagene) 점수 행렬을 만든다. 각 metagene의
    점수 컬럼은 0~1로 정규화한다(MinMaxScaler, 컬럼 단위) - metagene마다 유전자 수/발현
    스케일이 다르므로 원본 점수 그대로는 metagene 간 비교가 무의미하기 때문이다.

    adata.var의 gene_col(없으면 index)에서 metagene의 유전자가 2개 미만만 발견되면
    (예: 데이터에서 필터링돼 빠진 유전자만 있는 클러스터) 그 metagene은 건너뛴다.
    실패한 개별 metagene이 있어도 나머지는 계속 계산한다(try/except).
    """
    import pandas as pd
    import scanpy as sc
    from sklearn.preprocessing import MinMaxScaler

    if gene_col and gene_col in adata.var.columns:
        var_name_set = set(adata.var[gene_col].astype(str).tolist())
    else:
        var_name_set = set(adata.var_names.astype(str).tolist())

    score_cols = []
    for cluster_id, genes in metagenes.items():
        genes_present = [g for g in genes if g in var_name_set]
        if len(genes_present) < 2:
            continue
        score_name = f"metagene_{cluster_id}"
        try:
            sc.tl.score_genes(adata, gene_list=genes_present, score_name=score_name, use_raw=False)
            score_cols.append(score_name)
        except Exception as e:
            logger.warning(f"metagene {cluster_id} 점수화 실패 (건너뜀): {e}")

    if not score_cols:
        logger.warning("점수화에 성공한 metagene이 하나도 없습니다.")
        return pd.DataFrame()

    mean_by_celltype = adata.obs.groupby(celltype_col)[score_cols].mean()
    normalized = pd.DataFrame(
        MinMaxScaler().fit_transform(mean_by_celltype),
        index=mean_by_celltype.index,
        columns=mean_by_celltype.columns,
    )
    logger.info(f"metagene {len(score_cols)}개 x cell type {normalized.shape[0]}개 점수 행렬 계산 완료")
    return normalized


def save_metagene_scores(score_df, output_dir: Path) -> Optional[Path]:
    """
    score_metagenes()가 계산한 (cell type x metagene) 점수 전체를 CSV로 저장한다.
    metagene이 수백 개(예: 실제 GPU 실행에서 resolution=40으로 391개 확인됨,
    Phase 12)면 save_metagene_heatmap()이 가독성을 위해 상위 일부만 그리므로,
    잘려나간 나머지 metagene의 점수도 항상 어딘가에는 남기기 위한 함수 - heatmap과
    달리 이건 생략되지 않는다(단, score_df가 비어있으면 파일 자체를 만들지 않는다).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if score_df is None or score_df.empty:
        logger.warning("metagene 점수 데이터가 비어있어 grn_metagene_scores.csv를 생략합니다.")
        return None
    out_path = output_dir / "grn_metagene_scores.csv"
    score_df.to_csv(out_path)
    logger.info(f"metagene 점수 전체({score_df.shape[1]}개 metagene) 저장: {out_path.name}")
    return out_path


def _select_top_n_metagene_columns(score_df, top_n: Optional[int]):
    """
    heatmap용으로 표시할 metagene(컬럼)을 상위 top_n개로 제한한다. score_df의 컬럼은
    score_metagenes()가 get_metagenes()의 순서(유전자 수 많은 순 정렬)를 그대로
    보존해서 만들기 때문에, 앞에서부터 top_n개를 자르면 곧 "가장 큰 metagene부터
    top_n개"가 된다 - 순서를 다시 계산하지 않는다. top_n이 None이거나 컬럼 수보다
    크거나 같으면 전체를 그대로 반환한다(자르지 않음).
    """
    if top_n is None or score_df.shape[1] <= top_n:
        return score_df
    return score_df.iloc[:, :top_n]


def save_metagene_heatmap(score_df, output_path: Path, top_n: Optional[int] = None) -> Optional[Path]:
    """
    metagene별 cell type 발현 점수 heatmap을 저장한다. metagene(컬럼) 수가 많으면
    (예: resolution을 높게 잡아 수백 개가 나온 경우) 컬럼마다 폭을 확보하는 기존 방식이
    극단적으로 넓고 얇은(그래서 사실상 못 읽는) 이미지를 만드는 문제가 실제 GPU
    실행에서 확인됐다(Phase 12, 391개 metagene → 그림 폭 195인치). 그래서 top_n이
    주어지면 (get_metagenes()가 이미 정렬해둔 순서상) 가장 큰 top_n개 metagene만
    그리고, 전체 데이터는 잘리지 않는다는 걸 보장하기 위해 save_metagene_scores()가
    항상 별도로 전체를 CSV에 저장한다(이 함수는 표시만 담당, 데이터 보존은 그쪽 책임).
    """
    if score_df is None or score_df.empty:
        logger.warning("metagene 점수 데이터가 비어있어 heatmap을 생략합니다.")
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    total_n = score_df.shape[1]
    plot_df = _select_top_n_metagene_columns(score_df, top_n)
    shown_n = plot_df.shape[1]

    fig_w = max(6.0, min(40.0, shown_n * 0.5))
    fig_h = max(4.0, plot_df.shape[0] * 0.4)
    g = sns.clustermap(plot_df, cmap="viridis", figsize=(fig_w, fig_h))
    if shown_n < total_n:
        g.fig.suptitle(f"상위 {shown_n}개 / 전체 {total_n}개 metagene (나머지는 grn_metagene_scores.csv 참고)", y=1.02)
    g.savefig(output_path, dpi=150)
    plt.close("all")
    logger.info(
        f"metagene heatmap 저장: {Path(output_path).name}"
        + (f" (전체 {total_n}개 중 상위 {shown_n}개만 표시)" if shown_n < total_n else "")
    )
    return Path(output_path)


# ---------------------------------------------------------------------------
# 3) pathway enrichment (온라인 API - 실패해도 전체 파이프라인은 계속됨)
# ---------------------------------------------------------------------------
def enrich_metagenes(
    metagenes: Dict[str, List[str]],
    top_n: int = 10,
    gene_sets: Optional[List[str]] = None,
    organism: str = "Human",
):
    """
    유전자 수 기준 상위 top_n개 metagene에 대해 gseapy.enrichr()로 pathway enrichment를
    계산한다 (Reactome_2022가 원본 튜토리얼의 선택 - GitHub 원본에서 직접 확인함).
    gseapy.enrichr()는 Enrichr(maayanlab.cloud) 온라인 API를 호출하므로 이 함수가
    실행되는 서버에 아웃바운드 인터넷 접근이 필요하다 - 이 프로젝트의 HPC 실행 환경
    (gnode01)이 실제로 인터넷에 접근 가능한지는 아직 확인된 적이 없다(README에 위험으로
    명시). 그래서:
      - gseapy 자체가 설치돼 있지 않으면(ImportError) 전체를 건너뛰고 전부 None 반환.
      - 클러스터 하나가 실패해도(네트워크 오류, API 무응답 등) 나머지 클러스터는 계속
        시도한다(클러스터 단위 try/except).
      - run.py 쪽에서 config의 grn_skip_enrichment: true로 이 단계 자체를 아예 끌 수도
        있다.
    """
    if gene_sets is None:
        gene_sets = ["Reactome_2022"]

    top_clusters = list(metagenes.items())[:top_n]

    try:
        import gseapy as gp
    except ImportError:
        logger.warning("gseapy가 설치되어 있지 않아 pathway enrichment를 건너뜁니다.")
        return {cluster_id: None for cluster_id, _ in top_clusters}

    results: Dict[str, Optional[object]] = {}
    for cluster_id, genes in top_clusters:
        if len(genes) < 3:
            results[cluster_id] = None
            continue
        try:
            enr = gp.enrichr(gene_list=genes, gene_sets=gene_sets, organism=organism, outdir=None, cutoff=0.5)
            results[cluster_id] = enr.results
        except Exception as e:
            logger.warning(f"metagene {cluster_id} enrichment 실패 (네트워크/API 문제 가능성, 건너뜀): {e}")
            results[cluster_id] = None

    n_ok = sum(1 for v in results.values() if v is not None)
    logger.info(f"pathway enrichment: {n_ok}/{len(top_clusters)}개 metagene 성공")
    return results


def save_enrichment_results(results: Dict[str, Optional[object]], output_dir: Path, top_k_terms: int = 10) -> Path:
    """metagene별 enrichment 결과에서 P-value 기준 상위 top_k_terms개 pathway만 추려
    하나의 CSV로 합쳐 저장한다. 전부 실패(None)했으면 헤더만 있는 빈 CSV를 만든다 -
    "enrichment를 시도했지만 결과가 없다"와 "아예 시도하지 않았다(grn_skip_enrichment)"를
    output 파일 존재 여부로 구분할 수 있게 하기 위함."""
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cluster_id, df in results.items():
        if df is None or getattr(df, "empty", True):
            continue
        top = df.sort_values("P-value").head(top_k_terms)
        for _, r in top.iterrows():
            rows.append({
                "metagene": cluster_id,
                "term": r.get("Term"),
                "p_value": r.get("P-value"),
                "adjusted_p_value": r.get("Adjusted P-value"),
                "genes": r.get("Genes"),
            })

    out_path = output_dir / "grn_enrichment.csv"
    columns = ["metagene", "term", "p_value", "adjusted_p_value", "genes"]
    pd.DataFrame(rows, columns=columns).to_csv(out_path, index=False)
    logger.info(f"pathway enrichment 결과 저장: {out_path.name} ({len(rows)}개 term)")
    return out_path


# ---------------------------------------------------------------------------
# 4) 유전자 유사도 네트워크
# ---------------------------------------------------------------------------
def save_metagene_network(
    gene_embeddings: Dict[str, np.ndarray],
    genes: List[str],
    output_path: Path,
    similarity_threshold: float = 0.5,
    title: str = "",
) -> Optional[Path]:
    """
    genes 목록(보통 대표 metagene 하나, 너무 크면 run.py가 미리 상한을 잘라서 넘긴다)에
    대해 cosine similarity 기반 네트워크를 그린다 - scGPT GRN 튜토리얼의 NetworkX 그래프
    구성(유전자 쌍 간 cosine similarity가 threshold를 넘는 엣지만 남김)과 동일한 방식.
    """
    genes_present = [g for g in genes if g in gene_embeddings]
    if len(genes_present) < 2:
        logger.warning("네트워크를 그리기엔 유전자가 너무 적어 생략합니다.")
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    from sklearn.metrics.pairwise import cosine_similarity

    vectors = np.stack([gene_embeddings[g] for g in genes_present])
    sim = cosine_similarity(vectors)

    G = nx.Graph()
    G.add_nodes_from(genes_present)
    for i in range(len(genes_present)):
        for j in range(i + 1, len(genes_present)):
            s = float(sim[i, j])
            if s >= similarity_threshold:
                G.add_edge(genes_present[i], genes_present[j], weight=round(s, 2))

    fig, ax = plt.subplots(figsize=(10, 10))
    pos = nx.spring_layout(G, k=0.4, iterations=15, seed=3)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color="skyblue", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
    if weights:
        nx.draw_networkx_edges(G, pos, width=[w * 2 for w in weights], alpha=0.5, ax=ax)
    ax.set_title(
        f"{title} (cosine similarity >= {similarity_threshold}, "
        f"유전자 {G.number_of_nodes()}개, 엣지 {G.number_of_edges()}개)"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"metagene 네트워크 저장: {Path(output_path).name} (노드 {G.number_of_nodes()}, 엣지 {G.number_of_edges()})")
    return Path(output_path)
