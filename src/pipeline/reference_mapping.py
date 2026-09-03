"""
pipeline/reference_mapping.py

mode: embed (reference mapping)에서 쓰는, 모델에 무관한 k-NN 기반 label 전파 로직.

scGPT Tutorial_Reference_Mapping.ipynb의 워크플로우 - reference를 임베딩하고,
query를 임베딩한 뒤, 임베딩 공간에서 k-NN 다수결로 reference의 label을 query에
전파한다 - 를 그대로 옮기되, "임베딩을 어떻게 만드는가"는 adapter.embed()가
담당하고 여기서는 "임베딩이 주어졌을 때 어떻게 매핑하는가"만 다룬다. 그래서
scGPT뿐 아니라 embed()를 구현하는 어떤 모델에도 그대로 재사용 가능하다
(run.py/pipeline 쪽에서는 이 파일도 어떤 모델의 임베딩인지 전혀 모른다).
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def knn_label_transfer(
    ref_embeddings: np.ndarray,
    ref_labels: np.ndarray,
    query_embeddings: np.ndarray,
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    reference 임베딩+label로 query 임베딩의 label을 예측한다.

    반환: (예측 label 배열, 신뢰도 점수 배열)
      신뢰도 점수 = k개 이웃 중 다수결로 뽑힌 label의 비율(0~1) - pipeline/report.py의
      low_confidence 로직/컬럼 관례(pred_score)와 그대로 호환된다.

    faiss가 설치돼 있으면(scGPT 환경엔 requirements.txt의 faiss-gpu로 기본 포함) 그걸
    쓰고, 없으면 scikit-learn의 NearestNeighbors로 폴백한다 - 공식 튜토리얼도 동일한
    폴백 패턴("faiss not installed! We highly recommend installing it...")을 쓴다.
    """
    n_ref = len(ref_labels)
    if n_ref == 0:
        raise ValueError("reference 임베딩/label이 비어 있습니다.")
    k = min(k, n_ref)

    ref_embeddings = np.ascontiguousarray(ref_embeddings, dtype=np.float32)
    query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)

    try:
        import faiss

        index = faiss.IndexFlatL2(ref_embeddings.shape[1])
        index.add(ref_embeddings)
        _, neighbor_idx = index.search(query_embeddings, k)
        logger.info(f"faiss로 k-NN 탐색 완료 (k={k})")
    except ImportError:
        logger.warning(
            "faiss가 설치돼 있지 않아 scikit-learn NearestNeighbors로 대체합니다 "
            "(reference/query가 크면 느릴 수 있음 - faiss 설치를 권장)."
        )
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(ref_embeddings)
        _, neighbor_idx = nn.kneighbors(query_embeddings)

    ref_labels = np.asarray(ref_labels)
    pred_labels = []
    pred_scores = []
    for row in neighbor_idx:
        neighbor_labels = ref_labels[row]
        labels, counts = np.unique(neighbor_labels, return_counts=True)
        winner = np.argmax(counts)
        pred_labels.append(labels[winner])
        pred_scores.append(counts[winner] / len(row))

    return np.array(pred_labels), np.array(pred_scores, dtype=float)
