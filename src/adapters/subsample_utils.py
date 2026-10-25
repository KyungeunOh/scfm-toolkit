"""reference type별 subsampling (교수님 2026-10-08 요청)."""
import numpy as np


def stratified_subsample_indices(y, n_per_type, seed):
    """y: 1차원 정수 라벨. 각 type에서 최대 n_per_type개의 인덱스를 고른다
    (세포 수가 n_per_type보다 적은 type은 있는 만큼 전부).

    - type마다 RandomState([seed, label])로 순열을 만들고 앞에서 n개를 취한다.
      -> 같은 (seed, y)이면 scope/모델과 무관하게 항상 같은 subsample (scope 간 paired 비교),
         같은 seed에서 N=10 subsample은 N=50 subsample의 부분집합 (nested)."""
    y = np.asarray(y)
    picked = []
    for lab in np.unique(y):
        idx = np.flatnonzero(y == lab)
        perm = np.random.RandomState([int(seed), int(lab)]).permutation(len(idx))
        picked.append(idx[perm[: int(n_per_type)]])
    return np.sort(np.concatenate(picked))
