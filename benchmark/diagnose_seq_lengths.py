"""
benchmark/diagnose_seq_lengths.py

2026-09-17 priority_grid 결과에서 max_seq_len=1500과 max_seq_len=3001(smoke)이
peak_allocated_mb를 거의 완전히 동일하게(9746.87MB vs ~9740MB) 낸 이유를 확인하기
위한 1회성 진단 스크립트. GPU/모델 로드 없이 데이터 전처리(Step 3~5)까지만 실행해서
"실제로 모델에 들어가는 시퀀스 길이"의 원천이 되는 값 - 세포별 0이 아닌(발현된)
유전자 수 분포 - 를 직접 찍어본다.

가설: adapters/scgpt_adapter.py의 prepare_inputs()가 쓰는
scgpt.tokenizer.tokenize_and_pad_batch(..., max_len=max_seq_len, include_zero_gene=False)는
"각 세포마다 최대 max_seq_len개까지 발현 유전자를 담고, 부족하면 <pad>로 채운다"는
방식이다. 만약 이 데이터셋에서 세포당 발현 유전자 수(0이 아닌 값의 개수)가 이미
1500 근처(또는 그 이하)에서 정체돼 있다면 - 즉 max_seq_len=1500이든 3001이든
"부족해서 <pad>로 채우는" 상황이 똑같이 발생한다면 - 두 설정의 실제 시퀀스 길이가
같아져서 메모리도 같게 나온다. 반대로 max_seq_len=500처럼 실제 발현 유전자 수보다
작은 값을 주면 그때는 진짜로 잘라내야 하니 메모리가 줄어든다(실측 결과와 일치).

이 스크립트는 그 "세포당 발현 유전자 수" 분포를 직접 찍어서 위 가설이 맞는지
확인한다. torch/GPU 불필요 - Step 3~5(load_data/load_vocab_full/preprocess)만 재사용.

사용법 (Docker, run_benchmark.sh와 동일한 마운트로):
  docker run --rm \\
    -v <REPO_ROOT>/data:/workspace/data:ro \\
    -v <REPO_ROOT>/model:/workspace/model:ro \\
    -v <REPO_ROOT>/config/config.yaml:/workspace/config.yaml:ro \\
    -v <REPO_ROOT>/src:/workspace/src:ro \\
    -v <REPO_ROOT>/benchmark:/workspace/benchmark:ro \\
    --entrypoint python scgpt-toolkit:v0.1 \\
    /workspace/benchmark/diagnose_seq_lengths.py --config /workspace/config.yaml
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
from scipy.sparse import issparse  # noqa: E402

from adapters import get_adapter  # noqa: E402
from pipeline.config import load_config  # noqa: E402

from benchmark.run_scgpt_sweep import _build_ctx  # noqa: E402 - Step 3~5 재사용


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg_base = load_config(args.config)
    adapter = get_adapter("scgpt")
    ctx = _build_ctx(cfg_base, adapter, quiet=False)

    adata = ctx["adata"]
    # prepare_inputs()와 동일하게 reference batch만 본다.
    adata_ref = adata[adata.obs["str_batch"] == "0"]
    X = adata_ref.layers["X_binned"]
    X = X.toarray() if issparse(X) else np.asarray(X)
    nonzero_per_cell = (X != 0).sum(axis=1)

    print("=" * 80)
    print(f"reference 세포 수: {X.shape[0]}, HVG+vocab 교집합 후 전체 gene 수: {X.shape[1]}")
    print("세포당 '0이 아닌(발현된) gene 수' 분포 (tokenize_and_pad_batch가 include_zero_gene="
          f"{cfg_base.get('include_zero_gene', False)}로 이 값 이하만 실제로 채운다):")
    print(f"  min    = {int(nonzero_per_cell.min())}")
    print(f"  median = {float(np.median(nonzero_per_cell)):.0f}")
    print(f"  p95    = {float(np.percentile(nonzero_per_cell, 95)):.0f}")
    print(f"  max    = {int(nonzero_per_cell.max())}")
    print("=" * 80)
    print(
        "이 max 값이 1500보다 작다면: max_seq_len=1500과 3001이 똑같은 메모리로 나온 "
        "이유가 확인된 것 (요청한 max_seq_len과 무관하게, 실제 데이터의 세포당 최대 "
        "발현 gene 수가 사실상의 상한 역할을 함). max_seq_len 스윕을 다시 설계한다면 "
        "이 max 값보다 작은 지점(예: 그 절반, 3/4 지점 등) 여러 개를 잡아야 실제로 "
        "메모리가 달라지는 구간을 볼 수 있다."
    )


if __name__ == "__main__":
    main()
