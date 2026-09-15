#!/usr/bin/env bash
# benchmark/run_benchmark.sh
# run.sh와 같은 패턴(Docker 이미지 재사용, src/를 바인드 마운트해서 이미지 재빌드
# 없이 코드 변경 반영)으로 scGPT memory-benchmark sweep을 돌린다.
#
# run.sh와의 차이: ENTRYPOINT(python /workspace/src/run.py)를 그대로 쓰지 않고
# --entrypoint python으로 덮어써서 benchmark/run_scgpt_sweep.py를 대신 실행한다.
# 이미지 자체(scgpt-toolkit:v0.1)는 run.sh가 이미 빌드해둔 것을 그대로 재사용하므로
# 이 스크립트를 쓰기 전에 별도 재빌드는 필요 없다(benchmark/, src/ 모두 바인드
# 마운트라 이미지 안에 없어도 됨 - Dockerfile의 COPY src/는 빌드 시점 스냅샷일
# 뿐이고 run.sh/이 스크립트 둘 다 런타임에 호스트의 최신 파일로 덮어쓴다).
#
# 사용법:
#   bash benchmark/run_benchmark.sh smoke                  # 5개 조합 (처음엔 이것부터)
#   bash benchmark/run_benchmark.sh coarse                 # grid.py의 전체 조합
#   bash benchmark/run_benchmark.sh smoke 12               # 12GB memory budget 흉내
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
IMAGE="scgpt-toolkit:v0.1"

GRID="${1:-smoke}"
MEMORY_BUDGET_GB="${2:-}"

DATA_DIR="${REPO_ROOT}/data"
MODEL_DIR="${REPO_ROOT}/model"
CONFIG_PATH="${REPO_ROOT}/config/config.yaml"
BENCHMARK_RESULTS_DIR="${REPO_ROOT}/benchmark_results"
mkdir -p "${BENCHMARK_RESULTS_DIR}"

OUT_NAME="scgpt_sweep_${GRID}.csv"
if [ -n "${MEMORY_BUDGET_GB}" ]; then
  OUT_NAME="scgpt_sweep_${GRID}_${MEMORY_BUDGET_GB}gb_budget.csv"
fi

EXTRA_ARGS=()
if [ -n "${MEMORY_BUDGET_GB}" ]; then
  EXTRA_ARGS+=(--memory-budget-gb "${MEMORY_BUDGET_GB}")
fi

echo "==================================================="
echo "scGPT memory-benchmark sweep 시작"
echo "  grid:          ${GRID}"
echo "  memory budget: ${MEMORY_BUDGET_GB:-없음 (GPU 전체 사용)}"
echo "  결과:          ${BENCHMARK_RESULTS_DIR}/${OUT_NAME}"
echo "==================================================="

docker run --rm \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  -v "${DATA_DIR}":/workspace/data:ro \
  -v "${MODEL_DIR}":/workspace/model:ro \
  -v "${CONFIG_PATH}":/workspace/config.yaml:ro \
  -v "${REPO_ROOT}/src":/workspace/src:ro \
  -v "${REPO_ROOT}/benchmark":/workspace/benchmark:ro \
  -v "${BENCHMARK_RESULTS_DIR}":/workspace/benchmark_results \
  --entrypoint python \
  "${IMAGE}" \
  /workspace/benchmark/run_scgpt_sweep.py \
    --config /workspace/config.yaml \
    --grid "${GRID}" \
    --out "/workspace/benchmark_results/${OUT_NAME}" \
    "${EXTRA_ARGS[@]}"

echo "==================================================="
echo "완료. 결과: ${BENCHMARK_RESULTS_DIR}/${OUT_NAME}"
echo "요약 보기: python3 benchmark/summarize_results.py ${BENCHMARK_RESULTS_DIR}/${OUT_NAME}"
echo "==================================================="
