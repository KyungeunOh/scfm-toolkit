#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="scgpt-toolkit:v0.1"

DATA_DIR="${SCRIPT_DIR}/data"
MODEL_DIR="${SCRIPT_DIR}/model"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
CONFIG_PATH="${SCRIPT_DIR}/config/config.yaml"

mkdir -p "${OUTPUT_DIR}"

echo "==================================================="
echo "scGPT 어노테이션 툴킷 시작"
echo "  데이터:  ${DATA_DIR}"
echo "  모델:    ${MODEL_DIR}"
echo "  결과:    ${OUTPUT_DIR}"
echo "==================================================="

docker run --rm \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  -e WANDB_MODE=offline \
  -v "${DATA_DIR}":/workspace/data:ro \
  -v "${MODEL_DIR}":/workspace/model:ro \
  -v "${OUTPUT_DIR}":/workspace/outputs \
  -v "${CONFIG_PATH}":/workspace/config.yaml:ro \
  -v "${SCRIPT_DIR}/src":/workspace/src:ro \
  "${IMAGE}" \
  --config /workspace/config.yaml

echo "==================================================="
echo "완료. 결과: ${OUTPUT_DIR}"
echo "==================================================="
