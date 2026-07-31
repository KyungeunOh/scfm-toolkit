#!/usr/bin/env bash
# =========================================================================
# test_gpu.sh - 기존 코드(src/run.py, run_annotation.py) 전혀 건드리지 않고
# 기존 이미지(scgpt-toolkit:v0.1)로 GPU만 빠르게 확인하는 스크립트.
#
# nvidia-container-toolkit이 이제 설치됐다는 전제로 --gpus all을 사용.
# 파이프라인 전체(7 step)를 돌리지 않고, cuBLAS까지 실제로 찍어보는
# 최소 테스트만 수행 -> 몇 초 안에 결과 확인 가능.
# =========================================================================
set -euo pipefail

IMAGE="scgpt-toolkit:v0.1"

echo "==================================================="
echo "GPU 스모크 테스트 (--gpus all, 기존 코드 미변경)"
echo "==================================================="

docker run --rm --gpus all --entrypoint python "${IMAGE}" -c "
import torch
print('torch version   :', torch.__version__)
print('torch CUDA build:', torch.version.cuda)
print('cuda available  :', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('CUDA를 사용할 수 없습니다. nvidia-smi / --gpus all 마운트를 확인하세요.')

print('device name     :', torch.cuda.get_device_name(0))
print('capability      :', torch.cuda.get_device_capability(0))

# 22436.log에서 실패했던 지점: cuBLAS 행렬곱 (첫 forward pass에서 재현됨)
a = torch.randn(256, 256, device='cuda')
b = torch.randn(256, 256, device='cuda')
c = a @ b
torch.cuda.synchronize()
print('cuBLAS matmul OK, sum =', c.sum().item())

# scGPT가 실제로 쓰는 amp/autocast 경로까지 확인
with torch.cuda.amp.autocast():
    d = a @ b
torch.cuda.synchronize()
print('autocast(amp) matmul OK')

print()
print('=> GPU 정상 동작. 이제 ./run.sh (--gpus all로 수정 후) 로 전체 파이프라인 실행 가능.')
"
