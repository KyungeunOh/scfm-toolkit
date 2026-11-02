#!/bin/bash
# scFoundation sweep 실행 스크립트. 사용: bash ~/scfm_run_grid.sh <grid이름> <config파일이름> [gpu_uuid]
#   예) bash ~/scfm_run_grid.sh pancreas_smoke config_scfoundation_pancreas.yaml
# GPU uuid 를 안 주면 호스트 이름으로 고른다 (gnode01 -> RTX 3090, gnode02 -> RTX A6000).
# DRY=1 을 앞에 붙이면 실행하지 않고 docker 명령만 출력한다. 예) DRY=1 bash ~/scfm_run_grid.sh seed_repeat config_scfoundation.yaml
GRID=$1
CFG=$2
if [ -z "$GRID" ] || [ -z "$CFG" ]; then
  echo "사용: bash ~/scfm_run_grid.sh <grid이름> <config파일이름> [gpu_uuid]"
  exit 1
fi
GPU=$3
if [ -z "$GPU" ]; then
  case "$(hostname)" in
    gnode01*) GPU=GPU-a2bdb475-8359-6e95-24bf-ae77ca09cc2a ;;
    gnode02*) GPU=GPU-cd8efe34-ffe3-c975-f634-397731d15321 ;;
    *) echo "이 호스트($(hostname))의 GPU uuid 를 모릅니다. 세 번째 인자로 주세요."; exit 1 ;;
  esac
fi
cd ~/scfm-toolkit || exit 1
if [ ! -f "config/${CFG}" ]; then
  echo "config 파일이 없습니다: config/${CFG}"
  exit 1
fi
mkdir -p benchmark_results/logs
TS=$(date +%Y%m%dT%H%M%S)
LOGFILE=benchmark_results/logs/scfoundation_${GRID}_${TS}.log
OUT=/workspace/benchmark_results/scfoundation_sweep_${GRID}.csv
CMD=(docker run --rm --gpus "\"device=${GPU}\""
  -v ~/scfm-toolkit/data:/workspace/data:ro
  -v ~/scfm-toolkit/model:/workspace/model:ro
  -v ~/scfm-toolkit/config/${CFG}:/workspace/config_scfoundation.yaml:ro
  -v ~/scfm-toolkit/src:/workspace/src:ro
  -v ~/scfm-toolkit/benchmark:/workspace/benchmark:ro
  -v ~/scfm-toolkit/benchmark_results:/workspace/benchmark_results
  -v ~/scFoundation:/workspace/scFoundation:ro
  --entrypoint python scgpt-toolkit:v0.2-scfoundation
  /workspace/benchmark/run_scfoundation_sweep.py
  --config /workspace/config_scfoundation.yaml
  --grid "${GRID}"
  --out "${OUT}")
if [ "${DRY:-0}" = "1" ]; then
  echo "[DRY] host=$(hostname) GPU=${GPU}"
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi
if [ "${FG:-0}" = "1" ]; then
  echo "[FG] 시작 $(date +%T) grid=${GRID} host=$(hostname)"
  "${CMD[@]}" > "$LOGFILE" 2>&1
  RC=$?
  echo "[FG] 종료 $(date +%T) grid=${GRID} exit=${RC} 로그=${LOGFILE}"
  exit 0
fi
nohup "${CMD[@]}" > "$LOGFILE" 2>&1 &
echo "시작함: grid=${GRID} config=${CFG} host=$(hostname) GPU=${GPU}"
echo "로그: ~/scfm-toolkit/${LOGFILE}"
echo "결과 CSV: ~/scfm-toolkit/benchmark_results/scfoundation_sweep_${GRID}.csv"
