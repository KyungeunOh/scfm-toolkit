#!/bin/bash
# Geneformer 어댑터 시험 실행(G2). 기본은 실행하지 않고 docker 명령만 출력한다(DRY).
# 실제 실행은 GO=1 을 앞에 붙인다.
#   명령만 보기:  bash ~/scfm-toolkit/scripts/run_geneformer_smoke.sh
#   실행:         GO=1 bash ~/scfm-toolkit/scripts/run_geneformer_smoke.sh
#   표본 크기:    NPT=10 NQ=100 (유형당 reference 세포 수, query 세포 수; 기본값)
#   root 로 실행: ASROOT=1  (기본은 내 계정으로 실행해서 결과 파일을 내가 지울 수 있게 한다)
# GPU 가 정확히 1장 보이는 srun 세션에서 실행한다.
NPT=${NPT:-10}
NQ=${NQ:-100}
REPO=~/scfm-toolkit

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NGPU" != "1" ]; then
  echo "GPU 가 정확히 1장 보이는 srun 세션에서 실행하세요 (지금 보이는 GPU 수: $NGPU)"
  exit 1
fi
GPU=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d ' ')

# 컨테이너에는 .git 이 없으므로 호스트에서 구해 넘긴다 (코드가 커밋과 다르면 -dirty)
GIT_COMMIT=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)
if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no -- src benchmark config scripts 2>/dev/null)" ]; then
  GIT_COMMIT="${GIT_COMMIT}-dirty"
fi
GF_COMMIT=$(git -C ~/Geneformer rev-parse HEAD 2>/dev/null)

mkdir -p "$REPO/benchmark_results/logs"
LOGFILE="$REPO/benchmark_results/logs/geneformer_smoke_$(date +%Y%m%dT%H%M%S).log"

USER_ARGS=(--user "$(id -u):$(id -g)" -e HOME=/tmp)
if [ "${ASROOT:-0}" = "1" ]; then USER_ARGS=(); fi

# 2026-11-19 batch: EPOCHS, LR, SEED 환경변수가 있으면 config 값을 덮어쓴다
EXTRA=()
[ -n "${EPOCHS:-}" ] && EXTRA+=(--epochs "$EPOCHS")
[ -n "${LR:-}" ] && EXTRA+=(--lr "$LR")
[ -n "${SEED:-}" ] && EXTRA+=(--seed "$SEED")

CMD=(docker run --rm --gpus "\"device=${GPU}\"" "${USER_ARGS[@]}"
  -e "SCFM_GIT_COMMIT=${GIT_COMMIT}" -e "GENEFORMER_COMMIT=${GF_COMMIT}"
  -v "$REPO/data:/workspace/data:ro"
  -v ~/Geneformer:/workspace/Geneformer:ro
  -v "$REPO/config/config_geneformer.yaml:/workspace/config_geneformer.yaml:ro"
  -v "$REPO/src:/workspace/src:ro"
  -v "$REPO/benchmark:/workspace/benchmark:ro"
  -v "$REPO/benchmark_results:/workspace/benchmark_results"
  --entrypoint python scgpt-toolkit:v0.3-geneformer
  /workspace/benchmark/geneformer_smoke.py --config /workspace/config_geneformer.yaml
  --n-per-type "$NPT" --n-query "$NQ" "${EXTRA[@]}")

if [ "${GO:-0}" != "1" ]; then
  echo "[DRY] host=$(hostname) GPU=${GPU} commit=${GIT_COMMIT} geneformer=${GF_COMMIT}"
  printf '%q ' "${CMD[@]}"; echo
  echo "(실행하려면 GO=1 을 앞에 붙이세요. 로그: $LOGFILE)"
  exit 0
fi
echo "[GO] 시작 $(date +%T) host=$(hostname) GPU=${GPU} commit=${GIT_COMMIT}"
"${CMD[@]}" 2>&1 | tee "$LOGFILE"
RC=${PIPESTATUS[0]}
echo "[GO] 종료 $(date +%T) rc=${RC} 로그=$LOGFILE"
exit $RC
