#!/bin/bash
# Geneformer N=10 반복 실행(G3 1차). 기본은 DRY(목록만 출력), 실행은 GO=1.
#   DRY:  bash ~/scfm-toolkit/scripts/run_geneformer_batch.sh
#   실행: GO=1 bash ~/scfm-toolkit/scripts/run_geneformer_batch.sh
# GPU 가 정확히 1장 보이는 srun 세션에서 실행하고, 끝날 때까지 셸을 열어 둔다.
# 각 실행은 benchmark_results/geneformer_smoke/smoke_<시각>/ 에 결과를 따로 남기므로 중간에 끊겨도 그때까지는 남는다.
# 요약: python3 ~/scfm-toolkit/benchmark/tools/summarize_smoke.py
SMOKE=~/scfm-toolkit/scripts/run_geneformer_smoke.sh
export NPT=${NPT:-10}      # 유형당 reference 세포 수
export NQ=${NQ:-100000}    # query 는 전부 사용(표본이 더 많으면 있는 만큼)
export EPOCHS=${EPOCHS:-10}
# "lr seed" 목록. 앞의 두 줄은 같은 설정 반복(같은 GPU 종류에서 비트 단위로 같은지 확인).
RUNS=(
  "5e-5 42" "5e-5 42"
  "1e-5 42" "1e-4 42"
  "5e-5 43" "1e-5 43" "1e-4 43"
  "5e-5 44" "1e-5 44" "1e-4 44"
)
i=0
for r in "${RUNS[@]}"; do
  i=$((i+1)); set -- $r
  echo "######## [$i/${#RUNS[@]}] lr=$1 seed=$2 epochs=$EPOCHS N=$NPT ########"
  LR=$1 SEED=$2 bash "$SMOKE"
  RC=$?
  if [ "${GO:-0}" = "1" ] && [ $RC -ne 0 ]; then
    if [ $i -le 2 ]; then echo "처음 두 실행 중 실패(rc=$RC): 같은 원인으로 반복 실패할 수 있어 배치를 중단합니다."; exit $RC; fi
    echo "실행 $i 실패(rc=$RC), 다음으로 진행"
  fi
done
echo "######## 배치 종료 $(date +%T) ########"
