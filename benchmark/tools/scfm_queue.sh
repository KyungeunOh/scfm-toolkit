#!/bin/bash
CFG=$1; shift
for g in "$@"; do FG=1 bash ~/scfm_run_grid.sh "$g" "$CFG"; done
echo "[QUEUE] 전체 끝 $(date +%T)"
