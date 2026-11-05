#!/bin/bash
# Launch SuperMimic training in a tmux session inside the 'supermimic' container.
# Usage: [CPUS=0-23] ./run_train.sh <run_name> [train.py args...]
# CPUS optionally pins the run with taskset (rootless docker ignores --cpuset-cpus).
set -euo pipefail
RUN=${1:-lafan1_privileged_$(date -u +%Y%m%dT%H%M%SZ)}
shift || true
SUPERMIMIC_ROOT=${SUPERMIMIC_ROOT:-/workspace/supermimic}
PIN=()
if [ -n "${CPUS:-}" ]; then PIN=(taskset -c "$CPUS"); fi
printf -v ROOT_Q '%q' "$SUPERMIMIC_ROOT"
printf -v LOG_Q '%q' "$SUPERMIMIC_ROOT/logs/$RUN"
printf -v COMMAND '%q ' env "SUPERMIMIC_ROOT=$SUPERMIMIC_ROOT" \
  "PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}" \
  "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  "OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}" "MKL_NUM_THREADS=${MKL_NUM_THREADS:-3}" \
  "${PIN[@]}" python supermimic/train.py --log_dir "$SUPERMIMIC_ROOT/logs/$RUN" "$@"
printf -v SESSION_COMMAND '%q' "cd $ROOT_Q && mkdir -p $LOG_Q && $COMMAND 2>&1 | tee -a $LOG_Q/stdout.log"
docker exec supermimic tmux new-session -d -s "$RUN" "bash -o pipefail -c $SESSION_COMMAND"
