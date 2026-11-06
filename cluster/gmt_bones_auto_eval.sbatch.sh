#!/bin/bash
#SBATCH --job-name=gmt-bones-auto-eval
#SBATCH --partition=batch
#SBATCH --account=liu-account
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/projects/ll/tianzizh/gmt_bones26k_20261106_02/supervisor-%j.log
set -euo pipefail
ROOT=/mnt/projects/ll/tianzizh/gmt_bones26k_20261106_02
RUN=/mnt/projects/ll/tianzizh/runs/gmt_bones_filtered_from26k_20k_20261106_02
RUNTIME=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128
SDK=/mnt/projects/ll/tianzizh/gmt_l40_20261105_01/telemetry_venv
TRAIN_JOB_ID=$(cat "$ROOT/train_jobid.txt")
case "$TRAIN_JOB_ID" in ''|*[!0-9]*) echo 'Invalid training job ID' >&2; exit 2;; esac
export PYTHONPATH="$SDK/lib/python3.12/site-packages:$RUNTIME/lib/python3.12/site-packages"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export WANDB_DISABLE_CODE=true WANDB_CONSOLE=off WANDB_MODE=online
cd "$ROOT"
"$SDK/bin/python" "$ROOT/source/tools/gmt_auto_eval.py" \
  --run-dir "$RUN" --source-root "$ROOT/source" --control-dir "$ROOT/control" \
  --train-job-id "$TRAIN_JOB_ID" --wandb-entity tianzizh-university-of-pennsylvania \
  --wandb-project supermimic_gmt --wandb-id d1c131df \
  --create-run --run-name gmt_bones_filtered_from26k_20k_20261106_02 \
  --expected-clips 512 --eval-manifest "$ROOT/reference/eval512/manifest.json" \
  --eval-script "$ROOT/eval_worker.sbatch.sh" --total-iterations 20000 --interval 2000
