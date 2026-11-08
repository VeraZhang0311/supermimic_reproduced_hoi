#!/bin/bash
#SBATCH --job-name=hoi-unified-monitor
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
#SBATCH --no-requeue
set -euo pipefail
SOURCE_ROOT=${1:?Specify new standalone logging source}
CONTROL_ROOT=${2:?Specify independent unified logging control}
TRAINING_ROOT=${3:?Specify unchanged active training run}
CAMPAIGN=${4:?Specify its campaign}
EVAL_SOURCE=${5:?Specify unchanged GPU evaluator source}
TRAINING_SOURCE=${6:?Specify unchanged frozen training source}
EVAL_CONTROL=${7:?Specify existing durable evaluation ledger}
TRAIN_JOB=${8:?Specify the training Slurm job}
ASSETS=${9:?Specify exact ordered comma-separated native asset names}
SUPERVISOR_MODULE=${10:-hoi_tables.eval_monitor}
TELEMETRY=/mnt/projects/ll/tianzizh/gmt_l40_20261105_01/telemetry_venv
RUNTIME=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128
COMPAT=/mnt/projects/ll/tianzizh/hoi_residual_ppo_20261106_01/source_smoke08/tools
export PYTHONPATH="$COMPAT:$TELEMETRY/lib/python3.12/site-packages:$RUNTIME/lib/python3.12/site-packages:$SOURCE_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export WANDB_DISABLE_CODE=true WANDB_CONSOLE=off WANDB_SILENT=true
cd "$SOURCE_ROOT"
exec "$TELEMETRY/bin/python" -m hoi_tables.unified_monitor --root "$CONTROL_ROOT" \
  --training-root "$TRAINING_ROOT" --campaign "$CAMPAIGN" --eval-control "$EVAL_CONTROL" \
  --eval-source "$EVAL_SOURCE" --training-source "$TRAINING_SOURCE" --train-job "$TRAIN_JOB" \
  --assets "$ASSETS" --supervisor-module "$SUPERVISOR_MODULE"
