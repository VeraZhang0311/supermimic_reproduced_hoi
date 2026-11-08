#!/bin/bash
#SBATCH --job-name=hoi-tables-eval-monitor
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
#SBATCH --no-requeue
set -euo pipefail
CAMPAIGN=${1:?Specify the tables campaign}
TRAINING_ROOT=${2:?Specify the unchanged active formal run}
SOURCE_ROOT=${3:?Specify separate frozen evaluation source}
TRAINING_SOURCE=${4:?Specify unchanged source_train03}
CONTROL_ROOT=${5:?Specify fresh evaluation watcher control directory}
EVAL_ID=${6:?Specify the independently registered evaluation W&B run}
TRAIN_JOB=${7:?Specify the active training Slurm job}
TELEMETRY=/mnt/projects/ll/tianzizh/gmt_l40_20261105_01/telemetry_venv
RUNTIME=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128
COMPAT=/mnt/projects/ll/tianzizh/hoi_residual_ppo_20261106_01/source_smoke08/tools
export PYTHONPATH="$COMPAT:$TELEMETRY/lib/python3.12/site-packages:$RUNTIME/lib/python3.12/site-packages:$SOURCE_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export WANDB_DISABLE_CODE=true WANDB_CONSOLE=off WANDB_SILENT=true
cd "$SOURCE_ROOT"
exec "$TELEMETRY/bin/python" -m hoi_tables.eval_monitor --campaign "$CAMPAIGN" \
  --training-root "$TRAINING_ROOT" --source-root "$SOURCE_ROOT" --training-source "$TRAINING_SOURCE" \
  --root "$CONTROL_ROOT" --eval-id "$EVAL_ID" --parent-run-id 44ad3b72 --train-job "$TRAIN_JOB" \
  --partition batch --gres gpu:a10:1
