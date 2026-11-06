#!/bin/bash
#SBATCH --job-name=gmt-bones-validate
#SBATCH --partition=batch
#SBATCH --account=liu-account
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/projects/ll/tianzizh/gmt_bones26k_20261106_02/validation-%j.log
set -euo pipefail
ROOT=/mnt/projects/ll/tianzizh/gmt_bones26k_20261106_02
RUNTIME=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128
export PYTHONPATH="$ROOT/source:$ROOT/source/tests" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export VIRTUAL_ENV="$RUNTIME" ISAACLAB_PATH=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi/IsaacLab
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
cd "$ROOT/source"
"$RUNTIME/bin/python" -m unittest test_gmt_transfer test_gmt_auto_eval test_gmt_eval_worker \
  test_prepare_bones_filtered test_wandb_gmt_monitor -v
"$RUNTIME/bin/python" "$ROOT/args_probe.py"
"$RUNTIME/bin/python" "$ROOT/validate_data.py"
