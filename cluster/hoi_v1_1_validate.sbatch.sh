#!/usr/bin/env bash
#SBATCH --job-name=sm-hoi-v11-validate
#SBATCH --partition=liu-compute
#SBATCH --account=liu-account
#SBATCH --qos=ll-med
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
source "$SUPERMIMIC_ROOT/cluster/hoi_v1_1_runtime.sh"
hoi_v1_1_setup
export WANDB_SYNC=0
python -m cluster.hoi_v1_1.validate
