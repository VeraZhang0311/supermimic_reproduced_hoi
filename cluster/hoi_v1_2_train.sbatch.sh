#!/usr/bin/env bash
#SBATCH --job-name=sm-hoi-v12-scratch20k
#SBATCH --partition=liu-compute
#SBATCH --account=liu-account
#SBATCH --qos=ll-med
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${RUN_DIR:?Set the exact fresh training output recorded in the stage manifest}"
: "${EXPECTED_COMMIT_RECEIPT_SHA256:?Bind the post-validation commit/push receipt}"
source "$SUPERMIMIC_ROOT/cluster/hoi_v1_2_runtime.sh"
hoi_v1_2_setup
test "${WANDB_SYNC:-}" = 1
test "${WANDB_ENTITY:-}" = tianzizh-university-of-pennsylvania
test "${WANDB_PROJECT:-}" = SuperMimic_HOI_v1_20261023
test -z "${WANDB_RUN_ID:-}" && test -z "${WANDB_RESUME:-}"
export WANDB_MODE=online
python tools/hoi_v1_2_launch_train.py
