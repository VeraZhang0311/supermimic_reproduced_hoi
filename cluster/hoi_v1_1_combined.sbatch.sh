#!/usr/bin/env bash
#SBATCH --job-name=sm-hoi-v11-rms-pipeline
#SBATCH --partition=liu-compute
#SBATCH --account=liu-account
#SBATCH --qos=ll-med
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --no-requeue
set -euo pipefail
source "$SUPERMIMIC_ROOT/cluster/hoi_v1_1_runtime.sh"
hoi_v1_1_setup
: "${OPERATIONS_ROOT:?Set the immutable combined-job operations directory}"
: "${EXPECTED_OPERATIONS_SHA256:?Bind the combined-job manifest}"
export WANDB_SYNC=0
python -m cluster.hoi_v1_1.pipeline before
# Separate Python processes release all probe models and CUDA contexts before
# fresh training. A failed/timed-out validation exits the entire allocation.
timeout --signal=TERM --kill-after=30s 7200 python -m cluster.hoi_v1_1.validate
python -m cluster.hoi_v1_1.pipeline after-validation
export WANDB_SYNC=1
exec python -m cluster.hoi_v1_1.train
