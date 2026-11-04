#!/usr/bin/env bash
#SBATCH --partition=liu-compute
#SBATCH --account=liu-account
#SBATCH --qos=ll-med
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
: "${SUPERMIMIC_ROOT:?frozen source required}"
: "${SUPERMIMIC_RUNTIME:?read-only runtime required}"
: "${CAMPAIGN_ROOT:?fresh campaign directory required}"
: "${INITIAL_CHECKPOINT:?verified initial checkpoint required}"
: "${INITIAL_CHECKPOINT_SHA256:?initial checkpoint hash required}"
[[ "${SLURM_JOB_PARTITION:-}" == liu-compute ]]
[[ -d "$CAMPAIGN_ROOT" ]]
export VIRTUAL_ENV="$SUPERMIMIC_RUNTIME" PATH="$SUPERMIMIC_RUNTIME/bin:$PATH"
export PYTHONPATH="$SUPERMIMIC_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
export WANDB_ENTITY=tianzizh-university-of-pennsylvania WANDB_PROJECT=superteacher WANDB_SYNC=1
# flock remains held throughout exec/children. An accidental duplicate producer
# exits instead of collecting or mutating an existing attempt.
exec flock --nonblock --conflict-exit-code 75 "$CAMPAIGN_ROOT/producer.lock" \
  "$SUPERMIMIC_RUNTIME/bin/python" "$SUPERMIMIC_ROOT/cluster/compact_teacher_requeue.py"
