#!/usr/bin/env bash
#SBATCH --job-name=hoi-native-full
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${SUPERMIMIC_ROOT:?frozen source checkout required}"
: "${SUPERMIMIC_RUNTIME:?existing read-only native runtime required}"
: "${ISAACLAB_PATH:?existing read-only IsaacLab source required}"
: "${NATIVE_SCHEDULE:?fresh hash-bound schedule required}"
: "${SLURM_ARRAY_TASK_ID:?submit with schedule array_argument, capped at %4}"
: "${NATIVE_CACHE_ROOT:?task-owned compiled-code cache root required}"
export VIRTUAL_ENV="$SUPERMIMIC_RUNTIME" PATH="$SUPERMIMIC_RUNTIME/bin:$PATH"
export PYTHONPATH="$SUPERMIMIC_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
# Select only CUDA >= 12.1 driver nodes when submitting; native preflight also
# rejects old drivers. No runtime installation or shared cache changes occur.
export XDG_CACHE_HOME="$NATIVE_CACHE_ROOT"
export WARP_CACHE_PATH="$NATIVE_CACHE_ROOT/warp"
export TORCHINDUCTOR_CACHE_DIR="$NATIVE_CACHE_ROOT/inductor" MPLCONFIGDIR="$NATIVE_CACHE_ROOT/mpl"
export UV_NO_SYNC=1
source "$SUPERMIMIC_ROOT/cluster/hoi_v1_runtime.sh"
hoi_v1_short_tmpdir
mkdir -p "$MPLCONFIGDIR"
cd "$SUPERMIMIC_ROOT"
exec "$SUPERMIMIC_RUNTIME/bin/python" -m hoi.v1.native_batch run-slot --schedule "$NATIVE_SCHEDULE" --slot "$SLURM_ARRAY_TASK_ID"
