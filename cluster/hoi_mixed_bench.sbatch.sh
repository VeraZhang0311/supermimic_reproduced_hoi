#!/bin/bash
#SBATCH --job-name=hoi-mix-bench
#SBATCH --partition=liu-compute
#SBATCH --account=liu-account
#SBATCH --qos=ll-med
#SBATCH --gres=gpu:l40:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --no-requeue
set -euo pipefail
ROOT=${1:?Specify the prepared mixed campaign}
SOURCE=${2:?Specify a frozen source snapshot}
OUTPUT=${3:?Specify a fresh output directory}
ITERATIONS=${4:-10}
shift "$(( $# < 4 ? $# : 4 ))"
RUNTIME=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128
export VIRTUAL_ENV="$RUNTIME" PATH="$RUNTIME/bin:$PATH"
export ISAACLAB_PATH=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi/IsaacLab
export PYTHONPATH="$SOURCE" SUPERMIMIC_ROOT="$SOURCE" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export WANDB_MODE=disabled OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WARP_CACHE_PATH="$ROOT/cache/warp" TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/inductor"
export TMPDIR=$(mktemp -d /tmp/hoi-mix.XXXXXXXX)
trap 'rmdir "$TMPDIR" 2>/dev/null || true' EXIT
mkdir -p "$WARP_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR"
test -f "$ROOT/tables/prepared"
cd "$SOURCE"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv
nvidia-smi --query-gpu=timestamp,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 200 > "$OUTPUT.nvml.csv" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true; rmdir "$TMPDIR" 2>/dev/null || true' EXIT
python -m hoi_tables.mixed_train --campaign "$ROOT" --output "$OUTPUT" --benchmark "$ITERATIONS" "$@"
