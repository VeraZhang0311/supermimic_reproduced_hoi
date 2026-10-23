#!/usr/bin/env bash
#SBATCH --job-name=sm-hoi80
#SBATCH --partition=liu-compute
#SBATCH --account=liu-account
#SBATCH --qos=ll-med
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${SUPERMIMIC_ROOT:?Set a separate development checkout}"
: "${HOI_PREPARED:?Set prepared HOI80 input path}"
: "${CHECKPOINT:?Set frozen checkpoint path}"
: "${OUTPUT:?Set a fresh result directory}"
r=$(realpath "$SUPERMIMIC_ROOT")
for old in supermimic_reproduced_hoi supermimic_reproduced_hoi_selfcollision supermimic_hoi_zeroshot supermimic_hoi_zeroshot_handcollision SuperMimic_HOI; do
  test "$r" != "/mnt/projects/ll/tianzizh/$old" || { echo "Protected historical checkout: $r" >&2; exit 2; }
done
cd "$r"
test ! -e "$OUTPUT" || { echo 'Existing results must be preserved' >&2; exit 2; }
export VIRTUAL_ENV="$r/envs/kitless-cu128" PATH="$r/envs/kitless-cu128/bin:$PATH"
export ISAACLAB_PATH="$r/IsaacLab" UV_NO_SYNC=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WARP_CACHE_PATH="$r/.cache/warp" TORCHINDUCTOR_CACHE_DIR="$r/.cache/inductor"
export MPLCONFIGDIR="$r/.cache/mpl" TMPDIR="$r/tmp"
mkdir -p "$r/tmp" "$r/.cache/mpl"
case "${HAND_COLLISION:-1}" in 1) hand=--hand-collision;; 0) hand=--no-hand-collision;; *) exit 2;; esac
case "${SELF_COLLISION:-1}" in 1) self=--self-collision;; 0) self=--no-self-collision;; *) exit 2;; esac
python -m hoi.run --prepared "$HOI_PREPARED" --checkpoint "$CHECKPOINT" --output "$OUTPUT" "$hand" "$self" "$@"
if [ "${TRACK_METRICS:-1}" = 1 ]; then
  TRACKING_PYTHON=${TRACKING_PYTHON:-$r/tools/python-portable/cpython-3.12.4-linux-x86_64-gnu/bin/python3.12}
  TRACKING_SITE=${TRACKING_SITE:-/mnt/projects/ll/tianzizh/SuperMimic_HOI/envs/tracking-py312/lib/python3.12/site-packages}
  run_id=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["wandb_run_id"])' "$OUTPUT/evaluation.json")
  PYTHONPATH="$r:$TRACKING_SITE" "$TRACKING_PYTHON" -m hoi.track "$OUTPUT/evaluation.json" --run-id "$run_id"
fi
