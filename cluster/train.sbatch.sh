#!/usr/bin/env bash
#SBATCH --job-name=sm-development
#SBATCH --partition=liu-compute
#SBATCH --account=liu-account
#SBATCH --qos=ll-med
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${SUPERMIMIC_ROOT:?Set a separate development checkout, not a frozen experiment}"
: "${RUN_DIR:?Set a fresh run directory}"
r=$(realpath "$SUPERMIMIC_ROOT")
for old in supermimic_reproduced_hoi supermimic_reproduced_hoi_selfcollision supermimic_hoi_zeroshot supermimic_hoi_zeroshot_handcollision SuperMimic_HOI; do
  test "$r" != "/mnt/projects/ll/tianzizh/$old" || { echo "Protected historical checkout: $r" >&2; exit 2; }
done
cd "$r"
test ! -e "$RUN_DIR" || { echo 'Existing results must be preserved' >&2; exit 2; }
export VIRTUAL_ENV="$r/envs/kitless-cu128" PATH="$r/envs/kitless-cu128/bin:$PATH"
export ISAACLAB_PATH="$r/IsaacLab" UV_NO_SYNC=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export WARP_CACHE_PATH="$r/.cache/warp" TORCHINDUCTOR_CACHE_DIR="$r/.cache/inductor"
export MPLCONFIGDIR="$r/.cache/mpl" TMPDIR="$r/tmp"
mkdir -p "$r/tmp" "$r/.cache/mpl"
case "${HAND_COLLISION:-1}" in 1) hand=--hand-collision;; 0) hand=--no-hand-collision;; *) exit 2;; esac
case "${SELF_COLLISION:-1}" in 1) self=--self-collision;; 0) self=--no-self-collision;; *) exit 2;; esac
python cluster/compat_launch.py supermimic/train.py --log_dir "$RUN_DIR" "$hand" "$self" "$@"
