#!/usr/bin/env bash
#SBATCH --job-name=sm-hoi-v1-native
#SBATCH --partition=liu-compute
#SBATCH --account=liu-account
#SBATCH --qos=ll-med
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
set -euo pipefail
: "${SUPERMIMIC_ROOT:?Set an independent development checkout}"
: "${SUPERMIMIC_RUNTIME:?Set the prepared read-only Linux runtime}"
: "${ISAACLAB_PATH:?Set the prepared read-only IsaacLab source}"
: "${OUTPUT:?Set a fresh native-validation output directory}"
if [[ -n "${LEGACY_PREPARED_FIXTURE:-}" ]]; then
  input=(--legacy-prepared-fixture "$LEGACY_PREPARED_FIXTURE")
else
  : "${BUNDLE:?Set the independently prepared HOI v1 bundle or legacy read-only fixture}"
  input=(--bundle "$BUNDLE")
fi
r=$(realpath "$SUPERMIMIC_ROOT")
for old in supermimic_reproduced_hoi supermimic_reproduced_hoi_selfcollision supermimic_reproduced_hoi_handcollision supermimic_hoi_zeroshot supermimic_hoi_zeroshot_handcollision SuperMimic_HOI; do
  test "$r" != "/mnt/projects/ll/tianzizh/$old" || { echo "Protected historical checkout: $r" >&2; exit 2; }
done
cd "$r"
test ! -e "$OUTPUT" || { echo 'Existing native validation results must be preserved' >&2; exit 2; }
export VIRTUAL_ENV="$SUPERMIMIC_RUNTIME" PATH="$SUPERMIMIC_RUNTIME/bin:$PATH"
export UV_NO_SYNC=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
cache="${NATIVE_CACHE_ROOT:-$r/.cache/hoi-v1-native/${SLURM_JOB_ID}}"
export WARP_CACHE_PATH="$cache/warp"
export TORCHINDUCTOR_CACHE_DIR="$cache/inductor"
export MPLCONFIGDIR="$cache/mpl"
source "$r/cluster/hoi_v1_runtime.sh"
hoi_v1_short_tmpdir
mkdir -p "$MPLCONFIGDIR"
python -m hoi.v1.native "${input[@]}" --asset-id "${ASSET_ID:-0}" --num-envs "${NUM_ENVS:-8}" \
  --steps "${STEPS:-64}" --output "$OUTPUT" --hand-collision --self-collision "$@"
