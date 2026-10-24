#!/usr/bin/env bash
#SBATCH --job-name=sm-hoi-v1-prepare
#SBATCH --partition=liu-compute
#SBATCH --account=liu-account
#SBATCH --qos=ll-med
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${SUPERMIMIC_ROOT:?Set an independent frozen development checkout}"
: "${SUPERMIMIC_RUNTIME:?Set the prepared read-only CPU Python runtime}"
: "${MANIFEST:?Set the frozen data manifest}"
: "${ASSET_RECIPE:?Set the master-hand full object recipe}"
: "${OUTPUT:?Set a fresh bundle output directory}"
root=$(realpath "$SUPERMIMIC_ROOT")
for old in supermimic_reproduced_hoi supermimic_reproduced_hoi_selfcollision supermimic_reproduced_hoi_handcollision supermimic_hoi_zeroshot supermimic_hoi_zeroshot_handcollision SuperMimic_HOI; do
  test "$root" != "/mnt/projects/ll/tianzizh/$old" || { echo "Protected historical checkout: $root" >&2; exit 2; }
done
cd "$root"
test ! -e "$OUTPUT" || { echo 'Existing preparation results must be preserved' >&2; exit 2; }
export VIRTUAL_ENV="$SUPERMIMIC_RUNTIME" PATH="$SUPERMIMIC_RUNTIME/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_NO_SYNC=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
source "$root/cluster/hoi_v1_runtime.sh"
hoi_v1_short_tmpdir
jobs=${PREP_JOBS:-${SLURM_CPUS_PER_TASK:-1}}
args=(--manifest "$MANIFEST" --assets "$ASSET_RECIPE" --output "$OUTPUT" --partition "${PARTITION:-train}" --jobs "$jobs")
if [[ -n "${ROBOT_MJCF:-}" ]]; then args+=(--robot-mjcf "$ROBOT_MJCF"); fi
if [[ -n "${LIMIT:-}" ]]; then args+=(--limit "$LIMIT"); fi
if [[ "${RESOURCE_FIXTURE:-0}" == 1 ]]; then args+=(--one-per-asset-resource-fixture); fi
/usr/bin/time -v python -m hoi.v1.prepare "${args[@]}"
