#!/usr/bin/env bash
#SBATCH --job-name=sm-hoi-v1-data
#SBATCH --partition=liu-compute
#SBATCH --account=liu-account
#SBATCH --qos=ll-med
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
set -euo pipefail
: "${SUPERMIMIC_ROOT:?Set the independent development checkout}"
: "${SUPERMIMIC_RUNTIME:?Set the prepared read-only Python runtime}"
: "${MANIFEST:?Set the separately prepared canonical manifest}"
: "${OUTPUT:?Set a fresh data-validation output directory}"
root=$(realpath "$SUPERMIMIC_ROOT")
for old in supermimic_reproduced_hoi supermimic_reproduced_hoi_selfcollision supermimic_reproduced_hoi_handcollision supermimic_hoi_zeroshot supermimic_hoi_zeroshot_handcollision SuperMimic_HOI; do
  test "$root" != "/mnt/projects/ll/tianzizh/$old" || { echo "Protected historical checkout: $root" >&2; exit 2; }
done
cd "$root"
test ! -e "$OUTPUT" || { echo 'Existing validation results must be preserved' >&2; exit 2; }
export VIRTUAL_ENV="$SUPERMIMIC_RUNTIME" PATH="$SUPERMIMIC_RUNTIME/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_NO_SYNC=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m hoi.data.prepare validate --manifest "$MANIFEST" --partition train --output "$OUTPUT"
