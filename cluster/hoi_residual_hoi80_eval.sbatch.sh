#!/bin/bash
#SBATCH --job-name=hoi80-eval
#SBATCH --partition=liu-compute
#SBATCH --account=liu-account
#SBATCH --qos=ll-med
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
ROOT=${1:?experiment root}
CODE=${2:?evaluation code directory}
REQUEST=${3:?dispatch receipt}
SOURCE=${HOI_TRAINING_SOURCE:-$ROOT/source_smoke08}
PROJECT=/mnt/projects/ll/tianzizh
HELPERS="$PROJECT/gmt26k_hoi80_zeroshot_20261106_03/source"
CANONICAL="$PROJECT/supermimic_hoi_zeroshot/results/hoi80_inputs/manifest.json"
ATTEMPT="$(dirname "$REQUEST")/job_${SLURM_JOB_ID:?scheduler required}"
mkdir "$ATTEMPT"
export VIRTUAL_ENV="$PROJECT/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128"
export ISAACLAB_PATH="$PROJECT/supermimic_reproduced_hoi/IsaacLab"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export SUPERMIMIC_ROOT="$SOURCE" PYTHONPATH="$CODE:$SOURCE:$HELPERS"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_NO_SYNC=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
export WARP_CACHE_PATH="$ATTEMPT/cache/warp" CUDA_CACHE_PATH="$ATTEMPT/cache/cuda"
export MPLCONFIGDIR="$ATTEMPT/cache/mpl" TORCHINDUCTOR_CACHE_DIR="$ATTEMPT/cache/inductor"
export TMPDIR="$ATTEMPT/tmp"
mkdir -p "$WARP_CACHE_PATH" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR"
cd "$CODE"
mapfile -t FIELDS < <(python -c 'import sys,json; r=json.load(open(sys.argv[1])); print(r["checkpoint"],r["sha256"],r["iteration"],sep="\n")' "$REQUEST")
[[ ${#FIELDS[@]} -eq 3 ]]
exec python -m hoi_residual.evaluate_hoi80 --checkpoint "${FIELDS[0]}" --checkpoint-sha256 "${FIELDS[1]}" \
    --iteration "${FIELDS[2]}" --gmt-checkpoint "$ROOT/input/ckpt_004000.pt" \
    --manifest "$ROOT/input/ultra_manifest.json" --canonical "$CANONICAL" \
    --asset-bundle "$ROOT/input/assets/assets.json" --shape-bank "$ROOT/input/shape_bank_v2.json" \
    --output "$ATTEMPT/results" --device cuda:0
