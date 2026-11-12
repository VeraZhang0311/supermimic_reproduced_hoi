#!/bin/bash
#SBATCH --job-name=hoi-ultra-hoi80-eval
#SBATCH --partition=batch
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
REQUEST=${1:?Specify the immutable eight-checkpoint HOI80 request}
SOURCE_ROOT=${2:?Specify the frozen direct-teacher evaluation source}
export VIRTUAL_ENV=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128
export ISAACLAB_PATH=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi/IsaacLab
export PATH="$VIRTUAL_ENV/bin:/usr/local/bin:/usr/bin:/bin"
export SUPERMIMIC_ROOT="$SOURCE_ROOT" PYTHONPATH="$SOURCE_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_NO_SYNC=1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 WANDB_MODE=disabled
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$SOURCE_ROOT"
python -m hoi_tables.hoi80_monitor --validate-request "$REQUEST"
mapfile -t INPUTS < <(python - "$REQUEST" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
for key in ('output','training_root','training_source','iteration','manifest','asset_bundle','shape_bank',
            'reference_contacts','parent_run_id','geometry_bundle','canonical','helper_root'):
    print(r[key])
PY
)
OUTPUT=${INPUTS[0]}
mkdir "$OUTPUT"
eval_tmp=""
finish() {
  code=$?
  if [[ -n "$eval_tmp" ]]; then rm -rf -- "$eval_tmp" || true; fi
  printf '%s\n' "$code" > "$OUTPUT/worker_exitcode"
  exit "$code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export PYTHONPATH="$SOURCE_ROOT:${INPUTS[11]}"
export WARP_CACHE_PATH="$OUTPUT/cache/warp" TORCHINDUCTOR_CACHE_DIR="$OUTPUT/cache/inductor"
export MPLCONFIGDIR="$OUTPUT/cache/mpl" CUDA_CACHE_PATH="$OUTPUT/cache/cuda"
# NVRTC can reject the deeply nested evaluation output path as a temp directory.
eval_tmp=$(mktemp -d /tmp/hoi80.XXXXXXXX)
export TMPDIR="$eval_tmp" TMP="$eval_tmp" TEMP="$eval_tmp"
mkdir -p "$WARP_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUTPUT/gpu.txt"
scontrol show job "${SLURM_JOB_ID:?HOI80 GPU evaluation requires Slurm}" > "$OUTPUT/slurm.txt"
sha256sum supermimic/*.py hoi_residual/*.py hoi_tables/*.py hoi_tables/v11/*.py > "$OUTPUT/source.sha256"
for asset in {0..7}; do
  python -m hoi_tables.evaluate_hoi80 --checkpoint-root "${INPUTS[1]}" --training-source "${INPUTS[2]}" \
    --iteration "${INPUTS[3]}" --manifest "${INPUTS[4]}" --asset-bundle "${INPUTS[5]}" \
    --shape-bank "${INPUTS[6]}" --reference-contacts "${INPUTS[7]}" --parent-run-id "${INPUTS[8]}" \
    --geometry-bundle "${INPUTS[9]}" --canonical "${INPUTS[10]}" --helper-root "${INPUTS[11]}" \
    --asset "$asset" --device cuda:0 --output "$OUTPUT/asset_0$asset" \
    2>&1 | tee "$OUTPUT/asset_0$asset.log"
done
python -m hoi_tables.evaluate_hoi80 --aggregate "$OUTPUT" --output "$OUTPUT/result.json"
python -m hoi_tables.hoi80_monitor --validate-request "$REQUEST"
sha256sum --check "$OUTPUT/source.sha256" > "$OUTPUT/source_unchanged.txt"
