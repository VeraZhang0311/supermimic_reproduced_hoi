#!/bin/bash
#SBATCH --job-name=tables6000-contact-AB
#SBATCH --partition=batch
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -euo pipefail
HELPERS=${1:?immutable experiment helpers}
OUTPUT=${2:?fresh output}
CAMPAIGN=/mnt/projects/ll/tianzizh/hoi_tables_teacher_20261107_01
SOURCE="$CAMPAIGN/source_eval01"
mkdir "$OUTPUT"
finish(){ code=$?; printf '%s\n' "$code" > "$OUTPUT/exitcode"; exit "$code"; }
trap finish EXIT
export VIRTUAL_ENV=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128
export ISAACLAB_PATH=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi/IsaacLab
export PATH="$VIRTUAL_ENV/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="$SOURCE:$HELPERS" SUPERMIMIC_ROOT="$SOURCE"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_NO_SYNC=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTORCH_ALLOC_CONF=expandable_segments:True
export WARP_CACHE_PATH="$OUTPUT/cache/warp" CUDA_CACHE_PATH="$OUTPUT/cache/cuda"
export TORCHINDUCTOR_CACHE_DIR="$OUTPUT/cache/inductor" TMPDIR="$OUTPUT/tmp" MPLCONFIGDIR="$OUTPUT/cache/mpl"
export MUJOCO_GL=egl
mkdir -p "$WARP_CACHE_PATH" "$CUDA_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR" "$MPLCONFIGDIR"
# This job is pinned to A10, matching the verified A10 render cache.
cp -a "$CAMPAIGN/videos_6000_01/cache/warp/." "$WARP_CACHE_PATH/"
cp -a "$CAMPAIGN/videos_6000_01/cache/cuda/." "$CUDA_CACHE_PATH/"
cd "$SOURCE"
sha256sum supermimic/*.py hoi_residual/*.py hoi_tables/*.py hoi_tables/v11/*.py > "$OUTPUT/source.sha256"
sha256sum "$CAMPAIGN/formal_03"/rank_*/checkpoint_006000.pt > "$OUTPUT/checkpoints.sha256"
sha256sum "$CAMPAIGN/protected_eval_2000_01/iter_006000/evaluation/result.json" > "$OUTPUT/original_result.sha256"
sha256sum "$HELPERS"/*.py "$HELPERS/run.sbatch.sh" > "$OUTPUT/helpers.sha256"
scontrol show job -o 1512413 > "$OUTPUT/tables_before.txt"
scontrol show job -o 1516579 > "$OUTPUT/ultra_before.txt"
nvidia-smi > "$OUTPUT/gpu.txt"
python "$HELPERS/verify_hook.py" 2>&1 | tee "$OUTPUT/hook_verification.log"
for variant in A B; do
  python "$HELPERS/rollout.py" --variant "$variant" --audit-root "$OUTPUT/$variant" \
    --checkpoint-root "$CAMPAIGN/formal_03" --training-source "$CAMPAIGN/source_train03" \
    --iteration 6000 --manifest "$CAMPAIGN/input/manifest.json" --asset-bundle "$CAMPAIGN/input/assets.json" \
    --shape-bank "$CAMPAIGN/input/shape_bank.json" --reference-contacts "$CAMPAIGN/formal_inputs01/reference_contacts.pt" \
    --parent-run-id 44ad3b72 --asset 1 --device cuda:0 --output "$OUTPUT/${variant}_environment" \
    2>&1 | tee "$OUTPUT/$variant.log"
done
python "$HELPERS/analyze.py" --root "$OUTPUT" 2>&1 | tee "$OUTPUT/analysis.log"
python "$HELPERS/render.py" --root "$OUTPUT" --campaign "$CAMPAIGN" 2>&1 | tee "$OUTPUT/render.log"
sha256sum --check "$OUTPUT/source.sha256" > "$OUTPUT/source_unchanged.txt"
sha256sum --check "$OUTPUT/checkpoints.sha256" > "$OUTPUT/checkpoints_unchanged.txt"
sha256sum --check "$OUTPUT/original_result.sha256" > "$OUTPUT/original_result_unchanged.txt"
sha256sum --check "$OUTPUT/helpers.sha256" > "$OUTPUT/helpers_unchanged.txt"
scontrol show job -o 1512413 > "$OUTPUT/tables_after.txt"
scontrol show job -o 1516579 > "$OUTPUT/ultra_after.txt"
echo CONTACT_AB_COMPLETE
