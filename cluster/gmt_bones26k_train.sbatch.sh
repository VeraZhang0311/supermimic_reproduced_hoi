#!/bin/bash
#SBATCH --job-name=gmt-bones26k-20k
#SBATCH --partition=liu-compute
#SBATCH --account=liu-account
#SBATCH --qos=ll-med
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/projects/ll/tianzizh/gmt_bones26k_20261106_02/train-%j.log
set -euo pipefail
ROOT=/mnt/projects/ll/tianzizh/gmt_bones26k_20261106_02
RUN=/mnt/projects/ll/tianzizh/runs/gmt_bones_filtered_from26k_20k_20261106_02
SOURCE="$ROOT/source"
test -f "$ROOT/parent_verified.json" && test -f "$ROOT/reference/full/manifest.json"
test -f "$ROOT/validation_passed.json"
test -f "$ROOT/launch.json" && test -d "$RUN" && test ! -f "$RUN/args.json"
mkdir "$RUN/trainer_claim"
finish() {
  result=$?
  trap - EXIT TERM INT
  printf '%s\n' "$result" > "$RUN/training_exitcode"
  exit "$result"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export VIRTUAL_ENV=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128
export ISAACLAB_PATH=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi/IsaacLab
export PATH="$VIRTUAL_ENV/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="$SOURCE" SUPERMIMIC_ROOT="$SOURCE" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_NO_SYNC=1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WARP_CACHE_PATH="$RUN/cache/warp" TORCHINDUCTOR_CACHE_DIR="$RUN/cache/inductor"
export MPLCONFIGDIR="$RUN/cache/mpl" CUDA_CACHE_PATH="$RUN/cache/cuda"
mkdir -p "$WARP_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
cd "$SOURCE"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$RUN/gpu.txt"
scontrol show job "$SLURM_JOB_ID" > "$RUN/slurm.txt"
sha256sum supermimic/*.py tools/*.py > "$RUN/source.sha256"
# The independent CPU supervisor is the only W&B writer. No SDK in the trainer.
for ((attempt=0; attempt<1200; attempt++)); do
  if [ -f "$RUN/wandb_error.json" ]; then echo 'W&B initialization failed' >&2; exit 1; fi
  if [ -f "$RUN/wandb_ready.json" ]; then break; fi
  sleep 1
done
test -f "$RUN/wandb_ready.json"
python -m supermimic.train --dataset boneseed --motion_dir "$ROOT/reference/full" \
  --num_envs 4096 --buffer_steps 1000 --horizon 32 --amp bf16 \
  --iterations 20000 --wm_only_iters 0 --ckpt_every 2000 \
  --transfer_from "$ROOT/input/parent_026000_full.pt" --transfer_parent_iteration 26000 \
  --log_dir "$RUN" 2>&1 | tee "$RUN/stdout.log"
