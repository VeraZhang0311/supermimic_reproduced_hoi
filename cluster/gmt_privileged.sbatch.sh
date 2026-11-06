#!/bin/bash
# Submit explicitly; this file is never invoked by the validation CLI.
# Example: sbatch --export=ALL,SUPERMIMIC_ROOT=/absolute/frozen/source,GMT_RUN_DIR=/absolute/new/run cluster/gmt_privileged.sbatch.sh
#SBATCH --job-name=gmt-hw3-privileged
#SBATCH --partition=liu-compute
#SBATCH --account=liu-account
#SBATCH --qos=ll-med
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=gmt-hw3-%j.log
set -euo pipefail
: "${SUPERMIMIC_ROOT:?Set SUPERMIMIC_ROOT to the frozen boneseed/privileged source}"
: "${GMT_RUN_DIR:?Set GMT_RUN_DIR to an unused absolute output directory}"
case "$SUPERMIMIC_ROOT:$GMT_RUN_DIR" in /*:/*) ;; *) echo 'Both paths must be absolute' >&2; exit 2;; esac
if [ -e "$GMT_RUN_DIR" ]; then echo "Run path already exists: $GMT_RUN_DIR" >&2; exit 2; fi
export VIRTUAL_ENV=${GMT_RUNTIME:-/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128}
export ISAACLAB_PATH=${ISAACLAB_PATH:-/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi/IsaacLab}
export PATH="$VIRTUAL_ENV/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="$SUPERMIMIC_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_NO_SYNC=1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$GMT_RUN_DIR"
export WARP_CACHE_PATH="$GMT_RUN_DIR/cache/warp" TORCHINDUCTOR_CACHE_DIR="$GMT_RUN_DIR/cache/inductor"
export MPLCONFIGDIR="$GMT_RUN_DIR/cache/mpl" CUDA_CACHE_PATH="$GMT_RUN_DIR/cache/cuda"
mkdir -p "$WARP_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
cd "$SUPERMIMIC_ROOT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$GMT_RUN_DIR/gpu.txt"
scontrol show job "$SLURM_JOB_ID" > "$GMT_RUN_DIR/slurm.txt"
sha256sum supermimic/*.py > "$GMT_RUN_DIR/source.sha256"
export WANDB_PROJECT=${WANDB_PROJECT:-supermimic_gmt}
export WANDB_ENTITY=${WANDB_ENTITY:-tianzizh-university-of-pennsylvania}
export WANDB_DISABLE_CODE=true WANDB_CONSOLE=off WANDB_MODE=online
WANDB_RUN_ID=${WANDB_RUN_ID:-$(python -c 'import uuid; print(uuid.uuid4().hex[:8])')}
SDK_PYTHON=${GMT_WANDB_PYTHON:-$SUPERMIMIC_ROOT/../telemetry_venv/bin/python}
SDK_PATH=${GMT_WANDB_PATH:-$SUPERMIMIC_ROOT/../telemetry_venv/lib/python3.12/site-packages:$VIRTUAL_ENV/lib/python3.12/site-packages}
test -x "$SDK_PYTHON"
# The SDK/dependency path is scoped to this process, never to the trainer.
env PYTHONPATH="$SDK_PATH" "$SDK_PYTHON" "$SUPERMIMIC_ROOT/tools/wandb_gmt_monitor.py" \
  --run-dir "$GMT_RUN_DIR" --entity "$WANDB_ENTITY" --project "$WANDB_PROJECT" \
  --run-id "$WANDB_RUN_ID" --run-name "$(basename "$GMT_RUN_DIR")" \
  > "$GMT_RUN_DIR/wandb_stdout.log" 2>&1 &
MONITOR_PID=$!
finish() {
  result=$?
  trap - EXIT TERM INT
  printf '%s\n' "$result" > "$GMT_RUN_DIR/training_exitcode"
  wait "$MONITOR_PID" || true
  exit "$result"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
for ((attempt=0; attempt<120; attempt++)); do
  if [ -f "$GMT_RUN_DIR/wandb_error.json" ]; then echo 'W&B initialization failed' >&2; exit 1; fi
  if [ -f "$GMT_RUN_DIR/wandb_ready.json" ]; then break; fi
  if ! kill -0 "$MONITOR_PID" 2>/dev/null; then echo 'W&B monitor exited before handshake' >&2; exit 1; fi
  sleep 1
done
test -f "$GMT_RUN_DIR/wandb_ready.json"
test ! -f "$GMT_RUN_DIR/wandb_error.json"
echo "[gmt] W&B ready: $WANDB_ENTITY/$WANDB_PROJECT/$WANDB_RUN_ID"
set +e
python -m supermimic.train --dataset lafan1 \
  --motion_dir "${GMT_MOTION_DIR:-/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi/data/lafan1_g1}" \
  --num_envs 4096 --buffer_steps 1000 --horizon 32 --amp bf16 --iterations 30000 \
  --log_dir "$GMT_RUN_DIR" "$@" 2>&1 | tee "$GMT_RUN_DIR/stdout.log"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e
exit "$TRAIN_EXIT"
