#!/usr/bin/env bash
#SBATCH --job-name=sm-stock-lafan1-s0
#SBATCH --partition=liu-compute
#SBATCH --account=liu-account
#SBATCH --qos=ll-med
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
set -euo pipefail
root=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi
export VIRTUAL_ENV="$root/envs/kitless-cu128"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export ISAACLAB_PATH="$root/IsaacLab" UV_NO_SYNC=1
export WARP_CACHE_PATH="$root/.cache/warp" TORCHINDUCTOR_CACHE_DIR="$root/.cache/inductor"
export MPLCONFIGDIR="$root/.cache/mpl" TMPDIR="$root/tmp" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TOKENIZERS_PARALLELISM=false
cd "$root"
test -s "$root/provenance/prepare-complete.json"
git diff --exit-code HEAD -- supermimic tools
test "$(git rev-parse HEAD)" = a1508d42c9d1699a52e8233ab16d08c488441ab9
test "$(git -C IsaacLab rev-parse HEAD)" = 46d6cc3d7b115906fad12efd372fbf69f4e3d3ed
python "$root/cluster/run_pipeline.py"
