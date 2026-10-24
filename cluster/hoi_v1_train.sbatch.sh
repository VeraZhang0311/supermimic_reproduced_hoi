#!/usr/bin/env bash
#SBATCH --job-name=sm-hoi-v1
#SBATCH --partition=liu-compute
#SBATCH --account=liu-account
#SBATCH --qos=ll-med
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${SUPERMIMIC_ROOT:?Set an independent development checkout}"
: "${SUPERMIMIC_RUNTIME:?Set the prepared read-only Linux runtime}"
: "${ISAACLAB_PATH:?Set the prepared read-only IsaacLab source}"
: "${BUNDLE:?Set an immutable train bundle.json}"
: "${NATIVE_EVIDENCE:?Set completed native evidence for this exact bundle}"
: "${EVAL_BUNDLE:?Set the complete canonical HOI80 evaluation bundle}"
: "${RUN_DIR:?Set a new experiment output directory}"
: "${INIT_CHECKPOINT:?Set the frozen motion-only initialization checkpoint}"
r=$(realpath "$SUPERMIMIC_ROOT")
for old in supermimic_reproduced_hoi supermimic_reproduced_hoi_selfcollision supermimic_reproduced_hoi_handcollision supermimic_hoi_zeroshot supermimic_hoi_zeroshot_handcollision SuperMimic_HOI; do
  test "$r" != "/mnt/projects/ll/tianzizh/$old" || { echo "Protected historical checkout: $r" >&2; exit 2; }
done
cd "$r"
test ! -e "$RUN_DIR" || { echo 'Existing results must be preserved; choose a new RUN_DIR' >&2; exit 2; }
export VIRTUAL_ENV="$SUPERMIMIC_RUNTIME" PATH="$SUPERMIMIC_RUNTIME/bin:$PATH"
export UV_NO_SYNC=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
# Eight simulator processes share the allocation with one central learner.
# Match the validated three-thread profile; preserve explicit caller budgets.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export WARP_CACHE_PATH="$r/.cache/hoi-v1/${SLURM_JOB_ID}/warp"
export TORCHINDUCTOR_CACHE_DIR="$r/.cache/hoi-v1/${SLURM_JOB_ID}/inductor"
export MPLCONFIGDIR="$r/.cache/hoi-v1/${SLURM_JOB_ID}/mpl"
source "$r/cluster/hoi_v1_runtime.sh"
hoi_v1_short_tmpdir
mkdir -p "$MPLCONFIGDIR"
# This is a separate physics-change initialization, not continuation of the
# hand-OFF/self-ON historical experiment. No W&B run is resumed here.
python -m hoi.v1.train --config "${CONFIG:-configs/hoi_v1_first.json}" \
  --bundle "$BUNDLE" --native-evidence "$NATIVE_EVIDENCE" --output "$RUN_DIR" \
  --eval-bundle "$EVAL_BUNDLE" \
  --init-checkpoint "$INIT_CHECKPOINT" --no-legacy-hand-collision --legacy-self-collision \
  --allow-initialization-physics-change --hand-collision --self-collision "$@"
# WANDB_SYNC=1 streams curated metrics from the single fresh training run.
# Do not start a second batch tracking run when training exits.
