#!/usr/bin/env bash
#SBATCH --job-name=hoi-superteacher
#SBATCH --no-requeue
# Supply partition/account/qos, GPU count/type, CPUs, RAM and time to sbatch.
set -euo pipefail
: "${SUPERMIMIC_ROOT:?Set the new frozen teacher checkout}"
: "${SUPERMIMIC_RUNTIME:?Set the prepared read-only runtime}"
: "${ISAACLAB_PATH:?Set the read-only IsaacLab source}"
: "${SOURCE_CONTENT_SHA256:?Set the frozen source_identity content SHA256}"
: "${BUNDLE:?Set the complete immutable train bundle}"
: "${EVAL_BUNDLE:?Set the complete protected_eval bundle}"
: "${NATIVE_EVIDENCE:?Set the train partition native evidence}"
: "${RUN_DIR:?Set a new training output directory}"
: "${PROBE_DIR:?Set a new full-resource probe output directory}"
: "${COLLECTOR_DEVICES:?Set space-separated job-local cuda:N collector devices}"
: "${WORKER_STARTUP_CONCURRENCY:?Set the maximum concurrent worker initializations}"
[[ $# == 0 ]] || { echo 'Use declared environment variables; extra CLI overrides are forbidden' >&2; exit 2; }
r=$(realpath "$SUPERMIMIC_ROOT")
for old in supermimic_reproduced_hoi supermimic_reproduced_hoi_selfcollision supermimic_reproduced_hoi_handcollision supermimic_hoi_zeroshot supermimic_hoi_zeroshot_handcollision SuperMimic_HOI; do
  [[ "$r" != "/mnt/projects/ll/tianzizh/$old" ]] || { echo 'Protected historical checkout' >&2; exit 2; }
done
cd "$r"
[[ ! -e "$RUN_DIR" && ! -e "$PROBE_DIR" ]] || { echo 'Choose fresh RUN_DIR and PROBE_DIR; existing outputs are preserved' >&2; exit 2; }
[[ $(realpath -m "$RUN_DIR") != $(realpath -m "$PROBE_DIR") ]] || { echo 'Probe and training outputs must differ' >&2; exit 2; }
[[ "$WORKER_STARTUP_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || exit 2
read -r -a collectors <<< "$COLLECTOR_DEVICES"
for device in "${collectors[@]}"; do [[ "$device" =~ ^cuda:[0-9]+$ ]] || exit 2; done
export VIRTUAL_ENV="$SUPERMIMIC_RUNTIME" PATH="$SUPERMIMIC_RUNTIME/bin:$PATH"
export UV_NO_SYNC=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_ENTITY="${WANDB_ENTITY:-tianzizh-university-of-pennsylvania}"
export WANDB_PROJECT="${WANDB_PROJECT:-superteacher}" WANDB_SYNC="${WANDB_SYNC:-1}"
# The project must already have been created and verified PRIVATE. Never export credentials.
export WARP_CACHE_PATH="$r/.cache/teacher/${SLURM_JOB_ID:-local}/warp"
export TORCHINDUCTOR_CACHE_DIR="$r/.cache/teacher/${SLURM_JOB_ID:-local}/inductor"
export MPLCONFIGDIR="$r/.cache/teacher/${SLURM_JOB_ID:-local}/mpl"
source "$r/cluster/hoi_v1_runtime.sh"
hoi_v1_short_tmpdir
mkdir -p "$MPLCONFIGDIR"
config="$r/configs/hoi_v1_privileged_teacher.json"
learner_device="${LEARNER_DEVICE:-cuda:0}"
common=(--config "$config" --bundle "$BUNDLE" --from-scratch --device "$learner_device"
        --hand-collision --self-collision --collector-devices "${collectors[@]}"
        --worker-startup-concurrency "$WORKER_STARTUP_CONCURRENCY")
if [[ -n "${ROBOT_MJCF:-}" ]]; then common+=(--robot-mjcf "$ROBOT_MJCF"); fi
python -m hoi.v1.teacher_launch --source-sha "$SOURCE_CONTENT_SHA256"
python -m hoi.v1.resource_probe "${common[@]}" --expected-groups 273 --worker-cpu-threads 3 --output "$PROBE_DIR"
python -m hoi.v1.teacher_launch --source-sha "$SOURCE_CONTENT_SHA256" --report "$PROBE_DIR/report.json"   --bundle "$BUNDLE" --device "$learner_device" --collector-devices "${collectors[@]}"   --worker-startup-concurrency "$WORKER_STARTUP_CONCURRENCY"
python -m hoi.v1.train "${common[@]}" --native-evidence "$NATIVE_EVIDENCE" --eval-bundle "$EVAL_BUNDLE"   --evaluation-launcher "${EVALUATION_LAUNCHER:-$r/cluster/hoi_teacher_evaluate.sbatch.sh}"   --evaluation-runtime "$SUPERMIMIC_RUNTIME" --evaluation-isaaclab "$ISAACLAB_PATH" --output "$RUN_DIR"
