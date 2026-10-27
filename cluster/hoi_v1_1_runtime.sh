#!/usr/bin/env bash
# Independent frozen v1.1 jobs; shared environments/assets remain read-only.
hoi_v1_1_setup() {
  : "${SUPERMIMIC_ROOT:?Set a new frozen source directory}"
  : "${SUPERMIMIC_RUNTIME:?Set the validated read-only runtime}"
  : "${ISAACLAB_PATH:?Set the validated read-only IsaacLab source}"
  : "${EXPECTED_STAGE_SHA256:?Bind the exact staged source and helpers}"
  cd "$(realpath "$SUPERMIMIC_ROOT")"
  test -z "${INIT_CHECKPOINT:-}" && test -z "${RESUME_CHECKPOINT:-}"
  test -z "${WANDB_API_KEY:-}" && test -z "${WANDB_IDENTITY_TOKEN:-}"
  export VIRTUAL_ENV="$SUPERMIMIC_RUNTIME" PATH="$SUPERMIMIC_RUNTIME/bin:$PATH"
  export UV_NO_SYNC=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
  export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export WARP_CACHE_PATH="$SUPERMIMIC_ROOT/.cache/${SLURM_JOB_ID}/warp"
  export TORCHINDUCTOR_CACHE_DIR="$SUPERMIMIC_ROOT/.cache/${SLURM_JOB_ID}/inductor"
  export MPLCONFIGDIR="$SUPERMIMIC_ROOT/.cache/${SLURM_JOB_ID}/mpl"
  source "$SUPERMIMIC_ROOT/cluster/hoi_v1_runtime.sh"
  hoi_v1_short_tmpdir
  mkdir -p "$MPLCONFIGDIR"
}
