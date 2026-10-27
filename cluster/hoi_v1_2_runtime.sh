#!/usr/bin/env bash
# Immutable v1.2 source; existing shared runtime/assets are read-only.
hoi_v1_2_setup() {
  : "${SUPERMIMIC_ROOT:?Set the new frozen v1.2 source directory}"
  : "${SUPERMIMIC_RUNTIME:?Set the validated read-only runtime}"
  : "${ISAACLAB_PATH:?Set the validated read-only IsaacLab source}"
  : "${EXPECTED_STAGE_SHA256:?Bind the exact staged source and helpers}"
  local root
  root=$(realpath "$SUPERMIMIC_ROOT")
  for old in supermimic_reproduced_hoi supermimic_reproduced_hoi_selfcollision supermimic_reproduced_hoi_handcollision supermimic_hoi_zeroshot supermimic_hoi_zeroshot_handcollision SuperMimic_HOI; do
    test "$root" != "/mnt/projects/ll/tianzizh/$old" || { echo 'Protected historical source checkout' >&2; exit 2; }
  done
  cd "$root"
  test -z "${INIT_CHECKPOINT:-}" && test -z "${RESUME_CHECKPOINT:-}"
  test -z "${WANDB_API_KEY:-}" && test -z "${WANDB_IDENTITY_TOKEN:-}"
  export VIRTUAL_ENV="$SUPERMIMIC_RUNTIME" PATH="$SUPERMIMIC_RUNTIME/bin:$PATH"
  export UV_NO_SYNC=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
  export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export WARP_CACHE_PATH="$root/.cache/${SLURM_JOB_ID}/warp"
  export TORCHINDUCTOR_CACHE_DIR="$root/.cache/${SLURM_JOB_ID}/inductor"
  export MPLCONFIGDIR="$root/.cache/${SLURM_JOB_ID}/mpl"
  source "$root/cluster/hoi_v1_runtime.sh"
  hoi_v1_short_tmpdir
  mkdir -p "$MPLCONFIGDIR"
}
