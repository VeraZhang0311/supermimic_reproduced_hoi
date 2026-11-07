#!/bin/bash
#SBATCH --job-name=hoi-residual-smoke
#SBATCH --partition=liu-compute
#SBATCH --account=liu-account
#SBATCH --qos=ll-med
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=hoi-residual-smoke-%j.log
set -euo pipefail

# Submit only after SOURCE_ROOT is frozen and CPU checks have completed.
ROOT=${1:?Usage: sbatch script.sh NEW_EXPERIMENT_ROOT FROZEN_SOURCE_ROOT [all|ASSET_ID]}
SOURCE_ROOT=${2:?Specify the complete frozen residual teacher source}
SHAPE_BANK=${HOI_SHAPE_BANK:-$ROOT/input/shape_bank_v2.json}
GMT_CHECKPOINT=${HOI_GMT_CHECKPOINT:-$ROOT/input/ckpt_004000.pt}
GMT_ITERATION=${HOI_GMT_ITERATION:-4000}
SELECTION=${3:-all}
REWARD_ARGS=()
if [[ $# -gt 3 ]]; then
  REWARD_ARGS=(--experiment-config "$4" --v11-train-bundle "${5:?Specify the v1.1 training bundle}")
fi
case "$SELECTION" in
  all) ASSETS=(0 1 2 3 4 5 6 7); STEPS=32;;
  [0-7]) ASSETS=("$SELECTION"); STEPS=24;;
  *) echo 'Asset selection must be all or one integer 0..7' >&2; exit 2;;
esac
test -d "$ROOT" && test -d "$SOURCE_ROOT"
test -f "$SOURCE_ROOT/hoi_residual/train.py"
test -f "$GMT_CHECKPOINT"
test -f "$ROOT/input/ultra_manifest.json"
test -f "$ROOT/input/assets/assets.json"
test -f "$SHAPE_BANK"
RUN="$ROOT/smoke_${SLURM_JOB_ID:?Run GPU workloads through Slurm}"
mkdir "$RUN"
finish() {
  result=$?
  trap - EXIT TERM INT
  printf '%s\n' "$result" > "$RUN/exitcode"
  exit "$result"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

export VIRTUAL_ENV=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128
export ISAACLAB_PATH=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi/IsaacLab
export PATH="$VIRTUAL_ENV/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="$SOURCE_ROOT" SUPERMIMIC_ROOT="$SOURCE_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_NO_SYNC=1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WARP_CACHE_PATH="$RUN/cache/warp" TORCHINDUCTOR_CACHE_DIR="$RUN/cache/inductor"
export MPLCONFIGDIR="$RUN/cache/mpl" CUDA_CACHE_PATH="$RUN/cache/cuda"
export TMPDIR="$RUN/tmp"
mkdir -p "$WARP_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH" "$TMPDIR"
cd "$SOURCE_ROOT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$RUN/gpu.txt"
scontrol show job "$SLURM_JOB_ID" > "$RUN/slurm.txt"
sha256sum supermimic/*.py hoi_residual/*.py > "$RUN/source.sha256"
sha256sum "$GMT_CHECKPOINT" "$ROOT/input/ultra_manifest.json" "$ROOT/input/assets/assets.json" "$SHAPE_BANK" > "$RUN/inputs.sha256"

# Eight independent disposable simulators validate every collision asset. The
# resulting tiny per-asset checkpoints are smoke artifacts, not a trained teacher.
for asset in "${ASSETS[@]}"; do
  python -m hoi_residual.train \
    --manifest "$ROOT/input/ultra_manifest.json" \
    --asset-bundle "$ROOT/input/assets/assets.json" \
    --shape-bank "$SHAPE_BANK" "${REWARD_ARGS[@]}" \
    --gmt-checkpoint "$GMT_CHECKPOINT" --gmt-iteration "$GMT_ITERATION" \
    --output "$RUN/asset_${asset}" --asset "$asset" --smoke \
    --num-envs 8 --rollout-steps "$STEPS" --iterations 2 --epochs 5 --minibatches 4 \
    --save-every 2 --seed 0 --device cuda:0 2>&1 | tee "$RUN/asset_${asset}.log"
done
sha256sum --check "$RUN/source.sha256" > "$RUN/source_unchanged.txt"
sha256sum --check "$RUN/inputs.sha256" > "$RUN/inputs_unchanged.txt"
python - "$RUN" "${ASSETS[@]}" <<'PY'
import json, pathlib, sys
from hoi_residual.features import ORACLE_DIM, observation_contract
root = pathlib.Path(sys.argv[1])
asset_ids = [int(i) for i in sys.argv[2:]]
reports = [json.loads((root / f"asset_{i}/rank_00/report.json").read_text()) for i in asset_ids]
configs = [json.loads((root / f"asset_{i}/rank_00/config.json").read_text()) for i in asset_ids]
assert all(r["passed"] for r in reports)
assert len({r["asset"] for r in reports}) == len(asset_ids)
for report, config in zip(reports, configs):
    assert all(report["shape_encoders_changed"].values())
    assert set(report["shape_encoders_changed"]) == {"actor_shape_encoder", "critic_shape_encoder"}
    assert report["object_history_reset_isolation"] and report["object_history_resets"] > 0
    assert report["observation_contract"] == observation_contract()
    assert config["network"]["oracle_dim"] == ORACLE_DIM == 3647
    assert config["network"]["mlp_input_dim"] == ORACLE_DIM + 29 + 64
    assert config["network"]["shape_encoder"]["input"] == [128, 3]
with (root / "smoke_report.json").open("x") as stream:
    json.dump({"passed": True, "scope": "independent real-physics PPO smoke tests; no convergence claim",
               "asset_count": len(asset_ids), "reports": reports}, stream, indent=2, allow_nan=False)
print(f"PASS: {len(asset_ids)} ULTRA assets completed real-physics residual PPO smoke")
PY
