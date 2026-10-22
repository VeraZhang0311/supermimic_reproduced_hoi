#!/usr/bin/env bash
#SBATCH --job-name=sm-stock-prepare
#SBATCH --partition=liu-compute
#SBATCH --account=liu-account
#SBATCH --qos=ll-med
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
root=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi
old=/mnt/projects/ll/tianzizh/SuperMimic_HOI
uv="$old/tools/bin/uv"
export UV_CACHE_DIR="$root/.cache/uv" TMPDIR="$root/tmp"
export UV_HTTP_TIMEOUT=120 UV_HTTP_RETRIES=5
mkdir -p "$root/tools/python-portable" "$root/envs" "$root/tmp" "$root/provenance" "$root/logs"
exec > >(tee "$root/logs/prepare-${SLURM_JOB_ID}.log") 2>&1
test "$(git -C "$root" rev-parse HEAD)" = a1508d42c9d1699a52e8233ab16d08c488441ab9
test "$(git -C "$root/IsaacLab" rev-parse HEAD)" = 46d6cc3d7b115906fad12efd372fbf69f4e3d3ed
if [ ! -x "$root/tools/python-portable/cpython-3.12.4-linux-x86_64-gnu/bin/python3.12" ]; then
    cp -a "$old/tools/python-portable/cpython-3.12.4-linux-x86_64-gnu" "$root/tools/python-portable/"
fi
python_base="$root/tools/python-portable/cpython-3.12.4-linux-x86_64-gnu/bin/python3.12"
if [ ! -x "$root/envs/kitless-cu128/bin/python" ]; then
    "$uv" venv --python "$python_base" "$root/envs/kitless-cu128"
fi
"$uv" export --project "$root/IsaacLab" --frozen --extra all --extra importers --extra test \
    --no-hashes --no-header --no-annotate --no-emit-project --no-emit-workspace \
    --format requirements-txt --output-file "$root/provenance/author-lock-requirements.txt" >/dev/null
"$python_base" "$root/cluster/prepare_requirements.py"
"$uv" pip install --python "$root/envs/kitless-cu128/bin/python" --no-deps \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match \
    -r "$root/provenance/grasp-cu128-requirements.txt"
for package in "$root"/IsaacLab/source/*; do
    if [ -f "$package/pyproject.toml" ]; then
        "$uv" pip install --python "$root/envs/kitless-cu128/bin/python" --no-deps -e "$package"
    fi
done
"$uv" pip freeze --python "$root/envs/kitless-cu128/bin/python" > "$root/provenance/environment.txt"
"$root/envs/kitless-cu128/bin/python" "$root/cluster/download_data.py"
cd "$root"
"$root/envs/kitless-cu128/bin/python" tools/urdf_to_mjcf.py
"$root/envs/kitless-cu128/bin/python" tools/mjcf_to_usd.py
"$python_base" - <<'PY'
import json, pathlib, datetime
r=pathlib.Path('/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi')
(r/'provenance/prepare-complete.json').write_text(json.dumps({'status':'prepared','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'native_gpu_validated':False},indent=2)+'\n')
PY
