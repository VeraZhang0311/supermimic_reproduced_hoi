#!/usr/bin/env bash
set -euo pipefail
r=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi
test -s "$r/provenance/prepare-complete.json"
"$r/envs/kitless-cu128/bin/python" "$r/cluster/verify_environment.py"
bash -n "$r/cluster/gpu.sbatch.sh"
"$r/tools/python-portable/cpython-3.12.4-linux-x86_64-gnu/bin/python3.12" - <<'PY'
from pathlib import Path
import ast, hashlib, json
r=Path('/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi')
files=list((r/'cluster').glob('*.py'))
for p in files: ast.parse(p.read_text())
identity={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'cluster').iterdir() if p.is_file()}
(r/'provenance/cluster-wrapper-hashes.json').write_text(json.dumps(identity,indent=2)+'\n')
PY
test ! -e "$r/provenance/gpu-job.txt"
/opt/slurm/bin/sbatch --parsable --output="$r/logs/gpu-%j.out" "$r/cluster/gpu.sbatch.sh" > "$r/provenance/gpu-job.txt"
cat "$r/provenance/gpu-job.txt"
