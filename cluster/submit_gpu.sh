#!/usr/bin/env bash
set -euo pipefail
r=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision
cd "$r"
test -s provenance/prepare-complete.json
test ! -e provenance/gpu-job.txt
test ! -d logs/lafan1_selfcollision_seed0
queue=$(/opt/slurm/bin/squeue -u "$USER" -h -o '%i|%j|%T|%Z')
printf '%s\n' "$queue" > provenance/queue-before-submit.txt
if printf '%s\n' "$queue" | grep -F "$r"; then
    echo 'An existing job already uses this experiment root; not submitting a duplicate' >&2
    exit 1
fi
bash -n cluster/gpu.sbatch.sh
export PYTHONDONTWRITEBYTECODE=1
envs/kitless-cu128/bin/python cluster/verify_environment.py
envs/kitless-cu128/bin/python cluster/validate_experiment.py
envs/kitless-cu128/bin/python - <<'PY'
from pathlib import Path
import ast,hashlib,json
r=Path.cwd()
for p in (r/'cluster').glob('*.py'): ast.parse(p.read_text())
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'cluster').iterdir() if p.is_file()}
(r/'provenance/cluster-wrapper-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
PY
job=$(/opt/slurm/bin/sbatch --parsable --output="$r/logs/gpu-%j.out" "$r/cluster/gpu.sbatch.sh")
printf '%s\n' "$job" > provenance/gpu-job.txt
envs/kitless-cu128/bin/python - "$job" <<'PY'
from pathlib import Path
import datetime,json,sys
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'slurm_job_id':sys.argv[1],
 'experiment':'selfcollision_on','status':'submitted','physical_self_collisions':True,
 'native_self_collision_verified':False,'formal_training_stable':False,
 'next':'Run OFF/ON effective-filter diagnostics, 64env native/FK, full4096env501 smoke, then fresh14000 formal run.'}
Path('provenance/startup-status.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
PY
