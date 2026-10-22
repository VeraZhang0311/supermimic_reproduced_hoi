#!/usr/bin/env bash
set -euo pipefail
r=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi
live=$(/opt/slurm/bin/squeue -u tianzizh --name=sm-stock-lafan1-s0,sm-stock-graphfix -h -o '%i')
if [ -n "$live" ]; then
    printf 'Existing live producer(s): %s\n' "$live"
    exit 3
fi
test ! -d "$r/logs/lafan1_stock_seed0"
mkdir -p "$r/logs/failed-1454300"
for p in "$r/logs/native-fk-gate.log" "$r/logs/gpu-1454300.out" "$r/provenance/gpu-hardware.json"; do
    if [ -f "$p" ]; then cp -n "$p" "$r/logs/failed-1454300/"; fi
done
/opt/slurm/bin/sacct -X -j 1454300 --format=JobIDRaw,State,Start,End,Elapsed,ExitCode,NodeList -P > "$r/logs/failed-1454300/accounting.txt"
cp -n "$r/provenance/gpu-job.txt" "$r/provenance/gpu-job-initial-1454300.txt"
bash -n "$r/cluster/gpu.sbatch.sh"
"$r/tools/python-portable/cpython-3.12.4-linux-x86_64-gnu/bin/python3.12" - <<'PY'
import ast,hashlib,json
from pathlib import Path
r=Path('/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi')
for p in (r/'cluster').glob('*.py'): ast.parse(p.read_text())
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'cluster').iterdir() if p.is_file()}
(r/'provenance/cluster-wrapper-hashes-graphfix.json').write_text(json.dumps(files,indent=2)+'\n')
PY
git -C "$r" diff --exit-code HEAD -- supermimic tools
/opt/slurm/bin/sbatch --parsable --job-name=sm-stock-graphfix --output="$r/logs/gpu-%j.out" "$r/cluster/gpu.sbatch.sh" > "$r/provenance/gpu-retry-conditional-job.txt"
cp "$r/provenance/gpu-retry-conditional-job.txt" "$r/provenance/gpu-job.txt"
/mnt/projects/ll/tianzizh/SuperMimic_HOI/envs/tracking-py312/bin/python - <<'PY'
from pathlib import Path
import json,datetime,wandb
r=Path('/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi')
job=int((r/'provenance/gpu-job.txt').read_text().strip())
s={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'active_gpu_job_id':job,
 'previous_failed_gpu_job_id':1454300,'previous_failure':'MuJoCo-Warp conditional graph nodes require driver CUDA12.4+, L40 driver535 only exposes12.2',
 'repair':'Independent compat_launch.py chooses Model.opt.graph_conditional=False only if Warp reports conditional graph unsupported; solver iterations, tolerance and training hyperparameters unchanged.',
 'compatibility_unit_cases_passed':4,'repair_native_gpu_validated':False,
 'formal_training_started':False,'automation_id':'supermimic-2','automation_interval_minutes':15,
 'stop_monitor_when':'Formal iteration>=550 after500WM-only iterations, finite losses and gradients, observed log growth and W&B cloud readback'}
(r/'provenance/startup-status.json').write_text(json.dumps(s,indent=2)+'\n')
api=wandb.Api()
run=api.run('tianzizh-university-of-pennsylvania/SuperMimic_Upstream_Reproduction/setup-a1508d4-20261022')
run.summary['formal_gpu_job_id']=job
run.summary['previous_failed_gpu_job_id']=1454300
run.summary['setup_status']='graph scheduling compatibility patched; awaiting new native GPU gate'
run.summary['repair_native_gpu_validated']=False
run.summary.update()
print(json.dumps(s))
PY
