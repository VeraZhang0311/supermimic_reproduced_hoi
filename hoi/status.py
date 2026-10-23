"""Read actual Slurm, saved evaluation and rollout state; never submit or mutate a run."""
from pathlib import Path
import datetime,hashlib,json,subprocess
root=Path(__file__).resolve().parents[1]
job=(root/'provenance/eval-job.txt').read_text().strip()
def command(args):return subprocess.check_output(args,cwd=root,text=True,stderr=subprocess.STDOUT).strip()
def read(path):return path.read_text(errors='replace') if path.exists() else ''
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
out=root/f'results/hoi80_stock14000_handcollision-{job}'
manifest=json.loads((root/'results/hoi80_inputs/manifest.json').read_text())
groups={}
for asset in manifest['assets']:
    path=out/asset/'evaluation.json'
    groups[asset]=json.loads(path.read_text()) if path.exists() else None
report=json.loads((out/'evaluation.json').read_text()) if (out/'evaluation.json').exists() else None
utc=datetime.datetime.now(datetime.timezone.utc)
receipt={
    'utc':utc.isoformat(),'job_id':job,
    'queue':command(['/opt/slurm/bin/squeue','-u','tianzizh','--noheader','--format=%i|%j|%T|%M|%R']),
    'accounting':command(['/opt/slurm/bin/sacct','-j',job,'--format=JobID,State,Elapsed,ExitCode,NodeList','-P']),
    'source_branch':command(['git','branch','--show-current']),
    'source_commit':command(['git','rev-parse','HEAD']),
    'author_core_diff':command(['git','diff','--name-only','a1508d4','--','supermimic','tools']),
    'canonical_cases':len(manifest['cases']),'physical_objects':True,'object_observation_channels':0,
    'completed_full_cases':sum(len(value['cases']) for value in groups.values() if value),
    'completed_asset_groups':[key for key,value in groups.items() if value],
    'all80_evaluation_completed':bool(report and report['status']=='completed' and len(report['cases'])==80),
    'aggregate':report['aggregate'] if report else None,
    'pipeline_tail':read(root/f'logs/eval-{job}.out').splitlines()[-20:],
    'failure':json.loads((out/'failure.json').read_text()) if (out/'failure.json').exists() else None,
    'group_log_tails':{p.name:read(p).splitlines()[-12:] for p in out.glob('*.stdout.log')},
    'output':str(out),'training_or_configuration_mutated':False,
    'hand_asset':json.loads((out/'hand_asset/hand-asset.json').read_text()) if (out/'hand_asset/hand-asset.json').exists() else None,
    'hand_gate_passed':(out/'hand-gate-passed.json').exists(),
}
if report:
    receipt['rollout_sha256_verified']=all(sha(out/case['rollout'])==case['rollout_sha256'] for case in report['cases'])
    receipt['wandb']=json.loads((out/'wandb-readback.json').read_text()) if (out/'wandb-readback.json').exists() else None
else:receipt['rollout_sha256_verified']=False
path=root/('provenance/hoi80-status-'+utc.strftime('%Y%m%dT%H%M%SZ')+'.json')
path.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
print(json.dumps({'receipt_path':str(path),'receipt':receipt},indent=2))
