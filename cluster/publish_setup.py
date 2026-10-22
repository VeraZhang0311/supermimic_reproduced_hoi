"""Publish queued ON experiment metadata only; formal metrics use a separate run."""
from pathlib import Path
import datetime
import json
import os
import subprocess
import wandb

ROOT=Path('/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision')
identity=json.loads((ROOT/'provenance/source-identity.json').read_text())
job=(ROOT/'provenance/gpu-job.txt').read_text().strip()
status=subprocess.check_output(['/opt/slurm/bin/squeue','-j',job,'-h','-o','%T|%R'],text=True).strip()
run=wandb.init(entity='tianzizh-university-of-pennsylvania',project='SuperMimic_Upstream_Reproduction',
 id='setup-selfcoll-20261022',name='LAFAN1-selfcollision-ON-setup',job_type='setup',resume='allow',
 config={'stage':'setup','experiment':'selfcollision_on','physical_self_collisions':True,
 'extra_self_collision_reward_penalty':False,'comparison_group':'LAFAN1-seed0-selfcollision',
 'training_parameters':identity['training_parameters'],'source_identity':identity,
 'baseline_run_id':'stock-lafan1-20261022-s0','formal_run_id':'selfcoll-lafan1-20261022-s0'},
 save_code=False,mode='online',settings=wandb.Settings(disable_code=True),
 tags=['selfcollision-on','controlled-comparison','LAFAN1','L40'])
run.summary.update({'slurm_job_id':int(job),'slurm_snapshot':status,'cpu_setup_passed':True,
 'native_self_collision_verified':False,'formal_training_stable':False,'checkpoint_uploads':False})
url=run.url
run.finish()
cloud=wandb.Api().run('tianzizh-university-of-pennsylvania/SuperMimic_Upstream_Reproduction/setup-selfcoll-20261022')
assert cloud.config['physical_self_collisions'] is True
assert int(cloud.summary['slurm_job_id'])==int(job)
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'url':url,'id':cloud.id,
 'cloud_readback':True,'physical_self_collisions':True,'slurm_job_id':job,'slurm_snapshot':status}
(ROOT/'provenance/wandb-setup.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
