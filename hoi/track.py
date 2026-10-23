"""Sync only curated HOI80 evaluation metrics/configuration; no raw artifacts."""
from pathlib import Path
import argparse,json,os,time
os.environ['WANDB_BASE_URL']='https://api.forge.coreweave.com'
os.environ['WANDB_MODE']='online'
os.environ['WANDB_DISABLE_CODE']='true'
os.environ['WANDB_DISABLE_GIT']='true'
import wandb
p=argparse.ArgumentParser();p.add_argument('report',type=Path)
p.add_argument('--run-id',required=True);a=p.parse_args()
report=json.loads(a.report.read_text());assert report['status']=='completed' and len(report['cases'])==80
assert report['hand_collision_comparison'] and a.run_id==report['wandb_run_id']
assert a.run_id!='zeroshot-hoi80-stock14000-20261022'
run=wandb.init(entity='tianzizh-university-of-pennsylvania',project='SuperMimic_Upstream_Reproduction',
    id=a.run_id,name='ZeroShot-HOI80-stock14000-rubber-hand-collision',
    job_type='evaluation',resume='allow',save_code=False,
    settings=wandb.Settings(disable_code=True),
    config={key:report[key] for key in ('source_branch','source_commit','checkpoint_iteration','checkpoint_sha256',
        'input_manifest_sha256','zero_shot','optimizer_updates','normalization_updates','physical_objects',
        'object_observation_channels','no_failure_reset','reference_count','limitations')})
run.config.update({'stage':'zero_shot_evaluation','hand_collision_comparison':True,
    'only_scientific_change':report['only_scientific_change'],
    'baseline_report_sha256':report['baseline_report_sha256'],
    'hand_asset':report['hand_asset'],'hand_cooking_limitation':report['hand_cooking_limitation']},allow_val_change=True)
metrics={'hoi80/'+key:value for key,value in report['aggregate'].items()}
for key in ('phc_pass','object_pose_pass','body_and_object_pose_pass'):
    metrics['hoi80/'+key+'_fraction']=report['aggregate'][key]/80
run.log(metrics)
for key,value in report['by_object'].items():run.summary['hoi80/by_object/'+key]=value
for key,value in report['paired_comparison']['aggregate_deltas'].items():run.summary['comparison/delta/'+key]=value
columns=['id','object_asset','supermimic_norm_return','phc_pass','object_pose_pass','body_and_object_pose_pass','first_supermimic_failure_step','first_phc_failure_reference_s','first_object_failure_reference_s']
run.summary['hoi80/per_case']=wandb.Table(columns=columns,data=[[c[key] for key in columns] for c in report['cases']])
run.summary['hoi80/rollouts_saved_on_cluster']=80
run.summary['evaluation_status']='completed'
run.summary['slurm/job_id']=report['slurm_job_id']
url=run.url;run.finish()
for attempt in range(10):
    cloud=wandb.Api(timeout=30).run('tianzizh-university-of-pennsylvania/SuperMimic_Upstream_Reproduction/'+a.run_id)
    if cloud.summary.get('hoi80/count')==80 and all(abs(float(cloud.summary.get(key,float('nan')))-float(value))<1e-7 for key,value in metrics.items()):break
    time.sleep(3)
else:raise RuntimeError('W&B metrics uploaded but cloud readback has not matched yet; preserve evaluation and retry only tracking')
(a.report.parent/'wandb-readback.json').write_text(json.dumps({'url':url,'cloud_metrics_verified':True,'uploaded_raw_rollouts':False,'uploaded_weights':False},indent=2)+'\n')
print(json.dumps({'wandb_url':url,'cloud_metrics_verified':True}))
