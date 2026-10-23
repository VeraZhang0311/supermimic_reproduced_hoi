"""Sync only curated HOI80 evaluation metrics/configuration; no raw artifacts."""
from pathlib import Path
import argparse,json
import wandb
p=argparse.ArgumentParser();p.add_argument('report',type=Path);a=p.parse_args()
report=json.loads(a.report.read_text());assert report['status']=='completed' and len(report['cases'])==80
run=wandb.init(entity='tianzizh-university-of-pennsylvania',project='SuperMimic_Upstream_Reproduction',
    id='zeroshot-hoi80-stock14000-20261022',name='ZeroShot-HOI80-stock14000-objects-no-object-obs',
    job_type='evaluation',resume='allow',save_code=False,
    settings=wandb.Settings(disable_code=True),
    config={key:report[key] for key in ('source_branch','source_commit','checkpoint_iteration','checkpoint_sha256',
        'input_manifest_sha256','zero_shot','optimizer_updates','normalization_updates','physical_objects',
        'object_observation_channels','no_failure_reset','reference_count','limitations')})
run.config.update({'stage':'zero_shot_evaluation'},allow_val_change=True)
metrics={'hoi80/'+key:value for key,value in report['aggregate'].items()}
for key in ('phc_pass','object_pose_pass','body_and_object_pose_pass'):
    metrics['hoi80/'+key+'_fraction']=report['aggregate'][key]/80
run.log(metrics)
for key,value in report['by_object'].items():run.summary['hoi80/by_object/'+key]=value
columns=['id','object_asset','supermimic_norm_return','phc_pass','object_pose_pass','body_and_object_pose_pass','first_supermimic_failure_step','first_phc_failure_reference_s','first_object_failure_reference_s']
run.summary['hoi80/per_case']=wandb.Table(columns=columns,data=[[c[key] for key in columns] for c in report['cases']])
run.summary['hoi80/rollouts_saved_on_cluster']=80
url=run.url;run.finish()
cloud=wandb.Api(timeout=30).run('tianzizh-university-of-pennsylvania/SuperMimic_Upstream_Reproduction/zeroshot-hoi80-stock14000-20261022')
assert cloud.summary['hoi80/count']==80
for key,value in metrics.items():assert abs(float(cloud.summary[key])-float(value))<1e-7
(a.report.parent/'wandb-readback.json').write_text(json.dumps({'url':url,'cloud_metrics_verified':True,'uploaded_raw_rollouts':False,'uploaded_weights':False},indent=2)+'\n')
print(json.dumps({'wandb_url':url,'cloud_metrics_verified':True}))
