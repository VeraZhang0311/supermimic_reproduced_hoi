"""One immutable-checkpoint HOI80 evaluation; preserve every completed rollout."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import traceback
from hoi.hand_collision import prepare as prepare_hands, assert_same_physics, LIMITATION

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    baseline=Path('/mnt/projects/ll/tianzizh/supermimic_hoi_zeroshot/results/hoi80_stock14000-1457069/evaluation.json')
    baseline_report=json.loads(baseline.read_text())
    checkpoint_hash=sha(a.checkpoint)
    assert checkpoint_hash==baseline_report['checkpoint_sha256']
    for name,digest in baseline_report['source_sha256'].items():
        if name.startswith('supermimic/') or name.startswith('hoi/ultra_') or name=='hoi/prepare.py':
            assert sha(ROOT/name)==digest,name
    manifest=json.loads((a.prepared/'manifest.json').read_text())
    assert sha(a.prepared/'manifest.json')==baseline_report['input_manifest_sha256']
    source={str(p.relative_to(ROOT)):sha(p) for folder in ('hoi','supermimic') for p in sorted((ROOT/folder).glob('*.py'))}
    identity={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'source_branch':'codex/hoi80-rubber-hand-collision',
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'checkpoint':str(a.checkpoint),'checkpoint_iteration':14000,'checkpoint_sha256':checkpoint_hash,
        'input_manifest_sha256':sha(a.prepared/'manifest.json'),'source_sha256':source,
        'zero_shot':True,'optimizer_updates':0,'normalization_updates':0,
        'physical_objects':True,'object_observation_channels':0,'no_failure_reset':True,
        'reference_count':80,'checkpoint_uploads':False,'rollout_uploads':False,'source_uploads':False,
        'hand_collision_comparison':True,'baseline_report':str(baseline),'baseline_report_sha256':sha(baseline),
        'wandb_run_id':'zeroshot-hoi80-stock14000-rubberhands-20261022',
        'only_scientific_change':'Add original HOI rubber-hand collision geometry; existing wrists/robot/object/model settings unchanged',
        'hand_cooking_limitation':LIMITATION}
    (a.output/'manifest.json').write_text(json.dumps(identity,indent=2)+'\n')
    def evaluate(asset,directory,gate=0,robot_usd=None):
        command=[sys.executable,str(ROOT/'hoi/launch.py'),str(ROOT/'hoi/evaluate.py'),
            '--prepared',str(a.prepared),'--checkpoint',str(a.checkpoint),
            '--asset',asset,'--output',str(directory)]
        if gate:command+=['--gate-steps',str(gate)]
        if robot_usd:command+=['--robot-usd',str(robot_usd)]
        log=a.output/(directory.name+'.stdout.log')
        print(json.dumps({'command':command,'log':str(log)}),flush=True)
        with log.open('w') as stream:
            subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=2700)
        return json.loads((directory/'evaluation.json').read_text())
    try:
        assets=sorted(manifest['assets'])
        hand_asset=prepare_hands(ROOT,a.output/'hand_asset',a.prepared/'metadata/ultra_g1.urdf')
        identity['hand_asset']=hand_asset
        (a.output/'manifest.json').write_text(json.dumps(identity,indent=2)+'\n')
        gate_off=evaluate(assets[0],a.output/'native_gate_off',16)
        gate=evaluate(assets[0],a.output/'native_gate_on',16,hand_asset['usd'])
        assert gate['gate_only'] and gate['native']['compiled_mujoco_allowed_object_robot_geom_pairs']>0
        assert_same_physics(gate_off['native']['hand_collision_verification'],gate['native']['hand_collision_verification'])
        (a.output/'hand-gate-passed.json').write_text(json.dumps({'off':gate_off['native']['hand_collision_verification'],
            'on':gate['native']['hand_collision_verification'],'same_robot_mass_inertia_joints':True,
            'original_authored_properties_unchanged':hand_asset['original_authored_values_unchanged']},indent=2)+'\n')
        cases=[]
        for asset in assets:
            result=evaluate(asset,a.output/asset,robot_usd=hand_asset['usd'])
            assert not result['gate_only'] and result['checkpoint_sha256']==checkpoint_hash
            assert all(case['full_reference_completed'] for case in result['cases'])
            for case in result['cases']:
                case['rollout']=asset+'/'+case['rollout'];cases.append(case)
            assert sha(a.checkpoint)==checkpoint_hash
        assert len(cases)==80 and {c['id'] for c in cases}=={c['id'] for c in manifest['cases']}
        assert all(sha(ROOT/name)==digest for name,digest in source.items())
        assert sha(baseline)==identity['baseline_report_sha256']
        def aggregate(items):
            return {'count':len(items),'phc_pass':sum(c['phc_pass'] for c in items),
                'object_pose_pass':sum(c['object_pose_pass'] for c in items),
                'body_and_object_pose_pass':sum(c['body_and_object_pose_pass'] for c in items),
                'supermimic_n_failed':sum(c['first_supermimic_failure_step']>=0 for c in items),
                'supermimic_norm_return':sum(c['supermimic_norm_return'] for c in items)/len(items)}
        report={**identity,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'status':'completed','aggregate':aggregate(cases),
            'by_object':{obj:aggregate([c for c in cases if c['id'].startswith(obj+'_')]) for obj in ('smallbox','largebox','plasticbox','suitcase')},
            'limitations':['HOI object-pose pass is the existing engineering pose diagnostic, not a contact/grasp-success judge.',
                'Uses frozen SuperMimic G1/PD and Newton MuJoCo-Warp, adding only original HOI rubber-hand collision geometry.',
                LIMITATION,
                'Object meshes retain geometry and nominal mass/COM/inertia; Newton CoACD differs from PhysX VHACD cooking.',
                'These are the existing canonical validation80, not a newly held-out test split.'],
            'cases':cases}
        old={c['id']:c for c in baseline_report['cases']}
        report['paired_comparison']={'baseline_aggregate':baseline_report['aggregate'],
            'aggregate_deltas':{k:report['aggregate'][k]-baseline_report['aggregate'][k] for k in report['aggregate'] if k!='count'},
            'per_case':[{'id':c['id'],'return_delta':c['supermimic_norm_return']-old[c['id']]['supermimic_norm_return'],
                'baseline_phc_pass':old[c['id']]['phc_pass'],'hand_collision_phc_pass':c['phc_pass'],
                'baseline_object_pose_pass':old[c['id']]['object_pose_pass'],'hand_collision_object_pose_pass':c['object_pose_pass']} for c in cases]}
        (a.output/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'output':str(a.output),'aggregate':report['aggregate']}),flush=True)
    except BaseException as error:
        (a.output/'failure.json').write_text(json.dumps({'error':repr(error),'traceback':traceback.format_exc()},indent=2)+'\n')
        raise

if __name__=='__main__':main()
