"""Prepare a fresh immutable direct-teacher campaign; never start a GPU job."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from .config import TrainConfig
from .experiments import SCHEMA, PROFILES
from .telemetry import atomic_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(source, target):
    """Copy code/config only, and retain the existing robot assets read-only."""
    source, target = Path(source).resolve(), Path(target).resolve()
    if target.exists() or target == source or source.is_relative_to(target):
        raise ValueError("Frozen source must be a fresh separate directory")
    inherited = source/'SOURCE_MANIFEST.json'
    if inherited.exists():
        provenance = json.loads(inherited.read_text())
        branch = provenance['branch']
        revision = provenance.get('revision', provenance.get('parent'))
        for name,digest in provenance.get('files',{}).items():
            path=(source/name).resolve()
            if not path.is_relative_to(source) or not path.is_file() or sha(path)!=digest:
                raise ValueError('Existing frozen source changed: '+name)
    else:
        branch = subprocess.check_output(['git','branch','--show-current'],cwd=source,text=True).strip()
        revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
        if subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip():
            raise ValueError("Commit tracked source changes before launching a new experiment")
    if branch not in ('hoi-retarget/tables-v1.0','hoi-retarget/tables-boxes-v1.0'):
        raise ValueError("Unexpected direct-teacher source branch")
    target.mkdir(parents=True)
    files = {}
    for folder in ('supermimic','hoi_residual','hoi_tables','configs','cluster','tests'):
        for path in sorted((source/folder).rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix not in ('.py','.sh','.json'):
                continue
            name=path.relative_to(source).as_posix(); raw=path.read_bytes()
            if path.suffix=='.sh':raw=raw.replace(b'\r\n',b'\n')
            copy=target/name;copy.parent.mkdir(parents=True,exist_ok=True);copy.write_bytes(raw)
            files[name]=hashlib.sha256(raw).hexdigest()
    robot=(source/'data/g1_hw3').resolve()
    if not robot.is_dir():
        raise ValueError("Prepare the existing hw3 robot assets before freezing an experiment")
    (target/'data').mkdir();(target/'data/g1_hw3').symlink_to(robot,target_is_directory=True)
    value={'branch':branch,'revision':revision,'files':files}
    atomic_json(target/'SOURCE_MANIFEST.json',value)
    return value


def prepare(config, campaign, source, *, workers_per_gpu=None):
    config, campaign, source = Path(config).resolve(), Path(campaign).resolve(), Path(source).resolve()
    if campaign.exists():raise FileExistsError('Use a new campaign; active runs are never overwritten')
    value=json.loads(config.read_text());name=value.get('profile')
    if value.get('schema')!=SCHEMA or name not in PROFILES:raise ValueError('Invalid experiment configuration')
    from .train import actor_backward_schedule, build_parser
    from .experiments import resolve_arguments
    args=build_parser().parse_args(['--experiment-config',str(config),'--output',str(campaign/'training')])
    if workers_per_gpu is not None:args.workers_per_gpu=workers_per_gpu
    profile,binding=resolve_arguments(args)
    if profile.world_size%args.workers_per_gpu:raise ValueError('Workers must divide equally among GPUs')
    inputs={key:Path(getattr(args,key)).resolve() for key in ('manifest','asset_bundle','shape_bank','reference_contacts')}
    if name=='ultra':inputs.update(geometry_bundle=Path(args.geometry_bundle).resolve(),canonical=Path(args.canonical).resolve())
    if args.resume_bundle:inputs['resume_bundle']=Path(args.resume_bundle).resolve()
    if args.native_hands:
        from .native_hands import load_native_hands
        native_hands = load_native_hands(args.native_hands)
        inputs['native_hands'] = Path(native_hands.identity['path'])
    for key,path in inputs.items():
        if not path.is_file():raise FileNotFoundError(f'Missing {key}: {path}; run reference-only preparation first')
    campaign.mkdir(parents=True);(campaign/'input').mkdir()
    destinations={'manifest':'manifest.json','asset_bundle':'assets.json','shape_bank':'shape_bank.json',
                  'reference_contacts':'reference_contacts.pt','geometry_bundle':'geometry_bundle.json','canonical':'hoi80_manifest.json',
                  'resume_bundle':'resume_bundle.json','native_hands':'native_hands.json'}
    paths={}
    for key,path in inputs.items():
        target=campaign/'input'/destinations[key];target.symlink_to(path);paths[key]=str(target)
    if name=='tables':
        # Existing protected evaluator's request format is retained.
        (campaign/'formal_inputs01').mkdir()
        (campaign/'formal_inputs01/reference_contacts.pt').symlink_to(inputs['reference_contacts'])
    frozen=campaign/'source';manifest=freeze(source,frozen)
    normalized=json.loads(config.read_text());normalized['paths'].update({k:v for k,v in paths.items() if k!='canonical'})
    normalized['execution']['workers_per_gpu']=args.workers_per_gpu
    normalized['execution']['checkpoint_every']=args.checkpoint_every
    if name=='ultra':normalized['execution']['actor_ranks_per_turn']=args.actor_ranks_per_turn
    if name=='ultra':normalized['evaluation']['canonical']=paths['canonical']
    atomic_json(campaign/'experiment.json',normalized)
    training=campaign/'training';training.mkdir()
    from .networks import feature_schema
    run_config=TrainConfig(num_envs=args.num_envs,iterations=args.iterations,seed=args.seed,ckpt_every=args.checkpoint_every).validate()
    metadata={'branch':manifest['branch'],'from_scratch':True,'global_config':asdict(run_config),
              'schema':feature_schema(),'gpu_type':'L40','gpus':profile.world_size//args.workers_per_gpu,
              'total_env':args.num_envs,'source_sha256':sha(frozen/'SOURCE_MANIFEST.json'),
              'manifest_sha256':sha(inputs['manifest']),'asset_catalog_sha256':sha(inputs['asset_bundle']),
              'shape_bank_sha256':sha(inputs['shape_bank']),'reference_contacts_sha256':sha(inputs['reference_contacts'])}
    raw_manifest=json.loads(inputs['manifest'].read_text())
    metadata['train_clips']=raw_manifest.get('partitions',{}).get('train',{}).get('motions',0) if name=='ultra' else sum(c['partition']=='train' for c in raw_manifest['clips'])
    metadata['evaluation_clips']=80 if name=='ultra' else 22
    metadata['asset_count']=profile.world_size
    if name=='tables':metadata['protected_eval']=22
    else:metadata['train_source_groups']=raw_manifest['partitions']['train']['source_groups']
    metadata['experiment']={'profile':name,'assets':list(profile.assets),'evaluation':normalized['evaluation'],
                            'workers_per_gpu':args.workers_per_gpu,'config_sha256':sha(config),
                            'actor_backward_schedule':actor_backward_schedule(name,args.workers_per_gpu,args.actor_ranks_per_turn)}
    if name=='ultra':metadata['experiment']['actor_ranks_per_turn']=args.actor_ranks_per_turn
    if args.native_hands:
        metadata['experiment']['native_hands_sha256'] = native_hands.identity['sha256']
    if args.ls_iterations!=50 or args.resume_bundle:
        from .solver_health import policy
        metadata['solver_health']=policy(args.ls_iterations)
    if args.resume_bundle:
        from .resume import bundle,lineage
        parent=bundle(inputs['resume_bundle'])
        metadata.update(from_scratch=False,resume=lineage(parent),start_iteration=parent['iteration']+1)
    atomic_json(training/'run_metadata.json',metadata)
    receipt={'campaign':str(campaign),'source':str(frozen),'profile':name,'assets':list(profile.assets),
             'workers':profile.world_size,'workers_per_gpu':args.workers_per_gpu,
             'gpus':profile.world_size//args.workers_per_gpu,'input_sha256':{k:sha(v) for k,v in inputs.items()},
             'optimizer_updates':0,'gpu_job_started':False,'evaluation_owning_run':True}
    if args.resume_bundle:receipt.update(resume=metadata['resume'],start_iteration=metadata['start_iteration'])
    atomic_json(campaign/'preparation.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--campaign',required=True);p.add_argument('--source',required=True)
    p.add_argument('--workers-per-gpu',type=int)
    a=p.parse_args();print(json.dumps(prepare(a.config,a.campaign,a.source,workers_per_gpu=a.workers_per_gpu),indent=2))


if __name__=='__main__':main()
