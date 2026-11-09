"""Identity-bound full learner restore with fresh simulator/replay, not restart."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re

from .experiments import ULTRA_ASSETS
from .solver_health import identity_policy

SCHEMA = 'hoi_direct_continuation_v1'


def bundle(path):
    path=Path(path);raw=path.read_bytes();value=json.loads(raw)
    if (value.get('schema')!='hoi_resume_bundle_v1' or value.get('passed') is not True
        or value.get('world_size')!=8 or value.get('assets')!=list(ULTRA_ASSETS)
        or type(value.get('iteration')) is not int or value['iteration']<500 or value['iteration']%500
        or value.get('replay_and_simulator_serialized') is not False):
        raise ValueError('Invalid complete Ultra continuation bundle')
    records=value.get('records',[])
    if len(records)!=8 or [r['rank'] for r in records]!=list(range(8)) or [r['asset'] for r in records]!=list(ULTRA_ASSETS):
        raise ValueError('Resume bundle must bind every exact rank/asset')
    if any(not re.fullmatch('[a-f0-9]{64}',r['sha256']) for r in records):raise ValueError('Invalid checkpoint hash')
    if not re.fullmatch('[A-Za-z0-9_-]{4,64}',value.get('parent_run_id','')):raise ValueError('Invalid parent run')
    value['bundle_sha256']=hashlib.sha256(raw).hexdigest()
    return value


def lineage(value):
    return {'schema':SCHEMA,'iteration':value['iteration'],'parent_run_id':value['parent_run_id'],
        'bundle_sha256':value['bundle_sha256'],
        'checkpoint_sha256':{f"rank_{r['rank']:02d}":r['sha256'] for r in value['records']},
        'fresh_replay_and_simulator':True,'repeat_wm_warmup':False}


def validate_lineage(identity):
    if identity.get('from_scratch') is True and identity.get('base_checkpoint') is None:return
    base=identity.get('base_checkpoint',{})
    if (identity.get('from_scratch') is not False or not isinstance(base,dict)
        or base.get('schema')!=SCHEMA or type(base.get('iteration')) is not int or base['iteration']<500
        or base['iteration']%500 or base.get('fresh_replay_and_simulator') is not True
        or base.get('repeat_wm_warmup') is not False
        or not re.fullmatch('[a-f0-9]{64}',base.get('bundle_sha256',''))
        or not re.fullmatch('[A-Za-z0-9_-]{4,64}',base.get('parent_run_id',''))
        or set(base.get('checkpoint_sha256',{}))!={f'rank_{i:02d}' for i in range(8)}
        or any(not re.fullmatch('[a-f0-9]{64}',v) for v in base['checkpoint_sha256'].values())):
        raise ValueError('Invalid explicitly bound Ultra continuation lineage')
    identity_policy(identity)


def _semantic(identity):
    from .evaluate import _reward_identity
    result=deepcopy(identity)
    for name in ('from_scratch','base_checkpoint','solver_health'):result.pop(name,None)
    result['reward']=_reward_identity(result['reward'])
    result.get('experiment',{}).get('config_binding',{}).pop('config_sha256',None)
    return json.dumps(result,sort_keys=True,separators=(',',':'))


def load_rank(path,rank,current_identity, *, enforce_model_source=True):
    import torch
    value=bundle(path);record=value['records'][rank]
    raw=Path(record['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=record['sha256']:raise ValueError('Resume checkpoint bytes changed')
    checkpoint=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    if checkpoint.get('schema')!='hoi_tables_checkpoint_v1' or checkpoint.get('iteration')!=value['iteration']:
        raise ValueError('Resume checkpoint iteration/schema changed')
    old=checkpoint['identity'];validate_lineage(old)
    if _semantic(old)!=_semantic(current_identity):raise ValueError('Resume scientific/data/config identity differs')
    before,after=identity_policy(old),identity_policy(current_identity)
    if after['ls_iterations']<before['ls_iterations']:raise ValueError('Resume cannot lower solver budget')
    if old['asset_id']!=rank or old['asset']!=ULTRA_ASSETS[rank]:raise ValueError('Resume rank/asset mismatch')
    if enforce_model_source:
        source=Path(value['parent_source']);manifest_path=source/'SOURCE_MANIFEST.json'
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=value['parent_source_manifest_sha256']:
            raise ValueError('Resume parent source manifest changed')
        manifest=json.loads(manifest_path.read_text());runtime=Path(__file__).resolve().parents[1]
        # Learning equations, normalizers, RSI, data, contacts, robot and reward
        # must remain byte-identical; orchestration/diagnostic additions below
        # are the only reviewed exceptions.
        allowed={'hoi_tables/train.py','hoi_tables/env.py','hoi_tables/experiments.py','hoi_tables/launch.py',
                 'hoi_tables/evaluate_hoi80.py','hoi_tables/telemetry.py'}
        for name,digest in manifest['files'].items():
            if name.endswith('.py') and name.split('/')[0] in ('supermimic','hoi_residual','hoi_tables'):
                if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:raise ValueError('Parent source changed: '+name)
                if name not in allowed and hashlib.sha256((runtime/name).read_bytes()).hexdigest()!=digest:
                    raise ValueError('Resume changed protected scientific implementation: '+name)
    return checkpoint,value


def restore(checkpoint,value,learner,env, *, restore_rng=True):
    import torch
    from .capacity import _hash_state
    from .evaluate import _validate_state,state_dict_hash
    model_hashes={}
    for name,module in (('policy',learner.pi),('wm',learner.wm),('critic',learner.critic)):
        state=checkpoint[name];_validate_state(state,module.state_dict(),name)
        module.load_state_dict(state,strict=True)
        model_hashes[name]=state_dict_hash(module.state_dict())
        if model_hashes[name]!=state_dict_hash(state):raise RuntimeError('Resume model/RMS restore differs')
    if set(checkpoint['optimizers'])!=set(learner.optimizers):raise ValueError('Resume optimizer set differs')
    for name,optimizer in learner.optimizers.items():
        optimizer.load_state_dict(checkpoint['optimizers'][name])
        if _hash_state(optimizer.state_dict())!=_hash_state(checkpoint['optimizers'][name]):
            raise RuntimeError('Resume Adam state restore differs: '+name)
    env.sampler.load_state_dict(checkpoint['sampler'])
    if _hash_state(env.sampler.state_dict())!=_hash_state(checkpoint['sampler']):raise RuntimeError('Resume RSI restore differs')
    for name in ('contact_positive','contact_negative'):
        getattr(learner,name).copy_(checkpoint[name])
        if not torch.equal(getattr(learner,name).cpu(),checkpoint[name].cpu()):raise RuntimeError('Resume contact statistics differ')
    learner.iteration=env.iteration=checkpoint['iteration']
    if restore_rng:
        torch.set_rng_state(checkpoint['rng']['torch'].cpu())
        if torch.device(learner.device).type=='cuda':torch.cuda.set_rng_state(checkpoint['rng']['cuda'].cpu(),learner.device)
    return {'iteration':checkpoint['iteration'],'next_iteration':checkpoint['iteration']+1,
        'models_and_rms_sha256':model_hashes,'optimizer_steps':{k:sorted({int(v['step']) for v in o.state.values()}) for k,o in learner.optimizers.items()},
        'optimizer_sha256':{k:_hash_state(o.state_dict()) for k,o in learner.optimizers.items()},
        'sampler_sha256':_hash_state(env.sampler.state_dict()),'lineage':lineage(value),
        'model_optimizer_RSI_contact_counts_restored':True,'replay_and_simulator_recreated':True}
