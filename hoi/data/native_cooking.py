"""Hash-bound, resumable native cooking batches; never a native acceptance gate."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import subprocess
import sys
if __package__:
    from .native_assets import canonical, digest, sha, exact_surface
else:
    from hoi.data.native_assets import canonical, digest, sha, exact_surface

PLAN_SCHEMA='hoi_native_cooking_batch_plan_v1'
SUCCESS_SCHEMA='hoi_native_cooking_task_success_v1'
DISTANCE_SCHEMA='hoi_source_native_sampled_unsigned_distance_v1'
_PLAN_IDENTITY_CACHE=ContextVar('native_cooking_plan_identity_cache',default=None)


@contextmanager
def _planning_identity_cache():
    """Share verified immutable bindings only during one plan creation call."""
    if _PLAN_IDENTITY_CACHE.get() is not None:
        yield
        return
    token=_PLAN_IDENTITY_CACHE.set({})
    try:
        yield
    finally:
        _PLAN_IDENTITY_CACHE.reset(token)


def _stat_signature(path):
    stat=path.stat()
    return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)


def _json(path):
    return json.loads(Path(path).read_bytes())


def _write(path,value):
    with Path(path).open('xb') as stream:
        stream.write(canonical(value)+b'\n')


def _identity(path,expected=None,kind='bytes'):
    path=Path(path).resolve()
    cache=_PLAN_IDENTITY_CACHE.get()
    key=(str(path),expected,kind) if cache is not None and isinstance(expected,str) and expected else None
    before=_stat_signature(path) if key is not None else None
    if key is not None and key in cache:
        signature,binding=cache[key]
        if signature==before:
            return dict(binding)
        del cache[key]
    actual=digest(_json(path)) if kind=='canonical_json' else sha(path)
    if expected is not None and actual!=expected:
        raise ValueError('input identity changed: '+str(path))
    binding={'path':str(path),'sha256':actual,'kind':kind}
    if key is not None:
        after=_stat_signature(path)
        if after!=before:
            raise ValueError('input changed while hashing: '+str(path))
        cache[key]=(after,dict(binding))
    return binding


def verify_identity(binding):
    return _identity(binding['path'],binding['sha256'],binding.get('kind','bytes'))


def _source_files(source):
    source=Path(source).resolve();files=[]
    for folder in ('hoi','supermimic'):
        files.extend((source/folder).rglob('*.py'))
    files.extend((source/'configs').glob('hoi_v1*.json'))
    files.extend((source/'cluster').glob('hoi_v1*.sh'))
    if not files or not (source/'hoi/v1/native.py').exists():
        raise ValueError('native source checkout is missing')
    return {p.relative_to(source).as_posix():sha(p) for p in sorted(files)}


def _fixture_inputs(fixture_path):
    f=_json(fixture_path);asset=f['asset'];record=f['source_record'];build=asset['build_receipt'];robot=f['native_robot_asset']
    result=[_identity(fixture_path),verify_identity(build),_identity(asset['native_asset'],asset['native_asset_sha256']),
            _identity(Path(record['output'])/'metadata.json',record['metadata_sha256']),
            _identity(Path(record['output'])/'labels.npz',record['labels_sha256']),
            _identity(record['object_geometry_path'],record['object_geometry_sha256'],'canonical_json'),
            _identity(robot['path'],robot['sha256']),verify_identity(f['reference_kinematics']['urdf'])]
    for name in ('hand_asset_manifest','foot_asset_manifest'):
        if name in robot:result.append(verify_identity(robot[name]))
    if 'robot_recipe' in f:result.append(verify_identity(f['robot_recipe']))
    if f.get('repair_binding'):
        from hoi.data.native_repairs import fixture_repair_context, dependency_bindings
        fixture_repair_context(f)
        result.extend(verify_identity(x) for x in dependency_bindings(f['repair_binding']['path']))
    return result


def _runtime_versions(python):
    code="import importlib.metadata as m,json;print(json.dumps({k:m.version(v) for k,v in [('torch','torch'),('warp','warp-lang'),('mujoco','mujoco'),('mujoco_warp','mujoco-warp'),('newton','newton')]}))"
    result=subprocess.run([str(python),'-c',code],check=True,capture_output=True,text=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


@_planning_identity_cache()
def make_plan(registry_path,source,python,native_args,output,*,surface_samples=1024,seed=20261031,defer_surface_report=False):
    registry_path=Path(registry_path).resolve();registry=_json(registry_path)
    if registry.get('schema')!='hoi_curated_native_source_registry_v1' or registry.get('content_sha256')!=digest({k:v for k,v in registry.items() if k!='content_sha256'}):
        raise ValueError('native registry schema/content mismatch')
    from hoi.data.native_repairs import validate_derived_registry
    validate_derived_registry(registry)
    if not isinstance(native_args,list) or not all(isinstance(x,str) for x in native_args):
        raise ValueError('native args must be a JSON list of argument strings')
    reserved=('--output','--asset-id','--num-envs','--steps','--bundle','--legacy-prepared-fixture','--curated-cooking-fixture')
    if any(any(x==key or x.startswith(key+'=') for key in reserved) for x in native_args):
        raise ValueError('batch owns native input/output/env-count arguments')
    if '--hand-collision' not in native_args or '--self-collision' not in native_args or any(x in native_args for x in ('--no-hand-collision','--no-self-collision')):
        raise ValueError('cooking batch requires explicit hand and self collision enabled')
    if type(surface_samples) is not int or surface_samples<64:
        raise ValueError('surface report requires at least 64 deterministic samples per direction')
    source=Path(source).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    files=_source_files(source);tasks=[]
    for group in registry['object_cooking_groups']:
        fixture=verify_identity(group['cooking_fixture']);f=_json(fixture['path'])
        task={'kind':'native_cook','task_id':'object_'+group['object_build_group_sha256'][:20],
              'object_build_group_sha256':group['object_build_group_sha256'],
              'covered_geometry_identities':group['members'],'fixture':fixture,
              'source_geometry':{'path':f['source_record']['object_geometry_path'],'sha256':f['source_record']['object_geometry_sha256'],'kind':'canonical_json'},
              'inputs':_fixture_inputs(fixture['path'])}
        task['task_sha256']=digest(task);tasks.append(task)
    for binding in registry['bindings']:
        if binding['dataset']!='ultra':continue
        asset=binding['asset_recipe'];record=binding['source_record']
        task={'kind':'bootstrap_verify','task_id':'ultra_'+binding['object_geometry_sha256'][:20],
              'covered_geometry_identities':[binding['object_geometry_sha256']], 'asset_recipe':asset,
              'source_geometry':{'path':record['object_geometry_path'],'sha256':record['object_geometry_sha256'],'kind':'canonical_json'},
              'inputs':[_identity(asset['native_asset'],asset['native_asset_sha256']),
                        _identity(record['object_geometry_path'],record['object_geometry_sha256'],'canonical_json')]}
        task['task_sha256']=digest(task);tasks.append(task)
    if len({t['task_id'] for t in tasks})!=len(tasks):raise ValueError('task ID collision')
    covered=[h for task in tasks for h in task['covered_geometry_identities']]
    if len(covered)!=registry['selected_geometry_identities'] or len(set(covered))!=len(covered):
        raise ValueError('batch tasks do not cover each registry identity exactly once')
    execution={'source':str(source),'source_files':files,'python':str(Path(python).absolute()),'native_args':native_args,
               'surface_samples':surface_samples,'surface_seed':seed,'surface_report_schema':DISTANCE_SCHEMA,
               'defer_surface_report':bool(defer_surface_report),'driver_identity':_identity(__file__),
               'python_identity':_identity(python),
               'runtime_versions':_runtime_versions(python) if registry['object_cooking_groups'] else None}
    plan={'schema':PLAN_SCHEMA,'registry':_identity(registry_path),'output':str(output),'execution':execution,
          'execution_sha256':digest(execution),'tasks':tasks,'geometry_identity_count':len(covered),
          'task_counts':{kind:sum(t['kind']==kind for t in tasks) for kind in ('native_cook','bootstrap_verify')},
          'full_selection':registry.get('full_selection',False),'training_started':False,'scope':'cooking/bootstrap only; every final native asset still requires the standard native gate'}
    plan['content_sha256']=digest(plan);_write(output/'plan.json',plan)
    return plan


def read_plan(path):
    plan=_json(path)
    if plan.get('schema')!=PLAN_SCHEMA or plan.get('content_sha256')!=digest({k:v for k,v in plan.items() if k!='content_sha256'}) or plan.get('execution_sha256')!=digest(plan['execution']):
        raise ValueError('cooking plan identity changed')
    verify_identity(plan['registry'])
    verify_identity(plan['execution']['python_identity'])
    verify_identity(plan['execution']['driver_identity'])
    versions=plan['execution'].get('runtime_versions')
    if versions is not None and _runtime_versions(plan['execution']['python'])!=versions:
        raise ValueError('native runtime package versions changed after planning')
    if _source_files(plan['execution']['source'])!=plan['execution']['source_files']:
        raise ValueError('native executable source changed after planning')
    for task in plan['tasks']:
        if task.get('task_sha256')!=digest({k:v for k,v in task.items() if k!='task_sha256'}):raise ValueError('cooking task identity changed')
    return plan


def _meshes(shapes):
    import numpy as np
    import trimesh
    result=[]
    for shape in exact_surface(shapes):
        if any(k in shape for k in ('transform','position','quaternion_wxyz')):raise ValueError('surface audit needs composed object-local geometry')
        if shape['kind']=='cube':
            box=trimesh.creation.box(extents=shape['size']);v,f=box.vertices,box.faces
        elif shape['kind']=='mesh':v,f=shape['vertices'],shape['faces']
        else:raise ValueError('unsupported unsigned surface shape')
        result.append({'vertices':np.asarray(v,dtype=np.float64),'faces':np.asarray(f,dtype=np.int64)})
    return result


def _sample(meshes,count,seed):
    import numpy as np
    tri=np.concatenate([m['vertices'][m['faces']] for m in meshes]);area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    if not np.isfinite(tri).all() or area.sum()<=0:raise ValueError('source/cooked mesh has no finite area')
    rng=np.random.default_rng(seed);indices=rng.choice(len(tri),count,p=area/area.sum());chosen=tri[indices]
    u=np.sqrt(rng.random(count));v=rng.random(count)
    return (1-u[:,None])*chosen[:,0]+(u*(1-v))[:,None]*chosen[:,1]+(u*v)[:,None]*chosen[:,2]


def unsigned_distances(points,meshes):
    """Exact point-to-triangle/segment distance; closure and winding unnecessary."""
    import numpy as np
    import torch
    triangles=torch.as_tensor(np.concatenate([m['vertices'][m['faces']] for m in meshes]),dtype=torch.float64)
    a,b,c=triangles.unbind(1);e0,e1=b-a,c-a;normal=torch.cross(e0,e1,dim=-1);norm2=normal.square().sum(-1)
    d00,d01,d11=e0.square().sum(-1),(e0*e1).sum(-1),e1.square().sum(-1);denom=d00*d11-d01.square()
    out=[];chunk=max(1,min(256,131072//len(triangles)))
    for point in torch.as_tensor(points,dtype=torch.float64).split(chunk):
        pa=point[:,None]-a;h=(pa*normal).sum(-1)/norm2.clamp_min(1e-300)
        rel=point[:,None]-h[...,None]*normal-a;d20,d21=(rel*e0).sum(-1),(rel*e1).sum(-1)
        u=(d11*d20-d01*d21)/denom.clamp_min(1e-300);v=(d00*d21-d01*d20)/denom.clamp_min(1e-300)
        inside=(norm2>0)&(denom>0)&(u>=0)&(v>=0)&(u+v<=1)
        d2=torch.where(inside,h.square()*norm2,torch.inf)
        for start,end in ((a,b),(b,c),(c,a)):
            edge=end-start;weight=((point[:,None]-start)*edge).sum(-1)/edge.square().sum(-1).clamp_min(1e-300)
            nearest=start+weight.clamp(0,1)[...,None]*edge
            d2=torch.minimum(d2,(point[:,None]-nearest).square().sum(-1))
        out.append(d2.min(-1).values.clamp_min(0).sqrt())
    result=torch.cat(out).numpy()
    if not np.isfinite(result).all():raise ValueError('nonfinite surface comparison')
    return result


def surface_distance_report(source_geometry,compiled_asset,*,samples=1024,seed=20261031):
    import numpy as np
    source=_meshes(source_geometry['shapes']);cooked=_meshes([dict(m,kind='mesh') for m in compiled_asset['collision_meshes']])
    compiled_hash=digest(compiled_asset['collision_meshes'])
    if compiled_asset.get('native_compiled_surface_sha256',compiled_hash)!=compiled_hash:
        raise ValueError('compiled surface receipt hash differs from exported triangle arrays')
    points_source=_sample(source,samples,seed);points_cooked=_sample(cooked,samples,seed+1)
    def stats(values):return {'samples':len(values),'mean_m':float(np.mean(values)),'p50_m':float(np.quantile(values,.5)),
                              'p95_m':float(np.quantile(values,.95)),'p99_m':float(np.quantile(values,.99)),'sampled_max_m':float(np.max(values))}
    return {'schema':DISTANCE_SCHEMA,'source_geometry_sha256':digest(source_geometry),
            'source_exact_surface_sha256':digest(exact_surface(source_geometry['shapes'])),
            'native_compiled_surface_sha256':compiled_hash,
            'native_exact_surface_sha256':digest(exact_surface([dict(m,kind='mesh') for m in compiled_asset['collision_meshes']])),
            'source_to_cooked_component_surfaces':stats(unsigned_distances(points_source,cooked)),
            'cooked_component_surfaces_to_source':stats(unsigned_distances(points_cooked,source)),
            'seed':seed,'samples_each_direction':samples,'sampling':'deterministic_area_weighted_triangle_barycentric',
            'threshold':None,'observational_only':True,'native_acceptance_pass':False,
            'limitations':['Sampled diagnostic, not an exhaustive Hausdorff bound.',
                           'Uses the complete compiled component triangle surfaces, including surfaces inside overlapping components; it is not a Boolean-union boundary distance.',
                           'No PhysX cooked-geometry equivalence is claimed.']}


def _validate_cook(task,native_output,source_files,runtime_versions=None):
    from hoi.v1.native_acceptance import runtime_blockers
    report_path=Path(native_output)/'cooking.json';report=_json(report_path)
    if (Path(native_output)/'native.json').exists():
        raise ValueError('cooking-only adoption cannot mix a native acceptance report into the same output')
    if report.get('schema')!='hoi_native_geometry_cooking_report_v1' or report.get('physics_steps')!=0 or report.get('formal_training_started') is not False or report.get('training_ready') is not False or report.get('native_pair_calibration_complete') is not False or report.get('num_envs')!=1:
        raise ValueError('output is not a geometry-only cooking receipt')
    if report['source'].get('fixture_manifest_sha256')!=task['fixture']['sha256'] or report['source'].get('scope')!='curated_native_cooking_fixture_not_training':
        raise ValueError('cooking receipt source fixture changed')
    if runtime_blockers(report.get('native_runtime')):raise ValueError('cooking receipt lacks actual CUDA/Newton/MuJoCo runtime evidence')
    if runtime_versions is not None and report['native_runtime']['versions']!=runtime_versions:
        raise ValueError('native child runtime differs from planned package versions')
    if _json(Path(native_output)/'source.json')!=source_files:raise ValueError('native child executed different sources')
    export=Path(report['compiled_geometry_export']);_identity(export,report['compiled_geometry_export_sha256'])
    data=_json(export)
    if data.get('schema')!='hoi_v1_compiled_geometry_recipe_v1' or len(data.get('assets',[]))!=1:raise ValueError('invalid actual native geometry export')
    asset=data['assets'][0];fixture=_json(task['fixture']['path'])
    if fixture.get('repair_binding'):
        from hoi.data.native_repairs import fixture_repair_context
        binding,proposal,_,_,_,build,_=fixture_repair_context(fixture)
        from hoi.data.native_repairs import compare_candidate_export
        if asset.get('repair_import_comparison')!=compare_candidate_export(fixture['repair_binding']['path'],asset['collision_meshes']):
            raise ValueError('repaired native import lacks recomputed full convex-solid comparison')
        if (asset.get('repair_binding')!=fixture['repair_binding'] or report['source'].get('repair_binding')!=fixture['repair_binding'] or
            asset.get('cooking_input_geometry_sha256')!=build['object_geometry_sha256'] or asset.get('cooking_input_surface_sha256')!=build['exact_surface_sha256'] or
            asset.get('source_exact_surface_sha256')!=proposal['source_label_surface_sha256']):
            raise ValueError('repaired native cook lost approved original/input source bindings')
    elif asset.get('repair_binding') or report['source'].get('repair_binding'):
        raise ValueError('unrequested repair provenance in ordinary cooking output')
    if asset['source_native_asset_sha256']!=fixture['asset']['native_asset_sha256'] or not asset.get('native_convex_evidence'):
        raise ValueError('native export is not bound to the requested source asset')
    _identity(asset['native_asset'],asset['native_asset_sha256'])
    if asset.get('native_compiled_surface_sha256')!=digest(asset['collision_meshes']) or asset.get('object_geometry_sha256')!=task['source_geometry']['sha256']:
        raise ValueError('actual native arrays or source label geometry identity changed')
    if asset.get('source_exact_surface_sha256')!=report.get('source_exact_surface_sha256') or asset.get('native_compiled_surface_sha256')!=report.get('native_compiled_surface_sha256'):
        raise ValueError('native/source surface hashes disagree between receipts')
    return report,data


@contextmanager
def _task_lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0);stream.write(b'0');stream.flush();stream.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def valid_success(plan,task,attempt):
    receipt=_json(Path(attempt)/'SUCCESS.json')
    if receipt.get('schema')!=SUCCESS_SCHEMA or receipt.get('content_sha256')!=digest({k:v for k,v in receipt.items() if k!='content_sha256'}) or receipt.get('task_sha256')!=task['task_sha256'] or receipt.get('execution_sha256')!=plan['execution_sha256']:
        raise ValueError('success receipt task/executable identity mismatch')
    for binding in task['inputs']+receipt['artifacts']:verify_identity(binding)
    if task['kind']=='native_cook':_validate_cook(task,receipt.get('native_output',str(Path(attempt)/'native')),plan['execution']['source_files'],plan['execution'].get('runtime_versions'))
    return receipt


def _completed(plan,task):
    root=Path(plan['output'])/'tasks'/task['task_id']
    for attempt in sorted(root.glob('attempt_*'),reverse=True):
        try:return attempt,valid_success(plan,task,attempt)
        except (ValueError,KeyError,OSError,json.JSONDecodeError):continue
    return None


def run_task(plan_path,index):
    plan=read_plan(plan_path)
    if not 0<=index<len(plan['tasks']):raise ValueError('task index outside plan')
    task=plan['tasks'][index];root=Path(plan['output'])/'tasks'/task['task_id']
    with _task_lock(root/'task.lock'):
        for binding in task['inputs']:verify_identity(binding)
        previous=_completed(plan,task)
        if previous:
            print(json.dumps({'task':task['task_id'],'resumed_verified':True,'attempt':str(previous[0])}));return previous[1]
        number=1
        while (root/f'attempt_{number:04d}').exists():number+=1
        attempt=root/f'attempt_{number:04d}';attempt.mkdir();artifacts=[];execution=plan['execution']
        if task['kind']=='native_cook':
            command=[execution['python'],'-m','hoi.v1.native','--curated-cooking-fixture',task['fixture']['path'],
                     '--asset-id','0','--num-envs','1','--steps','0','--seed',str(execution['surface_seed']),
                     '--output',str(attempt/'native'),*execution['native_args']]
            _write(attempt/'command.json',{'argv':command,'cwd':execution['source'],'training_started':False})
            env=os.environ.copy();env['PYTHONPATH']=execution['source']+os.pathsep+env.get('PYTHONPATH','')
            with (attempt/'stdout.log').open('wb') as log:
                result=subprocess.run(command,cwd=execution['source'],env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
            if result.returncode:
                _write(attempt/'FAILED.json',{'task_sha256':task['task_sha256'],'returncode':result.returncode,'training_started':False})
                raise RuntimeError('native cooking failed; inspect '+str(attempt/'stdout.log'))
            report,data=_validate_cook(task,attempt/'native',execution['source_files'],execution.get('runtime_versions'))
            artifacts=[_identity(attempt/'native/cooking.json'),_identity(attempt/'native/source.json'),
                       _identity(report['compiled_geometry_export']),_identity(data['assets'][0]['native_asset'])]
            if not execution.get('defer_surface_report',False):
                import torch
                old_threads=torch.get_num_threads();torch.set_num_threads(1)
                try:distance=surface_distance_report(_json(task['source_geometry']['path']),data['assets'][0],samples=execution['surface_samples'],seed=execution['surface_seed'])
                finally:torch.set_num_threads(old_threads)
                _write(attempt/'surface_distances.json',distance)
                artifacts.append(_identity(attempt/'surface_distances.json'))
        else:
            geometry=_json(task['source_geometry']['path']);asset=task['asset_recipe']
            if digest(exact_surface(geometry['shapes']))!=digest(exact_surface([dict(m,kind='mesh') for m in asset['collision_meshes']])):
                raise ValueError('bootstrap native/source surface mismatch')
            _write(attempt/'bootstrap.json',{'schema':'hoi_bound_ultra_native_bootstrap_v1','task_sha256':task['task_sha256'],
                'native_asset':_identity(asset['native_asset'],asset['native_asset_sha256']),
                'status':'existing_bound_hulls_reused; standard_region16_native_gate_still_required',
                'physics_steps':0,'native_acceptance_pass':False,'training_started':False})
            artifacts=[_identity(attempt/'bootstrap.json')]
        receipt={'schema':SUCCESS_SCHEMA,'task_sha256':task['task_sha256'],'execution_sha256':plan['execution_sha256'],
                 'kind':task['kind'],'artifacts':artifacts,'surface_report_deferred':execution.get('defer_surface_report',False),'native_acceptance_pass':False,'training_started':False}
        receipt['content_sha256']=digest(receipt);_write(attempt/'SUCCESS.json',receipt)
        return receipt


def adopt_existing_output(plan_path,index,native_output):
    """Wrap a verified existing zero-step cook without changing its provenance."""
    plan=read_plan(plan_path)
    if not 0<=index<len(plan['tasks']):raise ValueError('task index outside plan')
    task=plan['tasks'][index]
    if task['kind']!='native_cook':raise ValueError('adoption is only for actual native cooking outputs')
    root=Path(plan['output'])/'tasks'/task['task_id'];native_output=Path(native_output).resolve()
    with _task_lock(root/'task.lock'):
        for binding in task['inputs']:verify_identity(binding)
        previous=_completed(plan,task)
        if previous:return previous[1]
        report,data=_validate_cook(task,native_output,plan['execution']['source_files'],plan['execution'].get('runtime_versions'))
        number=1
        while (root/f'attempt_{number:04d}').exists():number+=1
        attempt=root/f'attempt_{number:04d}';attempt.mkdir()
        _write(attempt/'adoption.json',{'schema':'hoi_existing_native_cooking_adoption_v1',
            'producer_native_output':str(native_output),'producer_source':plan['execution']['source'],
            'producer_source_files_sha256':digest(plan['execution']['source_files']),
            'producer_seed':report.get('seed'),'fixture':task['fixture'],
            'statement':'existing source/runtime/fixture/native bytes verified; no source rebinding and no new native execution',
            'native_acceptance_pass':False,'training_started':False})
        artifacts=[_identity(native_output/'cooking.json'),_identity(native_output/'source.json'),
                   _identity(report['compiled_geometry_export']),_identity(data['assets'][0]['native_asset']),_identity(attempt/'adoption.json')]
        execution=plan['execution']
        if not execution.get('defer_surface_report',False):
            import torch
            old_threads=torch.get_num_threads();torch.set_num_threads(1)
            try:distance=surface_distance_report(_json(task['source_geometry']['path']),data['assets'][0],samples=execution['surface_samples'],seed=execution['surface_seed'])
            finally:torch.set_num_threads(old_threads)
            _write(attempt/'surface_distances.json',distance);artifacts.append(_identity(attempt/'surface_distances.json'))
        receipt={'schema':SUCCESS_SCHEMA,'task_sha256':task['task_sha256'],'execution_sha256':plan['execution_sha256'],
                 'kind':'native_cook','native_output':str(native_output),'adopted_existing_output':True,
                 'artifacts':artifacts,'surface_report_deferred':execution.get('defer_surface_report',False),
                 'native_acceptance_pass':False,'training_started':False}
        receipt['content_sha256']=digest(receipt);_write(attempt/'SUCCESS.json',receipt)
        return receipt


def collect_exports(plan_path,output):
    plan=read_plan(plan_path);exports={};completed=[];missing=[]
    for task in plan['tasks']:
        result=_completed(plan,task)
        if result is None:missing.append(task['task_id']);continue
        attempt,receipt=result;completed.append({'task_id':task['task_id'],'success':_identity(attempt/'SUCCESS.json')})
        if task['kind']=='native_cook':exports[task['object_build_group_sha256']]=_json(Path(receipt.get('native_output',str(attempt/'native')))/'cooking.json')['compiled_geometry_export']
    if missing:raise ValueError('incomplete/unverified cooking tasks: '+', '.join(missing[:12]))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    _write(output/'exports.json',exports)
    _write(output/'collection.json',{'schema':'hoi_native_cooking_collection_v1','plan':_identity(plan_path),'tasks':completed,
            'exports':_identity(output/'exports.json'),'native_acceptance_pass':False,'training_started':False})
    return exports


@_planning_identity_cache()
def plan_fixtures(fixtures,source,python,native_args,output,**kwargs):
    """Plan completed source inputs while the full registry producer is running."""
    groups=[];bindings=[];seen=set()
    for path in fixtures:
        fixture=_json(path);asset=fixture['asset'];build=_json(asset['build_receipt']['path'])
        if fixture.get('content_sha256')!=digest({k:v for k,v in fixture.items() if k!='content_sha256'}):
            raise ValueError('partial fixture content identity changed')
        _fixture_inputs(path)
        repair=None
        if fixture.get('repair_binding'):
            from hoi.data.native_repairs import fixture_repair_context
            repair=fixture_repair_context(fixture)
        if (repair is None and build['object_geometry_sha256']!=asset['object_geometry_sha256']) or build['physical_profile']!=asset['physical_profile']:
            raise ValueError('partial source fixture/build mismatch')
        key=digest({'exact_surface_sha256':build['exact_surface_sha256'],'physical_profile':build['physical_profile'],
                    'inertia_mode':build.get('inertia_mode','explicit'),'collision_approximation':build['collision_approximation']})
        if repair is not None:key=repair[0]['object_build_group_sha256']
        if key in seen:raise ValueError('partial plan repeats a native object group')
        seen.add(key);label=asset['object_geometry_sha256']
        groups.append({'object_build_group_sha256':key,'members':[label],'cooking_fixture':_identity(path)})
        bindings.append({'dataset':fixture['source_record']['dataset'],'object_geometry_sha256':label})
    if not groups:raise ValueError('partial fixture plan is empty')
    registry={'schema':'hoi_curated_native_source_registry_v1','selected_geometry_identities':len(groups),
              'object_cooking_groups':groups,'bindings':bindings,'full_selection':False,
              'scope':'partial_ready_fixture_groups_only; full-registry coverage is not claimed'}
    registry['content_sha256']=digest(registry)
    output=Path(output).resolve();partial=output.with_name(output.name+'.partial_registry.json')
    _write(partial,registry)
    return make_plan(partial,source,python,native_args,output,**kwargs)


def collect_registry_exports(registry_path,plan_paths,output):
    registry=_json(registry_path)
    if registry.get('content_sha256')!=digest({k:v for k,v in registry.items() if k!='content_sha256'}):
        raise ValueError('full registry identity changed')
    from hoi.data.native_repairs import validate_derived_registry
    validate_derived_registry(registry)
    required={g['object_build_group_sha256']:g for g in registry['object_cooking_groups']}
    bootstrap={b['object_geometry_sha256'] for b in registry['bindings'] if b['dataset']=='ultra'}
    exports={};signatures={};covered_bootstrap=set();successes=[]
    for plan_path in plan_paths:
        plan=read_plan(plan_path)
        for task in plan['tasks']:
            completed=_completed(plan,task)
            if completed is None:continue
            attempt,receipt=completed
            if task['kind']=='bootstrap_verify':
                labels=set(task['covered_geometry_identities'])
                if not labels<=bootstrap:raise ValueError('bootstrap task is outside the full registry')
                covered_bootstrap.update(labels)
            else:
                key=task['object_build_group_sha256']
                if key not in required or not set(task['covered_geometry_identities'])<=set(required[key]['members']):
                    raise ValueError('partial cooking task does not match a full registry group')
                export=_json(Path(receipt.get('native_output',str(attempt/'native')))/'cooking.json')['compiled_geometry_export'];asset=_json(export)['assets'][0]
                signature=digest({'mesh':asset['native_compiled_surface_sha256'],'native_asset':asset['native_asset_sha256'],'physical_profile':asset['physical_profile']})
                if key in signatures and signatures[key]!=signature:
                    raise ValueError('conflicting actual native cooks; select one explicit plan for group '+key)
                signatures[key]=signature;exports.setdefault(key,export)
            successes.append({'plan':_identity(plan_path),'task_id':task['task_id'],'success':_identity(attempt/'SUCCESS.json')})
    if set(exports)!=set(required) or covered_bootstrap!=bootstrap:
        raise ValueError(f'full registry incomplete: missing {len(set(required)-set(exports))} cooked groups and {len(bootstrap-covered_bootstrap)} bootstrap identities')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);_write(output/'exports.json',exports)
    _write(output/'collection.json',{'schema':'hoi_full_registry_cooking_collection_v1','registry':_identity(registry_path),
        'exports':_identity(output/'exports.json'),'completed_tasks':successes,'selected_geometry_identities':registry['selected_geometry_identities'],
        'native_acceptance_pass':False,'training_started':False})
    return exports


def report_registry_surfaces(registry_path,exports_path,output,*,samples=1024,seed=20261031):
    import torch
    registry=_json(registry_path);exports=_json(exports_path)
    groups={g['object_build_group_sha256']:g for g in registry['object_cooking_groups']}
    if set(exports)!=set(groups):raise ValueError('surface report needs the full exported object-group mapping')
    bindings={b['object_geometry_sha256']:b for b in registry['bindings']}
    output=Path(output);output.mkdir(parents=True,exist_ok=False);reports=[]
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        for key,group in sorted(groups.items()):
            binding=bindings[group['representative_geometry_sha256']];record=binding['source_record']
            _identity(record['object_geometry_path'],record['object_geometry_sha256'],'canonical_json')
            asset=_json(exports[key])['assets'][0];_identity(asset['native_asset'],asset['native_asset_sha256'])
            report=surface_distance_report(_json(record['object_geometry_path']),asset,samples=samples,seed=seed)
            path=output/(key+'.json');_write(path,report);reports.append({'object_build_group_sha256':key,'report':_identity(path)})
    finally:torch.set_num_threads(old_threads)
    _write(output/'index.json',{'schema':'hoi_registry_surface_diagnostic_index_v1','registry':_identity(registry_path),
        'exports':_identity(exports_path),'reports':reports,'observational_only':True,'threshold':None,'native_acceptance_pass':False})
    return reports


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('plan');p.add_argument('--registry',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--python',type=Path,required=True);p.add_argument('--native-args',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--surface-samples',type=int,default=1024);p.add_argument('--seed',type=int,default=20261031);p.add_argument('--defer-surface-report',action='store_true')
    p=sub.add_parser('plan-fixtures');p.add_argument('--fixture',type=Path,action='append',required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--python',type=Path,required=True)
    p.add_argument('--native-args',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--surface-samples',type=int,default=1024);p.add_argument('--seed',type=int,default=20261031);p.add_argument('--defer-surface-report',action='store_true')
    p=sub.add_parser('run');p.add_argument('--plan',type=Path,required=True);p.add_argument('--task-index',type=int,required=True)
    p=sub.add_parser('adopt');p.add_argument('--plan',type=Path,required=True);p.add_argument('--task-index',type=int,required=True);p.add_argument('--native-output',type=Path,required=True)
    p=sub.add_parser('collect');p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('collect-registry');p.add_argument('--registry',type=Path,required=True);p.add_argument('--plan',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('report-surfaces');p.add_argument('--registry',type=Path,required=True);p.add_argument('--exports',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--surface-samples',type=int,default=1024);p.add_argument('--seed',type=int,default=20261031)
    args=parser.parse_args(argv)
    if args.command=='plan':
        plan=make_plan(args.registry,args.source,args.python,_json(args.native_args),args.output,surface_samples=args.surface_samples,seed=args.seed,defer_surface_report=args.defer_surface_report)
        print(json.dumps({'plan':str(args.output/'plan.json'),'tasks':len(plan['tasks']),'task_counts':plan['task_counts'],'identities':plan['geometry_identity_count']}))
    elif args.command=='plan-fixtures':
        plan=plan_fixtures(args.fixture,args.source,args.python,_json(args.native_args),args.output,surface_samples=args.surface_samples,seed=args.seed,defer_surface_report=args.defer_surface_report)
        print(json.dumps({'plan':str(args.output/'plan.json'),'tasks':len(plan['tasks']),'partial':True}))
    elif args.command=='collect-registry':
        exports=collect_registry_exports(args.registry,args.plan,args.output);print(json.dumps({'exports':str(args.output/'exports.json'),'groups':len(exports)}))
    elif args.command=='report-surfaces':
        reports=report_registry_surfaces(args.registry,args.exports,args.output,samples=args.surface_samples,seed=args.seed);print(json.dumps({'reports':len(reports),'native_acceptance_pass':False}))
    elif args.command=='run':run_task(args.plan,args.task_index)
    elif args.command=='adopt':adopt_existing_output(args.plan,args.task_index,args.native_output)
    else:
        exports=collect_exports(args.plan,args.output);print(json.dumps({'exports':str(args.output/'exports.json'),'native_cooking_groups':len(exports)}))


if __name__=='__main__':main()
