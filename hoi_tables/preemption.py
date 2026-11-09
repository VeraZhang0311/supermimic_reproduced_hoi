"""Preemption-only Ultra continuation, outside the immutable science runtime.

An afterany CPU handler submits a fresh two-L40 attempt only for PREEMPTED.
The original run remains the logical metrics/checkpoint/evaluation namespace.
Replayed metric iterations stay in attempt logs and are not uploaded twice.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

SCHEMA='hoi_ultra_preemption_recovery_v1'
ASSETS=tuple(sorted(f'{o}_{s}_{s}_{s}' for o in ('largebox','smallbox','plasticbox','suitcase') for s in ('080','100')))
RUNTIME='/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128'
TELEMETRY='/mnt/projects/ll/tianzizh/gmt_l40_20261105_01/telemetry_venv'
COMPAT='/mnt/projects/ll/tianzizh/hoi_residual_ppo_20261106_01/source_smoke08/tools'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def atomic(p,value):
    p=Path(p);tmp=p.with_name(f'.{p.name}.{os.getpid()}.tmp')
    with tmp.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def utc():return datetime.now(timezone.utc).isoformat()
def can_recover(state):return state=='PREEMPTED'

def reopen_evaluation_ledger(state):
    """Reopen liveness only; keep every bound request/result and active job."""
    for key in ('finished','parent_training_exitcode','missing_milestones','failed_milestones','finished_utc'):state.pop(key,None)
    return state

@contextmanager
def locked(root):
    import fcntl
    root=Path(root);root.mkdir(exist_ok=True,parents=True)
    with (root/'controller.lock').open('a+b') as f:
        fcntl.flock(f,fcntl.LOCK_EX);yield

def settings(root):
    v=read(Path(root)/'policy.json')
    if v.get('schema')!=SCHEMA or v.get('profile')!='ultra' or v.get('assets')!=list(ASSETS):raise ValueError('Recovery is Ultra-only')
    from hoi_tables.unified_monitor import owning_run
    owner=owning_run(v['training_root'],v['run_id'])
    if owner['project']!='hoiretarget-tables':raise ValueError('Wrong owning project')
    if sha(Path(v['source'])/'SOURCE_MANIFEST.json')!=v['source_sha256']:raise ValueError('Frozen source identity changed')
    if sha(v['fallback_bundle'])!=v['fallback_bundle_sha256']:raise ValueError('Original recovery bundle changed')
    manifest=read(Path(v['source'])/'SOURCE_MANIFEST.json')
    for name,digest in manifest['files'].items():
        if sha(Path(v['source'])/name)!=digest:raise ValueError('Frozen source changed: '+name)
    for name,digest in v['helper_sha256'].items():
        if sha(Path(v['helpers'])/name)!=digest:raise ValueError('Recovery helper changed: '+name)
    return v

class Scheduler:
    def call(self,args):
        return subprocess.run(args,capture_output=True,text=True,check=True,timeout=60).stdout.strip()
    def status(self,job):
        from hoi_tables.eval_monitor import Slurm
        return Slurm().status(str(job))
    def find(self,name,created):
        from hoi_tables.eval_monitor import Slurm
        return Slurm().find(name,created)
    def submit(self,args):
        env=os.environ.copy()
        for key in ('CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'):env.pop(key,None)
        result=subprocess.run(['sbatch','--parsable',*args],env=env,capture_output=True,text=True,check=True,timeout=60).stdout.strip().split(';')[0]
        if not result.isdigit():raise ValueError('Invalid scheduler submission reply')
        return result
    def stop_logger(self,job):
        state=self.status(job)['state']
        if state=='PENDING':self.call(['scancel',str(job)])
        elif state in ('RUNNING','CONFIGURING','COMPLETING'):
            self.call(['scancel','--signal=TERM','--full',str(job)])
    def release(self,job):
        if self.status(job)['state']!='PENDING':return
        detail=self.call(['scontrol','show','job','-o',str(job)])
        if 'Reason=JobHeldUser' in detail:self.call(['scontrol','release',str(job)])

def initialize(root,campaign,train_job,logger_job,helpers):
    root,campaign,helpers=Path(root).resolve(),Path(campaign).resolve(),Path(helpers).resolve()
    if root.exists():raise FileExistsError('Use a fresh recovery controller')
    training=campaign/'training';metadata=read(training/'run_metadata.json')
    if metadata.get('experiment',{}).get('profile')!='ultra' or metadata.get('gpus')!=2 or metadata['experiment']['assets']!=list(ASSETS):
        raise ValueError('Only the existing two-L40 Ultra run can be armed')
    from hoi_tables.unified_monitor import owning_run
    owner=owning_run(training);root.mkdir(parents=True)
    source=campaign/'source'
    value={'schema':SCHEMA,'profile':'ultra','assets':list(ASSETS),'run_id':owner['id'],
        'campaign':str(campaign),'training_root':str(training),'source':str(source),'source_sha256':sha(source/'SOURCE_MANIFEST.json'),
        'eval_control':str(campaign/'evaluation'),'fallback_bundle':str((campaign/'input/resume_bundle.json').resolve()),
        'fallback_bundle_sha256':sha(campaign/'input/resume_bundle.json'),
        'helpers':str(helpers),'helper_sha256':{p.name:sha(p) for p in helpers.iterdir() if p.suffix in ('.py','.sh')},
        'trigger_states':['PREEMPTED'],'gpu_type':'L40','gpus':2,'tables_auto_resume':False,'created_utc':utc()}
    atomic(root/'policy.json',value)
    atomic(root/'state.json',{'schema':SCHEMA,'current':{'attempt':0,'train_job':str(train_job),'logger_job':str(logger_job),
        'actual_training_root':str(training),'control':str(campaign/'unified_logging')},'handlers':{},'handled':{}})
    settings(root)
    return value

def submit_once(record,scheduler,args):
    """Resolve a lost acceptance reply by unique name; never blindly retry."""
    if record.get('job_id'):return record['job_id']
    found=scheduler.find(record['name'],record['created_utc'])
    if len(found)>1:raise RuntimeError('Duplicate recovery jobs detected')
    if found:record['job_id']=str(found[0]);return record['job_id']
    if record.get('submission_started'):
        raise RuntimeError('Unresolved scheduler acceptance; refuse duplicate submission')
    record['submission_started']=utc()
    return None

def ensure_handler(root,parent,scheduler=None):
    scheduler=scheduler or Scheduler();root=Path(root)
    with locked(root):
        v=settings(root);s=read(root/'state.json');key=str(parent)
        intent=s.setdefault('handlers',{}).setdefault(key,{'name':f"ultra-watch-{v['run_id']}-{key}",'created_utc':utc()})
        job=submit_once(intent,scheduler,[])
        if job:return job
        atomic(root/'state.json',s)
        status=scheduler.status(parent)['state']
        args=['--job-name='+intent['name'],'--output='+str(root/f'handler-{parent}-%j.log')]
        if status in ('PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED'):
            args+=['--dependency=afterany:'+str(parent)]
        args +=[str(Path(v['helpers'])/'handler.sbatch.sh'),str(root),str(parent)]
        intent['job_id']=scheduler.submit(args);atomic(root/'state.json',s)
        return intent['job_id']

def metric_rows(path):
    """Only complete newline-terminated records; reject internal corruption."""
    path=Path(path)
    if not path.exists():return [],b''
    raw=path.read_bytes();boundary=raw.rfind(b'\n')+1;rows=[];last=None
    for line in raw[:boundary].splitlines(keepends=True):
        if not line.strip():continue
        row=json.loads(line);i=row['iteration']
        if type(i) is not int or i<1 or (last is not None and i!=last+1):raise ValueError('Noncontiguous rank metric history')
        rows.append((i,line));last=i
    return rows,raw[boundary:]

def align_metrics(training_root,archive):
    """After trainer/logger termination, preserve and trim only unpaired tails."""
    training_root,archive=Path(training_root),Path(archive);archive.mkdir(parents=True,exist_ok=True)
    allrows=[metric_rows(training_root/f'rank_{i:02d}/metrics.jsonl') for i in range(8)]
    if all(rows for rows,tail in allrows) and len({rows[0][0] for rows,tail in allrows})!=1:
        raise ValueError('Rank histories lost an earlier common prefix')
    sets=[{i for i,_ in rows} for rows,tail in allrows];common=set.intersection(*sets)
    frontier=max(common,default=0)
    for rank,(rows,tail) in enumerate(allrows):
        keep=[line for i,line in rows if i<=frontier];removed=b''.join(line for i,line in rows if i>frontier)+tail
        if removed:
            p=training_root/f'rank_{rank:02d}/metrics.jsonl'
            backup=archive/f'rank_{rank:02d}_unpaired_tail.jsonl'
            if backup.exists() and backup.read_bytes()!=removed:raise ValueError('Unpaired metric archive already differs')
            backup.write_bytes(removed);tmp=p.with_suffix('.recovery.tmp');tmp.write_bytes(b''.join(keep));tmp.replace(p)
    return frontier

def complete_sets(training_root):
    from hoi_tables.eval_monitor import complete_checkpoint
    root=Path(training_root);ranks=[{int(p.stem.split('_')[-1]):p for p in (root/f'rank_{i:02d}').glob('checkpoint_*.pt') if complete_checkpoint(p)} for i in range(8)]
    common=set.intersection(*(set(r) for r in ranks))
    return {i:[r[i] for r in ranks] for i in sorted(common) if i>=500 and i%500==0}

class Bridge:
    """Publish paired new metrics and complete checkpoint sets, preserving history."""
    def __init__(self,actual,logical):
        self.actual,self.logical=Path(actual),Path(logical)
        rows=[metric_rows(self.logical/f'rank_{i:02d}/metrics.jsonl')[0] for i in range(8)]
        last=[r[-1][0] if r else 0 for r in rows]
        if len(set(last))!=1:raise ValueError('Logical metrics need a paired frontier before bridging')
        self.frontier=last[0];self.offsets=[0]*8;self.last=[None]*8;self.pending=[{} for _ in range(8)]
        self.published=set(complete_sets(self.logical))
    def flush(self):
        for rank in range(8):
            p=self.actual/f'rank_{rank:02d}/metrics.jsonl'
            if not p.exists():continue
            if p.stat().st_size<self.offsets[rank]:raise ValueError('Attempt metrics were truncated')
            with p.open('rb') as f:f.seek(self.offsets[rank]);data=f.read()
            boundary=data.rfind(b'\n')+1;self.offsets[rank]+=boundary
            for line in data[:boundary].splitlines(keepends=True):
                if not line.strip():continue
                row=json.loads(line);iteration=row['iteration']
                if type(iteration) is not int or iteration<1 or (self.last[rank] is not None and iteration!=self.last[rank]+1):raise ValueError('Noncontiguous attempt metrics')
                self.last[rank]=iteration
                if iteration>self.frontier:self.pending[rank][iteration]=line
        while all(self.pending):
            heads=[min(p) for p in self.pending]
            if len(set(heads))!=1:raise ValueError('Attempt rank iterations diverged')
            iteration=heads[0]
            if self.frontier and iteration!=self.frontier+1:raise ValueError('Bridge would skip a logical iteration')
            for rank in range(8):
                p=self.logical/f'rank_{rank:02d}/metrics.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
                with p.open('ab') as f:f.write(self.pending[rank].pop(iteration));f.flush();os.fsync(f.fileno())
            self.frontier=iteration
        from hoi_tables.eval_monitor import complete_checkpoint
        candidates={int(p.stem.split('_')[-1]) for p in (self.actual/'rank_00').glob('checkpoint_*.pt')}-self.published
        for iteration in sorted(candidates):
            paths=[self.actual/f'rank_{rank:02d}/checkpoint_{iteration:06d}.pt' for rank in range(8)]
            if iteration%500 or not all(complete_checkpoint(p) for p in paths):continue
            # A previously published checkpoint is immutable, including on a
            # replay after preemption. New attempts start after that checkpoint.
            targets=[self.logical/f'rank_{rank:02d}/checkpoint_{iteration:06d}.pt' for rank in range(8)]
            if all(p.exists() for p in targets):
                if any(p.resolve()!=q.resolve() and sha(p)!=sha(q) for p,q in zip(targets,paths)):
                    raise ValueError('Bridge would replace a checkpoint with different weights')
                self.published.add(iteration)
                continue
            for target,source in zip(targets,paths):
                if target.exists():
                    if sha(target)!=sha(source):raise ValueError('Partial checkpoint conflicts with continuation')
                else:target.symlink_to(source.resolve())
            self.published.add(iteration)
        return self.frontier

def audit_bundle(v,training_root,destination):
    """Audit the latest whole eight-rank snapshot; never initialize random weights."""
    import torch
    from hoi_tables.capacity import _hash_state
    from hoi_tables.evaluate import state_dict_hash
    from hoi_tables.resume import bundle,load_rank,_semantic
    sets=complete_sets(training_root)
    if not sets:
        fallback=Path(v['fallback_bundle']);parent=bundle(fallback)
        for rank in range(8):
            cp=torch.load(parent['records'][rank]['path'],map_location='cpu',weights_only=True)
            current=cp['identity'].copy();current['solver_health']={'schema':'hoi_native_solver_health_v1','ls_iterations':200,'fatal_on_all_native_flags':True,'clear_or_mask_flags':False,'physical_contact_or_motor_changes':False}
            load_rank(fallback,rank,current)
        return str(fallback),parent['iteration']
    iteration=max(sets);records=[];models=[];optimizers=[];science=[];counts=[]
    def finite(value):
        if isinstance(value,torch.Tensor):return not value.is_floating_point() or bool(torch.isfinite(value).all())
        if isinstance(value,dict):return all(finite(x) for x in value.values())
        if isinstance(value,(list,tuple)):return all(finite(x) for x in value)
        if type(value) is float:return math.isfinite(value)
        return True
    for rank,path in enumerate(sets[iteration]):
        cp=torch.load(path,map_location='cpu',weights_only=True);identity=cp['identity']
        if cp.get('schema')!='hoi_tables_checkpoint_v1' or cp.get('iteration')!=iteration or identity.get('asset')!=ASSETS[rank] or identity.get('asset_id')!=rank or identity.get('world_size')!=8:
            raise ValueError('Checkpoint rank/schema/iteration mismatch')
        for field in ('policy','wm','critic','optimizers','sampler','contact_positive','contact_negative'):
            if field not in cp or not finite(cp[field]):raise ValueError('Incomplete/nonfinite recovery learner state: '+field)
        models.append(tuple(state_dict_hash(cp[k]) for k in ('policy','wm','critic')))
        optimizers.append(_hash_state(cp['optimizers']))
        counts.append(_hash_state((cp['contact_positive'],cp['contact_negative'])))
        semantic=json.loads(_semantic(identity));semantic.pop('asset');semantic.pop('asset_id');science.append(json.dumps(semantic,sort_keys=True))
        if set(cp.get('rng',{}))!={'torch','cuda'} or 'sampler' not in cp:raise ValueError('Incomplete training recovery state')
        records.append({'rank':rank,'asset':ASSETS[rank],'path':str(path.resolve()),'sha256':sha(path),'bytes':path.stat().st_size})
        del cp
    if len(set(models))!=1 or len(set(optimizers))!=1 or len(set(science))!=1 or len(set(counts))!=1:raise ValueError('Eight-rank learner or scientific identity is not synchronized')
    value={'schema':'hoi_resume_bundle_v1','passed':True,'world_size':8,'assets':list(ASSETS),'iteration':iteration,
        'parent_run_id':v['run_id'],'checkpoint_root':str(training_root),'parent_source':v['source'],
        'parent_source_manifest_sha256':v['source_sha256'],'records':records,'replay_and_simulator_serialized':False}
    atomic(destination,value)
    # Reuse the production protected-source/identity loader on all exact bytes.
    for rank,record in enumerate(records):
        cp=torch.load(record['path'],map_location='cpu',weights_only=True)
        load_rank(destination,rank,cp['identity'])
    return str(destination),iteration

def handle(root,parent,scheduler=None):
    root=Path(root);scheduler=scheduler or Scheduler()
    with locked(root):previous=read(root/'state.json').get('handled',{}).get(str(parent))
    if previous:
        if previous['status']=='queued':
            ensure_handler(root,previous['job_id'],scheduler);scheduler.release(previous['job_id'])
        return previous
    # UNKNOWN accounting is transient after a scheduler transition, not a
    # reason to restart. Bound this wait and fail closed.
    status=scheduler.status(parent)
    for _ in range(24):
        if status['state']!='UNKNOWN':break
        time.sleep(5);status=scheduler.status(parent)
    with locked(root):
        v=settings(root);s=read(root/'state.json');key=str(parent)
        if key in s.get('handled',{}):return s['handled'][key]
        if not can_recover(status['state']):
            result={'status':'not_restarted','reason':status['state'],'time':utc()}
            s.setdefault('handled',{})[key]=result;atomic(root/'state.json',s);return result
        current=s['current']
        if str(current['train_job'])!=key:raise ValueError('Preempted job is no longer the active recovery parent')
        # SIGTERM is caught by the existing unified writer and causes its
        # documented graceful handoff, including SDK flush and lock release.
        logger=current.get('logger_job')
        if logger:
            scheduler.stop_logger(logger)
            for _ in range(120):
                if scheduler.status(logger)['state'] not in ('RUNNING','PENDING','COMPLETING','CONFIGURING'):break
                time.sleep(5)
            else:raise RuntimeError('Owning-run writer did not finish its graceful handoff')
        number=int(current['attempt'])+1;attempt=root/f'attempt_{number:03d}';attempt.mkdir(exist_ok=True)
        if current.get('actual_training_root')!=v['training_root']:
            # Complete any interrupted bridge append first, then drain the
            # stopped attempt. Never mutate logs while its trainer is live.
            align_metrics(v['training_root'],attempt/'unpaired_before_drain')
            Bridge(current['actual_training_root'],v['training_root']).flush()
        frontier=align_metrics(v['training_root'],attempt/'unpaired_metrics')
        selected,iteration=audit_bundle(v,v['training_root'],attempt/'resume_bundle.json')
        # Incomplete newer milestones cannot be mixed with a new attempt's
        # weights. Preserve them outside the logical checkpoint namespace.
        for rank in range(8):
            for p in (Path(v['training_root'])/f'rank_{rank:02d}').glob('checkpoint_*.pt'):
                if int(p.stem.split('_')[-1])>iteration:
                    orphan=attempt/'orphan_checkpoints'/f'rank_{rank:02d}'/p.name;orphan.parent.mkdir(parents=True,exist_ok=True);p.rename(orphan)
        marker=Path(v['training_root'])/'training_exitcode'
        if marker.exists():marker.rename(attempt/'previous_training_exitcode')
        config=read(Path(v['campaign'])/'experiment.json');config['paths']['resume_bundle']=selected
        atomic(attempt/'experiment.json',config)
        detail={'attempt':number,'previous_train_job':key,'resume_iteration':iteration,'metrics_frontier':frontier,
            'actual_training_root':str(attempt/'training'),'control':str(attempt/'logging'),
            'experiment':str(attempt/'experiment.json'),'run_id':v['run_id'],'created_utc':utc()}
        atomic(attempt/'attempt.json',detail)
        intent=s.setdefault('submissions',{}).setdefault(key,{'name':f"ultra-auto-{v['run_id']}-{number:03d}",'created_utc':utc()})
        job=submit_once(intent,scheduler,[])
        if job is None:
            atomic(root/'state.json',s)
            # Hold until both the durable attempt identity and its successor
            # handler exist; a fast allocation cannot outrun its metadata.
            job=scheduler.submit(['--hold','--job-name='+intent['name'],'--output='+str(attempt/'gpu-%j.log'),
                str(Path(v['helpers'])/'continue.sbatch.sh'),str(root),str(attempt)])
            intent['job_id']=job
        detail['train_job']=job;atomic(attempt/'attempt.json',detail)
        s['current']={**detail,'logger_job':None};result={'status':'queued','job_id':job,'attempt':number,'resume_iteration':iteration}
        s.setdefault('handled',{})[key]=result;atomic(root/'state.json',s)
    ensure_handler(root,job,scheduler)
    scheduler.release(job)
    return result

def logger(root,attempt):
    import signal
    from hoi_tables import unified_monitor as unified
    from hoi_tables.hoi80_monitor import Supervisor
    from hoi_tables.telemetry import _sdk
    root,attempt=Path(root),Path(attempt);v=settings(root);a=read(attempt/'attempt.json');stopped={'yes':False};active={}
    previous_signals={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    signal.signal(signal.SIGTERM,lambda *x:stopped.update(yes=True))
    signal.signal(signal.SIGINT,lambda *x:stopped.update(yes=True))
    class ContinuingSupervisor(Supervisor):
        def __init__(self,**kwargs):
            super().__init__(**kwargs)
            reopen_evaluation_ledger(self.state)
            self.save()
    module=_sdk(None)
    class SDK:
        def __getattr__(self,key):return getattr(module,key)
        def init(self,**kwargs):
            if kwargs.get('id')!=v['run_id'] or kwargs.get('resume')!='must':raise ValueError('Recovery cannot register another run')
            run=module.init(**kwargs)
            active['run']=run
            run.summary.update({'recovery/attempt':a['attempt'],'recovery/checkpoint_iteration':a['resume_iteration'],
                'recovery/previous_job':a['previous_train_job'],'recovery/current_job':a['train_job'],
                'recovery/replayed_metrics_retained_in_attempt_logs':True,'recovery/metrics_rejoin_iteration':a['metrics_frontier']+1})
            return run
    def stopping():
        if 'run' in active:active['run'].summary['recovery/phase']='training' if (attempt/'capacity_passed.json').exists() else 'validating'
        return stopped['yes']
    try:
        return unified.monitor(a['control'],v['training_root'],assets=ASSETS,eval_control=v['eval_control'],
            expected_run_id=v['run_id'],train_job=a['train_job'],wandb_module=SDK(),stop_requested=stopping,
            supervisor_factory=ContinuingSupervisor,supervisor_kwargs={'root':v['eval_control'],'campaign':v['campaign'],
            'training_root':v['training_root'],'source_root':v['source'],'training_source':v['source'],
            'parent_run_id':v['run_id'],'train_job':a['train_job']})
    finally:
        for signum,handler in previous_signals.items():signal.signal(signum,handler)

def run(root,attempt):
    root,attempt=Path(root),Path(attempt);v=settings(root);a=read(attempt/'attempt.json');job=os.environ['SLURM_JOB_ID']
    if str(a['train_job'])!=job:raise ValueError('Attempt allocated under another job identity')
    ensure_handler(root,job)
    import torch
    if torch.cuda.device_count()!=2:raise ValueError('Ultra continuation requires exactly2 visible L40 GPUs')
    del torch
    command=[RUNTIME+'/bin/python','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=8',
        '-m','hoi_tables.train','--experiment-config',a['experiment']]
    scheduler=Scheduler();logger_job=scheduler.submit(['--job-name='+f"ultra-log-{v['run_id']}-{a['attempt']:03d}",
        '--output='+str(attempt/'logger-%j.log'),str(Path(v['helpers'])/'logger.sbatch.sh'),str(root),str(attempt)])
    with locked(root):
        s=read(root/'state.json');s['current']['logger_job']=logger_job;atomic(root/'state.json',s)
    for _ in range(600):
        if (Path(a['control'])/'unified_error.json').exists():raise RuntimeError('Recovery owning-run writer failed')
        if (Path(a['control'])/'unified_ready.json').exists():break
        time.sleep(1)
    else:raise RuntimeError('Owning-run logger did not become ready')
    # Keep all original guard checks. Every new allocation exercises the full
    # restored-policy backward/capacity gate before any optimizer updates.
    # Resume the existing SDK owner first so validation failures are visible.
    with (attempt/'capacity.log').open('w') as log:
        subprocess.run([*command,'--capacity-only','--output',str(attempt/'capacity')],stdout=log,stderr=subprocess.STDOUT,check=True)
    reports=[read(attempt/f'capacity/rank_{i:02d}/capacity_report.json') for i in range(8)]
    if not all(r['passed'] and r['production_configuration'] and r['optimizer_updates']==0 for r in reports):raise RuntimeError('Recovery GPU capacity gate failed')
    atomic(attempt/'capacity_passed.json',{'passed':True,'workers':8})
    bridge=Bridge(a['actual_training_root'],v['training_root']);stopped={'yes':False}
    signal.signal(signal.SIGTERM,lambda *x:stopped.update(yes=True))
    signal.signal(signal.SIGINT,lambda *x:stopped.update(yes=True))
    with (attempt/'training.log').open('w') as log:
        child=subprocess.Popen([*command,'--output',a['actual_training_root']],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while child.poll() is None:
                bridge.flush()
                if stopped['yes']:os.killpg(child.pid,signal.SIGTERM);break
                if (Path(a['control'])/'unified_error.json').exists():raise RuntimeError('Owning-run telemetry stopped during continuation')
                time.sleep(3)
            try:code=child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
            bridge.flush()
        except BaseException:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
            raise
    code=code if code>=0 else 128-code
    (Path(v['training_root'])/'training_exitcode').write_text(str(code)+'\n')
    atomic(attempt/'training_result.json',{'exitcode':code,'time':utc()})
    return code

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('arm','handle','run','logger'));p.add_argument('--root',type=Path,required=True)
    p.add_argument('--parent');p.add_argument('--attempt',type=Path);a=p.parse_args()
    try:
        if a.mode=='arm':print(ensure_handler(a.root,a.parent));return 0
        if a.mode=='handle':print(json.dumps(handle(a.root,a.parent)));return 0
        if a.mode=='logger':return logger(a.root,a.attempt)
        return run(a.root,a.attempt)
    except Exception as error:
        atomic(a.root/f"error_{os.environ.get('SLURM_JOB_ID',str(os.getpid()))}.json",{'mode':a.mode,'error_type':type(error).__name__,'error':str(error),'time':utc()})
        if a.mode=='run':
            v=settings(a.root);(Path(v['training_root'])/'training_exitcode').write_text('1\n')
        raise

if __name__=='__main__':raise SystemExit(main())
