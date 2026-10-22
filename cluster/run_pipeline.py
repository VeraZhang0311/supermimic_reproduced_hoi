"""Run the single-flag self-collision experiment: native gate, full-sized warm-up gate, then 14000 updates."""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time

ROOT=Path('/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision')
RUN=ROOT/'logs/lafan1_selfcollision_seed0'
SMOKE=ROOT/('logs/smoke_full501-'+os.environ.get('SLURM_JOB_ID','local'))
TRACKER_SITE='/mnt/projects/ll/tianzizh/SuperMimic_HOI/envs/tracking-py312/lib/python3.12/site-packages'
PORTABLE=ROOT/'tools/python-portable/cpython-3.12.4-linux-x86_64-gnu/bin/python3.12'
def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write_json(path,value):
    path=Path(path); temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); temporary.replace(path)
def logged(command,path,timeout):
    print(json.dumps({'utc':stamp(),'command':command,'log':str(path)}),flush=True)
    with Path(path).open('w') as stream:
        subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=timeout)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def upstream_command(script,*arguments):
    return [sys.executable,str(ROOT/'cluster/compat_launch.py'),script,*arguments]
def evaluation_complete():
    p=RUN/'eval.csv'
    if not p.exists(): return False
    try:
        rows=list(csv.DictReader(p.open()))
        return any(int(x['iteration'])==14000 and math.isfinite(float(x['norm_return'])) for x in rows)
    except (ValueError,KeyError): return False
def terminate(process):
    if process is None or process.poll() is not None: return
    process.terminate()
    try: process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill(); process.wait(timeout=15)

def validate_identity():
    sys.path.insert(0,str(ROOT/'cluster'))
    from validate_experiment import validate
    return validate()

def main():
    import torch
    assert torch.cuda.device_count()==1,'Exactly one allocated GPU required'
    assert torch.cuda.is_available()
    result=(torch.randn(32,32,device='cuda') @ torch.randn(32,32,device='cuda')).float()
    assert torch.isfinite(result).all()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        result=torch.randn(32,32,device='cuda') @ torch.randn(32,32,device='cuda')
    assert torch.isfinite(result).all()
    hardware={'name':torch.cuda.get_device_name(0),'memory_bytes':torch.cuda.get_device_properties(0).total_memory,
        'torch':torch.__version__,'torch_cuda_build':torch.version.cuda,'visible_gpu_count':torch.cuda.device_count(),
        'slurm_job_id':os.environ['SLURM_JOB_ID'],'cpus':os.environ['SLURM_CPUS_PER_TASK'],
        'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()}
    write_json(ROOT/'provenance/gpu-hardware.json',hardware)
    write_json(ROOT/f"provenance/gpu-hardware-{os.environ['SLURM_JOB_ID']}.json",hardware)
    validate_identity()
    for setting in ('off', 'on'):
        logged(upstream_command('cluster/check_selfcollision.py','--setting',setting),
               ROOT/f"logs/collision-gate-{setting}-{os.environ['SLURM_JOB_ID']}.log",1800)
    off=json.loads((ROOT/f"provenance/collision-off-{os.environ['SLURM_JOB_ID']}.json").read_text())
    on=json.loads((ROOT/f"provenance/collision-on-{os.environ['SLURM_JOB_ID']}.json").read_text())
    assert off['newton_allowed_self_shape_pairs']==0,off
    assert off['mujoco_allowed_nonadjacent_geom_pairs']==0,off
    assert on['newton_allowed_self_shape_pairs']>0,on
    assert on['mujoco_allowed_nonadjacent_geom_pairs']>0,on
    assert on['cross_body_shape_pairs']==off['cross_body_shape_pairs']
    gate_log=ROOT/f"logs/native-fk-gate-{os.environ['SLURM_JOB_ID']}.log"
    logged(upstream_command('tools/test_env.py','--num_envs','64'),gate_log,1800)
    text=gate_log.read_text()
    errors=dict(re.findall(r'roundtrip (\w+): max err ([0-9.eE+-]+)',text))
    assert set(errors)=={'p','R','v','w','q','qd'},errors
    assert all(float(x)<2e-4 for x in errors.values()),errors
    fk=float(re.search(r'FK pos err ([0-9.eE+-]+)',text)[1])
    assert math.isfinite(fk) and fk<1e-4,fk
    assert 'motion: 40 clips' in text
    assert not SMOKE.exists(),'Existing smoke directory preserved; inspect before retry'
    logged(upstream_command('supermimic/train.py','--iterations','501','--ckpt_every','501','--log_every','1','--log_dir',str(SMOKE)),ROOT/f"logs/smoke501-{os.environ['SLURM_JOB_ID']}.stdout.log",3600)
    smoke_rows=list(csv.DictReader((SMOKE/'train_log.csv').open()))
    assert smoke_rows and int(smoke_rows[-1]['iteration'])==501
    assert all(math.isfinite(float(row[key])) for row in smoke_rows for key in ('wm_loss','wm_err8'))
    assert all(math.isfinite(float(smoke_rows[-1][key])) for key in ('pi_loss','v_loss','pi_gradnorm','imag_survival'))
    # The final checkpoint includes the one post-warm-up BPTT/critic update at iteration501.
    checkpoint=torch.load(SMOKE/'ckpt_000501.pt',map_location='cpu',weights_only=False)
    assert checkpoint['iteration']==501
    assert all(torch.isfinite(v).all() for group in ('wm','pi') for v in checkpoint[group].values() if torch.is_floating_point(v))
    del checkpoint
    torch.cuda.empty_cache()
    write_json(ROOT/'provenance/gate-passed.json',{'utc':stamp(),'native_fk_max_m':fk,'roundtrip':errors,'full_environment_count':4096,'wm_batch':2048,'policy_batch':4096,'warmup_iterations':500,'bptt_updates':1,'policy_convergence_validated':False})
    assert not RUN.exists(),'Preserve existing formal run; no overwrite'
    RUN.mkdir()
    source={str(path.relative_to(ROOT)):sha(path) for folder in ('supermimic','tools') for path in sorted((ROOT/folder).glob('*.py'))}
    manifest={'stage':'formal','created_utc':stamp(),'upstream_commit':'a1508d42c9d1699a52e8233ab16d08c488441ab9',
        'experiment':'selfcollision_on','comparison_group':'LAFAN1-seed0-selfcollision',
        'physical_self_collisions':True,'extra_self_collision_reward_penalty':False,
        'baseline_root':'/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi',
        'scientific_change':'Only G1 articulation enabled_self_collisions=False -> True; same reward/termination and all training hyperparameters.',
        'collision_backend_verification':{'off':off,'on':on},
        'source_identity':validate_identity(),
        'isaaclab_commit':'46d6cc3d7b115906fad12efd372fbf69f4e3d3ed','hardware':hardware,
        'source_sha256':source,'data_manifest_sha256':sha(ROOT/'provenance/download-manifest.json'),
        'environment_deviation':json.loads((ROOT/'provenance/environment-deviation.json').read_text()),
        'parameters':{'num_envs':4096,'iterations':14000,'buffer_steps':1000,'utd':0.4,'horizon':32,'goal_frames':10,'hist':10,'wm_window':8,'wm_batches':10,'wm_batch':2048,'pi_batch':4096,'lr':0.0002,'gamma':0.99,'lam':0.95,'action_noise':0.1,'critic_epochs':5,'critic_minibatches':4,'warmup_steps':64,'wm_only_iters':500,'lr_decay_iters':250000,'ckpt_every':2000,'log_every':50,'seed':0,'amp':'bf16'},
        'acceptance':{'mean_normalized_return':0.82,'failed_clips':0,'clip_count':40},
        'timing':'Unchanged upstream wall_h includes training-loop compile and concurrent same-GPU evaluation; initial data/simulator/replay warm-up, setup and this separate smoke gate are outside its clock.',
        'checkpoint_uploads':False,
        'graph_compatibility_launcher_sha256':sha(ROOT/'cluster/compat_launch.py'),
        'graph_compatibility_receipts':[json.loads(p.read_text()) for p in sorted((ROOT/'provenance').glob(f"graph-compat-{os.environ['SLURM_JOB_ID']}-*.json"))]}
    write_json(RUN/'manifest.json',manifest)
    validate_identity()
    tracker_env=dict(os.environ,PYTHONPATH=TRACKER_SITE,WANDB_CACHE_DIR=str(ROOT/'.cache/wandb'),WANDB_DATA_DIR=str(ROOT/'.cache/wandb-data'),WANDB_DIR=str(ROOT/'logs'))
    track_log=(ROOT/'logs/tracker.stdout.log').open('w')
    tracker=subprocess.Popen([str(PORTABLE),str(ROOT/'cluster/track_csv.py')],cwd=ROOT,env=tracker_env,stdout=track_log,stderr=subprocess.STDOUT)
    train_log=(RUN/'stdout.log').open('w')
    eval_log=(RUN/'eval.stdout.log').open('w')
    train=None; evaluator=None; status='failed'; error=None
    try:
        train=subprocess.Popen(upstream_command('supermimic/train.py','--iterations','14000','--log_dir',str(RUN)),cwd=ROOT,stdout=train_log,stderr=subprocess.STDOUT)
        evaluator=subprocess.Popen(upstream_command('supermimic/evaluate.py','--run_dir',str(RUN),'--watch'),cwd=ROOT,stdout=eval_log,stderr=subprocess.STDOUT)
        write_json(RUN/'processes.json',{'training_pid':train.pid,'evaluation_pid':evaluator.pid,'tracker_pid':tracker.pid,'slurm_job_id':os.environ['SLURM_JOB_ID']})
        attempts=1
        while train.poll() is None:
            if evaluator.poll() is not None and attempts<3:
                time.sleep(10)
                evaluator=subprocess.Popen(upstream_command('supermimic/evaluate.py','--run_dir',str(RUN),'--watch'),cwd=ROOT,stdout=eval_log,stderr=subprocess.STDOUT)
                attempts+=1
            time.sleep(10)
        if train.returncode: raise RuntimeError(f'Upstream training exited {train.returncode}')
        deadline=time.monotonic()+3600
        while not evaluation_complete():
            if evaluator.poll() is not None:
                raise RuntimeError(f'Evaluator exited {evaluator.returncode}; full evaluation incomplete')
            if time.monotonic()>deadline: raise TimeoutError('Final full40 evaluation not produced within one hour')
            time.sleep(10)
        rows=list(csv.DictReader((RUN/'eval.csv').open()))
        result=next(x for x in rows if int(x['iteration'])==14000)
        actual=json.loads((RUN/'args.json').read_text())
        for key,value in manifest['parameters'].items():
            assert actual[key]==value,(key,actual[key],value)
        for path,digest in source.items(): assert sha(ROOT/path)==digest,path
        write_json(RUN/'acceptance.json',{'utc':stamp(),'iteration':14000,'norm_return':float(result['norm_return']),
            'n_failed':int(result['n_failed']),'completion_fraction':1-int(result['n_failed'])/40,
            'train_wall_h':float(result['train_wall_h']),
            'target_reproduced':float(result['norm_return'])>=0.82 and int(result['n_failed'])==0,
            'hardware_identical_to_author':False,'source_unchanged_since_manifest':True,'upstream_core_diff':'one self-collision flag'})
        status='completed'
    except BaseException as exc:
        error=repr(exc)
        raise
    finally:
        terminate(train); terminate(evaluator)
        write_json(RUN/'pipeline.json',{'status':status,'error':error,'utc':stamp(),'training_exit_code':None if train is None else train.returncode})
        try: tracker.wait(timeout=150)
        except subprocess.TimeoutExpired: terminate(tracker)
        track_log.close(); train_log.close(); eval_log.close()
    assert tracker.returncode==0,'W&B sync failed; raw CSV and checkpoints preserved'

if __name__=='__main__':
    main()
