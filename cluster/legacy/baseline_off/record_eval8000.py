"""Record an immutable checkpoint with the author's unmodified full-40 evaluation function."""
from pathlib import Path
import argparse
import ast
import datetime
import hashlib
import json
import os
import sys
import time

ROOT=Path('/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi')
sys.path.insert(0,str(ROOT))
from isaaclab.app import add_launcher_args,launch_simulation
parser=argparse.ArgumentParser()
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
add_launcher_args(parser)
args=parser.parse_args()
OUT=args.output
OUT.mkdir(exist_ok=False,parents=True)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(name,value): (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

checkpoint_sha=sha(args.checkpoint)
source=ROOT/'supermimic/evaluate.py'
tree=ast.parse(source.read_text())
evaluate_node=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='evaluate')
write('recording.json',{'status':'initializing','started_utc':stamp(),'checkpoint':str(args.checkpoint),
    'checkpoint_sha256':checkpoint_sha,'evaluation_source_sha256':sha(source),
    'evaluation_function':'Compiled directly from upstream evaluate.py AST without edits',
    'optimizer_updates':0,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
    'slurm_step_id':os.environ.get('SLURM_STEP_ID'),
    'concurrent_with_training':True,
    'timing_note':'Extra read-only evaluation in the existing GPU allocation may slow concurrent training during this recording window; exclude this window from clean speed comparisons.'})

import numpy as np
import torch
import warp as wp
import mujoco_warp
from cluster.compat_launch import install_graph_compatibility
from supermimic.env import TrackingSim,make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MAX_STEP_REWARD,OBS_DIM,MDP
from supermimic.motion import MotionLib
from supermimic.networks import Policy
import supermimic.networks as networks
from supermimic.robot import G1_CFG,default_joint_pos_list

assert G1_CFG.spawn.articulation_props.enabled_self_collisions is False
torch.backends.cuda.matmul.allow_tf32=True
wp.init()
install_graph_compatibility(mujoco_warp,wp.is_conditional_graph_supported())
assert torch.cuda.device_count()==1
free,total=torch.cuda.mem_get_info()
assert free>6*2**30, 'Insufficient free GPU memory for additional evaluation; preserve training'
dev='cuda:0'
kin=G1Kinematics(dev)
motion=MotionLib.from_dir(str(ROOT/'data/lafan1_g1'),kin,dev)
assert len(motion.names)==40
mdp=MDP(kin,motion,torch.tensor(default_joint_pos_list(),device=dev))
C=len(motion.names)
namespace=globals()
exec(compile(ast.Module(body=[evaluate_node],type_ignores=[]),str(source),'exec'),namespace)
sim_cfg=make_sim(dev,args)
started=stamp(); t0=time.monotonic()
with launch_simulation(sim_cfg,args):
    env=TrackingSim(sim_cfg,C,motion,mdp,eval_mode=True)
    checkpoint=torch.load(args.checkpoint,map_location=dev,weights_only=False)
    assert checkpoint['iteration']==8000
    print(json.dumps({'stage':'full40_recording_started','utc':stamp(),'checkpoint_iteration':8000,
        'gpu':torch.cuda.get_device_name(0),'free_gpu_gib_before_setup':free/2**30}),flush=True)
    result=evaluate(env,checkpoint,str(OUT/'full40_rollout.npz'))
    write('replayed_eval.json',result)
assert sha(args.checkpoint)==checkpoint_sha
original=json.loads(args.checkpoint.with_name(args.checkpoint.stem+'_eval.json').read_text())
write('original_eval.json',original)
chosen=['dance1_subject1','jumps1_subject2','fight1_subject3','walk3_subject1','fallAndGetUp1_subject4']
with np.load(OUT/'full40_rollout.npz',allow_pickle=False) as z:
    names=z['names'].tolist()
    ids=[names.index(name) for name in chosen]
    # Keep original reference frame indices; only the live robot batch is reduced to five.
    selected={key:z[key][:,ids] for key in ('root_p','root_quat','q','f','alive')}
    selected.update({key:z[key] for key in ('ref_root_p','ref_root_R','ref_q')})
    selected.update(names=np.array(chosen),clip_start=z['clip_start'][ids],clip_len=z['clip_len'][ids])
    np.savez_compressed(OUT/'selected5_rollout.npz',**selected)
    clips=[]
    for i,name in enumerate(chosen):
        live=selected['alive'][:,i]
        length=int(live.sum())
        clips.append({'name':name,'recorded_alive_steps':length,'recorded_seconds':length*.02,
            'reference_seconds':(int(selected['clip_len'][i])-1)*.02,
            'failed_in_original_eval':name in original['failed'],
            'failed_in_recording':name in result['failed'],
            'original_norm_return':original['per_clip'][name],
            'replayed_norm_return':result['per_clip'][name]})
report={'status':'actual_policy_rollouts_recorded','started_utc':started,'finished_utc':stamp(),
    'elapsed_seconds':time.monotonic()-t0,'checkpoint':str(args.checkpoint),'checkpoint_iteration':8000,
    'checkpoint_sha256':checkpoint_sha,'evaluation_source_sha256':sha(source),
    'evaluation_protocol':'Author full40 evaluation function, first frame, friction1, no DR or observation/action noise; no optimizer updates',
    'simulator':'Isaac Lab Newton MuJoCo-Warp, self-collision OFF; identical graph compatibility fallback',
    'optimizer_updates':0,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
    'slurm_step_id':os.environ.get('SLURM_STEP_ID'),'concurrent_with_training':True,
    'timing_note':'Concurrent training speed during this extra evaluation window is not a clean throughput benchmark.',
    'original_full40':{k:original[k] for k in ('norm_return','n_failed','failed')},
    'replayed_full40':{k:result[k] for k in ('norm_return','n_failed','failed')},
    'return_difference':result['norm_return']-original['norm_return'],
    'same_failed_clips':result['failed']==original['failed'],'clips':clips,
    'files_sha256':{name:sha(OUT/name) for name in ('full40_rollout.npz','selected5_rollout.npz','original_eval.json','replayed_eval.json')}}
write('recording.json',report)
print(json.dumps(report,indent=2),flush=True)
