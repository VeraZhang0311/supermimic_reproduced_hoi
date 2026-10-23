"""Inspect effective USD, Newton filter pairs and compiled MuJoCo collision masks.

The OFF diagnostic runs in a separate process; formal and smoke training keep the ON source default.
Adjacent/welded body exclusions remain as specified by the importer and MuJoCo.
"""
from pathlib import Path
import argparse
import datetime
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from isaaclab.app import add_launcher_args, launch_simulation
parser=argparse.ArgumentParser()
parser.add_argument('--setting',choices=('off','on'),required=True)
add_launcher_args(parser)
args=parser.parse_args()

import numpy as np
import torch
import mujoco
from newton import ShapeFlags
from isaaclab.sim.utils.stage import get_current_stage
from supermimic.env import TrackingSim, TrackingSceneCfg, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.motion import MotionLib
from supermimic.mdp import MDP
from supermimic.robot import G1_CFG, default_joint_pos_list

expected=args.setting=='on'
assert G1_CFG.spawn.articulation_props.enabled_self_collisions is True
# configclass stores mutable fields in instance factories, not class attributes.
assert TrackingSceneCfg(num_envs=1,env_spacing=0.0,replicate_physics=True).robot.spawn.articulation_props.enabled_self_collisions is True
if not expected:
    import supermimic.env as env_module
    original_scene_config=env_module.TrackingSceneCfg
    def diagnostic_scene_config(*arguments,**keywords):
        config=original_scene_config(*arguments,**keywords)
        config.robot.spawn.articulation_props.enabled_self_collisions=False
        return config
    env_module.TrackingSceneCfg=diagnostic_scene_config
dev='cuda:0'
torch.manual_seed(0)
kin=G1Kinematics(dev)
motion=MotionLib.from_dir(str(ROOT/'data/lafan1_g1'),kin,dev)
mdp=MDP(kin,motion,torch.tensor(default_joint_pos_list(),device=dev))
sim_cfg=make_sim(dev,args)
with launch_simulation(sim_cfg,args):
    env=TrackingSim(sim_cfg,1,motion,mdp,eval_mode=True)
    attrs=[]
    for prim in get_current_stage().Traverse():
        if str(prim.GetPath()).startswith('/World/envs/env_0/Robot'):
            attr=prim.GetAttribute('newton:selfCollisionEnabled')
            if attr and attr.HasAuthoredValueOpinion(): attrs.append({'prim':str(prim.GetPath()),'value':bool(attr.Get())})
    assert attrs and all(item['value']==expected for item in attrs),attrs
    manager=env.sim.backend_manager
    model=manager.get_model()
    bodies=model.shape_body.numpy()
    flags=model.shape_flags.numpy()
    shapes=np.flatnonzero((bodies>=0)&((flags&int(ShapeFlags.COLLIDE_SHAPES))!=0))
    a,b=np.triu_indices(len(shapes),1)
    pairs=np.column_stack((shapes[a],shapes[b]))
    pairs=pairs[bodies[pairs[:,0]]!=bodies[pairs[:,1]]]
    assert len(pairs)>0
    allowed=~model.shape_collision_filter_mask(pairs)
    n_newton=int(allowed.sum())
    assert (n_newton>0)==expected,(expected,n_newton,len(pairs))
    # Read the actual compiled MuJoCo model, not only the Newton source filter.
    mj=manager._solver.mj_model
    excluded={int(signature) for signature in mj.exclude_signature}
    samples=[]
    n_mujoco=0
    for g1 in range(mj.ngeom):
        for g2 in range(g1+1,mj.ngeom):
            b1,b2=int(mj.geom_bodyid[g1]),int(mj.geom_bodyid[g2])
            # body 0 is the static world; exclude same/welded/parent-child cases.
            if b1==0 or b2==0 or b1==b2: continue
            w1,w2=int(mj.body_weldid[b1]),int(mj.body_weldid[b2])
            if w1==w2 or int(mj.body_parentid[w1])==w2 or int(mj.body_parentid[w2])==w1: continue
            signature=(min(b1,b2)<<16)+max(b1,b2)
            if signature in excluded: continue
            masks=(int(mj.geom_contype[g1])&int(mj.geom_conaffinity[g2]))|(int(mj.geom_contype[g2])&int(mj.geom_conaffinity[g1]))
            if not masks: continue
            n_mujoco+=1
            if len(samples)<5: samples.append([mujoco.mj_id2name(mj,mujoco.mjtObj.mjOBJ_BODY,b1),mujoco.mj_id2name(mj,mujoco.mjtObj.mjOBJ_BODY,b2)])
    assert (n_mujoco>0)==expected,(expected,n_mujoco)
    for _ in range(16):
        action=(motion.q[env.f+1]-env.default_q)/env.alpha
        output=env.step(action)
        assert torch.isfinite(output['r']).all()
        assert all(torch.isfinite(part).all() for part in env.state())
    receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'slurm_job_id':os.environ['SLURM_JOB_ID'],'diagnostic_setting':args.setting,
        'source_training_default':True,'usd_self_collision_attributes':attrs,
        'cross_body_shape_pairs':len(pairs),'newton_allowed_self_shape_pairs':n_newton,
        'mujoco_allowed_nonadjacent_geom_pairs':n_mujoco,'allowed_body_pair_examples':samples,
        'finite_native_steps':16,'num_envs':1}
    (ROOT/f"provenance/collision-{args.setting}-{os.environ['SLURM_JOB_ID']}.json").write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
