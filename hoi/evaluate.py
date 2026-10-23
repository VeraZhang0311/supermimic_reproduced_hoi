"""Frozen baseline policy in the original G1 simulator plus dynamic HOI objects.

Full reference duration, no reset after failures. Author return is accumulated only
until its first termination. The existing HOI pose diagnostics score the full clip.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from isaaclab.app import add_launcher_args, launch_simulation
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--prepared',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--asset',required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--gate-steps',type=int,default=0)
p.add_argument('--robot-usd',type=Path,default=None,help='Independent hand-collision overlay; original robot otherwise')
add_launcher_args(p)
args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=False)

import numpy as np
import torch
import warp as wp
import mujoco_warp
import mujoco
from newton import ModelFlags
from pxr import Usd,UsdPhysics,Gf
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
from cluster.compat_launch import install_graph_compatibility
import supermimic.env as env_module
from supermimic.env import TrackingSim,make_sim,TrackingSceneCfg
from supermimic.kinematics import G1Kinematics
from supermimic.motion import MotionLib,CONTROL_DT
from supermimic.mdp import MDP,MAX_STEP_REWARD,OBS_DIM
from supermimic.robot import G1_CFG,G1_JOINT_NAMES,TRACKED_BODIES,default_joint_pos_list
from supermimic.rotations import mat_to_quat
from supermimic.networks import Policy
import supermimic.networks as networks
from hoi.ultra_kinematics import UltraKinematics
from hoi.ultra_metrics import clip_metrics
from hoi.hand_collision import compiled_audit

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name,value):
    path=args.output/name
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

manifest=json.loads((args.prepared/'manifest.json').read_text())
asset=manifest['assets'][args.asset]
cases=[case for case in manifest['cases'] if case['object_asset']==args.asset]
meta=asset['metadata']
assert cases and OBS_DIM==99 and G1_CFG.spawn.articulation_props.enabled_self_collisions is False
assert list(G1_JOINT_NAMES)==manifest['robot_joint_names']==meta['dof_names']
assert sha(asset['usd'])==asset['usd_sha256']
checkpoint_hash=sha(args.checkpoint)

# Independent flattened USD: the shared HOI assets are never changed or cooked in place.
# Newton uses its supported CoACD convex decomposition rather than PhysX's VHACD.
stage=Usd.Stage.Open(asset['usd'])
local_usd=args.output/'object.usda'
stage.Flatten().Export(str(local_usd))
stage=Usd.Stage.Open(str(local_usd))
bodies=[prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
assert len(bodies)==1,[str(x.GetPath()) for x in bodies]
physical=meta['object_properties'][0]
mass=UsdPhysics.MassAPI.Apply(bodies[0])
mass.CreateMassAttr(float(physical['mass']))
mass.CreateCenterOfMassAttr(Gf.Vec3f(*physical['com']))
diagonal,axes=np.linalg.eigh(np.asarray(physical['inertia']))
assert np.all(diagonal>0)
if np.linalg.det(axes)<0:axes[:,0]*=-1
from scipy.spatial.transform import Rotation
quat=Rotation.from_matrix(axes).as_quat()
mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float,diagonal)))
mass.CreatePrincipalAxesAttr(Gf.Quatf(float(quat[3]),Gf.Vec3f(*map(float,quat[:3]))))
colliders=[]
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(True)
        if prim.GetTypeName()=='Mesh':
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexDecomposition')
        colliders.append(str(prim.GetPath()))
assert colliders,'Object must contain actual collision geometry'
stage.GetRootLayer().Save()

@configclass
class HOISceneCfg(TrackingSceneCfg):
    object=RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Object',
        spawn=sim_utils.UsdFileCfg(usd_path=str(local_usd),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,linear_damping=0.,angular_damping=0.)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.,4.,2.)))

if args.robot_usd:
    assert args.robot_usd.is_file()
    # Instantiate before editing: configclass fields are instance fields. Only
    # the robot spawn path changes; PD/inertia/joints/self-collision stay intact.
    def patched_scene_factory(**kwargs):
        config=HOISceneCfg(**kwargs)
        config.robot.spawn.usd_path=str(args.robot_usd)
        return config
    env_module.TrackingSceneCfg=patched_scene_factory
else:
    env_module.TrackingSceneCfg=HOISceneCfg
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32=True
wp.init()
assert torch.cuda.device_count()==1
dev='cuda:0'
kin=G1Kinematics(dev)
files=[str(args.prepared/'motions'/f"{c['id']}.csv") for c in cases]
for c,file in zip(cases,files):assert sha(file)==c['csv_sha256']
motion=MotionLib(files,kin,dev,src_fps=60.)
assert motion.clip_len.tolist()==[c['resampled_frames'] for c in cases]
mdp=MDP(kin,motion,torch.tensor(default_joint_pos_list(),device=dev))
refs=[torch.tensor(np.load(args.prepared/'references'/f"{c['id']}.npz")['reference'],device=dev) for c in cases]
fk=UltraKinematics(args.prepared/'metadata/ultra_g1.urdf',meta['body_names'],meta['dof_names'],dev)
ck=torch.load(args.checkpoint,map_location=dev,weights_only=False)
assert ck['iteration']==14000
K,Hh=ck['args']['goal_frames'],ck['args']['hist']
assert K==10 and Hh==10
networks.AMP_DTYPE=torch.bfloat16 if ck['args'].get('amp')=='bf16' else None
pi=Policy(OBS_DIM,73*K,hist=Hh).to(dev)
pi.load_state_dict(ck['pi']);pi.eval()
before={k:v.detach().cpu().clone() for k,v in pi.state_dict().items()}
started=time.time()
sim_cfg=make_sim(dev,args)

with launch_simulation(sim_cfg,args),torch.no_grad():
    env=TrackingSim(sim_cfg,len(cases),motion,mdp,eval_mode=True)
    obj=env.scene['object']
    expected_mass=torch.full_like(obj.data.body_mass.torch,float(physical['mass']))
    expected_com=torch.tensor(physical['com'],device=dev,dtype=torch.float32).reshape(1,1,3).expand(len(cases),1,3).contiguous()
    expected_inertia=torch.tensor(physical['inertia'],device=dev,dtype=torch.float32).reshape(1,1,9).expand(len(cases),1,9).contiguous()
    obj.set_masses_index(masses=expected_mass)
    obj.set_coms_index(coms=expected_com)
    obj.set_inertias_index(inertias=expected_inertia)
    env.sim.backend_manager.add_model_change(ModelFlags.BODY_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES)
    # Baseline's setup deliberately changes robot/ground only. Set object friction explicitly.
    model=env.sim.backend_manager.get_model()
    material=wp.to_torch(obj._root_view.get_attribute('shape_material_mu',model))
    material[:]=.6
    env.sim.backend_manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)
    mass_error=float((obj.data.body_mass.torch-expected_mass).abs().max())
    com_error=float((obj.data.body_com_pose_b.torch[...,:3]-expected_com).abs().max())
    inertia_error=float((obj.data.body_inertia.torch.reshape_as(expected_inertia)-expected_inertia).abs().max())
    assert mass_error<1e-6 and com_error<1e-6 and inertia_error<1e-6,(mass_error,com_error,inertia_error)
    ids=torch.arange(len(cases),device=dev)
    env.reset(ids,motion.clip_start.clone())
    object_start=torch.stack([reference[0,71:84] for reference in refs])
    obj.write_root_link_pose_to_sim_index(root_pose=object_start[:,:7].contiguous())
    obj.write_root_com_velocity_to_sim_index(root_velocity=object_start[:,7:13].contiguous())
    env.scene.reset(ids);env.refresh()
    actual=obj.data.root_link_pose_w.torch
    assert float((actual[:,:3]-object_start[:,:3]).abs().max())<1e-5
    assert float((actual[:,3:7]*object_start[:,3:7]).sum(-1).abs().min())>1-1e-5
    # Detect a possible URDF convention mismatch before assigning the existing 39-body PHC metric.
    S=env.state();wxyz=mat_to_quat(S[1]);root=torch.cat((S[0],wxyz[:,[1,2,3,0]],S[2],S[3]),-1)
    virtual=fk.forward(root,S[4],S[5],root_velocity_is_com=False)
    common=torch.tensor([meta['body_names'].index(name) for name in TRACKED_BODIES],device=dev)
    fk_error=float((virtual['body_pos'][:,common]-kin.forward(*S)[0]).abs().max())
    assert fk_error<2e-4,('HOI versus trained G1 kinematics',fk_error)
    mj=env.sim.backend_manager._solver.mj_model
    body_names=[mujoco.mj_id2name(mj,mujoco.mjtObj.mjOBJ_BODY,i) or '' for i in range(mj.nbody)]
    object_bodies={i for i,name in enumerate(body_names) if '_Object' in name or '/Object' in name}
    assert object_bodies,body_names
    assert all(int(mj.body_jntnum[i])>0 and int(mj.jnt_type[int(mj.body_jntadr[i])])==int(mujoco.mjtJoint.mjJNT_FREE) for i in object_bodies)
    excluded=set(map(int,mj.exclude_signature))
    object_robot_pairs=0
    for g1 in range(mj.ngeom):
        b1=int(mj.geom_bodyid[g1])
        if b1 not in object_bodies:continue
        for g2 in range(mj.ngeom):
            b2=int(mj.geom_bodyid[g2])
            if b2==0 or b2 in object_bodies:continue
            signature=(min(b1,b2)<<16)+max(b1,b2)
            masks=(int(mj.geom_contype[g1])&int(mj.geom_conaffinity[g2]))|(int(mj.geom_contype[g2])&int(mj.geom_conaffinity[g1]))
            if signature not in excluded and masks:object_robot_pairs+=1
    assert object_robot_pairs>0,'Compiled MuJoCo object/robot collision pairs disabled'
    hand_verification=compiled_audit(mj,enabled=bool(args.robot_usd))
    native={'object_asset':args.asset,'body_count':obj.num_bodies,'colliders':colliders,
        'object_mass_max_abs_error':mass_error,'object_com_max_abs_error':com_error,
        'object_inertia_max_abs_error':inertia_error,'virtual39_vs_physical14_fk_error_m':fk_error,
        'physical_objects':True,'object_friction':.6,'policy_observation_dim':99,'goal_dim':730,
        'object_observation_channels':0,'robot_self_collision':False,
        'compiled_mujoco_dynamic_object_bodies':len(object_bodies),
        'compiled_mujoco_allowed_object_robot_geom_pairs':object_robot_pairs,
        'gpu':torch.cuda.get_device_name(0),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
        'convex_decomposition':'Newton CoACD; different engine/cooking from frozen PhysX VHACD',
        'robot_physics':'Author G1/PD unchanged; only original HOI rubber-hand collision hulls added' if args.robot_usd else 'Original author G1/PD with no rubber-hand colliders',
        'hand_collision_verification':hand_verification,
        'robot_usd_override_sha256':sha(args.robot_usd) if args.robot_usd else None}
    write('native.json',native)
    lengths=[int(x)-1 for x in motion.clip_len.tolist()]
    max_steps=min(max(lengths),args.gate_steps) if args.gate_steps else max(lengths)
    records={k:[] for k in ('root_p','root_quat','root_v','root_w','q','qd','body_pos','body_rot','body_vel','body_ang_vel','object_state','physical_body_pose')}
    controls={k:[] for k in ('action','policy_observation','goal','reward','supermimic_termination')}
    hist=None
    first_failure=torch.full((len(cases),),-1,device=dev,dtype=torch.long)
    ret=torch.zeros(len(cases),device=dev)
    alive=torch.ones(len(cases),device=dev,dtype=torch.bool)
    capacity_mask=int(mujoco_warp.OverflowType.NEFC | mujoco_warp.OverflowType.NJMAX_NNZ |
        mujoco_warp.OverflowType.BROADPHASE | mujoco_warp.OverflowType.NARROWPHASE |
        mujoco_warp.OverflowType.CCD | mujoco_warp.OverflowType.HFIELD |
        mujoco_warp.OverflowType.NVMAX | mujoco_warp.OverflowType.EPA_HORIZON)
    def record_state():
        S=env.state();quat=mat_to_quat(S[1])
        root=torch.cat((S[0],quat[:,[1,2,3,0]],S[2],S[3]),-1)
        body=fk.forward(root,S[4],S[5],root_velocity_is_com=False)
        values={'root_p':S[0],'root_quat':quat,'root_v':S[2],'root_w':S[3],'q':S[4],'qd':S[5],**body,
                'object_state':torch.cat((obj.data.root_link_pose_w.torch,obj.data.root_com_vel_w.torch),-1),
                'physical_body_pose':env.robot.data.body_link_pose_w.torch}
        for key,value in values.items():
            assert torch.isfinite(value).all(),('non-finite state',key)
            records[key].append(value.detach().cpu().clone())
    record_state()
    for step in range(max_steps):
        env.f=torch.minimum(motion.clip_start+step,motion.clip_end-1)
        S=env.state()
        obs=mdp.obs(S,env.a_prev,env.xi,env.f)
        goal=motion.goal(env.f,K)
        assert obs.shape==(len(cases),99) and goal.shape==(len(cases),730)
        hist=obs[:,None].repeat(1,Hh,1) if hist is None else torch.cat((hist[:,1:],obs[:,None]),1)
        action=pi(hist,goal)
        assert torch.isfinite(action).all()
        out=env.step(action)
        overflow=wp.to_torch(env.sim.backend_manager._solver.mjw_data.overflow)
        if bool(((overflow&capacity_mask)!=0).any()):
            write('physics-overflow.json',{'step':step+1,'overflow_flags':overflow.cpu().tolist(),
                'nconmax':128,'njmax':500,'action':'stop without publishing invalid evaluation metrics'})
            Sbad=env.state()
            np.savez_compressed(args.output/'overflow-state.npz',root_p=Sbad[0].cpu().numpy(),
                q=Sbad[4].cpu().numpy(),action=action.cpu().numpy(),
                object_pose=obj.data.root_link_pose_w.torch.cpu().numpy())
            raise RuntimeError('MuJoCo-Warp collision/constraint capacity overflow; physical evidence preserved')
        active=step<torch.tensor(lengths,device=dev)
        ret+=torch.where(alive&active,out['r'],0.)
        new_fail=out['term']&(first_failure<0)&active
        first_failure=torch.where(new_fail,step+1,first_failure)
        alive&=~out['term']
        for key,value in {'action':action,'policy_observation':obs,'goal':goal,'reward':out['r'],'supermimic_termination':out['term']}.items():
            controls[key].append(value.detach().cpu().clone())
        record_state()
        if (step+1)%100==0:print(json.dumps({'asset':args.asset,'steps':step+1,'total':max_steps}),flush=True)
    assert all(torch.equal(before[key],value.detach().cpu()) for key,value in pi.state_dict().items()),'Policy weights or normalization updated'
    results=[]
    for i,case in enumerate(cases):
        count=min(lengths[i],max_steps)
        states={key:torch.stack(value)[:count+1,i] for key,value in records.items()}
        control={key:torch.stack(value)[:count,i] for key,value in controls.items()}
        reference=refs[i][:count+1].cpu()
        packed={**states,'reference':reference,'time':torch.arange(count+1,dtype=torch.float64)*.02,
                'action':torch.cat((torch.zeros(1,29),control['action']),0)}
        metric=clip_metrics(packed,.02,exclude_initial=1)
        target=args.output/f"{case['id']}.npz"
        np.savez_compressed(target,**{k:v.numpy() for k,v in states.items()},
            **{k:v.numpy() for k,v in control.items()},reference=reference.numpy(),time=packed['time'].numpy(),
            body_names=np.array(meta['body_names']),joint_names=np.array(G1_JOINT_NAMES),
            physical_body_names=np.array(env.robot.body_names))
        results.append({'id':case['id'],'object_asset':args.asset,'source_group':case['source_group'],
            'full_reference_completed':count==lengths[i], 'control_steps':count,
            'first_supermimic_failure_step':int(first_failure[i]),
            'supermimic_norm_return':float(ret[i]/(lengths[i]*MAX_STEP_REWARD)),
            'rollout':target.name,'rollout_sha256':sha(target),**metric})
    assert sha(args.checkpoint)==checkpoint_hash
    write('evaluation.json',{'protocol':manifest['protocol'],'checkpoint':str(args.checkpoint),
        'checkpoint_iteration':14000,'checkpoint_sha256':checkpoint_hash,'case_count':len(results),
        'gate_only':bool(args.gate_steps),'optimizer_updates':0,'normalization_updates':0,
        'no_failure_reset':True,'physical_objects':True,'object_observation_channels':0,
        'action_timing':'action/obs/goal[t] acts on state[t] to produce state[t+1]; states and reference have T+1 frames',
        'root_quaternion':'wxyz','object_and_body_quaternion':'xyzw','dt':.02,
        'elapsed_seconds':time.time()-started,'native':native,'cases':results})
    print(json.dumps({'asset':args.asset,'completed':len(results),'gate_only':bool(args.gate_steps)}),flush=True)
