"""CPU source-binding/reset and contact-isolation contracts; no GPU claims."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import torch
from hoi.v1.scene_supports import (bind_scene_supports, validate_bound_support, validate_native_scene_supports,
    NativeSceneSupportBank, SOURCE_REVISION, ENV_CFG_SHA256, SUPPORT_ROLES, compiled_support_role)
from hoi.v1.contacts import aggregate_pairs, aggregate_ground_pairs, BODY_ROLES, OBJECT, GROUND, CHANNELS
from hoi.v1.native import HOITrackingSim


def source(sid="table1", positions=None):
    if positions is None:
        positions=[[1.,2.,.3],[1.1,2.,.3],[1.2,2.,.3],[1.3,2.,.3],[1.4,2.,.3]]
    parked=all(p[2]<-5 for p in positions)
    uses=any(abs(v)>1e-6 for p in positions for v in p) and not parked
    return {"partner_id":sid,"kind":"cuboid","kinematic":True,"source_present":True,
        "source_field":sid+"_pos_w","size_m":[.3,.25,.01],"position":positions,
        "quaternion":[[1.,0.,0.,0.]]*len(positions),"parked_below_scene":parked,
        "official_table1_uses":uses if sid=="table1" else None,
        "recipe":{"repository":"Ingrid789/OmniContact","revision":SOURCE_REVISION,"sha256":ENV_CFG_SHA256,
                  "path":"source/omnicontact/omnicontact/tasks/omnicontact/env_cfg.py"}}


def bind(raw):
    return bind_scene_supports(raw,torch.arange(5,dtype=torch.float64)/100,torch.arange(3,dtype=torch.float64)/50)


class SceneSupportTests(unittest.TestCase):
    def test_source_to_target_positions_no_new_time_grid(self):
        raw=[source(),source("table2")];out=bind(raw)
        self.assertEqual([r['support_id'] for r in out],['table1','table2'])
        torch.testing.assert_close(out[0]['position'][:,0],torch.tensor([1.,1.2,1.4],dtype=torch.float64))
        self.assertEqual(out[0]['pose_mode'],'sampled_reference_frame_held_for_episode')
        validate_bound_support(out[0],raw[0],3)
        self.assertEqual(raw[0]['position'][1][0],1.1)

    def test_source_interpolation_90_to50_no_frame_index_alias(self):
        raw=source(positions=[[float(i),0.,1.] for i in range(5)])
        actual=bind_scene_supports([raw],torch.arange(5,dtype=torch.float64)/90,torch.arange(3,dtype=torch.float64)/50)
        torch.testing.assert_close(actual[0]['position'][:,0],torch.tensor([0.,1.8,3.6],dtype=torch.float64))

    def test_author_table1_activation_and_table2_preserved(self):
        raw=[source(positions=[[0.,0.,0.]]*5),source('table2',[[0.,0.,0.]]*5)]
        out=bind(raw)
        torch.testing.assert_close(out[0]['position'][:,2],torch.full((3,),-10.,dtype=torch.float64))
        self.assertEqual(float(out[1]['position'].abs().max()),0.)
        parked=bind([source(positions=[[2.,3.,-9.]]*5)])[0]
        torch.testing.assert_close(parked['position'],torch.tensor([[0.,0.,-10.]]*3,dtype=torch.float64))

    def test_unsupported_or_misbound_physical_values_fail_closed(self):
        for mutation in (lambda x:x['recipe'].update(sha256='a'*64),lambda x:x.update(size_m=[.3,.25,.02]),
                         lambda x:x.update(kinematic=False),lambda x:x.update(source_field='object_pos'),
                         lambda x:x.update(official_table1_uses=False),lambda x:x.update(parked_below_scene=True)):
            raw=source();mutation(raw)
            with self.assertRaises(ValueError):bind([raw])
        with self.assertRaises(ValueError):bind([source(),source()])
        with self.assertRaises(ValueError):bind([{'kind':'table','status':'incomplete_metadata'}])

    def test_no_extrapolation_or_changed_grid(self):
        for target in ([-.01,.01,.03],[0.,.02,.06],[0.,.01,.02]):
            with self.assertRaises(ValueError):
                bind_scene_supports([source()],torch.arange(5,dtype=torch.float64)/100,target)

    def test_tampered_bound_trajectory_or_source_rejected(self):
        raw=source();out=bind([raw])[0];out['position'][0,0]+=1
        with self.assertRaisesRegex(ValueError,'hash'):validate_bound_support(out)
        out=bind([raw])[0];raw['position'][0][0]+=1
        with self.assertRaisesRegex(ValueError,'source'):validate_bound_support(out,raw)

    def test_clip_frame_reset_and_missing_slot_parking(self):
        raw=[[source(),source('table2')],[]]
        motion=SimpleNamespace(names=['a','b'],scene_supports=raw,bound_scene_supports=[bind(raw[0]),[]],
             clip_start=torch.tensor([0,3]),clip_len=torch.tensor([3,2]),asset_ids=torch.zeros(5,dtype=torch.long))
        bank=NativeSceneSupportBank(motion,0,'cpu')
        frames=torch.tensor([2,3,0]);origins=torch.tensor([[10.,0.,0.],[0.,20.,0.],[0.,0.,30.]])
        p=bank.sample('table1',frames,origins)
        torch.testing.assert_close(p[:,:3],torch.tensor([[11.4,2.,.3],[0.,20.,-10.],[1.,2.,30.3]]))
        self.assertEqual(bank.sample('table2',torch.tensor([4]),torch.zeros(1,3))[0,2],-5.)
        p[0,0]=999
        self.assertAlmostEqual(float(bank.sample('table1',torch.tensor([2]),torch.zeros(1,3))[0,0]),1.4,places=6)

    def test_raw_support_cannot_be_omitted_or_misidentified(self):
        motion=SimpleNamespace(names=['a'],scene_supports=[[source()]],clip_len=torch.tensor([3]))
        with self.assertRaisesRegex(ValueError,'cannot omit'):validate_native_scene_supports(motion)
        motion.bound_scene_supports=[bind([source('table2')])]
        with self.assertRaisesRegex(ValueError,'source identity'):validate_native_scene_supports(motion)

    def test_support_pairs_have_separate_diagnostics_and_never_learned_channels(self):
        roles=torch.tensor([*SUPPORT_ROLES,BODY_ROLES[0],OBJECT,GROUND])
        pairs=torch.tensor([[0,2],[2,1],[0,3],[3,1],[4,0]])
        env=torch.zeros(5,dtype=torch.long);forces=torch.full((5,),100.)
        learned=aggregate_pairs(roles,pairs,env,forces,num_envs=1,substep_dt=.005)
        self.assertEqual(learned.shape,(1,len(CHANNELS)));self.assertEqual(float(learned.sum()),0.)
        ground=aggregate_ground_pairs(roles,pairs,env,forces,roles=(*BODY_ROLES,OBJECT),num_envs=1,substep_dt=.005)
        self.assertEqual(float(ground.sum()),0.)
        table=aggregate_ground_pairs(roles,pairs,env,forces,roles=(*BODY_ROLES,OBJECT),partner=SUPPORT_ROLES[0],num_envs=1,substep_dt=.005)
        self.assertEqual(float(table[0,0]),.5);self.assertEqual(float(table[0,-1]),.5)
        self.assertEqual(compiled_support_role('/env_0/Support_table1/geometry','/env_0/Support_table1'),20)
        self.assertIsNone(compiled_support_role('/ground','world'))

    def test_compiled_support_identity_survives_world_fixed_collider_names(self):
        import mujoco
        from hoi.v1.contacts import compiled_geom_roles
        bodies=''.join(f'<body name="{name}"><geom type="sphere" size=".02"/></body>' for name in
             ('left_wrist_yaw_link','right_wrist_yaw_link','left_ankle_roll_link','right_ankle_roll_link','Object'))
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="ground" type="plane" size="1 1 .1"/>'
             +bodies+'<body name="env_0_Support_table1"><geom name="env_0_Support_table1_geometry" type="box" size=".15 .125 .005"/></body>'
             +'<geom name="env_0_Support_table2_geometry" type="box" size=".15 .125 .005"/></worldbody></mujoco>')
        roles,envs,names=compiled_geom_roles(model)
        self.assertEqual(roles[names.index('env_0_Support_table1_geometry')],20)
        self.assertEqual(roles[names.index('env_0_Support_table2_geometry')],21)
        self.assertEqual(roles[names.index('ground')],GROUND)

    def test_held_native_pose_readback_rejects_drift(self):
        env=HOITrackingSim.__new__(HOITrackingSim);env.device='cpu';env.N=2
        pose=torch.tensor([[1.,2.,.3,0,0,0,1.],[3.,4.,.5,0,0,0,1.]])
        data=SimpleNamespace(root_link_pose_w=SimpleNamespace(torch=pose.clone()),root_link_vel_w=SimpleNamespace(torch=torch.zeros(2,6)))
        env.supports={'table1':SimpleNamespace(data=data)};env.support_reset_poses={'table1':pose}
        env.support_alignment={'reset_count':0,'step_count':0,'max_position_error_m':0.,'max_quaternion_error':0.}
        env._check_support_alignment(reset=True)
        data.root_link_pose_w.torch[:,3:]*=-1
        env._check_support_alignment()
        data.root_link_pose_w.torch[0,0]+=.001
        with self.assertRaisesRegex(RuntimeError,'drifted'):env._check_support_alignment()


class SupportProbePlacementTests(unittest.TestCase):
    """Regression for live geom-cache/root-frame mixing; no native acceptance claim."""
    @staticmethod
    def fixture(*, drop_joint_write=False):
        from supermimic.rotations import quat_to_mat
        sim=HOITrackingSim.__new__(HOITrackingSim);sim.N=2;sim.device='cpu'
        sim.scene=SimpleNamespace(env_origins=torch.tensor([[10.,20.,0.],[-10.,-20.,1.]]))
        state=[torch.zeros(2,3),torch.eye(3).expand(2,3,3).clone(),torch.ones(2,3),
               torch.ones(2,3),torch.zeros(2,29),torch.ones(2,29)]
        q=torch.linspace(-.4,.6,29)
        calls=[]
        def placement(role,*,ground):
            calls.append((role,ground))
            return ([.1,.2,.9],torch.diag(torch.tensor([-1.,1.,-1.])),q,
                    [.4,.5,-.002],[0.,0.,-1.],{'robot_minimum_world_z_m':-.002})
        def root_pose(*,root_pose,env_ids):
            state[0]=root_pose[:,:3]-sim.scene.env_origins
            state[1]=quat_to_mat(root_pose[:,[6,3,4,5]])
        def root_velocity(*,root_velocity,env_ids):
            state[2]=root_velocity[:,:3];state[3]=root_velocity[:,3:]
        def joint_position(*,position,env_ids):
            if not drop_joint_write:state[4]=position.clone()
        def joint_velocity(*,velocity,env_ids):state[5]=velocity.clone()
        sim.robot=SimpleNamespace(write_root_link_pose_to_sim_index=root_pose,
            write_root_link_velocity_to_sim_index=root_velocity,write_joint_position_to_sim_index=joint_position,
            write_joint_velocity_to_sim_index=joint_velocity)
        sim.env=SimpleNamespace(default_q=torch.full((1,29),.05),xi=torch.arange(60).reshape(2,30)/1000.,
            alpha=torch.tensor(.25),refresh=lambda:None)
        sim._compiled_probe_planner=SimpleNamespace(placement=placement)
        sim.state=lambda:(tuple(state),None)
        def forbidden(*args,**kwargs):raise AssertionError('live cache or physics must not supply placement')
        sim._region_surface=forbidden;sim._advance=forbidden
        return sim,q,calls

    def test_exact_compiled_pose_is_translated_with_surface_and_held(self):
        from hoi.v1.contacts import LEFT
        sim,q,calls=self.fixture()
        pose,actions,definition=sim._body_support_probe_placement(torch.arange(2))
        torch.testing.assert_close(pose[:,:3]-sim.scene.env_origins,torch.tensor([[.4,.5,2.995]]).expand(2,-1))
        torch.testing.assert_close(sim.state()[0][0],torch.tensor([[.1,.2,3.9]]).expand(2,-1))
        torch.testing.assert_close(sim.env.default_q+sim.env.xi[:,1:30]+sim.env.alpha*actions,q.expand(2,-1))
        torch.testing.assert_close(sim.state()[0][4],q.expand(2,-1))
        self.assertEqual(calls,[(LEFT,True)])
        self.assertEqual(definition['rigid_translation_world_m'],[0.,0.,3.])
        self.assertEqual(definition['extra_physics_steps'],0)
        self.assertLessEqual(max(definition['initial_state_readback']['maximum_errors'].values()),2e-5)
        # A zero action would move this legal pose back toward default_q.
        self.assertGreater(float(actions.abs().max()),1.)
        self.assertEqual(float(sim.state()[0][5].abs().max()),0.)

    def test_silent_joint_write_failure_is_a_hard_error(self):
        sim,_,_=self.fixture(drop_joint_write=True)
        with self.assertRaisesRegex(RuntimeError,'initial state differs'):
            sim._body_support_probe_placement(torch.arange(2))


if __name__=='__main__':unittest.main()
