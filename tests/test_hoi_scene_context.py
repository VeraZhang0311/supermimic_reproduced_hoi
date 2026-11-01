"""Observed stationary support input contracts; CPU synthetic evidence only."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from supermimic.networks import RunningNorm
from supermimic.rotations import so3_exp, mat_to_quat
from hoi.v1.scene_context import (empty_scene_state, scene_context, scene_from_record,
    normalize_scene_context, validate_scene_state)
from hoi.v1.networks import Policy, WorldModel, Critic, frozen_parameters
from hoi.v1.native import HOITrackingSim
from hoi.v1.state import unflatten_state
from hoi.v1.checkpoint import make_checkpoint, validate_checkpoint_features
from tests.test_hoi_v1_learner import make_learner, transition
from tests.test_hoi_v1_networks import states


def table_state(batch=2, dtype=torch.float32):
    record=empty_scene_state((batch,),dtype=dtype)
    record["support_present"][:,0]=True
    record["support_pose_xyzw"][:,0,:3]=torch.tensor([2.,3.,.7],dtype=dtype)
    return record


class SceneContextTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads=torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,self.previous_threads)
        torch.manual_seed(36)

    def test_mask_remains_exactly_zero_after_nonzero_running_statistics(self):
        S,_=states(2)
        record=empty_scene_state((2,))
        record["support_pose_xyzw"][:,:,:3]=1234.
        value=scene_from_record(S,record)
        self.assertTrue(torch.equal(value,torch.zeros(2,26)))
        norm=RunningNorm(26)
        norm.mean.copy_(torch.arange(26,dtype=torch.float32)+1)
        norm.var.fill_(.2)
        self.assertTrue(torch.equal(normalize_scene_context(norm,value),value))
        active=scene_from_record(S,table_state())
        normalized=normalize_scene_context(norm,active)
        self.assertTrue(torch.equal(normalized[:,13:],torch.zeros(2,13)))
        self.assertTrue(torch.equal(normalized[:,0],torch.ones(2)))
        invalid=value.clone();invalid[0,2]=1
        with self.assertRaisesRegex(ValueError,'absent slots'):
            normalize_scene_context(norm,invalid)

    def test_heading_relative_position_and_common_SE2_transform(self):
        S,_=states(2)
        record=table_state()
        original=scene_from_record(S,record)
        yaw=so3_exp(torch.tensor([[0.,0.,.63]])).expand(2,3,3)
        shift=torch.tensor([4.,-7.,0.])
        rotate=lambda x:(yaw@x.unsqueeze(-1)).squeeze(-1)
        changed=(rotate(S[0])+shift,yaw@S[1],rotate(S[2]),rotate(S[3]),S[4],S[5])
        moved=deepcopy(record)
        moved["support_pose_xyzw"][:,0,:3]=rotate(record["support_pose_xyzw"][:,0,:3])+shift
        q=mat_to_quat(yaw)
        moved["support_pose_xyzw"][:,0,3:]=q[:,[1,2,3,0]]
        torch.testing.assert_close(scene_from_record(changed,moved),original,atol=2e-6,rtol=2e-6)
        identity=(torch.tensor([[1.,2.,3.]]),torch.eye(3)[None],*tuple(x[:1] for x in S[2:]))
        value=scene_from_record(identity,table_state(1))
        torch.testing.assert_close(value[0,1:4],torch.tensor([1.,1.,-2.3]))
        torch.testing.assert_close(value[0,4:10],torch.tensor([1.,0.,0.,1.,0.,0.]))
        torch.testing.assert_close(value[0,10:13],torch.tensor([.3,.25,.01]))

    def test_world_pose_is_constant_and_state_jacobian_matches_finite_difference(self):
        S,_=states(1)
        root=S[0].double().requires_grad_()
        angle=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
        record=table_state(1,torch.float64)
        record["support_pose_xyzw"].requires_grad_()
        def observed(p,y):
            theta=torch.stack((y*0,y*0,y))[None]
            state=(p,so3_exp(theta),*tuple(x.double() for x in S[2:]))
            return scene_from_record(state,record)
        value=observed(root,angle)
        weight=torch.arange(26,dtype=torch.float64)[None]/26
        objective=(value*weight).sum()
        gp,gy=torch.autograd.grad(objective,(root,angle),retain_graph=True)
        self.assertGreater(float(gp.abs().sum()+gy.abs()),0.)
        eps=1e-6
        fd=((observed(root.detach(),angle.detach()+eps)-observed(root.detach(),angle.detach()-eps))*weight).sum()/(2*eps)
        torch.testing.assert_close(gy,fd,atol=1e-8,rtol=1e-7)
        objective.backward()
        self.assertIsNone(record["support_pose_xyzw"].grad)
        moved=observed(root.detach()+torch.tensor([[1.,0.,0.]]),angle.detach())
        self.assertFalse(torch.equal(moved,value))
        torch.testing.assert_close(record["support_pose_xyzw"][0,0,:3],torch.tensor([2.,3.,.7],dtype=torch.float64))

    def test_identical_robot_object_with_different_tables_changes_all_three_network_inputs(self):
        S,O=states(2)
        record=table_state();other=deepcopy(record);other["support_pose_xyzw"][:,0,0]+=1
        first,second=scene_from_record(S,record),scene_from_record(S,other)
        pi=Policy(hidden=(16,),object_hidden=(8,));wm=WorldModel(hidden=(16,),contact_hidden=8);critic=Critic(hidden=(16,))
        points=torch.randn(2,128,3);captured=[]
        actor_args=(torch.randn(2,10,99),torch.randn(2,730),torch.randn(2,10,18),torch.randn(2,470),points)
        with pi.dec[0].register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone())):
            pi(*actor_args,scene_context=first);pi(*actor_args,scene_context=second)
        torch.testing.assert_close(captured[0][:,:-26],captured[1][:,:-26],atol=0,rtol=0)
        self.assertFalse(torch.equal(captured[0][:,-26:],captured[1][:,-26:]))
        wm_args=(S,O,torch.randn(2,29),torch.randn(2,33),torch.randn(2,12),points)
        x,y=wm.inputs(*wm_args,first),wm.inputs(*wm_args,second)
        torch.testing.assert_close(x[:,:224],y[:,:224],atol=0,rtol=0)
        self.assertFalse(torch.equal(x[:,224:],y[:,224:]))
        critic_args=tuple(torch.randn(2,n) for n in (914,470,15,15,12,32))
        captured.clear()
        with critic.net[0].register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone())):
            critic(*critic_args,first,points);critic(*critic_args,second,points)
        torch.testing.assert_close(captured[0][:,:-26],captured[1][:,:-26],atol=0,rtol=0)
        self.assertFalse(torch.equal(captured[0][:,-26:],captured[1][:,-26:]))

    def test_native_snapshot_uses_actual_pose_and_environment_origin_without_reference_lookup(self):
        env=HOITrackingSim.__new__(HOITrackingSim);env.N=2;env.device='cpu'
        env.scene=SimpleNamespace(env_origins=torch.tensor([[10.,0.,0.],[0.,20.,0.]]))
        actual=table_state()["support_pose_xyzw"][:,0].clone();actual[:,:3]+=env.scene.env_origins
        env.supports={"table1":SimpleNamespace(data=SimpleNamespace(root_link_pose_w=SimpleNamespace(torch=actual)))}
        env.support_reset_poses={"table1":actual.clone()};env.support_reset_present=table_state()["support_present"]
        record=env.support_state()
        torch.testing.assert_close(record["support_pose_xyzw"],table_state()["support_pose_xyzw"])
        actual[:,0]+=.00001
        next_record=env.support_state()
        self.assertTrue(torch.equal(record["reference_support_pose_xyzw"],next_record["reference_support_pose_xyzw"]))
        self.assertFalse(torch.equal(record["support_pose_xyzw"],next_record["support_pose_xyzw"]))
        env.supports={};env.support_reset_present.zero_()
        self.assertTrue(torch.equal(scene_from_record(states(2)[0],env.support_state()),torch.zeros(2,26)))

    def test_replay_retains_world_pose_mask_and_rejects_missing_scene_fields(self):
        learner=make_learner(horizon=3)
        expected=table_state()
        for tick in range(12):
            row=transition(learner,[60+tick,200+tick],tick)
            row.update(deepcopy(expected));learner.buffer.add(row)
        window=learner.buffer._windows_at(torch.tensor([0]),torch.tensor([0]),8)
        torch.testing.assert_close(window["support_pose_xyzw"][0],expected["support_pose_xyzw"][:1].expand(8,-1,-1))
        start,history=learner.buffer.sample_starts(4,10,horizon=3,motion=learner.motion)
        torch.testing.assert_close(start["support_pose_xyzw"],expected["support_pose_xyzw"][:1].expand(4,-1,-1))
        self.assertTrue(bool(start["support_present"][:,0].all()))
        self.assertTrue(bool(history["support_present"][...,0].all()))
        bad=transition(learner,[72,212],12);bad.pop("support_pose_xyzw")
        with self.assertRaisesRegex(ValueError,'missing support_pose_xyzw'):learner.buffer.add(bad)
        with self.assertRaisesRegex(ValueError,'old scene-less'):scene_from_record(states(2)[0],{})

    def test_imagination_reuses_frozen_world_scene_recomputes_each_state_and_backpropagates(self):
        learner=make_learner(horizon=3)
        start=transition(learner,[60,200],10);start.update(table_state())
        start["support_pose_xyzw"].requires_grad_()
        history={key:value[:,None].expand((2,10)+value.shape[1:]).clone() for key,value in start.items()}
        noise=[torch.zeros(2,10,99)]+[torch.zeros(2,99) for _ in range(3)]
        before=deepcopy(learner.wm.state_dict())
        actor_scenes=[];critic_scenes=[]
        def actor_capture(module,args,kwargs):
            actor_scenes.append(kwargs['scene_context'].detach().clone())
        with learner.pi.register_forward_pre_hook(actor_capture,with_kwargs=True), \
                learner.critic.register_forward_pre_hook(lambda module,args:critic_scenes.append(args[-2].detach().clone())), \
                frozen_parameters(learner.wm,learner.critic),patch.object(learner.wm,'step',wraps=learner.wm.step) as call:
            learner.action(start,exploration=False,update_statistics=False)
            trajectory=learner.imagine(start,history,fixed_robot_noise=noise)
            for tick,args in enumerate(call.call_args_list):
                actual=args.args[-1]
                expected=scene_from_record(trajectory["states"][tick],start)
                torch.testing.assert_close(actual,expected)
                torch.testing.assert_close(actor_scenes[tick+1],actual)
            torch.testing.assert_close(actor_scenes[0],actor_scenes[1])
            torch.testing.assert_close(critic_scenes[-1],scene_from_record(trajectory['states'][-1],start))
            self.assertFalse(torch.equal(call.call_args_list[0].args[-1],call.call_args_list[-1].args[-1]))
            trajectory["actions"][0].retain_grad()
            trajectory["objective"].backward()
        self.assertIsNone(start["support_pose_xyzw"].grad)
        self.assertGreater(float(trajectory["actions"][0].grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is None for p in learner.wm.parameters()))
        for key,value in before.items():torch.testing.assert_close(value,learner.wm.state_dict()[key],atol=0,rtol=0)

    def test_old_sceneless_checkpoint_and_invalid_presence_are_rejected(self):
        learner=make_learner()
        ck=make_checkpoint(learner.pi,learner.wm,learner.critic,learner.optimizers,iteration=0,
            physics={"hand_collision":True,"self_collision":True},data_identity={"synthetic":True},initialization={"test":True})
        validate_checkpoint_features(ck)
        ck['schema_version']='supermimic_hoi_v1_checkpoint_2'
        with self.assertRaisesRegex(ValueError,'scene-less'):validate_checkpoint_features(ck)
        record=table_state();record['support_pose_xyzw'][0,0,6]=0
        with self.assertRaisesRegex(ValueError,'unit xyzw'):validate_scene_state(**{'pose':record['support_pose_xyzw'],'present':record['support_present']})
        record=table_state();record['support_present']=torch.full((2,2),.2)
        with self.assertRaisesRegex(ValueError,'binary'):validate_scene_state(record['support_pose_xyzw'],record['support_present'])


if __name__=='__main__':unittest.main()
