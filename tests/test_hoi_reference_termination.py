import math
from types import SimpleNamespace
import unittest
import torch

from hoi.v1.mdp import HOIMDP, HOIRewardCfg
from hoi.v1.native import HOITrackingSim
from hoi.v1.scene_context import empty_scene_state
from hoi.v1.termination import gravity_angle_error
from supermimic.mdp import MDP
from tests.test_hoi_v1_mdp import DummyKin, DistanceShapes, bank_fixture


def pitch(angle):
    c,s=angle.cos(),angle.sin();z=torch.zeros_like(angle);o=torch.ones_like(angle)
    return torch.stack((c,z,s,z,o,z,-s,z,c),-1).reshape(*angle.shape,3,3)


def yaw(angle):
    c,s=angle.cos(),angle.sin();z=torch.zeros_like(angle);o=torch.ones_like(angle)
    return torch.stack((c,-s,z,s,c,z,z,z,o),-1).reshape(*angle.shape,3,3)


class ReferenceTerminationTests(unittest.TestCase):
    def setUp(self):
        self.motion=bank_fixture();self.f=torch.tensor([3,int(self.motion.clip_start[1])+3])
        self.mdp=HOIMDP(DummyKin(),self.motion,torch.zeros(29),DistanceShapes())
        self.S,self.O=self.motion.ref_state(self.f),self.motion.ref_object(self.f)

    def test_matching_sitting_and_leaning_and_yaw_do_not_fail(self):
        self.motion.root_p[self.f,2]=torch.tensor([.3,.45])
        self.motion.root_R[self.f]=pitch(torch.tensor([math.pi/3,-math.pi/3]))
        S=list(self.motion.ref_state(self.f))
        S[1]=yaw(torch.tensor([1.2,-2.4]))@S[1]
        self.assertFalse(self.mdp.termination(tuple(S),self.O,self.f).any())
        torch.testing.assert_close(gravity_angle_error(S[1],self.motion.root_R[self.f]),torch.zeros(2),atol=1e-6,rtol=0)

    def test_opposite_lean_detected_and_shared_baseline_unchanged(self):
        self.motion.root_R[self.f]=pitch(torch.full((2,),math.pi/3))
        S=list(self.motion.ref_state(self.f));S[1]=pitch(torch.full((2,),-math.pi/3))
        torch.testing.assert_close(gravity_angle_error(S[1],self.motion.root_R[self.f]),torch.full((2,),2*math.pi/3))
        self.assertTrue(self.mdp.termination(tuple(S),self.O,self.f).all())
        old=MDP(DummyKin(),self.motion,torch.zeros(29))
        self.assertFalse(old.termination(tuple(S),self.f).any())

    def test_upright_boundary_and_soft_angle_agree(self):
        limit=self.mdp.cfg.robot_tilt_limit_rad
        self.assertAlmostEqual(math.cos(limit),.2)
        S=list(self.S);S[1]=pitch(torch.tensor([limit-1e-4,limit+1e-4]))
        height,tilt,obj=self.mdp.termination_components(tuple(S),self.O,self.f)
        self.assertEqual(tilt.tolist(),[False,True]);self.assertFalse(height.any()|obj.any())
        soft=self.mdp.human.soft_termination(tuple(S),self.f)
        self.assertLess(float(soft[0]),.5);self.assertGreater(float(soft[1]),.5)
        # Keep the original height threshold independent of reference absolute height.
        S=list(self.S);S[0]=S[0].clone();S[0][:,2]+=torch.tensor([.249,.251])
        self.assertEqual(self.mdp.human.termination(tuple(S),self.f).tolist(),[False,True])

    def test_object_scale_floor_intermediate_and_cap(self):
        class Scales(DistanceShapes):
            def bbox_diagonal(self,ids):return torch.tensor([.37,.75,1.,1.9521163])[ids]
        f=torch.tensor([3,4,int(self.motion.clip_start[1])+3,int(self.motion.clip_start[1])+4])
        self.motion.asset_ids[f]=torch.arange(4)
        mdp=HOIMDP(DummyKin(),self.motion,torch.zeros(29),Scales())
        thresholds=mdp.object_failure_threshold(f)
        torch.testing.assert_close(thresholds,torch.tensor([1.,1.5,2.,2.]))
        S,O=self.motion.ref_state(f),self.motion.ref_object(f)
        for delta,expected in ((-.001,False),(.001,True)):
            p=O[0].clone();p[:,0]+=thresholds+delta
            self.assertEqual(mdp.termination(S,(p,*O[1:]),f).tolist(),[expected]*4)
        # Soft cutoff uses the same capped value, even for the biggest asset.
        p=O[0].clone();p[:,0]+=thresholds
        robot_fail=mdp.human.soft_termination(S,f)
        torch.testing.assert_close(mdp.soft_termination(S,(p,*O[1:]),f),1-(1-robot_fail)*.5)

    def test_gradients_finite_at_alignment_opposite_and_boundary(self):
        for angle in (0.,1e-8,self.mdp.cfg.robot_tilt_limit_rad,math.pi-1e-8,math.pi):
            x=torch.full((2,),angle,dtype=torch.float64,requires_grad=True)
            S=list(self.S);S[1]=pitch(x)
            value=self.mdp.soft_termination(tuple(S),self.O,self.f)
            grad,=torch.autograd.grad(value.sum(),x)
            self.assertTrue(torch.isfinite(value).all() & torch.isfinite(grad).all())
        x=torch.tensor([.4,.8],dtype=torch.float64,requires_grad=True)
        ref=pitch(torch.tensor([-.1,.1],dtype=torch.float64))
        self.assertTrue(torch.autograd.gradcheck(lambda a:gravity_angle_error(pitch(a),ref),(x,)))

    def test_gravity_angle_supports_horizon_batches_and_low_precision(self):
        for dtype in (torch.float32,torch.bfloat16):
            R=pitch(torch.zeros(3,2)).to(dtype)
            torch.testing.assert_close(gravity_angle_error(R,R),torch.zeros(3,2))

    def test_invalid_thresholds_rejected(self):
        for cfg in (HOIRewardCfg(object_term_max_m=.5),HOIRewardCfg(object_term_max_m=float('inf')),
                    HOIRewardCfg(object_soft_temperature=0),HOIRewardCfg(robot_tilt_limit_rad=0)):
            with self.assertRaises(ValueError):HOIMDP(DummyKin(),self.motion,torch.zeros(29),DistanceShapes(),cfg)

    def test_native_collector_uses_new_predicates_for_reset_and_reason_bits(self):
        # Exercise the production collector step with tensor-only stand-ins for
        # physics; this is a control-flow test, not a native physics claim.
        class LargeShapes(DistanceShapes):
            def bbox_diagonal(self,ids):return torch.full_like(ids,1.95,dtype=torch.float32)
        mdp=HOIMDP(DummyKin(),self.motion,torch.zeros(29),LargeShapes())
        f1=self.f+1;S1=list(self.motion.ref_state(f1));O1=list(self.motion.ref_object(f1))
        self.motion.root_R[f1[0]]=pitch(torch.tensor(math.pi/3))
        S1[1]=self.motion.root_R[f1].clone();S1[1][0]=pitch(torch.tensor(-math.pi/3))
        O1[0]=O1[0].clone();O1[0][1,0]+=2.01
        e=SimpleNamespace(f=self.f.clone(),a_prev=torch.zeros(2,29),ep_step=torch.zeros(2,dtype=torch.long),
            max_episode_steps=500,xi=torch.zeros(2,33),ep_disc_ret=torch.zeros(2),ep_ret=torch.zeros(2),
            gamma=.99,eval_mode=False,start_bin=torch.zeros(2,dtype=torch.long),bin_G=torch.zeros(1))
        env=HOITrackingSim.__new__(HOITrackingSim);env.env=e;env.N=2;env.motion=self.motion;env.mdp=mdp
        env.asset_id=self.motion.asset_ids[self.f].clone();env.xi_o=torch.zeros(2,12)
        env.contact_missing_counter=torch.zeros(2,16,dtype=torch.long)
        env.support_state=lambda:empty_scene_state((2,))
        states=iter([(self.S,self.O),(tuple(S1),tuple(O1))]);env.state=lambda:next(states)
        env._advance=lambda action:dict(contact=torch.zeros(2,16),contact_valid=torch.ones(2,16,dtype=torch.bool))
        resets=[];env.reset=lambda ids:resets.append(ids.clone())
        tr=env.step(torch.zeros(2,29))
        self.assertEqual(tr['term'].tolist(),[True,True]);self.assertEqual(tr['failure_reason'].tolist(),[2,4])
        self.assertEqual(resets[0].tolist(),[0,1])


if __name__=='__main__':unittest.main()
