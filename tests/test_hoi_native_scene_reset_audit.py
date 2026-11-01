"""Terminal reset/readback contract only; mocked writes are not GPU evidence."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import tempfile
import unittest
import torch
from hoi.v1.native import HOITrackingSim
from hoi.v1.scene_supports import bind_scene_supports, NativeSceneSupportBank
from hoi.v1.native_scene_reset_audit import reset_sample_plan, audit_scene_resets, SCHEMA, PHASE
from tests.test_hoi_v1_scene_supports import source


def fixture():
    lengths=[6,3,4];starts=torch.tensor([0,6,9]);raw=[
        [source('table1',[[1.,2.,z] for z in [.3,.3,.3,-10.,-10.,.4]]),
         source('table2',[[3.,4.,.5]]*6)],[],[source('table2',[[0.,0.,-5.]]*4)]]
    bound=[bind_scene_supports(r,torch.arange(n,dtype=torch.float64)/50,torch.arange(n,dtype=torch.float64)/50)
           for r,n in zip(raw,lengths)]
    motion=SimpleNamespace(names=['present','missing','parked'],scene_supports=raw,bound_scene_supports=bound,
        clip_start=starts,clip_len=torch.tensor(lengths),asset_ids=torch.zeros(sum(lengths),dtype=torch.long))
    env=HOITrackingSim.__new__(HOITrackingSim);env.N=2;env.device='cpu';env.motion=motion;env.asset={'asset_id':0}
    env.scene=SimpleNamespace(env_origins=torch.tensor([[10.,20.,0.],[-20.,-30.,1.]]))
    env.support_bank=NativeSceneSupportBank(motion,0,'cpu');env.f=torch.zeros(env.N,dtype=torch.long)
    env.supports={};env.support_reset_poses={};env.support_reset_present=torch.zeros(env.N,2,dtype=torch.bool)
    for sid in env.support_bank.support_ids:
        pose=torch.zeros(env.N,7);pose[:,6]=1
        env.supports[sid]=SimpleNamespace(data=SimpleNamespace(root_link_pose_w=SimpleNamespace(torch=pose),
            root_link_vel_w=SimpleNamespace(torch=torch.zeros(env.N,6))))
        env.support_reset_poses[sid]=pose.clone()
    class Sim:
        def __init__(self):self.count=41
        def get_physics_step_count(self):return self.count
        def step(self):self.count+=1
    sim=Sim();env.env=SimpleNamespace(sim=sim,reset_noise=False)
    class Adapter:
        def step(self):env.contacts.calls+=1
    adapter=Adapter();env.contacts=SimpleNamespace(adapter=adapter,manager=SimpleNamespace(_solver=adapter),calls=4)
    def reset(ids,frames):
        env.f[ids]=frames
        for sid in env.supports:
            slot=('table1','table2').index(sid)
            pose=env.support_bank.sample(sid,frames,env.scene.env_origins[ids])
            env.supports[sid].data.root_link_pose_w.torch[ids]=pose
            env.supports[sid].data.root_link_vel_w.torch[ids]=0
            env.support_reset_poses[sid][ids]=pose
            env.support_reset_present[ids,slot]=env.support_bank.present[sid][frames]
    env.reset=reset
    return env


class TerminalSceneResetAuditTests(unittest.TestCase):
    def test_plan_covers_each_clip_and_both_neighbors_of_presence_changes(self):
        env=fixture();clips,samples=reset_sample_plan(env.motion,0)
        self.assertEqual(clips[0]['presence_change_frames'],[3,5])
        self.assertEqual(clips[0]['selected_local_frames'],[0,2,3,4,5])
        self.assertEqual(clips[1]['selected_local_frames'],[0,1,2])
        self.assertEqual(clips[2]['selected_local_frames'],[0,2,3])
        self.assertEqual(len(samples),11)
        self.assertEqual([(s['clip_name'],s['local_frame']) for s in samples],
            [(c['name'],f) for c in clips for f in c['selected_local_frames']])

    def test_terminal_reads_all_clips_and_masks_without_any_physics(self):
        env=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'audit.json';header=audit_scene_resets(env,path);full=json.loads(path.read_text())
            self.assertEqual(header['schema'],SCHEMA);self.assertEqual(header['phase'],PHASE)
            self.assertTrue(header['passed']);self.assertTrue(header['state_discarded'])
            self.assertEqual(header['requested_sample_count'],11);self.assertEqual(header['completed_sample_count'],11)
            self.assertEqual(header['reset_batches'],6);self.assertEqual(len(full['samples']),11)
            self.assertEqual(header['physics_steps_before'],header['physics_steps_after'])
            self.assertEqual(header['recorder_substeps_before'],header['recorder_substeps_after'])
            self.assertEqual(header['additional_physics_steps'],0)
            self.assertEqual(header['presence_mismatches'],0)
            self.assertEqual({k:v for k,v in header.items() if k!='artifact'},{k:v for k,v in full.items() if k!='samples'})
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),header['artifact']['sha256'])
            missing=[r for r in full['samples'] if r['clip_name']=='missing']
            self.assertTrue(all(r['actual_present']==[False,False] for r in missing))
            self.assertTrue(all(r['actual_pose_xyzw'][0][2]==-10 and r['actual_pose_xyzw'][1][2]==-5 for r in missing))
        env.env.sim.step();self.assertEqual(env.env.sim.count,42)  # original method restored

    def test_wrong_bank_index_is_caught_against_independent_bound_records(self):
        env=fixture();env.support_bank.poses['table1'][0,0]+=1.
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'audit.json'
            with self.assertRaisesRegex(RuntimeError,'readback differs'):audit_scene_resets(env,path)
            result=json.loads(path.read_text());self.assertFalse(result['passed'])
            self.assertGreater(result['maximum_errors']['position_m'],.9)
            self.assertEqual(result['additional_physics_steps'],0)

    def test_incorrect_presence_is_not_hidden_by_correct_pose(self):
        env=fixture();env.support_bank.present['table1'][0]=False
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'audit.json'
            with self.assertRaisesRegex(RuntimeError,'mask=1'):audit_scene_resets(env,path)
            self.assertEqual(json.loads(path.read_text())['presence_mismatches'],1)

    def test_sim_and_solver_integration_attempts_are_blocked_before_execution(self):
        for target,key in [('sim','attempted_sim_step_calls'),('solver','attempted_solver_step_calls')]:
            env=fixture();original=env.reset
            def reset(ids,frames):
                original(ids,frames)
                (env.env.sim if target=='sim' else env.contacts.adapter).step()
            env.reset=reset
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'audit.json'
                with self.assertRaisesRegex(RuntimeError,'prohibits step'):audit_scene_resets(env,path)
                result=json.loads(path.read_text());self.assertEqual(result[key],1)
                self.assertFalse(result['passed']);self.assertEqual(env.env.sim.count,41);self.assertEqual(env.contacts.calls,4)

    def test_all_no_table_clips_still_have_exact_sample_coverage(self):
        env=fixture();env.motion.scene_supports=[[],[],[]];env.motion.bound_scene_supports=[[],[],[]]
        env.support_bank=NativeSceneSupportBank(env.motion,0,'cpu');env.supports={};env.support_reset_poses={}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'audit.json';header=audit_scene_resets(env,path);full=json.loads(path.read_text())
            self.assertTrue(header['passed']);self.assertEqual(header['declared_support_ids'],[])
            self.assertEqual(header['clip_names'],['present','missing','parked'])
            self.assertEqual(header['completed_sample_count'],9)
            self.assertTrue(all(r['actual_present']==[False,False] for r in full['samples']))
            self.assertTrue(all(not c['bindings'] for c in header['clips']))

    def test_counter_change_or_nonzero_support_velocity_fails_without_step_call(self):
        for mutation in ('counter','velocity'):
            env=fixture();original=env.reset
            def reset(ids,frames):
                original(ids,frames)
                if mutation=='counter':env.env.sim.count+=1
                else:env.supports['table1'].data.root_link_vel_w.torch[ids,0]=.1
            env.reset=reset
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'audit.json'
                with self.assertRaises(RuntimeError):audit_scene_resets(env,path)
                result=json.loads(path.read_text());self.assertFalse(result['passed'])
                if mutation=='counter':self.assertGreater(result['additional_physics_steps'],0)
                else:self.assertGreater(result['maximum_errors']['linear_velocity_m_s'],.09)

    def test_unknown_noise_pending_capture_and_changed_binding_are_rejected(self):
        env=fixture();env.env.reset_noise=True
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'audit.json'
            with self.assertRaisesRegex(ValueError,'reset_noise'):audit_scene_resets(env,path)
            env.env.reset_noise=False;env.__dict__['_post_advance_capture']=lambda:None
            with self.assertRaisesRegex(ValueError,'capture finish'):audit_scene_resets(env,path)
            del env.__dict__['_post_advance_capture']
            env.motion.bound_scene_supports[0][0]['position'][0,0]+=1
            with self.assertRaisesRegex(ValueError,'hash'):audit_scene_resets(env,path)


if __name__=='__main__':unittest.main()
