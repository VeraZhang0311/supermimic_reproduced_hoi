"""Frozen pair identity, complete protected coverage and failure-continuation."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hoi_tables import evaluate as evaluation
from hoi_tables.config import TrainConfig
from hoi_tables.networks import feature_schema, Policy


class PairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ("manifest.json","assets.json","contacts.pt"):
            (self.root/name).write_bytes(name.encode())
        self.shape = {"bank_sha256":"shape","catalog_sha256":evaluation.file_sha256(self.root/"assets.json")}
        config = TrainConfig().to_dict()
        self.identity = {"schema":feature_schema(),"physics":{"frozen":"physics"},"global_config":config,
                         "worker_config":replace(TrainConfig(),num_envs=2048,wm_batch=1024,pi_batch=2048).to_dict(),
                         "world_size":2,"from_scratch":True,"base_checkpoint":None,"smoke":False,
                         "schedule_scope":"500WM_then_fixed_actor_critic",
                         "manifest_sha256":evaluation.file_sha256(self.root/"manifest.json"),
                         "asset_catalog_sha256":evaluation.file_sha256(self.root/"assets.json"),
                         "shape_bank":self.shape,"reward":{"source_commit":"66d9184",
                             "frozen_reference_contacts":{"file_sha256":evaluation.file_sha256(self.root/"contacts.pt"),
                                                          "selected_clips":["asset-specific-training-clips"]}}}
        self.spec = {"weight":torch.zeros(2),"norm.count":torch.zeros(())}
        self.checkpoints = []
        for rank,name in enumerate(evaluation.ASSETS):
            folder = self.root/f"rank_{rank:02d}"; folder.mkdir()
            identity = {**deepcopy(self.identity),"asset":name,"asset_id":rank}
            identity["reward"]["frozen_reference_contacts"]["selected_clips"]=[name]
            checkpoint = {"schema":"hoi_tables_checkpoint_v1","iteration":500,"identity":identity,
                          **{key:deepcopy(self.spec) for key in ("policy","wm","critic")}}
            self.checkpoints.append(checkpoint)
            torch.save(checkpoint,folder/"checkpoint_000500.pt")

    def load(self, **changes):
        args = dict(manifest_path=self.root/"manifest.json",asset_bundle=self.root/"assets.json",
                    shape_metadata=self.shape,physics_identity={"frozen":"physics"},source_identity={"frozen":True},
                    reference_contacts=self.root/"contacts.pt",smoke=True)
        args.update(changes)
        with patch.object(evaluation,"_model_specifications",return_value={key:self.spec for key in ("policy","wm","critic")}):
            return evaluation.load_checkpoint_pair(self.root,500,**args)

    def save(self, rank):
        torch.save(self.checkpoints[rank],self.root/f"rank_{rank:02d}/checkpoint_000500.pt")

    def test_pair_same_milestone_states_and_rms(self):
        result = self.load()
        self.assertEqual(result["iteration"],500)
        self.assertEqual(set(result["checkpoint_sha256"]),{"rank_00","rank_01"})
        self.assertEqual(set(result["model_state_sha256"]),{"policy","wm","critic"})
        self.assertEqual(result["model_state_sha256"]["policy"],evaluation.state_dict_hash(self.spec))
        # Different asset-specific training reference selections are legitimate;
        # model/RMS and all other scientific identity must still match exactly.
        self.checkpoints[1]["policy"]["norm.count"] += 1
        self.save(1)
        with self.assertRaisesRegex(ValueError,"not synchronized"): self.load()

    def test_wrong_iteration_input_physics_and_nonfinite_refused(self):
        with self.assertRaisesRegex(ValueError,"2000"): self.load(smoke=False)
        with self.assertRaisesRegex(ValueError,"physics"): self.load(physics_identity={"other":True})
        self.checkpoints[1]["iteration"]=1000; self.save(1)
        with self.assertRaisesRegex(ValueError,"iteration"): self.load()
        self.checkpoints[1]["iteration"]=500
        self.checkpoints[1]["policy"]["weight"][0]=float("nan"); self.save(1)
        with self.assertRaisesRegex(ValueError,"finite"): self.load()

    def test_changed_manifest_or_contact_cache_refused(self):
        (self.root/"manifest.json").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"data/asset"): self.load()
        (self.root/"manifest.json").write_bytes(b"manifest.json")
        (self.root/"contacts.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"reference-contact"): self.load()

    def test_meta_model_specifications_are_full_formal_shapes(self):
        specs = evaluation._model_specifications(TrainConfig())
        self.assertEqual(specs["policy"]["net.0.weight"].shape,(2048,3711))
        self.assertEqual(specs["wm"]["net.0.weight"].shape,(2048,313))
        self.assertEqual(specs["wm"]["contact_head.2.weight"].shape[0],5)
        self.assertTrue(all(value.device.type=="meta" for state in specs.values() for value in state.values()))
        with self.assertRaisesRegex(ValueError,"shape/dtype"):
            evaluation._validate_state({"weight":torch.zeros(3)}, {"weight":torch.zeros(2)},"policy")


class SourceTests(unittest.TestCase):
    def test_renamed_branch_preserves_exact_scientific_source_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/"frozen"; runtime=root/"eval"
            for base in (source,runtime):
                (base/"hoi_tables").mkdir(parents=True)
                (base/"hoi_tables/networks.py").write_bytes(b"unchanged direct teacher\n")
            manifest={"branch":"hoi-retarget/tables-boxes-v1.0","files":{"hoi_tables/networks.py":evaluation.file_sha256(source/"hoi_tables/networks.py")}}
            (source/"SOURCE_MANIFEST.json").write_text(json.dumps(manifest))
            self.assertEqual(len(evaluation.validate_source_compatibility(source,runtime)["matched_scientific_files"]),1)
            (runtime/"hoi_tables/networks.py").write_bytes(b"drift")
            with self.assertRaisesRegex(ValueError,"scientific source"): evaluation.validate_source_compatibility(source,runtime)
            manifest["branch"]="unrelated-branch"
            (source/"SOURCE_MANIFEST.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,"source manifest"): evaluation.validate_source_compatibility(source,runtime)

    def test_frozen_source_manifest_and_runtime_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/"frozen"; runtime=root/"eval"
            for base in (source,runtime):
                (base/"hoi_tables").mkdir(parents=True)
                (base/"hoi_tables/networks.py").write_bytes(b"frozen scientific source\n")
            manifest={"branch":"hoi-retarget/tables-v1.0","files":{"hoi_tables/networks.py":evaluation.file_sha256(source/"hoi_tables/networks.py")}}
            (source/"SOURCE_MANIFEST.json").write_text(json.dumps(manifest))
            identity=evaluation.validate_source_compatibility(source,runtime)
            self.assertEqual(len(identity["matched_scientific_files"]),1)
            (runtime/"hoi_tables/networks.py").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"scientific source"): evaluation.validate_source_compatibility(source,runtime)
            (source/"hoi_tables/networks.py").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"Frozen training source changed"): evaluation.validate_source_compatibility(source,runtime)

    def test_evaluation_import_does_not_load_torch_or_native_backend(self):
        program="import sys; import hoi_tables.evaluate; assert 'torch' not in sys.modules; assert 'isaaclab' not in sys.modules"
        result=subprocess.run([sys.executable,"-c",program],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)


class Motion:
    partition="protected_eval"; for_training=False
    def __init__(self, steps):
        self.clip_start=torch.tensor([0]); self.clip_end=torch.tensor([steps]); self.clip_len=torch.tensor([steps+1])
        self.clip_id=torch.zeros(steps+1,dtype=torch.long)
        self.entries=[{"clip_id":"omomo/unitree_g1/sub1_largetable_eval","source_group":"source-group",
                       "object_asset":"largetable"}]
    def ref_state(self,f):
        p=torch.zeros((*f.shape,3)); p[...,2]=1
        z=torch.zeros_like(p); R=torch.eye(3).expand(*f.shape,3,3); q=torch.zeros(*f.shape,29)
        return p,R,z,z,q,q
    def object_ref(self,f):
        return self.ref_state(f)[:4]
    def reference_bodies(self,f):
        S=self.ref_state(f)
        return tuple(v[...,None,:].expand(*f.shape,14,3) if v.ndim==len(f.shape)+1 else v[...,None,:,:].expand(*f.shape,14,3,3) for v in S[:4])


class FrozenPolicy(nn.Module):
    def __init__(self):
        super().__init__(); self.weight=nn.Parameter(torch.zeros(29)); self.register_buffer("norm_count",torch.tensor(7.))
    def forward(self,oracle,points):
        if torch.is_grad_enabled(): raise AssertionError("evaluation must disable autograd")
        return self.weight[None].expand(len(oracle),-1)


class Environment:
    N=1; eval_mode=True; sampler=None; device="cpu"
    randomizer=SimpleNamespace(enabled=False)
    def __init__(self,motion,autoreset=False):
        self.motion=motion; self.f=torch.zeros(1,dtype=torch.long); self.episode_id=torch.zeros(1,dtype=torch.long)
        self.reset_count=0; self.autoreset=autoreset; self.calls=0
        self.reward_model=SimpleNamespace(object_failure_threshold=lambda f:torch.ones_like(f,dtype=torch.float32))
    def reset(self,ids,frames):
        self.f=frames.clone(); self.episode_id+=1; self.reset_count+=1
    def refresh(self): pass
    def state(self): return self.motion.ref_state(self.f)
    def object_state(self): return self.motion.object_ref(self.f)
    def snapshot(self):
        xi=torch.zeros(1,33); xi[:,0]=1
        actuator=torch.ones(1,88); actuator[:,:29]=0; actuator[:,29]=3
        return {"f":self.f.clone(),"xi":xi,"actuator":actuator,"xi_o":torch.tensor([[6.,0,0,0,1,1,1,0,0,0,.9,1.]]),
                "a_prev":torch.zeros(1,29),"q_prev":torch.zeros(1,29),"qd_prev":torch.zeros(1,29),
                "object_history":torch.zeros(1,10,9)}
    def step(self,action):
        self.calls+=1; self.f+=1
        failed=self.f==2
        if self.autoreset and failed: self.episode_id+=1
        return {"terminated":failed,"done":failed | (self.f>=self.motion.clip_end[0]),
                "truncated":self.f>=500,"reward":torch.tensor([.15]),"terms":{"total":torch.tensor([.15])}}


class RolloutTests(unittest.TestCase):
    def run_clip(self,steps,autoreset=False):
        motion=Motion(steps); env=Environment(motion,autoreset); policy=FrozenPolicy().eval().requires_grad_(False)
        kin=SimpleNamespace(forward=lambda *S:tuple(v[:,None].expand(-1,14,*v.shape[1:]) for v in S[:4]))
        mdp=SimpleNamespace(kin=kin,termination=lambda S,f:f==2)
        asset={"physical_profile":{"mass":6.,"com":[0,0,0],"inertia":torch.eye(3).tolist(),"friction":.9}}
        before=evaluation.state_dict_hash(policy.state_dict())
        with patch("hoi_residual.features.torso_pose",side_effect=lambda kin,S:(S[0],S[1])):
            result=evaluation.rollout_clip(env,motion,mdp,lambda *args:torch.zeros(1,3371),policy,0,None,
                                           torch.zeros(1,128,3),asset)
        self.assertEqual(evaluation.state_dict_hash(policy.state_dict()),before)
        return result,env
    def test_failure_does_not_reset_or_shorten_reference_over500_steps(self):
        report,env=self.run_clip(602)
        self.assertEqual(env.reset_count,1)
        self.assertEqual(env.calls,602)
        self.assertEqual(report["control_steps"],602)
        self.assertEqual(report["state_frames"],603)
        self.assertEqual(report["final_frame"],602)
        self.assertTrue(report["failed"])
        self.assertEqual(report["first_failure_step"],2)
        self.assertEqual(report["failure_free_steps"],1)
        self.assertEqual(report["reset_calls_during_rollout"],0)
        self.assertAlmostEqual(report["normalized_return"],1.,places=5)
    def test_auto_reset_is_protocol_failure(self):
        with self.assertRaisesRegex(RuntimeError,"reset an episode"): self.run_clip(5,autoreset=True)
    def test_randomized_context_rejected(self):
        motion=Motion(3);env=Environment(motion);snapshot=env.snapshot();snapshot["xi"][:,0]=.3
        with self.assertRaisesRegex(RuntimeError,"randomized"):
            evaluation.validate_nominal_snapshot(env,snapshot,{"physical_profile":{}})


def asset_reports(root):
    all_clips=[]
    for asset in evaluation.ASSETS:
        for i in range(11):
            steps=i+2
            all_clips.append({"clip_id":f"protected/{asset}/{i}","asset":asset,"partition":"protected_eval",
                              "state_frames":steps+1,"control_steps":steps,"reference_control_steps":steps,
                              "start_frame":0,"reference_start_frame":0,"final_frame":steps,"reference_end_frame":steps,
                              "full_reference_duration":True,"no_failure_reset":True,"reset_calls_during_rollout":0,
                              "failed":i==0,"first_failure_step":1 if i==0 else None,"failure_free_seconds":steps*.02,
                              "return":steps*.1,"normalized_return":2/3,"tracking_quality_fraction":.8,
                              **{name:{"mean":.1,"max":.2} for name in ("object_position_error_m","object_orientation_error_rad","tracked_body_position_error_m")}})
    reports=[]
    for rank,asset in enumerate(evaluation.ASSETS):
        clips=[c for c in all_clips if c["asset"]==asset]
        value={"schema":evaluation.ASSET_SCHEMA,"passed":True,"asset":asset,"clip_count":11,"control_steps":sum(c["control_steps"] for c in clips),
               "iteration":2000,"parent_run_id":"frozenrun","checkpoint_sha256":{"rank_00":"0"*64,"rank_01":"1"*64},
               "model_state_sha256":{"policy":"same"},"source_identity":{"frozen":True},"smoke":False,"input_sha256":{"manifest":"manifest"},
               "manifest_sha256":"manifest","eval_source_sha256":"source","checkpoint_identity":{"same":True},
               "expected_clip_ids":sorted(c["clip_id"] for c in all_clips),"expected_control_steps":sum(c["control_steps"] for c in all_clips),
               "expected_clips":{c["clip_id"]:{"asset":c["asset"],"state_frames":c["state_frames"],"control_steps":c["control_steps"]} for c in all_clips},
               "policy_state_sha256_before":"same","policy_state_sha256_after":"same","clips":clips,
               **dict.fromkeys(("no_normalizer_update","no_optimizer_update","full_reference_duration","no_failure_reset"),True),
               **dict.fromkeys(("optimizer_updates","normalizer_updates","replay_updates","rsi_updates"),0)}
        folder=root/f"asset_{rank:02d}";folder.mkdir()
        (folder/"report.json").write_text(json.dumps(value));reports.append(value)
    return reports


class AggregateTests(unittest.TestCase):
    def test_full22coverage_metrics_and_write_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);asset_reports(root)
            result=evaluation.aggregate_reports(root,root/"result.json")
            self.assertEqual(result["clip_count"],22)
            self.assertEqual(result["per_asset"]["largetable"]["clip_count"],11)
            self.assertAlmostEqual(result["metrics"]["success_rate"],10/11)
            self.assertTrue(result["scheduled_evaluation"])
            with self.assertRaises(FileExistsError): evaluation.aggregate_reports(root,root/"result.json")
    def test_missing_truncated_reset_or_mixed_checkpoint_refused(self):
        mutations=[lambda r:r["clips"][0].update(control_steps=1),
                   lambda r:r["clips"][0].update(reset_calls_during_rollout=1),
                   lambda r:r.update(iteration=4000),
                   lambda r:r["checkpoint_sha256"].update(rank_00="changed"),
                   lambda r:r.update(policy_state_sha256_after="changed"),
                   lambda r:r["clips"].pop()]
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);reports=asset_reports(root);mutate(reports[1]);(root/"asset_01/report.json").write_text(json.dumps(reports[1]))
                with self.assertRaises(ValueError): evaluation.aggregate_reports(root)


if __name__=="__main__":unittest.main()
