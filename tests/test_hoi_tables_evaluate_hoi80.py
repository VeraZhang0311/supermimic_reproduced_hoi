"""Eight-checkpoint identity, canonical quotas and untruncated direct evaluation."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hoi_tables import evaluate_hoi80 as evaluation
from hoi_tables.config import TrainConfig
from hoi_tables.networks import feature_schema
from tests.test_hoi_tables_evaluate import Motion,Environment,FrozenPolicy
from tests.test_hoi_tables_trajectories import fixture_trajectory


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        for name in ("manifest","assets","geometry","contacts"): (self.root/name).write_bytes(name.encode())
        self.shape={"bank_sha256":"shape"}; self.spec={"weight":torch.zeros(2),"norm.count":torch.zeros(())}
        config=TrainConfig()
        self.identity={"schema":feature_schema(),"physics":{"frozen":True},"global_config":config.to_dict(),
                       "worker_config":replace(config,num_envs=512,wm_batch=256,pi_batch=512).to_dict(),
                       "world_size":8,"from_scratch":True,"base_checkpoint":None,"smoke":False,
                       "schedule_scope":"500WM_then_fixed_actor_critic","manifest_sha256":evaluation.file_sha256(self.root/"manifest"),
                       "asset_catalog_sha256":evaluation.file_sha256(self.root/"assets"),"shape_bank":self.shape,
                       "reward":{"frozen_reference_contacts":{"file_sha256":evaluation.file_sha256(self.root/"contacts"),"selected_clips":[]}},
                       "experiment":{"profile":"ultra","assets":list(evaluation.ASSETS),"workers_per_gpu":2,
                       "geometry_bundle_sha256":evaluation.file_sha256(self.root/"geometry"),
                       "evaluation":{"kind":"canonical_hoi80","every":2000,"owning_training_run":True}}}
        self.checkpoints=[]
        for rank,asset in enumerate(evaluation.ASSETS):
            folder=self.root/f"rank_{rank:02d}";folder.mkdir()
            identity={**deepcopy(self.identity),"asset":asset,"asset_id":rank}
            identity["reward"]["frozen_reference_contacts"]["selected_clips"]=[asset]
            self.checkpoints.append({"schema":"hoi_tables_checkpoint_v1","iteration":2000,"identity":identity,
                                     **{name:deepcopy(self.spec) for name in ("policy","wm","critic")}})
            self.save(rank)

    def save(self,rank): torch.save(self.checkpoints[rank],self.root/f"rank_{rank:02d}/checkpoint_002000.pt")

    def load(self):
        with patch.object(evaluation,"_model_specifications",return_value={name:self.spec for name in ("policy","wm","critic")}):
            return evaluation.load_checkpoints(self.root,2000,manifest=self.root/"manifest",asset_bundle=self.root/"assets",
                geometry_bundle=self.root/"geometry",shape_metadata=self.shape,reference_contacts=self.root/"contacts",physics={"frozen":True},source={})

    def test_all_eight_states_and_rms_must_match(self):
        result=self.load()
        self.assertEqual(set(result["checkpoint_sha256"]),{f"rank_{i:02d}" for i in range(8)})
        self.checkpoints[7]["policy"]["norm.count"]+=1;self.save(7)
        with self.assertRaisesRegex(ValueError,"not synchronized"): self.load()

    def test_explicit_continuation_is_evaluated_on_same_full80_contract(self):
        from hoi_tables.resume import lineage
        from hoi_tables.solver_health import policy
        parent={'iteration':500,'parent_run_id':'parent00','bundle_sha256':'a'*64,
                'records':[{'rank':i,'sha256':str(i)*64} for i in range(8)]}
        for rank in range(8):
            self.checkpoints[rank]['identity'].update(from_scratch=False,base_checkpoint=lineage(parent),solver_health=policy(200))
            self.save(rank)
        self.assertEqual(len(self.load()['checkpoint_sha256']),8)
        self.checkpoints[7]['identity']['base_checkpoint']['repeat_wm_warmup']=True;self.save(7)
        with self.assertRaisesRegex(ValueError,'continuation'):self.load()

    def test_wrong_asset_worker_batch_geometry_or_milestone_rejected(self):
        for key,value,match in (("asset_id",6,"protocol"),("world_size",2,"protocol"),("worker_config",TrainConfig().to_dict(),"worker batches")):
            old=self.checkpoints[7]["identity"][key];self.checkpoints[7]["identity"][key]=value;self.save(7)
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,match): self.load()
            self.checkpoints[7]["identity"][key]=old;self.save(7)
        (self.root/"geometry").write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError,"geometry"): self.load()


def reports(root):
    clips=[];index=0
    for asset,count in zip(evaluation.ASSETS,[5,15,6,14,4,16,11,9]):
        for _ in range(count):
            steps=629 if index==0 else 313 if index==1 else 292
            clips.append({"clip_id":f"ultra/clip_{index}","canonical_case_id":f"canonical_{index}","canonical_index":index,
                "asset":asset,"partition":"protected_eval","state_frames":steps+1,"control_steps":steps,"reference_control_steps":steps,
                "start_frame":0,"reference_start_frame":0,"final_frame":steps,"reference_end_frame":steps,"full_reference_duration":True,
                "no_failure_reset":True,"reset_calls_during_rollout":0,"failed":True,"failure_free_seconds":.02,
                "return":steps*.1,"normalized_return":2/3,"tracking_quality_fraction":.8,
                "human_pass":True,"object_pass":False,"joint_pass":False,"scored_frames":steps,
                "first_scored_state_index":1,"final_scored_state_index":steps,
                **{name:{"mean":.1,"max":.2} for name in ("object_position_error_m","object_orientation_error_rad","tracked_body_position_error_m")}})
            index+=1
    expected={c["clip_id"]:{"asset":c["asset"],"state_frames":c["state_frames"],"control_steps":c["control_steps"]} for c in clips}
    result=[]
    for rank,asset in enumerate(evaluation.ASSETS):
        local=[c for c in clips if c["asset"]==asset]
        report={"schema":evaluation.ASSET_SCHEMA,"passed":True,"asset":asset,"clip_count":len(local),"clips":local,
            "control_steps":sum(c["control_steps"] for c in local),"iteration":2000,"parent_run_id":"training-owner",
            "checkpoint_sha256":{f"rank_{i:02d}":str(i)*64 for i in range(8)},"model_state_sha256":{"policy":"same"},
            "source_identity":{"frozen":True},"smoke":False,"input_sha256":{},"checkpoint_identity":{},"manifest_sha256":"manifest",
            "canonical_sha256":evaluation.CANONICAL_SHA256,"eval_source_sha256":"source","expected_clip_ids":sorted(expected),
            "expected_clips":expected,"expected_control_steps":23718,"protocol_files_sha256":evaluation.PROTOCOL_FILES,
            "policy_state_sha256_before":"same","policy_state_sha256_after":"same",
            **dict.fromkeys(("no_normalizer_update","no_optimizer_update","full_reference_duration","no_failure_reset"),True),
            **dict.fromkeys(("optimizer_updates","normalizer_updates","replay_updates","rsi_updates"),0)}
        folder=root/f"asset_{rank:02d}";folder.mkdir()
        report['trajectory_schema'] = evaluation.TRAJECTORY_SCHEMA
        for clip in local: clip['trajectory'] = fixture_trajectory(folder,clip,report)
        (folder/"report.json").write_text(json.dumps(report));result.append(report)
    return result


class CoverageTests(unittest.TestCase):
    def test_heterogeneous_full80_and_separate_phc_survival_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);reports(root)
            result=evaluation.aggregate_reports(root,root/"result.json")
            self.assertEqual((result["clip_count"],result["control_steps"]),(80,23718))
            self.assertEqual([v["clip_count"] for v in result["per_asset"].values()],[5,15,6,14,4,16,11,9])
            self.assertEqual(result["metrics"]["human_pass_rate"],1.)
            self.assertEqual(result["metrics"]["joint_pass_rate"],0.)
            self.assertEqual(result["metrics"]["tables_reward_survival_rate"],0.)
            self.assertEqual(max(c["control_steps"] for c in result["clips"]),629)
            self.assertEqual((result['trajectory_count'],result['recorded_state_frames']),(80,23798))

    def test_truncated_629_clip_or_training_update_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);values=reports(root)
            values[0]["clips"][0]["control_steps"]=500
            values[0]["control_steps"]-=129
            (root/"asset_00/report.json").write_text(json.dumps(values[0]))
            with self.assertRaisesRegex(ValueError,"truncated"): evaluation.aggregate_reports(root)

    def test_missing_native_trajectory_prevents_complete_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);values=reports(root)
            (root/values[0]['clips'][0]['trajectory']['path']).unlink()
            with self.assertRaisesRegex(ValueError,'missing'): evaluation.aggregate_reports(root)

    def test_canonical_recorder_initializes_after_reset_and_scores629_after_failure(self):
        motion=Motion(629);motion.entries[0]["clip_id"]="ultra/canonical-test"
        env=Environment(motion);history_calls=[]
        env.history=SimpleNamespace(reset=lambda ids,x:history_calls.append(env.calls))
        policy=FrozenPolicy().eval().requires_grad_(False)
        kin=SimpleNamespace(forward=lambda *S:tuple(v[:,None].expand(-1,14,*v.shape[1:]) for v in S[:4]))
        mdp=SimpleNamespace(kin=kin,termination=lambda S,f:f==2)
        asset={"physical_profile":{"mass":6.,"com":[0,0,0],"inertia":torch.eye(3).tolist(),"friction":.9}}
        native=ModuleType("hoi_zero_shot.native")
        events=[]
        def initialize(e,reference):
            events.append((e.reset_count,e.calls,reference.shape));return {"physics_steps":0}
        native.initialize_canonical_object=initialize
        recorder=evaluation.CanonicalRecorder(env,torch.zeros(630,630).numpy(),None)
        with patch.dict(sys.modules,{"hoi_zero_shot.native":native}),patch("hoi_residual.features.torso_pose",side_effect=lambda kin,S:(S[0],S[1])):
            result=evaluation.rollout_clip(recorder,motion,mdp,lambda *a:torch.zeros(1,3371),policy,0,None,torch.zeros(1,128,3),asset)
        self.assertEqual(events,[(1,0,(1,630))]);self.assertEqual(history_calls,[0])
        self.assertEqual((env.calls,len(recorder.states),len(recorder.objects)),(629,630,630))
        self.assertEqual(result["first_failure_step"],2)
        self.assertEqual(result["final_frame"],629)

    def test_import_stays_stdlib_and_no_frozen_gmt(self):
        program="import sys; import hoi_tables.evaluate_hoi80; assert 'torch' not in sys.modules; assert 'isaaclab' not in sys.modules; assert 'hoi_residual.policy' not in sys.modules"
        result=subprocess.run([sys.executable,"-c",program],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=="__main__": unittest.main()
