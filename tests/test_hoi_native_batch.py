"""Synthetic orchestration tests. Mocked native children are never GPU evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from hoi.v1 import native_batch as b
from hoi.data.native_assets import digest
from hoi.v1.native_input_contract import SCHEMA as INPUT_SCHEMA


class NativeBatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.source=self.root/"source";(self.source/"hoi/v1").mkdir(parents=True)
        (self.source/"hoi/v1/native.py").write_text("# synthetic executable identity only")
        (self.source/"hoi/v1/native_snapshot_render.py").write_text("# synthetic renderer identity only")
        self.python=self.root/"python";self.python.write_bytes(b"synthetic Python identity")
        self.blob=self.root/"data";self.blob.write_bytes(b"synthetic bound asset")
        self.binding=b._identity(self.blob)
        self.versions={k:"synthetic" for k in ("torch","warp","mujoco","mujoco_warp","newton")}
        self.patch=patch.object(b,"_runtime_versions",return_value=self.versions);self.patch.start();self.addCleanup(self.patch.stop)
        self.bundles={};self.paths=[]
        for partition,aids in (("train",[0,1]),("protected_eval",[1,2])):
            clips=[]
            for aid in aids:
                folder=self.root/f"{partition}_{aid}";folder.mkdir()
                (folder/"metadata.json").write_bytes(b"{}");(folder/"labels.npz").write_bytes(b"synthetic")
                clips.append({"asset_id":aid,"name":f"{partition}_{aid}","resolved_source_output":str(folder),
                    "metadata_sha256":b.sha(folder/"metadata.json"),"labels_sha256":b.sha(folder/"labels.npz"),
                    "reference_cache":self.binding,"contact_cache":self.binding})
            assets=[{"asset_id":aid,"native_asset":str(self.blob),"native_asset_sha256":self.binding["sha256"],
                     "geometry_cache":self.binding,"physical_profile":{"mass":float(aid+1)}} for aid in range(4)]
            bundle={"partition":partition,"scope":{"purpose":"curated_reference_bundle","is_subset":False},"clips":clips,"assets":assets,
                    "manifest":self.binding,"recipes":self.binding,"native_robot_asset":self.binding,"reference_kinematics":{"urdf":self.binding},"hands":{"geometry_cache":self.binding}}
            bundle["bundle_content_sha256"]=digest(bundle);path=self.root/f"{partition}.json";b._write(path,bundle)
            self.paths.append(path);self.bundles[str(path)]=bundle
        self.header=patch("hoi.data.curated_bundle.read_curated_bundle_header",side_effect=lambda p:deepcopy(self.bundles[str(p)]))
        self.header.start();self.addCleanup(self.header.stop)
        self.plan=b.make_plan(self.paths,self.source,self.python,["--hand-collision","--self-collision","--capture-smoke-visual","--audit-scene-resets"],self.root/"batch")
        self.plan_path=Path(self.plan["output"])/"plan.json"

    def fake_child(self,command,**kwargs):
        out=Path(command[command.index("--output")+1]);out.mkdir()
        aid=int(command[command.index("--asset-id")+1]);bundle=self.bundles[command[command.index("--bundle")+1]]
        (out/"compiled_geometry").mkdir()
        geometry=out/"compiled_geometry/assets.json";b._write(geometry,{"synthetic":True})
        inp=out/"input.pt";inp.write_bytes(b"synthetic non-native test input")
        contract={"schema":INPUT_SCHEMA,"passed":True,"cuda_verified":True,"num_envs":8,"normalizer_updates":0,"optimizer_steps":0,
                  "additional_physics_steps":0,"rng_preserved":True,"outputs_applied_to_simulator":False,"untrained_validation_only":True,"artifact":b._identity(inp)}
        report={"asset_id":aid,"num_envs":8,"steps":64,"seed":self.plan["execution"]["seed"],"formal_training_started":False,
            "source_code_sha256":self.plan["execution"]["source_files"],"native_runtime":{"versions":self.versions},
            "bundle_content_sha256":bundle["bundle_content_sha256"],"pair_probes":{f"synthetic_case_{i}":{"ticks":8} for i in range(31)},
            "native_input_contract":contract,"compiled_geometry_export":str(geometry)}
        report["native_scene_supports"]={"declared_support_ids":[],"clips":[{"name":c["name"],"bindings":[]} for c in bundle["clips"] if c["asset_id"]==aid]}
        contact={"topology":"one_mujoco_world_per_environment","geom_names":["synthetic_geom"],"geom_roles":[0]}
        report["contact"]=contact
        visual_root=out/"smoke_visual";visual_root.mkdir();(visual_root/"render").mkdir()
        binary=visual_root/"compiled_model.mjb";binary.write_bytes(b"synthetic test binary, never native evidence")
        live=visual_root/"compiled_model_live_arrays.pt";live.write_bytes(b"synthetic test live arrays, never native evidence")
        manifest={"schema":"hoi_actual_native_model_dump_v1","physics_steps":0,
            "binary":{**b._identity(binary),"collision_arrays_exact_roundtrip":True},"live_gpu_arrays":b._identity(live),
            "scene_snapshot":{"schema":"hoi_actual_native_scene_snapshot_v1","num_envs":8,"capture_phase":b.SMOKE_PHASE,"declared_support_ids":[]},
            "contact_mapping":{**contact,"geom_environment_ids":[0]}}
        manifest_path=visual_root/"compiled_model_manifest.json";b._write(manifest_path,manifest)
        preview=visual_root/"render/preview.png";preview.write_bytes(b"synthetic PNG test identity")
        rendering={"schema":"hoi_actual_native_snapshot_render_v1","manifest_sha256":b.sha(manifest_path),
            "binary_sha256":b.sha(binary),"live_arrays_sha256":b.sha(live),"preview_sha256":b.sha(preview),
            "renderer_sha256":self.plan["execution"]["source_files"]["hoi/v1/native_snapshot_render.py"],
            "environment":0,"saved_frame":3,"visible_support_ids":[],"support_geom_pose_errors":{},"physics_steps":0,"fk_calls":0,"policy_calls":0}
        rendering_path=visual_root/"render/rendering.json";b._write(rendering_path,rendering)
        report["smoke_visual_snapshot"]={"enabled":True,"captured":True,"extra_physics_steps":0,"phase":b.SMOKE_PHASE,
            "manifest":str(manifest_path),"manifest_sha256":b.sha(manifest_path),"rendering_report":str(rendering_path),"captured_existing_control_interval":1}
        if "--audit-scene-resets" in command:
            clips=[{"name":c["name"],"clip_index":i,"state_frames":3,"selected_local_frames":[0,1,2],"presence_change_frames":[],"bindings":c["bindings"]}
                   for i,c in enumerate(report["native_scene_supports"]["clips"])]
            count=sum(len(c["selected_local_frames"]) for c in clips)
            audit={"schema":"hoi_native_terminal_scene_reset_audit_v1","enabled":True,"passed":True,
                "phase":"terminal_auxiliary_after_smoke_state_discarded",
                "selection_rule":"every_asset_clip_first_middle_last_and_presence_change_neighbors_v1",
                "asset_id":aid,"num_envs":8,"clip_names":[c["name"] for c in clips],"clips":clips,"declared_support_ids":[],"formal_training_started":False,
                "requested_sample_count":count,"completed_sample_count":count,"reset_batches":(count+7)//8,
                "additional_physics_steps":0,"physics_steps_before":1248,"physics_steps_after":1248,
                "recorder_substeps_before":4,"recorder_substeps_after":4,"attempted_advance_calls":0,"attempted_sim_step_calls":0,"attempted_solver_step_calls":0,
                "reset_noise":False,"state_discarded":True,"maximum_errors":{k:0. for k in ("position_m","quaternion","linear_velocity_m_s","angular_velocity_rad_s")},"presence_mismatches":0}
            samples=[{"clip_name":c["name"],"local_frame":f,"clip_index":i,"frame":3*i+f} for i,c in enumerate(clips) for f in c["selected_local_frames"]]
            for i,sample in enumerate(samples):sample["environment_id"]=i%8
            audit_path=out/"scene_reset_audit.json";b._write(audit_path,{**audit,"samples":samples})
            report["native_scene_reset_audit"]={**audit,"artifact":b._identity(audit_path)}
        b._write(out/"native.json",report);b._write(out/"source.json",self.plan["execution"]["source_files"])
        return SimpleNamespace(returncode=0)

    @staticmethod
    def fake_gate(report,**kwargs):
        # Only mock GPU physics internals. Keep the actual strict bundle/asset
        # boundary so cross-partition orchestration bugs cannot pass this test.
        ok=report["bundle_content_sha256"]==kwargs["expected_bundle_sha256"] and [report["asset_id"]]==kwargs["required_asset_ids"]
        return {"passed":ok,"blockers":[] if ok else ["wrong bundle or asset"]}

    def test_same_asset_is_separate_per_partition_and_unused_registry_is_excluded(self):
        self.assertEqual(self.plan["task_counts"],{"train":2,"protected_eval":2})
        self.assertEqual([(t["partition"],t["asset_id"]) for t in self.plan["tasks"]],[("train",0),("train",1),("protected_eval",1),("protected_eval",2)])
        self.assertNotEqual(self.plan["tasks"][1]["task_sha256"],self.plan["tasks"][2]["task_sha256"])
        with self.assertRaisesRegex(ValueError,"owns"):
            b._native_args(["--hand-collision","--self-collision","--num-envs=1"])
        self.bundles[str(self.paths[0])]["scope"]["is_subset"]=True
        with self.assertRaisesRegex(ValueError,"subset"):
            b.make_plan(self.paths,self.source,self.python,["--hand-collision","--self-collision","--capture-smoke-visual","--audit-scene-resets"],self.root/"bad")

    def test_failed_attempt_is_preserved_retry_is_fresh_and_success_skips_child(self):
        with patch.object(b.subprocess,"run",return_value=SimpleNamespace(returncode=137)):
            self.assertEqual(b.run_task(self.plan,0)["state"],"failed")
        with patch.object(b.subprocess,"run",side_effect=self.fake_child) as child,patch("hoi.v1.gates.validate_native_evidence",side_effect=self.fake_gate) as gate:
            success=b.run_task(self.plan,0)
            self.assertEqual(success["state"],"complete");self.assertTrue(success["attempt"].endswith("attempt_0002"))
            self.assertEqual(b.run_task(self.plan,0)["state"],"already_complete");self.assertEqual(child.call_count,1)
            self.assertGreaterEqual(gate.call_count,2)
            self.assertTrue((Path(success["attempt"]).parent/"attempt_0001/FAILURE.json").exists())
            schedule=b.make_schedule(self.plan_path,self.root/"schedule.json",tasks_per_slot=2)
            self.assertEqual(schedule["slots"],[[1,2],[3]]);self.assertEqual(schedule["array_argument"],"0-1%4")
            (Path(success["attempt"])/"native/input.pt").write_bytes(b"changed")
            states=b.status(self.plan);self.assertEqual(states[0]["state"],"invalid")
            with self.assertRaisesRegex(ValueError,"invalid evidence"):
                b.make_schedule(self.plan_path,self.root/"bad_schedule.json")

    def test_cross_partition_report_and_missing_gpu_input_contract_are_rejected(self):
        with patch.object(b.subprocess,"run",side_effect=self.fake_child),patch("hoi.v1.gates.validate_native_evidence",side_effect=self.fake_gate):
            result=b.run_task(self.plan,1)
            report=b._json(Path(result["attempt"])/"native/native.json")
            with self.assertRaisesRegex(ValueError,"wrong bundle"):
                b._validate_report(self.plan,self.plan["tasks"][2],report)
            report["native_input_contract"]["cuda_verified"]=False
            with self.assertRaisesRegex(ValueError,"actual CUDA"):
                b._validate_report(self.plan,self.plan["tasks"][1],report)

    def test_source_or_inputs_changed_prevents_reuse_and_collection_requires_all(self):
        with self.assertRaisesRegex(ValueError,"coverage incomplete"):
            b.collect(self.plan_path,self.root/"aggregate")
        self.blob.write_bytes(b"modified")
        self.assertTrue(all(s["state"]=="invalid" for s in b.status(self.plan)))
        (self.source/"hoi/v1/native.py").write_text("# changed")
        with self.assertRaisesRegex(ValueError,"source changed"):
            b.read_plan(self.plan_path)

    def test_schedule_rejects_duplicate_producers(self):
        schedule=b._seal({"schema":b.SCHEDULE_SCHEMA,"plan":b._identity(self.plan_path),"slots":[[0,0]],"maximum_concurrent_slots":4})
        path=self.root/"duplicates.json";b._write(path,schedule)
        with self.assertRaisesRegex(ValueError,"duplicate"):
            b.run_slot(path,0)

    def _successful_report(self,index=0):
        with patch.object(b.subprocess,"run",side_effect=self.fake_child),patch("hoi.v1.gates.validate_native_evidence",side_effect=self.fake_gate):
            result=b.run_task(self.plan,index)
        self.assertEqual(result["state"],"complete",result)
        attempt=Path(result["attempt"])
        return attempt,b._json(attempt/"native/native.json")

    def test_full_plan_requires_capture_and_read_plan_rechecks_flag(self):
        with self.assertRaisesRegex(ValueError,"capture-smoke-visual"):
            b._native_args(["--hand-collision","--self-collision"])
        plan=deepcopy(self.plan);plan["execution"]["native_args"].remove("--capture-smoke-visual")
        plan["execution_sha256"]=digest(plan["execution"])
        plan=b._seal({k:v for k,v in plan.items() if k!="content_sha256"})
        path=self.root/"uncaptured_plan.json";b._write(path,plan)
        with self.assertRaisesRegex(ValueError,"capture-smoke-visual"):
            b.read_plan(path)

    def test_all_visual_artifacts_are_bound_and_mutation_invalidates_resume(self):
        attempt,report=self._successful_report()
        success=b._json(attempt/"SUCCESS.json")
        artifacts=b._visual_artifacts(self.plan,report)
        self.assertEqual(set(artifacts),{"smoke_manifest","smoke_binary","smoke_live_arrays","smoke_rendering","smoke_preview"})
        self.assertTrue(all(success["artifacts"][k]==v for k,v in artifacts.items()))
        with patch("hoi.v1.gates.validate_native_evidence",side_effect=self.fake_gate):
            for key,binding in artifacts.items():
                with self.subTest(artifact=key):
                    path=Path(binding["path"]);original=path.read_bytes()
                    try:
                        path.write_bytes(original+b" changed")
                        self.assertEqual(b.status(self.plan)[0]["state"],"invalid")
                        with self.assertRaisesRegex(ValueError,"invalid evidence"):
                            b.make_schedule(self.plan_path,self.root/f"schedule_{key}.json")
                    finally:path.write_bytes(original)
            self.assertEqual(b.status(self.plan)[0]["state"],"complete")
            del success["artifacts"]["smoke_preview"]
            success=b._seal({k:v for k,v in success.items() if k!="content_sha256"})
            (attempt/"SUCCESS.json").write_text(json.dumps(success),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"does not bind"):
                b._success(self.plan,self.plan["tasks"][0],attempt)

    def test_rendering_provenance_and_no_added_work_are_checked(self):
        _,report=self._successful_report();path=Path(report["smoke_visual_snapshot"]["rendering_report"])
        original=b._json(path)
        changes={"renderer_sha256":"0"*64,"manifest_sha256":"0"*64,"binary_sha256":"0"*64,
            "live_arrays_sha256":"0"*64,"preview_sha256":"0"*64,"physics_steps":1,"fk_calls":1,"policy_calls":1,
            "environment":1,"saved_frame":-1,"visible_support_ids":["table1"],"support_geom_pose_errors":{"table1":{}}}
        for key,value in changes.items():
            with self.subTest(field=key):
                modified={**original,key:value};path.write_text(json.dumps(modified),encoding="utf-8")
                with self.assertRaises(ValueError):b._visual_artifacts(self.plan,report)
        path.write_text(json.dumps(original),encoding="utf-8")
        for key,value in (("captured",False),("extra_physics_steps",1),("phase","initial_model_export"),("captured_existing_control_interval",2)):
            with self.subTest(capture=key):
                modified=deepcopy(report);modified["smoke_visual_snapshot"][key]=value
                with self.assertRaises(ValueError):b._visual_artifacts(self.plan,modified)

    def test_initial_or_other_capture_and_mismatched_mapping_are_rejected(self):
        _,report=self._successful_report();path=Path(report["smoke_visual_snapshot"]["manifest"]);original=b._json(path)
        for change in ("phase","worlds","mapping","geom_environment","roundtrip","file_location"):
            with self.subTest(change=change):
                manifest=deepcopy(original)
                if change=="phase":manifest["scene_snapshot"]["capture_phase"]="initial_model_export"
                elif change=="worlds":manifest["scene_snapshot"]["num_envs"]=1
                elif change=="mapping":manifest["contact_mapping"]["geom_names"]=["wrong"]
                elif change=="geom_environment":manifest["contact_mapping"]["geom_environment_ids"]=[8]
                elif change=="roundtrip":manifest["binary"]["collision_arrays_exact_roundtrip"]=False
                else:manifest["live_gpu_arrays"]["path"]=str(self.blob)
                path.write_text(json.dumps(manifest),encoding="utf-8")
                changed=deepcopy(report);changed["smoke_visual_snapshot"]["manifest_sha256"]=b.sha(path)
                with self.assertRaises(ValueError):b._visual_artifacts(self.plan,changed)
        changed=deepcopy(report);changed["smoke_visual_snapshot"]["manifest"]=str(self.root/"other/compiled_model_manifest.json")
        with self.assertRaisesRegex(ValueError,"outside this native attempt"):
            b._visual_artifacts(self.plan,changed)

    def test_support_report_covers_all_asset_clips_including_empty_supports(self):
        _,report=self._successful_report();task=deepcopy(self.plan["tasks"][0]);task["clip_names"].append("another_no_support_clip")
        correct=[*report["native_scene_supports"]["clips"],{"name":"another_no_support_clip","bindings":[]}]
        plan=deepcopy(self.plan);plan["execution"]["native_args"].remove("--audit-scene-resets")
        with patch("hoi.v1.gates.validate_native_evidence",side_effect=self.fake_gate):
            for rows in (correct[:1],[correct[0],correct[0]],[correct[0],{"name":"wrong","bindings":[]}],correct+[correct[0]]):
                report["native_scene_supports"]["clips"]=rows
                with self.assertRaisesRegex(ValueError,"every planned asset clip"):
                    b._validate_report(plan,task,report)
            report["native_scene_supports"]["clips"]=list(reversed(correct))
            self.assertTrue(b._validate_report(plan,task,report)["passed"])

    def test_collect_binds_all_tasks_to_their_own_partition_bundle(self):
        calls=[]
        def aggregate(argv):
            bundle_path=argv[argv.index("--bundle")+1];bundle=self.bundles[bundle_path]
            reports=[b._json(argv[i+1]) for i,value in enumerate(argv) if value=="--native-report"]
            self.assertEqual(sorted(r["asset_id"] for r in reports),sorted({c["asset_id"] for c in bundle["clips"]}))
            self.assertTrue(all(r["bundle_content_sha256"]==bundle["bundle_content_sha256"] for r in reports))
            calls.append(bundle["partition"])
            output=Path(argv[argv.index("--output")+1]);output.mkdir()
            b._write(output/"gate.json",{"passed":True,"synthetic":True})
            b._write(output/"native_evidence.json",{"synthetic":True,"reports":reports})
            return 0
        with patch.object(b.subprocess,"run",side_effect=self.fake_child),patch("hoi.v1.gates.validate_native_evidence",side_effect=self.fake_gate),patch("hoi.v1.gates.main",side_effect=aggregate):
            for index in range(len(self.plan["tasks"])):
                self.assertEqual(b.run_task(self.plan,index)["state"],"complete")
            summary=b.collect(self.plan_path,self.root/"full_aggregate")
        self.assertEqual(calls,["train","protected_eval"])
        self.assertEqual({p:row["asset_count"] for p,row in summary["partitions"].items()},{"train":2,"protected_eval":2})
        self.assertTrue(summary["passed"]);self.assertFalse(summary["training_started"])

    def test_requested_terminal_audit_is_required_and_bound_on_resume(self):
        attempt,report=self._successful_report();task=self.plan["tasks"][0]
        artifacts=b._scene_reset_artifacts(self.plan,task,report)
        self.assertEqual(set(artifacts),{"scene_reset_audit"})
        self.assertEqual(b._json(attempt/"SUCCESS.json")["artifacts"]["scene_reset_audit"],artifacts["scene_reset_audit"])
        missing=deepcopy(report);del missing["native_scene_reset_audit"]
        with self.assertRaisesRegex(ValueError,"requested terminal"):
            b._scene_reset_artifacts(self.plan,task,missing)
        legacy_plan=deepcopy(self.plan);legacy_plan["execution"]["native_args"].remove("--audit-scene-resets")
        self.assertEqual(b._scene_reset_artifacts(legacy_plan,task,missing),{})
        Path(artifacts["scene_reset_audit"]["path"]).write_bytes(b"changed actual readback details")
        with patch("hoi.v1.gates.validate_native_evidence",side_effect=self.fake_gate):
            self.assertEqual(b.status(self.plan)[0]["state"],"invalid")

    def test_terminal_audit_rejects_extra_steps_bad_coverage_and_wrong_provenance(self):
        _,report=self._successful_report();task=self.plan["tasks"][0]
        for key,value in (("additional_physics_steps",1),("physics_steps_after",1249),("recorder_substeps_after",5),
                ("attempted_advance_calls",1),("attempted_sim_step_calls",1),("attempted_solver_step_calls",1),("presence_mismatches",1),
                ("state_discarded",False),("completed_sample_count",2),("reset_batches",2),("clip_names",["wrong"])):
            with self.subTest(field=key):
                changed=deepcopy(report);changed["native_scene_reset_audit"][key]=value
                with self.assertRaises(ValueError):b._scene_reset_artifacts(self.plan,task,changed)
        for change in ("selected","bindings","error"):
            with self.subTest(change=change):
                changed=deepcopy(report);audit=changed["native_scene_reset_audit"]
                if change=="selected":audit["clips"][0]["selected_local_frames"]=[0,2]
                elif change=="bindings":audit["clips"][0]["bindings"]=[{"support_id":"unbound"}]
                else:audit["maximum_errors"]["position_m"]=.001
                with self.assertRaises(ValueError):b._scene_reset_artifacts(self.plan,task,changed)
        changed=deepcopy(report);binding=changed["native_scene_reset_audit"]["artifact"];path=Path(binding["path"])
        detail=b._json(path);detail["samples"]=detail["samples"][:-1]
        path.write_text(json.dumps(detail),encoding="utf-8");binding["sha256"]=b.sha(path)
        with self.assertRaisesRegex(ValueError,"detailed artifact"):
            b._scene_reset_artifacts(self.plan,task,changed)

    def test_terminal_audit_producer_and_batch_agree_on_real_helper_contract(self):
        # Real audit implementation with mocked CPU state writes; not GPU evidence.
        from tests.test_hoi_native_scene_reset_audit import fixture
        from hoi.v1.native_scene_reset_audit import audit_scene_resets
        env=fixture();env.N=8
        env.scene.env_origins=env.scene.env_origins.repeat(4,1);env.f=env.f.repeat(4)
        env.support_reset_present=env.support_reset_present.repeat(4,1)
        for sid,support in env.supports.items():
            support.data.root_link_pose_w.torch=support.data.root_link_pose_w.torch.repeat(4,1)
            support.data.root_link_vel_w.torch=support.data.root_link_vel_w.torch.repeat(4,1)
            env.support_reset_poses[sid]=env.support_reset_poses[sid].repeat(4,1)
        root=self.root/"producer_contract";root.mkdir()
        audit=audit_scene_resets(env,root/"scene_reset_audit.json")
        task=deepcopy(self.plan["tasks"][0]);task["clip_names"]=env.motion.names
        report={"native_scene_reset_audit":audit,"compiled_geometry_export":str(root/"compiled_geometry/assets.json"),
            "native_scene_supports":{"declared_support_ids":list(env.supports),
                "clips":[{"name":c["name"],"bindings":c["bindings"]} for c in audit["clips"]]}}
        self.assertEqual(audit["requested_sample_count"],11);self.assertEqual(audit["reset_batches"],2)
        self.assertEqual(set(b._scene_reset_artifacts(self.plan,task,report)),{"scene_reset_audit"})
        detail=b._json(audit["artifact"]["path"])
        detail["samples"][1]=deepcopy(detail["samples"][0])
        Path(audit["artifact"]["path"]).write_text(json.dumps(detail),encoding="utf-8")
        audit["artifact"]["sha256"]=b.sha(audit["artifact"]["path"])
        with self.assertRaisesRegex(ValueError,"omit or replace"):
            b._scene_reset_artifacts(self.plan,task,report)
