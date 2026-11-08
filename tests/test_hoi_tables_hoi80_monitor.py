"""Canonical-80 scheduling contracts; no real scheduler, SDK or GPU calls."""
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

from hoi_tables import hoi80_monitor as monitor
from tests.test_hoi_tables_eval_monitor import FakeSlurm
from tests.test_hoi_tables_trajectories import fixture_trajectory


class HOI80MonitorTests(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).resolve().parents[1]/".cache/hoi80_monitor_tests"
        cache.mkdir(parents=True, exist_ok=True)
        self.temporary = TemporaryDirectory(dir=cache)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root/"input"
        self.inputs.mkdir()
        self.training = self.root/"formal_ultra_01"
        self.training.mkdir()
        self.control = self.root/"hoi80_every2000"
        self.source, self.training_source = self.root/"eval_source", self.root/"training_source"
        files = {}
        required = ("hoi_tables/networks.py", "hoi_tables/learner.py", "hoi_tables/train.py", "hoi_tables/ultra.py",
                    "hoi_tables/reward.py", "hoi_tables/native_checks.py")
        for name in required:
            data = ('# immutable scientific source: '+name+'\n').encode()
            files[name] = hashlib.sha256(data).hexdigest()
            for directory in (self.source, self.training_source):
                path = directory/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        (self.training_source/"SOURCE_MANIFEST.json").write_text(json.dumps({"branch": "hoi-retarget/tables-boxes-v1.0", "files": files}))
        self.helper = self.root/"frozen_helpers"
        (self.helper/"hoi_zero_shot").mkdir(parents=True)
        protocols = {}
        for name in monitor.PROTOCOL_FILES:
            path = self.helper/"hoi_zero_shot"/name; path.write_text('# frozen helper '+name)
            protocols[name] = monitor.base.sha256(path)
        self.protocol_patch = patch.object(monitor, "PROTOCOL_FILES", protocols)
        self.protocol_patch.start(); self.addCleanup(self.protocol_patch.stop)
        cases, position = [], 0
        lengths = [629]+[292]*78+[313]
        for asset, count in zip(monitor.ASSETS, monitor.ASSET_COUNTS):
            for _ in range(count):
                cases.append({"id": f"canonical_{position}", "motion": f"/source/motion_{position}.pt", "object_asset": asset,
                              "resampled_frames": lengths[position]+1, "scored_control_steps": lengths[position],
                              "motion_sha256": f"{position:064x}"})
                position += 1
        self.canonical = self.inputs/"hoi80_manifest.json"
        self.canonical.write_text(json.dumps({"cases": cases}))
        self.canonical_patch = patch.object(monitor, "CANONICAL_SHA256", monitor.base.sha256(self.canonical))
        self.canonical_patch.start(); self.addCleanup(self.canonical_patch.stop)
        for name in ("manifest.json", "assets.json", "shape_bank.json", "geometry_bundle.json", "reference_contacts.pt"):
            (self.inputs/name).write_text('{"identity":"'+name+'"}')
        metadata = {"source_sha256": "a"*64, "manifest_sha256": monitor.base.sha256(self.inputs/"manifest.json"),
                    "asset_catalog_sha256": monitor.base.sha256(self.inputs/"assets.json"),
                    "shape_bank_sha256": monitor.base.sha256(self.inputs/"shape_bank.json"),
                    "reference_contacts_sha256": monitor.base.sha256(self.inputs/"reference_contacts.pt")}
        raw = json.dumps(metadata).encode()
        (self.training/"run_metadata.json").write_bytes(raw)
        (self.training/"wandb_registration.json").write_text(json.dumps({"id": "ultra-run01", "entity": "team-name",
                    "project": "hoiretarget-boxes", "name": "ultra_v1-now", "metadata_sha256": hashlib.sha256(raw).hexdigest()}))
        self.experiment = self.root/"experiment.json"
        self.experiment.write_text(json.dumps({"schema": "hoi_direct_teacher_experiment_v1", "profile": "ultra",
                    "evaluation": {"kind": "canonical_hoi80", "every": 2000, "owning_training_run": True, "helper_root": str(self.helper)}}))
        self.slurm = FakeSlurm()

    def supervisor(self):
        return monitor.Supervisor(self.control, self.root, self.training, self.source, self.training_source,
                                  "ultra-run01", slurm=self.slurm)

    def checkpoints(self, iteration=2000, ranks=range(8)):
        for rank in ranks:
            path = self.training/f"rank_{rank:02d}/checkpoint_{iteration:06d}.pt"
            path.parent.mkdir(exist_ok=True)
            with zipfile.ZipFile(path, "w") as archive: archive.writestr("archive/data.pkl", f"rank{rank}:{iteration}")

    def request(self):
        self.checkpoints()
        self.supervisor().tick()
        return self.slurm.jobs["100"]["request"]

    @staticmethod
    def metrics(clips):
        result = dict.fromkeys(monitor.base.METRICS, .5)
        result["success_rate"] = result["tables_reward_survival_rate"] = sum(not c["failed"] for c in clips)/len(clips)
        result.update({key: sum(c[key.removesuffix("_rate")] for c in clips)/len(clips) for key in monitor.HOI80_METRICS})
        return result

    def result(self, request):
        expected = monitor.canonical_cases(self.canonical)
        clips = []
        for name, ref in expected.items():
            i, count = ref["canonical_index"], ref["control_steps"]
            clips.append({"clip_id": name, "asset": ref["asset"], "control_steps": count,
                          "reference_control_steps": count, "state_frames": count+1,
                          "canonical_case_id": ref["case_id"], "canonical_index": i, "motion_sha256": ref["motion_sha256"],
                          "start_frame": 0, "reference_start_frame": 0, "final_frame": count, "reference_end_frame": count,
                          "full_reference_duration": True, "no_failure_reset": True, "reset_calls_during_rollout": 0,
                          "scored_frames": count, "first_scored_state_index": 1, "final_scored_state_index": count,
                          "human_pass": i%2 == 0, "object_pass": i%4 == 0, "joint_pass": i%4 == 0, "failed": i%5 == 0})
        report = {"schema": monitor.SCHEMA, "passed": True, "smoke": False, "scheduled_evaluation": True,
                  "partition": "protected_eval", "clip_count": 80, "control_steps": 23718, "clips": clips,
                  "metrics": self.metrics(clips), "no_normalizer_update": True, "no_optimizer_update": True,
                  "full_reference_duration": True, "no_failure_reset": True,
                  "input_sha256": request["input_sha256"], "protocol_files_sha256": dict(monitor.PROTOCOL_FILES),
                  "model_state_sha256": {"policy": "a"*64, "wm": "b"*64, "critic": "c"*64},
                  "policy_state_sha256_before": "a"*64, "policy_state_sha256_after": "a"*64,
                  "optimizer_updates": 0, "normalizer_updates": 0, "replay_updates": 0, "rsi_updates": 0}
        report.update({key: request[key] for key in ("iteration", "parent_run_id", "checkpoint_sha256", "manifest_sha256", "canonical_sha256", "eval_source_sha256")})
        report["per_asset"] = {asset: {"clip_count": sum(c["asset"] == asset for c in clips),
                    "control_steps": sum(c["control_steps"] for c in clips if c["asset"] == asset),
                    "metrics": self.metrics([c for c in clips if c["asset"] == asset])} for asset in monitor.ASSETS}
        report.update(trajectory_schema=monitor.TRAJECTORY_SCHEMA,trajectory_count=80,recorded_state_frames=23798)
        for clip in clips:
            lane = self.control/f'asset_{monitor.ASSETS.index(clip["asset"]):02d}'
            clip['trajectory'] = fixture_trajectory(lane,clip,report)
        return report

    def test_exact_heterogeneous80_and23718_not_ten_per_asset(self):
        expected = monitor.expected_clips(self.inputs/"manifest.json")
        self.assertEqual(len(expected), 80)
        self.assertEqual(tuple(sum(c["asset"] == a for c in expected.values()) for a in monitor.ASSETS), monitor.ASSET_COUNTS)
        self.assertEqual(sum(c["control_steps"] for c in expected.values()), 23718)
        self.assertEqual(max(c["control_steps"] for c in expected.values()), 629)

    def test_waits_for_all_eight_checkpoints_and_binds_all_inputs(self):
        self.checkpoints(ranks=range(7))
        supervisor = self.supervisor(); supervisor.tick()
        self.assertEqual(self.slurm.calls, [])
        self.checkpoints(ranks=(7,)); supervisor.tick()
        request = self.slurm.jobs["100"]["request"]
        self.assertEqual(len(request["checkpoint_sha256"]), 8)
        self.assertEqual(set(request["input_sha256"]), {"manifest", "asset_bundle", "shape_bank", "geometry_bundle", "canonical", "reference_contacts"})
        self.assertEqual(request["reference_contacts"], str((self.inputs/"reference_contacts.pt").resolve()))
        self.assertEqual(len(monitor.validate_request(request)), 80)

    def test_complete_result_routes_canonical_success_separately_from_survival(self):
        request = self.request(); path = self.control/"test-result.json"
        path.write_text(json.dumps(self.result(request)))
        metrics = monitor.validate_result(path, request)
        self.assertEqual(metrics["eval/hoi80/joint_pass_rate"], .25)
        self.assertEqual(metrics["eval/tables_reward_survival_rate"], .8)
        self.assertEqual(metrics["eval/success_rate"], .8)
        self.assertEqual(metrics["eval/control_steps"], 23718)
        self.assertTrue(all(f"eval/hoi80/assets/{asset}/joint_pass_rate" in metrics for asset in monitor.ASSETS))

    def test_missing_cropped_reset_wrong_reference_and_frozen_state_fail(self):
        request = self.request(); original = self.result(request); path = self.control/"test-result.json"
        for change in ("missing", "cropped", "reset", "source", "rank", "canonical", "normalizer", "policy", "rate", "trajectory", "trajectory_count"):
            report = copy.deepcopy(original)
            if change == "missing": report["clips"].pop()
            if change == "cropped": report["clips"][0]["control_steps"] -= 1
            if change == "reset": report["clips"][0]["reset_calls_during_rollout"] = 1
            if change == "source": report["clips"][0]["motion_sha256"] = "f"*64
            if change == "rank": del report["checkpoint_sha256"]["rank_07"]
            if change == "canonical": report["canonical_sha256"] = "0"*64
            if change == "normalizer": report["normalizer_updates"] = 1
            if change == "policy": report["policy_state_sha256_after"] = "d"*64
            if change == "rate": report["metrics"]["joint_pass_rate"] = report["metrics"]["tables_reward_survival_rate"]
            if change == "trajectory": report['clips'][0].pop('trajectory')
            if change == "trajectory_count": report['trajectory_count'] = 79
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError, msg=change): monitor.validate_result(path, request)

    def test_metadata_experiment_helper_and_training_source_are_required(self):
        for change in ("metadata", "experiment", "helper", "training_source"):
            if change == "metadata":
                target = self.training/"run_metadata.json"; original = target.read_bytes(); target.unlink()
            elif change == "experiment":
                target = self.experiment; original = target.read_bytes(); target.write_text('{"profile":"tables"}')
            elif change == "helper":
                target = self.helper/"hoi_zero_shot/data.py"; original = target.read_bytes(); target.write_text("changed")
            else:
                target = self.training_source/"hoi_tables/networks.py"; original = target.read_bytes(); target.write_text("changed")
            with self.assertRaises((ValueError, FileNotFoundError), msg=change): self.supervisor()
            target.write_bytes(original)

    def test_worker_wrapper_is_eight_asset_direct_evaluation_not_residual(self):
        request = self.request()
        request_path = Path(self.slurm.calls[0][0])
        scheduler = monitor.Slurm()
        with patch.object(scheduler, "_run", return_value="12345") as run:
            self.assertEqual(scheduler.submit(request_path, self.source), "12345")
        command = run.call_args.args[0]
        self.assertIn(str(self.source/"cluster/hoi_ultra_eval.sbatch.sh"), command)
        self.assertIn("--no-requeue", command)
        self.assertIn("--gres=gpu:a10:1", command)
        script = (Path(__file__).resolve().parents[1]/"cluster/hoi_ultra_eval.sbatch.sh").read_text()
        self.assertIn("for asset in {0..7}", script)
        self.assertIn("-m hoi_tables.evaluate_hoi80", script)
        self.assertNotIn("--gmt-checkpoint", script)
        self.assertNotIn("--experiment-config", script)

    def test_invalid_input_prevents_submission_before_any_scheduler_call(self):
        self.request()
        path = Path(self.slurm.calls[0][0])
        (self.inputs/"geometry_bundle.json").write_text("changed")
        scheduler = monitor.Slurm()
        with patch.object(scheduler, "_run") as run:
            with self.assertRaisesRegex(RuntimeError, "Invalid immutable"):
                scheduler.submit(path, self.source)
        run.assert_not_called()


if __name__ == "__main__": unittest.main()
