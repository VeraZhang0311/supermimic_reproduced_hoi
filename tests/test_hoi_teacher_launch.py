"""No scheduler/GPU calls: teacher probe authorization and launcher syntax."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import patch

from hoi.v1.config import TrainConfig, source_identity
from hoi.v1.privileged import TEACHER_PROFILE
from hoi.v1.teacher_launch import validate_probe, verify_source


class TeacherLaunchTests(unittest.TestCase):
    def fixture(self):
        config = replace(TrainConfig(), actor_observation_profile=TEACHER_PROFILE)
        source, identity = {"content_sha256": "a" * 64}, {"bundle_content_sha256": "b" * 64}
        report = {"passed": True, "optimizer_steps": 0, "formal_training_started": False,
                  "scope": "full_training_bundle_resource_validation", "source_identity": source,
                  "production_config": config.to_dict(), "data_identity": identity,
                  "initialization": {"mode": "explicit_from_scratch"}, "worker_cpu_threads": 3,
                  "collector_devices": ["cuda:1", "cuda:2"], "worker_startup_concurrency": 2,
                  "probe": {"expected_groups": 273}, "real_start_mix": {"all_static_groups_present": True},
                  "resources": {"groups": 273, "num_envs": 4096, "device": "cuda:0",
                                "assets_per_process": 1, "processes": 273,
                                "collector_devices": ["cuda:1", "cuda:2"], "worker_startup_concurrency": 2}}
        args = dict(source=source, config=config, data_identity=identity, device="cuda:0",
                    collector_devices=["cuda:1", "cuda:2"], startup=2)
        return report, args

    def test_only_matching_full_zero_optimizer_teacher_probe_passes(self):
        report, args = self.fixture()
        self.assertTrue(validate_probe(report, **args))
        changes = {"passed": False, "optimizer_steps": 1, "formal_training_started": True,
                   "scope": "subset_resource_fixture", "source_identity": {}, "production_config": {},
                   "data_identity": {}, "initialization": {"mode": "migrated"},
                   "worker_cpu_threads": 24, "collector_devices": ["cuda:0"],
                   "worker_startup_concurrency": 1, "probe": {"expected_groups": 272},
                   "real_start_mix": {"all_static_groups_present": False}}
        for key, value in changes.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_probe({**report, key: value}, **args)
        for key, value in (("groups", 272), ("num_envs", 2048), ("device", "cuda:1"),
                           ("collector_devices", ["cuda:1"]), ("worker_startup_concurrency", 1)):
            changed = deepcopy(report)
            changed["resources"][key] = value
            with self.subTest(resource=key), self.assertRaises(ValueError):
                validate_probe(changed, **args)
        with self.assertRaises(ValueError):
            validate_probe(report, **{**args, "config": TrainConfig()})

    def test_source_mismatch_rejected_and_both_teacher_launchers_bound(self):
        identity = source_identity()
        for name in ("hoi_teacher_train.sbatch.sh", "hoi_teacher_evaluate.sbatch.sh"):
            self.assertIn("cluster/" + name, identity["files"])
        with patch("hoi.v1.teacher_launch.source_identity", return_value=identity):
            self.assertEqual(verify_source(identity["content_sha256"]), identity)
            with self.assertRaisesRegex(ValueError, "source identity changed"):
                verify_source("0" * 64)

    def test_generic_launcher_rejects_packed_or_undeclared_process_topology(self):
        report, args = self.fixture()
        for changes in ({"assets_per_process": 8, "processes": 35},
                        {"assets_per_process": 273, "processes": 1},
                        {"assets_per_process": 2}, {"processes": 272},
                        {"assets_per_process": True}, {"processes": 273.0}):
            changed = deepcopy(report)
            changed["resources"].update(changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "allocation mismatch"):
                validate_probe(changed, **args)
        for key in ("assets_per_process", "processes"):
            changed = deepcopy(report)
            del changed["resources"][key]
            with self.subTest(missing=key), self.assertRaisesRegex(ValueError, "allocation mismatch"):
                validate_probe(changed, **args)

    def test_2048_resource_comparison_cannot_authorize_generic_training(self):
        report, args = self.fixture()
        report["scope"] = "full_training_bundle_resource_comparison"
        with self.assertRaisesRegex(ValueError, "mismatch: scope"):
            validate_probe(report, **args)

    def test_launchers_parse_as_bash_without_submitting_any_jobs(self):
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
        if not bash:
            self.skipTest("Bash syntax checker is unavailable")
        root = Path(__file__).resolve().parents[1]
        for name in ("hoi_teacher_train.sbatch.sh", "hoi_teacher_evaluate.sbatch.sh"):
            subprocess.run([bash, "-n", (root / "cluster" / name).as_posix()], check=True,
                           capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
