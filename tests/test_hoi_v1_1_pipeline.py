"""CPU handoff failures and real shell ordering; no Slurm/GPU/train execution."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from cluster.hoi_v1_1.pipeline import check

ROOT = Path(__file__).resolve().parents[1]
BASH = (str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe") if os.name == "nt" else shutil.which("bash"))
if BASH and not Path(BASH).is_file():
    BASH = None

def shell_path(path):
    if os.name != "nt":
        return str(path)
    return subprocess.check_output([BASH, "-c", 'cygpath -u "$1"', "--", str(path)], encoding="utf-8").strip()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HandoffTests(unittest.TestCase):
    def test_packaged_helpers_import_without_slurm_or_ignored_python_files(self):
        from cluster.hoi_v1_1 import stage, validate, train, capacity_probe
        self.assertEqual(capacity_probe.ROOT, ROOT)
        self.assertTrue(callable(stage.verify) and callable(validate.main) and callable(train.main))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.root, self.ops = base / "source", base / "operations"
        self.root.mkdir(); self.ops.mkdir()
        (self.root / "fixture.py").write_text("fixture source", encoding="utf-8")
        (self.ops / "fixture.py").write_text("fixture operations", encoding="utf-8")
        self.config = {"iterations": 20000, "num_envs": 4096, "pi_batch": 4096, "horizon": 32, "wm_only_iters": 500}
        self.identity = {"content_sha256": "CPU_fixture"}
        stage = {"root": str(self.root.resolve()), "files": {"fixture.py": sha(self.root / "fixture.py")},
                 "production_config": self.config, "source_identity": self.identity}
        save(self.root / "data/v11_staging.json", stage)
        manifest = {"root": str(self.ops.resolve()), "stage_sha256": sha(self.root / "data/v11_staging.json"),
                    "files": {"fixture.py": sha(self.ops / "fixture.py")}}
        save(self.ops / "operations.json", manifest)
        self.env = {"SUPERMIMIC_ROOT": str(self.root), "OPERATIONS_ROOT": str(self.ops),
                    "EXPECTED_STAGE_SHA256": manifest["stage_sha256"],
                    "EXPECTED_OPERATIONS_SHA256": sha(self.ops / "operations.json"),
                    "SLURM_JOB_ID": "CPU_fixture", "SLURM_JOB_NODELIST": "CPU_fixture_node",
                    "CUDA_VISIBLE_DEVICES": "0", "SLURM_JOB_GPUS": "0"}

    def validated(self, *, job="CPU_fixture", steps=0):
        native, resource = self.root / "results/native.json", self.root / "results/resource.json"
        save(native, {"CPU_fixture": True})
        save(resource, {"passed": True, "actual_optimizer_steps": steps, "all_model_parameters_unchanged": True,
                        "config": self.config, "source_identity": self.identity,
                        "noop_optimizer_step_calls": {"pi": 1, "wm": 1, "critic": 20}})
        save(self.root / "results/v11_validation_complete.json", {"passed": True, "actual_optimizer_steps": 0,
             "source_identity": self.identity, "job_id": job, "native_evidence": str(native),
             "native_evidence_sha256": sha(native), "resource_report": str(resource), "resource_report_sha256": sha(resource)})

    def test_matching_same_allocation_handoff_records_fresh_training(self):
        check("before", self.env)
        self.validated()
        check("after-validation", self.env)
        record = json.loads((self.ops / "run/validation_passed.json").read_text())
        self.assertTrue(record["fresh_training_process_next"])
        self.assertFalse(record["second_training_submission"])

    def test_missing_validation_cannot_handoff(self):
        check("before", self.env)
        with self.assertRaises(FileNotFoundError): check("after-validation", self.env)
        self.assertFalse((self.ops / "run/validation_passed.json").exists())

    def test_another_job_cannot_supply_validation(self):
        check("before", self.env)
        self.validated(job="another_job")
        with self.assertRaisesRegex(ValueError, "current allocation"): check("after-validation", self.env)

    def test_gpu_assignment_cannot_change_between_phases(self):
        check("before", self.env)
        self.validated()
        with self.assertRaisesRegex(ValueError, "same Slurm allocation"):
            check("after-validation", {**self.env, "CUDA_VISIBLE_DEVICES": "1"})

    def test_real_optimizer_step_in_probe_blocks_training(self):
        check("before", self.env)
        self.validated(steps=1)
        with self.assertRaisesRegex(ValueError, "capacity/backward"): check("after-validation", self.env)

    def test_tampered_frozen_source_blocks_pipeline(self):
        (self.root / "fixture.py").write_text("modified", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "immutable file"): check("before", self.env)


@unittest.skipUnless(BASH, "POSIX shell ordering needs Bash (Git Bash on Windows)")
class ShellOrderingTests(unittest.TestCase):
    def exercise(self, fail=""):
        with tempfile.TemporaryDirectory(prefix="smhoi-pipeline-") as name:
            base = Path(name)
            root, ops, runtime = base / "source", base / "operations", base / "runtime"
            (root / "cluster").mkdir(parents=True); ops.mkdir(); (runtime / "bin").mkdir(parents=True)
            (root / "cluster/hoi_v1_1_runtime.sh").write_text('hoi_v1_1_setup() { export PATH="$SUPERMIMIC_RUNTIME/bin:$PATH"; }\n', encoding="utf-8", newline="\n")
            python = runtime / "bin/python"
            python.write_text('''#!/bin/bash
set -eu
[[ "$1" == "-m" ]]
phase="$2:${3:-}"
printf '%s|%s|%s|%s\\n' "$phase" "$WANDB_SYNC" "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES" >> "$TEST_TRACE"
if [[ "$phase" == "$TEST_FAIL" ]]; then exit 7; fi
''', encoding="utf-8", newline="\n")
            python.chmod(0o755)
            env = dict(os.environ, SUPERMIMIC_ROOT=shell_path(root), OPERATIONS_ROOT=shell_path(ops), SUPERMIMIC_RUNTIME=shell_path(runtime),
                       EXPECTED_OPERATIONS_SHA256="CPU_fixture", SLURM_JOB_ID="fixture_job", CUDA_VISIBLE_DEVICES="0",
                       TEST_TRACE=shell_path(base / "trace"), TEST_FAIL=fail)
            script = ROOT / "cluster/hoi_v1_1_combined.sbatch.sh"
            completed = subprocess.run([BASH, shell_path(script)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            rows = (base / "trace").read_text().splitlines()
            return completed.returncode, rows

    def test_success_uses_same_job_gpu_and_separate_validation_train_invocations(self):
        status, rows = self.exercise()
        self.assertEqual(status, 0)
        self.assertEqual([r.split("|")[0] for r in rows], ["cluster.hoi_v1_1.pipeline:before", "cluster.hoi_v1_1.validate:",
                         "cluster.hoi_v1_1.pipeline:after-validation", "cluster.hoi_v1_1.train:"])
        self.assertEqual([r.split("|")[1] for r in rows], ["0", "0", "0", "1"])
        self.assertTrue(all(r.endswith("|fixture_job|0") for r in rows))

    def test_validation_failure_never_invokes_training(self):
        status, rows = self.exercise("cluster.hoi_v1_1.validate:")
        self.assertEqual(status, 7)
        self.assertEqual(len(rows), 2)
        self.assertFalse(any("cluster.hoi_v1_1.train" in r for r in rows))

    def test_handoff_guard_failure_never_invokes_training(self):
        status, rows = self.exercise("cluster.hoi_v1_1.pipeline:after-validation")
        self.assertEqual(status, 7)
        self.assertEqual(len(rows), 3)

    def test_manifest_guard_failure_never_invokes_validation(self):
        status, rows = self.exercise("cluster.hoi_v1_1.pipeline:before")
        self.assertEqual(status, 7)
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
