"""CPU interruption and restart contracts; no GPU training or scheduler calls."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from hoi.v1.checkpoint import make_checkpoint, save_checkpoint
from tests.test_hoi_v1_learner import make_learner

spec = importlib.util.spec_from_file_location("requeue", Path(__file__).resolve().parents[1] / "cluster/compact_teacher_requeue.py")
requeue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(requeue)


class RequeueTests(unittest.TestCase):
    def setUp(self):
        before = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, before)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.learner = make_learner()
        self.args = (self.learner.pi, self.learner.wm, self.learner.critic, self.learner.optimizers)
        self.kwargs = dict(iteration=1300, physics={"hand_collision": True, "self_collision": True},
                           data_identity={"test": "synthetic"}, initialization={"mode": "test"},
                           extra={"config": self.learner.config.to_dict(), "sim_steps": 123,
                                  "contact_positive": torch.zeros(16), "contact_negative": torch.zeros(16),
                                  "sampling_bin_counts": torch.zeros(1)})
        self.seed = make_checkpoint(*self.args, **self.kwargs)
        self.seed_path = self.root / "seed.pt"
        torch.save(self.seed, self.seed_path)
        self.sha = requeue.digest(self.seed_path)

    def prepare(self, restart):
        return requeue.prepare_attempt(self.root, self.seed_path, self.sha, "42", str(restart), "teacher")

    def test_requeue_selects_latest_and_keeps_old_attempt(self):
        first = self.prepare(0)
        output = Path(first["output"])
        output.mkdir()
        newer = deepcopy(self.seed)
        newer["iteration"] = 1400
        path = output / "checkpoint_001400.pt"
        torch.save(newer, path)
        second = self.prepare(1)
        self.assertEqual(second["checkpoint"], str(path.resolve()))
        self.assertEqual(second["iteration"], 1400)
        self.assertNotEqual(first["output"], second["output"])
        self.assertNotEqual(first["run_name"], second["run_name"])
        self.assertTrue(path.exists())
        with self.assertRaises(FileExistsError):
            self.prepare(1)

    def test_interrupted_unpublished_and_corrupt_files_do_not_replace_seed(self):
        output = Path(self.prepare(0)["output"])
        output.mkdir()
        (output / "checkpoint_001500.pt.xyz.partial").write_bytes(b"partial")
        (output / "checkpoint_001400.pt").write_bytes(b"truncated")
        result = self.prepare(1)
        self.assertEqual(result["iteration"], 1300)
        self.assertEqual(len(result["rejected_checkpoints"]), 1)

    def test_nonfinite_and_foreign_checkpoint_rejected(self):
        output = Path(self.prepare(0)["output"])
        output.mkdir()
        bad = deepcopy(self.seed)
        bad["iteration"] = 1400
        next(iter(bad["pi"].values())).fill_(float("nan"))
        torch.save(bad, output / "checkpoint_001400.pt")
        bad = deepcopy(self.seed)
        bad["iteration"] = 1500
        bad["data_identity"] = {"other": "dataset"}
        torch.save(bad, output / "checkpoint_001500.pt")
        result = self.prepare(1)
        self.assertEqual(result["iteration"], 1300)
        self.assertEqual(len(result["rejected_checkpoints"]), 2)

    def test_seed_hash_is_required(self):
        self.seed_path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            self.prepare(0)

    def test_finished_checkpoint_does_not_launch_more_training(self):
        output = Path(self.prepare(0)["output"])
        output.mkdir()
        final = deepcopy(self.seed)
        final["iteration"] = final["extra"]["config"]["iterations"]
        torch.save(final, output / f"checkpoint_{final['iteration']:06d}.pt")
        self.assertTrue(self.prepare(1)["training_complete"])

    def test_failed_save_never_publishes_final_filename(self):
        path = self.root / "checkpoint_001300.pt"
        def interrupted(checkpoint, stream):
            stream.write(b"partial")
            self.assertFalse(path.exists())
            raise OSError("simulated interruption")
        with patch("hoi.v1.checkpoint.torch.save", side_effect=interrupted):
            with self.assertRaises(OSError):
                save_checkpoint(path, *self.args, **self.kwargs)
        self.assertFalse(path.exists())
        self.assertFalse(list(self.root.glob("*.partial")))

    def test_atomic_publish_roundtrip_and_no_overwrite(self):
        path = self.root / "checkpoint_001300.pt"
        save_checkpoint(path, *self.args, **self.kwargs)
        self.assertEqual(torch.load(path, weights_only=False)["iteration"], 1300)
        sha = requeue.digest(path)
        with self.assertRaises(FileExistsError):
            save_checkpoint(path, *self.args, **self.kwargs)
        self.assertEqual(sha, requeue.digest(path))

    def test_evaluation_recovery_reuses_checkpoint_owner_directory(self):
        receipt = {"iteration": 5000, "checkpoint": str(self.root / "old/run/checkpoint_005000.pt")}
        environment = {"SUPERMIMIC_ROOT": "/source", "COMPACT_DATASET": "/dataset",
                       "SUPERMIMIC_RUNTIME": "/runtime", "ISAACLAB_PATH": "/isaac"}
        with patch("hoi.v1.periodic_eval.schedule_evaluation", return_value={"status": "submitted"}) as schedule:
            requeue.recover_due_evaluation(receipt, environment)
        self.assertEqual(schedule.call_args.args[4], self.root / "old/run/evaluations")
        with patch("hoi.v1.periodic_eval.schedule_evaluation", return_value={"status": "submission_outcome_unknown"}):
            with self.assertRaises(RuntimeError):
                requeue.recover_due_evaluation(receipt, environment)
