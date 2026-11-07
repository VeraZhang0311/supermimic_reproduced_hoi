from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from hoi_residual.data import sha256_file
from hoi_residual.policy import ResidualActorCritic
from hoi_residual.ppo import PPO, PPOConfig
from hoi_residual.resume import checkpoint_paths, restore_training, save_checkpoint


class ResumeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(81)
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.capacity = {"nconmax": 512, "njmax": 2048}
        self.identity = {"residual_contract": {"schema":"test", "implementation_sha256": {"train.py":"a"}},
                         "gmt": {"checkpoint":"old.pt", "checkpoint_sha256":"g", "target_contract":{"a":1}},
                         "observation_contract":{"version":5}, "experiment":{"reward_profile":"resmimic"}}
        self.policy, self.ppo = self.make()
        self.policy.update_normalizer(torch.randn(4, 4), torch.randn(4, 2))
        self.gradient_update(self.policy, self.ppo)
        self.ck = {**self.identity, "schema":"hoi_residual_ppo_teacher_v2", "iteration":1,
                   "ppo_updates":1, "ppo_config":asdict(self.ppo.config), "seed":0, "num_envs_per_rank":4,
                   "capacity":self.capacity, "architecture":self.policy.architecture(),
                   "residual":self.policy.state_dict(), "optimizer":self.ppo.optimizer.state_dict()}

    @staticmethod
    def make():
        policy = ResidualActorCritic(4, 2, hidden=(8,))
        return policy, PPO(policy, PPOConfig(rollout_steps=2, epochs=1, minibatches=1))

    @staticmethod
    def gradient_update(policy, ppo):
        ppo.optimizer.zero_grad()
        sum((p - .25).square().sum() for p in policy.parameters()).backward()
        ppo.optimizer.step()

    def load(self, **overrides):
        policy, ppo = self.make()
        path = self.root / 'input.pt'
        torch.save(self.ck, path)
        settings = dict(expected_sha256=sha256_file(path), policy=policy, ppo=ppo,
                        identity=self.identity, total_iterations=2, num_envs=4, seed=0,
                        capacity={**self.capacity, "epa_horizon":64})
        settings.update(overrides)
        return policy, ppo, restore_training(path, **settings)

    def test_exact_next_adam_update_including_shape_normalizer_and_absolute_counter(self):
        policy, ppo, receipt = self.load(identity={**self.identity, "gmt":{**self.identity["gmt"], "checkpoint":"new.pt"}})
        self.assertEqual((ppo.updates, receipt["iteration"], receipt["normalizer_revision"]), (1, 1, 1))
        self.assertIn("per-rank RNG", receipt["fresh"])
        self.assertEqual(ppo.optimizer.state_dict()["param_groups"], self.ppo.optimizer.state_dict()["param_groups"])
        for before, after in zip(self.policy.state_dict().values(), policy.state_dict().values()):
            self.assertTrue(torch.equal(before, after))
        self.gradient_update(self.policy, self.ppo)
        self.gradient_update(policy, ppo)
        for before, after in zip(self.policy.state_dict().values(), policy.state_dict().values()):
            self.assertTrue(torch.equal(before, after))

    def test_reject_mismatch_before_loading(self):
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.load(expected_sha256="0"*64)
        with self.assertRaisesRegex(ValueError, "implementation"):
            self.load(identity={**self.identity, "residual_contract":{"unknown":True}})
        with self.assertRaisesRegex(ValueError, "observation_contract"):
            self.load(identity={**self.identity, "observation_contract":{"version":4}})
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.load(capacity={"nconmax":512,"njmax":128})
        with self.assertRaisesRegex(ValueError, "later"):
            self.load(total_iterations=1)
        self.ck["optimizer"]["state"][0]["step"].zero_()
        with self.assertRaisesRegex(ValueError, "Adam step"):
            self.load()

    def test_rolling_checkpoint_survives_failed_save_and_shares_archival_bytes(self):
        paths = checkpoint_paths(self.root, 10, 300, 100, 10)
        self.assertEqual([p.name for p in paths], ["recovery_latest.pt"])
        save_checkpoint({"iteration":10}, paths)
        with patch("torch.save", side_effect=OSError("write failed")), self.assertRaisesRegex(OSError,"write failed"):
            save_checkpoint({"iteration":20}, paths)
        self.assertEqual(torch.load(paths[0], weights_only=True)["iteration"], 10)
        self.assertFalse(list(self.root.glob("*.partial")))
        double = checkpoint_paths(self.root, 100, 300, 100, 10)
        self.assertEqual([p.name for p in double], ["checkpoint_000100.pt", "recovery_latest.pt"])
        with patch("torch.save", wraps=torch.save) as writer:
            save_checkpoint({"iteration":100}, double)
            writer.assert_called_once()
        self.assertEqual(sha256_file(double[0]), sha256_file(double[1]))
        save_checkpoint({"iteration":110}, paths)
        self.assertEqual(torch.load(double[0],weights_only=True)["iteration"], 100)
        self.assertEqual(torch.load(paths[0],weights_only=True)["iteration"], 110)
        self.assertEqual(checkpoint_paths(self.root, 111, 300, 100, 10), [])
        self.assertEqual(checkpoint_paths(self.root, 110, 300, 100, 0), [])
