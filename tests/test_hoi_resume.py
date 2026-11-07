from dataclasses import asdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from hoi_residual.data import sha256_file
from hoi_residual.contract import compatible_for_evaluation, implementation_contract
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

    def test_v11_pre_resume_checkpoint_binds_objective_and_contact_sources(self):
        live = implementation_contract("hoi_v1_1")
        old = deepcopy(live)
        files = old["implementation_sha256"]
        for name in ("resume.py", "capacity.py"):
            del files[name]
        for name in ("contract.py", "env.py", "train.py"):
            files[name] = "older"
        self.ck.update(experiment={"reward_profile":"hoi_v1_1"}, reward_inputs={"hand_labels":"fixture"},
                       residual_contract=old)
        for key in ("ppo_updates", "num_envs_per_rank", "seed"):
            del self.ck[key]
        implementation_sha = hashlib.sha256(json.dumps(old, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        path = self.root / "old.pt"
        torch.save(self.ck, path)
        hashed = sha256_file(path)
        identity = {**self.identity, "residual_contract":live, "experiment":self.ck["experiment"],
                    "reward_inputs":self.ck["reward_inputs"]}
        def restore(identity):
            policy, ppo = self.make()
            return restore_training(path, hashed, policy=policy, ppo=ppo, identity=identity,
                                    total_iterations=2, num_envs=512, seed=0, capacity=self.capacity)
        with self.assertRaisesRegex(ValueError, "implementation"):
            restore(identity)
        with patch.dict("hoi_residual.resume._LEGACY_PARENTS", {hashed: (1, implementation_sha)}):
            self.assertTrue(restore(identity)["legacy_migration"])
            for key, replacement in (("reward_inputs", {"hand_labels":"other"}), ("experiment", {"reward_profile":"resmimic"})):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    restore({**identity,key:replacement})
            for name in ("v11_reward.py", "v11_contacts.py"):
                changed = deepcopy(live)
                changed["implementation_sha256"][name] = "different source"
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "implementation"):
                    restore({**identity, "residual_contract":changed})

    def test_previous_protected_loaders_preserve_strict_training_and_evaluation(self):
        current = implementation_contract()
        old = deepcopy(current)
        old["implementation_sha256"].update({
            "contract.py":"98e8d8a2dc3571b03cad1a314b0733f1c778e31a9b068741f14746f9da2255e0",
            "resume.py":"3b65ab2367c927dd811fb3d72d7e75f197ef4ea72823988f82aca64dda7cd81a",
        })
        self.ck["residual_contract"] = old
        identity = {**self.identity, "residual_contract":current}
        self.assertTrue(compatible_for_evaluation(old,"resmimic"))
        self.assertFalse(self.load(identity=identity)[2]["legacy_migration"])
        del self.ck["ppo_updates"]
        with self.assertRaisesRegex(ValueError, "PPO configuration"):
            self.load(identity=identity)
        old["implementation_sha256"]["ppo.py"] = "other"
        self.assertFalse(compatible_for_evaluation(old,"resmimic"))
        with self.assertRaisesRegex(ValueError, "implementation"):
            self.load(identity=identity)
