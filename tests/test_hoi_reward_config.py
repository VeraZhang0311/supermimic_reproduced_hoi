import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from hoi_residual.contract import compatible_for_evaluation, implementation_contract
from hoi_residual.train import reward_experiment


class RewardConfigTests(unittest.TestCase):
    def test_baseline_and_v11_are_distinct_and_bound(self):
        config = Path(__file__).resolve().parents[1] / "configs/hoi_residual_v11_reward.json"
        self.assertEqual(reward_experiment(None, None), {"reward_profile": "resmimic"})
        with TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle.json"
            bundle.write_text("{}")
            self.assertEqual(reward_experiment(config, bundle), json.loads(config.read_text()))
            with self.assertRaises(ValueError):
                reward_experiment(None, bundle)
            with self.assertRaises(ValueError):
                reward_experiment(config, None)
            bad = Path(directory) / "experiment.json"
            bad.write_text(json.dumps({**json.loads(config.read_text()), "seed": 2}))
            with self.assertRaises(ValueError):
                reward_experiment(bad, bundle)
        current = implementation_contract("hoi_v1_1")
        self.assertTrue(compatible_for_evaluation(current, "hoi_v1_1"))
        self.assertFalse(compatible_for_evaluation(current, "resmimic"))
        for name in ("v11_reward.py", "v11_contacts.py"):
            with self.subTest(name=name):
                altered = implementation_contract("hoi_v1_1")
                altered["implementation_sha256"][name] = "0" * 64
                self.assertFalse(compatible_for_evaluation(altered, "hoi_v1_1"))
