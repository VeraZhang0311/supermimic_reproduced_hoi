"""Ablations change only their declared objective/input and require new runs."""
from dataclasses import asdict, replace
from pathlib import Path
import unittest

import torch

from hoi.v1.config import TrainConfig
from hoi.v1.contacts import CHANNELS
from hoi.v1.train import read_config
from hoi.v1.variants import geometry_input_enabled, max_step_reward, reward_config
from tests.test_hoi_v1_learner import make_learner, transition
from hoi.v1.state import unflatten_state, unflatten_object


class VariantTests(unittest.TestCase):
    def test_formal_ablation_profiles_preserve_every_numerical_training_default(self):
        root = Path(__file__).resolve().parents[1]
        for variant in ('no_contact_objective', 'no_geometry_input'):
            config, numerical_deviations = read_config(root / 'configs' / f'hoi_v1_{variant}.json')
            self.assertEqual(config.variant, variant)
            self.assertEqual(asdict(config), asdict(replace(TrainConfig(), variant=variant)))
            self.assertEqual(numerical_deviations, {})
        with self.assertRaisesRegex(ValueError, 'variant'):
            replace(TrainConfig(), variant='no_geometery').validate()

    def test_only_declared_contact_weight_changes_and_head_supervision_remains_separate(self):
        full, ablated = asdict(reward_config('hoi_v1')), asdict(reward_config('no_contact_objective'))
        self.assertEqual(ablated, {**full, 'w_contact': 0.})
        self.assertEqual(asdict(reward_config('no_geometry_input')), full)
        self.assertTrue(geometry_input_enabled('hoi_v1'))
        self.assertTrue(geometry_input_enabled('no_contact_objective'))
        self.assertFalse(geometry_input_enabled('no_geometry_input'))
        self.assertAlmostEqual(max_step_reward('hoi_v1'), .15)
        self.assertAlmostEqual(max_step_reward('no_contact_objective'), .145)

    def test_contact_ablation_drops_only_match_reward_under_identical_real_state(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            learner = make_learner()
            tr = transition(learner, [0, 70], 0)
            S, O, f = unflatten_state(tr['s1']), unflatten_object(tr['o1']), tr['f'] + 1
            # Exercise both a net positive match and a net contact penalty,
            # with one unknown region that must not affect either objective.
            learner.motion.expected_contact[f] = 0
            learner.motion.expected_contact[f, CHANNELS.index("pelvis_object")] = 1
            learner.motion.expected_valid[f] = 1
            learner.motion.expected_valid[f, CHANNELS.index("head_object")] = 0
            actual = torch.stack((torch.full((len(CHANNELS),), .2), torch.full((len(CHANNELS),), .7)))
            actual[:, CHANNELS.index("pelvis_object")] = torch.tensor([.8, .1])
            actual[:, CHANNELS.index("head_object")] = .99
            full, terms = learner.mdp.reward(S, O, f, tr['a'], tr['a_prev'], actual, terms=True)
            learner.mdp.cfg = reward_config('no_contact_objective')
            ablated, ablated_terms = learner.mdp.reward(S, O, f, tr['a'], tr['a_prev'], actual, terms=True)
            torch.testing.assert_close(full - ablated, terms['contact_match'] * .25)
            torch.testing.assert_close(terms['contact_match'], torch.tensor([.6, -.6]) * .02)
            for name in ('human_reward', 'object_position', 'object_orientation', 'hand_approach'):
                torch.testing.assert_close(terms[name], ablated_terms[name])
        finally:
            torch.set_num_threads(threads)


if __name__ == '__main__':
    unittest.main()
