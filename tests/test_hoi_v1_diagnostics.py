"""CPU checks for read-only, correctly aligned evaluation diagnostics."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from hoi.v1.diagnostics import contact_transition_events, executed_action_reward_gap, source_flag_comparison
from hoi.v1.contacts import CHANNELS, LABEL_SCHEMA
from hoi.v1.state import flatten_object, flatten_state
from tests.test_hoi_v1_evaluate import fixture


class ScriptedWM(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.))
        self.register_buffer("count", torch.tensor(0.))
        self.actions = []

    def step(self, S, O, a, xi, xi_o, points):
        self.actions.append(a.detach().clone())
        S1 = (S[0] + a[:, :3] * self.scale * .001, *S[1:])
        O1 = (O[0] + a[:, :3] * self.scale * .002, *O[1:])
        return S1, O1, a.new_full((len(a), len(CHANNELS)), 100.)


class EvaluationDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)

    def test_source_flags_use_verified_names_not_hardcoded_hand_indices_or_pair_truth(self):
        source = torch.zeros(4, 630)
        names = [f"body_{i}" for i in range(39)]
        names[4], names[17] = "left_rubber_hand", "right_rubber_hand"
        source[:, 591 + 4] = torch.tensor([1, 0, 1, 0])
        source[:, 591 + 17] = torch.tensor([0, 1, 0, 1])
        expected = torch.tensor([[1, 1], [1, 1], [0, 1], [0, 0]])
        valid = torch.tensor([[1, 1], [1, 0], [0, 1], [1, 1]])
        result = source_flag_comparison(source, expected, valid, names)
        self.assertEqual(result["source_body_indices"], [4, 17])
        self.assertEqual(result["valid_intervals"], [3, 3])
        self.assertEqual(result["mismatch_intervals"], [1, 3])
        self.assertEqual(result["mismatch_fraction_valid"], [1/3, 1.])
        self.assertFalse(result["source_is_pair_truth"])
        self.assertFalse(result["affects_reward_or_gate"])
        self.assertFalse(source_flag_comparison(source, expected, valid)["available"])

    def test_contact_and_support_events_only_connect_adjacent_valid_intervals(self):
        labels = torch.tensor([[0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0]])
        valid = torch.ones_like(labels).bool()
        valid[2, 0] = False
        result = contact_transition_events(labels, valid, ("left", "right", "support"))
        self.assertEqual(result["channels"]["left"]["onset_endpoint_steps"], [2, 6])
        self.assertEqual(result["channels"]["left"]["release_endpoint_steps"], [5])
        self.assertEqual(result["channels"]["support"]["release_endpoint_steps"], [3, 6])
        self.assertEqual(result["channels"]["support"]["onset_endpoint_steps"], [5])
        self.assertEqual(result["channels"]["support"]["onset_endpoint_times_s"], [.1])
        self.assertTrue(result["channels"]["support"]["initial_observed_contact"])
        self.assertTrue(result["first_positive_interval_is_not_an_onset"])

    def trajectory(self):
        _, motion, mdp, shapes, _, _ = fixture()
        wm = ScriptedWM()
        start = int(motion.clip_start[1])
        T = int(motion.clip_len[1]) - 1
        S = tuple(value[start:start + 1] for value in motion.ref_arrays())
        O = tuple(value[start:start + 1] for value in motion.ref_object_arrays())
        states = {"s": [flatten_state(S)[0]], "o": [flatten_object(O)[0]]}
        actions = torch.arange(T * 29).reshape(T, 29).float() / 100
        control = {"action": actions, "a_prev": torch.cat((torch.zeros_like(actions[:1]), actions[:-1])),
                   "xi": torch.ones(T, 33), "xi_o": torch.ones(T, 12),
                   "contact_valid": torch.ones(T, len(CHANNELS)).bool(), "reward": []}
        cloud = shapes.points(torch.tensor(0))
        with torch.no_grad():
            for step in range(T):
                S, O, logits = wm.step(S, O, actions[step:step + 1], control["xi"][step:step + 1],
                                       control["xi_o"][step:step + 1], cloud[None])
                states["s"].append(flatten_state(S)[0])
                states["o"].append(flatten_object(O)[0])
                reward = mdp.reward(S, O, torch.tensor([start + step + 1]), actions[step:step + 1],
                                    control["a_prev"][step:step + 1], logits.sigmoid(),
                                    contact_valid=control["contact_valid"][step:step + 1])
                control["reward"].append(reward[0])
        states = {key: torch.stack(value) for key, value in states.items()}
        control["reward"] = torch.stack(control["reward"])
        wm.actions.clear()
        return wm, mdp, states, control, torch.arange(start, start + T + 1), cloud

    def test_paired_reward_same_real_starts_executed_actions_freezes_everything(self):
        wm, mdp, states, control, frames, cloud = self.trajectory()
        before = deepcopy(wm.state_dict())
        saved = deepcopy(states)
        wm.train()
        result = executed_action_reward_gap(wm, mdp, states, control, frames, cloud, horizons=(1, 2, 4, 32), batch_size=2)
        for horizon, windows, intervals in ((1, 6, 6), (2, 3, 6), (4, 1, 4)):
            row = result["horizons"][str(horizon)]
            self.assertEqual(row["sequence_count"], windows)
            self.assertEqual(row["valid_intervals"], intervals)
            self.assertLess(row["absolute_reward_gap_mean"], 1e-8)
            self.assertLess(row["absolute_window_return_gap_mean"], 1e-8)
        self.assertFalse(result["horizons"]["32"]["available"])
        self.assertIsNone(result["horizons"]["32"]["absolute_reward_gap_mean"])
        self.assertFalse(result["used_as_training_data_or_gate"])
        self.assertEqual(result["contact_channels"], list(CHANNELS))
        self.assertEqual(result["contact_label_schema"], LABEL_SCHEMA)
        self.assertTrue(wm.training)
        self.assertTrue(all(parameter.grad is None for parameter in wm.parameters()))
        for key, value in before.items():
            torch.testing.assert_close(wm.state_dict()[key], value, atol=0, rtol=0)
        for key in saved:
            torch.testing.assert_close(states[key], saved[key], atol=0, rtol=0)
        # Distinct changing actions force the autoregressive diagnostic to use
        # the real executed sequence; it cannot substitute a target/constant.
        self.assertGreater(len({tuple(row.tolist()) for batch in wm.actions for row in batch}), 1)
        wm.scale.data.fill_(.1)
        changed = executed_action_reward_gap(wm, mdp, states, control, frames, cloud, horizons=(1, 4), batch_size=2)
        self.assertGreater(changed["horizons"]["4"]["absolute_reward_gap_mean"], 0)

    def test_paired_reward_rejects_cross_clip_frame_jump(self):
        wm, mdp, states, control, frames, cloud = self.trajectory()
        frames[2] += 20
        with self.assertRaisesRegex(ValueError, "contiguous"):
            executed_action_reward_gap(wm, mdp, states, control, frames, cloud)

    def test_paired_reward_rejects_old_validity_and_WM_logits_contracts(self):
        wm, mdp, states, control, frames, cloud = self.trajectory()
        old = {**control, "contact_valid": control["contact_valid"][:, :3]}
        with self.assertRaisesRegex(ValueError, "5 channels"):
            executed_action_reward_gap(wm, mdp, states, old, frames, cloud)
        class OldWM(ScriptedWM):
            def step(self, *args):
                S, O, logits = super().step(*args)
                return S, O, logits[:, :3]
        with self.assertRaisesRegex(ValueError, "five-channel"):
            executed_action_reward_gap(OldWM(), mdp, states, control, frames, cloud)

    def test_foot_logits_do_not_become_reference_reward_and_nonfinite_feet_are_rejected(self):
        wm, mdp, states, control, frames, cloud = self.trajectory()
        class FootWM(ScriptedWM):
            def step(self, *args):
                S, O, logits = super().step(*args)
                logits[:, 3:] = -100.
                return S, O, logits
        baseline = executed_action_reward_gap(wm, mdp, states, control, frames, cloud, horizons=(1, 2))
        feet = executed_action_reward_gap(FootWM(), mdp, states, control, frames, cloud, horizons=(1, 2))
        self.assertEqual(baseline["horizons"], feet["horizons"])
        class InvalidFootWM(ScriptedWM):
            def step(self, *args):
                S, O, logits = super().step(*args)
                logits[:, 4] = float("nan")
                return S, O, logits
        with self.assertRaisesRegex(FloatingPointError, "contact logits"):
            executed_action_reward_gap(InvalidFootWM(), mdp, states, control, frames, cloud, horizons=(1,))

    def test_full_duration_evaluation_persists_paired_frozen_WM_reports_and_named_source_flags(self):
        from hoi.v1.evaluate import evaluate_group
        from hoi.v1.networks import WorldModel
        from tests.test_hoi_v1_evaluate import SyntheticWorkers
        policy, motion, mdp, shapes, fk, metadata = fixture()
        fk.body_names[4], fk.body_names[17] = "left_rubber_hand", "right_rubber_hand"
        wm = WorldModel(hidden=(16,16), contact_hidden=8)
        before = deepcopy(wm.state_dict())
        with tempfile.TemporaryDirectory() as temp:
            reports = evaluate_group(policy,motion,mdp,shapes,SyntheticWorkers(motion,mdp),[0,1],
                fk39=fk,joint_names=fk.dof_names,output=Path(temp)/"evaluation",metadata=metadata,
                verify_tracked_fk=False,wm=wm,diagnostic_batch_size=2)
            for report,T in zip(reports,(3,6)):
                row=report["simulator_imagined_reward_gap"]["horizons"]["1"]
                self.assertEqual(row["valid_intervals"],T)
                self.assertTrue(row["absolute_reward_gap_mean"]>=0)
                self.assertEqual(report["source_flag_comparison"]["source_body_indices"],[4,17])
                self.assertEqual(report["source_flag_comparison"]["mismatch_intervals"],[T,T])
                saved=torch.load(report["rollout"],weights_only=True)
                self.assertEqual(saved["control"]["a_prev"].shape,(T,29))
                self.assertIn("object_ground_support",report["contact"]["actual_transition_events"]["channels"])
        for key,value in before.items():
            torch.testing.assert_close(wm.state_dict()[key],value,atol=0,rtol=0)


if __name__ == "__main__":
    unittest.main()
