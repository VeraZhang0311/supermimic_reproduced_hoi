import copy
from types import SimpleNamespace
import unittest

import torch

from hoi_tables.replay import FIELDS, ReplayBuffer


def transition(t, *, num_envs=4, episode=None, done=None):
    values = {name: torch.zeros((num_envs, *shape), dtype=dtype) for name, (shape, dtype) in FIELDS.items()}
    episode = torch.zeros(num_envs, dtype=torch.long) if episode is None else torch.tensor(episode)
    values.update(s=torch.full((num_envs, 71), float(t)), s_next=torch.full((num_envs, 71), float(t+1)),
                  o=torch.full((num_envs, 13), float(t)), o_next=torch.full((num_envs, 13), float(t+1)),
                  f=torch.full((num_envs,), t, dtype=torch.long), next_f=torch.full((num_envs,), t+1, dtype=torch.long),
                  episode_id=episode, done=torch.tensor(done) if done is not None else torch.zeros(num_envs, dtype=torch.bool),
                  contact_valid=torch.ones(num_envs, 5, dtype=torch.bool),
                  actuator=torch.full((num_envs, 88), float(t)), actuator_next=torch.full((num_envs, 88), float(t+1)))
    return values


class TestJointReplay(unittest.TestCase):
    def test_copy_ownership_and_pre_reset_terminal_state(self):
        replay = ReplayBuffer(4, 4)
        row = transition(0, done=[True]*4)
        row["s_next"].fill_(17.)
        replay.append(row)
        row["s_next"].zero_()
        self.assertTrue(torch.equal(replay.data["s_next"][0], torch.full((4, 71), 17.)))
        sampled = replay.sample_transitions(32)
        self.assertTrue(bool(sampled["done"].all()))
        self.assertTrue(bool((sampled["s_next"] == 17).all()))

    def test_ring_wrap_windows_remain_in_time_order(self):
        replay = ReplayBuffer(5, 4, holdout_fraction=0., seed=3)
        for i in range(11):
            replay.append(transition(i))
        sampled = replay.sample_windows(100, 4)
        self.assertTrue(bool((sampled["f"] >= 6).all()))
        active = sampled["mask"][:, 1:].bool()
        self.assertTrue(torch.equal(sampled["s"][:, 1:][active], sampled["s_next"][:, :-1][active]))
        self.assertTrue(torch.equal(sampled["actuator"][:, 1:][active], sampled["actuator_next"][:, :-1][active]))
        self.assertTrue(bool(((sampled["f"][:, 1:]-sampled["f"][:, :-1])[active] == 1).all()))
        self.assertEqual(tuple(sampled["object_history"].shape), (100, 4, 10, 9))

    def test_windows_never_cross_done_episode_or_asset_change(self):
        replay = ReplayBuffer(8, 4, holdout_fraction=0., seed=8)
        for i in range(8):
            episode = [int(i >= 3)]*4
            row = transition(i, episode=episode, done=[i == 2]*4)
            row["asset_id"][0] = int(i >= 6)
            replay.append(row)
        sampled = replay.sample_windows(1000, 3)
        self.assertFalse(bool((sampled["done"][:, :-1] & sampled["mask"][:, 1:].bool()).any()))
        self.assertTrue(bool((sampled["episode_id"] == sampled["episode_id"][:, :1]).all()))
        self.assertTrue(bool((sampled["asset_id"] == sampled["asset_id"][:, :1]).all()))
        # A terminal transition is still a valid *last* interval in a window.
        self.assertTrue(bool(sampled["done"][:, -1].any()))

    def test_disjoint_episode_level_wm_holdout(self):
        replay = ReplayBuffer(6, 12)
        for t in range(6):
            replay.append(transition(t, num_envs=12))
        # The same episode ID means different envs here: give each env a
        # marker in q_prev to inspect exactly which worlds were returned.
        for t in range(6):
            replay.data["q_prev"][t, :, 0] = torch.arange(12)
        fit, holdout = replay.sample_windows(256, 3, "fit"), replay.sample_windows(256, 3, "holdout")
        self.assertTrue(set(fit["q_prev"][..., 0].flatten().tolist()).isdisjoint(set(holdout["q_prev"][..., 0].flatten().tolist())))
        mask = replay.fit_mask(transition(5, num_envs=12))
        self.assertTrue(bool(mask[fit["q_prev"][:, 0, 0].long()].all()))
        self.assertFalse(bool(mask[holdout["q_prev"][:, 0, 0].long()].any()))

    def test_no_half_written_row_on_invalid_append_or_eval(self):
        replay = ReplayBuffer(3, 4)
        replay.append(transition(0))
        for variant in ("protected", "nonfinite", "missing", "shape"):
            row = transition(1)
            if variant == "protected":
                row["partition"] = "protected_eval"
            elif variant == "nonfinite":
                row["actuator_next"][1, 5] = float("nan")
            elif variant == "missing":
                del row["actuator"]
            else:
                row["contact"] = torch.zeros(4, 3)
            with self.assertRaises(ValueError):
                replay.append(row)
            self.assertEqual((replay.ptr, replay.size, replay.appended_steps), (1, 1, 1))
        with self.assertRaises(ValueError):
            ReplayBuffer(3, 4, partition="protected_eval")

    def test_one_step_failures_are_retained_and_right_padding_never_reads_reset(self):
        replay = ReplayBuffer(8, 4, holdout_fraction=0., seed=6)
        for t in range(5):
            row = transition(t, episode=[t]*4, done=[True]*4)
            row["s_next"].fill_(100+t)
            replay.append(row)
        windows = replay.sample_windows(256, 8)
        self.assertTrue(torch.equal(windows["mask"], torch.tensor([[1.]+[0.]*7]).expand(256, -1)))
        for key in ("s", "s_next", "o", "o_next", "actuator", "actuator_next", "f", "episode_id"):
            first = windows[key][:, :1].expand_as(windows[key])
            self.assertTrue(torch.equal(windows[key], first), key)
        self.assertTrue(bool((windows["s_next"][:, 0, 0] == 100+windows["f"][:, 0]).all()))
        self.assertEqual(set(windows["episode_id"][:, 0].tolist()), set(range(5)))

    def test_partial_window_includes_terminal_interval_then_repeats_its_row(self):
        replay = ReplayBuffer(3, 4, holdout_fraction=0., seed=9)
        replay.append(transition(0))
        terminal = transition(1, done=[True]*4)
        terminal["s_next"].fill_(44.)
        replay.append(terminal)
        replay.append(transition(2, episode=[1]*4, done=[True]*4))
        windows = replay.sample_windows(256, 8)
        starts = windows["f"][:, 0] == 0
        self.assertTrue(bool(starts.any()))
        self.assertTrue(bool((windows["mask"][starts, :2] == 1).all()))
        self.assertTrue(bool((windows["mask"][starts, 2:] == 0).all()))
        self.assertTrue(bool((windows["f"][starts, 1:] == 1).all()))
        self.assertTrue(bool((windows["s_next"][starts, 1:] == 44).all()))

    def test_single_observed_control_is_enough_for_a_masked_wm_window(self):
        replay = ReplayBuffer(8, 4, holdout_fraction=0.)
        replay.append(transition(0, done=[True]*4))
        windows = replay.sample_windows(3, 8)
        self.assertEqual(windows["mask"].sum(1).tolist(), [1., 1., 1.])

    def test_snapshot_restores_sample_rng_and_live_ring(self):
        replay = ReplayBuffer(5, 4, seed=123, holdout_fraction=0.)
        for t in range(9):
            replay.append(transition(t))
        state = copy.deepcopy(replay.state_dict())
        restored = ReplayBuffer(5, 4, seed=0, holdout_fraction=0.)
        restored.load_state_dict(state)
        expected, actual = replay.sample_windows(32, 3), restored.sample_windows(32, 3)
        self.assertTrue(torch.equal(expected["s"], actual["s"]))
        replay.append(transition(9))
        restored.append(transition(9))
        self.assertTrue(torch.equal(replay.sample_windows(32, 3)["s"], restored.sample_windows(32, 3)["s"]))

    def test_imagination_horizon_respects_clip_end(self):
        replay = ReplayBuffer(6, 4, holdout_fraction=0.)
        for t in range(6):
            replay.append(transition(t))
        reference = SimpleNamespace(clip_id=torch.zeros(8, dtype=torch.long), clip_end=torch.tensor([7]))
        starts = replay.sample_starts(256, horizon=4, motion=reference)
        self.assertTrue(bool((starts["f"] <= 3).all()))
        with self.assertRaisesRegex(ValueError, "no reset-safe"):
            replay.sample_starts(1, horizon=9, motion=reference)


if __name__ == "__main__":
    unittest.main()
