"""CPU structural/differentiation checks; never claims a native GPU pass."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from supermimic.gmt_preflight import (HELPERS, assert_unchanged, learning_checks,
                                     load_training_helpers, parameter_snapshot,
                                     projected_replay_bytes, snapshot_hash)


class PreflightTests(unittest.TestCase):
    def test_ast_does_not_execute_training_top_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.py"
            path.write_text("raise RuntimeError('training started')\n" + "\n".join(
                f"def {name}():\n    return 7\n" for name in HELPERS))
            helpers = load_training_helpers({}, path)
            self.assertEqual([fun() for fun in helpers.values()], [7] * len(HELPERS))

    def test_unchanged_parameters_detects_updates(self):
        model = nn.Linear(2, 1)
        models = {"model": model}
        before = parameter_snapshot(models)
        self.assertEqual(assert_unchanged(models, before), snapshot_hash(before))
        with torch.no_grad():
            model.bias.add_(1)
        with self.assertRaises(AssertionError):
            assert_unchanged(models, before)

    def test_actual_wm_rollout_ignores_future_recorded_context(self):
        from supermimic.actuator_state import advance_actuator_state
        from supermimic.mdp import clamp_action, normalize_xi, unflatten_state
        calls = []
        class SpyWM:
            def step(self, S, a, xn, context):
                calls.append(context.clone())
                return S
        namespace = dict(torch=torch, wm=SpyWM(), args=SimpleNamespace(wm_window=8),
                         unflatten_state=unflatten_state, normalize_xi=normalize_xi,
                         clamp_action=clamp_action,
                         advance_actuator_state=advance_actuator_state)
        helper = load_training_helpers(namespace)["_wm_rollout"]
        state = torch.zeros(2, 8, 71)
        state[..., 3] = 1
        xi = torch.zeros(2, 8, 33)
        xi[..., 0] = .75
        xi[..., 1:30] = .003
        action = torch.arange(16).reshape(2, 8, 1).expand(-1, -1, 29).float() * .02
        context = torch.full((2, 8, 88), float("nan"))
        context[:, 0] = 1
        context[:, 0, 29] = torch.tensor([2., 4.])
        context[:, 0, 30:59] = torch.tensor([.9, 1.1]).unsqueeze(-1)
        helper(state, action, xi, context)
        expected = context[:, 0]
        self.assertEqual(len(calls), 8)
        for j, actual in enumerate(calls):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            expected = advance_actuator_state(expected, action[:, j], xi[:, j])

    def test_actual_helpers_h32_backward_without_weight_changes(self):
        from supermimic.actuator_state import advance_actuator_state
        from supermimic.buffer import ReplayBuffer
        from supermimic.mdp import flatten_state
        from supermimic.networks import Critic, WorldModel

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(3)
            N = 2
            buffer = ReplayBuffer(48, N, "cpu")
            ctx = torch.zeros(N, 88)
            ctx[:, 29] = 3
            ctx[:, 30:] = 1
            xi = torch.zeros(N, 33)
            xi[:, 0] = .75
            p, R = torch.tensor([[0., 0., .8]]).expand(N, -1), torch.eye(3).expand(N, -1, -1)
            v, q = torch.zeros(N, 3), torch.zeros(N, 29)
            for t in range(40):
                action = torch.full((N, 29), .01 * (t % 4))
                S = (p, R, v, v, q + t * .001, q)
                S1 = (p + .001, R, v, v, q + (t+1) * .001, q)
                buffer.add(dict(s=flatten_state(S), s1=flatten_state(S1), a=action, a_prev=action,
                                q_prev=S[4], qd_prev=q, xi=xi, actuator=ctx, r=torch.zeros(N),
                                f=torch.full((N,), t, dtype=torch.long), ep_step=torch.full((N,), t),
                                done=torch.zeros(N, dtype=torch.bool)))
                ctx = advance_actuator_state(ctx, action, xi)

            class TinyPolicy(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(290 + 730 + 88, 29)
                def forward(self, hist, goal, privileged):
                    return .01 * self.linear(torch.cat((hist.flatten(1), goal, privileged), -1))

            class TinyMDP:
                def obs(self, S, ap, xi, f):
                    return S[4] + .01 * ap
                def critic_features(self, S, f, xn):
                    return torch.cat((S[4], xn), -1)
                def reward(self, S, f, a, ap):
                    return .1 - .001 * a.square().sum(-1) - .001 * S[4].square().sum(-1)
                def soft_termination(self, S, f):
                    return torch.zeros_like(f, dtype=torch.float32)

            motion = SimpleNamespace(clip_end=torch.tensor([200]), clip_id=torch.zeros(201, dtype=torch.long),
                                     goal=lambda f, K: torch.zeros(*f.shape, 73*K))
            models = {"wm": WorldModel(hidden=(16, 16)), "pi": TinyPolicy(), "critic": Critic(29+33+730+88)}
            before = parameter_snapshot(models)
            result = learning_checks(buffer, motion, TinyMDP(), lambda *args: args[-1], models, "cpu")
            self.assertEqual(result["horizon"], 32)
            self.assertGreater(result["wm_gradient_max"], 0)
            self.assertGreater(result["actor_gradient_max"], 0)
            self.assertGreater(result["analytic_target_action_gradient_max"], 0)
            self.assertEqual(assert_unchanged(models, before), snapshot_hash(before))
            self.assertGreater(projected_replay_bytes(), 0)
        finally:
            torch.set_num_threads(old_threads)

    def test_missing_native_dependencies_never_pass(self):
        import importlib.util
        if importlib.util.find_spec("isaaclab") is not None:
            self.skipTest("Real native runtime installed")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            command = [sys.executable, "-m", "supermimic.gmt_preflight", "--motion_dir", tmp, "--output", str(output)]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, run.stderr)
            report = json.loads(output.read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(report["optimizer_updates"], 0)


if __name__ == "__main__":
    unittest.main()
