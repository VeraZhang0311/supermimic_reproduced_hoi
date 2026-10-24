"""CPU regression against the pre-chunking value-fitting calculation.

The small integration fixture uses the real critic and PointNet. The separate
production row-count fixture measures scheduling only, not GPU capacity or
native dynamics. Neither fixture starts a simulator or trains an experiment.
"""
from copy import deepcopy
from dataclasses import replace
from types import MethodType
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hoi.v1.config import TrainConfig
from hoi.v1.learner import Learner, _detach, _flatten, _stack
from hoi.v1.networks import frozen_parameters
from supermimic.mdp import normalize_xi
from tests.test_hoi_v1_learner import make_learner, transition


def full_batch_reference(learner, trajectory, start):
    """The previous full-forward implementation, retained only as a baseline."""
    c = learner.config
    H, B = c.horizon, len(start["f"])
    with torch.no_grad():
        S = _flatten(_stack([_detach(state) for state in trajectory["states"]]))
        O = _flatten(_stack([_detach(state) for state in trajectory["objects"]]))
        f = torch.stack(trajectory["frames"]).flatten()
        expand = lambda value: value[None].expand((H + 1,) + value.shape).flatten(0, 1)
        arguments = learner._critic_args(S, O, f, expand(normalize_xi(start["xi"])),
                                         expand(start["xi_o"]), expand(start["asset_id"]))
    learner.critic.update_normalizers(*arguments[:-1])
    with torch.no_grad():
        values = learner.critic(*arguments).reshape(H + 1, B)
        target, targets = values[-1], []
        for step in range(H - 1, -1, -1):
            target = trajectory["reward"][step].detach() + c.gamma * (1 - trajectory["failure"][step].detach()) * (
                (1 - c.lam) * values[step + 1] + c.lam * target)
            targets.append(target)
        targets = torch.stack(targets[::-1]).flatten()
    inputs = tuple(value[:H * B].detach() for value in arguments)
    size, total_loss, count = H * B, 0., 0
    for _ in range(c.critic_epochs):
        for ids in torch.randperm(size, device=learner.device).tensor_split(c.critic_minibatches):
            if not len(ids):
                continue
            value = learner.critic(*(feature[ids] for feature in inputs))
            loss = (value - targets[ids]).square().mean()
            optimizer = learner.optimizers["critic"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = nn.utils.clip_grad_norm_(learner.critic.parameters(), 1.)
            if not bool(torch.isfinite(grad)):
                raise FloatingPointError("nonfinite critic gradient")
            optimizer.step()
            total_loss += float(loss.detach())
            count += 1
    return total_loss / max(count, 1), values, targets


def capture_forward(calls):
    def hook(module, arguments, output):
        call = {"gradient_enabled": torch.is_grad_enabled(), "rows": len(output),
                "arguments": tuple(value.detach().clone() for value in arguments),
                "value": output.detach().clone()}
        calls.append(call)
        if output.requires_grad:
            # MSE's output gradient recovers the target actually consumed by
            # backward, independently of the implementation's local variables.
            def gradient_hook(gradient):
                call["target"] = call["value"] - gradient.detach() * len(output) / 2
            output.register_hook(gradient_hook)
    return hook


class CriticTargetChunkingTests(unittest.TestCase):
    def setUp(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.manual_seed(417)

    def test_fp32_repeated_updates_keep_full_targets_and_bounded_optimizer_drift(self):
        self.compare_repeated_updates(torch.float32)

    def test_float64_repeated_updates_match_full_parameters_and_adam_strictly(self):
        self.compare_repeated_updates(torch.float64)

    def compare_repeated_updates(self, dtype):
        H, B = 5, 3
        chunked = make_learner(horizon=H)
        chunked.config = replace(chunked.config, critic_epochs=5, critic_minibatches=4)
        frames = torch.tensor([60, 200, 80])
        start = transition(chunked, frames, 10)
        start["s"][:, 13:42] = torch.linspace(-.2, .2, B * 29).reshape(B, 29)
        start["xi_o"][:, 0] = torch.tensor([1., 1.5, 2.])
        history_rows = [transition(chunked, frames - 9 + tick, tick) for tick in range(10)]
        history = {key: torch.stack([row[key] for row in history_rows], 1) for key in start}
        noise = [torch.zeros(B, 10, 99)] + [torch.zeros(B, 99) for _ in range(H)]
        with torch.no_grad(), frozen_parameters(chunked.wm, chunked.critic):
            trajectory = chunked.imagine(start, history, fixed_robot_noise=noise)
        # Exercise nontrivial lambda continuation/termination weights, rather
        # than relying on a fixture that happens to never terminate.
        trajectory["failure"] = torch.linspace(0., .35, H * B).reshape(H, B)
        trajectory["reward"] += .01 * torch.arange(H * B).reshape(H, B)
        if dtype == torch.float64:
            # Isolate batch-GEMM rounding from optimization semantics. The
            # trajectory is identical; only this test's critic/point inputs
            # use double precision, never the production learner or config.
            chunked.critic.double()
            chunked.shapes.cloud = chunked.shapes.cloud.double()
        full = deepcopy(chunked)
        initial_shape = deepcopy(chunked.critic.shape_encoder.state_dict())
        initial_others = {name: deepcopy(model.state_dict()) for name, model in (("pi", chunked.pi), ("wm", chunked.wm))}
        target_batch = (H * B + 3) // 4
        self.assertEqual(target_batch, 4)
        atol, rtol = (1e-10, 1e-10) if dtype == torch.float64 else (2e-6, 2e-5)
        # PyTorch 2.7 CPU: forward values differ by at most 1.20e-7 and
        # lambda targets by 5.97e-8, but 40 Adam steps amplify the difference
        # to 9.55e-6 in PointNet weights. Keep a 2e-5 FP32 absolute bound,
        # and independently require 1e-10 parameter/Adam parity in float64.
        parameter_atol = 1e-10 if dtype == torch.float64 else 2e-5

        # The second update also checks existing Adam moments and accumulated
        # normalizers, matching the failure's occurrence after a prior update.
        for cycle in range(2):
            full_calls, chunk_calls = [], []
            with full.critic.register_forward_hook(capture_forward(full_calls)), \
                    patch.object(full.critic, "update_normalizers", wraps=full.critic.update_normalizers) as full_norm:
                torch.manual_seed(700 + cycle)
                full_loss, full_values, _ = full_batch_reference(full, trajectory, start)
                full_rng = torch.get_rng_state().clone()
            with chunked.critic.register_forward_hook(capture_forward(chunk_calls)), \
                    patch.object(chunked.critic, "update_normalizers", wraps=chunked.critic.update_normalizers) as chunk_norm:
                torch.manual_seed(700 + cycle)
                chunk_loss = chunked.critic_update(trajectory, start)
                chunk_rng = torch.get_rng_state().clone()
            self.assertEqual((full_norm.call_count, chunk_norm.call_count), (1, 1))
            for old, new in zip(full_norm.call_args.args, chunk_norm.call_args.args):
                self.assertEqual(len(new), (H + 1) * B)
                torch.testing.assert_close(old, new, rtol=0, atol=0)
            for name in ("norm", "object_norm"):
                self.assertEqual(float(getattr(chunked.critic, name).count), (cycle + 1) * (H + 1) * B)
                for key, old in getattr(full.critic, name).state_dict().items():
                    torch.testing.assert_close(old, getattr(chunked.critic, name).state_dict()[key], rtol=0, atol=0)
            target_calls = [call for call in chunk_calls if not call["gradient_enabled"]]
            self.assertEqual([call["rows"] for call in target_calls], [4, 4, 4, 4, 2])
            self.assertEqual(sum(call["rows"] for call in target_calls), (H + 1) * B)
            for feature_index in range(7):
                reconstructed = torch.cat([call["arguments"][feature_index] for call in target_calls])
                torch.testing.assert_close(reconstructed, full_calls[0]["arguments"][feature_index], rtol=0, atol=0)
            chunk_values = torch.cat([call["value"] for call in target_calls]).reshape(H + 1, B)
            torch.testing.assert_close(chunk_values, full_values, rtol=rtol, atol=atol)
            old_training = [call for call in full_calls if call["gradient_enabled"]]
            new_training = [call for call in chunk_calls if call["gradient_enabled"]]
            self.assertEqual((len(old_training), len(new_training)), (20, 20))
            for old, new in zip(old_training, new_training):
                self.assertLessEqual(new["rows"], target_batch)
                # Identical raw rows demonstrate unchanged mini permutation.
                for old_arg, new_arg in zip(old["arguments"], new["arguments"]):
                    torch.testing.assert_close(old_arg, new_arg, rtol=0, atol=0)
                torch.testing.assert_close(old["target"], new["target"], rtol=rtol, atol=atol)
            self.assertAlmostEqual(chunk_loss, full_loss, delta=atol)
            torch.testing.assert_close(chunk_rng, full_rng, rtol=0, atol=0)
            for name, old in full.critic.state_dict().items():
                torch.testing.assert_close(chunked.critic.state_dict()[name], old, rtol=rtol, atol=parameter_atol)
            old_opt, new_opt = full.optimizers["critic"].state_dict(), chunked.optimizers["critic"].state_dict()
            self.assertEqual(old_opt["param_groups"], new_opt["param_groups"])
            self.assertTrue(old_opt["state"])
            self.assertEqual(old_opt["state"].keys(), new_opt["state"].keys())
            for parameter, old_state in old_opt["state"].items():
                for key, old in old_state.items():
                    torch.testing.assert_close(new_opt["state"][parameter][key], old, rtol=rtol, atol=atol)
        self.assertTrue(any(not torch.equal(initial_shape[key], value) for key, value in chunked.critic.shape_encoder.state_dict().items()))
        for name, model in (("pi", chunked.pi), ("wm", chunked.wm)):
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, initial_others[name][key], rtol=0, atol=0)

    def test_production_135168_rows_are_all_evaluated_with_32768_forward_cap(self):
        # Lightweight scalar critic: exercise production scheduling without
        # allocating production PointNet activations on a CPU test machine.
        class CountingCritic(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(.1))
                self.calls, self.normalizer_rows = [], []
            def update_normalizers(self, *arguments):
                self.normalizer_rows.append(len(arguments[0]))
            def forward(self, *arguments):
                self.calls.append((torch.is_grad_enabled(), arguments[0].detach().clone()))
                return self.weight * arguments[0][:, 0]
        learner = Learner.__new__(Learner)
        learner.config, learner.device, learner.critic = TrainConfig(), torch.device("cpu"), CountingCritic()
        learner.optimizers = {"critic": torch.optim.Adam(learner.critic.parameters(), lr=learner.config.lr)}
        H, B = learner.config.horizon, learner.config.pi_batch
        rows = torch.arange((H + 1) * B).float()[:, None] / ((H + 1) * B)
        learner._critic_args = MethodType(lambda self, *args: (rows,) * 7, learner)
        trajectory = {"states": [(torch.full((B, 1), float(tick)),) for tick in range(H + 1)],
                      "objects": [(torch.zeros(B, 1),) for _ in range(H + 1)],
                      "frames": [torch.full((B,), tick) for tick in range(H + 1)],
                      "reward": torch.full((H, B), .1), "failure": torch.zeros(H, B)}
        start = {"f": torch.zeros(B, dtype=torch.long), "xi": torch.ones(B, 33),
                 "xi_o": torch.ones(B, 12), "asset_id": torch.zeros(B, dtype=torch.long)}
        loss = learner.critic_update(trajectory, start)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertEqual(learner.critic.normalizer_rows, [135168])
        targets = [arguments for gradients, arguments in learner.critic.calls if not gradients]
        self.assertEqual([len(arguments) for arguments in targets], [32768, 32768, 32768, 32768, 4096])
        torch.testing.assert_close(torch.cat(targets), rows, rtol=0, atol=0)
        self.assertEqual([len(arguments) for gradients, arguments in learner.critic.calls if gradients], [32768] * 20)


if __name__ == "__main__":
    unittest.main()
