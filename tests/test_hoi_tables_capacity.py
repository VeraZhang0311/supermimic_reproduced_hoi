"""Capacity guards perform real backwards without committing optimizer updates."""
from copy import deepcopy
from dataclasses import replace
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from hoi_tables.capacity import CapacityValidationError, _hash_state, validate_capacity
from hoi_tables.config import TrainConfig


class Model(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 1))
        self.register_buffer("rms_count", torch.tensor(7.))


class RealTensorReplay:
    def __init__(self, complete=True):
        self.T, self.N, self.size, self.ptr, self.appended_steps = 8, 2, 8, 0, 8
        self.data = {"x": torch.randn(8, 2, 3)}
        self.generator = torch.Generator().manual_seed(11)
        self.complete = complete

    def sample_windows(self, B, K, split):
        ids = torch.randint(16, (B,), generator=self.generator)
        mask = torch.ones(B, K)
        if not self.complete: mask[:, 1:] = 0.
        return {"x": self.data["x"].flatten(0, 1)[ids], "mask": mask, "partition": "train", "wm_split": split}

    def sample_starts(self, B, **kwargs):
        ids = torch.randint(16, (B,), generator=self.generator)
        return {"x": self.data["x"].flatten(0, 1)[ids], "partition": "train"}


class ProbeLearner:
    def __init__(self, replay):
        self.pi, self.wm, self.critic = Model(), Model(), Model()
        self.optimizers = {name: torch.optim.Adam(getattr(self, name).parameters(), lr=.01) for name in ("pi", "wm", "critic")}
        self.config = TrainConfig(experiment_kind="smoke", num_envs=2, buffer_steps=8, wm_window=3,
                                  wm_batch=4, pi_batch=4, wm_batches=2, wm_only_iters=1, horizon=3,
                                  critic_epochs=2, critic_minibatches=2)
        self.device, self.buffer, self.motion = "cpu", replay, SimpleNamespace()
        self.reward = SimpleNamespace(contract={"fixture": True})
        self.iteration = 17
        self.contact_positive, self.contact_negative = torch.ones(5), torch.ones(5)
        self.gradient_sync = None
        self.mutate_rms, self.fail_on_actor = False, False
        self.saw_reserved_adam = False

    def fit(self, name, x):
        optimizer, model = self.optimizers[name], getattr(self, name)
        optimizer.zero_grad(set_to_none=True)
        loss = model(x).square().mean()
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    def wm_update(self, window):
        self.saw_reserved_adam = all("exp_avg" in self.optimizers["wm"].state[p] for p in self.wm.parameters())
        torch.rand(2); random.random(); np.random.rand()
        if self.mutate_rms: self.wm.rms_count.add_(1)
        return {"loss": self.fit("wm", window["x"])}

    def policy_update(self, start):
        if self.fail_on_actor: raise RuntimeError("synthetic allocation failure")
        actor = self.fit("pi", start["x"])
        losses = [self.fit("critic", start["x"]) for _ in range(self.config.critic_epochs*self.config.critic_minibatches)]
        return {"pi_loss": actor, "v_loss": sum(losses)/len(losses)}


class CapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, complete=True):
        replay = RealTensorReplay(complete)
        learner = ProbeLearner(replay)
        env = SimpleNamespace(N=2, physics=None, sampler=SimpleNamespace(generator=torch.Generator().manual_seed(19)))
        return learner, replay, env

    def test_zero_steps_full_callback_counts_and_exact_state_rng_restoration(self):
        learner, replay, env = self.fixture()
        before = {name: _hash_state(getattr(learner, name).state_dict()) for name in learner.optimizers}
        torch_state, replay_rng = torch.get_rng_state().clone(), replay.generator.get_state().clone()
        sampler_rng, python_rng, numpy_rng = env.sampler.generator.get_state().clone(), random.getstate(), np.random.get_state()
        report = validate_capacity(learner, replay, env, learner.config, require_production=False)
        self.assertTrue(report["passed"])
        worker = report["workers"][0]
        self.assertEqual(worker["blocked_optimizer_calls"], {"pi": 1, "wm": 2, "critic": 4})
        self.assertEqual(worker["optimizer_updates"], 0)
        self.assertTrue(worker["models_and_rms_unchanged"] and worker["optimizer_states_unchanged"])
        self.assertTrue(worker["replay_tensor_versions_unchanged"])
        self.assertTrue(worker["wm_validation_all_rows_all_ticks_active"])
        self.assertTrue(learner.saw_reserved_adam)
        self.assertGreater(worker["adam_state_bytes"]["wm"], 0)
        self.assertGreater(worker["adam_foreach_scratch_reserved_bytes"], 0)
        self.assertEqual(worker["replay_index_scratch_reserved_bytes"], 128*replay.T*replay.N)
        self.assertFalse(worker["replay_allocation"]["fabricated_transitions"])
        self.assertEqual(report["max_peak_allocated_bytes"], 0)
        self.assertFalse(worker["memory_before"]["available"])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertTrue(torch.equal(replay.generator.get_state(), replay_rng))
        self.assertTrue(torch.equal(env.sampler.generator.get_state(), sampler_rng))
        self.assertEqual(random.getstate(), python_rng)
        np.testing.assert_equal(np.random.get_state(), numpy_rng)
        for name in learner.optimizers:
            self.assertEqual(before[name], _hash_state(getattr(learner, name).state_dict()))
            self.assertFalse(learner.optimizers[name].state)
        # The actual Adam method is restored; a later explicit update works.
        old = learner.wm[0].weight.detach().clone()
        learner.wm_update(replay.sample_windows(4, 3, "fit"))
        self.assertFalse(torch.equal(old, learner.wm[0].weight))

    def test_existing_adam_state_and_existing_gradients_are_preserved(self):
        learner, replay, env = self.fixture()
        learner.fit("wm", replay.data["x"][0])
        before = _hash_state(learner.optimizers["wm"].state_dict())
        gradients = [p.grad.clone() for p in learner.wm.parameters()]
        report = validate_capacity(learner, replay, env, learner.config, require_production=False)
        self.assertTrue(report["passed"])
        self.assertEqual(_hash_state(learner.optimizers["wm"].state_dict()), before)
        for parameter, gradient in zip(learner.wm.parameters(), gradients):
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)

    def test_mutated_rms_rejected_and_restored_on_error(self):
        learner, replay, env = self.fixture()
        before = _hash_state(learner.wm.state_dict())
        learner.mutate_rms = True
        with self.assertRaises(CapacityValidationError) as caught:
            validate_capacity(learner, replay, env, learner.config, require_production=False)
        self.assertIn("RMS", str(caught.exception))
        self.assertFalse(caught.exception.report["passed"])
        self.assertTrue(caught.exception.report["models_and_rms_unchanged"])
        self.assertEqual(_hash_state(learner.wm.state_dict()), before)
        self.assertTrue(all(not optimizer.state for optimizer in learner.optimizers.values()))

    def test_backward_failure_still_restores_guards_rng_and_state(self):
        learner, replay, env = self.fixture()
        before, rng = _hash_state(learner.wm.state_dict()), torch.get_rng_state().clone()
        learner.fail_on_actor = True
        with self.assertRaises(CapacityValidationError) as caught:
            validate_capacity(learner, replay, env, learner.config, require_production=False)
        report = caught.exception.report
        self.assertEqual(report["blocked_optimizer_calls"]["wm"], 2)
        self.assertIn("synthetic allocation failure", report["error"]["message"])
        self.assertEqual(_hash_state(learner.wm.state_dict()), before)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertNotIn("step", learner.optimizers["wm"].__dict__)

    def test_masked_short_prefixes_cannot_masquerade_as_full_window_capacity(self):
        learner, replay, env = self.fixture(complete=False)
        with self.assertRaises(CapacityValidationError) as caught:
            validate_capacity(learner, replay, env, learner.config, require_production=False)
        self.assertIn("complete real WM fit window", str(caught.exception))
        self.assertTrue(caught.exception.report["optimizer_states_unchanged"])
        self.assertEqual(caught.exception.report["blocked_optimizer_calls"]["wm"], 0)

    def test_cpu_or_reduced_replay_is_never_production_evidence(self):
        learner, replay, env = self.fixture()
        with self.assertRaisesRegex(ValueError, "two live CUDA"):
            validate_capacity(learner, replay, env, learner.config)
        replay.T = 7
        with self.assertRaisesRegex(ValueError, "full replay"):
            validate_capacity(learner, replay, env, learner.config, require_production=False)

    def test_short_real_prefill_reserves_future_scan_workspace_without_fabrication(self):
        learner, replay, env = self.fixture()
        replay.size, replay.appended_steps = 4, 4
        report = validate_capacity(learner, replay, env, learner.config, require_production=False)
        worker = report["workers"][0]
        self.assertEqual(worker["replay_allocation"]["real_prefilled_steps"], 4)
        self.assertEqual(worker["replay_allocation"]["capacity_steps"], 8)
        self.assertFalse(worker["replay_allocation"]["full_replay_scan_measured"])
        self.assertEqual(worker["replay_index_scratch_reserved_bytes"], 128*8*2)
        self.assertEqual(replay.size, 4)


if __name__ == "__main__":
    unittest.main()
