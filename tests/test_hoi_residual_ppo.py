"""Meaningful CPU contracts for frozen GMT, residual likelihoods and PPO/GAE."""
import copy
from datetime import timedelta
import multiprocessing
import socket
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist

from hoi_residual.policy import FrozenGMT, ResidualActorCritic, validate_frozen_gmt_contract
from hoi_residual.ppo import PPO, PPOConfig, RolloutBatch, compute_gae
from supermimic.networks import Policy
from supermimic.privileged import PRIVILEGED_DIM, SCHEMA
from supermimic.robot_metadata import G1_BODY_NAMES, G1_JOINT_NAMES


def contract(dataset="boneseed", manifest="a"):
    return {**copy.deepcopy(SCHEMA), "body_names": G1_BODY_NAMES.copy(), "joint_names": G1_JOINT_NAMES.copy(),
            "physics": {"profile": "fengqiao_hw3_mujoco_v1", "implementation_sha256": {"env.py": "b" * 64}},
            "analytic_heads": {"observation_dim": 99, "control_dt": .02,
                               "implementation_sha256": {"networks.py": "c" * 64}},
            "urdf_sha256": "d" * 64, "dataset": dataset, "motion_manifest_sha256": manifest * 64}


def rollout(agent, offset=0.0):
    oracle = torch.randn(4, 3, agent.oracle_dim) + offset
    base = torch.randn(4, 3, agent.action_dim)
    agent.update_normalizer(oracle, base)
    sampled = agent.sample(oracle, base)
    term = torch.zeros(4, 3, dtype=torch.bool)
    trunc = torch.zeros_like(term)
    term[1, 0] = True
    trunc[2, 1] = True
    return RolloutBatch(oracle, base, sampled["actions"], sampled["log_prob"], sampled["value"],
                        torch.randn(4, 3) + offset, term, trunc, torch.randn(4, 3), agent.normalizer_revision)


def distributed_worker(rank, init_uri, output_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=init_uri, rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        torch.manual_seed(71)
        agent = ResidualActorCritic(5, action_dim=3, hidden=(16, 8))
        torch.manual_seed(100 + rank)
        batch = rollout(agent, offset=rank * 5.0)
        ppo = PPO(agent, PPOConfig(epochs=2, minibatches=2))
        metrics = ppo.update(batch)
        torch.save({"state": agent.state_dict(), "metrics": metrics,
                    "raw": torch.cat((batch.oracle, batch.base_actions), -1).flatten(0, 1)},
                   Path(output_dir) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


class ResidualPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_zero_mean_preserves_base_and_likelihood_uses_unclipped_residual(self):
        agent = ResidualActorCritic(5, action_dim=3, hidden=(16, 8))
        oracle, base = torch.randn(6, 5), torch.full((6, 3), 19.99)
        agent.update_normalizer(oracle, base)
        self.assertTrue(torch.equal(agent.act(oracle, base), torch.zeros_like(base)))
        sampled = agent.sample(oracle, base)
        evaluated = agent.evaluate(oracle, base, sampled["actions"])
        torch.testing.assert_close(sampled["log_prob"], evaluated["log_prob"], atol=0, rtol=0)
        # The physically clipped combined action is not the random variable in PPO.
        executed_residual = (base + sampled["actions"]).clamp(-20, 20) - base
        altered = agent.evaluate(oracle, base, executed_residual)
        self.assertFalse(torch.equal(sampled["log_prob"], altered["log_prob"]))

    def test_gae_bootstraps_timeout_from_terminal_observation_without_cross_reset_trace(self):
        rewards = torch.tensor([[1., 2.], [3., 4.], [100., 200.]])
        values = torch.tensor([[.5, .6], [.7, .8], [.9, 1.]])
        next_values = torch.tensor([[.7, .8], [10., 20.], [30., 40.]])
        term = torch.tensor([[False, False], [True, False], [False, False]])
        trunc = torch.tensor([[False, False], [False, True], [False, False]])
        advantages, returns = compute_gae(rewards, values, next_values, term, trunc, .9, .8)
        expected1 = torch.tensor([3. - .7, 4. + .9 * 20. - .8])
        expected0 = rewards[0] + .9 * next_values[0] - values[0] + .9 * .8 * expected1
        torch.testing.assert_close(advantages[1], expected1)
        torch.testing.assert_close(advantages[0], expected0)
        torch.testing.assert_close(returns, advantages + values)
        # A simultaneous true failure and timeout must suppress bootstrapping.
        term[1, 1] = True
        advantages2, _ = compute_gae(rewards, values, next_values, term, trunc, .9, .8)
        self.assertAlmostEqual(float(advantages2[1, 1]), 3.2, places=6)

    def test_real_ppo_changes_actor_and_critic_with_immutable_normalizer(self):
        torch.manual_seed(8)
        agent = ResidualActorCritic(5, action_dim=3, hidden=(16, 8))
        batch = rollout(agent)
        norm = copy.deepcopy(agent.norm.state_dict())
        actor_before = agent.actor[-1].weight.detach().clone()
        critic_before = agent.critic[-1].weight.detach().clone()
        ppo = PPO(agent, PPOConfig(epochs=2, minibatches=2))
        metrics = ppo.update(batch)
        self.assertEqual(metrics["optimizer_steps"], 4)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
        self.assertFalse(torch.equal(actor_before, agent.actor[-1].weight))
        self.assertFalse(torch.equal(critic_before, agent.critic[-1].weight))
        for key, value in agent.norm.state_dict().items():
            self.assertTrue(torch.equal(value, norm[key]))
        agent.update_normalizer(batch.oracle, batch.base_actions)
        with self.assertRaisesRegex(RuntimeError, "normalizer changed"):
            ppo.update(batch)

    def test_nonfinite_rollout_is_a_hard_error(self):
        agent = ResidualActorCritic(5, action_dim=3, hidden=(16, 8))
        batch = rollout(agent)
        batch.rewards[0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            PPO(agent).update(batch)

    def test_contract_allows_dataset_only_but_rejects_physics_and_schema(self):
        source, target = contract(), contract("ultra", "e")
        validate_frozen_gmt_contract(source, target)
        for key, replacement in (("physics", {"implementation_sha256": {"env.py": "f" * 64}}),
                                 ("dim", 3283), ("lookahead_frames", [1, 12])):
            changed = copy.deepcopy(target)
            changed[key] = replacement
            with self.assertRaises(ValueError):
                validate_frozen_gmt_contract(source, changed)

    def test_real_gmt_policy_load_is_strict_frozen_and_constructs_no_world_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ckpt_026000.pt"
            torch.manual_seed(2)
            original = Policy(99, 730, hist=10, privileged_dim=PRIVILEGED_DIM)
            original.obs_norm.update(torch.randn(20, 99))
            original.goal_norm.update(torch.randn(20, 730))
            original.privileged_norm.update(torch.randn(20, PRIVILEGED_DIM))
            torch.save({"pi": original.state_dict(), "wm": {"unused": torch.tensor(0)},
                        "args": {"hist": 10, "goal_frames": 10, "amp": "none"},
                        "gmt_contract": contract(), "iteration": 26000}, path)
            with patch("supermimic.networks.WorldModel", side_effect=AssertionError("WM must not be constructed")) as wm:
                frozen = FrozenGMT(path, contract("ultra", "e"))
                wm.assert_not_called()
            inputs = (torch.randn(2, 10, 99), torch.randn(2, 730), torch.randn(2, PRIVILEGED_DIM))
            with torch.no_grad():
                torch.testing.assert_close(frozen(*inputs), original(*inputs), atol=0, rtol=0)
            self.assertFalse(frozen(*inputs).requires_grad)
            frozen.train()
            self.assertFalse(frozen.policy.training)
            self.assertTrue(frozen.assert_unchanged())
            frozen.policy.obs_norm.count.add_(1)
            with self.assertRaisesRegex(RuntimeError, "weight/RMS changed"):
                frozen.assert_unchanged()

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "PyTorch Gloo is unavailable")
    def test_two_rank_asset_gradients_and_normalizers_are_synchronized(self):
        with tempfile.TemporaryDirectory() as directory:
            # FileStore cannot open this Windows host's Unicode user path;
            # loopback TCP rendezvous avoids imposing an ASCII workspace.
            with socket.socket() as server:
                server.bind(("127.0.0.1", 0))
                port = server.getsockname()[1]
            uri = f"tcp://127.0.0.1:{port}?use_libuv=0"
            context = multiprocessing.get_context("spawn")
            processes = [context.Process(target=distributed_worker, args=(rank, uri, directory)) for rank in range(2)]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=55)
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()
                    self.fail("Distributed PPO test timed out")
                self.assertEqual(process.exitcode, 0)
            first, second = [torch.load(Path(directory) / f"rank{rank}.pt", weights_only=True) for rank in range(2)]
            for key in first["state"]:
                torch.testing.assert_close(first["state"][key], second["state"][key], atol=0, rtol=0)
            all_raw = torch.cat((first["raw"], second["raw"]))
            torch.testing.assert_close(first["state"]["norm.mean"], all_raw.mean(0))
            torch.testing.assert_close(first["state"]["norm.var"], all_raw.var(0, unbiased=False))
            self.assertEqual(first["metrics"], second["metrics"])
            self.assertEqual(first["metrics"]["world_size"], 2)


if __name__ == "__main__":
    unittest.main()
