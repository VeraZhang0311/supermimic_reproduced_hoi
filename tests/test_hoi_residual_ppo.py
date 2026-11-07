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

from hoi_residual.policy import FrozenGMT, ResidualActorCritic, ShapeEncoder, validate_frozen_gmt_contract
from hoi_residual.ppo import PPO, PPOConfig, RolloutBatch, compute_gae
from hoi_residual.distributed import all_reduce_tensor, broadcast_tensor
from supermimic.networks import Policy
from supermimic.privileged import PRIVILEGED_DIM, SCHEMA
from supermimic.robot_metadata import G1_BODY_NAMES, G1_JOINT_NAMES


def contract(dataset="boneseed", manifest="a"):
    return {**copy.deepcopy(SCHEMA), "body_names": G1_BODY_NAMES.copy(), "joint_names": G1_JOINT_NAMES.copy(),
            "physics": {"profile": "fengqiao_hw3_mujoco_v1", "implementation_sha256": {"env.py": "b" * 64}},
            "analytic_heads": {"observation_dim": 99, "control_dt": .02,
                               "implementation_sha256": {"networks.py": "c" * 64}},
            "urdf_sha256": "d" * 64, "dataset": dataset, "motion_manifest_sha256": manifest * 64}


def rollout(agent, offset=0.0, fixed_shape=False):
    oracle = torch.randn(4, 3, agent.oracle_dim) + offset
    base = torch.randn(4, 3, agent.action_dim)
    points = torch.randn((128, 3) if fixed_shape else (4, 3, 128, 3)) * (.2 + .01 * offset)
    agent.update_normalizer(oracle, base)
    sampled = agent.sample(oracle, base, points)
    term = torch.zeros(4, 3, dtype=torch.bool)
    trunc = torch.zeros_like(term)
    term[1, 0] = True
    trunc[2, 1] = True
    return RolloutBatch(oracle, base, sampled["actions"], sampled["log_prob"], sampled["value"],
                        torch.randn(4, 3) + offset, term, trunc, torch.randn(4, 3), agent.normalizer_revision, points)


def distributed_worker(rank, init_uri, output_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=init_uri, rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        torch.manual_seed(71 + rank)
        agent = ResidualActorCritic(5, action_dim=3, hidden=(16, 8))
        for value in agent.state_dict().values():
            broadcast_tensor(value, src=0)
        torch.manual_seed(100 + rank)
        batch = rollout(agent, offset=rank * 5.0, fixed_shape=True)
        shape_before = {name: parameter.detach().clone() for name, parameter in agent.named_parameters()
                        if "shape_encoder" in name}
        ppo = PPO(agent, PPOConfig(epochs=2, minibatches=2))
        metrics = ppo.update(batch)
        torch.save({"state": agent.state_dict(), "metrics": metrics, "optimizer": ppo.optimizer.state_dict(),
                    "shape_input": list(batch.shape_points.shape),
                    "shape_changed": {prefix: any(not torch.equal(parameter, shape_before[name])
                        for name, parameter in agent.named_parameters() if name.startswith(prefix))
                        for prefix in ("actor_shape_encoder", "critic_shape_encoder")},
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
        points = torch.randn(128, 3) * .2
        agent.update_normalizer(oracle, base)
        self.assertTrue(torch.equal(agent.act(oracle, base, points), torch.zeros_like(base)))
        sampled = agent.sample(oracle, base, points)
        evaluated = agent.evaluate(oracle, base, sampled["actions"], points)
        torch.testing.assert_close(sampled["log_prob"], evaluated["log_prob"], atol=0, rtol=0)
        # The physically clipped combined action is not the random variable in PPO.
        executed_residual = (base + sampled["actions"]).clamp(-20, 20) - base
        altered = agent.evaluate(oracle, base, executed_residual, points)
        self.assertFalse(torch.equal(sampled["log_prob"], altered["log_prob"]))

    def test_shape_encoder_is_permutation_invariant_and_preserves_metric_scale(self):
        torch.manual_seed(101)
        encoder = ShapeEncoder()
        points = torch.randn(2, 128, 3) * .2
        value = encoder(points)
        self.assertEqual(value.shape, (2, 64))
        torch.testing.assert_close(value, encoder(points[:, torch.randperm(128)]), atol=0, rtol=0)
        self.assertFalse(torch.allclose(value, encoder(points * 2)))
        from hoi_residual.features import ORACLE_DIM
        agent = ResidualActorCritic(ORACLE_DIM)
        self.assertEqual(agent.actor[0].in_features, ORACLE_DIM + 29 + 64)
        self.assertEqual(agent.norm.mean.numel(), ORACLE_DIM + 29)
        self.assertIsNot(agent.actor_shape_encoder, agent.critic_shape_encoder)
        self.assertTrue(agent.architecture()["shape_encoder"]["independent_actor_critic"])

    def test_shape_input_is_required_finite_and_aligned_to_rollout(self):
        agent = ResidualActorCritic(5, action_dim=3, hidden=(16, 8))
        oracle, base = torch.randn(2, 5), torch.randn(2, 3)
        for points in (torch.randn(2, 64), torch.randn(3, 128, 3), torch.randn(2, 128, 3).double()):
            with self.assertRaises(ValueError):
                agent.act(oracle, base, points)
        with self.assertRaises(FloatingPointError):
            agent.act(oracle, base, torch.full((128, 3), float("nan")))
        batch = rollout(agent)
        batch.shape_points = torch.randn(4, 3, 64)
        with self.assertRaisesRegex(ValueError, "raw.*coordinates"):
            PPO(agent).update(batch)

    def test_fixed_shape_matches_repeated_outputs_encoder_gradients_and_adam(self):
        # Compare the independent repeated-cloud path against a shared cloud.
        # Nonzero actor output weights exercise BOTH encoders on every step.
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                torch.manual_seed(83)
                fixed = ResidualActorCritic(5, action_dim=3, hidden=(16, 8)).to(dtype=dtype)
                torch.nn.init.normal_(fixed.actor[-1].weight, std=.03)
                repeated = copy.deepcopy(fixed)
                oracle, base = torch.randn(12, 5, dtype=dtype), torch.randn(12, 3, dtype=dtype)
                points = torch.randn(128, 3, dtype=dtype) * .2
                copies = points.expand(12, -1, -1).clone()
                actions, targets = torch.randn(12, 3, dtype=dtype) * .1, torch.randn(12, dtype=dtype)
                optimizers = [torch.optim.Adam(agent.parameters(), lr=2e-4) for agent in (fixed, repeated)]
                atol, rtol = (3e-6, 3e-4) if dtype == torch.float32 else (1e-11, 1e-9)
                for _ in range(2):
                    outputs = []
                    for agent, optimizer, cloud in zip((fixed, repeated), optimizers, (points, copies)):
                        result = agent.evaluate(oracle, base, actions, cloud)
                        loss = -(result["log_prob"] * targets).mean() + (result["value"] - targets).square().mean()
                        loss -= .005 * result["entropy"].mean()
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        outputs.append(result)
                    for key in ("mean", "value", "log_prob", "entropy"):
                        torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=atol, rtol=rtol)
                    second_parameters = dict(repeated.named_parameters())
                    for prefix in ("actor_shape_encoder", "critic_shape_encoder"):
                        gradients = []
                        for name, parameter in fixed.named_parameters():
                            if name.startswith(prefix):
                                gradients.append(parameter.grad)
                                torch.testing.assert_close(parameter.grad, second_parameters[name].grad,
                                                           atol=atol, rtol=rtol)
                        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.)
                    for optimizer in optimizers:
                        optimizer.step()
                    for parameter, other in zip(fixed.parameters(), repeated.parameters()):
                        # Adam can amplify FP32 rounding of near-zero gradients.
                        torch.testing.assert_close(parameter, other, atol=max(atol, 1e-5 if dtype == torch.float32 else 0),
                                                   rtol=rtol)
                        for key in ("step", "exp_avg", "exp_avg_sq"):
                            torch.testing.assert_close(optimizers[0].state[parameter][key], optimizers[1].state[other][key],
                                                       atol=atol, rtol=rtol)

    def test_fixed_shape_ppo_recomputes_single_cloud_each_minibatch(self):
        torch.manual_seed(19)
        agent = ResidualActorCritic(5, action_dim=3, hidden=(16, 8))
        batch = rollout(agent, fixed_shape=True)
        before = {name: parameter.detach().clone() for name, parameter in agent.named_parameters()
                  if "shape_encoder" in name}
        seen = {name: [] for name in ("actor_shape_encoder", "critic_shape_encoder")}
        handles = []
        for name in seen:
            handles.append(getattr(agent, name).register_forward_pre_hook(
                lambda module, inputs, name=name: seen[name].append((tuple(inputs[0].shape), torch.is_grad_enabled()))))
        try:
            metrics = PPO(agent, PPOConfig(epochs=2, minibatches=2)).update(batch)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(metrics["optimizer_steps"], 4)
        for name, calls in seen.items():
            self.assertEqual(calls, [((128, 3), True)] * 4)
            self.assertTrue(any(not torch.equal(parameter, before[key]) for key, parameter in agent.named_parameters()
                                if key.startswith(name)))

    def test_object_history_normalizer_shares_current_tick_statistics_only(self):
        from hoi_residual.features import BLOCKS, ORACLE_DIM, SCHEMA
        agent = ResidualActorCritic(ORACLE_DIM, schema=SCHEMA, hidden=(16, 8))
        oracle, base = torch.randn(4, ORACLE_DIM), torch.randn(4, 29)
        start, end = BLOCKS["robot_privileged"], BLOCKS["robot_privileged"] + 90
        history = torch.full((4, 10, 9), 1e6)
        current = torch.arange(36, dtype=torch.float32).reshape(4, 9) * .01
        history[:, -1] = current
        oracle[:, start:end] = history.flatten(-2)
        before = oracle.clone()
        agent.update_normalizer(oracle, base)
        means = agent.norm.mean[start:end].reshape(10, 9)
        variances = agent.norm.var[start:end].reshape(10, 9)
        torch.testing.assert_close(means, current.mean(0).expand(10, -1))
        torch.testing.assert_close(variances, current.var(0, unbiased=False).expand(10, -1))
        self.assertEqual(float(agent.norm.count), 4.)
        self.assertTrue(torch.equal(oracle, before))
        normalized = agent.norm(agent._inputs(oracle, base))[:, start:end].reshape(4, 10, 9)
        self.assertTrue(bool((normalized[:, :-1] > 1e5).all()))
        self.assertLess(float(normalized[:, -1].abs().max()), 2.)
        # A second collection counts four new ticks, never 40 history values.
        current2 = current + .4
        oracle[:, end - 9:end] = current2
        agent.update_normalizer(oracle, base)
        pooled = torch.cat((current, current2))
        torch.testing.assert_close(agent.norm.mean[start:end].reshape(10, 9), pooled.mean(0).expand(10, -1))
        torch.testing.assert_close(agent.norm.var[start:end].reshape(10, 9), pooled.var(0, unbiased=False).expand(10, -1))
        self.assertEqual(float(agent.norm.count), 8.)

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
        shapes_before = {name: value.detach().clone() for name, value in agent.named_parameters()
                         if "shape_encoder" in name}
        ppo = PPO(agent, PPOConfig(epochs=2, minibatches=2))
        metrics = ppo.update(batch)
        self.assertEqual(metrics["optimizer_steps"], 4)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
        self.assertFalse(torch.equal(actor_before, agent.actor[-1].weight))
        self.assertFalse(torch.equal(critic_before, agent.critic[-1].weight))
        for prefix in ("actor_shape_encoder", "critic_shape_encoder"):
            parameters = [(name, value) for name, value in agent.named_parameters() if name.startswith(prefix)]
            self.assertTrue(any(not torch.equal(value, shapes_before[name]) for name, value in parameters), prefix)
            self.assertTrue(all(value.grad is not None and torch.isfinite(value.grad).all() for _, value in parameters))
            self.assertGreater(sum(float(value.grad.abs().sum()) for _, value in parameters), 0.)
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

    def test_staged_collectives_preserve_operators_and_copy_back_synchronously(self):
        # CPU-only transport simulation tests the staging path, not a claim
        # of native CUDA/Gloo support; the L40 job validates that separately.
        peer = torch.tensor([[4., -5.], [7., 1.]])
        for op in (dist.ReduceOp.SUM, dist.ReduceOp.MIN, dist.ReduceOp.MAX):
            value = torch.tensor([[2., 6.], [-3., 8.]], requires_grad=True).T
            before = value.detach().clone()
            expected = (before + peer if op == dist.ReduceOp.SUM else
                        torch.minimum(before, peer) if op == dist.ReduceOp.MIN else
                        torch.maximum(before, peer))
            def transport(buffer, *, op, async_op):
                self.assertFalse(async_op)
                self.assertFalse(torch.is_grad_enabled())
                self.assertEqual(buffer.device.type, "cpu")
                self.assertTrue(buffer.is_contiguous())
                self.assertNotEqual(buffer.data_ptr(), value.data_ptr())
                torch.testing.assert_close(value, before, atol=0, rtol=0)
                buffer.copy_(expected)
            with patch("hoi_residual.distributed._requires_cpu_staging", return_value=True), patch(
                    "hoi_residual.distributed.dist.all_reduce", side_effect=transport) as reduce:
                self.assertIs(all_reduce_tensor(value, op=op), value)
                self.assertEqual(reduce.call_args.kwargs["op"], op)
            torch.testing.assert_close(value, expected, atol=0, rtol=0)
            self.assertTrue(value.requires_grad)
        value, source = torch.zeros(3, dtype=torch.int64), torch.tensor([6, 2, 9])
        def broadcast(buffer, *, src, async_op):
            self.assertEqual(src, 3)
            self.assertFalse(async_op)
            self.assertNotEqual(buffer.data_ptr(), value.data_ptr())
            buffer.copy_(source)
        with patch("hoi_residual.distributed._requires_cpu_staging", return_value=True), patch(
                "hoi_residual.distributed.dist.broadcast", side_effect=broadcast):
            self.assertIs(broadcast_tensor(value, src=3), value)
        torch.testing.assert_close(value, source, atol=0, rtol=0)

    def test_cpu_collective_uses_native_in_place_transport(self):
        value = torch.tensor([1., 2., 3.])
        def transport(buffer, *, op, async_op):
            self.assertIs(buffer, value)
            self.assertEqual(op, dist.ReduceOp.SUM)
            self.assertFalse(async_op)
            buffer.mul_(2)
        with patch("hoi_residual.distributed.dist.all_reduce", side_effect=transport):
            self.assertIs(all_reduce_tensor(value), value)
        torch.testing.assert_close(value, torch.tensor([2., 4., 6.]), atol=0, rtol=0)

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
            self.assertEqual(first["optimizer"]["param_groups"], second["optimizer"]["param_groups"])
            for key, state in first["optimizer"]["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(value, second["optimizer"]["state"][key][name], atol=0, rtol=0)
            all_raw = torch.cat((first["raw"], second["raw"]))
            torch.testing.assert_close(first["state"]["norm.mean"], all_raw.mean(0))
            torch.testing.assert_close(first["state"]["norm.var"], all_raw.var(0, unbiased=False))
            self.assertEqual(first["metrics"], second["metrics"])
            self.assertEqual(first["metrics"]["world_size"], 2)
            for result in (first, second):
                self.assertEqual(result["shape_input"], [128, 3])
                self.assertEqual(result["shape_changed"], {"actor_shape_encoder": True, "critic_shape_encoder": True})


if __name__ == "__main__":
    unittest.main()
