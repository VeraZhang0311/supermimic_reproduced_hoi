"""CPU architecture, checkpoint and differentiable-rollout evidence for v1.2.

Reduced MLPs in most fixtures bound CPU runtime; the attention dimensions stay
at their production values. These tests provide no native/GPU training evidence.
"""
from copy import deepcopy
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hoi.v1.actor_fusion import FUSION_ACTOR, MLP_ACTOR, actor_architecture_contract
from hoi.v1.checkpoint import (checkpoint_actor_architecture, load_checkpoint,
                               make_checkpoint, validate_checkpoint_features)
from hoi.v1.learner import Learner
from hoi.v1.networks import Policy, frozen_parameters
from hoi.v1.observations import MATCHING
from hoi.v1.randomization import MATCHING_DR
from hoi.v1.train import read_config
from test_hoi_v1_1_observations import matching_learner, ChainKinematics
from test_hoi_v1_learner import fill


def fusion_policy(**kwargs):
    return Policy(hidden=(32, 16), object_hidden=(16,), observation_profile=MATCHING,
                  actor_architecture=FUSION_ACTOR, **kwargs)


def inputs(batch=(2,), *, dtype=torch.float32, requires_grad=False):
    return tuple((torch.randn(batch + shape, dtype=dtype) * .2).requires_grad_(requires_grad)
                 for shape in ((10, 93), (448,), (10, 9), (63,), (128, 3)))


def wake_fusion(policy):
    # The production initialization starts with zero new decoder columns. This
    # models their having learned nonzero weights, so gradients reach attention.
    with torch.no_grad():
        policy.dec[0].weight[:, 1148:].normal_(std=.01)


def fusion_learner():
    base = matching_learner()
    return Learner(fusion_policy(), base.wm, base.critic, base.motion, base.mdp,
                   base.shapes, base.buffer,
                   replace(base.config, actor_architecture=FUSION_ACTOR))


class PerceptionFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12025)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)

    def assert_finite_nonzero_gradients(self, module):
        gradients = [parameter.grad for parameter in module.parameters()
                     if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.)

    def test_additive_initialization_preserves_v1_1_modules_and_decoder_prefix(self):
        torch.manual_seed(121)
        original = Policy(hidden=(32, 16), object_hidden=(16,), observation_profile=MATCHING)
        original_rng = torch.get_rng_state().clone()
        torch.manual_seed(121)
        policy = fusion_policy()
        torch.testing.assert_close(torch.get_rng_state(), original_rng, atol=0, rtol=0)
        self.assertEqual(policy.dec[0].in_features, 1660)
        self.assertEqual([layer.out_features for layer in policy.dec if isinstance(layer, nn.Linear)],
                         [32, 16, 29])
        self.assertEqual(torch.count_nonzero(policy.dec[0].weight[:, 1148:]).item(), 0)
        for key, expected in original.state_dict().items():
            actual = policy.state_dict()[key]
            if key == "dec.0.weight":
                actual = actual[:, :1148]
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        data = inputs()
        torch.testing.assert_close(policy(*data), original(*data), atol=1e-8, rtol=2e-5)
        self.assertEqual(len(policy.fusion.layers), 2)
        for layer in policy.fusion.layers:
            self.assertEqual((layer.self_attn.embed_dim, layer.self_attn.num_heads), (256, 4))
            self.assertEqual(layer.linear1.out_features, 1024)
            self.assertTrue(layer.norm_first)
            self.assertEqual(layer.dropout.p, 0)
        self.assertFalse(torch.equal(policy.fusion.layers[0].linear1.weight,
                                     policy.fusion.layers[1].linear1.weight))

    def test_normalized_raw_goals_feed_tokens_and_old_1148_prefix_is_retained(self):
        policy, data = fusion_policy(), inputs()
        # Distinct statistics make accidental raw inputs or repeated normalization
        # observable; object values deliberately exceed the robot clipping range.
        with torch.no_grad():
            policy.obs_norm.mean.fill_(.1)
            policy.goal_norm.var.fill_(.5)
            policy.object_obs_norm.mean.fill_(-7.)
            policy.object_goal_norm.mean.fill_(9.)
        normalized = (policy.obs_norm(data[0]).flatten(-2), policy.goal_norm(data[1]),
                      policy.object_obs_norm(data[2]).flatten(-2), policy.object_goal_norm(data[3]),
                      policy.shape_encoder(data[4]))
        token_inputs, decoder_inputs, fused_outputs = [], [], []
        handles = [policy.fusion.register_forward_pre_hook(lambda module, args: token_inputs.append(args)),
                   policy.fusion.register_forward_hook(lambda module, args, value: fused_outputs.append(value)),
                   policy.dec[0].register_forward_pre_hook(lambda module, args: decoder_inputs.append(args[0]))]
        self.addCleanup(lambda: [handle.remove() for handle in handles])
        statistics = {name: value.clone() for name, value in policy.state_dict().items()
                      if "_norm." in name}
        policy(*data)
        self.assertEqual([value.shape[-1] for value in token_inputs[0]], [930, 448, 90, 63, 64])
        for actual, expected in zip(token_inputs[0], normalized):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertGreater(float(token_inputs[0][2].abs().max()), 5.)
        self.assertGreater(float(token_inputs[0][3].abs().max()), 5.)
        expected_prefix = torch.cat((policy.human_latent(data[1]), normalized[0], normalized[2],
                                     torch.tanh(policy.object_enc(normalized[3])), normalized[4]), -1)
        torch.testing.assert_close(decoder_inputs[0][..., :1148], expected_prefix, atol=0, rtol=0)
        self.assertEqual(fused_outputs[0].shape, (2, 512))
        torch.testing.assert_close(decoder_inputs[0][..., 1148:], fused_outputs[0], atol=0, rtol=0)
        for name, expected in statistics.items():
            torch.testing.assert_close(policy.state_dict()[name], expected, atol=0, rtol=0)

    def test_fusion_reads_robot_and_object_tokens_after_joint_attention(self):
        policy, data = fusion_policy(), inputs()
        outputs = []
        handle = policy.fusion.output_norm.register_forward_hook(
            lambda module, args, value: outputs.append(value))
        self.addCleanup(handle.remove)
        normalized = (policy.obs_norm(data[0]).flatten(-2), policy.goal_norm(data[1]),
                      policy.object_obs_norm(data[2]).flatten(-2), policy.object_goal_norm(data[3]),
                      policy.shape_encoder(data[4]))
        fused = policy.fusion(*normalized)
        self.assertEqual(outputs[0].shape, (2, 5, 256))
        torch.testing.assert_close(fused, torch.cat((outputs[0][..., 0, :], outputs[0][..., 2, :]), -1))
        # These modalities are not selected for readout. Their gradients show
        # that they nevertheless contribute through joint attention.
        raw = tuple(value.detach().requires_grad_() for value in normalized)
        score = (policy.fusion(*raw) * torch.linspace(-1., 1., 512)).sum()
        gradients = torch.autograd.grad(score, raw)
        for gradient in gradients:
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_zero_extended_decoder_can_learn_without_a_dead_fusion_gate(self):
        policy, data = fusion_policy(), inputs()
        optimizer = torch.optim.SGD(policy.parameters(), lr=.1)
        policy(*data).sum().backward()
        self.assertGreater(float(policy.dec[0].weight.grad[:, 1148:].abs().sum()), 0.)
        for parameter in policy.fusion.parameters():
            if parameter.grad is not None:
                self.assertEqual(float(parameter.grad.abs().sum()), 0.)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        policy(*data).sum().backward()
        self.assert_finite_nonzero_gradients(policy.fusion)
        self.assert_finite_nonzero_gradients(policy.shape_encoder)

    def test_deterministic_forward_has_no_cross_batch_attention_and_supports_leading_dims(self):
        policy, data = fusion_policy(), inputs((2, 3))
        wake_fusion(policy)
        first = policy(*data)
        self.assertEqual(first.shape, (2, 3, 29))
        policy.eval()
        torch.testing.assert_close(policy(*data), first, atol=0, rtol=0)
        flat = tuple(value.reshape((-1,) + value.shape[2:]) for value in data)
        torch.testing.assert_close(policy(*flat).reshape(2, 3, 29), first, atol=1e-8, rtol=2e-5)
        changed = tuple(value.clone() for value in flat)
        for value in changed:
            value[1:] += 20.
        torch.testing.assert_close(policy(*changed)[0], first[0, 0], atol=1e-8, rtol=2e-5)

    def test_geometry_disable_zeros_both_branches_and_preserves_shape_module(self):
        policy, data = fusion_policy(geometry_input_enabled=False), inputs()
        wake_fusion(policy)
        other = (*data[:4], torch.randn_like(data[4]) * 30.)
        torch.testing.assert_close(policy(*data), policy(*other), atol=0, rtol=0)
        torch.testing.assert_close(policy(*data), policy(*data, shape_embedding=torch.randn(2, 64)), atol=0, rtol=0)
        policy(*data).sum().backward()
        for parameter in policy.shape_encoder.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertEqual(float(parameter.grad.abs().sum()), 0.)

    def test_full_production_policy_forward_backward_and_all_inputs_have_finite_gradients(self):
        policy = Policy(observation_profile=MATCHING, actor_architecture=FUSION_ACTOR)
        wake_fusion(policy)
        data = inputs(requires_grad=True)
        action = policy(*data)
        self.assertEqual(action.shape, (2, 29))
        self.assertEqual([layer.out_features for layer in policy.dec if isinstance(layer, nn.Linear)],
                         [2048, 1024, 512, 256, 29])
        action.square().sum().backward()
        for value in data:
            self.assertTrue(bool(torch.isfinite(value.grad).all()))
            self.assertGreater(float(value.grad.abs().sum()), 0.)
        for module in (policy.enc, policy.object_enc, policy.dec, policy.shape_encoder, policy.fusion):
            self.assert_finite_nonzero_gradients(module)

    def test_float64_fused_action_jacobian_matches_finite_difference(self):
        policy = fusion_policy().double()
        wake_fusion(policy)
        data = inputs((1,), dtype=torch.float64, requires_grad=True)
        weights = torch.linspace(-.5, .8, 29, dtype=torch.float64)
        score = (policy(*data) * weights).sum()
        gradient = torch.autograd.grad(score, data[3])[0]
        self.assertGreater(float(gradient.abs().sum()), 0.)
        direction = gradient / gradient.norm()
        step, values = 1e-5, []
        for sign in (1, -1):
            shifted = (*data[:3], data[3].detach() + sign * step * direction, data[4])
            values.append((policy(*shifted) * weights).sum())
        torch.testing.assert_close((gradient * direction).sum(), (values[0] - values[1]) / (2 * step),
                                   atol=2e-10, rtol=2e-5)

    def test_imagination_recomputes_fusion_with_live_gradients_and_frozen_wm_critic(self):
        learner = fusion_learner()
        wake_fusion(learner.pi)
        fill(learner)
        start, history = learner.buffer.sample_starts(2, 10, horizon=3, motion=learner.motion)
        noise = [torch.zeros(2, 10, 93)] + [torch.zeros(2, 93) for _ in range(3)]
        snapshots = [deepcopy(module.state_dict()) for module in (learner.wm, learner.critic)]
        normalizers = {name: value.clone() for name, value in learner.pi.state_dict().items()
                       if "_norm." in name}
        fusion_inputs = []
        handle = learner.pi.fusion.register_forward_pre_hook(
            lambda module, args: fusion_inputs.append(args))
        self.addCleanup(handle.remove)
        with frozen_parameters(learner.wm, learner.critic):
            trajectory = learner.imagine(start, history, fixed_robot_noise=noise)
            forward_inputs = list(fusion_inputs)
            (-trajectory["objective"]).backward()
        self.assertEqual(len(forward_inputs), 3)
        self.assertGreater(len(fusion_inputs), 3)  # Backward recomputes activations.
        for step, arguments in enumerate(forward_inputs):
            state, obj, frame = trajectory["states"][step], trajectory["objects"][step], trajectory["frames"][step]
            torch.testing.assert_close(arguments[1], learner.pi.goal_norm(learner.actor_obs.human_goal(state, frame)))
            torch.testing.assert_close(arguments[3], learner.pi.object_goal_norm(learner.actor_obs.object_goal(state, obj, frame)))
        for module in (learner.pi.fusion, learner.pi.dec, learner.pi.shape_encoder):
            self.assert_finite_nonzero_gradients(module)
        for module, before in zip((learner.wm, learner.critic), snapshots):
            self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))
            for name, value in module.state_dict().items():
                torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        for name, value in normalizers.items():
            torch.testing.assert_close(learner.pi.state_dict()[name], value, atol=0, rtol=0)

    def test_H32_checkpoint_preserves_trajectory_action_jacobian_and_all_actor_gradients(self):
        learner = fusion_learner()
        learner.config = replace(learner.config, horizon=32)
        wake_fusion(learner.pi)
        fill(learner)
        start, history = learner.buffer.sample_starts(2, 10, horizon=32, motion=learner.motion)
        reference = deepcopy(learner)
        noise = [torch.zeros(2, 10, 93)] + [torch.zeros(2, 93) for _ in range(32)]

        def run(value):
            saved_bytes = []
            def pack(tensor):
                saved_bytes.append(tensor.numel() * tensor.element_size())
                return tensor
            with frozen_parameters(value.wm, value.critic):
                with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                    result = value.imagine(start, history, fixed_robot_noise=noise)
                rng_before_backward = torch.get_rng_state().clone()
                jacobian = torch.autograd.grad(result["states"][1][0].sum(),
                                               result["actions"][0], retain_graph=True)[0]
                (-result["objective"]).backward()
                torch.testing.assert_close(torch.get_rng_state(), rng_before_backward, atol=0, rtol=0)
            self.assertTrue(all(parameter.grad is None for model in (value.wm, value.critic)
                                for parameter in model.parameters()))
            return result, jacobian, sum(saved_bytes)

        # Compare the same weights, data, objective and H32 against direct
        # execution. Only the activation retention strategy differs.
        with patch.object(reference, "_imagination_network",
                          new=lambda function, *args, **kwargs: function(*args, **kwargs)):
            direct, direct_jacobian, direct_bytes = run(reference)
        actual, actual_jacobian, actual_bytes = run(learner)
        self.assertLess(actual_bytes, direct_bytes * .5)
        torch.testing.assert_close(actual["objective"], direct["objective"], atol=0, rtol=0)
        torch.testing.assert_close(actual_jacobian, direct_jacobian, atol=1e-7, rtol=2e-5)
        self.assertGreater(float(actual_jacobian.abs().sum()), 0.)
        for key in ("reward", "failure", "survival", "terminal_survival"):
            torch.testing.assert_close(actual[key], direct[key], atol=0, rtol=0)
        for action, expected in zip(actual["actions"], direct["actions"]):
            torch.testing.assert_close(action, expected, atol=0, rtol=0)
        for (name, parameter), (expected_name, expected) in zip(
                learner.pi.named_parameters(), reference.pi.named_parameters()):
            self.assertEqual(name, expected_name)
            self.assertIsNotNone(parameter.grad, name)
            torch.testing.assert_close(parameter.grad, expected.grad, atol=1e-7, rtol=3e-5, msg=name)
        self.assert_finite_nonzero_gradients(learner.pi.fusion)
        self.assert_finite_nonzero_gradients(learner.pi.shape_encoder)

    def test_checkpoint_roundtrip_old_mlp_compatibility_and_cross_architecture_rejection(self):
        learner = fusion_learner()
        kwargs = dict(iteration=7, physics={"hand_collision": True, "self_collision": True},
                      data_identity={"synthetic": True}, initialization={"mode": "CPU_fixture"},
                      extra={"config": learner.config.to_dict()})
        saved = make_checkpoint(learner.pi, learner.wm, learner.critic, learner.optimizers, **kwargs)
        self.assertEqual(checkpoint_actor_architecture(saved), FUSION_ACTOR)
        self.assertEqual(saved["actor_architecture_contract"], actor_architecture_contract(FUSION_ACTOR, MATCHING))
        load_checkpoint(saved, learner.pi, learner.wm, learner.critic, learner.optimizers,
                        physics=kwargs["physics"], data_identity=kwargs["data_identity"],
                        domain_randomization_profile=MATCHING_DR)
        original = matching_learner()
        old_config = original.config.to_dict()
        old_config.pop("actor_architecture")
        old = make_checkpoint(original.pi, original.wm, original.critic, original.optimizers,
                              **{**kwargs, "extra": {"config": old_config}})
        self.assertNotIn("actor_architecture_contract", old)
        self.assertEqual(checkpoint_actor_architecture(old), MLP_ACTOR)
        load_checkpoint(old, original.pi, original.wm, original.critic,
                        physics=kwargs["physics"], data_identity=kwargs["data_identity"])
        for snapshot, target in ((old, learner), (saved, original)):
            before = deepcopy(target.pi.state_dict())
            with self.assertRaisesRegex(ValueError, "architecture"):
                load_checkpoint(snapshot, target.pi, target.wm, target.critic,
                                physics=kwargs["physics"], data_identity=kwargs["data_identity"])
            for name, value in target.pi.state_dict().items():
                torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        bad = deepcopy(saved)
        bad.pop("actor_architecture_contract")
        with self.assertRaisesRegex(ValueError, "architecture|contract"):
            validate_checkpoint_features(bad)
        bad = deepcopy(saved)
        bad["extra"]["config"]["actor_architecture"] = MLP_ACTOR
        with self.assertRaisesRegex(ValueError, "architecture"):
            validate_checkpoint_features(bad)

    def test_formal_config_preserves_numerical_schedule_and_preflight_starts_no_workers(self):
        root = Path(__file__).resolve().parents[1]
        config, changed = read_config(root / "configs/hoi_v1_2_perception_fusion.json")
        self.assertEqual(changed, {})
        self.assertEqual((config.actor_architecture, config.observation_profile,
                          config.domain_randomization_profile), (FUSION_ACTOR, MATCHING, MATCHING_DR))
        self.assertEqual((config.num_envs, config.pi_batch, config.horizon, config.lr, config.wm_only_iters),
                         (4096, 4096, 32, .0002, 500))
        with self.assertRaisesRegex(ValueError, "observation|profile"):
            replace(config, observation_profile="v1").validate()
        from hoi.v1.train import main
        bundle = {"partition": "train", "scope": {"is_subset": False}, "assets": [], "clips": [],
                  "bundle_content_sha256": "a" * 64, "manifest": {"sha256": "b" * 64},
                  "recipes": {}, "robot_fk": {}, "reference_kinematics": {}, "native_robot_asset": {},
                  "hands": {"geometry_cache": {"sha256": "c" * 64}}, "timebase": {}}
        with tempfile.TemporaryDirectory() as temp:
            # This preflight fixture has no evaluation corpus. A two-iteration
            # limit avoids the unrelated full-HOI80 dispatch requirement.
            short_config = Path(temp) / "config.json"
            short_config.write_text(json.dumps({**config.to_dict(), "iterations": 2}), encoding="utf-8")
            with patch("hoi.v1.evaluate._read_bundle_header", return_value=bundle), patch(
                    "hoi.v1.workers.WorkerGroup") as workers, redirect_stdout(io.StringIO()):
                plan = main(["--bundle", str(Path(temp) / "unused.json"), "--config",
                             str(short_config), "--output",
                             str(Path(temp) / "preflight"), "--preflight"])
            workers.assert_not_called()
            self.assertFalse(plan["training_started"])
            self.assertEqual(plan["actor_architecture_contract"], actor_architecture_contract(FUSION_ACTOR, MATCHING))

    def test_frozen_evaluation_retains_full_duration_and_fusion_contract_without_updates(self):
        from hoi.v1.evaluate import evaluate_group
        from hoi.v1.mdp import HOIMDP
        from test_hoi_v1_evaluate import fixture, SyntheticWorkers
        _, motion, _, shapes, fk, metadata = fixture()
        kin = ChainKinematics()
        motion.body_p, motion.body_R, motion.body_v, motion.body_w = kin.forward(*motion.ref_arrays())
        mdp = HOIMDP(kin, motion, torch.zeros(29), shapes)
        policy = fusion_policy()
        wake_fusion(policy)
        before = deepcopy(policy.state_dict())
        workers = SyntheticWorkers(motion, mdp)
        with tempfile.TemporaryDirectory() as temp:
            cases = evaluate_group(policy, motion, mdp, shapes, workers, [0, 1], fk39=fk,
                                   joint_names=fk.dof_names, output=Path(temp) / "eval", metadata=metadata,
                                   verify_tracked_fk=False)
            for index, length in enumerate((3, 6)):
                raw = torch.load(cases[index]["rollout"], weights_only=True)
                self.assertEqual(raw["states"]["s"].shape[0], length + 1)
                self.assertEqual(raw["control"]["action"].shape[0], length)
                self.assertEqual(raw["metadata"]["actor_architecture_contract"],
                                 actor_architecture_contract(FUSION_ACTOR, MATCHING))
        for name, value in policy.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
