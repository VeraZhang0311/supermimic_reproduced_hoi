"""CPU contracts for additive networks, physical gradients and migration."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn as nn

from supermimic.networks import Policy as LegacyPolicy, WorldModel as LegacyWM, Critic as LegacyCritic, mlp
from supermimic.rotations import so3_exp
from hoi.v1.state import (flatten_object, unflatten_object, object_features, object_targets,
                          object_state_error, integrate_object)
from hoi.v1.networks import Policy, WorldModel, Critic, ShapeEncoder, frozen_parameters
from hoi.v1.contacts import CHANNELS, LABEL_SCHEMA
from hoi.v1.checkpoint import make_checkpoint, load_checkpoint, save_checkpoint, migrate_legacy
from hoi.v1.checkpoint import checkpoint_training_progress


def states(batch=3):
    def vec(n):
        return torch.randn(batch, n) * .2
    S = vec(3), so3_exp(vec(3)), vec(3), vec(3), vec(29), vec(29)
    O = vec(3), so3_exp(vec(3)), vec(3), vec(3)
    return S, O


def modules():
    return Policy(hidden=(32, 16), object_hidden=(16,)), WorldModel(hidden=(32, 32), contact_hidden=12), Critic(hidden=(32, 16))


def legacy_models():
    policy = LegacyPolicy(99, 730)
    policy.enc, policy.dec = mlp(730, [32, 16], 32), mlp(1022, [32, 16], 29)
    wm = LegacyWM(hidden=(32, 32))
    nn.init.normal_(wm.net[-1].weight, std=.02)
    nn.init.normal_(wm.net[-1].bias, std=.01)
    critic = LegacyCritic(914)
    critic.net = mlp(914, [32, 16], 1)
    for model in (policy, wm, critic):
        for child in model.modules():
            if hasattr(child, "count"):
                child.mean.normal_(std=.2)
                child.var.uniform_(.2, 1.)
                child.count.fill_(2e8)
    return policy, wm, critic


PHYSICS = {"hand_collision": True, "self_collision": True}
DATA = {"manifest_sha256": "a" * 64, "timebase": {"source_hz": 60, "control_hz": 50},
        "contact_schema": "pair_interval_1N", "asset_sha256": "b" * 64}


class NetworksTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261023)

    def test_checkpoint_training_stage_uses_explicit_current_run_counts_only(self):
        cases=((12,3,"policy_updated_this_run"),(12,0,"wm_only_this_run"),(0,0,"no_optimizer_updates_this_run"))
        for wm,policy,stage in cases:
            with self.subTest(stage=stage):
                checkpoint={"iteration":20000,"extra":{"wm_updates_this_run":wm,"policy_updates_this_run":policy,
                            "wm_gate":{"passed":False}},"initialization":{"mode":"legacy_initialization"}}
                progress=checkpoint_training_progress(checkpoint)
                self.assertEqual(progress["stage"],stage)
                self.assertEqual(progress["wm_updates_this_run"],wm)
                self.assertEqual(progress["policy_updates_this_run"],policy)
                self.assertEqual(progress["counts_status"],"known")
                self.assertIn("reset on continuation",progress["count_scope"])
        unknown=checkpoint_training_progress({"iteration":20000,"extra":{"wm_gate":{"passed":True},
            "wm_updates":200000,"policy_updates":19000},"initialization":{"origin":{"policy_updates_this_run":999}}})
        self.assertEqual(unknown["stage"],"unknown")
        self.assertIsNone(unknown["wm_updates_this_run"])
        self.assertIsNone(unknown["policy_updates_this_run"])
        partial=checkpoint_training_progress({"extra":{"wm_updates_this_run":10,"policy_updates_this_run":False}})
        self.assertEqual(partial["stage"],"unknown")
        self.assertEqual(partial["wm_updates_this_run"],10)
        self.assertIsNone(partial["policy_updates_this_run"])
        resumed=checkpoint_training_progress({"extra":{"wm_updates_this_run":4,"policy_updates_this_run":0},
            "initialization":{"origin":{"policy_updates_this_run":999}}})
        self.assertEqual(resumed["stage"],"wm_only_this_run")
        self.assertEqual(resumed["policy_updates_this_run"],0)

    def test_no_geometry_variant_zeros_all_three_shape_interfaces_without_removing_modules(self):
        pi = Policy(hidden=(32,16), object_hidden=(16,), geometry_input_enabled=False)
        wm = WorldModel(hidden=(32,32), contact_hidden=12, geometry_input_enabled=False)
        critic = Critic(hidden=(32,16), geometry_input_enabled=False)
        points, changed = torch.randn(3,128,3), torch.randn(3,128,3)*10
        args = (torch.randn(3,10,99), torch.randn(3,730), torch.randn(3,10,18), torch.randn(3,190))
        action = pi(*args, points)
        torch.testing.assert_close(action, pi(*args, changed), atol=0, rtol=0)
        torch.testing.assert_close(action, pi(*args, points, shape_embedding=torch.randn(3,64)), atol=0, rtol=0)
        S,O = states()
        wm_args = (S,O,action.detach(),torch.randn(3,33),torch.randn(3,12))
        inputs = wm.inputs(*wm_args, points)
        torch.testing.assert_close(inputs[:,148:212], torch.zeros(3,64), atol=0, rtol=0)
        torch.testing.assert_close(inputs, wm.inputs(*wm_args, changed), atol=0, rtol=0)
        critic_args = (torch.randn(3,914),torch.randn(3,190),torch.randn(3,15),torch.randn(3,15),torch.randn(3,12),torch.randn(3,4))
        value = critic(*critic_args, points)
        torch.testing.assert_close(value, critic(*critic_args, changed), atol=0, rtol=0)
        _,_,logits = wm.step(*wm_args, points)
        (action.sum()+value.sum()+logits.sum()).backward()
        for model in (pi,wm,critic):
            self.assertIsInstance(model.shape_encoder, ShapeEncoder)
            for parameter in model.shape_encoder.parameters():
                self.assertIsNotNone(parameter.grad)
                torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter.grad), atol=0, rtol=0)

    def test_checkpoint_variants_strict_restore_and_old_main_only_compatibility(self):
        for variant in ("hoi_v1","no_contact_objective","no_geometry_input"):
            enabled = variant != "no_geometry_input"
            models = (Policy(hidden=(32,16),object_hidden=(16,),geometry_input_enabled=enabled),
                      WorldModel(hidden=(32,32),contact_hidden=12,geometry_input_enabled=enabled),
                      Critic(hidden=(32,16),geometry_input_enabled=enabled))
            optimizers = {key:torch.optim.Adam(model.parameters()) for key,model in zip(("pi","wm","critic"),models)}
            ck = make_checkpoint(*models,optimizers,iteration=3,physics=PHYSICS,data_identity=DATA,
                                 initialization={"mode":"CPU_variant_test"},extra={"config":{"variant":variant}})
            self.assertEqual(ck["variant"],variant)
            load_checkpoint(ck,*models,physics=PHYSICS,data_identity=DATA,variant=variant)
            other = "no_contact_objective" if variant == "hoi_v1" else "hoi_v1"
            with self.assertRaisesRegex(ValueError,"variant"):
                load_checkpoint(ck,*models,physics=PHYSICS,data_identity=DATA,variant=other)
            bad = copy.deepcopy(ck)
            bad["extra"]["config"]["variant"] = other
            with self.assertRaisesRegex(ValueError,"variant"):
                load_checkpoint(bad,*models,physics=PHYSICS,data_identity=DATA,variant=variant)
            if variant == "hoi_v1":
                old = copy.deepcopy(ck)
                old.pop("variant")
                old["extra"]["config"].pop("variant")
                for record in old["architecture"].values():
                    record.pop("geometry_input_enabled")
                load_checkpoint(old,*models,physics=PHYSICS,data_identity=DATA)
                with self.assertRaisesRegex(ValueError,"variant"):
                    load_checkpoint(old,*models,physics=PHYSICS,data_identity=DATA,variant="no_contact_objective")

    def test_default_production_architecture(self):
        pi, wm, critic = Policy(), WorldModel(), Critic()
        self.assertEqual(pi.enc[0].in_features, 730)
        self.assertEqual(pi.enc[-1].out_features, 32)
        self.assertEqual(pi.object_enc[0].in_features, 190)
        self.assertEqual(pi.object_enc[-1].out_features, 32)
        self.assertEqual([x.out_features for x in pi.dec if isinstance(x, nn.Linear)], [2048, 1024, 512, 256, 29])
        self.assertEqual(pi.dec[0].in_features, 1298)
        self.assertEqual([x.out_features for x in wm.net if isinstance(x, nn.Linear)], [2048, 2048, 2048, 82])
        self.assertEqual(wm.net[0].in_features, 224)
        self.assertEqual(wm.contact_head[0].in_features, 2048)
        self.assertEqual(wm.contact_head[-1].out_features, 5)
        self.assertEqual(critic.net[0].in_features, 1214)
        self.assertEqual([x.out_features for x in critic.net if isinstance(x, nn.Linear)], [256, 128, 1])
        self.assertEqual(len({id(m.shape_encoder) for m in (pi, wm, critic)}), 3)

    def test_state_roundtrip_and_heading_increment(self):
        S, O = states()
        restored = unflatten_object(flatten_object(O))
        for actual, expected in zip(restored, O):
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        d = torch.randn(3, 12) * .01
        O1 = integrate_object(S, O, d)
        torch.testing.assert_close(object_targets(S, O, O1), d, atol=2e-6, rtol=2e-4)
        self.assertEqual(object_features(S, O).shape, (3, 15))
        torch.testing.assert_close(object_state_error(S, O, O), torch.zeros(3, 12), atol=2e-6, rtol=0)

    def test_legacy_migration_exact_outputs_and_independent_statistics(self):
        legacy_pi, legacy_wm, legacy_v = legacy_models()
        pi, wm, critic = modules()
        ck = {"pi": legacy_pi.state_dict(), "wm": legacy_wm.state_dict(), "critic": legacy_v.state_dict(),
              "args": PHYSICS, "iteration": 100}
        report = migrate_legacy(ck, pi, wm, critic, physics=PHYSICS)
        self.assertEqual(report["iteration"], 0)
        self.assertEqual(report["source_iteration"], 100)
        S, O = states()
        obs, hg, oh, og, points = torch.randn(3, 10, 99), torch.randn(3, 730), torch.randn(3, 10, 18), torch.randn(3, 190), torch.randn(3, 128, 3)
        xi = torch.randn(3, 33)
        action = legacy_pi(obs, hg)
        torch.testing.assert_close(pi.human_latent(hg), torch.tanh(legacy_pi.enc(legacy_pi.goal_norm(hg))))
        torch.testing.assert_close(pi(obs, hg, oh, og, points), action, atol=2e-6, rtol=2e-5)
        S1 = legacy_wm.step(S, action, xi)
        newS, _, _ = wm.step(S, O, action, xi, torch.randn(3, 12), points)
        for new, old in zip(newS, S1):
            torch.testing.assert_close(new, old, atol=2e-6, rtol=2e-5)
        v = torch.randn(3, 914)
        torch.testing.assert_close(critic(v, og, torch.randn(3, 15), torch.randn(3, 15), torch.randn(3, 12), torch.randn(3, 4), points), legacy_v(v))
        self.assertEqual(float(pi.object_obs_norm.count), 0)
        self.assertEqual(float(wm.object_in_norm.count), 0)
        self.assertEqual(float(wm.in_norm.count), 2e8)
        wm.update_normalizers(S, O, action, xi, torch.randn(3, 12), points, S1, O)
        self.assertEqual(float(wm.object_in_norm.count), 3)
        self.assertEqual(float(wm.in_norm.count), 2e8)
        self.assertEqual(torch.count_nonzero(pi.dec[0].weight[:, 1022:]), 0)
        self.assertEqual(torch.count_nonzero(wm.net[0].weight[:, 133:]), 0)

    def test_world_model_has_bidirectional_trainable_coupling(self):
        _, wm, _ = modules()
        with torch.no_grad():
            wm.net[-1].weight.normal_(std=.05)
        S, O = states()
        S = tuple(x.detach().requires_grad_() for x in S)
        O = tuple(x.detach().requires_grad_() for x in O)
        action = torch.randn(3, 29, requires_grad=True)
        S1, O1, contact = wm.step(S, O, action, torch.randn(3, 33), torch.randn(3, 12), torch.randn(3, 128, 3))
        robot_depends_on_object = torch.autograd.grad(S1[0].square().sum(), O[0], retain_graph=True)[0]
        object_depends_on_action = torch.autograd.grad(O1[0].square().sum(), action, retain_graph=True)[0]
        self.assertGreater(robot_depends_on_object.abs().sum().item(), 0)
        self.assertGreater(object_depends_on_action.abs().sum().item(), 0)
        (S1[0].sum() + O1[0].sum() + contact.sum()).backward()
        self.assertGreater(wm.net[0].weight.grad[:, 133:].abs().sum().item(), 0)
        self.assertGreater(sum(p.grad.abs().sum().item() for p in wm.shape_encoder.parameters() if p.grad is not None), 0)

    def test_frozen_wm_retains_action_gradient_and_matches_finite_difference(self):
        _, wm, _ = modules()
        with torch.no_grad():
            wm.net[-1].weight.normal_(std=.05)
        wm.double()
        S, O = states(1)
        S, O = tuple(x.double() for x in S), tuple(x.double() for x in O)
        action = torch.randn(1, 29, dtype=torch.float64, requires_grad=True)
        xi, xio, points = torch.randn(1, 33, dtype=torch.float64), torch.randn(1, 12, dtype=torch.float64), torch.randn(1, 128, 3, dtype=torch.float64)
        before = copy.deepcopy(wm.state_dict())
        def objective(a):
            _, O1, logits = wm.step(S, O, a, xi, xio, points)
            return O1[0].square().sum() + torch.sigmoid(logits).sum() * .1
        with frozen_parameters(wm):
            value = objective(action)
            grad = torch.autograd.grad(value, action)[0]
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)
            direction = torch.randn_like(action)
            step = 1e-5
            fd = (objective(action.detach() + step * direction) - objective(action.detach() - step * direction)) / (2 * step)
            torch.testing.assert_close((grad * direction).sum(), fd, rtol=1e-5, atol=1e-7)
            self.assertTrue(all(p.grad is None for p in wm.parameters()))
        self.assertTrue(all(p.requires_grad for p in wm.parameters()))
        for k, v in before.items():
            torch.testing.assert_close(wm.state_dict()[k], v)

    def test_critic_pointnet_receives_value_loss_gradients(self):
        _, _, v = modules()
        features = [torch.randn(3, d).detach() for d in (914, 190, 15, 15, 12, 4)]
        value = v(*features, torch.randn(3, 128, 3).detach())
        value.square().mean().backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in v.shape_encoder.parameters() if p.grad is not None), 0)

    def test_actor_shape_same_graph_reuse_matches_repeated_objective_and_all_gradients(self):
        policy = Policy(hidden=(32, 16), object_hidden=(16,)).double()
        reused = copy.deepcopy(policy)
        robot, obj = torch.randn(2, 10, 99, dtype=torch.float64), torch.randn(2, 10, 18, dtype=torch.float64)
        hg, og, points = torch.randn(2, 730, dtype=torch.float64), torch.randn(2, 190, dtype=torch.float64), torch.randn(2, 128, 3, dtype=torch.float64)
        robot_map = torch.randn(29, 99, dtype=torch.float64) * .01
        object_map = torch.randn(29, 18, dtype=torch.float64) * .01
        goal_map = torch.randn(29, 190, dtype=torch.float64) * .01
        def objective(actor, use_reuse):
            r, o, goal = robot.clone(), obj.clone(), og.clone()
            encoded = actor.shape_encoder(points) if use_reuse else None
            result = torch.zeros((), dtype=torch.float64)
            for tick in range(4):
                action = actor(r, hg, o, goal, points, shape_embedding=encoded)
                # A differentiable feedback rollout makes later observations
                # depend on earlier actions; the cache stays on this one graph.
                r = torch.cat((r[:, 1:], (.99 * r[:, -1] + action @ robot_map)[:, None]), 1)
                o = torch.cat((o[:, 1:], (o[:, -1] + action @ object_map)[:, None]), 1)
                goal = goal + action @ goal_map
                result = result + action.square().mean() + o[:, -1].square().mean()
            return result
        original, cached = objective(policy, False), objective(reused, True)
        original.backward()
        cached.backward()
        torch.testing.assert_close(original, cached, atol=1e-12, rtol=1e-10)
        for (name, first), (other_name, second) in zip(policy.named_parameters(), reused.named_parameters()):
            self.assertEqual(name, other_name)
            self.assertIsNotNone(first.grad, name)
            self.assertIsNotNone(second.grad, name)
            torch.testing.assert_close(first.grad, second.grad, atol=1e-10, rtol=1e-7, msg=name)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in reused.shape_encoder.parameters()), 0)
        with self.assertRaisesRegex(ValueError, "64 features"):
            reused(robot, hg, obj, og, points, shape_embedding=torch.zeros(2, 63))
        with self.assertRaisesRegex(ValueError, "batch"):
            reused(robot, hg, obj, og, points, shape_embedding=torch.zeros(3, 64))

    def test_terminal_mask_contact_validity_and_loss(self):
        _, wm, _ = modules()
        S, O = states(2)
        S = tuple(x[:, None].expand((2, 2) + x.shape[1:]) for x in S)
        O = tuple(x[:, None].expand((2, 2) + x.shape[1:]) for x in O)
        O_bad = tuple(x + .01 if i != 1 else x for i, x in enumerate(O))
        logits = torch.zeros(2, 2, len(CHANNELS), requires_grad=True)
        valid = torch.tensor([[[1., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]], [[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.]]])
        mask = torch.tensor([[1., 0.], [1., 0.]])
        losses = wm.loss(S, O_bad, S, O, logits, torch.zeros_like(logits), valid, mask)
        torch.testing.assert_close(losses["contact"], torch.tensor(.6931472))
        self.assertEqual(float(losses["robot"]), 0)
        self.assertGreater(float(losses["object"]), 0)
        losses["total"].backward()
        self.assertEqual(float(logits.grad[:, 1].abs().sum()), 0)
        self.assertEqual(float(logits.grad[0, 0, 1:].abs().sum()), 0)

    def test_strict_checkpoint_roundtrip_and_mismatch_rejection(self):
        pi, wm, critic = modules()
        optimizers = {key: torch.optim.Adam(m.parameters()) for key, m in zip(("pi", "wm", "critic"), (pi, wm, critic))}
        checkpoint = make_checkpoint(pi, wm, critic, optimizers, iteration=12, physics=PHYSICS,
                                     data_identity=DATA, initialization={"kind": "fresh", "seed": 0})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ckpt.pt"
            save_checkpoint(path, pi, wm, critic, optimizers, iteration=12, physics=PHYSICS,
                            data_identity=DATA, initialization={"kind": "fresh", "seed": 0})
            restored = load_checkpoint(path, pi, wm, critic, optimizers, physics=PHYSICS, data_identity=DATA)
            self.assertEqual(restored["iteration"], 12)
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, pi, wm, critic, optimizers, iteration=12, physics=PHYSICS,
                                data_identity=DATA, initialization={"kind": "fresh"})
        for kwargs in ({"physics": {**PHYSICS, "self_collision": False}, "data_identity": DATA},
                       {"physics": PHYSICS, "data_identity": {**DATA, "contact_schema": "net_force"}}):
            with self.assertRaises(ValueError):
                load_checkpoint(checkpoint, pi, wm, critic, **kwargs)
        bad = copy.deepcopy(checkpoint)
        bad["feature_schema"]["version"] = "unknown"
        with self.assertRaises(ValueError):
            load_checkpoint(bad, pi, wm, critic, physics=PHYSICS, data_identity=DATA)
        bad = copy.deepcopy(checkpoint)
        bad["wm"]["object_in_norm.var"][0] = -1
        with self.assertRaises(ValueError):
            load_checkpoint(bad, pi, wm, critic, physics=PHYSICS, data_identity=DATA)
        altered_eps = copy.deepcopy(wm)
        altered_eps.object_in_norm.eps = .5
        with self.assertRaises(ValueError):
            load_checkpoint(checkpoint, pi, altered_eps, critic, physics=PHYSICS, data_identity=DATA)

    def test_feet_supervision_reaches_each_head_row_and_shared_dynamics_features(self):
        _, wm, _ = modules()
        S, O = states(2)
        S1, O1, logits = wm.step(S, O, torch.randn(2, 29), torch.randn(2, 33),
                                  torch.randn(2, 12), torch.randn(2, 128, 3))
        self.assertEqual(CHANNELS, ("left_hand_object", "right_hand_object", "object_ground",
                                    "left_foot_object", "right_foot_object"))
        self.assertEqual(logits.shape, (2, 5))
        # Two distinct executed labels: left kick then right kick. Only the
        # feet are supervised here, so old channels cannot create this gradient.
        targets = torch.tensor([[0., 0., 0., 1., 0.], [0., 0., 0., 0., 1.]])
        valid = torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.bool).expand(2, -1)
        loss = wm.loss(S1, O1, S1, O1, logits, targets, valid, torch.ones(2),
                       pos_weight=torch.tensor([1., 1., 1., 3., 7.]))["contact"]
        logits.retain_grad()
        loss.backward()
        self.assertTrue(bool((logits.grad[:, 3:] * (1 - 2 * targets[:, 3:]) > 0).all()))
        self.assertEqual(float(logits.grad[:, :3].abs().sum()), 0.)
        self.assertTrue(bool((wm.contact_head[-1].weight.grad[3:].abs().sum(-1) > 0).all()))
        self.assertEqual(float(wm.contact_head[-1].weight.grad[:3].abs().sum()), 0.)
        self.assertGreater(float(wm.net[0].weight.grad.abs().sum()), 0.)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in wm.shape_encoder.parameters() if p.grad is not None), 0.)
        # Invalid labels do not contribute, even for a foot-positive target.
        wm.zero_grad(set_to_none=True)
        independent = torch.zeros(2, 5, requires_grad=True)
        valid = valid.clone()
        valid[:, 3] = False
        wm.loss(S1, O1, S1, O1, independent, targets, valid, torch.ones(2))["contact"].backward()
        self.assertEqual(float(independent.grad[:, 3].abs().sum()), 0.)
        self.assertGreater(float(independent.grad[:, 4].abs().sum()), 0.)
        with self.assertRaisesRegex(ValueError, "dimension 5"):
            wm.loss(S1, O1, S1, O1, independent[:, :3], targets[:, :3], valid[:, :3], torch.ones(2))

    def test_three_channel_hoi_checkpoint_is_rejected_before_any_mutation(self):
        models = modules()
        optimizers = {key: torch.optim.Adam(model.parameters()) for key, model in zip(("pi", "wm", "critic"), models)}
        checkpoint = make_checkpoint(*models, optimizers, iteration=12, physics=PHYSICS,
                                     data_identity=DATA, initialization={"kind": "fresh"})
        self.assertEqual(checkpoint["feature_schema"]["contact"], list(CHANNELS))
        self.assertEqual(checkpoint["feature_schema"]["actual_contact_label"]["schema"], LABEL_SCHEMA)
        old = copy.deepcopy(checkpoint)
        old["feature_schema"]["version"] = "supermimic_hoi_v1_full_reference_1"
        old["feature_schema"]["contact"] = list(CHANNELS[:3])
        old["feature_schema"]["actual_contact_label"]["schema"] = "hoi_v1_pair_normal_impulse_v1"
        last = str(len(models[1].contact_head) - 1)
        for field in ("weight", "bias"):
            key = "contact_head." + last + "." + field
            old["wm"][key] = old["wm"][key][:3].clone()
        before = [copy.deepcopy(model.state_dict()) for model in models]
        with self.assertRaisesRegex(ValueError, "old three-channel HOI checkpoints"):
            load_checkpoint(old, *models, optimizers, physics=PHYSICS, data_identity=DATA)
        for model, state in zip(models, before):
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, state[name], rtol=0, atol=0)

    def test_legacy_unknown_physics_and_no_critic(self):
        lpi, lwm, _ = legacy_models()
        pi, wm, critic = modules()
        ck = {"pi": lpi.state_dict(), "wm": lwm.state_dict()}
        with self.assertRaises(ValueError):
            migrate_legacy(ck, pi, wm, critic, physics=PHYSICS)
        report = migrate_legacy(ck, pi, wm, critic, physics=PHYSICS, legacy_physics=PHYSICS)
        self.assertEqual(report["critic"], "fresh_initialization")
        with self.assertRaises(ValueError):
            migrate_legacy(ck, pi, wm, critic, physics=PHYSICS, legacy_physics={**PHYSICS, "hand_collision": False})


if __name__ == "__main__":
    unittest.main()
