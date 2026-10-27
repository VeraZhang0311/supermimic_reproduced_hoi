"""CPU end-to-end dynamics-context checks; no native physics or training claim."""
from copy import deepcopy
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hoi.v1.checkpoint import (make_checkpoint, load_checkpoint, validate_checkpoint_features,
                               checkpoint_torso_mass_context)
from hoi.v1.evaluate import evaluate_group
from hoi.v1.native import HOITrackingSim
from hoi.v1.networks import feature_schema, WorldModel, Critic
from hoi.v1.robot_context import MASS_FIELDS, mass_context_from, normalize_torso_mass, torso_mass_contract
from hoi.v1.state import unflatten_state, unflatten_object
from hoi.v1.workers import _snapshot
from test_hoi_v1_1_observations import matching_learner, ChainKinematics
from test_hoi_v1_learner import fill, transition
from test_hoi_v1_evaluate import fixture, SyntheticWorkers


class NominalMassWorkers(SyntheticWorkers):
    def reset(self, starts):
        super().reset(starts)
        self.value.update(torso_mass=torch.full((self.N, 1), 8.), torso_mass_nominal=torch.full((self.N, 1), 8.))
        return self.snapshot()

    def step(self, action):
        masses = {key: self.value[key].clone() for key in MASS_FIELDS}
        result = super().step(action)
        result.update(masses)
        return result


class TorsoMassContextTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(229)
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        self.learner = matching_learner()

    def test_raw_kg_validation_and_fixed_scale_without_clipping(self):
        actual = torch.tensor([[5.], [8.], [11.], [14.]], requires_grad=True)
        nominal = torch.full_like(actual, 8.)
        value = normalize_torso_mass(actual, nominal)
        torch.testing.assert_close(value, torch.tensor([[-1.], [0.], [1.], [2.]]))
        torch.testing.assert_close(torch.autograd.grad(value.sum(), actual)[0], torch.full_like(actual, 1 / 3))
        with self.assertRaisesRegex(ValueError, "missing torso_mass"):
            mass_context_from({})
        for invalid in (torch.tensor([[0.]]), torch.tensor([[float("nan")]]), torch.tensor([[-1.]])):
            with self.assertRaisesRegex(ValueError, "finite positive"):
                normalize_torso_mass(invalid, torch.ones_like(invalid))
        with self.assertRaisesRegex(ValueError, "shapes"):
            normalize_torso_mass(torch.ones(2), torch.ones(2))

    def test_effective_solver_readback_tracks_each_world_and_rejects_stale_properties(self):
        from hoi.v1.dr_validation import _solver_readback
        names = ["world", "torso_link", "Object"]
        mass = torch.tensor([[0., 8.1, .6], [0., 7.8, 1.2]])
        com = torch.zeros(2, 3, 3)
        inertia = mass[..., None].expand(-1, -1, 3).clone()
        quat = torch.zeros(2, 3, 4)
        quat[..., 0] = 1
        mu = torch.full((2, 2, 3), .6)
        mu[1, :, 0] = .7
        mj = SimpleNamespace(nbody=3, ngeom=2, geom_bodyid=[2, 2])
        model = SimpleNamespace(body_mass=mass, body_ipos=com, body_inertia=inertia, body_iquat=quat, geom_friction=mu)
        host = SimpleNamespace(N=2, env=SimpleNamespace(torso_id=1,
               sim=SimpleNamespace(backend_manager=SimpleNamespace(_solver=SimpleNamespace(mj_model=mj, mjw_model=model)))))
        props = {"robot_mass": mass[:, 1:2].expand(-1, 3).clone(), "robot_com": com.clone(),
                 "robot_inertia": torch.diag_embed(inertia), "object_mass": mass[:, 2].clone(),
                 "object_com": com[:, 2].clone(), "object_inertia": torch.diag_embed(inertia[:, 2]),
                 "object_friction": mu[:, :, 0].clone()}
        fake = SimpleNamespace(mjtObj=SimpleNamespace(mjOBJ_BODY=1), mj_id2name=lambda m, kind, index: names[index])
        with patch.dict(sys.modules, {"mujoco": fake, "warp": SimpleNamespace(to_torch=lambda value: value)}):
            self.assertTrue(all(value == 0 for value in _solver_readback(host, props).values()))
            mass[1, 2] += .3
            with self.assertRaisesRegex(RuntimeError, "effective_solver_object_mass"):
                _solver_readback(host, props)

    def test_models_condition_predictions_and_value_on_mass_and_keep_action_jacobian(self):
        learner = self.learner
        tr = transition(learner, [8, 80], 0)
        S, O = unflatten_state(tr["s"]), unflatten_object(tr["o"])
        action = tr["a"].clone().requires_grad_()
        points = learner._points(tr["asset_id"])
        args = (S, O, action, tr["xi"], tr["xi_o"], points)
        zero, one = torch.zeros(2, 1), torch.ones(2, 1)
        self.assertEqual((learner.wm.in_dim, learner.critic.net[0].in_features), (225, 1215))
        x0 = learner.wm.inputs(*args, mass_context=zero)
        x1 = learner.wm.inputs(*args, mass_context=one)
        torch.testing.assert_close(x0[:, :-1], x1[:, :-1], atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "mass context"):
            learner.wm.step(*args)
        s0, o0, _ = learner.wm.step(*args, mass_context=zero)
        s1, o1, _ = learner.wm.step(*args, mass_context=one)
        self.assertGreater(float((s1[4] - s0[4]).detach().abs().sum()), 0.)
        self.assertGreater(float((o1[0] - o0[0]).detach().abs().sum()), 0.)
        cargs = learner._critic_args(S, O, tr["f"], tr["xi"], tr["xi_o"], tr["asset_id"], zero)
        v0, v1 = learner.critic(*cargs), learner.critic(*cargs[:-1], one)
        self.assertGreater(float((v1 - v0).detach().abs().sum()), 0.)
        grad = torch.autograd.grad(s1[4].sum() + o1[0].sum(), action)[0]
        self.assertTrue(bool(torch.isfinite(grad).all()))
        self.assertGreater(float(grad.abs().sum()), 0.)
        self.assertEqual((WorldModel(hidden=(8,)).in_dim, Critic(hidden=(8,)).net[0].in_features), (224, 1214))

    def test_replay_preserves_raw_masses_across_ring_wrap_and_rejects_missing_fields(self):
        learner = self.learner
        tr = transition(learner, [8, 80], 0)
        bad = deepcopy(tr)
        del bad["torso_mass"]
        with self.assertRaisesRegex(ValueError, "missing torso_mass"):
            learner.observe(bad)
        self.assertEqual(learner.buffer.size, 0)
        fill(learner, steps=70)
        window = learner.buffer.sample_windows(10, 8, split="all")
        start, history = learner.buffer.sample_starts(10, 10, horizon=3, motion=learner.motion)
        for values in (window, start, history):
            expected = .1 + .1 * (values["f"] >= 70).float()
            torch.testing.assert_close(mass_context_from(values).squeeze(-1), expected)
            torch.testing.assert_close(values["torso_mass_nominal"], torch.full_like(values["torso_mass_nominal"], 8.))

    def test_wm_fit_and_one_step_and_autoregressive_validation_receive_mass(self):
        learner = self.learner
        fill(learner, steps=40)
        seen = []
        original = learner.wm.step
        def record(*args, **kwargs):
            context = kwargs["mass_context"]
            self.assertEqual(context.shape, (len(args[2]), 1))
            self.assertTrue(bool(torch.isfinite(context).all()))
            seen.append(context.clone())
            return original(*args, **kwargs)
        with patch.object(learner.wm, "step", side_effect=record):
            loss = learner.wm_update()
            self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in loss.values()))
            before_validation = len(seen)
            report = learner.wm_validation(batch_size=2)
        self.assertTrue(report["finite"])
        self.assertEqual(len(seen) - before_validation, 64)
        for value in seen[before_validation:]:
            torch.testing.assert_close(value, torch.full_like(value, .1))

    def test_imagination_bootstrap_and_value_minibatches_keep_start_mass(self):
        learner = self.learner
        fill(learner)
        start, history = learner.buffer.sample_starts(2, 10, horizon=3, motion=learner.motion)
        context = mass_context_from(start)
        with patch.object(learner.wm, "step", wraps=learner.wm.step) as wm, patch.object(
                learner.critic, "forward", wraps=learner.critic.forward) as critic:
            trajectory = learner.imagine(start, history)
            self.assertEqual(wm.call_count, 3)
            for call in wm.call_args_list:
                torch.testing.assert_close(call.kwargs["mass_context"], context, atol=0, rtol=0)
            torch.testing.assert_close(critic.call_args.args[-1], context, atol=0, rtol=0)
            with patch.object(learner.critic, "update_normalizers", wraps=learner.critic.update_normalizers) as norms:
                value_loss = learner.critic_update(trajectory, start)
                self.assertEqual(len(norms.call_args.args), 6)
            for call in critic.call_args_list[1:]:
                self.assertEqual(call.args[-1].shape[-1], 1)
                for value in call.args[-1]:
                    self.assertTrue(any(torch.equal(value, row) for row in context))
        self.assertTrue(torch.isfinite(torch.tensor(value_loss)))
        del start["torso_mass"]
        with self.assertRaisesRegex(ValueError, "missing torso_mass"):
            learner.imagine(start, history)

    def test_actor_action_is_identical_when_only_mass_metadata_changes(self):
        learner = self.learner
        tr = transition(learner, [8, 80], 0)
        with patch.object(learner.actor_obs, "add_noise", side_effect=lambda value: value):
            first = learner.action(tr, exploration=False, update_statistics=False)
            tr["torso_mass"] += 2.
            tr["torso_mass_nominal"] -= 1.
            second = learner.action(tr, exploration=False, update_statistics=False)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        self.assertFalse(hasattr(learner.pi, "torso_mass_context"))

    def test_checkpoint_binds_mass_contract_and_rejects_contextless_or_wrong_model(self):
        learner = self.learner
        kwargs = dict(iteration=3, physics={"hand_collision": True, "self_collision": True},
                      data_identity={"synthetic": True}, initialization={"mode": "CPU_fixture"},
                      extra={"config": learner.config.to_dict()})
        cp = make_checkpoint(learner.pi, learner.wm, learner.critic, learner.optimizers, **kwargs)
        self.assertTrue(checkpoint_torso_mass_context(cp))
        self.assertEqual(cp["architecture"]["wm"]["robot_mass_context"], torso_mass_contract())
        self.assertNotIn("robot_mass_context", cp["architecture"]["pi"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "context.pt"
            torch.save(cp, path)
            restored = load_checkpoint(path, learner.pi, learner.wm, learner.critic, learner.optimizers,
                                      physics=kwargs["physics"], data_identity=kwargs["data_identity"])
            self.assertEqual(restored["feature_schema"], cp["feature_schema"])
        before = deepcopy(learner.pi.state_dict())
        old = deepcopy(cp)
        old["feature_schema"] = feature_schema(learner.pi.observation_profile)
        with self.assertRaisesRegex(ValueError, "torso mass context"):
            validate_checkpoint_features(old)
        with self.assertRaisesRegex(ValueError, "torso mass context"):
            load_checkpoint(cp, learner.pi, WorldModel(hidden=(32, 32)), learner.critic,
                            physics=kwargs["physics"], data_identity=kwargs["data_identity"])
        for key, value in learner.pi.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_native_snapshot_and_terminal_transition_use_effective_pre_reset_mass(self):
        learner = self.learner
        tr = transition(learner, [8, 80], 0)
        host = HOITrackingSim.__new__(HOITrackingSim)
        mass = torch.tensor([[2., 8.3, 1.], [2., 8.6, 1.]])
        e = SimpleNamespace(torso_id=1, robot=SimpleNamespace(data=SimpleNamespace(body_mass=SimpleNamespace(torch=mass))),
            f=tr["f"].clone(), a_prev=tr["a_prev"].clone(), ep_step=tr["ep_step"].clone(), xi=tr["xi"],
            max_episode_steps=1, gamma=.99, eval_mode=False, ep_disc_ret=torch.zeros(2), ep_ret=torch.zeros(2),
            start_bin=torch.arange(2), bin_G=torch.zeros(2))
        host.env, host.N, host.motion, host.mdp = e, 2, learner.motion, learner.mdp
        host.torso_mass_context, host.torso_mass_nominal = True, tr["torso_mass_nominal"].clone()
        host.xi_o, host.asset_id = tr["xi_o"], tr["asset_id"]
        host.contact_missing_counter = torch.zeros(2, 2, dtype=torch.int32)
        host.state = lambda: (learner.motion.ref_state(e.f), learner.motion.ref_object(e.f))
        host._advance = lambda action: {"contact": tr["contact"], "contact_valid": tr["contact_valid"], "impulse": tr["impulse"]}
        def reset(ids):
            mass[ids, 1] += 1.
        host.reset = reset
        snapshot = _snapshot(host)
        result = host.step(tr["a"])
        for key in MASS_FIELDS:
            torch.testing.assert_close(result[key], snapshot[key], atol=0, rtol=0)
        self.assertTrue(bool(result["done"].all()))
        self.assertFalse(torch.equal(_snapshot(host)["torso_mass"], result["torso_mass"]))
        mass[:, 1] += .25
        torch.testing.assert_close(host.robot_mass_fields()["torso_mass"], mass[:, 1:2])

    def test_frozen_evaluation_records_nominal_mass_and_pairs_wm_diagnostics(self):
        learner = self.learner
        _, motion, _, shapes, fk, metadata = fixture()
        kin = ChainKinematics()
        motion.body_p, motion.body_R, motion.body_v, motion.body_w = kin.forward(*motion.ref_arrays())
        from hoi.v1.mdp import HOIMDP
        mdp = HOIMDP(kin, motion, torch.zeros(29), shapes)
        workers = NominalMassWorkers(motion, mdp)
        before = [deepcopy(module.state_dict()) for module in (learner.pi, learner.wm, learner.critic)]
        with tempfile.TemporaryDirectory() as temp, patch.object(learner.wm, "step", wraps=learner.wm.step) as step:
            cases = evaluate_group(learner.pi, motion, mdp, shapes, workers, [0, 1], fk39=fk,
                joint_names=fk.dof_names, output=Path(temp) / "eval", metadata=metadata, verify_tracked_fk=False, wm=learner.wm)
            self.assertGreater(step.call_count, 0)
            for call in step.call_args_list:
                torch.testing.assert_close(call.kwargs["mass_context"], torch.zeros_like(call.kwargs["mass_context"]))
            for index, length in enumerate((3, 6)):
                raw = torch.load(cases[index]["rollout"], weights_only=True)
                self.assertEqual(raw["control"]["torso_mass"].shape, (length, 1))
                self.assertEqual(raw["metadata"]["robot_mass_context"], torso_mass_contract())
                torch.testing.assert_close(raw["control"]["torso_mass_context"], torch.zeros(length, 1))
            with self.assertRaisesRegex(ValueError, "missing torso_mass"):
                evaluate_group(learner.pi, motion, mdp, shapes, SyntheticWorkers(motion, mdp), [0, 1], fk39=fk,
                    joint_names=fk.dof_names, output=Path(temp) / "bad", verify_tracked_fk=False, wm=learner.wm)
            self.assertFalse((Path(temp) / "bad").exists())
        for module, snapshot in zip((learner.pi, learner.wm, learner.critic), before):
            for key, value in module.state_dict().items():
                torch.testing.assert_close(value, snapshot[key], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
