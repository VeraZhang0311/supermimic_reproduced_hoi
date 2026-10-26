"""Shared learner: real transitions -> joint WM -> differentiable policy + value.

This module has no simulator imports. Native workers own physical state; the
learner owns the single policy/WM/critic and never treats a reference as a real
transition. WM and critic parameters are frozen, not detached, in actor BPTT.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from supermimic.mdp import normalize_xi
from .config import TrainConfig
from .contacts import CHANNELS
from .networks import frozen_parameters
from .state import object_features, unflatten_object, unflatten_state
from .observations import ActorObservations
from .robot_context import uses_torso_mass, mass_context_from


def _stack(states):
    return tuple(torch.stack([state[index] for state in states]) for index in range(len(states[0])))


def _flatten(state):
    return tuple(value.flatten(0, 1) for value in state)


def _detach(state):
    return tuple(value.detach() for value in state)


def _finite_gradients(module):
    return all(bool(torch.isfinite(parameter.grad).all())
               for parameter in module.parameters() if parameter.grad is not None)


class Learner:
    def __init__(self, policy, wm, critic, motion, mdp, shapes, replay, config=None):
        self.config = (config if config is not None else TrainConfig()).validate()
        self.pi, self.wm, self.critic = policy, wm, critic
        self.motion, self.mdp, self.shapes, self.buffer = motion, mdp, shapes, replay
        self.torso_mass_context = uses_torso_mass(self.config.domain_randomization_profile)
        if any(getattr(value, "torso_mass_context", False) != self.torso_mass_context
               for value in (wm, critic, replay)):
            raise ValueError("WM, critic and replay torso mass context must match the DR profile")
        self.device = next(policy.parameters()).device
        self.optimizers = {
            "pi": torch.optim.Adam(policy.parameters(), lr=self.config.lr, betas=(.9, .95)),
            "wm": torch.optim.Adam(wm.parameters(), lr=self.config.lr),
            "critic": torch.optim.Adam(critic.parameters(), lr=self.config.lr),
        }
        if policy.observation_profile != self.config.observation_profile:
            raise ValueError("actor observation profile differs from learner configuration")
        self.actor_obs = ActorObservations(mdp, self.config.observation_profile)
        self.robot_history = torch.zeros((replay.N, 10, self.actor_obs.spec.robot_dim), device=self.device)
        self.object_history = torch.zeros((replay.N, 10, self.actor_obs.spec.object_dim), device=self.device)
        self.iteration = self.sim_steps = 0
        self.contact_positive = torch.zeros(len(CHANNELS), device=self.device, dtype=torch.float64)
        self.contact_negative = torch.zeros(len(CHANNELS), device=self.device, dtype=torch.float64)

    def set_lr(self, iteration):
        rate = self.config.lr_at(iteration)
        for optimizer in self.optimizers.values():
            for group in optimizer.param_groups:
                group["lr"] = rate
        return rate

    def _points(self, assets):
        return self.shapes.points(assets)

    def _mass_kwargs(self, values, index=None):
        if not getattr(self, "torso_mass_context", False):
            return {}
        context = mass_context_from(values)
        return {"mass_context": context if index is None else context[index]}

    def _check_assets(self, frames, assets):
        if not torch.equal(self.motion.asset_ids[frames], assets):
            raise ValueError("collector asset geometry does not match the reference frame")

    @torch.no_grad()
    def action(self, snapshot, *, exploration=True, update_statistics=True):
        S, O = unflatten_state(snapshot["s"]), unflatten_object(snapshot["o"])
        f, xi, assets = snapshot["f"], snapshot["xi"], snapshot["asset_id"]
        self._check_assets(f, assets)
        robot = self.actor_obs.add_noise(self.actor_obs.robot(S, snapshot["a_prev"], xi, f))
        obj = self.actor_obs.object(S, O, f)
        fresh = snapshot["ep_step"] == 0
        self.robot_history = torch.where(fresh[:, None, None], robot[:, None].expand_as(self.robot_history),
                                        torch.cat((self.robot_history[:, 1:], robot[:, None]), 1))
        self.object_history = torch.where(fresh[:, None, None], obj[:, None].expand_as(self.object_history),
                                         torch.cat((self.object_history[:, 1:], obj[:, None]), 1))
        human_goal = self.actor_obs.human_goal(S, f)
        object_goal = self.actor_obs.object_goal(S, O, f)
        if update_statistics:
            # Match the upstream current-observation count, not ten repeated
            # histories per tick; each new normalizer has its own count.
            self.pi.obs_norm.update(robot)
            self.pi.goal_norm.update(human_goal)
            self.pi.object_obs_norm.update(obj)
            self.pi.object_goal_norm.update(object_goal)
        a = self.pi(self.robot_history, human_goal, self.object_history, object_goal, self._points(assets))
        if exploration:
            a = (a + self.config.action_noise * torch.randn_like(a)).clamp(-20, 20)
        if not bool(torch.isfinite(a).all()):
            raise FloatingPointError("collector produced nonfinite actions")
        return a

    @torch.no_grad()
    def observe(self, transition):
        self._check_assets(transition["f"], transition["asset_id"])
        self.buffer.add(transition)
        fit = ((transition["f"] // 50).remainder(20) != 0)[:, None]
        valid = transition["contact_valid"] & fit
        positive = (transition["contact"] > .5) & valid
        self.contact_positive += positive.sum(0)
        self.contact_negative += (valid & ~positive).sum(0)
        fit_rows = fit.squeeze(-1)
        if bool(fit_rows.any()):
            S, O = unflatten_state(transition["s"][fit_rows]), unflatten_object(transition["o"][fit_rows])
            S1, O1 = unflatten_state(transition["s1"][fit_rows]), unflatten_object(transition["o1"][fit_rows])
            self.wm.update_normalizers(S, O, transition["a"][fit_rows], normalize_xi(transition["xi"][fit_rows]),
                                       transition["xi_o"][fit_rows], self._points(transition["asset_id"][fit_rows]), S1, O1)
        self.sim_steps += self.buffer.N

    def wm_rollout(self, window):
        S, O = unflatten_state(window["s"][:, 0]), unflatten_object(window["o"][:, 0])
        states, objects, logits = [], [], []
        for step in range(window["a"].shape[1]):
            active = window["mask"][:, step].bool().nonzero().flatten()
            contact = window["a"].new_zeros(len(window["a"]), len(CHANNELS))
            if len(active):
                next_s, next_o, active_contact = self.wm.step(
                    tuple(value[active] for value in S), tuple(value[active] for value in O),
                    window["a"][active, step], normalize_xi(window["xi"][active, step]),
                    window["xi_o"][active, step], self._points(window["asset_id"][active, step]),
                    **self._mass_kwargs(window, (active, step)))
                S = tuple(previous.index_copy(0, active, value) for previous, value in zip(S, next_s))
                O = tuple(previous.index_copy(0, active, value) for previous, value in zip(O, next_o))
                contact = contact.index_copy(0, active, active_contact)
            states.append(S)
            objects.append(O)
            logits.append(contact)
        # Network loss windows have batch first, unlike imagination time lists.
        transpose = lambda values: tuple(value.transpose(0, 1) for value in _stack(values))
        return transpose(states), transpose(objects), torch.stack(logits, 1)

    def wm_update(self):
        window = self.buffer.sample_windows(self.config.wm_batch, self.config.wm_window, split="fit")
        predicted_s, predicted_o, logits = self.wm_rollout(window)
        ratio = self.contact_negative / self.contact_positive.clamp_min(1)
        weights = torch.where(self.contact_positive > 0, ratio, torch.ones_like(ratio)).clamp(1e-3, 100).float()
        losses = self.wm.loss(predicted_s, predicted_o, unflatten_state(window["s1"]), unflatten_object(window["o1"]),
                              logits, window["contact"], window["contact_valid"], window["mask"], pos_weight=weights)
        optimizer = self.optimizers["wm"]
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        norm = nn.utils.clip_grad_norm_(self.wm.parameters(), 1.)
        if not bool(torch.isfinite(norm)) or not _finite_gradients(self.wm):
            raise FloatingPointError("nonfinite joint world-model gradient")
        optimizer.step()
        return {key: float(value.detach()) / self.config.wm_window for key, value in losses.items()}

    def _critic_args(self, S, O, f, xi_n, xi_o, assets, mass_context=None):
        expected, valid = self.motion.expected(f)
        legacy = torch.cat((self.mdp.legacy_critic_features(S, f, xi_n), self.motion.goal(f, 10)), -1)
        arguments = (legacy, self.motion.object_goal(S, O, f, 10), object_features(S, O),
                self.mdp.object_errors(S, O, f), xi_o, torch.cat((expected, valid), -1), self._points(assets))
        if self.torso_mass_context:
            if mass_context is None:
                raise ValueError("critic is missing torso mass context")
            arguments += (mass_context,)
        return arguments

    def imagine(self, start, history, *, fixed_robot_noise=None):
        c = self.config
        S, O = unflatten_state(start["s"]), unflatten_object(start["o"])
        xi, xi_o, f, previous = start["xi"], start["xi_o"], start["f"], start["a_prev"]
        assets, xi_n = start["asset_id"], normalize_xi(xi)
        mass_kwargs = self._mass_kwargs(start)
        with torch.no_grad():
            robot_hist = self.actor_obs.robot(unflatten_state(history["s"]), history["a_prev"], history["xi"], history["f"])
            obj_hist = self.actor_obs.object(unflatten_state(history["s"]), unflatten_object(history["o"]), history["f"])
        robot_hist = self.actor_obs.add_noise(robot_hist) if fixed_robot_noise is None else robot_hist + fixed_robot_noise[0]
        states, objects, frames = [S], [O], [f]
        actions, rewards, failures, contact_probabilities = [], [], [], []
        points = self._points(assets)
        # Geometry is constant within this imagination. Reuse the same graph,
        # not a detached embedding or a cache across parameter updates.
        policy_shape = self.pi.shape_encoder(points)
        for step in range(c.horizon):
            human_goal = self.actor_obs.human_goal(S, f)
            object_goal = self.actor_obs.object_goal(S, O, f)
            action = self.pi(robot_hist, human_goal, obj_hist, object_goal, points, shape_embedding=policy_shape)
            S, O, logits = self.wm.step(S, O, action, xi_n, xi_o, points, **mass_kwargs)
            f = f + 1
            probabilities = torch.sigmoid(logits)
            rewards.append(self.mdp.reward(S, O, f, action, previous, probabilities))
            failures.append(self.mdp.soft_termination(S, O, f))
            observation = self.actor_obs.robot(S, action, xi, f)
            observation = self.actor_obs.add_noise(observation) if fixed_robot_noise is None else observation + fixed_robot_noise[step + 1]
            robot_hist = torch.cat((robot_hist[:, 1:], observation[:, None]), 1)
            obj_hist = torch.cat((obj_hist[:, 1:], self.actor_obs.object(S, O, f)[:, None]), 1)
            states.append(S)
            objects.append(O)
            frames.append(f)
            actions.append(action)
            contact_probabilities.append(probabilities)
            previous = action
        reward, failure = torch.stack(rewards), torch.stack(failures)
        alive = torch.cumprod(1 - failure, 0)
        survival = torch.cat((torch.ones_like(alive[:1]), alive[:-1]), 0)
        discounts = c.gamma ** torch.arange(c.horizon, device=self.device, dtype=reward.dtype)
        returns = (discounts[:, None] * survival * reward).sum(0)
        if not c.no_critic:
            returns = returns + c.gamma ** c.horizon * alive[-1] * self.critic(
                *self._critic_args(S, O, f, xi_n, xi_o, assets, **mass_kwargs))
        return {"objective": returns.mean(), "reward": reward, "failure": failure, "survival": survival,
                "states": states, "objects": objects, "frames": frames, "actions": actions,
                "contact": contact_probabilities, "terminal_survival": alive[-1]}

    def critic_update(self, trajectory, start):
        if self.config.no_critic:
            return 0.
        c = self.config
        H, B = c.horizon, len(start["f"])
        with torch.no_grad():
            S, O = _flatten(_stack([_detach(state) for state in trajectory["states"]])), _flatten(_stack([_detach(state) for state in trajectory["objects"]]))
            f = torch.stack(trajectory["frames"]).flatten()
            expand = lambda value: value[None].expand((H + 1,) + value.shape).flatten(0, 1)
            mass_kwargs = {key: expand(value) for key, value in self._mass_kwargs(start).items()}
            arguments = self._critic_args(S, O, f, expand(normalize_xi(start["xi"])), expand(start["xi_o"]),
                                          expand(start["asset_id"]), **mass_kwargs)
        # Cache raw inputs/points only. PointNet remains inside each minibatch
        # forward so it is trained by the value loss.
        self.critic.update_normalizers(*arguments[:6])
        with torch.no_grad():
            # Bound PointNet's temporary activations by the existing largest
            # training minibatch; include every bootstrap row in time order.
            target_batch = (H * B + c.critic_minibatches - 1) // c.critic_minibatches
            values = torch.cat([self.critic(*(feature[offset:offset + target_batch] for feature in arguments))
                                for offset in range(0, (H + 1) * B, target_batch)]).reshape(H + 1, B)
            target, targets = values[-1], []
            for step in range(H - 1, -1, -1):
                target = trajectory["reward"][step].detach() + c.gamma * (1 - trajectory["failure"][step].detach()) * (
                    (1 - c.lam) * values[step + 1] + c.lam * target)
                targets.append(target)
            targets = torch.stack(targets[::-1]).flatten()
        inputs = tuple(value[:H * B].detach() for value in arguments)
        size, total_loss, count = H * B, 0., 0
        for _ in range(c.critic_epochs):
            for ids in torch.randperm(size, device=self.device).tensor_split(c.critic_minibatches):
                if not len(ids):
                    continue
                value = self.critic(*(feature[ids] for feature in inputs))
                loss = (value - targets[ids]).square().mean()
                optimizer = self.optimizers["critic"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = nn.utils.clip_grad_norm_(self.critic.parameters(), 1.)
                if not bool(torch.isfinite(grad)):
                    raise FloatingPointError("nonfinite critic gradient")
                optimizer.step()
                total_loss += float(loss.detach())
                count += 1
        return total_loss / max(count, 1)

    def policy_update(self):
        start, history = self.buffer.sample_starts(self.config.pi_batch, 10, horizon=self.config.horizon, motion=self.motion)
        optimizer = self.optimizers["pi"]
        self.optimizers["wm"].zero_grad(set_to_none=True)
        self.optimizers["critic"].zero_grad(set_to_none=True)
        with frozen_parameters(self.wm, self.critic):
            trajectory = self.imagine(start, history)
            loss = -trajectory["objective"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.)
            nn.utils.clip_grad_value_(self.pi.parameters(), .1)
            if not bool(torch.isfinite(norm)) or not _finite_gradients(self.pi):
                raise FloatingPointError("nonfinite policy BPTT gradient")
            optimizer.step()
        value_loss = self.critic_update(trajectory, start)
        return {"pi_loss": float(loss.detach()), "v_loss": value_loss, "pi_gradnorm": float(norm),
                "imag_human_and_hoi_return_per_step": float(trajectory["reward"].detach().mean()),
                "imag_survival": float(trajectory["terminal_survival"].detach().mean())}

    @torch.no_grad()
    def wm_validation(self, batch_size=None):
        """Real heldout 1/8/32-step errors, phases, simple baselines and pairs.

        One-step predictions restart from each real state in the 32-step window.
        Eight/32-step predictions are autoregressive under executed actions.
        Terminal transitions remain valid; post-reset transitions are masked.
        """
        from supermimic.rotations import so3_exp
        from .state import object_state_error

        batch = self.config.wm_batch if batch_size is None else batch_size
        length = max(32, self.config.wm_window)
        window = self.buffer.sample_windows(batch, length, split="holdout")
        S, O, logits = self.wm_rollout(window)
        actual_s, actual_o = unflatten_state(window["s1"]), unflatten_object(window["o1"])
        n = self.config.wm_window
        prefix = lambda values: tuple(value[:, :n] for value in values)
        loss = self.wm.loss(prefix(S), prefix(O), prefix(actual_s), prefix(actual_o), logits[:, :n],
                            window["contact"][:, :n], window["contact_valid"][:, :n], window["mask"][:, :n])
        mask = window["mask"][:, :n]
        valid_state = window["mask"].bool()[..., None]
        position_error = torch.where(valid_state, O[0] - actual_o[0], 0.).square().sum(-1)
        initial = unflatten_object(window["o"][:, 0])
        ticks = torch.arange(1, length + 1, device=self.device)[None, :, None] * .02
        constant_velocity = initial[0][:, None] + ticks * initial[2][:, None]
        baseline_error = torch.where(valid_state, constant_velocity - actual_o[0], 0.).square().sum(-1)
        zero_error = torch.where(valid_state, initial[0][:, None] - actual_o[0], 0.).square().sum(-1)
        average = lambda x: float((x[:, :n] * mask).sum() / mask.sum().clamp_min(1))
        # Teacher-forced one-step contact/physics diagnostics use actual inputs,
        # rather than confusing contact errors with accumulated 32-step drift.
        one_objects, one_logits = [], []
        for step in range(length):
            active = window["mask"][:, step].bool()
            one_o = tuple(value[:, step].clone() for value in actual_o)
            one_c = window["a"].new_zeros((len(active), len(CHANNELS)))
            if bool(active.any()):
                _, selected_o, selected_c = self.wm.step(unflatten_state(window["s"][active, step]), unflatten_object(window["o"][active, step]),
                                                        window["a"][active, step], normalize_xi(window["xi"][active, step]),
                                                        window["xi_o"][active, step], self._points(window["asset_id"][active, step]),
                                                        **self._mass_kwargs(window, (active, step)))
                for target, source in zip(one_o, selected_o):
                    target[active] = source
                one_c[active] = selected_c
            one_objects.append(one_o)
            one_logits.append(one_c)
        one_o = tuple(value.transpose(0, 1) for value in _stack(one_objects))
        one_logits = torch.stack(one_logits, 1)
        valid = window["contact_valid"] & window["mask"][..., None].bool()
        positive = (window["contact"] > .5) & valid
        negative = (window["contact"] <= .5) & valid
        probabilities = torch.sigmoid(one_logits)
        prediction = probabilities >= .5
        recall = ((prediction & positive).sum((0, 1)) / positive.sum((0, 1)).clamp_min(1)).tolist()
        precision = ((prediction & positive).sum((0, 1)) / (prediction & valid).sum((0, 1)).clamp_min(1)).tolist()
        calibration = (probabilities - window["contact"]).square()
        counts = valid.sum((0, 1)).clamp_min(1)
        prevalence = positive.sum((0, 1)) / counts
        hand_valid = valid[..., :2].all(-1)
        in_contact = positive[..., :2].any(-1)
        connected = torch.zeros_like(in_contact)
        connected[:, 1:] = hand_valid[:, 1:] & hand_valid[:, :-1]
        onset, release = torch.zeros_like(in_contact), torch.zeros_like(in_contact)
        onset[:, 1:] = in_contact[:, 1:] & ~in_contact[:, :-1] & connected[:, 1:]
        release[:, 1:] = ~in_contact[:, 1:] & in_contact[:, :-1] & connected[:, 1:]
        phases = {"no_contact": ~in_contact & hand_valid, "contact": in_contact & hand_valid,
                  "onset": onset, "release": release}
        scales = torch.maximum(self.wm.object_out_norm.std(), self.wm.object_error_floor)
        def error(predicted, truth, robot, selection):
            result = truth[0].new_zeros(selection.shape + (4,))
            if bool(selection.any()):
                selected = lambda state: tuple(value[selection] for value in state)
                raw = object_state_error(selected(robot), selected(predicted), selected(truth))
                result[selection] = (raw / scales).square().reshape(-1, 4, 3).mean(-1)
            return result
        def baseline(origin, dt):
            p, r, v, w = origin
            return p + dt * v, so3_exp(dt * w) @ r, v, w
        def summarize(predicted, zero, cv, truth, robot, selection, phase_masks):
            errors = [error(value, truth, robot, selection) for value in (predicted, zero, cv)]
            def group(selected):
                count = int(selected.sum())
                mean = lambda value: float((value.mean(-1) * selected).sum() / selected.sum().clamp_min(1))
                return {"valid_samples": count, "object_normalized_error": mean(errors[0]),
                        "zero_increment_normalized_error": mean(errors[1]), "constant_velocity_normalized_error": mean(errors[2]),
                        "object_component_normalized_error": ((errors[0] * selected[..., None]).sum(tuple(range(selected.ndim))) /
                                                               selected.sum().clamp_min(1)).tolist()}
            return {**group(selection), "phases": {name: group(selection & value) for name, value in phase_masks.items()}}
        real_o = unflatten_object(window["o"])
        horizon_reports = {"1": summarize(one_o, real_o, baseline(real_o, .02), actual_o, actual_s,
                                            window["mask"].bool(), phases)}
        horizon_reports["1"]["mode"] = "teacher_forced_single_steps_in_reserved_real_windows"
        for horizon in (8, 32):
            last = lambda values: tuple(value[:, horizon - 1] for value in values)
            horizon_reports[str(horizon)] = summarize(last(O), initial, baseline(initial, horizon * .02), last(actual_o), last(actual_s),
                                                       window["mask"][:, horizon - 1].bool(),
                                                       {name: value[:, horizon - 1] for name, value in phases.items()})
            horizon_reports[str(horizon)]["mode"] = "autoregressive_under_real_executed_actions"
        return {"holdout_object_position_mse": average(position_error),
                "constant_velocity_position_mse": average(baseline_error), "zero_increment_position_mse": average(zero_error),
                "loss": {key: float(value) / self.config.wm_window for key, value in loss.items()},
                "contact_positive_counts": positive.sum((0, 1)).tolist(), "contact_negative_counts": negative.sum((0, 1)).tolist(),
                "contact_precision": precision, "contact_recall": recall,
                "contact_brier": float((calibration * valid).sum() / valid.sum().clamp_min(1)),
                "horizon_reports": horizon_reports, "real_executed_actions": True,
                "contact_diagnostics": {"channels": list(CHANNELS),
                                        "positive_counts": positive.sum((0, 1)).tolist(), "negative_counts": negative.sum((0, 1)).tolist(),
                                        "precision": precision, "recall": recall,
                                        "brier_per_channel": ((calibration * valid).sum((0, 1)) / counts).tolist(),
                                        "prevalence_baseline_brier": (prevalence * (1 - prevalence)).tolist(),
                                        "probability_source": "teacher_forced_one_step_actual_state_inputs",
                                        "phase_definition": "actual_left_or_right_hand_object; onset/release require both adjacent intervals valid"},
                "finite": all(math.isfinite(float(value)) for value in loss.values()) and
                          all(bool(torch.isfinite(value).all()) for value in (position_error, baseline_error, zero_error, one_logits)),
                "scope": "training_reference_bins_reserved_from_wm_fit_not_eval80_or_test40"}
