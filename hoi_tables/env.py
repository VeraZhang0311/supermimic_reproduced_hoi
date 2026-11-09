"""Table scene on the unchanged privileged GMT physics and motor runtime."""
from __future__ import annotations

import torch
from pathlib import Path

from hoi_residual.env import HOISim
from hoi_residual.features import ObjectHistory, object_observation, torso_pose
from .native_checks import audit_object_solver
from supermimic.mdp import flatten_state
from supermimic.rotations import mat_to_quat, rot_angle_sq
from .contacts import PairContacts
from .sampling import AdaptiveRSI


class TableSim(HOISim):
    def __init__(self, sim_cfg, num_envs, motion, mdp, asset, output, *, teacher_kin,
                 reward, eval_mode=False, max_episode_steps=500, solver_ls_iterations=50):
        self.iteration, self.teacher_kin, self.reward_model = 0, teacher_kin, reward
        self._solver_failure_output = Path(output)
        self.sampler = None if eval_mode else AdaptiveRSI(motion, asset_index=int(asset["asset_id"]))
        self._history_ready = False
        super().__init__(sim_cfg, num_envs, motion, mdp, asset, output,
                         eval_mode=eval_mode, max_episode_steps=max_episode_steps)
        self.contacts = PairContacts(self)
        from .solver_health import configure
        self.solver_health = configure(self,solver_ls_iterations)
        self.history = ObjectHistory(object_observation(torso_pose(teacher_kin, self.state()), self.object_state()))
        self.q_prev, self.qd_prev = (x.clone() for x in self.state()[4:])
        self.episode_id = torch.arange(self.N, device=self.device)
        self.quality_successes = torch.zeros(self.N, device=self.device)
        self._nominal_torso_mass = self.robot.data.body_mass.torch[:, self.torso_id].clone()
        self._history_ready = True
        self.G_max = .15 / (1 - self.gamma)

    def sample_frames(self, n):
        if self.sampler is not None:
            return self.sampler.sample_frames(n, iteration=self.iteration)
        return super().sample_frames(n)

    def reset(self, env_ids, frames=None):
        super().reset(env_ids, frames)
        if self._history_ready and len(env_ids):
            S, O = self.state(), self.object_state()
            self.history.reset(env_ids, object_observation(torso_pose(self.teacher_kin, S), O))
            self.q_prev[env_ids], self.qd_prev[env_ids] = S[4][env_ids], S[5][env_ids]
            self.episode_id[env_ids] += self.N
            self.quality_successes[env_ids] = 0

    def snapshot(self):
        S, O = self.state(), self.object_state()
        return {"s": flatten_state(S), "o": torch.cat((O[0], mat_to_quat(O[1]), O[2], O[3]), -1),
                "xi": self.xi.clone(), "xi_o": self.object_context(),
                "mass_context": ((self.robot.data.body_mass.torch[:, self.torso_id] - self._nominal_torso_mass) / 3.)[:, None],
                "actuator": self.actuator_state(), "f": self.f.clone(), "a_prev": self.a_prev.clone(),
                "q_prev": self.q_prev.clone(), "qd_prev": self.qd_prev.clone(),
                "object_history": self.history.values.clone(), "episode_id": self.episode_id.clone(),
                "asset_id": self.asset_ids.clone(), "bin_id": self.start_bin.clone(),
                "ep_step": self.ep_step.clone(), "partition": "protected_eval" if self.eval_mode else "train"}

    def step(self, action):
        if action.shape != (self.N, 29) or not bool(torch.isfinite(action).all()):
            raise ValueError("Table teacher requires finite direct [N,29] actions")
        S0, previous, f0 = self.state(), self.a_prev.clone(), self.f.clone()
        target = self.default_q + self.xi[:, 1:30] + self.alpha * action
        try:
            contact, valid = self.contacts.step(target)
        except RuntimeError as error:
            if str(error) == 'native contact/constraint buffer overflow':
                from .solver_health import raise_failure
                raise_failure(self,error)
            raise
        if self._check_object_after_step:
            self.object_solver_report = audit_object_solver(self)
            self.object_solver_checks += 1
            self._check_object_after_step = False
        S1, O1, f1 = self.state(), self.object_state(), f0 + 1
        if any(not bool(torch.isfinite(x).all()) for x in (*S1, *O1)):
            raise FloatingPointError("Nonfinite native table state")
        terms = self.reward_model.terms(S1, O1, f1, action, previous, contact, valid)
        r = sum(terms.values())
        if not bool(torch.isfinite(r).all()):
            raise FloatingPointError("Nonfinite v1.1 table reward")
        terminated = self.reward_model.termination(S1, O1, f1)
        truncated = (f1 >= self.motion.clip_end[self.motion.clip_id[f0]]) | (self.ep_step + 1 >= self.max_episode_steps)
        op, oR, _, _ = self.motion.object_ref(f1)
        bp = self.mdp.kin.forward(*S1)[0]
        ref_bp = self.motion.reference_bodies(f1)[0]
        quality = ((bp-ref_bp).norm(dim=-1).amax(-1) < .5) & ((O1[0]-op).norm(dim=-1) < .1) & (rot_angle_sq(O1[1], oR) < (.5235987756 ** 2))
        self.quality_successes += quality.float()
        self.ep_disc_ret += self.gamma ** self.ep_step.float() * r
        self.ep_ret += r
        self.ep_step += 1
        self.q_prev, self.qd_prev = S0[4].clone(), S0[5].clone()
        self.f, self.a_prev = f1, action.clone()
        self.history.advance(object_observation(torso_pose(self.teacher_kin, S1), O1))
        done = terminated | truncated
        ids = done.nonzero().flatten()
        if len(ids) and self.sampler is not None:
            self.sampler.update(self.start_bin[ids], success=~terminated[ids],
                                quality=self.quality_successes[ids]/self.ep_step[ids],
                                failure_frames=torch.where(terminated[ids], f1[ids], -torch.ones_like(f1[ids])),
                                elapsed_steps=self.ep_step[ids])
        return {"reward": r, "terminated": terminated, "truncated": truncated,
                "done": done, "contact": contact, "contact_valid": valid.bool(), "terms": terms}
