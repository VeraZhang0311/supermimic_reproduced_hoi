"""Real, reset-safe joint robot/object replay for the table teacher.

Rows are synchronous control steps and columns are native environments.  ``s``
and ``o`` are pre-action state; ``s_next`` and ``o_next`` always mean the actual
last state before reset, including for a terminated episode. Windows retain
short failed prefixes and mask right padding without crossing resets. A deterministic
episode/environment split provides a disjoint WM diagnostic holdout using only
training references; protected evaluation is never accepted by this buffer.
"""
from __future__ import annotations

import torch


FIELDS = {
    "s": ((71,), torch.float32), "s_next": ((71,), torch.float32),
    "o": ((13,), torch.float32), "o_next": ((13,), torch.float32),
    "a": ((29,), torch.float32), "a_prev": ((29,), torch.float32),
    "xi": ((33,), torch.float32), "xi_o": ((12,), torch.float32),
    "mass_context": ((1,), torch.float32),
    "actuator": ((88,), torch.float32), "actuator_next": ((88,), torch.float32),
    "f": ((), torch.long), "next_f": ((), torch.long),
    "q_prev": ((29,), torch.float32), "qd_prev": ((29,), torch.float32),
    "object_history": ((10, 9), torch.float32),
    "done": ((), torch.bool), "episode_id": ((), torch.long),
    "asset_id": ((), torch.long),
    "contact": ((5,), torch.float32), "contact_valid": ((5,), torch.bool),
}
OPTIONAL_FIELDS = {"r": ((), torch.float32), "bin_id": ((), torch.long), "ep_step": ((), torch.long)}
ALIASES = {"s1": "s_next", "o1": "o_next", "f1": "next_f", "actual_contact": "contact"}


class ReplayBuffer:
    schema = "hoi_tables_real_joint_replay_v1"

    def __init__(self, capacity_steps=None, num_envs=None, device="cpu", *, partition="train", seed=0,
                 holdout_fraction=.1, T=None, N=None):
        capacity_steps = T if capacity_steps is None else capacity_steps
        num_envs = N if num_envs is None else num_envs
        if partition != "train":
            raise ValueError("Replay only accepts the frozen training partition")
        if type(capacity_steps) is not int or type(num_envs) is not int or min(capacity_steps, num_envs) < 1:
            raise ValueError("Replay capacity and environment count must be positive integers")
        if not 0 <= holdout_fraction < 1:
            raise ValueError("WM diagnostic holdout fraction must be in [0,1)")
        self.T, self.N, self.device = capacity_steps, num_envs, torch.device(device)
        self.partition, self.holdout_fraction = partition, float(holdout_fraction)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.data = {name: torch.empty((self.T, self.N, *shape), device=self.device, dtype=dtype)
                     for name, (shape, dtype) in {**FIELDS, **OPTIONAL_FIELDS}.items()}
        self.ptr, self.size = 0, 0
        self.appended_steps = 0

    def __len__(self):
        return self.size*self.N

    def append(self, transition):
        if transition.get("partition", "train") != "train" or transition.get("allowed_for_training", True) is not True:
            raise ValueError("Protected evaluation transitions cannot enter replay")
        values = dict(transition)
        for old, new in ALIASES.items():
            if old in values and new not in values:
                values[new] = values[old]
        missing = set(FIELDS)-values.keys()
        if missing:
            raise ValueError("Replay transition missing fields: " + ", ".join(sorted(missing)))
        prepared = {}
        for name, (shape, dtype) in {**FIELDS, **OPTIONAL_FIELDS}.items():
            value = torch.as_tensor(values.get(name, torch.zeros((self.N, *shape))), device=self.device)
            if value.shape != (self.N, *shape):
                raise ValueError(f"Replay {name}: expected {(self.N, *shape)}, received {tuple(value.shape)}")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError("Nonfinite real replay field: " + name)
            prepared[name] = value.to(dtype=dtype)
        if bool(((prepared["contact"] < 0) | (prepared["contact"] > 1)).any()):
            raise ValueError("Contact labels must lie in [0,1]")
        for name in ("f", "next_f", "episode_id", "asset_id"):
            if bool((prepared[name] < 0).any()):
                raise ValueError("Replay reference and identity indices must be nonnegative")
        # Validate the complete row before mutating a ring slot.  copy_ owns
        # storage so a subsequent in-place native reset cannot rewrite replay.
        for name, value in prepared.items():
            self.data[name][self.ptr].copy_(value)
        self.ptr = (self.ptr+1) % self.T
        self.size = min(self.size+1, self.T)
        self.appended_steps += 1

    add = append

    def _row(self, age):
        return (self.ptr-self.size+age) % self.T

    def _subset(self, ages, env, split):
        if split not in ("fit", "holdout", "all"):
            raise ValueError("Replay split must be fit, holdout or all")
        if split == "all":
            return torch.ones_like(ages, dtype=torch.bool)
        episode = self.data["episode_id"][self._row(ages), env]
        bucket = torch.remainder(episode*104729 + env*15485863, 1000)
        holdout = bucket < round(self.holdout_fraction*1000)
        return holdout if split == "holdout" else ~holdout

    def fit_mask(self, transition):
        """The same per-episode WM fit partition before a row is appended.

        World-model normalizers must use this mask as well as WM fit losses;
        otherwise the diagnostic holdout has already affected model fitting.
        Actor/critic statistics may still use all real training transitions.
        """
        if transition.get("partition", "train") != "train":
            raise ValueError("Protected evaluation cannot update WM fit statistics")
        episode = torch.as_tensor(transition["episode_id"], device=self.device, dtype=torch.long)
        if episode.shape != (self.N,) or bool((episode < 0).any()):
            raise ValueError("WM fit mask requires one nonnegative episode ID per native environment")
        env = torch.arange(self.N, device=self.device)
        return torch.remainder(episode*104729+env*15485863, 1000) >= round(self.holdout_fraction*1000)

    def _sample_indices(self, batch_size, *, window=1, split="all", horizon=None, motion=None):
        if type(batch_size) is not int or batch_size < 1 or type(window) is not int or window < 1:
            raise ValueError("Replay sample batch/window must be positive integers")
        if self.size < window:
            raise ValueError(f"Replay has {self.size} control steps; {window} are required")
        ages = torch.arange(self.size-window+1, device=self.device)[:, None].expand(-1, self.N)
        env = torch.arange(self.N, device=self.device)[None].expand_as(ages)
        valid = self._subset(ages, env, split)
        rows = self._row(ages)
        first_episode, first_asset = self.data["episode_id"][rows, env], self.data["asset_id"][rows, env]
        for offset in range(1, window):
            before, row = self._row(ages+offset-1), self._row(ages+offset)
            valid &= ~self.data["done"][before, env]
            valid &= self.data["episode_id"][row, env] == first_episode
            valid &= self.data["asset_id"][row, env] == first_asset
            valid &= self.data["next_f"][before, env] == self.data["f"][row, env]
        if horizon is not None:
            if type(horizon) is not int or horizon < 1 or motion is None:
                raise ValueError("A positive imagination horizon requires the motion library")
            frame = self.data["f"][rows, env]
            valid &= motion.clip_end[motion.clip_id[frame]]-frame >= horizon
        locations = valid.nonzero()
        if not len(locations):
            raise ValueError(f"Replay contains no reset-safe {split} windows of length {window}")
        chosen = locations[torch.randint(len(locations), (batch_size,), device=self.device, generator=self.generator)]
        return chosen[:, 0], chosen[:, 1]

    def sample_transitions(self, batch_size, *, split="all"):
        age, env = self._sample_indices(batch_size, split=split)
        result = {name: value[self._row(age), env] for name, value in self.data.items()}
        result["partition"] = "train"
        result["wm_split"] = split
        return result

    def sample_starts(self, batch_size, horizon=None, motion=None):
        age, env = self._sample_indices(batch_size, split="fit", horizon=horizon, motion=motion)
        result = {name: value[self._row(age), env] for name, value in self.data.items()}
        result["partition"] = "train"
        return result

    def sample_windows(self, batch_size, window, split="fit"):
        if type(window) is not int or window < 1:
            raise ValueError("Replay window must be a positive integer")
        # Sample every real start, including episodes that terminate before
        # `window` controls. Discarding those episodes would systematically
        # hide the exact difficult transitions targeted by adaptive RSI.
        age, env = self._sample_indices(batch_size, window=1, split=split)
        rows = self._row(age)
        first_episode = self.data["episode_id"][rows, env]
        first_asset = self.data["asset_id"][rows, env]
        selected_rows, validity = [rows], [torch.ones(batch_size, device=self.device, dtype=torch.bool)]
        for offset in range(1, window):
            available = age+offset < self.size
            candidate = self._row(torch.minimum(age+offset, age.new_full(age.shape, self.size-1)))
            previous = selected_rows[-1]
            active = (validity[-1] & available & ~self.data["done"][previous, env]
                      & (self.data["episode_id"][candidate, env] == first_episode)
                      & (self.data["asset_id"][candidate, env] == first_asset)
                      & (self.data["next_f"][previous, env] == self.data["f"][candidate, env]))
            # Repeat a real row for padding. Zero-fill would create invalid
            # quaternions/physical contexts before masking the WM loss.
            selected_rows.append(torch.where(active, candidate, previous))
            validity.append(active)
        rows, mask = torch.stack(selected_rows, 1), torch.stack(validity, 1)
        envs = env[:, None].expand_as(rows)
        result = {name: value[rows, envs] for name, value in self.data.items()}
        result["mask"] = mask.float()
        result["partition"] = "train"
        result["wm_split"] = split
        return result

    def state_dict(self):
        # The source training lineage historically omitted replay from normal
        # checkpoints.  Explicit snapshots support exact software resumption;
        # simulator and RNG restoration still need their separate contracts.
        rows = self._row(torch.arange(self.size, device=self.device))
        return {"schema": self.schema, "capacity_steps": self.T, "num_envs": self.N,
                "partition": self.partition, "holdout_fraction": self.holdout_fraction,
                "appended_steps": self.appended_steps, "generator_state": self.generator.get_state().clone(),
                "data": {name: value[rows].clone() for name, value in self.data.items()}}

    def load_state_dict(self, state):
        if (state.get("schema"), state.get("capacity_steps"), state.get("num_envs"), state.get("partition"), state.get("holdout_fraction")) != (self.schema, self.T, self.N, self.partition, self.holdout_fraction):
            raise ValueError("Replay checkpoint shape, partition or schema mismatch")
        if set(state["data"]) != set(self.data):
            raise ValueError("Replay checkpoint field schema mismatch")
        sizes = {len(value) for value in state["data"].values()}
        if len(sizes) != 1 or next(iter(sizes)) > self.T:
            raise ValueError("Replay checkpoint row count mismatch")
        size = next(iter(sizes))
        for name, value in state["data"].items():
            if value.shape != self.data[name][:size].shape or value.dtype != self.data[name].dtype:
                raise ValueError("Replay checkpoint field shape or dtype mismatch: " + name)
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError("Replay checkpoint contains nonfinite data")
        for name, value in state["data"].items():
            self.data[name][:size].copy_(value)
        self.size, self.ptr = size, size % self.T
        self.appended_steps = int(state["appended_steps"])
        self.generator.set_state(state["generator_state"].cpu())


JointReplayBuffer = ReplayBuffer
