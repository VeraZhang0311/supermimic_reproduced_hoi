"""Joint robot/object replay with same-episode histories and explicit WM holdout."""

from __future__ import annotations

import torch
from .contacts import CHANNELS


FIELDS = {
    "s": (71, torch.float32), "s1": (71, torch.float32),
    "o": (13, torch.float32), "o1": (13, torch.float32),
    "a": (29, torch.float32), "a_prev": (29, torch.float32),
    "xi": (33, torch.float32), "xi_o": (12, torch.float32),
    "r": ((), torch.float32), "r_human": ((), torch.float32), "r_hoi": ((), torch.float32),
    "f": ((), torch.long), "ep_step": ((), torch.int32),
    "done": ((), torch.bool), "term": ((), torch.bool), "asset_id": ((), torch.long),
    "failure_reason": ((), torch.int32),
    "contact": (len(CHANNELS), torch.float32), "contact_valid": (len(CHANNELS), torch.bool),
    "impulse": (len(CHANNELS), torch.float32), "contact_missing_counter": (2, torch.int32),
}


class ReplayBuffer:
    def __init__(self, T, N, device="cpu"):
        if min(T, N) < 1:
            raise ValueError("replay dimensions must be positive")
        self.T, self.N, self.device = T, N, torch.device(device)
        self.data = {key: torch.zeros((T, N) + ((shape,) if isinstance(shape, int) else shape),
                                      dtype=dtype, device=device)
                     for key, (shape, dtype) in FIELDS.items()}
        self.ptr = self.size = 0

    @torch.no_grad()
    def add(self, transition):
        for key, (shape, _) in FIELDS.items():
            if key not in transition:
                raise ValueError(f"joint replay transition is missing {key}")
            expected = (self.N,) + ((shape,) if isinstance(shape, int) else shape)
            if tuple(transition[key].shape) != expected:
                raise ValueError(f"wrong {key} shape: {tuple(transition[key].shape)} != {expected}")
            if transition[key].is_floating_point() and not bool(torch.isfinite(transition[key]).all()):
                raise ValueError(f"nonfinite real transition: {key}")
        if not bool(((transition["contact"] == 0) | (transition["contact"] == 1)).all()):
            raise ValueError("real pair-contact labels must be binary")
        if bool((transition["impulse"] < 0).any()):
            raise ValueError("normal contact impulse must be nonnegative")
        for key in FIELDS:
            self.data[key][self.ptr].copy_(transition[key].detach())
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def _row(self, age):
        return (self.ptr - self.size + age) % self.T

    def _windows_at(self, ages, env, k):
        rows = self._row(ages[:, None] + torch.arange(k, device=self.device))
        index = env[:, None].expand_as(rows)
        values = {key: item[rows, index] for key, item in self.data.items()}
        connected = (~values["done"][:, :-1]) & (
            values["ep_step"][:, 1:] == values["ep_step"][:, :-1] + 1
        ) & (values["asset_id"][:, 1:] == values["asset_id"][:, :-1]) & (
            values["f"][:, 1:] == values["f"][:, :-1] + 1)
        values["mask"] = torch.cumprod(torch.cat((torch.ones((len(ages), 1), device=self.device),
                                                  connected.float()), -1), -1)
        return values

    def _eligible_window_starts(self, k, split):
        """Enumerate only metadata; keep the same conditional uniform law.

        A valid window ends its supervised prefix at the first reset, asset
        change or reference jump, exactly as in ``_windows_at``. The resulting
        flat indices address (start age, environment), not reference frames.
        """
        rows = self._row(torch.arange(self.size, device=self.device))
        frame = self.data["f"][rows]
        step = self.data["ep_step"][rows]
        asset = self.data["asset_id"][rows]
        done = self.data["done"][rows]
        connected = (~done[:-1]) & (step[1:] == step[:-1] + 1) & (
            asset[1:] == asset[:-1]) & (frame[1:] == frame[:-1] + 1)
        held = (frame // 50).remainder(20) == 0
        del frame, step, asset, done
        count = self.size - k + 1
        active = torch.ones((count, self.N), dtype=torch.bool, device=self.device)
        eligible = torch.ones_like(active)
        for offset in range(k):
            if offset:
                active &= connected[offset - 1:offset - 1 + count]
            selected = held[offset:offset + count]
            if split == "fit":
                selected = ~selected
            eligible &= selected | ~active
        return eligible.flatten().nonzero().flatten()

    def sample_windows(self, B, k, *, split="fit"):
        """A fixed 1/20 of reference bins is reserved for WM validation.

        Whole k-step reference ranges must lie on the same side of this split;
        no held-out transition appears in a fitted window. Terminal transitions
        remain supervised, with subsequent reset transitions masked out.
        """
        if self.size < k or k < 1 or B < 1:
            raise ValueError("not enough replay for the requested WM window")
        if split not in ("fit", "holdout", "all"):
            raise ValueError("unknown replay WM split")
        batches, remaining = [], B
        for _ in range(32):
            ages = torch.randint(self.size - k + 1, (max(2 * remaining, 32),), device=self.device)
            env = torch.randint(self.N, ages.shape, device=self.device)
            window = self._windows_at(ages, env, k)
            held = (window["f"] // 50).remainder(20) == 0
            valid = window["mask"].bool()
            if split == "fit":
                keep = (~held | ~valid).all(-1)
            elif split == "holdout":
                keep = (held | ~valid).all(-1)
            else:
                keep = torch.ones_like(ages, dtype=torch.bool)
            ids = keep.nonzero().flatten()[:remaining]
            if len(ids):
                batches.append({key: value[ids] for key, value in window.items()})
                remaining -= len(ids)
            if not remaining:
                return {key: torch.cat([batch[key] for batch in batches]) for key in batches[0]}
        # Holdout covers only 1/20 reference bins. With 32-step windows its
        # eligible start rate can be ~1.9%, so a fixed rejection budget can
        # fail even though many valid windows exist. Complete the batch by
        # drawing uniformly, with replacement, from the exact eligible starts.
        # This preserves the rejection sampler's distribution and split/mask
        # rules; only truly absent windows now report insufficient coverage.
        eligible = self._eligible_window_starts(k, split)
        if not len(eligible):
            raise ValueError(f"insufficient {split} replay coverage for WM windows: no eligible starts")
        selected = eligible[torch.randint(len(eligible), (remaining,), device=self.device)]
        batches.append(self._windows_at(selected // self.N, selected % self.N, k))
        return {key: torch.cat([batch[key] for batch in batches]) for key in batches[0]}

    def sample_starts(self, B, hist, *, horizon, motion):
        if self.size < 1 or min(B, hist, horizon) < 1:
            raise ValueError("invalid imagination start request")
        starts, histories, remaining = [], [], B
        for _ in range(32):
            ages = torch.randint(self.size, (max(2 * remaining, 32),), device=self.device)
            env = torch.randint(self.N, ages.shape, device=self.device)
            rows = self._row(ages)
            f = self.data["f"][rows, env]
            ok = f + horizon <= motion.clip_end[motion.clip_id[f]]
            ids = ok.nonzero().flatten()[:remaining]
            if not len(ids):
                continue
            ages, env, rows = ages[ids], env[ids], rows[ids]
            step = self.data["ep_step"][rows, env].long()
            # Restore only the contiguous suffix even if a collector supplied
            # inconsistent episode metadata. Never stitch across an asset,
            # reference jump or terminal boundary; pad the earliest valid row.
            usable = torch.zeros_like(step)
            continuous = torch.ones_like(step, dtype=torch.bool)
            for offset in range(1, hist):
                past = self._row(ages - offset)
                continuous = continuous & (step >= offset) & (ages >= offset)
                continuous = continuous & (~self.data["done"][past, env]) & (
                    self.data["asset_id"][past, env] == self.data["asset_id"][rows, env]) & (
                    self.data["ep_step"][past, env].long() == step - offset) & (
                    self.data["f"][past, env] == self.data["f"][rows, env] - offset)
                usable += continuous.long()
            back = torch.arange(hist - 1, -1, -1, device=self.device)
            back = torch.minimum(back[None], usable[:, None])
            hrows = self._row(ages[:, None] - back)
            henv = env[:, None].expand_as(hrows)
            keys = ("s", "o", "a_prev", "xi", "xi_o", "f", "asset_id", "contact_missing_counter")
            starts.append({key: self.data[key][rows, env] for key in keys})
            histories.append({key: self.data[key][hrows, henv] for key in keys})
            remaining -= len(ids)
            if not remaining:
                combine = lambda items: {key: torch.cat([item[key] for item in items]) for key in items[0]}
                return combine(starts), combine(histories)
        raise ValueError("replay has no sufficient full-horizon reference starts; do not loop indefinitely")
