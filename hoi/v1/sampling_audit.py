"""Count actual collected control intervals, independently of reset lotteries."""
from __future__ import annotations

import torch

from .task_sampling import allocation_report, build_sampling_plan


class ActualCollectionAudit:
    """Map worker-global start frames to frozen clips and verify every vector tick.

    Counts describe collection, not replay minibatches or optimizer updates.
    Reset/group/clip counts are separate because episode lengths can differ.
    """

    def __init__(self, bundle, specs, device="cpu"):
        self.device = torch.device(device)
        self.allocation = allocation_report(bundle, specs)
        records = build_sampling_plan(bundle)["records"]
        self.category_names = sorted(self.allocation["category_env_counts"])
        self.stratum_names = sorted(self.allocation["stratum_env_counts"])
        self.asset_names = sorted(self.allocation["asset_env_counts"])
        self.source_group_names = sorted({r["source_group"] for r in records})
        self.clip_names = [c.get("name", c["clip_id"]) for c in bundle["clips"]]
        make = lambda values: torch.tensor(values, dtype=torch.long, device=self.device)
        lengths = make([c["state_frames"] for c in bundle["clips"]])
        if bool((lengths < 2).any()):
            raise ValueError("collection audit needs outgoing reference intervals")
        self.ends_exclusive = lengths.cumsum(0)
        self.clip_category = make([self.category_names.index(r["category"]) for r in records])
        self.clip_stratum = make([self.stratum_names.index(r["stratum"]) for r in records])
        self.clip_asset = make([self.asset_names.index(int(c["asset_id"])) for c in bundle["clips"]])
        self.clip_group = make([self.source_group_names.index(r["source_group"]) for r in records])
        self.asset_values = make(self.asset_names)
        self.expected_row_assets = make([s.asset_id for s in specs for _ in range(s.num_envs)])
        self.N = len(self.expected_row_assets)
        self.expected = {
            "category": make([self.allocation["category_env_counts"][k] for k in self.category_names]),
            "stratum": make([self.allocation["stratum_env_counts"][k] for k in self.stratum_names]),
            "asset": make([self.allocation["asset_env_counts"][k] for k in self.asset_names]),
        }
        self.counts = {k: torch.zeros_like(v) for k, v in self.expected.items()}
        self.reset_category = torch.zeros_like(self.expected["category"])
        self.reset_stratum = torch.zeros_like(self.expected["stratum"])
        self.clip_resets = torch.zeros(len(records), dtype=torch.long, device=self.device)
        self.clip_transitions = torch.zeros_like(self.clip_resets)
        self.group_resets = torch.zeros(len(self.source_group_names), dtype=torch.long, device=self.device)
        self.group_transitions = torch.zeros_like(self.group_resets)
        self.vector_steps = self.terminations = self.completed_episodes = 0
        self._lane_strata = None

    @torch.no_grad()
    def observe(self, transition):
        values = {}
        for key in ("f", "asset_id", "ep_step", "done", "term"):
            raw = transition[key]
            if not isinstance(raw, torch.Tensor) or raw.shape != (self.N,):
                raise ValueError("sampling audit transition shape differs: " + key)
            if key in ("f", "asset_id", "ep_step") and raw.dtype not in (torch.int32, torch.int64):
                raise ValueError("sampling audit requires integer frame/identity fields")
            if key in ("done", "term") and raw.dtype != torch.bool:
                raise ValueError("sampling audit requires boolean episode fields")
            values[key] = raw.detach().to(self.device)
        frames = values["f"]
        if bool(((frames < 0) | (frames >= self.ends_exclusive[-1])).any()):
            raise ValueError("collected frame is outside the frozen reference bank")
        clips = torch.bucketize(frames, self.ends_exclusive, right=True)
        if bool((frames >= self.ends_exclusive[clips] - 1).any()):
            raise ValueError("collected interval begins at a terminal-only frame")
        category, stratum, asset = self.clip_category[clips], self.clip_stratum[clips], self.clip_asset[clips]
        if (not torch.equal(values["asset_id"], self.expected_row_assets)
                or not torch.equal(self.asset_values[asset], values["asset_id"])):
            raise ValueError("collected asset/frame identity differs from worker allocation")
        observed = {k: torch.bincount(v, minlength=len(self.expected[k]))
                    for k, v in (("category", category), ("stratum", stratum), ("asset", asset))}
        if any(not torch.equal(observed[k], self.expected[k]) for k in observed):
            raise ValueError("actual collected transition ratios differ from frozen task quotas")
        if self._lane_strata is not None and not torch.equal(self._lane_strata, stratum):
            raise ValueError("collector changed a persistent task lane across reset")
        if bool((values["ep_step"] < 0).any()):
            raise ValueError("collected episode step must be nonnegative")
        # Only count after all checks pass; failed vectors cannot pollute receipts.
        self._lane_strata = stratum.clone()
        for k in observed:
            self.counts[k] += observed[k]
        resets = values["ep_step"] == 0
        groups = self.clip_group[clips]
        self.reset_category += torch.bincount(category[resets], minlength=len(self.category_names))
        self.reset_stratum += torch.bincount(stratum[resets], minlength=len(self.stratum_names))
        self.clip_resets += torch.bincount(clips[resets], minlength=len(self.clip_names))
        self.clip_transitions += torch.bincount(clips, minlength=len(self.clip_names))
        self.group_resets += torch.bincount(groups[resets], minlength=len(self.source_group_names))
        self.group_transitions += torch.bincount(groups, minlength=len(self.source_group_names))
        self.vector_steps += 1
        self.completed_episodes += int(values["done"].sum())
        self.terminations += int(values["term"].sum())

    def report(self, *, detail=False):
        total = self.vector_steps * self.N
        def named(names, values):
            return dict(zip(names, values.detach().cpu().tolist()))
        category = named(self.category_names, self.counts["category"])
        fractions = {k: n / total if total else 0. for k, n in category.items()}
        result = {"schema": "hoi_actual_transition_sampling_v1", "num_envs": self.N,
                  "vector_steps": self.vector_steps, "total_transitions": total,
                  "allocation_match": bool(total) and all(torch.equal(self.counts[k], self.expected[k] * self.vector_steps) for k in self.expected),
                  "category_counts": category, "category_fractions": fractions,
                  "category_requested_fractions": self.allocation["category_requested_fraction"],
                  "category_error_percentage_points": {k: 100 * (fractions[k] - v) for k, v in self.allocation["category_requested_fraction"].items()},
                  "stratum_counts": named(self.stratum_names, self.counts["stratum"]),
                  "asset_counts": named(self.asset_names, self.counts["asset"]),
                  "reset_category_counts": named(self.category_names, self.reset_category),
                  "reset_stratum_counts": named(self.stratum_names, self.reset_stratum),
                  "completed_episodes": self.completed_episodes, "terminations": self.terminations,
                  "scope": "actual transition start frames; collection only, not replay/optimizer minibatch proportions",
                  "allocation": self.allocation}
        if detail:
            result.update(clip_names=self.clip_names, clip_reset_counts=self.clip_resets.cpu().tolist(),
                          clip_transition_counts=self.clip_transitions.cpu().tolist(),
                          source_group_reset_counts=named(self.source_group_names, self.group_resets),
                          source_group_transition_counts=named(self.source_group_names, self.group_transitions))
        return result
