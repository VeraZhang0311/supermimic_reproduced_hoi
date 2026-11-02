"""Explicit task-balanced collection for the compact privileged teacher.

Outer probabilities are realized as persistent environment quotas, so episode
length and termination frequency cannot change category transition shares.
Reset probabilities condition a global source-group/clip hierarchy on the
worker asset; the existing difficult-bin preference remains inside each clip.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math

import torch


SCHEMA = "hoi_compact_teacher_task_sampling_v1"
ULTRA_OBJECTS = ("largebox", "smallbox", "plasticbox", "suitcase")
HOI_OBJECTS = ("clothesstand", "floorlamp", "largetable", "monitor", "smalltable",
               "trashcan", "tripod", "whitechair", "woodchair")
SOCCER_TASKS = ("soccer/case1_kick_forward", "soccer/case2_kick_left", "soccer/case3_kick_right",
                "soccer/case4_carry", "soccer/case5_pickball_kick")
GRAIL_TRAIN_ASSETS = (2, 122, 169, 172, 200)
CATEGORY_PERCENT = {**{"ultra/" + obj: 6 for obj in ULTRA_OBJECTS},
                    **{"hoi_retarget/" + obj: 4 for obj in HOI_OBJECTS},
                    "omnicontact/table": 15, "omnicontact/soccer": 15, "grail_sitting/sitting": 10}


def default_sampling_contract():
    return {"schema": SCHEMA, "category_percent": dict(CATEGORY_PERCENT),
            "soccer_tasks": list(SOCCER_TASKS), "grail_train_original_asset_ids": list(GRAIL_TRAIN_ASSETS),
            "hierarchy": "category;equal_task_or_chair_stratum;equal_source_group;equal_clip;adaptive_bin;uniform_outgoing_frame",
            "transition_balance": "persistent_asset_and_soccer_task_environment_quotas",
            "bin_adaptation": "legacy_0.5_uniform_0.4_soft_0.1_ema_conditioned_on_selected_clip"}


def validate_sampling_contract(bundle):
    contract = bundle.get("task_sampling")
    if contract is None:
        return None
    if bundle.get("partition") != "train":
        raise ValueError("weighted task sampling is training-only; protected evaluation must remain exhaustive")
    if contract != default_sampling_contract():
        raise ValueError("unsupported or modified compact task sampling contract")
    return contract


def sampling_clip_metadata(entry):
    provenance = entry.get("provenance", {})
    source = entry.get("source_provenance", provenance.get("source_provenance", {}))
    dataset = entry.get("dataset", provenance.get("dataset"))
    task = entry.get("task", source.get("task", source.get("subset")))
    group = entry.get("source_group", provenance.get("source_group"))
    if not isinstance(group, str) or not group:
        raise ValueError("weighted clips require a nonempty frozen source_group")
    obj = source.get("object")
    if dataset == "ultra" and obj in ULTRA_OBJECTS:
        category = "ultra/" + obj
        stratum = category
    elif dataset == "hoi_retarget" and obj in HOI_OBJECTS:
        category = "hoi_retarget/" + obj
        stratum = category
    elif dataset == "omnicontact" and task == "box/case1_carry":
        category = "omnicontact/table"
        stratum = category
    elif dataset == "omnicontact" and task in SOCCER_TASKS:
        category = "omnicontact/soccer"
        stratum = "omnicontact/" + task
    elif dataset == "grail_sitting" and task == "sitting":
        original = entry.get("original_asset_id", entry.get("asset_id"))
        if original not in GRAIL_TRAIN_ASSETS:
            raise ValueError("weighted sitting clip is not one of the five selected training chairs")
        category = "grail_sitting/sitting"
        stratum = "grail_sitting/chair_{:04d}".format(original)
    else:
        raise ValueError("clip is outside the authorized compact sampling categories")
    return {"category": category, "stratum": stratum, "source_group": group}


def _clip_key(entry):
    provenance = entry.get("provenance", {})
    dataset = entry.get("dataset", provenance.get("dataset"))
    clip_id = entry.get("clip_id", provenance.get("clip_id"))
    if not isinstance(dataset, str) or not isinstance(clip_id, str) or not clip_id:
        raise ValueError("weighted clips require immutable dataset and clip_id identity")
    return dataset, clip_id


def build_sampling_plan(bundle):
    if validate_sampling_contract(bundle) is None:
        raise ValueError("weighted allocation requires an explicit task_sampling contract")
    clips = bundle["clips"]
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seen = set()
    records = []
    for entry in clips:
        key = _clip_key(entry)
        if key in seen:
            raise ValueError("duplicate weighted clip identity")
        seen.add(key)
        identity = sampling_clip_metadata(entry)
        record = {**identity, "key": key, "asset_id": int(entry["asset_id"])}
        records.append(record)
        groups[identity["category"]][identity["stratum"]][identity["source_group"]].append(record)
    if set(groups) != set(CATEGORY_PERCENT):
        raise ValueError("weighted dataset must contain all sixteen authorized categories")
    if set(groups["omnicontact/soccer"]) != {"omnicontact/" + task for task in SOCCER_TASKS}:
        raise ValueError("weighted soccer requires all five task strata")
    if set(groups["grail_sitting/sitting"]) != {"grail_sitting/chair_{:04d}".format(i) for i in GRAIL_TRAIN_ASSETS}:
        raise ValueError("weighted sitting requires all five chair strata")
    asset_categories = defaultdict(set)
    for category, strata in groups.items():
        for stratum, sources in strata.items():
            for source, members in sources.items():
                probability = CATEGORY_PERCENT[category] / 100 / len(strata) / len(sources) / len(members)
                for record in members:
                    record["probability"] = probability
                    asset_categories[record["asset_id"]].add(category)
    if any(len(categories) != 1 for categories in asset_categories.values()):
        raise ValueError("an asset cannot span multiple weighted outer categories")
    soccer_assets = {r["asset_id"] for r in records if r["category"] == "omnicontact/soccer"}
    if len(soccer_assets) != 1:
        raise ValueError("soccer task lanes require the one shared frozen soccer asset")
    for stratum, sources in groups["grail_sitting/sitting"].items():
        if len({r["asset_id"] for members in sources.values() for r in members}) != 1:
            raise ValueError("each sitting stratum must bind exactly one physical chair")
    assert math.isclose(sum(r["probability"] for r in records), 1., abs_tol=1e-12)
    return {"schema": SCHEMA, "records": records, "category_percent": dict(CATEGORY_PERCENT)}


def apportion(weights, total, minimum=None):
    """Deterministic constrained Hamilton allocation; keys must be sortable.

    A lower bound is enforced only if an unconstrained allocation would violate
    it. It is not added to every weight, which would bias small categories.
    """
    if type(total) is not int or total < 1 or not weights:
        raise ValueError("apportionment needs positive integer budget and nonempty weights")
    if any(not math.isfinite(float(w)) or w <= 0 for w in weights.values()):
        raise ValueError("apportionment weights must be finite and positive")
    lower = {k: 0 for k in weights} if minimum is None else dict(minimum)
    if set(lower) != set(weights) or any(type(n) is not int or n < 0 for n in lower.values()):
        raise ValueError("invalid apportionment lower bounds")
    if sum(lower.values()) > total:
        raise ValueError("environment budget cannot cover every retained asset/stratum")
    pending = dict(weights)
    result = {}
    remaining = total
    while pending:
        norm = sum(pending.values())
        exact = {k: remaining * w / norm for k, w in pending.items()}
        forced = [k for k in pending if exact[k] < lower[k]]
        if not forced:
            floor = {k: int(math.floor(x)) for k, x in exact.items()}
            order = sorted(pending, key=lambda k: (-(exact[k] - floor[k]), k))
            for k in order[:remaining - sum(floor.values())]:
                floor[k] += 1
            result.update(floor)
            break
        for k in forced:
            result[k] = lower[k]
            remaining -= lower[k]
            del pending[k]
    assert sum(result.values()) == total and all(result[k] >= lower[k] for k in result)
    return result


def weighted_asset_quotas(bundle, num_envs):
    plan = build_sampling_plan(bundle)
    category_assets = defaultdict(lambda: defaultdict(float))
    for record in plan["records"]:
        category_assets[record["category"]][record["asset_id"]] += record["probability"]
    minimum = {c: len(assets) for c, assets in category_assets.items()}
    minimum["omnicontact/soccer"] = max(minimum["omnicontact/soccer"], len(SOCCER_TASKS))
    category_quota = apportion(CATEGORY_PERCENT, num_envs, minimum)
    quotas = {}
    for category, weights in category_assets.items():
        quotas.update(apportion(weights, category_quota[category], {a: 1 for a in weights}))
    return quotas


def allocation_report(bundle, specs):
    plan = build_sampling_plan(bundle)
    quotas = {s.asset_id: s.num_envs for s in specs}
    if len(quotas) != len(specs) or quotas != weighted_asset_quotas(bundle, sum(quotas.values())):
        raise ValueError("worker quotas do not match the frozen weighted allocation")
    total = sum(quotas.values())
    category = defaultdict(int)
    strata = defaultdict(int)
    asset_records = defaultdict(list)
    for r in plan["records"]:
        asset_records[r["asset_id"]].append(r)
    for aid, n in quotas.items():
        rr = asset_records[aid]
        category[rr[0]["category"]] += n
        names = sorted({r["stratum"] for r in rr})
        if len(names) > 1 and rr[0]["category"] != "omnicontact/soccer":
            raise ValueError("only soccer may contain multiple persistent task lanes per asset")
        for name, count in apportion({name: 1 for name in names}, n, {name: 1 for name in names}).items():
            strata[name] += count
    return {"schema": SCHEMA, "num_envs": total,
            "category_env_counts": dict(sorted(category.items())), "stratum_env_counts": dict(sorted(strata.items())),
            "asset_env_counts": dict(sorted(quotas.items())),
            "category_requested_fraction": {c: p / 100 for c, p in CATEGORY_PERCENT.items()},
            "category_realized_fraction": {c: n / total for c, n in category.items()},
            "scope": "expected transition proportions for one complete collector vector step; resets logged separately"}


def attach_sampling_metadata(bank, clips, contract, population):
    bank.task_sampling = None
    if contract is None:
        return
    if bank.partition != "train" or population is None:
        raise ValueError("weighted bank requires the full frozen training population")
    plan = build_sampling_plan({"partition": bank.partition, "task_sampling": contract, "clips": population})
    lookup = {r["key"]: r for r in plan["records"]}
    bank.task_sampling = deepcopy(contract)
    bank.sampling_category_names = sorted(CATEGORY_PERCENT)
    bank.sampling_stratum_names = sorted({r["stratum"] for r in plan["records"]})
    selected = []
    for entry in clips:
        r = lookup[_clip_key(entry)]
        if r["asset_id"] != int(entry["asset_id"]) or sampling_clip_metadata(entry) != {k: r[k] for k in ("category", "stratum", "source_group")}:
            raise ValueError("filtered motion metadata differs from the full weighted population")
        selected.append(dict(r))
    bank.sampling_records = selected
    bank.sampling_clip_categories = torch.tensor([bank.sampling_category_names.index(r["category"]) for r in selected], dtype=torch.long, device=bank.device)
    bank.sampling_clip_strata = torch.tensor([bank.sampling_stratum_names.index(r["stratum"]) for r in selected], dtype=torch.long, device=bank.device)
    bank.sampling_clip_probabilities = torch.tensor([r["probability"] for r in selected], dtype=torch.float64, device=bank.device)


class PersistentTaskSampler:
    """Weighted reset selection while every environment keeps its task stratum."""
    def __init__(self, motion, asset_id, num_envs):
        if not getattr(motion, "task_sampling", None) or motion.partition != "train":
            raise ValueError("persistent weighted sampling requires a weighted training bank")
        self.motion, self.asset_id, self.num_envs = motion, int(asset_id), num_envs
        self.device = motion.device
        clip_ids = [i for i, r in enumerate(motion.sampling_records) if r["asset_id"] == self.asset_id]
        if not clip_ids:
            raise ValueError("weighted worker asset has no clips")
        names = sorted({motion.sampling_records[i]["stratum"] for i in clip_ids})
        quota = apportion({s: 1 for s in names}, num_envs, {s: 1 for s in names})
        self.stratum_names = names
        self.lane_strata = torch.tensor([motion.sampling_stratum_names.index(s) for s in names for _ in range(quota[s])], dtype=torch.long, device=self.device)
        self.clips = {}
        self.weights = {}
        for name in names:
            sid = motion.sampling_stratum_names.index(name)
            ids = torch.tensor([i for i in clip_ids if motion.sampling_records[i]["stratum"] == name], dtype=torch.long, device=self.device)
            self.clips[sid] = ids
            weights = motion.sampling_clip_probabilities[ids]
            self.weights[sid] = weights / weights.sum()
        self.clip_bins = {i: torch.nonzero(motion.bin_clip == i, as_tuple=False).flatten() for i in clip_ids}

    def validate_frames(self, env_ids, frames):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        frames = torch.as_tensor(frames, dtype=torch.long, device=self.device)
        if env_ids.ndim != 1 or frames.shape != env_ids.shape or bool(((env_ids < 0) | (env_ids >= self.num_envs)).any()):
            raise ValueError("weighted reset environment/frame shapes differ")
        if bool(((frames < 0) | (frames >= self.motion.num_frames)).any()):
            raise ValueError("weighted reset frame is outside the reference bank")
        if not bool((self.motion.asset_ids[frames] == self.asset_id).all()):
            raise ValueError("weighted reset frame belongs to another asset")
        if not torch.equal(self.motion.sampling_clip_strata[self.motion.clip_id[frames]], self.lane_strata[env_ids]):
            raise ValueError("explicit reset would change a persistent task lane")

    def first_frames(self, env_ids, horizon=1):
        """Deterministic compatible reset for probes, with explicit reference headroom."""
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        result = torch.empty_like(env_ids)
        for sid, ids in self.clips.items():
            available = ids[self.motion.clip_len[ids] > horizon]
            if not len(available):
                raise ValueError("task stratum has no clip with requested reference headroom")
            chosen = available[torch.argmax(self.motion.clip_len[available])]
            result[self.lane_strata[env_ids] == sid] = self.motion.clip_start[chosen]
        self.validate_frames(env_ids, result)
        return result

    def sample(self, env, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        n = len(env_ids)
        selected_clips = torch.empty(n, dtype=torch.long, device=self.device)
        for sid, ids in self.clips.items():
            mask = self.lane_strata[env_ids] == sid
            count = int(mask.sum())
            if count:
                selected_clips[mask] = ids[torch.multinomial(self.weights[sid], count, replacement=True)]
        sampled = torch.empty_like(selected_clips)
        for clip in selected_clips.unique().tolist():
            mask = selected_clips == clip
            bins = self.clip_bins[clip]
            soft = torch.softmax(-25 * env.bin_G[bins] / env.G_max, 0)
            env.bin_w_avg[bins] = .999 * env.bin_w_avg[bins] + .001 * soft
            avg = env.bin_w_avg[bins]
            probs = .5 / len(bins) + .4 * soft + .1 * avg / avg.sum()
            sampled[mask] = bins[torch.multinomial(probs, int(mask.sum()), replacement=True)]
        start = self.motion.bin_start[sampled]
        end = torch.minimum(start + 50, self.motion.clip_end[self.motion.bin_clip[sampled]])
        frames = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        self.validate_frames(env_ids, frames)
        return frames, sampled
