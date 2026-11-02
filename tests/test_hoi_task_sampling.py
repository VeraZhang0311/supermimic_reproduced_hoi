"""Task quotas and reset probabilities under unequal clip/episode lengths."""
import ast
import copy
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from hoi.v1.task_sampling import (CATEGORY_PERCENT, GRAIL_TRAIN_ASSETS, HOI_OBJECTS,
    SOCCER_TASKS, ULTRA_OBJECTS, PersistentTaskSampler, allocation_report, apportion,
    attach_sampling_metadata, build_sampling_plan, default_sampling_contract,
    sampling_clip_metadata, validate_sampling_contract, weighted_asset_quotas)
from hoi.v1.workers import allocate_workers


def population():
    assets, clips = [], []

    def asset():
        aid = len(assets)
        assets.append({"asset_id": aid, "object_asset": "fixture_" + str(aid), "exact_scale": [1., 1., 1.],
                       "collision_hash": str(aid), "physical_profile": {"mass": 2.}})
        return aid

    def clip(aid, dataset, group, *, obj=None, task="unspecified", original=None, length=101):
        row = {"asset_id": aid, "dataset": dataset, "clip_id": "clip_" + str(len(clips)),
               "source_group": group, "task": task, "source_provenance": {"object": obj},
               "state_frames": length}
        if original is not None:
            row["original_asset_id"] = original
        clips.append(row)

    for obj in ULTRA_OBJECTS:
        aid = asset()
        # One long, unaugmented source versus nine shorter augmentation clips.
        clip(aid, "ultra", obj + "_A", obj=obj, length=401)
        for _ in range(9):
            clip(aid, "ultra", obj + "_B", obj=obj, length=51)
    for obj in HOI_OBJECTS:
        clip(asset(), "hoi_retarget", obj, obj=obj)
    aid = asset()
    clip(aid, "omnicontact", "table_A", task="box/case1_carry")
    aid = asset()
    for i, task in enumerate(SOCCER_TASKS):
        for _ in range(i + 1):
            clip(aid, "omnicontact", "soccer_" + str(i), task=task, length=101 + i * 50)
    for original in GRAIL_TRAIN_ASSETS:
        clip(asset(), "grail_sitting", "chair_" + str(original), task="sitting", original=original)
    return {"partition": "train", "task_sampling": default_sampling_contract(), "assets": assets, "clips": clips}


def motion_for(bundle, aid=None):
    clips = [c for c in bundle["clips"] if aid is None or c["asset_id"] == aid]
    lengths = torch.tensor([c["state_frames"] for c in clips])
    starts = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]))
    bins, bin_clips = [], []
    for i, (start, length) in enumerate(zip(starts.tolist(), lengths.tolist())):
        for offset in range(0, length - 1, 50):
            bins.append(start + offset)
            bin_clips.append(i)
    bank = SimpleNamespace(partition=bundle["partition"], device="cpu", clip_len=lengths, clip_start=starts,
                           clip_end=starts + lengths - 1, num_frames=int(lengths.sum()),
                           clip_id=torch.repeat_interleave(torch.arange(len(clips)), lengths),
                           asset_ids=torch.cat([torch.full((int(n),), c["asset_id"], dtype=torch.long) for c,n in zip(clips,lengths)]),
                           bin_start=torch.tensor(bins), bin_clip=torch.tensor(bin_clips), num_bins=len(bins))
    bank.bin_asset = bank.asset_ids[bank.bin_start]
    attach_sampling_metadata(bank, clips, bundle.get("task_sampling"), bundle["clips"])
    return bank


def adaptive_state(bank):
    # Gross differences in clip difficulty must not change source-group mass.
    return SimpleNamespace(bin_G=torch.arange(bank.num_bins).float() * 100,
                           bin_w_avg=torch.full((bank.num_bins,), 1 / bank.num_bins), G_max=100.)


class TaskSamplingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.bundle = population()

    def test_category_quota_2048_and_4096_and_header_match(self):
        bank = motion_for(self.bundle)
        for n in (2048, 4096):
            specs = allocate_workers(self.bundle, bank, n)
            report = allocation_report(self.bundle, specs)
            self.assertEqual(sum(s.num_envs for s in specs), n)
            self.assertEqual({s.asset_id for s in specs}, set(bank.bin_asset.tolist()))
            for category, percent in CATEGORY_PERCENT.items():
                self.assertLessEqual(abs(report["category_env_counts"][category] - n * percent / 100), 1)
            soccer = [v for k,v in report["stratum_env_counts"].items() if k.startswith("omnicontact/soccer/")]
            self.assertEqual(len(soccer), 5)
            self.assertLessEqual(max(soccer) - min(soccer), 1)
            grail = [v for k,v in report["stratum_env_counts"].items() if k.startswith("grail_sitting/chair_")]
            self.assertEqual(len(grail), 5)
            self.assertLessEqual(max(grail) - min(grail), 1)
            header_motion = SimpleNamespace(bin_asset=bank.bin_asset)
            self.assertEqual(specs, allocate_workers(self.bundle, header_motion, n))

    def test_full_hierarchy_equal_sources_despite_augmentations_and_duration(self):
        records = build_sampling_plan(self.bundle)["records"]
        selected = [r for r in records if r["category"] == "ultra/largebox"]
        masses = Counter()
        for r in selected:
            masses[r["source_group"]] += r["probability"]
        self.assertAlmostEqual(masses["largebox_A"], .03)
        self.assertAlmostEqual(masses["largebox_B"], .03)
        bank = motion_for(self.bundle, 0)
        sampler = PersistentTaskSampler(bank, 0, 50000)
        torch.manual_seed(73)
        frames, bins = sampler.sample(adaptive_state(bank), torch.arange(50000))
        chosen = bank.clip_id[frames]
        self.assertAlmostEqual(float(chosen.eq(0).float().mean()), .5, delta=.01)
        for clip in range(1,10):
            self.assertAlmostEqual(float(chosen.eq(clip).float().mean()), 1 / 18, delta=.006)
        self.assertTrue(bool((frames < bank.clip_end[chosen]).all()))
        self.assertTrue(torch.equal(bank.bin_clip[bins], chosen))

    def test_cross_asset_source_group_probability_is_conditioned_globally(self):
        # Move one of B's nine variants into a second scale asset. The new asset
        # gets 1/18 of the category, not half merely because there are two assets.
        b = copy.deepcopy(self.bundle)
        new_asset = copy.deepcopy(b["assets"][0])
        new_asset.update(asset_id=len(b["assets"]), object_asset="second_scale", collision_hash="second")
        b["assets"].append(new_asset)
        b["clips"][1]["asset_id"] = new_asset["asset_id"]
        records = build_sampling_plan(b)["records"]
        mass = sum(r["probability"] for r in records if r["asset_id"] == new_asset["asset_id"])
        self.assertAlmostEqual(mass, .06 / 18)
        filtered = motion_for(b, 0)
        weights = filtered.sampling_clip_probabilities / filtered.sampling_clip_probabilities.sum()
        self.assertAlmostEqual(float(weights[0]), 9 / 17)
        quotas = weighted_asset_quotas(b, 4096)
        category_n = quotas[0] + quotas[new_asset["asset_id"]]
        self.assertLessEqual(abs(quotas[new_asset["asset_id"]] - category_n / 18), 1)

    def test_persistent_soccer_transition_counts_ignore_episode_lengths(self):
        aid = next(r["asset_id"] for r in build_sampling_plan(self.bundle)["records"] if r["category"] == "omnicontact/soccer")
        bank = motion_for(self.bundle, aid)
        sampler = PersistentTaskSampler(bank, aid, 75)
        env = adaptive_state(bank)
        ids = torch.arange(75)
        frames = sampler.first_frames(ids, horizon=32)
        ages = torch.zeros(75, dtype=torch.long)
        lengths = torch.repeat_interleave(torch.tensor([1,2,5,11,23]), 15)
        transition_counts = Counter()
        reset_counts = Counter()
        torch.manual_seed(9)
        for _ in range(120):
            strata = bank.sampling_clip_strata[bank.clip_id[frames]]
            transition_counts.update(strata.tolist())
            ages += 1
            # Include clip-end resets as well as deliberately unequal episodes.
            done = (ages >= lengths) | (frames + 1 >= bank.clip_end[bank.clip_id[frames]])
            frames += 1
            reset_ids = ids[done]
            if len(reset_ids):
                reset_counts.update(sampler.lane_strata[reset_ids].tolist())
                frames[reset_ids], _ = sampler.sample(env, reset_ids)
                ages[reset_ids] = 0
        self.assertEqual(sorted(transition_counts.values()), [1800] * 5)
        self.assertGreater(max(reset_counts.values()), 5 * min(reset_counts.values()))

    def test_explicit_reset_cannot_change_soccer_lane(self):
        aid = next(r["asset_id"] for r in build_sampling_plan(self.bundle)["records"] if r["category"] == "omnicontact/soccer")
        bank = motion_for(self.bundle, aid)
        sampler = PersistentTaskSampler(bank, aid, 10)
        frames = sampler.first_frames(torch.arange(10), horizon=32)
        sampler.validate_frames(torch.arange(10), frames)
        bad = frames.clone()
        bad[0] = frames[-1]
        with self.assertRaisesRegex(ValueError, "persistent task lane"):
            sampler.validate_frames(torch.arange(10), bad)
        with self.assertRaisesRegex(ValueError, "headroom"):
            sampler.first_frames(torch.arange(10), horizon=10000)

    def test_native_bridge_initial_constructor_partial_and_explicit_resets(self):
        # Execute the real nested override against a minimal base reset; no
        # IsaacLab dependency or replacement of the production sampling body.
        source=Path(__file__).parents[1]/"hoi/v1/native.py"
        tree=ast.parse(source.read_text(encoding="utf-8"))
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=="AssetTrackingSim")
        aid=next(r["asset_id"] for r in build_sampling_plan(self.bundle)["records"] if r["category"]=="omnicontact/soccer")
        bank=motion_for(self.bundle,aid)
        sampler=PersistentTaskSampler(bank,aid,25)

        class Base:
            def __init__(self):
                self.motion,self.N,self.device=bank,25,"cpu"
                self.__dict__.update(vars(adaptive_state(bank)))
                self.f=torch.zeros(25,dtype=torch.long)
                self.reset(torch.arange(25))

            def reset(self,env_ids,frames=None):
                if not len(env_ids):return
                if frames is None:frames,_=self.sample_frames(len(env_ids))
                self.f[env_ids]=frames

        namespace={"TrackingSim":Base,"task_sampler":sampler,"torch":torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),"exec"),namespace)
        env=namespace["AssetTrackingSim"]()
        sampler.validate_frames(torch.arange(25),env.f)
        self.assertIsNone(env._weighted_reset_env_ids)
        before=env.f.clone()
        env.reset(torch.tensor([1,9,24]))
        sampler.validate_frames(torch.arange(25),env.f)
        untouched=torch.tensor([i for i in range(25) if i not in (1,9,24)])
        self.assertTrue(torch.equal(before[untouched],env.f[untouched]))
        compatible=sampler.first_frames(torch.arange(25),horizon=32)
        env.reset(torch.arange(25),compatible)
        self.assertTrue(torch.equal(env.f,compatible))
        with self.assertRaisesRegex(ValueError,"persistent task lane"):
            env.reset(torch.tensor([0]),compatible[-1:])
        with self.assertRaisesRegex(RuntimeError,"environment identities"):
            env.sample_frames(1)

    def test_rejects_modified_missing_eval_and_unknown_contracts(self):
        b = copy.deepcopy(self.bundle)
        b["task_sampling"]["category_percent"]["ultra/largebox"] = 7
        with self.assertRaisesRegex(ValueError, "modified"):
            build_sampling_plan(b)
        b = copy.deepcopy(self.bundle)
        b["partition"] = "protected_eval"
        with self.assertRaisesRegex(ValueError, "training-only"):
            validate_sampling_contract(b)
        b = copy.deepcopy(self.bundle)
        b["clips"] = [c for c in b["clips"] if c["dataset"] != "grail_sitting"]
        with self.assertRaisesRegex(ValueError, "sixteen"):
            build_sampling_plan(b)
        c = copy.deepcopy(self.bundle["clips"][0])
        c["source_provenance"]["object"] = "unknown"
        with self.assertRaisesRegex(ValueError, "outside"):
            sampling_clip_metadata(c)
        b = copy.deepcopy(self.bundle)
        b["clips"].append(copy.deepcopy(b["clips"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_sampling_plan(b)

    def test_legacy_allocation_unchanged_without_contract(self):
        b = copy.deepcopy(self.bundle)
        b.pop("task_sampling")
        bank = motion_for(b)
        self.assertIsNone(bank.task_sampling)
        n = 4096
        counts = torch.bincount(bank.bin_asset, minlength=len(b["assets"])).tolist()
        residual = n - len(counts)
        exact = [residual * c / sum(counts) for c in counts]
        quota = [1 + int(x) for x in exact]
        order = sorted(range(len(counts)), key=lambda i: (-(exact[i]-int(exact[i])), i))
        for i in order[:n-sum(quota)]:
            quota[i] += 1
        self.assertEqual([s.num_envs for s in allocate_workers(b, bank, n)], quota)

    def test_small_budget_rejects_dropping_retained_assets(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            weighted_asset_quotas(self.bundle, 16)
        self.assertEqual(apportion({"a":9,"b":1}, 10, {"a":1,"b":1}), {"a":9,"b":1})
        self.assertEqual(apportion({"a":999,"b":1}, 10, {"a":1,"b":1}), {"a":9,"b":1})
        for bad in (float("nan"), float("inf"), -1, 0):
            with self.assertRaises(ValueError):
                apportion({"a":bad}, 10)


if __name__ == "__main__":
    unittest.main()
