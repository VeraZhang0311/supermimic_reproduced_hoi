"""Training-only, asset-balanced reference-state initialization.

The adaptive pools are based on real episode outcomes, never WM predictions or
protected evaluation.  Raw return is deliberately not used: its scale depends
on how much reference remains after an RSI start.  A success/quality observation
must cover a meaningful amount of that remaining reference before it can count
as mastery.  Failure-time credit also includes the preceding one-second bin.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

import torch


@dataclass(frozen=True)
class RSIConfig:
    uniform_fraction: float = .5
    hard_fraction: float = .3
    mastered_fraction: float = .2
    warmup_iterations: int = 500
    ramp_iterations: int = 200
    ema_rate: float = .1
    min_mastery_visits: int = 10
    mastery_success: float = .8
    mastery_quality: float = .8
    minimum_mastery_steps: int = 25
    bin_frames: int = 50
    failure_floor: float = .05
    maximum_hard_ratio: float = 10.

    def validate(self):
        fractions = (self.uniform_fraction, self.hard_fraction, self.mastered_fraction)
        if any(x < 0 for x in fractions) or abs(sum(fractions)-1) > 1e-9:
            raise ValueError("RSI mixture fractions must be nonnegative and sum to one")
        if not 0 < self.ema_rate <= 1 or not 0 <= self.mastery_success <= 1 or not 0 <= self.mastery_quality <= 1:
            raise ValueError("Invalid RSI EMA or mastery thresholds")
        if min(self.min_mastery_visits, self.minimum_mastery_steps, self.bin_frames) < 1:
            raise ValueError("RSI visit and time thresholds must be positive")
        if self.warmup_iterations < 0 or self.ramp_iterations < 1 or self.failure_floor <= 0 or self.maximum_hard_ratio < 1:
            raise ValueError("Invalid RSI schedule or failure priority")
        return self


class AdaptiveRSI:
    """One sampler for all assets, or the fixed asset of a native worker.

    ``sample_frames`` returns global reference frame and global bin IDs.  All
    slots in ``update`` must refer to actual sampled episode starts.  ``quality``
    is the fraction of observed control intervals meeting both robot and object
    tracking criteria, or an equivalent Boolean.  ``elapsed_steps`` records the
    actual observed duration; omitting it means mastery is not certified.
    """
    schema = "hoi_tables_adaptive_rsi_v1"

    def __init__(self, motion, asset_index=None, *, config=None, seed=0):
        if not bool(getattr(motion, "allowed_for_training", getattr(motion, "for_training", False))) or motion.partition != "train":
            raise ValueError("Adaptive RSI only accepts the frozen training partition")
        self.motion, self.config = motion, (config or RSIConfig()).validate()
        self.device = torch.device(motion.bin_start.device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.asset_index = asset_index
        all_bins = torch.arange(motion.num_bins, device=self.device)
        assets = motion.clip_asset_ids[motion.bin_clip]
        ends = motion.clip_end[motion.bin_clip]
        # A last reference state has no outgoing control interval.
        usable = motion.bin_start < ends
        if asset_index is not None:
            usable &= assets == asset_index
        self.bins = all_bins[usable]
        self.assets = assets
        self.asset_values = torch.unique(assets[self.bins], sorted=True)
        if not len(self.bins):
            raise ValueError("No usable training RSI bins for this asset")
        self._eligible = usable
        self._remaining = (ends-motion.bin_start).long()
        self.iteration = 0
        self.visits = torch.zeros(motion.num_bins, device=self.device, dtype=torch.long)
        self.success_ema = torch.zeros(motion.num_bins, device=self.device)
        self.quality_ema = torch.zeros_like(self.success_ema)
        self.failure_ema = torch.zeros_like(self.success_ema)
        self.failure_observations = torch.zeros_like(self.visits)
        self.mastery_visits = torch.zeros_like(self.visits)
        self.last_sample_counts = {}
        identity = torch.stack((motion.bin_start, ends, assets), -1).detach().cpu().numpy().tobytes()
        self.identity = hashlib.sha256(identity).hexdigest()

    def _ramp(self, iteration):
        return min(1., max(0., (iteration-self.config.warmup_iterations)/self.config.ramp_iterations))

    def probabilities(self, *, iteration=None, asset_index=None):
        """Return global bin IDs and probabilities, conditional on one asset."""
        it = self.iteration if iteration is None else int(iteration)
        asset = self.asset_index if asset_index is None else asset_index
        bins = self.bins if asset is None else self.bins[self.assets[self.bins] == asset]
        if not len(bins):
            raise ValueError("Asset has no eligible training bins")
        if asset is None and len(self.asset_values) > 1:
            # Equal mass per physical table, independently of clip duration.
            pieces = [self.probabilities(iteration=it, asset_index=int(a)) for a in self.asset_values]
            return torch.cat([p[0] for p in pieces]), torch.cat([p[1]/len(pieces) for p in pieces])
        uniform = torch.full((len(bins),), 1/len(bins), device=self.device)
        hard = torch.zeros_like(uniform)
        seen = (self.failure_observations[bins] > 0) & (self.failure_ema[bins] > 0)
        hard[seen] = self.failure_ema[bins][seen] + self.config.failure_floor
        hard = self._capped_distribution(hard, uniform)
        good = ((self.mastery_visits[bins] >= self.config.min_mastery_visits)
                & (self.success_ema[bins] >= self.config.mastery_success)
                & (self.quality_ema[bins] >= self.config.mastery_quality)
                & (self._remaining[bins] >= self.config.minimum_mastery_steps)).float()
        good = good/good.sum() if bool(good.sum() > 0) else uniform
        ramp = self._ramp(it)
        hard_fraction, good_fraction = self.config.hard_fraction*ramp, self.config.mastered_fraction*ramp
        result = (1-hard_fraction-good_fraction)*uniform + hard_fraction*hard + good_fraction*good
        return bins, result/result.sum()

    def _capped_distribution(self, weight, uniform):
        if not bool(weight.sum() > 0):
            return uniform
        probability = weight/weight.sum()
        # Redistribute capped excess into the remaining bins; a single bad bin
        # cannot absorb the entire hard-pool allocation in a large dataset.
        cap = min(1., self.config.maximum_hard_ratio/len(weight))
        for _ in range(len(weight)+1):
            excessive = probability > cap+1e-7
            if not bool(excessive.any()):
                break
            probability[excessive] = cap
            free = probability < cap-1e-7
            if not bool(free.any()):
                break
            probability[free] += (1-probability.sum())/free.sum()
        return probability/probability.sum()

    def sample_frames(self, n, iteration=None):
        if type(n) is not int or n < 1:
            raise ValueError("RSI sample count must be a positive integer")
        if iteration is not None:
            if int(iteration) < 0:
                raise ValueError("RSI iteration must be nonnegative")
            self.iteration = int(iteration)
        # Exact equal allocation up to a single sample, randomly rotating the
        # remainder so tiny/native smoke batches also cover both assets.
        asset_order = self.asset_values[torch.randperm(len(self.asset_values), generator=self.generator, device=self.device)]
        choices = []
        self.last_sample_counts = {}
        for i, asset in enumerate(asset_order):
            count = n//len(asset_order) + int(i < n % len(asset_order))
            if not count:
                continue
            bins, probability = self.probabilities(asset_index=int(asset))
            choices.append(bins[torch.multinomial(probability, count, replacement=True, generator=self.generator)])
            self.last_sample_counts[int(asset)] = count
        bins = torch.cat(choices)
        bins = bins[torch.randperm(n, generator=self.generator, device=self.device)]
        start = self.motion.bin_start[bins]
        end = torch.minimum(start+self.config.bin_frames, self.motion.clip_end[self.motion.bin_clip[bins]])
        jitter = torch.rand(n, device=self.device, generator=self.generator)
        frame = start + (jitter*(end-start)).long()
        return frame, bins

    def _ema(self, ids, values, destination, observations):
        count = torch.bincount(ids, minlength=len(destination))
        total = torch.zeros_like(destination).scatter_add_(0, ids, values.float())
        touched = count > 0
        average = total[touched]/count[touched]
        rate = 1-(1-self.config.ema_rate)**count[touched].float()
        rate = torch.where(observations[touched] == 0, torch.ones_like(rate), rate)
        destination[touched] += rate*(average-destination[touched])
        return count

    def update(self, bin_ids, success, quality, *, failure_frames=None, elapsed_steps=None):
        bins = torch.as_tensor(bin_ids, device=self.device, dtype=torch.long).flatten()
        success = torch.as_tensor(success, device=self.device).flatten().float()
        quality = torch.as_tensor(quality, device=self.device).flatten().float()
        if bins.shape != success.shape or bins.shape != quality.shape:
            raise ValueError("RSI outcomes must have one bin, success and quality per episode")
        if not len(bins):
            return
        if bool(((bins < 0) | (bins >= len(self.visits))).any()) or not bool(self._eligible[bins].all()):
            raise ValueError("Outcome refers to a protected, incompatible or terminal RSI bin")
        if not bool(torch.isfinite(quality).all()) or bool(((quality < 0) | (quality > 1)).any()) or bool(((success != 0) & (success != 1)).any()):
            raise ValueError("RSI success must be binary and quality must lie in [0,1]")
        elapsed, frames = None, None
        if elapsed_steps is not None:
            elapsed = torch.as_tensor(elapsed_steps, device=self.device).flatten()
            if elapsed.shape != bins.shape or not bool(torch.isfinite(elapsed).all()) or bool((elapsed < 0).any()):
                raise ValueError("RSI elapsed steps must match sampled episodes")
        if failure_frames is not None:
            frames = torch.as_tensor(failure_frames, device=self.device, dtype=torch.long).flatten()
            if frames.shape != bins.shape:
                raise ValueError("Failure frame must index the training reference library")
            failed = success == 0
            if bool(((frames[failed] < 0) | (frames[failed] >= self.motion.num_frames)).any()):
                raise ValueError("Actual failures require an in-range failure frame")
            if bool(((frames[~failed] < -1) | (frames[~failed] >= self.motion.num_frames)).any()):
                raise ValueError("Successful outcomes allow an in-range frame or -1 sentinel")
            observed_clip = self.motion.bin_clip[self.motion.frame_bin[frames[failed]]]
            if not bool((observed_clip == self.motion.bin_clip[bins[failed]]).all()):
                raise ValueError("Failure frame and RSI start must belong to the same source clip")
        previous_visits = self.visits.clone()
        self._ema(bins, success, self.success_ema, previous_visits)
        count = self._ema(bins, quality, self.quality_ema, previous_visits)
        self.visits += count
        self._ema(bins, 1-success, self.failure_ema, self.failure_observations)
        self.failure_observations += count
        if elapsed is not None:
            certified = elapsed >= self.config.minimum_mastery_steps
            self.mastery_visits += torch.bincount(bins[certified], minlength=len(self.visits))
        if frames is not None:
            failed = success == 0
            current = self.motion.frame_bin[frames[failed]]
            clip = self.motion.bin_clip[current]
            before = torch.maximum(self.motion.clip_start[clip], frames[failed]-self.config.bin_frames)
            preceding = self.motion.frame_bin[before]
            # Dedupe the same bin within each failure event.
            credit = torch.cat((current, preceding[preceding != current]))
            credit = credit[self._eligible[credit]]
            if len(credit):
                count = self._ema(credit, torch.ones_like(credit), self.failure_ema, self.failure_observations)
                self.failure_observations += count

    def state_dict(self):
        return {"schema": self.schema, "identity": self.identity, "config": asdict(self.config),
                "asset_index": self.asset_index, "iteration": self.iteration,
                "generator_state": self.generator.get_state().clone(),
                **{name: getattr(self, name).clone() for name in ("visits", "success_ema", "quality_ema", "failure_ema", "failure_observations", "mastery_visits")}}

    def load_state_dict(self, state):
        if (state.get("schema"), state.get("identity"), state.get("config"), state.get("asset_index")) != (self.schema, self.identity, asdict(self.config), self.asset_index):
            raise ValueError("RSI checkpoint does not match the frozen training reference and schedule")
        for name in ("visits", "success_ema", "quality_ema", "failure_ema", "failure_observations", "mastery_visits"):
            value = torch.as_tensor(state[name], device=self.device)
            target = getattr(self, name)
            if value.shape != target.shape or not bool(torch.isfinite(value).all()):
                raise ValueError("Invalid RSI checkpoint statistics")
            if bool((value < 0).any()) or (target.is_floating_point() and bool((value > 1).any())):
                raise ValueError("Out-of-range RSI checkpoint statistics")
        for name in ("visits", "success_ema", "quality_ema", "failure_ema", "failure_observations", "mastery_visits"):
            getattr(self, name).copy_(state[name])
        self.iteration = int(state["iteration"])
        self.generator.set_state(state["generator_state"].cpu())
