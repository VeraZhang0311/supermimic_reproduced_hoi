"""Versioned experiment settings; original SuperMimic update defaults retained."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import math


@dataclass(frozen=True)
class TrainConfig:
    num_envs: int = 4096
    iterations: int = 20000
    buffer_steps: int = 1000
    utd: float = 0.4
    horizon: int = 32
    goal_frames: int = 10  # Legacy critic/reference path; actor slots belong to observation_profile.
    hist: int = 10
    wm_window: int = 8
    wm_batches: int = 10
    wm_batch: int = 2048
    pi_batch: int = 4096
    lr: float = 2e-4
    gamma: float = 0.99
    lam: float = 0.95
    action_noise: float = 0.1
    critic_epochs: int = 5
    critic_minibatches: int = 4
    warmup_steps: int = 64
    wm_only_iters: int = 500
    lr_decay_iters: int = 250000
    ckpt_every: int = 500
    eval_every: int = 5000
    log_every: int = 50
    gate_every: int = 50
    seed: int = 0
    amp: str = "bf16"
    no_critic: bool = False
    experiment_kind: str = "formal"
    variant: str = "hoi_v1"
    observation_profile: str = "v1"
    domain_randomization_profile: str = "v1"

    def validate(self):
        from .variants import validate_variant
        validate_variant(self.variant)
        from .observations import observation_spec
        from .randomization import randomization_contract
        observation_spec(self.observation_profile)
        randomization_contract(self.domain_randomization_profile)
        if type(self.seed) is not int or self.seed < 0 or type(self.no_critic) is not bool:
            raise ValueError("seed must be a nonnegative integer and no_critic must be boolean")
        for key in ("num_envs", "iterations", "buffer_steps", "horizon", "goal_frames", "hist",
                    "wm_window", "wm_batches", "wm_batch", "pi_batch", "critic_epochs",
                    "critic_minibatches", "warmup_steps", "lr_decay_iters", "ckpt_every",
                    "log_every", "gate_every", "eval_every"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.wm_only_iters) is not int or self.wm_only_iters < 0 or self.buffer_steps < self.wm_window:
            raise ValueError("invalid WM warmup/window/buffer settings")
        for key in ("lr", "utd", "gamma", "lam", "action_noise"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite real number")
        if (self.goal_frames, self.hist) != (10, 10):
            raise ValueError("v1 preserves K=10 and history=10")
        if not (0 < self.gamma < 1 and 0 <= self.lam <= 1 and self.lr > 0 and self.utd > 0):
            raise ValueError("invalid optimization settings")
        if self.action_noise < 0 or self.amp not in ("none", "bf16"):
            raise ValueError("invalid action noise or AMP mode")
        if self.experiment_kind not in ("formal", "smoke"):
            raise ValueError("experiment_kind must be formal or smoke")
        return self

    def lr_at(self, iteration):
        import math

        if iteration >= self.lr_decay_iters:
            return self.lr * 0.03
        return self.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * iteration / self.lr_decay_iters)))

    def to_dict(self):
        return asdict(self)


def json_identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def bundle_identity(bundle):
    """Portable checkpoint identity, including all training feature contracts."""
    assets = []
    for asset in bundle["assets"]:
        assets.append({key: asset[key] for key in (
            "asset_id", "object_asset", "exact_scale", "collision_hash", "native_asset_sha256", "physical_profile")})
        assets[-1]["geometry_sha256"] = asset["geometry_cache"]["sha256"]
    return {"bundle_content_sha256": bundle["bundle_content_sha256"], "partition": bundle["partition"],
            "manifest": bundle["manifest"], "timebase": bundle["timebase"],
            "recipes": bundle.get("recipes"), "robot_fk": bundle.get("robot_fk"),
            "native_robot_asset": bundle.get("native_robot_asset"),
            "reference_kinematics": bundle.get("reference_kinematics"), "assets": assets,
            "hands_geometry_sha256": bundle["hands"]["geometry_cache"]["sha256"]}


def fresh_directory(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def source_identity():
    """Hash executable local sources, including uncommitted HOI implementation."""
    root = Path(__file__).resolve().parents[2]
    selected = []
    for folder in ("supermimic", "hoi"):
        selected.extend((root / folder).rglob("*.py"))
    selected.extend((root / "configs").glob("hoi_v1*.json"))
    selected.extend((root / "cluster").glob("hoi_v1*.sh"))
    files = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(selected)}
    return {"files": files, "content_sha256": json_identity(files)}


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
