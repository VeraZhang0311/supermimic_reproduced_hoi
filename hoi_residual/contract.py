"""Bind the complete residual learning path to checkpoint provenance.

Training orchestration is included because rollout history, reset timing and
normalizer updates are part of the algorithm. Even logging-only trainer edits
therefore require a new implementation identity. Evaluation/reporting code is
excluded so that read-only reporting improvements can inspect old checkpoints.
The shared robot/GMT code is independently bound by the GMT source contract.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "hoi_residual_learning_implementation_v1"
FILES = ("assets.py", "contract.py", "data.py", "distributed.py", "env.py", "features.py",
         "native_checks.py", "policy.py", "ppo.py", "randomization.py",
         "rewards.py", "runtime.py", "shape.py", "train.py")


def implementation_contract(reward_profile="resmimic"):
    directory = Path(__file__).resolve().parent
    files = FILES
    if reward_profile == "hoi_v1_1":
        files += ("v11_reward.py", "v11_contacts.py")
    elif reward_profile != "resmimic":
        raise ValueError(f"unknown training reward profile: {reward_profile}")
    return {"schema": SCHEMA,
            "implementation_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                      for name in files}}


def compatible_for_evaluation(stored, reward_profile):
    current = implementation_contract(reward_profile)
    if stored == current:
        return True
    # 1509634 used the exact baseline from commit 541f7bd. Its eval path
    # still uses the same reward and step semantics; bind the full old digest.
    if reward_profile != "resmimic" or not isinstance(stored, dict):
        return False
    digest = hashlib.sha256(json.dumps(stored, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != "471e9be4e34aa5d94ec9b569999d30944229d6a1794ae36915831a836bc47e9a":
        return False
    return all(stored["implementation_sha256"][key] == current["implementation_sha256"][key]
               for key in FILES if key not in ("contract.py", "train.py", "env.py"))
