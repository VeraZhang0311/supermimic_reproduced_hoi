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
         "rewards.py", "runtime.py", "shape.py", "train.py", "capacity.py", "resume.py")

# The first protected trainer already saved complete resume metadata. This
# migration changes only the two modules which authenticate older checkpoints.
_PREVIOUS_PROTECTED_LOADERS = {
    "contract.py": "98e8d8a2dc3571b03cad1a314b0733f1c778e31a9b068741f14746f9da2255e0",
    "resume.py": "3b65ab2367c927dd811fb3d72d7e75f197ef4ea72823988f82aca64dda7cd81a",
}


def compatible_protected_implementation(stored, current):
    if stored == current:
        return True
    if not isinstance(stored, dict) or stored.keys() != current.keys() or stored.get("schema") != SCHEMA:
        return False
    files = current["implementation_sha256"]
    if not _PREVIOUS_PROTECTED_LOADERS.keys() <= files.keys():
        return False
    return stored.get("implementation_sha256") == {**files, **_PREVIOUS_PROTECTED_LOADERS}


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
    if compatible_protected_implementation(stored, current):
        return True
    # Both earlier runs predate resumable PPO and configurable EPA scratch.
    # Permit only the exact published source bundles through this migration.
    if not isinstance(stored, dict):
        return False
    digest = hashlib.sha256(json.dumps(stored, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    allowed = {
        "resmimic": {"471e9be4e34aa5d94ec9b569999d30944229d6a1794ae36915831a836bc47e9a",
                     "f75d4b55c0585799476c6b6faf7700ad9e2255bb01498b48c01b539ee437609c"},
        "hoi_v1_1": {"c83896ec3d7f826ae2fbffa9441eda6eee11107c93d5108ca3ee8b2b5c924048"},
    }
    if digest not in allowed[reward_profile]:
        return False
    return all(stored["implementation_sha256"][key] == current["implementation_sha256"][key]
               for key in stored["implementation_sha256"] if key not in ("contract.py", "train.py", "env.py"))
