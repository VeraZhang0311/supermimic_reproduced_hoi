"""Bind the complete residual learning path to checkpoint provenance.

Training orchestration is included because rollout history, reset timing and
normalizer updates are part of the algorithm. Even logging-only trainer edits
therefore require a new implementation identity. Evaluation/reporting code is
excluded so that read-only reporting improvements can inspect old checkpoints.
The shared robot/GMT code is independently bound by the GMT source contract.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

SCHEMA = "hoi_residual_learning_implementation_v1"
FILES = ("assets.py", "contract.py", "data.py", "distributed.py", "env.py", "features.py",
         "native_checks.py", "policy.py", "ppo.py", "randomization.py",
         "rewards.py", "runtime.py", "shape.py", "train.py")


def implementation_contract():
    directory = Path(__file__).resolve().parent
    return {"schema": SCHEMA,
            "implementation_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                      for name in FILES}}
