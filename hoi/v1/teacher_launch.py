"""Fail closed before training if the full teacher resource probe differs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import bundle_identity, source_identity
from .privileged import TEACHER_PROFILE
from .train import read_config


def verify_source(expected_sha):
    identity = source_identity()
    if identity["content_sha256"] != expected_sha:
        raise ValueError("teacher source identity changed")
    return identity


def validate_probe(report, *, source, config, data_identity, device, collector_devices, startup):
    required = {
        "passed": True, "optimizer_steps": 0, "formal_training_started": False,
        "scope": "full_training_bundle_resource_validation", "source_identity": source,
        "production_config": config.to_dict(), "data_identity": data_identity,
        "initialization": {"mode": "explicit_from_scratch"},
        "worker_cpu_threads": 3, "collector_devices": collector_devices,
        "worker_startup_concurrency": startup,
    }
    if config.actor_observation_profile != TEACHER_PROFILE:
        raise ValueError("teacher launcher requires the explicit teacher configuration")
    for key, expected in required.items():
        if key not in report or report[key] != expected or type(report[key]) is not type(expected):
            raise ValueError(f"teacher resource probe mismatch: {key}")
    if report.get("probe", {}).get("expected_groups") != 273:
        raise ValueError("teacher resource probe must cover all 273 train assets")
    resources = report.get("resources", {})
    for key, expected in (("groups", 273), ("num_envs", config.num_envs), ("device", device),
                          ("collector_devices", collector_devices), ("worker_startup_concurrency", min(startup, 273))):
        if resources.get(key) != expected:
            raise ValueError(f"teacher resource allocation mismatch: {key}")
    if report.get("real_start_mix", {}).get("all_static_groups_present") is not True:
        raise ValueError("teacher resource probe did not exercise every asset in real replay")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collector-devices", nargs="+")
    parser.add_argument("--worker-startup-concurrency", type=int)
    args = parser.parse_args(argv)
    source = verify_source(args.source_sha)
    if args.report:
        if not args.bundle or not args.collector_devices or not args.worker_startup_concurrency:
            parser.error("probe verification requires bundle and exact collector configuration")
        from .evaluate import _read_bundle_header
        header = _read_bundle_header(args.bundle)
        if header["partition"] != "train" or header["scope"]["is_subset"]:
            raise ValueError("teacher launcher requires the complete train bundle")
        config, _ = read_config(Path(__file__).resolve().parents[2] / "configs/hoi_v1_privileged_teacher.json")
        validate_probe(json.loads(args.report.read_text(encoding="utf-8")), source=source, config=config,
                       data_identity=bundle_identity(header), device=args.device,
                       collector_devices=args.collector_devices, startup=args.worker_startup_concurrency)
    print(json.dumps({"source_content_sha256": source["content_sha256"], "probe_verified": args.report is not None}))


if __name__ == "__main__":
    main()
