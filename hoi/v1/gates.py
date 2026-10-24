"""Evidence gates for native HOI physics and the first policy update.

Missing evidence closes a gate. A CPU test, submitted job, or an uncalibrated
threshold proposal is never relabelled as a successful native experiment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch


class GateError(RuntimeError):
    pass


def _read(value):
    if isinstance(value, (str, Path)):
        raw = Path(value).read_bytes()
        return json.loads(raw), hashlib.sha256(raw).hexdigest()
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return value, hashlib.sha256(raw).hexdigest()


def _finite(value):
    try:
        tensor = torch.as_tensor(value, dtype=torch.float64)
        return tensor.numel() > 0 and bool(torch.isfinite(tensor).all())
    except (TypeError, ValueError, RuntimeError):
        return False


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _native_geometry_evidence(physics, prefix):
    """Require a collision-engine hull export and an independent distance audit.

    Comparing cached triangles with a second copy of the same triangles cannot
    establish agreement with the collision engine. The graph provenance and
    native GJK query are independent parts of this requirement.
    """
    blockers = []
    export = physics.get("native_convex_hull_export", {})
    sides = ("left", "right")
    count = export.get("object_hulls", 0)
    hand_counts = export.get("hand_hulls", {})
    if (export.get("source") != "mjModel.mesh_graph" or export.get("all_geometries_graph_backed") is not True or
        type(count) is not int or count < 1 or not _sha256(export.get("object_evidence_sha256")) or
        any(type(hand_counts.get(side)) is not int or hand_counts[side] < 1 or
            not _sha256(export.get("hand_evidence_sha256", {}).get(side)) for side in sides)):
        blockers.append(f"{prefix}: native collision-engine convex-hull graph export is incomplete")
    audit = physics.get("independent_native_geometry_queries", {})
    tolerance = audit.get("tolerance_m")
    valid_tolerance = (type(tolerance) in (int, float) and math.isfinite(tolerance) and 0 < tolerance <= 5e-5)
    if (audit.get("passed") is not True or audit.get("api") != "mujoco.mj_geomDistance convex collision distance" or
        not valid_tolerance):
        blockers.append(f"{prefix}: independent native convex-distance audit is missing or too permissive")
    samples, covered = audit.get("samples", []), set()
    if not isinstance(samples, list) or not samples:
        blockers.append(f"{prefix}: independent native geometry query samples are missing")
    else:
        for sample in samples:
            side, pose = sample.get("hand"), sample.get("pose")
            native, cached, error = (sample.get(key) for key in ("native_distance_m", "cached_surface_distance_m", "abs_error_m"))
            queried, total = sample.get("cache_pairs_queried"), sample.get("cache_pairs_total")
            valid = (side in sides and type(pose) is int and pose >= 0 and
                     all(type(x) in (int, float) and math.isfinite(x) for x in (native, cached, error)) and
                     native > 0 and cached >= 0 and error >= 0 and valid_tolerance and error <= tolerance and
                     abs(abs(native - cached) - error) <= 1e-9 and type(queried) is int and type(total) is int and
                     0 < queried <= total and type(count) is int and type(hand_counts.get(side)) is int and
                     total == count * hand_counts[side])
            if not valid:
                blockers.append(f"{prefix}: independent native geometry sample is invalid or fails tolerance")
            else:
                covered.add((side, pose))
        if any(len({pose for hand, pose in covered if hand == side}) < 2 for side in sides):
            blockers.append(f"{prefix}: independent native geometry audit requires two separated poses per hand")
    fk = physics.get("robot_fk_comparison", {})
    valid = fk.get("passed") is True
    for measured, threshold in (("max_link_position_error_m", "position_tolerance_m"),
                                ("max_rotation_matrix_error", "rotation_tolerance")):
        value, limit = fk.get(measured), fk.get(threshold)
        valid &= (type(value) in (int, float) and type(limit) in (int, float) and
                  math.isfinite(value) and math.isfinite(limit) and 0 <= value <= limit <= 5e-5 and limit > 0)
    if not valid:
        blockers.append(f"{prefix}: native robot FK does not match the reference kinematics")
    return blockers


def validate_native_evidence(report_or_path, *, expected_physics=None, required_asset_ids=None,
                             expected_bundle_sha256=None):
    if report_or_path is None:
        return {"schema_version": "hoi_v1_native_gate_v1", "passed": False,
                "blockers": ["no native evidence supplied"], "evidence_sha256": None, "asset_ids": [], "status": "blocked"}
    payload, digest = _read(report_or_path)
    reports = payload if isinstance(payload, list) else payload.get("reports", [payload])
    blockers, assets = [], set()
    if not reports:
        blockers.append("no native asset reports supplied")
    for number, report in enumerate(reports):
        prefix = f"report[{number}]"
        physics, contact = report.get("physics", {}), report.get("contact", {})
        asset_id = report.get("asset_id", report.get("spec", {}).get("asset_id"))
        if asset_id is None:
            blockers.append(f"{prefix}: missing static asset identity")
        else:
            asset_id = int(asset_id)
            if asset_id in assets:
                blockers.append(f"{prefix}: duplicate static asset report {asset_id}")
            assets.add(asset_id)
        observed_flags = physics.get("physics", {})
        desired = expected_physics or {"hand_collision": True, "self_collision": True}
        for key, expected in desired.items():
            if observed_flags.get(key) != expected:
                blockers.append(f"{prefix}: compiled physics {key} does not match {expected!r}")
        for key, expected in (("control_dt", .02), ("substeps", 4), ("integration_dt", .005)):
            if physics.get(key) != expected:
                blockers.append(f"{prefix}: native {key} must be {expected}")
        for key in ("mass", "com", "inertia", "object_friction", "ground_friction", "gravity"):
            if key not in physics or not _finite(physics[key]):
                blockers.append(f"{prefix}: missing/nonfinite native {key} readback")
        try:
            com, inertia = torch.as_tensor(physics.get("com", [])), torch.as_tensor(physics.get("inertia", []))
            if com.shape[-1:] != (3,) or inertia.shape[-2:] != (3, 3):
                blockers.append(f"{prefix}: invalid native COM/inertia shape")
            elif not torch.allclose(inertia, inertia.transpose(-1, -2)) or not bool((torch.linalg.eigvalsh(inertia) > 0).all()):
                blockers.append(f"{prefix}: native inertia is not symmetric positive definite")
            for key in ("object_friction", "ground_friction"):
                if not bool((torch.as_tensor(physics.get(key, [])) >= 0).all()):
                    blockers.append(f"{prefix}: negative native {key}")
        except (TypeError, ValueError, RuntimeError):
            blockers.append(f"{prefix}: malformed native physical readback")
        if _finite(physics.get("mass", [])) and not bool((torch.as_tensor(physics.get("mass", [])) > 0).all()):
            blockers.append(f"{prefix}: object mass is not positive")
        if physics.get("velocity_reference_point") != "link_origin":
            blockers.append(f"{prefix}: object velocity reference point has not been verified")
        if physics.get("object_pose_origin") != "link":
            blockers.append(f"{prefix}: object pose origin has not been verified")
        gravity = torch.as_tensor(physics.get("gravity", []))
        if gravity.numel() == 0 or gravity.shape[-1:] != (3,) or not bool((gravity[..., 2] < 0).all()):
            blockers.append(f"{prefix}: downward native gravity missing")
        if not physics.get("collision_hash") or not physics.get("flattened_object_usd_sha256"):
            blockers.append(f"{prefix}: effective collider provenance is missing")
        if physics.get("compiled_collision_geometry_verified_against_bundle") is not True:
            blockers.append(f"{prefix}: compiled collider and cached reward geometry are not verified equal")
        blockers.extend(_native_geometry_evidence(physics, prefix))
        geometry = report.get("geometry", {})
        if report.get("palm_selection_verified", geometry.get("palm_selection_verified")) is not True:
            blockers.append(f"{prefix}: actual palm-surface probe selection is not calibrated")
        hand = report.get("compiled_hand_audit", {})
        if hand.get("enabled") is not True:
            blockers.append(f"{prefix}: compiled rubber-hand collisions are not verified ON")
        for side in ("left", "right"):
            h = hand.get("hands", {}).get(side, {})
            if not h.get("collision_geom_count", 0) or not h.get("allowed_object_pairs", 0) or not h.get("cpu_probe_contacts"):
                blockers.append(f"{prefix}: {side} hand collider/pair mask probe missing")
        self_collision = report.get("compiled_self_collision_audit", {})
        if self_collision.get("enabled") is not True or self_collision.get("allowed_nonadjacent_robot_geom_pairs", 0) < 1:
            blockers.append(f"{prefix}: compiled nonadjacent self-collision audit missing")
        expected_channels = ["left_hand_object", "right_hand_object", "object_ground"]
        if (contact.get("schema") != "hoi_v1_pair_normal_impulse_v1" or list(contact.get("channels", [])) != expected_channels or
            contact.get("interval") != "(t,t+1]" or contact.get("num_substeps") != 4 or contact.get("substep_dt") != .005):
            blockers.append(f"{prefix}: pair labels do not cover the four physics substeps of (t,t+1]")
        if contact.get("normal_force_api") != "mujoco_warp.contact_force(to_world_frame=False)[0]":
            blockers.append(f"{prefix}: pair-specific normal-force instrumentation is unverified")
        if contact.get("use_cuda_graph") is not False or physics.get("use_cuda_graph") is not False:
            blockers.append(f"{prefix}: per-substep recorder requires the verified non-graph execution path")
        threshold = contact.get("threshold_N")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold != 1.:
            blockers.append(f"{prefix}: v1 actual-pair force threshold must be the versioned 1N definition")
            threshold = math.inf
        probes = report.get("pair_probes", {})
        if report.get("native_pair_calibration_complete") is not True or probes.get("passed") is not True:
            blockers.append(f"{prefix}: native pair calibration is incomplete")
        for channel, index in (("left_hand_object", 0), ("right_hand_object", 1), ("object_ground", 2)):
            values = probes.get(channel, {}).get("maximum_interval_normal_force_N", [])
            if probes.get("checks", {}).get(channel) is not True or len(values) != 3 or not _finite(values) or values[index] <= threshold:
                blockers.append(f"{prefix}: {channel} positive pair probe is missing")
        negative = probes.get("hand_ground_negative", {})
        negative_pair, hand_ground = negative.get("maximum_interval_normal_force_N", []), negative.get("maximum_hand_ground_interval_force_N", [])
        if (probes.get("checks", {}).get("hand_ground_not_mislabelled") is not True or len(negative_pair) != 3 or len(hand_ground) != 2 or
            not _finite(negative_pair + hand_ground) or max(hand_ground, default=0) <= threshold or max(negative_pair[:2], default=math.inf) != 0):
            blockers.append(f"{prefix}: hand-ground negative identity control did not pass")
        if report.get("zero_action_smoke_completed") is not True:
            blockers.append(f"{prefix}: native smoke did not complete")
        if expected_bundle_sha256 is not None and report.get("bundle_content_sha256") != expected_bundle_sha256:
            blockers.append(f"{prefix}: native evidence belongs to another bundle or legacy fixture")
    if required_asset_ids is not None:
        missing = set(int(x) for x in required_asset_ids) - assets
        if missing:
            blockers.append(f"missing native geometry groups: {sorted(missing)}")
    return {"schema_version": "hoi_v1_native_gate_v1", "passed": not blockers, "blockers": blockers,
            "evidence_sha256": digest, "asset_ids": sorted(assets), "status": "passed" if not blockers else "blocked"}


@dataclass(frozen=True)
class WMThresholdProposal:
    """Explicit startup proposal; this is not an empirically calibrated threshold.

    It asks the learned normalized object error to improve on simple baselines
    and pair probabilities to improve on prevalence-only predictions. Counts
    >0 and positive precision/recall are elementary coverage checks, not a
    claim that one positive example establishes task success.
    """
    name: str = "train_only_startup_relative_baselines_v1"
    calibration_status: str = "proposal_not_empirically_calibrated"
    require_horizons: tuple = (1, 8, 32)
    baseline_error_ratio: float = 1.
    require_baseline_improvement: bool = True
    require_pair_brier_improvement: bool = True
    require_contact_phase_coverage: bool = True


def validate_wm_evidence(report, proposal=None):
    proposal = proposal if proposal is not None else WMThresholdProposal()
    if isinstance(proposal, dict):
        proposal = WMThresholdProposal(**proposal)
    if not math.isfinite(proposal.baseline_error_ratio) or not 0 < proposal.baseline_error_ratio <= 1:
        raise ValueError("relative baseline error ratio must be finite, positive, and at most one")
    blockers = []
    if report.get("finite") is not True:
        blockers.append("WM holdout contains nonfinite diagnostics or has not run")
    if report.get("scope") != "training_reference_bins_reserved_from_wm_fit_not_eval80_or_test40":
        blockers.append("WM policy gate requires reserved train-only real transitions")
    if report.get("real_executed_actions") is not True:
        blockers.append("WM diagnostics are not bound to real executed action sequences")
    horizon_reports = report.get("horizon_reports", {})
    for horizon in proposal.require_horizons:
        value = horizon_reports.get(str(horizon), {})
        if value.get("valid_samples", 0) < 1:
            blockers.append(f"no valid {horizon}-step real heldout sequences")
            continue
        metrics = [value.get(key) for key in ("object_normalized_error", "zero_increment_normalized_error", "constant_velocity_normalized_error")]
        if not all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in metrics):
            blockers.append(f"missing/nonfinite {horizon}-step object and baseline errors")
        elif proposal.require_baseline_improvement:
            learned, zero, cv = metrics
            if learned > proposal.baseline_error_ratio * min(zero, cv) or (learned == min(zero, cv) and learned > 0):
                blockers.append(f"{horizon}-step object prediction has not improved over both simple baselines")
        if proposal.require_contact_phase_coverage:
            phases = value.get("phases", {})
            for name in ("no_contact", "contact", "onset", "release"):
                phase = phases.get(name, {})
                if phase.get("valid_samples", 0) < 1:
                    blockers.append(f"{horizon}-step {name} phase has no real holdout coverage")
                elif proposal.require_baseline_improvement:
                    metrics = [phase.get(key) for key in ("object_normalized_error", "zero_increment_normalized_error", "constant_velocity_normalized_error")]
                    if not all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in metrics):
                        blockers.append(f"{horizon}-step {name} phase lacks finite baseline comparisons")
                    elif metrics[0] > proposal.baseline_error_ratio * min(metrics[1:]) or (metrics[0] == min(metrics[1:]) and metrics[0] > 0):
                        blockers.append(f"{horizon}-step {name} phase has not improved over both baselines")
    contact = report.get("contact_diagnostics", {})
    for channel in range(3):
        for key in ("positive_counts", "negative_counts"):
            values = contact.get(key, [])
            if len(values) != 3 or values[channel] < 1:
                blockers.append(f"actual pair channel {channel} lacks {key}")
        for key in ("precision", "recall"):
            values = contact.get(key, [])
            if len(values) != 3 or not _finite(values) or not 0 < values[channel] <= 1:
                blockers.append(f"actual pair channel {channel} has no positive {key}")
        brier, baseline = contact.get("brier_per_channel", []), contact.get("prevalence_baseline_brier", [])
        if len(brier) != 3 or len(baseline) != 3 or not _finite(brier + baseline):
            blockers.append(f"actual pair channel {channel} calibration diagnostics missing")
        elif proposal.require_pair_brier_improvement and brier[channel] >= baseline[channel]:
            blockers.append(f"actual pair channel {channel} probabilities do not improve on prevalence-only Brier")
    return {"schema_version": "hoi_v1_wm_policy_gate_v1", "passed": not blockers, "blockers": blockers,
            "threshold_proposal": asdict(proposal), "status": "passed_startup_proposal" if not blockers else "blocked",
            "interpretation": "Train-only startup proposal; passing does not establish calibrated thresholds or simulator HOI success"}


def require_native_evidence(*args, **kwargs):
    result = validate_native_evidence(*args, **kwargs)
    if not result["passed"]:
        raise GateError("Native HOI gate closed: " + "; ".join(result["blockers"]))
    return result


def require_wm_evidence(*args, **kwargs):
    result = validate_wm_evidence(*args, **kwargs)
    if not result["passed"]:
        raise GateError("WM policy gate closed: " + "; ".join(result["blockers"]))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bind actual native report bytes into one auditable multi-asset evidence manifest")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--native-report", action="append", required=True, type=Path,
                        help="actual native.json report; repeat for all static geometry groups")
    parser.add_argument("--output", required=True, type=Path, help="fresh evidence directory; blocked reports are retained")
    args = parser.parse_args(argv)
    try:
        from .prepare import BUNDLE_SCHEMA, _digest_json
        bundle, bundle_sha = _read(args.bundle)
        content_sha = bundle.pop("bundle_content_sha256", None)
        if bundle.get("schema_version") != BUNDLE_SCHEMA or content_sha != _digest_json(bundle):
            raise ValueError("bundle schema/content hash changed")
        bundle["bundle_content_sha256"] = content_sha
        inputs, reports = [], []
        for path in args.native_report:
            report, digest = _read(path)
            reports.extend(report["reports"] if "reports" in report else [report])
            inputs.append({"path": str(path.resolve()), "sha256": digest})
        payload = {"schema_version": "hoi_v1_native_evidence_manifest_v1",
                   "bundle": {"path": str(args.bundle.resolve()), "sha256": bundle_sha, "bundle_content_sha256": content_sha},
                   "inputs": inputs, "reports": reports}
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        evidence_path = output / "native_evidence.json"
        with evidence_path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        required = sorted(set(clip["asset_id"] for clip in bundle["clips"]))
        gate = validate_native_evidence(evidence_path, expected_physics={"hand_collision": True, "self_collision": True},
                                        required_asset_ids=required, expected_bundle_sha256=content_sha)
        with (output / "gate.json").open("x", encoding="utf-8") as stream:
            json.dump(gate, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"evidence": str(evidence_path), "gate": str(output / "gate.json"),
                          "passed": gate["passed"], "blockers": gate["blockers"]}, allow_nan=False))
        return 0 if gate["passed"] else 1
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Native evidence aggregation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
