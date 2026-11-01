"""CPU-auditable contracts for saved GPU-native acceptance evidence.

These checks do not execute physics and cannot create successful GPU evidence.
They recompute probe outcomes from recorded interval traces and bind physical
readbacks to the native asset profile that the caller actually installed.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math

import torch

from .contacts import CHANNELS

RUNTIME_SCHEMA = "hoi_native_cuda_execution_v1"


def physical_profile_identity(profile):
    return hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def capture_native_runtime(recorder, *, compilation=None):
    """Read live backend-buffer devices; call only inside the native process."""
    buffers = {
        "contact_geom": recorder.wp.to_torch(recorder.solver.mjw_data.contact.geom),
        "normal_force": recorder.wp.to_torch(recorder.forces),
        "qpos": recorder.wp.to_torch(recorder.solver.mjw_data.qpos),
        "interval_impulse": recorder.impulse,
    }
    devices = {name: str(value.device) for name, value in buffers.items()}
    if any(value.device.type != "cuda" for value in buffers.values()) or len(set(devices.values())) != 1:
        raise RuntimeError("native acceptance requires contact, force, state and impulse buffers on one CUDA device")
    versions = {}
    for name, distribution in (("torch", "torch"), ("warp", "warp-lang"),
                               ("mujoco", "mujoco"), ("mujoco_warp", "mujoco-warp"), ("newton", "newton")):
        try:
            versions[name] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = str(getattr(importlib.import_module(name), "__version__", "unavailable"))
    device = next(iter(buffers.values())).device
    report = {"schema": RUNTIME_SCHEMA, "backend": "Newton/MuJoCo-Warp", "buffer_devices": devices,
            "verified_from_live_buffers": True, "gpu_name": torch.cuda.get_device_name(device),
            "versions": versions, "torch_cuda_version": torch.version.cuda}
    if compilation is not None:
        report["warp_compilation"] = compilation
    return report


def _tensor(value, shape):
    try:
        tensor = torch.as_tensor(value, dtype=torch.float64)
        if tuple(tensor.shape) != tuple(shape) or not bool(torch.isfinite(tensor).all()):
            return None
        return tensor
    except (ValueError, TypeError, RuntimeError):
        return None


def runtime_blockers(runtime):
    problems = []
    if not isinstance(runtime, dict):
        return ["native CUDA execution evidence is missing"]
    devices = runtime.get("buffer_devices", {})
    required = {"contact_geom", "normal_force", "qpos", "interval_impulse"}
    valid_devices = (isinstance(devices, dict) and set(devices) == required and
                     all(isinstance(value, str) and value.startswith("cuda:") and value[5:].isdigit()
                         for value in devices.values()) and len(set(devices.values())) == 1)
    if (runtime.get("schema") != RUNTIME_SCHEMA or runtime.get("backend") != "Newton/MuJoCo-Warp" or
            runtime.get("verified_from_live_buffers") is not True or not valid_devices or
            not isinstance(runtime.get("gpu_name"), str) or not runtime["gpu_name"]):
        problems.append("contact/force/state/impulse buffers are not verified on the same native CUDA device")
    versions = runtime.get("versions", {})
    if not isinstance(versions, dict) or any(not isinstance(versions.get(name), str) or not versions[name]
                                            for name in ("torch", "warp", "mujoco", "mujoco_warp", "newton")):
        problems.append("native backend package version identities are incomplete")
    return problems


def bound_physics_blockers(physics, environments):
    profile = physics.get("bound_physical_profile")
    if not isinstance(profile, dict):
        return ["native physical readback has no bound asset physical profile"]
    try:
        if physical_profile_identity(profile) != physics.get("bound_physical_profile_sha256"):
            return ["bound native physical profile SHA256 differs"]
    except (ValueError, TypeError):
        return ["bound native physical profile is not finite JSON"]
    problems = []
    fields = (("mass", "mass", (environments,), ()), ("com", "com", (environments, 3), (3,)),
              ("inertia", "inertia", (environments, 3, 3), (3, 3)),
              ("object_friction", "friction", (environments,), ()),
              ("ground_friction", "ground_friction", (), ()))
    for observed_key, bound_key, observed_shape, bound_shape in fields:
        actual, expected = _tensor(physics.get(observed_key), observed_shape), _tensor(profile.get(bound_key), bound_shape)
        if actual is None or expected is None or not torch.allclose(actual, expected.expand_as(actual), atol=1e-6, rtol=1e-5):
            problems.append(f"native {observed_key} readback does not match the bound physical profile in every environment")
    return problems


def probe_trace_blockers(probes, environments, threshold=1.):
    """Recompute force/maxima and require each simulated world to be covered."""
    problems = []
    negative = {("hand_ground_negative" if i == 0 else channel.removesuffix("_object") + "_ground_negative"): i
                for i, channel in enumerate(CHANNELS[:-1])}
    for name in (*CHANNELS, *negative):
        record = probes.get(name, {})
        ticks = record.get("ticks")
        if (type(ticks) is not int or ticks < 1 or record.get("control_dt_s") != .02 or
                record.get("substeps_per_interval") != 4):
            problems.append(f"{name}: complete four-substep interval trace timing is missing")
            continue
        impulse = _tensor(record.get("interval_normal_impulse_Ns"), (ticks, environments, len(CHANNELS)))
        force = _tensor(record.get("interval_mean_normal_force_N"), (ticks, environments, len(CHANNELS)))
        ground = _tensor(record.get("interval_body_ground_impulse_Ns"), (ticks, environments, len(CHANNELS) - 1))
        maximum = _tensor(record.get("maximum_interval_normal_force_N"), (len(CHANNELS),))
        ground_maximum = _tensor(record.get("maximum_body_ground_interval_force_N"), (len(CHANNELS) - 1,))
        if any(value is None for value in (impulse, force, ground, maximum, ground_maximum)):
            problems.append(f"{name}: recorded contact/ground trace does not cover every tick, environment and region")
            continue
        if (bool((impulse < 0).any()) or bool((ground < 0).any()) or
                not torch.allclose(force, impulse / .02, atol=1e-5, rtol=1e-5) or
                not torch.allclose(maximum, force.amax((0, 1)), atol=1e-5, rtol=1e-5) or
                not torch.allclose(ground_maximum, (ground / .02).amax((0, 1)), atol=1e-5, rtol=1e-5)):
            problems.append(f"{name}: force/maxima disagree with the recorded nonnegative interval impulses")
            continue
        if name in negative:
            index = negative[name]
            if not bool(((ground[..., index] / .02).amax(0) > threshold).all()):
                problems.append(f"{name}: intended body-ground contact was not observed in every environment")
            if bool((impulse[..., :-1] != 0).any()):
                problems.append(f"{name}: body-ground contact contaminated a learned body-object channel")
        elif not bool((force[..., CHANNELS.index(name)].amax(0) > threshold).all()):
            problems.append(f"{name}: positive pair contact was not observed in every environment")
    return problems


def native_acceptance_blockers(report):
    environments = report.get("num_envs", report.get("spec", {}).get("num_envs"))
    if type(environments) is not int or environments < 1:
        return ["native report requires a positive integer environment count"]
    problems = runtime_blockers(report.get("native_runtime"))
    problems.extend(compiled_mapping_blockers(report))
    problems.extend(bound_physics_blockers(report.get("physics", {}), environments))
    problems.extend(probe_trace_blockers(report.get("pair_probes", {}), environments))
    problems.extend(region_pose_blockers(report.get("pair_probes", {})))
    problems.extend(support_evidence_blockers(report, environments))
    if report.get("formal_training_started") is not False:
        problems.append("native acceptance must explicitly attest that formal training was not started")
    return problems

def region_pose_blockers(probes):
    """Recompute non-target floor penetration from every compiled collider."""
    from .contacts import BODY_ROLES
    from .probe_pose import SCHEMA
    problems = []
    for index, channel in enumerate(CHANNELS[:-1]):
        for name, negative in ((channel, False), (("hand_ground_negative" if index == 0 else channel.removesuffix("_object") + "_ground_negative"), True)):
            audit = probes.get(name, {}).get("definition", {}).get("region_pose_audit", {})
            definition = probes.get(name, {}).get("definition", {})
            if not negative and definition.get("object_support_selection") == "center_of_one_actual_exposed_convex_component_facet":
                facet = definition.get("object_support_facet", {})
                errors = definition.get("initial_state_readback", {}).get("maximum_errors", {})
                error_values = list(errors.values())
                if (facet.get("schema") != "hoi_native_object_support_facet_v1" or
                        facet.get("point_rule") != "mean_of_coplanar_vertices_in_one_actual_convex_hull" or
                        facet.get("all_worlds_same_orientation") is not True or
                        facet.get("support_plane_error_m", math.inf) > 2e-6 or
                        facet.get("maximum_union_support_excess_m", math.inf) > 2e-6 or
                        definition.get("prescribed_relative_approach_speed_m_s") != .2 or
                        len(error_values) != 9 or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 2e-5 for x in error_values)):
                    problems.append(f"{name}: exposed object facet or prescribed initial state was not verified")
            ids, roles = audit.get("robot_geom_ids"), audit.get("robot_geom_roles")
            if (audit.get("schema") != SCHEMA or audit.get("region") != channel or
                    audit.get("ground_negative") is not negative or not isinstance(ids, list) or not ids or
                    len(set(ids)) != len(ids) or not isinstance(roles, list) or len(roles) != len(ids)):
                problems.append(f"{name}: exact whole-robot compiled geometry pose audit is missing")
                continue
            low = _tensor(audit.get("per_geom_minimum_world_z_m"), (len(ids),))
            target = torch.tensor([role == BODY_ROLES[index] for role in roles])
            if low is None or not bool(target.any()) or not bool((~target).any()):
                problems.append(f"{name}: per-collider floor clearance is incomplete")
                continue
            expected_min, robot_min = float(low[target].min()), float(low.min())
            depth = max(0., -float(low[~target].min()))
            scalar = lambda x: type(x) in (int, float) and math.isfinite(x)
            values = (audit.get("maximum_non_target_ground_penetration_m"), audit.get("target_minimum_world_z_m"), audit.get("robot_minimum_world_z_m"))
            search = audit.get("search", {})
            surface = audit.get("compiled_surface", {})
            if (surface.get("source") != "actual_compiled_collision_surface" or surface.get("geom_id") not in ids or
                    (surface.get("geom_id") in ids and roles[ids.index(surface["geom_id"])] != BODY_ROLES[index]) or
                    not all(scalar(x) for x in values) or abs(values[0]-depth) > 2e-6 or
                    abs(values[1]-expected_min) > 2e-6 or abs(values[2]-robot_min) > 2e-6 or
                    audit.get("maximum_allowed_non_target_ground_penetration_m") != .002 or depth > .002+2e-6 or
                    audit.get("prescribed_penetration_m") != .002 or
                    (negative and abs(expected_min + .002) > 2e-6) or (not negative and robot_min < .002-2e-6) or
                    not scalar(audit.get("maximum_self_penetration_m")) or audit["maximum_self_penetration_m"] > .002+2e-6 or
                    search.get("physics_steps") != 0 or search.get("collisions_disabled") is not False or
                    search.get("joint_limits_preserved") is not True):
                problems.append(f"{name}: probe placement deeply penetrates non-target geometry or its audit is inconsistent")
    return problems


def support_evidence_blockers(report, environments):
    from .scene_supports import omni_support_recipe, digest, SUPPORT_IDS, SUPPORT_ROLES, POSE_MODE
    data = report.get("native_scene_supports")
    if not isinstance(data, dict) or data.get("schema") != "hoi_native_scene_supports_v1":
        return ["explicit native scene-support evidence is missing (including empty declarations)"]
    problems = []
    ids, supports = data.get("declared_support_ids"), data.get("supports")
    if (not isinstance(ids, list) or len(set(ids)) != len(ids) or any(sid not in SUPPORT_IDS for sid in ids) or
            not isinstance(supports, list) or sorted(row.get("support_id", "") for row in supports) != sorted(ids)):
        return ["native support declaration and compiled support entities disagree"]
    recipe = omni_support_recipe()
    if (data.get("geometry_recipe") != recipe or data.get("geometry_recipe_sha256") != digest(recipe) or
            data.get("pose_mode") != POSE_MODE):
        problems.append("native support geometry/material/reset timing differs from the pinned official recipe")
    clips = data.get("clips")
    bound = set()
    sha = lambda value: isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    if not isinstance(clips, list):
        problems.append("support source clip bindings are missing")
    else:
        for clip in clips:
            for binding in clip.get("bindings", []):
                sid = binding.get("support_id")
                if sid not in ids or any(not sha(binding.get(key)) for key in ("binding_sha256", "source_support_sha256", "target_timebase_sha256")):
                    problems.append("support source/timebase/binding identity is missing")
                bound.add(sid)
        if bound != set(ids):
            problems.append("declared supports lack source-bound clip coverage")
    isolation = data.get("isolation_probes", {})
    if not isinstance(isolation, dict) or set(isolation) != set(ids):
        return problems + ["native support isolation probes do not cover declared support entities"]
    topology = report.get("contact", {}).get("topology")
    count = 1 if topology == "one_mujoco_world_per_environment" else environments
    for item in supports:
        sid = item["support_id"]
        instances = item.get("instances", [])
        geoms = item.get("compiled_geom_ids", [])
        if (item.get("geometry_recipe_sha256") != digest(recipe) or item.get("kinematic_verified") is not True or
                len(instances) != count or len(geoms) != count or [row.get("geom_id") for row in instances] != geoms or
                len(set(geoms)) != len(geoms) or not item.get("native_body_flags")):
            problems.append(f"{sid}: independent kinematic compiled cuboid evidence is incomplete")
        role_map = report.get("contact", {}).get("geom_roles", [])
        if any(type(g) is not int or not 0 <= g < len(role_map) or role_map[g] != SUPPORT_ROLES[SUPPORT_IDS.index(sid)] for g in geoms):
            problems.append(f"{sid}: support collider IDs do not map to the actual independent compiled partner role")
        for instance in instances:
            size, friction = _tensor(instance.get("size_m"), (3,)), _tensor(instance.get("geom_friction"), (3,))
            if (size is None or not torch.allclose(size, torch.tensor(recipe["size_m"], dtype=torch.float64), atol=1e-7, rtol=0) or
                    friction is None or abs(float(friction[0])-recipe["dynamic_friction"]) > 1e-6 or
                    instance.get("role") != SUPPORT_ROLES[SUPPORT_IDS.index(sid)] or
                    type(instance.get("body_id")) is not int or instance["body_id"] <= 0):
                problems.append(f"{sid}: actual cuboid size/material/partner role differs from its recipe")
        cases = isolation.get(sid, {})
        if set(cases) != {"object_support", "body_support"}:
            problems.append(f"{sid}: both body-support and object-support native controls are required")
            continue
        for name, probe in cases.items():
            ticks = probe.get("ticks")
            if type(ticks) is not int or ticks < 1 or probe.get("control_dt_s") != .02 or probe.get("substeps_per_interval") != 4:
                problems.append(f"{sid}/{name}: support interval timing is incomplete")
                continue
            learned = _tensor(probe.get("interval_normal_impulse_Ns"), (ticks, environments, len(CHANNELS)))
            event = _tensor(probe.get("interval_support_impulse_Ns"), (ticks, environments))
            if (learned is None or event is None or bool((learned != 0).any()) or bool((event < 0).any()) or
                    not bool(((event/.02).amax(0) > 1.).all()) or probe.get("passed") is not True):
                problems.append(f"{sid}/{name}: true support contact is missing in some world or leaked into the 16 learned channels")
    if ids:
        alignment = data.get("alignment", {})
        if any(type(alignment.get(key)) is not int or alignment[key] < 1 for key in ("reset_count", "step_count")) or any(
                type(alignment.get(key)) not in (float, int) or not math.isfinite(alignment[key]) or not 0 <= alignment[key] <= 2e-5
                for key in ("max_position_error_m", "max_quaternion_error")):
            problems.append("native support reset-held alignment was not verified over real steps")
        reset = data.get("reference_reset_audit", {})
        if reset.get("environment_ids") != list(range(environments)) or set(reset.get("supports", {})) != set(ids):
            problems.append("support reference-reset readback does not cover every world and entity")
        else:
            for sid, item in reset["supports"].items():
                expected = _tensor(item.get("expected_pose_xyzw"), (environments, 7))
                actual = _tensor(item.get("actual_pose_xyzw"), (environments, 7))
                velocity = _tensor(item.get("actual_velocity"), (environments, 6))
                if (expected is None or actual is None or velocity is None or
                        not torch.allclose(actual[:, :3], expected[:, :3], atol=2e-5, rtol=0) or
                        not torch.allclose((actual[:, 3:]*expected[:, 3:]).sum(-1).abs(), torch.ones(environments,dtype=torch.float64), atol=2e-5, rtol=0) or
                        not bool((velocity.abs() <= 2e-5).all())):
                    problems.append(f"{sid}: reference reset pose/zero velocity differs from actual native readback")
    if data.get("passed") is not True:
        problems.append("native scene-support calibration did not pass")
    return problems


def compiled_mapping_blockers(report):
    """Validate actual collider IDs/roles; absent theoretical links are allowed."""
    from .contacts import BODY_ROLES, LINK_ROLES
    contact = report.get("contact", {})
    names, roles, links = (contact.get(key) for key in ("geom_names", "geom_roles", "geom_canonical_links"))
    if (not isinstance(names, list) or not names or not isinstance(roles, list) or not isinstance(links, list) or
            len(names) != len(roles) or len(names) != len(links)):
        return ["raw compiled collider identity/role/link arrays are missing or misaligned"]
    problems = []
    for geom, (role, link) in enumerate(zip(roles, links)):
        if role in BODY_ROLES and (link not in LINK_ROLES or LINK_ROLES[link] != role):
            problems.append(f"compiled collider {geom} region role disagrees with its canonical active link")
    for channel, role in zip(CHANNELS[:-1], BODY_ROLES):
        if contact.get("region_collision_geom_counts", {}).get(channel) != roles.count(role):
            problems.append(f"{channel}: reported collider count differs from actual compiled role mapping")
    for side in ("left", "right"):
        count = sum(link == side+"_ankle_roll_link" and role == LINK_ROLES[side+"_ankle_roll_link"] for link,role in zip(links,roles))
        if contact.get("foot_collision_geom_counts", {}).get(side) != count:
            problems.append(f"{side} mesh-foot collider count differs from actual canonical mapping")
    first_geometry = None
    for name, probe in report.get("pair_probes", {}).items():
        if not isinstance(probe, dict) or "definition" not in probe:
            continue
        audit = probe["definition"].get("region_pose_audit")
        if not isinstance(audit, dict):
            continue
        ids, assigned = audit.get("robot_geom_ids"), audit.get("robot_geom_roles")
        if (not isinstance(ids,list) or not isinstance(assigned,list) or len(ids)!=len(assigned) or
                any(type(g) is not int or not 0 <= g < len(roles) for g in ids)):
            problems.append(f"{name}: pose audit collider IDs are not actual compiled IDs")
            continue
        if assigned != [roles[g] for g in ids]:
            problems.append(f"{name}: pose audit relabelled an actual compiled collider")
        if contact.get("topology") == "one_mujoco_world_per_environment" and not set(g for g,r in enumerate(roles) if r in BODY_ROLES).issubset(ids):
            problems.append(f"{name}: whole-robot pose audit omitted an active body-region collider")
        if first_geometry is None:
            first_geometry=ids
        elif ids != first_geometry:
            problems.append(f"{name}: whole-robot collider inventory differs across probes")
    return problems

