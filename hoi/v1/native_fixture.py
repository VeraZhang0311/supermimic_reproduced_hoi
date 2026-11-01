"""Read-only first-stage native cooking fixtures, explicitly outside training."""
from __future__ import annotations
from pathlib import Path
import torch

FIXTURE_SCHEMA = "hoi_curated_native_cooking_fixture_v1"
DEFERRED_INERTIA_MODE = "native_cooking_only_uniform_density"


def validate_cooking_mass_profile(profile):
    import math
    if set(profile) != {"mass","friction","ground_friction"}:
        raise ValueError("deferred cooking profile requires only explicit mass and friction; no placeholder COM/inertia")
    values = [float(profile[k]) for k in ("mass","friction","ground_friction")]
    if not all(math.isfinite(v) for v in values) or values[0] <= 0 or min(values[1:]) < 0:
        raise ValueError("invalid explicit cooking mass/friction")
    return values



def read_curated_cooking_fixture(path):
    from hoi.data.curated import load_json, load_curated_clip, digest, sha
    from hoi.data.native_assets import BUILD_SCHEMA, exact_surface
    from hoi.data.manifest import robot_joint_names
    from .prepare import native_robot_asset_binding
    from .native import validate_physical_profile
    path = Path(path).resolve()
    fixture = load_json(path)
    from hoi.data.native_repairs import FIXTURE_SCHEMA as REPAIR_FIXTURE_SCHEMA, fixture_repair_context
    if fixture.get("schema") not in (FIXTURE_SCHEMA,REPAIR_FIXTURE_SCHEMA) or fixture.get("content_sha256") != digest({k:v for k,v in fixture.items() if k != "content_sha256"}):
        raise ValueError("curated native cooking fixture schema/content identity mismatch")
    repair = fixture_repair_context(fixture) if fixture.get("schema") == REPAIR_FIXTURE_SCHEMA else None
    if repair is None and (fixture.get("repair_binding") or fixture.get("asset",{}).get("repair_binding")):
        raise ValueError("repair provenance requires its explicit v2 fixture schema")
    asset = fixture["asset"]
    if asset.get("asset_id") != 0:
        raise ValueError("single-asset cooking fixture must use local asset_id zero")
    if asset.get("inertia_mode") == DEFERRED_INERTIA_MODE:
        validate_cooking_mass_profile(asset["physical_profile"])
    elif asset.get("inertia_mode") is not None:
        raise ValueError("unknown fixture inertia mode")
    else:
        validate_physical_profile(asset["physical_profile"])
    build_identity = asset["build_receipt"]
    if sha(build_identity["path"]) != build_identity["sha256"]:
        raise ValueError("curated native build receipt bytes changed")
    build = load_json(build_identity["path"])
    if build.get("schema") != BUILD_SCHEMA or build.get("status") != "native_cooking_and_geometry_validation_required" or build.get("training_ready") is not False:
        raise ValueError("cooking fixture needs an explicit unverified first-stage build")
    for key in ("native_asset", "native_asset_sha256", "physical_profile", "object_geometry_sha256", "inertia_mode"):
        if key == "object_geometry_sha256" and repair is not None:
            continue
        if asset.get(key) != build.get(key):
            raise ValueError("cooking fixture/build mismatch: " + key)
    if sha(asset["native_asset"]) != asset["native_asset_sha256"]:
        raise ValueError("curated native cooking input bytes changed")
    if build.get("collision_approximation") not in ("convexHull", "convexDecomposition") or not build.get("physical_evidence"):
        raise ValueError("explicit source physics and collision cooking recipe required")
    if fixture["source_record"]["object_geometry_sha256"] != asset["object_geometry_sha256"]:
        raise ValueError("cooking fixture source label/native geometry mismatch")
    recipe_identity=fixture["robot_recipe"]
    if sha(recipe_identity["path"]) != recipe_identity["sha256"]:
        raise ValueError("cooking fixture robot recipe bytes changed")
    robot_recipe=load_json(recipe_identity["path"])
    if any(fixture[key] != robot_recipe.get(key) for key in ("native_robot_asset","reference_kinematics","robot_geometry_sha256")):
        raise ValueError("cooking fixture robot identity differs from its bound recipe")
    robot = fixture["native_robot_asset"]
    if native_robot_asset_binding(robot["path"],robot["sha256"]) != robot or "foot_asset_manifest" not in robot:
        raise ValueError("cooking fixture requires the exact audited hand/foot robot binding")
    virtual = fixture["reference_kinematics"]
    if len(virtual["body_names"]) != 39 or len(set(virtual["body_names"])) != 39 or virtual["dof_names"] != robot_joint_names() or sha(virtual["urdf"]["path"]) != virtual["urdf"]["sha256"]:
        raise ValueError("cooking fixture reference kinematics identity mismatch")
    clip = load_curated_clip(fixture["source_record"],fixture.get("path_maps",()))
    if clip["label_metadata"]["robot_geometry_sha256"] != fixture.get("robot_geometry_sha256"):
        raise ValueError("cooking fixture label/native robot geometry mismatch")
    expected_source_surface = repair[1]["source_label_surface_sha256"] if repair is not None else build.get("exact_surface_sha256")
    if digest(exact_surface(clip["object_geometry"]["shapes"])) != expected_source_surface:
        raise ValueError("cooking fixture source surface differs from label geometry")
    return fixture, build, clip


def load_curated_cooking_fixture(path, kin, device):
    from types import SimpleNamespace
    from supermimic.mdp import MDP
    from supermimic.robot import default_joint_pos_list
    from hoi.data.curated import sha
    from .motion import TrainingMotionBank
    from .scene_supports import validate_native_scene_supports
    fixture, build, clip = read_curated_cooking_fixture(path)
    bank = TrainingMotionBank([{"name":clip["name"],"asset_id":0,"reference":clip["reference"],
        "expected_contact":clip["expected_contact"],"expected_valid":clip["expected_valid"]}],kin,device=device,partition="validation")
    bank.scene_supports = [clip["scene_supports"]]
    bank.bound_scene_supports = [clip["bound_scene_supports"]]
    validate_native_scene_supports(bank)
    mdp = SimpleNamespace(human=MDP(kin,bank,torch.tensor(default_joint_pos_list(),device=device)),max_step_reward=.15)
    source = {"fixture_manifest":str(Path(path).resolve()),"fixture_manifest_sha256":sha(path),
        "fixture_content_sha256":fixture["content_sha256"],"scope":"curated_native_cooking_fixture_not_training",
        "case":fixture["source_record"],"reference_kinematics":fixture["reference_kinematics"],
        "robot_geometry_sha256":fixture["robot_geometry_sha256"],
        "source_object_geometry_sha256":fixture["asset"]["object_geometry_sha256"],
        "source_exact_surface_sha256":build["exact_surface_sha256"],
        "physical_evidence":build["physical_evidence"],"collision_approximation":build["collision_approximation"],
        "build_receipt":fixture["asset"]["build_receipt"],
        "geometry_stage":"source_input_for_native_cook; not_a_verified_compiled_union",
        "training_ready":False,"inertia_mode":fixture["asset"].get("inertia_mode")}
    if fixture.get("repair_binding"):
        from hoi.data.native_repairs import fixture_repair_context
        binding,proposal,*_=fixture_repair_context(fixture)
        source.update(repair_binding=fixture["repair_binding"],source_exact_surface_sha256=proposal["source_label_surface_sha256"],
            cooking_input_geometry_sha256=build["object_geometry_sha256"],cooking_input_surface_sha256=build["exact_surface_sha256"],
            original_source_build=proposal["original_build"])
    return bank,mdp,fixture["asset"],source,fixture["native_robot_asset"]


def finish_geometry_only_cooking(env, source, output, *, hand_audit, self_audit, seed=0):
    """Write only a cooking receipt after export; never calibrate or integrate.

    Both complete and deferred source mass profiles stop here. Full pair and
    physical acceptance belongs to the final compiled-geometry bundle.
    """
    import json
    from hoi.data.curated import sha, load_json
    from .native_acceptance import capture_native_runtime
    if not env.cooking_only or source.get("scope") != "curated_native_cooking_fixture_not_training":
        raise ValueError("geometry-only completion requires an explicit curated cooking fixture")
    output=Path(output)
    export_path=output/"compiled_geometry/assets.json"
    exported=load_json(export_path)
    if exported.get("schema") != "hoi_v1_compiled_geometry_recipe_v1" or len(exported.get("assets",[])) != 1:
        raise ValueError("cooking completion requires the actual single-asset native hull export")
    native=exported["assets"][0]
    if not native.get("native_compiled_surface_sha256") or not native.get("source_exact_surface_sha256"):
        raise ValueError("cooking completion must distinguish source and native compiled surface identities")
    report={"schema":"hoi_native_geometry_cooking_report_v1","source":source,
        "native_runtime":capture_native_runtime(env.contacts),"physics_import_readback":env.physics_report,
        "compiled_geometry_export":str(export_path),"compiled_geometry_export_sha256":sha(export_path),
        "native_compiled_surface_sha256":native["native_compiled_surface_sha256"],
        "source_exact_surface_sha256":native["source_exact_surface_sha256"],
        "compiled_hand_audit":hand_audit,"compiled_self_collision_audit":self_audit,
        "compiled_foot_audit":env.compiled_foot_audit,"formal_training_started":False,
        "num_envs":env.N,"seed":seed,"physics_steps":0,"native_pair_calibration_complete":False,
        "zero_action_smoke_completed":False,"training_ready":False,
        "next_stage":("integrate closed compound hulls using explicit mass, then bind the final full profile and run complete native bundle gate"
                      if source.get("inertia_mode") else "bind exported native hulls and full physical profile into the final bundle, then run complete native gate")}
    with (output/"cooking.json").open("x",encoding="utf-8") as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n")
    env.close()
    return report
