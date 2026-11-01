"""Lossless actual-model dumps and portable source-spec diagnostics.

The binary is the real host compiled model, never a reconstruction. Newton's
per-world GPU properties may differ from its host template and are saved
separately, so neither the XML nor MJB is advertised as a simulator checkpoint.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

# Arrays used by CPU pose/collision queries. No dynamics is executed here.
COLLISION_ARRAYS = (
    "body_parentid", "body_weldid", "body_pos", "body_quat", "body_jntadr", "body_jntnum",
    "jnt_type", "jnt_bodyid", "jnt_axis", "jnt_pos", "jnt_qposadr", "jnt_limited", "jnt_range",
    "geom_bodyid", "geom_type", "geom_dataid", "geom_pos", "geom_quat", "geom_size",
    "geom_contype", "geom_conaffinity", "geom_margin", "geom_gap", "geom_condim",
    "mesh_vertadr", "mesh_vertnum", "mesh_vert", "mesh_faceadr", "mesh_facenum", "mesh_face",
    "pair_geom1", "pair_geom2", "exclude_signature", "qpos0",
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def portable_xml(source, destination):
    """Copy official generated XML and relocate only external resource paths."""
    source, destination = Path(source), Path(destination)
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    resources = []
    for item in root.findall(".//*[@file]"):
        file = Path(item.get("file"))
        folder = ""
        if compiler is not None:
            folder = compiler.get("texturedir" if item.tag == "texture" else "meshdir", "")
        resolved = file if file.is_absolute() else source.parent / folder / file
        if not resolved.is_file():
            raise FileNotFoundError(f"Generated native XML resource is missing: {resolved}")
        sha = _sha(resolved)
        target = destination.parent / "compiled_model_assets" / (sha + resolved.suffix)
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            shutil.copyfile(resolved, target)
        item.set("file", target.relative_to(destination.parent).as_posix())
        resources.append({"source": str(resolved), "path": str(target.relative_to(destination.parent)), "sha256": sha})
    if compiler is not None:
        for key in ("meshdir", "texturedir", "assetdir"):
            compiler.attrib.pop(key, None)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return resources


def export_actual_model(model, output, generated_xml, *, recorder=None, scene_snapshot=None):
    import mujoco
    if scene_snapshot is not None and recorder is None:
        raise ValueError("A scene snapshot must bind the same live native recorder dump")
    output = Path(output)
    binary, portable = output / "compiled_model.mjb", output / "compiled_model.xml"
    if binary.exists() or portable.exists():
        raise FileExistsError("Actual model diagnostics require fresh output files")
    # Native serializer preserves the real compiled arrays and topology.
    mujoco.mj_saveModel(model, str(binary))
    restored = mujoco.MjModel.from_binary_path(str(binary))
    binary_equal = all(np.array_equal(np.asarray(getattr(model,key)),np.asarray(getattr(restored,key))) for key in COLLISION_ARRAYS)
    if not binary_equal:
        raise RuntimeError("Actual compiled MJB did not round-trip collision state exactly")
    resources = portable_xml(generated_xml, portable)
    xml_model = mujoco.MjModel.from_xml_path(str(portable))
    differences = {}
    for key in COLLISION_ARRAYS:
        actual, written = np.asarray(getattr(model,key)),np.asarray(getattr(xml_model,key))
        if actual.shape != written.shape:
            differences[key] = {"actual_shape":list(actual.shape),"xml_shape":list(written.shape)}
        elif not np.allclose(actual,written,atol=2e-6,rtol=1e-7):
            differences[key] = {"maximum_absolute_difference":float(np.max(np.abs(actual-written)))}
    manifest = {"schema":"hoi_actual_native_model_dump_v1","mujoco_version":mujoco.__version__,
        "binary":{"path":str(binary),"sha256":_sha(binary),"origin":"mujoco.mj_saveModel(actual_solver_mj_model)",
                  "collision_arrays_exact_roundtrip":binary_equal},
        "generated_xml":{"path":str(generated_xml),"sha256":_sha(generated_xml),
            "origin":"Newton SolverMuJoCo official save_to_mjcf; generated MjSpec before runtime per-world updates"},
        "portable_xml":{"path":str(portable),"sha256":_sha(portable),"relocated_resources":resources,
            "host_collision_array_comparison":differences, "host_collision_arrays_agree":not differences},
        "model_counts":{key:int(getattr(model,key)) for key in ("nq","nv","nbody","njnt","ngeom","nmesh","npair","nexclude")},
        "host_opt":{"disableflags":int(model.opt.disableflags),"enableflags":int(model.opt.enableflags)},
        "scope":"CPU host template and generated spec for collision diagnostics; not a complete GPU simulation checkpoint",
        "physics_steps":0}
    if recorder is not None:
        import torch
        # Preserve live per-world state/property arrays separately, without
        # overwriting the CPU model or implying host/GPU properties coincide.
        arrays = {}
        for container_name, container, fields in (
            ("data",recorder.solver.mjw_data,("qpos","qvel","geom_xpos","geom_xmat")),
            ("model",recorder.solver.mjw_model,("body_mass","body_inertia","body_ipos","body_iquat",
                "geom_contype","geom_conaffinity","geom_friction","geom_margin","geom_gap","jnt_range"))):
            for key in fields:
                if hasattr(container,key):
                    value=getattr(container,key)
                    if hasattr(value,"numpy"):
                        arrays[container_name+"."+key]=torch.as_tensor(np.array(value.numpy(),copy=True))
        if scene_snapshot is not None:
            from .scene_context import validate_scene_state
            from .scene_supports import SUPPORT_IDS, omni_support_recipe, digest
            actual = scene_snapshot["support_pose_xyzw"]
            reference = scene_snapshot["reference_support_pose_xyzw"]
            present = scene_snapshot["support_present"]
            origins = scene_snapshot["environment_origins"]
            frames = scene_snapshot["reference_frames"]
            validate_scene_state(actual, present)
            validate_scene_state(reference, present)
            n = len(actual)
            declared = scene_snapshot["declared_support_ids"]
            if (actual.ndim != 3 or origins.shape != (n,3) or frames.shape != (n,) or
                    not bool(torch.isfinite(origins).all()) or frames.dtype not in (torch.int32,torch.int64) or
                    len(set(declared)) != len(declared) or any(sid not in SUPPORT_IDS for sid in declared)):
                raise ValueError("Native scene snapshot environment/slot/frame identity differs")
            for slot,sid in enumerate(SUPPORT_IDS):
                if sid not in declared and bool(present[:,slot].any()):
                    raise ValueError("An undeclared support cannot be present in native scene snapshot")
            for key in ("support_pose_xyzw","reference_support_pose_xyzw","support_present","environment_origins","reference_frames"):
                arrays["scene."+key] = scene_snapshot[key].detach().cpu().clone()
            manifest["scene_snapshot"] = {"schema":"hoi_actual_native_scene_snapshot_v1",
                "support_ids":list(SUPPORT_IDS),"declared_support_ids":list(declared),"num_envs":n,
                "geometry_recipe":omni_support_recipe(),"geometry_recipe_sha256":digest(omni_support_recipe()),
                "actual_pose_source":"observed native root_link_pose_w at the same stopped export",
                "reference_pose_source":"saved sampled reset-frame support pose held for episode",
                "pose_coordinates":"environment-relative world axes; xyzw quaternion; add saved environment_origins for global world",
                "reference_frame_scope":"current global motion-bank frame; reference support pose remains sampled reset pose",
                "timing":"same stopped model/live-array dump; no physics, FK or reference resampling",
                "capture_phase":scene_snapshot.get("capture_phase","initial_model_export"),
                "geom_pose_consistency":"renderer must validate saved support root poses against saved geom transforms"}
        arrays_path=output/"compiled_model_live_arrays.pt"
        torch.save(arrays,arrays_path)
        manifest["live_gpu_arrays"]={"path":str(arrays_path),"sha256":_sha(arrays_path),
            "fields":{key:list(value.shape) for key,value in arrays.items()},
            "origin":"actual mjw_model/mjw_data arrays at export time"}
        manifest["contact_mapping"]=recorder.manifest()
        manifest["contact_mapping"]["geom_environment_ids"]=list(recorder.geom_envs)
    (output/"compiled_model_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return manifest



def enable_smoke_visual_capture(env, output, generated_xml):
    """Arm one post-advance diagnostic only after native pair/support probes.

    The hook runs before the collector can reset. It adds no integration steps
    and is forbidden for geometry-only cooking. Rendering happens after close.
    """
    if env.__dict__.get("cooking_only",False) or env.asset.get("inertia_mode") is not None:
        raise ValueError("smoke visual capture cannot add physics to geometry-only cooking")
    if env.__dict__.get("active_pair_probe") is not None or "_post_advance_capture" in env.__dict__:
        raise ValueError("smoke visual capture must be armed once after pair/support probes")
    output=Path(output)
    if output.exists():
        raise FileExistsError("smoke visual capture requires a fresh output directory")
    status={"enabled":True,"captured":False,"extra_physics_steps":0,
            "phase":"first_existing_smoke_control_interval_after_probes_before_autoreset",
            "manifest":str(output/"compiled_model_manifest.json"),
            "rendering_report":str(output/"render/rendering.json")}
    def capture(current):
        import torch
        if current is not env or current.__dict__.get("active_pair_probe") is not None:
            raise RuntimeError("smoke visual callback ran outside its armed smoke stage")
        output.mkdir(parents=True,exist_ok=False)
        arrival=torch.minimum(current.f+1,current.motion.clip_end[current.motion.clip_id[current.f]])
        export_actual_model(current.contacts.solver.mj_model,output,generated_xml,recorder=current.contacts,
            scene_snapshot={**current.support_state(),"environment_origins":current.scene.env_origins,
                "reference_frames":arrival,"declared_support_ids":list(current.supports),
                "capture_phase":status["phase"]})
        status.update(captured=True,manifest_sha256=_sha(output/"compiled_model_manifest.json"),
                      captured_existing_control_interval=1)
    env.__dict__["_post_advance_capture"]=capture
    return status
