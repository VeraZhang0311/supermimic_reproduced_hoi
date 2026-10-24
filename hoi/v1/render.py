"""Offline display of v1 saved transforms; no policy, FK or physics stepping.

Actual visuals keep their own colours and the reference is translucent blue.
Object artwork and an explicit collision-geometry display are separate choices;
the latter is never silently substituted for an unprovided visual mesh.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np

FPS = 50
REFERENCE_RGBA = (.08, .43, 1., .35)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def load_saved(path, expected_sha256=None):
    import torch
    from .evaluate import ROLLOUT_SCHEMA
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("saved rollout SHA256 differs from its evaluation report")
    import io
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if payload.get("schema_version") != ROLLOUT_SCHEMA:
        raise ValueError("renderer requires the explicit HOI v1 recorded-rollout schema")
    states, control, meta = payload["states"], payload["control"], payload["metadata"]
    T = len(control["action"])
    names = list(meta["body_names"])
    if len(names) != 39 or len(set(names)) != 39 or T < 1:
        raise ValueError("renderer requires original 39 saved body names and nonempty controls")
    values = {key: states[key].detach().cpu().numpy() for key in ("body_pos", "body_rot", "o", "reference_source630")}
    n = T + 1
    if (values["body_pos"].shape != (n, 39, 3) or values["body_rot"].shape != (n, 39, 4) or
            values["o"].shape != (n, 13) or values["reference_source630"].shape != (n, 630)):
        raise ValueError("saved display fields must retain all T+1 state/reference frames")
    timing = payload["time"].detach().cpu().numpy().astype(np.float64)
    if timing.shape != (n,) or not np.allclose(timing, np.arange(n) / FPS, rtol=0, atol=1e-10):
        raise ValueError("recorded control clock must be the unmodified complete 50 Hz grid")
    if any(not np.isfinite(value).all() for value in values.values()):
        raise ValueError("recorded transforms contain nonfinite values")
    ref = values["reference_source630"]
    data = {"actual_pos": values["body_pos"], "actual_quat": values["body_rot"][..., [3, 0, 1, 2]],
            "reference_pos": ref[:, 84:201].reshape(n, 39, 3),
            "reference_quat": ref[:, 201:357].reshape(n, 39, 4)[..., [3, 0, 1, 2]],
            "actual_object_pos": values["o"][:, :3], "actual_object_quat": values["o"][:, 3:7],
            "reference_object_pos": ref[:, 71:74], "reference_object_quat": ref[:, 74:78][:, [3, 0, 1, 2]],
            "time": timing, "names": names, "metadata": meta, "metrics": payload.get("metrics", {}),
            "rollout_sha256": digest, "source": str(path.resolve()), "control_steps": T}
    for key in ("actual_quat", "reference_quat", "actual_object_quat", "reference_object_quat"):
        if not np.allclose(np.linalg.norm(data[key], axis=-1), 1., atol=1e-3, rtol=0):
            raise ValueError(f"invalid saved quaternion: {key}")
    return data


def fixed_camera_parameters(data):
    """One camera enclosing both complete trajectories, with no alignment."""
    positions = np.concatenate([data[key].reshape(-1, 3) for key in
                                ("actual_pos", "reference_pos", "actual_object_pos", "reference_object_pos")])
    lo, hi = positions.min(0), positions.max(0)
    center = (lo + hi) / 2
    radius = np.linalg.norm((hi - lo) / 2) + .35
    return {"lookat": center.tolist(), "distance": max(3., float(radius / math.sin(math.radians(21)) * 1.1)),
            "azimuth": 135., "elevation": -17., "shared_world": True, "trajectory_alignment": "none"}


def _triangles_ascii_usda(path):
    """Strict fallback for identity-transform, metric-Z-up flattened USDs."""
    text = Path(path).read_text(encoding="utf-8")
    if not re.search(r"metersPerUnit\s*=\s*1(?:\.0*)?\b", text) or not re.search(r'upAxis\s*=\s*"Z"', text):
        raise ValueError("ASCII USD fallback requires declared metric Z-up geometry")
    if "visibility = \"invisible\"" in text or "xformOp:transform" in text:
        raise ValueError("USD visibility/matrix transforms require the Pixar USD reader")
    identities = {"orient": (1., 0., 0., 0.), "translate": (0., 0., 0.), "scale": (1., 1., 1.)}
    operations = re.findall(r"\bxformOp:([^\s=]+)\s*=", text)
    decoded = re.findall(r"(?:quat[d-f]?|double3|float3)\s+xformOp:(\w+)\s*=\s*\(([^\n]+)\)", text)
    if len(operations) != len(decoded):
        raise ValueError("unsupported USD transforms require the Pixar USD reader")
    for kind, value in decoded:
        parsed = tuple(float(x) for x in value.split(","))
        if kind not in identities or parsed != identities[kind]:
            raise ValueError("nonidentity USD transforms require the Pixar USD reader")
    triangles = []
    for match in re.finditer(r'\bdef\s+Mesh\s+"[^"]+"\s*(?:\([^{}]*\)\s*)?\{', text):
        start, depth, end = match.end(), 1, match.end()
        while depth and end < len(text):
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        if depth:
            raise ValueError("unbalanced ASCII USD mesh block")
        block = text[start:end - 1]
        def array(name, dtype):
            value = re.findall(r"\b" + name + r"\s*=\s*\[([^\]]+)\]", block)
            if len(value) != 1:
                raise ValueError(f"USD mesh lacks one explicit {name} array")
            return np.fromstring(re.sub(r"[(),]", " ", value[0]), sep=" ", dtype=dtype)
        p, counts, faces = array("points", np.float64).reshape(-1, 3), array("faceVertexCounts", np.int64), array("faceVertexIndices", np.int64)
        if not np.all(counts == 3) or len(faces) != 3 * len(counts) or faces.min() < 0 or faces.max() >= len(p):
            raise ValueError("USD display requires explicit valid triangle faces")
        triangles.append(p[faces.reshape(-1, 3)])
    if not triangles:
        raise ValueError("USD contains no explicit triangle meshes")
    return np.concatenate(triangles), {"reader": "strict_identity_ascii_USD", "mesh_count": len(triangles)}


def _triangles_usd(path):
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        return _triangles_ascii_usda(path)
    stage = Usd.Stage.Open(str(path))
    if stage is None or UsdGeom.GetStageUpAxis(stage) != "Z" or UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError("object visual USD must have metric Z-up coordinates")
    root, cache, meshes = stage.GetDefaultPrim(), UsdGeom.XformCache(), []
    if not root:
        raise ValueError("object USD requires a default prim defining its pose origin")
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
            continue
        mesh = UsdGeom.Mesh(prim)
        transform = cache.ComputeRelativeTransform(prim, root)[0]
        p = np.array([transform.Transform(point) for point in mesh.GetPointsAttr().Get()], dtype=np.float64)
        counts, faces = np.array(mesh.GetFaceVertexCountsAttr().Get()), np.array(mesh.GetFaceVertexIndicesAttr().Get())
        if not np.all(counts == 3) or len(faces) != 3 * len(counts):
            raise ValueError("object visual USD requires explicit triangle topology")
        meshes.append(p[faces.reshape(-1, 3)])
    if not meshes:
        raise ValueError("object USD contains no visible triangle meshes")
    return np.concatenate(meshes), {"reader": "Pixar_USD_relative_to_default_prim", "mesh_count": len(meshes)}


def write_stl(path, triangles):
    triangles = np.asarray(triangles, dtype=np.float32)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles) or not np.isfinite(triangles).all():
        raise ValueError("display triangles must be finite [F,3,3]")
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-30)
    entries = np.zeros(len(triangles), dtype=[("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    entries["normal"], entries["vertices"] = normal, triangles
    with Path(path).open("xb") as stream:
        stream.write(b"HOI recorded display triangles; no physical collider".ljust(80, b"\0"))
        stream.write(struct.pack("<I", len(triangles)))
        stream.write(entries.tobytes())


def object_display(asset, output, *, explicit_visual=None, use_collision_geometry=False):
    """Return static display mesh paths and an explicit source/equivalence record."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if explicit_visual is not None:
        source, expected = Path(explicit_visual), None
        role = "explicit_user_visual"
    elif asset.get("visual_mesh"):
        record = asset["visual_mesh"]
        source, expected, role = Path(record["path"]), record["sha256"], "bound_visual_mesh"
    elif asset.get("source_native_asset") and not use_collision_geometry:
        source, expected = Path(asset["source_native_asset"]), asset["source_native_asset_sha256"]
        role = "bound_original_source_USD_meshes"
    elif use_collision_geometry:
        import torch
        record = asset["geometry_cache"]
        if sha(record["path"]) != record["sha256"]:
            raise ValueError("bound collision geometry cache bytes changed")
        geometry = torch.load(record["path"], map_location="cpu", weights_only=True)
        triangles = np.concatenate([m["vertices"].numpy()[m["faces"].numpy()] for m in geometry["collision_meshes"]])
        target = output / "explicit_collision_display.stl"
        write_stl(target, triangles)
        return [target], {"role": "explicit_collision_geometry_display", "artwork_substitution": "explicitly_requested",
                          "source_sha256": record["sha256"], "collision_hash": asset["collision_hash"],
                          "triangle_count": len(triangles), "mesh_count": len(geometry["collision_meshes"]), "mesh_sha256": sha(target)}
    else:
        raise ValueError("provide a bound/explicit object visual or explicitly select --use-collision-geometry")
    digest = sha(source)
    if expected is not None and digest != expected:
        raise ValueError("object visual bytes differ from their binding")
    record = {"role": role, "source": str(source.resolve()), "source_sha256": digest,
              "visual_equivalence_to_native_collider": "not_asserted; display geometry only"}
    if source.suffix.lower() in (".usd", ".usda", ".usdc"):
        triangles, reader = _triangles_usd(source)
        target = output / "object_visual_triangles.stl"
        write_stl(target, triangles)
        return [target], {**record, **reader, "triangle_count": len(triangles), "mesh_sha256": sha(target)}
    if source.suffix.lower() not in (".stl", ".obj", ".ply"):
        raise ValueError("object display supports triangle USD/STL/OBJ/PLY")
    return [source.resolve()], {**record, "mesh_count": 1}


def _quat_rpy(value):
    r, p, y = (float(x) / 2 for x in value.split())
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]


def robot_visual_assets(urdf, names, *, mesh_root=None, verify_hands=True):
    """Resolve the original URDF visuals, with explicit relative-path relocation.

    mesh_root replaces the URDF directory for relative mesh names. Absolute
    names retain their meaning. There is no basename search or fallback to
    collision artwork. Every selected visual is recorded with its actual SHA.
    """
    from hoi.hand_collision import HAND_SHA
    urdf, visuals, sources, seen_hands = Path(urdf), {}, [], set()
    mesh_root = Path(mesh_root).resolve() if mesh_root is not None else None
    if mesh_root is not None and not mesh_root.is_dir():
        raise ValueError("explicit robot mesh root must be an existing asset directory")
    mesh_specs = []
    for link in ET.parse(urdf).getroot().findall("link"):
        name = link.get("name")
        for index, visual in enumerate(link.findall("visual")):
            if name not in names:
                raise ValueError(f"visual body {name} has no saved transform")
            mesh, origin = visual.find("geometry/mesh"), visual.find("origin")
            if mesh is None:
                raise ValueError("G1 display requires its actual URDF triangle visual meshes")
            source = Path(mesh.get("filename"))
            if not source.is_absolute():
                source = ((mesh_root if mesh_root is not None else urdf.parent) / source).resolve()
                if mesh_root is not None and not source.is_relative_to(mesh_root):
                    raise ValueError("relative URDF visual mesh escapes the explicit robot mesh root")
            if not source.is_file():
                raise FileNotFoundError(f"URDF visual mesh missing: {source}; a relocated metadata URDF requires explicit --robot-mesh-root asset_directory")
            digest = sha(source)
            for side in ("left", "right"):
                if name == side + "_rubber_hand":
                    if verify_hands and digest != HAND_SHA[side]:
                        raise ValueError("rubber-hand display must use the original byte-verified STL")
                    visual_xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
                    visual_rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
                    if verify_hands and (not np.array_equal(visual_xyz, np.zeros(3)) or not np.array_equal(visual_rpy, np.zeros(3)) or
                                         not np.array_equal(np.fromstring(mesh.get("scale", "1 1 1"), sep=" "), np.ones(3))):
                        raise ValueError("recorded hand-link visuals must retain the original zero visual offset and unit scale")
                    seen_hands.add(side)
            mesh_name = f"robot_{len(sources)}"
            mesh_specs.append({"name": mesh_name, "file": source.resolve().as_posix(), "scale": mesh.get("scale", "1 1 1")})
            colour = visual.find("material/color")
            record = {"mesh": mesh_name, "pos": origin.get("xyz", "0 0 0") if origin is not None else "0 0 0",
                      "quat": " ".join(map(str, _quat_rpy(origin.get("rpy", "0 0 0")))) if origin is not None else "1 0 0 0",
                      "rgba": colour.get("rgba") if colour is not None else ".65 .65 .65 1"}
            visuals.setdefault(name, []).append(record)
            sources.append({"body": name, "path": str(source.resolve()), "sha256": digest})
    if verify_hands and seen_hands != {"left", "right"}:
        raise ValueError("G1 display URDF must retain both real rubber-hand visual meshes")
    return mesh_specs, visuals, {"urdf": str(urdf.resolve()), "urdf_sha256": sha(urdf), "visual_meshes": sources,
                                 "relative_visual_mesh_root": str(mesh_root) if mesh_root is not None else str(urdf.parent.resolve()),
                                 "relative_visual_mesh_root_explicit": mesh_root is not None,
                                 "real_hand_STL_verified": verify_hands and seen_hands == {"left", "right"}}


def display_model(urdf, object_paths, names, *, width=1280, height=720, verify_hands=True, mesh_root=None):
    """Mocap-only display model with zero generalized coordinates."""
    import mujoco
    mesh_specs, visuals, robot_record = robot_visual_assets(urdf, names, mesh_root=mesh_root, verify_hands=verify_hands)
    root = ET.Element("mujoco", model="HOI_v1_saved_transforms")
    ET.SubElement(root, "compiler", angle="radian")
    assets, world = ET.SubElement(root, "asset"), ET.SubElement(root, "worldbody")
    ET.SubElement(assets, "texture", name="sky", type="skybox", builtin="gradient", rgb1=".97 .98 1", rgb2=".78 .84 .92", width="256", height="256")
    ET.SubElement(assets, "texture", name="floor", type="2d", builtin="checker", rgb1=".89 .91 .94", rgb2=".75 .79 .84", width="512", height="512")
    ET.SubElement(assets, "material", name="floor_material", texture="floor", texrepeat="2 2", reflectance="0")
    ET.SubElement(world, "geom", type="plane", size="0 0 .05", material="floor_material", contype="0", conaffinity="0")
    ET.SubElement(world, "light", pos="-3 -4 7", dir=".4 .5 -1", directional="true", diffuse=".75 .75 .75", ambient=".4 .4 .4")
    ET.SubElement(world, "light", pos="3 4 5", dir="-.4 -.5 -1", directional="true", diffuse=".35 .35 .35")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(width), offheight=str(height))
    ET.SubElement(visual, "quality", shadowsize="1024", offsamples="1")
    for spec in mesh_specs:
        ET.SubElement(assets, "mesh", **spec)
    for index, path in enumerate(object_paths):
        ET.SubElement(assets, "mesh", name=f"object_{index}", file=Path(path).resolve().as_posix())
    for prefix in ("actual", "reference"):
        for name in names:
            body = ET.SubElement(world, "body", name=prefix + "/" + name, mocap="true")
            for item in visuals.get(name, []):
                fields = {key: value for key, value in item.items()}
                fields["rgba"] = " ".join(map(str, REFERENCE_RGBA)) if prefix == "reference" else fields["rgba"]
                ET.SubElement(body, "geom", type="mesh", contype="0", conaffinity="0", **fields)
        body = ET.SubElement(world, "body", name=prefix + "/object", mocap="true")
        for index in range(len(object_paths)):
            ET.SubElement(body, "geom", type="mesh", mesh=f"object_{index}", contype="0", conaffinity="0",
                          rgba=" ".join(map(str, REFERENCE_RGBA)) if prefix == "reference" else ".72 .36 .12 1")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if model.nq != 0 or model.nv != 0:
        raise RuntimeError("offline display model unexpectedly contains simulated degrees of freedom")
    model.vis.headlight.ambient[:] = .4
    model.vis.headlight.diffuse[:] = .55
    maps = {}
    for prefix in ("actual", "reference"):
        maps[prefix] = np.array([model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "/" + n)] for n in names])
        maps[prefix + "_object"] = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "/object")]
    return model, maps, robot_record


def apply_saved_frame(model, sim_data, maps, data, frame):
    """Assign recorded world transforms only; mj_kinematics does no integration."""
    import mujoco
    for prefix in ("actual", "reference"):
        sim_data.mocap_pos[maps[prefix]] = data[prefix + "_pos"][frame]
        sim_data.mocap_quat[maps[prefix]] = data[prefix + "_quat"][frame]
        sim_data.mocap_pos[maps[prefix + "_object"]] = data[prefix + "_object_pos"][frame]
        sim_data.mocap_quat[maps[prefix + "_object"]] = data[prefix + "_object_quat"][frame]
    mujoco.mj_kinematics(model, sim_data)


def _ffmpeg(explicit=None):
    if explicit:
        return explicit
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("video rendering requires ffmpeg; --preview-only still writes recorded-frame images") from exc


def verify_decoded_frames(ffmpeg, path, expected):
    process = subprocess.run([str(ffmpeg), "-v", "error", "-i", str(path), "-f", "null", "-", "-progress", "pipe:1", "-nostats"],
                             capture_output=True, text=True, check=True)
    count = [int(line.partition("=")[2]) for line in process.stdout.splitlines() if line.startswith("frame=")]
    if not count or count[-1] != expected or process.stderr.strip():
        raise RuntimeError(f"video decode frame count differs: {count[-1:]}, expected {expected}; {process.stderr}")
    return count[-1]


def render_saved(rollout, urdf, asset, output, *, expected_sha256=None, explicit_visual=None, use_collision_geometry=False,
                 preview_only=False, width=1280, height=720, ffmpeg=None, verify_hands=True, robot_mesh_root=None):
    os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")
    import mujoco
    from PIL import Image, ImageDraw
    renderer_source_sha256 = sha(__file__)
    data = load_saved(rollout, expected_sha256)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    meshes, object_record = object_display(asset, output / "object_display", explicit_visual=explicit_visual,
                                            use_collision_geometry=use_collision_geometry)
    model, maps, robot_record = display_model(urdf, meshes, data["names"], width=width, height=height,
                                            verify_hands=verify_hands, mesh_root=robot_mesh_root)
    sim_data, camera = mujoco.MjData(model), mujoco.MjvCamera()
    parameters = fixed_camera_parameters(data)
    camera.lookat[:] = parameters["lookat"]
    camera.distance, camera.azimuth, camera.elevation = parameters["distance"], parameters["azimuth"], parameters["elevation"]
    renderer = mujoco.Renderer(model, height, width)
    n, frames = len(data["time"]), []
    chosen = sorted({0, n // 2, n - 1})
    encode, target, executable = None, output / "rollout.mp4", None
    if not preview_only:
        executable = _ffmpeg(ffmpeg)
        encode = subprocess.Popen([str(executable), "-n", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                                   "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-threads", "4", "-preset", "fast", "-crf", "20",
                                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)], stdin=subprocess.PIPE)
    started = time.monotonic()
    try:
        for frame in chosen if preview_only else range(n):
            apply_saved_frame(model, sim_data, maps, data, frame)
            renderer.update_scene(sim_data, camera=camera)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            image = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, width, 42), fill=(244, 247, 252))
            draw.text((12, 7), f"{data['metadata'].get('name', Path(rollout).stem)} | ACTUAL: original colours / orange object | REFERENCE: translucent blue", fill=(22, 44, 70))
            draw.text((12, 23), f"Saved world transforms | fixed shared camera | t={data['time'][frame]:.2f}/{data['time'][-1]:.2f}s | frame {frame+1}/{n}", fill=(22, 44, 70))
            if frame in chosen:
                image.save(output / f"frame_{frame:06d}.png")
                frames.append({"frame": frame, "time_s": float(data["time"][frame]), "path": f"frame_{frame:06d}.png"})
            if encode is not None:
                encode.stdin.write(np.asarray(image).tobytes())
        if encode is not None:
            encode.stdin.close()
            if encode.wait() != 0:
                raise RuntimeError("ffmpeg failed while encoding complete recorded frames")
    finally:
        renderer.close()
        if encode is not None and encode.poll() is None:
            encode.kill()
            encode.wait()
    decoded = verify_decoded_frames(executable, target, n) if not preview_only else None
    if sha(__file__) != renderer_source_sha256:
        raise RuntimeError("renderer source changed during verification; preserve this output and rerun in a fresh directory")
    report = {"schema_version": "hoi_v1_offline_render_v1", "status": "preview_only" if preview_only else "all_frames_decode_verified",
              "rollout": data["source"], "rollout_sha256": data["rollout_sha256"], "recorded_frames": n, "control_steps": data["control_steps"],
              "fps": FPS, "decoded_frames": decoded, "last_recorded_time_s": float(data["time"][-1]),
              "full_reference_completed": data["metrics"].get("full_reference_completed"),
              "video_display_duration_s": n / FPS, "video": target.name if not preview_only else None,
              "video_sha256": sha(target) if not preview_only else None, "previews": frames,
              "reference_rgba": REFERENCE_RGBA, "actual_robot": "original URDF visual colours", "actual_object": "solid orange",
              "camera": parameters, "robot": robot_record, "object": object_record, "render_seconds": time.monotonic() - started,
              "render_method": "saved body/object world transforms; mocap visual bodies; mj_kinematics only",
              "physics_steps": 0, "policy_inferences": 0, "forward_kinematics_reconstructions": 0,
              "coordinates": "unaltered saved common world; no separate robot/object/reference alignment", "mujoco_version": mujoco.__version__}
    report["renderer_source_sha256"] = renderer_source_sha256
    _json(output / "rendering.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--rollout", type=Path)
    inputs.add_argument("--evaluation", type=Path, help="v1 evaluation.json; all cases unless --case is explicit")
    parser.add_argument("--bundle", type=Path, help="bound evaluation bundle; supplies object visual/collision identity")
    parser.add_argument("--robot-urdf", type=Path, required=True)
    parser.add_argument("--robot-mesh-root", type=Path,
                        help="explicit asset directory replacing URDF directory for relative mesh names (directory containing meshes/)")
    parser.add_argument("--object-visual", type=Path, help="explicit artwork for a single asset")
    parser.add_argument("--use-collision-geometry", action="store_true", help="explicitly display bound collision meshes in place of artwork")
    parser.add_argument("--case", action="append", help="exact reported case ID; repeated flags select explicit cases")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--ffmpeg")
    args = parser.parse_args(argv)
    if args.width < 160 or args.height < 120 or args.width % 2 or args.height % 2:
        parser.error("video dimensions must be even and at least 160x120")
    if args.object_visual and args.use_collision_geometry:
        parser.error("choose artwork or an explicit collision display")
    bundle = None
    if args.bundle:
        from .evaluate import _read_bundle_header
        bundle = _read_bundle_header(args.bundle)
    if args.evaluation:
        evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
        cases = evaluation["cases"]
        if args.case:
            selected = set(args.case)
            cases = [case for case in cases if case["id"] in selected]
            if {case["id"] for case in cases} != selected:
                raise ValueError("requested rendering case is absent from the evaluation report")
    else:
        saved = load_saved(args.rollout)
        cases = [{"id": saved["metadata"].get("name", args.rollout.stem), "rollout": str(args.rollout.resolve()),
                  "rollout_sha256": saved["rollout_sha256"], "asset_id": saved["metadata"]["asset_id"]}]
    if not cases:
        raise ValueError("no recorded cases selected")
    if args.object_visual and len({case["asset_id"] for case in cases}) != 1:
        raise ValueError("one --object-visual cannot substitute multiple different assets")
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for index, case in enumerate(cases):
        asset = next(asset for asset in bundle["assets"] if asset["asset_id"] == case["asset_id"]) if bundle else {}
        source = Path(case["rollout"])
        if not source.is_absolute():
            source = args.evaluation.parent / source
        reports.append(render_saved(source, args.robot_urdf, asset, args.output / f"case_{index:05d}",
                                    expected_sha256=case.get("rollout_sha256"), explicit_visual=args.object_visual,
                                    use_collision_geometry=args.use_collision_geometry, preview_only=args.preview_only,
                                    width=args.width, height=args.height, ffmpeg=args.ffmpeg, robot_mesh_root=args.robot_mesh_root))
        print(json.dumps({"case": case["id"], "recorded_frames": reports[-1]["recorded_frames"], "decoded_frames": reports[-1]["decoded_frames"]}), flush=True)
    _json(args.output / "rendering.json", {"reports": reports, "selection": [case["id"] for case in cases],
                                          "selection_rule": "all evaluation cases or exact explicit case IDs; no score filtering",
                                          "native_robot_asset": bundle.get("native_robot_asset") if bundle else None,
                                          "bundle_content_sha256": bundle.get("bundle_content_sha256") if bundle else None,
                                          "evaluation_sha256": sha(args.evaluation) if args.evaluation else None,
                                          "native_collision_vs_visual": "bound native overlay is physics provenance; robot artwork remains original URDF/STL"})
    cards = []
    for index, (case, report) in enumerate(zip(cases, reports)):
        base = f"case_{index:05d}"
        preview = report["previews"][len(report["previews"]) // 2]["path"]
        media = f'<img src="{base}/{preview}" width="640">' if args.preview_only else f'<video controls width="640" poster="{base}/{preview}" src="{base}/rollout.mp4"></video>'
        cards.append(f'<article><h2>{html.escape(case["id"])}</h2>{media}<p>{report["recorded_frames"]} saved frames, 50 fps; no simulation steps.</p></article>')
    (args.output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Recorded HOI v1</title><h1>Recorded HOI v1</h1><p>Actual: original colours/orange. Reference: translucent blue. Shared fixed world camera.</p>' + "".join(cards), encoding="utf-8")


if __name__ == "__main__":
    main()
