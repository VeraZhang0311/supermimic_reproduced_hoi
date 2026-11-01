"""Independent HOI native scene and joint robot/object rollout contract.

Importing this module does not initialize IsaacLab. Instantiate HOITrackingSim
inside the launcher context. Geometry is fixed for an environment's lifetime;
reset sampling is restricted to its exact asset/scale/profile group.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from supermimic.mdp import flatten_state
from supermimic.motion import CONTROL_DT
from supermimic.rotations import mat_to_quat, quat_to_mat
from .contacts import (CHANNELS, CHANNEL_INDEX, CHANNEL_ROLES, BODY_ROLES, GROUND, LEFT, OBJECT,
                       RIGHT, LEFT_FOOT, RIGHT_FOOT, PairContactRecorder, compiled_geom_links)
from .state import flatten_object
from .probes import palm_patch_points, _write_palm_qa

LEGACY_CAPACITY = {"nconmax": 128, "njmax": 500}
HOI_CAPACITY = {"nconmax": 512, "njmax": 2048}


def configure_hoi_capacity(config):
    """Increase only HOI allocation limits; retain the legacy source config."""
    from copy import deepcopy
    result = deepcopy(config)
    result.physics.use_cuda_graph = False
    for name, value in HOI_CAPACITY.items():
        setattr(result.physics.solver_cfg, name, value)
    return result


def compiled_convex_mesh(model, mesh_id):
    """Read the actual collision hull facets from MuJoCo's native graph.

    ``mesh_face`` is a visual surface and is not the collision authority.
    Graph layout is documented at the linked primary API reference. Global
    vertex IDs are relative to this full mesh, before unused vertices are cut.
    """
    import hashlib
    import numpy as np
    address = int(model.mesh_graphadr[mesh_id])
    if address < 0:
        raise RuntimeError("Native mesh has no collision convex graph; refusing a visual-mesh approximation")
    graph = np.asarray(model.mesh_graph)
    nv, nf = map(int, graph[address:address + 2])
    vertex_offset, vertex_count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])
    facet_offset = address + 2 + 3 * nv + 3 * nf
    global_faces = graph[facet_offset:facet_offset + 3 * nf].reshape(nf, 3).astype(np.int64)
    referenced = np.unique(global_faces)
    if nv < 4 or nf < 4 or len(referenced) != nv or int(referenced.min()) < 0 or int(referenced.max()) >= vertex_count:
        raise RuntimeError("Malformed native convex-hull graph")
    original_vertices = np.asarray(model.mesh_vert[vertex_offset:vertex_offset + vertex_count])
    vertices = original_vertices[referenced].astype(np.float64)
    faces = np.searchsorted(referenced, global_faces)
    record = graph[address:facet_offset + 3 * nf]
    evidence = {"source": "mjModel.mesh_graph native collision hull facets", "graph_address": address,
                "reference": "https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#convex-hulls",
                "raw_vertex_count": vertex_count, "raw_face_count": face_count, "hull_vertex_count": nv, "hull_face_count": nf,
                "source_vertex_ids": referenced.tolist(), "graph_record_sha256": hashlib.sha256(record.tobytes()).hexdigest(),
                "full_mesh_vertex_buffer_sha256": hashlib.sha256(original_vertices.tobytes()).hexdigest()}
    return vertices, faces, evidence


def compiled_geom_support(model, geom, direction, *, facet=False):
    """Actual compiled collider support, in geom frame; no joint-center proxy."""
    import mujoco
    d = torch.as_tensor(direction, dtype=torch.float64)
    if d.shape != (3,) or not bool(torch.isfinite(d).all()) or float(d.norm()) <= 1e-12:
        raise ValueError("support direction must be a finite nonzero 3-vector")
    d = d / d.norm()
    size = torch.as_tensor(model.geom_size[geom], dtype=torch.float64)
    kind = int(model.geom_type[geom])
    proof = {"geom_id": int(geom), "primitive": int(kind), "source": "actual_compiled_collision_surface"}
    if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        v, f, hull_proof = compiled_convex_mesh(model, int(model.geom_dataid[geom]))
        vertices, faces = torch.tensor(v), torch.tensor(f)
        if facet:
            triangles = vertices[faces]
            normals = torch.linalg.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            lengths = normals.norm(dim=-1)
            good = lengths > 1e-12
            normals = normals / lengths.clamp_min(1e-12)[:, None]
            signs = ((triangles.mean(1) - vertices.mean(0)) * normals).sum(-1)
            normals = torch.where((signs < 0)[:, None], -normals, normals)
            score = torch.where(good, normals @ d, torch.full_like(lengths, -2))
            chosen = int(score.argmax())
            normal = normals[chosen]
            plane = vertices @ normal
            point = vertices[(plane - plane.max()).abs() < 1e-7].mean(0)
        else:
            point, normal = vertices[int((vertices @ d).argmax())], d
        proof["native_convex_evidence"] = hull_proof
    elif kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        point, normal = size[0] * d, d
    elif kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        point, normal = size[0] * d, d
        point[2] += size[1] * torch.sign(d[2])
    elif kind == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        point, normal = size.square() * d / (size * d).norm(), d
    elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        if facet:
            normal = torch.zeros(3, dtype=torch.float64)
            axis = int(d.abs().argmax()); normal[axis] = torch.sign(d[axis])
            point = normal * size
        else:
            point, normal = torch.sign(d) * size, d
    elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        radial = d[:2] / d[:2].norm().clamp_min(1e-12)
        point = torch.cat((radial * size[0], (size[1] * torch.sign(d[2])).reshape(1)))
        normal = d
    else:
        raise RuntimeError(f"Unsupported actual body collider primitive {kind}; native evidence cannot be inferred")
    if not bool(torch.isfinite(point).all() and torch.isfinite(normal).all()):
        raise RuntimeError("Nonfinite compiled collider support")
    return point, normal, proof


def link_to_com_velocity(R, v_link, omega_world, local_com):
    offset = (R @ local_com.unsqueeze(-1)).squeeze(-1)
    return v_link + torch.linalg.cross(omega_world, offset, dim=-1)


def com_to_link_velocity(R, v_com, omega_world, local_com):
    offset = (R @ local_com.unsqueeze(-1)).squeeze(-1)
    return v_com - torch.linalg.cross(omega_world, offset, dim=-1)


def shallow_support_contact(palm_world, outward_normal_world, object_offsets_world, penetration=.002, *, lowest_support=False):
    """Place a convex-union support vertex against a measured palm surface."""
    normal = outward_normal_world
    if penetration <= 0 or not torch.allclose(normal.norm(dim=-1), torch.ones_like(normal[..., 0]), atol=2e-6, rtol=0):
        raise ValueError("shallow contact needs a positive penetration and unit outward normals")
    projection = (object_offsets_world * normal[:, None]).sum(-1)
    index = projection.argmin(-1)
    if lowest_support:
        # Select the lower edge of a support face, not an arbitrarily ordered
        # upper corner that could place a tall box underground next to a toe.
        on_plane = projection <= projection.amin(-1, keepdim=True) + 1e-7
        index = torch.where(on_plane, object_offsets_world[..., 2], torch.inf).argmin(-1)
    support = object_offsets_world[torch.arange(len(index), device=index.device), index]
    position = palm_world - support - penetration * normal
    if lowest_support:
        # A numerical coplanarity tie must not increase the normal overlap.
        chosen_projection = projection.gather(1, index[:, None]).squeeze(1)
        position = position + (chosen_projection - projection.amin(-1))[:, None] * normal
    return position, index


def lift_foot_probe_above_ground(root_world, object_world, surface_world, object_offsets_world, clearance=.002):
    """Translate only a disposable calibration pair, preserving its contact."""
    initial = (object_world[:, None, 2] + object_offsets_world[..., 2]).amin(-1)
    lift = (clearance - initial).clamp_min(0)
    delta = torch.zeros_like(object_world)
    delta[:, 2] = lift
    return root_world + delta, object_world + delta, surface_world + delta, lift, initial, initial + lift


def _compiled_foot_meshes(model, geom_ids):
    """Actual collision hulls transformed from compiled geom to ankle axes."""
    import mujoco
    ids = list(map(int, geom_ids))
    if not ids or len({int(model.geom_bodyid[g]) for g in ids}) != 1:
        raise RuntimeError("Foot colliders must belong to one actual ankle-roll body")
    meshes = []
    for geom in ids:
        if int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            raise RuntimeError("Foot mesh calibration requires all actual compiled ankle-roll colliders to be meshes")
        vertices, faces, proof = compiled_convex_mesh(model, int(model.geom_dataid[geom]))
        rotation = quat_to_mat(torch.tensor(model.geom_quat[geom], dtype=torch.float64))
        vertices = torch.tensor(vertices, dtype=torch.float64) @ rotation.T + torch.tensor(model.geom_pos[geom], dtype=torch.float64)
        if not bool(torch.isfinite(vertices).all()):
            raise RuntimeError("Nonfinite actual compiled foot hull")
        meshes.append((geom, vertices, torch.tensor(faces, dtype=torch.long), proof))
    return meshes


def compiled_foot_probe_surface(model, geom_ids, *, bottom=False):
    """Use an exposed collision-hull facet interior and its real outward normal.

    All foot hull vertices must lie behind the selected facet plane. Placing
    the object's opposing support plane 2 mm inside it therefore cannot embed
    an object center or a different foot hull deeply. Spheres remain supported
    solely for historical CPU fixtures; the ULTRA profile gate rejects them.
    """
    import mujoco
    ids = list(map(int, geom_ids))
    direction = torch.tensor([0., 0., -1.] if bottom else [1., 0., 0.], dtype=torch.float64)
    if ids and all(int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH) for g in ids):
        meshes = _compiled_foot_meshes(model, ids)
        all_vertices = torch.cat([mesh[1] for mesh in meshes])
        candidates = []
        for geom, vertices, faces, proof in meshes:
            center = vertices.mean(0)
            for face in faces:
                triangle = vertices[face]
                normal = torch.linalg.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
                length = normal.norm()
                if float(length) <= 1e-14:
                    continue
                normal = normal / length
                if float(normal @ (triangle.mean(0) - center)) < 0:
                    normal = -normal
                alignment = float(normal @ direction)
                if alignment < .25:
                    continue
                plane = triangle[0] @ normal
                excess = float((all_vertices @ normal - plane).max())
                if excess > 1e-7:
                    continue
                # Coplanar triangle vertices describe one convex facet. Their
                # mean is in its interior, rather than on a support corner.
                coplanar = (vertices @ normal - plane).abs() <= 1e-7
                point = vertices[coplanar].mean(0)
                candidates.append((alignment, float(point @ direction), geom, point, normal, proof, excess,
                                   coplanar.nonzero().flatten().tolist()))
        if not candidates:
            raise RuntimeError("No exposed actual foot-hull facet can support a shallow toe probe")
        _, _, geom, point, normal, proof, excess, vertex_ids = max(candidates, key=lambda item: item[:2])
        evidence = {"geom_id": geom, "body_id": int(model.geom_bodyid[geom]),
                    "geom_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom),
                    "body_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])),
                    "primitive": "actual_compiled_convex_mesh", "native_convex_evidence": proof,
                    "selection": "interior_of_exposed_global_support_facet",
                    "facet_vertex_ids": vertex_ids, "all_foot_hull_count": len(meshes),
                    "maximum_other_vertex_plane_excess_m": excess,
                    "surface_point_link": point.tolist(), "outward_normal_link": normal.tolist()}
        return point, normal, evidence
    if not ids or any(int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_SPHERE) for g in ids):
        raise RuntimeError("Foot pair calibration requires actual compiled ankle-roll collision spheres or convex meshes")
    centers = torch.tensor(model.geom_pos[ids], dtype=torch.float64)
    radii = torch.tensor(model.geom_size[ids, 0], dtype=torch.float64)
    if not bool(torch.isfinite(centers).all() and torch.isfinite(radii).all()) or bool((radii <= 0).any()):
        raise RuntimeError("Invalid compiled foot collision sphere")
    selected = int((centers @ direction + radii).argmax())
    geom = ids[selected]
    point = centers[selected] + radii[selected] * direction
    definition = {"geom_id": geom, "body_id": int(model.geom_bodyid[geom]),
                  "geom_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom),
                  "body_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])),
                  "primitive": "actual_compiled_sphere", "radius_m": float(radii[selected]),
                  "surface_point_link": point.tolist(), "outward_normal_link": direction.tolist()}
    return point, direction, definition


def compiled_foot_world_minimum(model, geom_ids, link_position, link_rotation):
    """Lowest actual collider point in world z, including rotated mesh feet."""
    import mujoco
    ids = list(map(int, geom_ids))
    if ids and all(int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH) for g in ids):
        meshes = _compiled_foot_meshes(model, ids)
        vertices = torch.cat([mesh[1] for mesh in meshes]).to(link_position)
        z = (link_rotation[:, 2, :] @ vertices.T) + link_position[:, 2, None]
        return z.amin(-1), {"source": "all_actual_compiled_convex_hull_vertices",
                           "hull_count": len(meshes), "vertex_count": len(vertices)}
    # A rotated sphere still has a world-z radius, unlike a rotated local bottom.
    compiled_foot_probe_surface(model, ids, bottom=True)
    centers = torch.tensor(model.geom_pos[ids], device=link_position.device, dtype=link_position.dtype)
    radii = torch.tensor(model.geom_size[ids, 0], device=link_position.device, dtype=link_position.dtype)
    z = (link_rotation[:, 2, :] @ centers.T) + link_position[:, 2, None] - radii
    return z.amin(-1), {"source": "actual_compiled_sphere_centers_minus_world_radius", "sphere_count": len(ids)}


def compiled_foot_geometry_audit(model, *, num_envs, topology, manifest, manifest_sha256, robot_asset_sha256,
                                 reference_model, reference_mjcf_sha256):
    """Every active ankle-roll collider in every compiled world is a bound mesh."""
    import hashlib
    import itertools
    import re
    import mujoco
    from .foot_collision import FOOT_PROFILE, FOOT_SCHEMA, validate_hulls
    if manifest.get("schema") != FOOT_SCHEMA or manifest.get("profile") != FOOT_PROFILE:
        raise RuntimeError("ULTRA-aligned foot manifest/profile is missing")
    if manifest["overlay"]["sha256"] != robot_asset_sha256:
        raise RuntimeError("Compiled robot asset differs from the bound foot overlay")
    if topology not in ("one_mujoco_world_per_environment", "one_world_with_named_environments") or num_envs < 1:
        raise RuntimeError("Unknown native foot world topology")
    sides = {}
    for side in ("left", "right"):
        groups = {}
        link = side + "_ankle_roll_link"
        for geom in range(model.ngeom):
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])) or ""
            if not re.search(r"(?:^|[/_])" + link + r"$", body) or not (int(model.geom_contype[geom]) or int(model.geom_conaffinity[geom])):
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
            match = re.search(r"env[_/](\d+)", name + " " + body)
            environment = int(match.group(1)) if match else 0
            if topology == "one_mujoco_world_per_environment" and environment != 0:
                raise RuntimeError("Single-world foot model contains an unexpected environment name")
            if topology == "one_world_with_named_environments" and match is None:
                raise RuntimeError("Foot collider is not bound to an explicit environment")
            groups.setdefault(environment, []).append(geom)
        expected_worlds = 1 if topology == "one_mujoco_world_per_environment" else num_envs
        if set(groups) != set(range(expected_worlds)):
            raise RuntimeError("Actual compiled foot colliders do not cover every environment")
        expected_count = manifest["sides"][side]["hull_count"]
        if type(expected_count) is not int or not 1 <= expected_count <= 5:
            raise RuntimeError("Foot hull count violates the bound five-hull recipe")
        cooked_record = manifest["sides"][side]["cooked_geometry"]
        cooked_bytes = Path(cooked_record["path"]).read_bytes()
        if hashlib.sha256(cooked_bytes).hexdigest() != cooked_record["sha256"]:
            raise RuntimeError("Bound foot cooked geometry bytes changed")
        expected_pieces = validate_hulls(json.loads(cooked_bytes), side)
        expected_vertices = [torch.tensor(piece["vertices"], dtype=torch.float64) for piece in expected_pieces]
        if len(expected_vertices) != expected_count:
            raise RuntimeError("Bound foot cook hull count differs from its manifest")
        records = []
        body_ids = []
        max_geometry_error = 0.
        for environment, ids in sorted(groups.items()):
            if len(ids) != expected_count:
                raise RuntimeError("Actual foot geometry count differs from its bound mesh cook")
            meshes = _compiled_foot_meshes(model, ids)
            if any(len(vertices) > 16 for _, vertices, _, _ in meshes):
                raise RuntimeError("Actual foot hull exceeds the bound sixteen-vertex recipe")
            errors = []
            for _, vertices, _, _ in meshes:
                row = []
                for expected in expected_vertices:
                    distance = torch.cdist(vertices, expected)
                    row.append(float(torch.maximum(distance.amin(0).amax(), distance.amin(1).amax())))
                errors.append(row)
            # Match hulls and vertices without relying on USD importer order or
            # triangle tessellation. Count equality alone cannot verify scale.
            best = min(max(errors[index][matched] for index, matched in enumerate(permutation))
                       for permutation in itertools.permutations(range(expected_count)))
            if best > 5e-5:
                raise RuntimeError(f"Actual compiled foot hulls differ from bound ankle-local cooked geometry: {best:.9g} m")
            max_geometry_error = max(max_geometry_error, best)
            records.extend({"environment": environment, "geom_id": geom, "native_convex_evidence": proof}
                           for geom, _, _, proof in meshes)
            body_ids.append(int(model.geom_bodyid[ids[0]]))
        reference_body = mujoco.mj_name2id(reference_model, mujoco.mjtObj.mjOBJ_BODY, link)
        if reference_body < 0:
            raise RuntimeError("Bound reference MJCF lacks the original foot body inertia")
        def read_physics(source, body):
            return {"body_mass": float(source.body_mass[body]), "body_ipos": source.body_ipos[body].tolist(),
                    "body_inertia": source.body_inertia[body].tolist(), "body_iquat": source.body_iquat[body].tolist()}
        original = read_physics(reference_model, reference_body)
        expected_mass = torch.tensor(original["body_mass"], dtype=torch.float64)
        expected_com = torch.tensor(original["body_ipos"], dtype=torch.float64)
        ir = quat_to_mat(torch.tensor(original["body_iquat"], dtype=torch.float64))
        expected_inertia = ir @ torch.diag(torch.tensor(original["body_inertia"], dtype=torch.float64)) @ ir.T
        maxima = {"mass_absolute_error": 0., "com_max_absolute_error_m": 0., "inertia_tensor_max_absolute_error": 0.}
        for body in body_ids:
            actual = read_physics(model, body)
            mass = torch.tensor(actual["body_mass"], dtype=torch.float64)
            com = torch.tensor(actual["body_ipos"], dtype=torch.float64)
            ir = quat_to_mat(torch.tensor(actual["body_iquat"], dtype=torch.float64))
            inertia = ir @ torch.diag(torch.tensor(actual["body_inertia"], dtype=torch.float64)) @ ir.T
            if (not all(bool(torch.isfinite(x).all()) for x in (mass, com, inertia)) or
                    not bool(mass > 0) or not bool((torch.linalg.eigvalsh(inertia) > 0).all()) or
                    any(not torch.allclose(a, b, atol=1e-8, rtol=1e-5) for a, b in (
                        (mass, expected_mass), (com, expected_com), (inertia, expected_inertia)))):
                raise RuntimeError("Compiled mesh foot mass/COM/inertia differs from the bound original MJCF: " +
                                   json.dumps({"link": link, "body_id": body, "actual": actual, "original": original}))
            for key, difference in (("mass_absolute_error", (mass - expected_mass).abs()),
                                    ("com_max_absolute_error_m", (com - expected_com).abs().max()),
                                    ("inertia_tensor_max_absolute_error", (inertia - expected_inertia).abs().max())):
                maxima[key] = max(maxima[key], float(difference))
        sides[side] = {"link": link, "hulls_per_environment": expected_count, "model_environment_count": expected_worlds,
                       "simulated_environment_count": num_envs, "model_collision_geom_count": len(records),
                       "all_active_colliders_mesh": True, "all_geometries_graph_backed": True,
                       "native_convex_evidence_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
                       "native_vs_cooked_geometry": {"passed": True, "comparison": "unordered_bidirectional_vertex_set_distance",
                           "coordinate_frame": "ankle_roll_link", "cooked_geometry_sha256": cooked_record["sha256"],
                           "tolerance_m": 5e-5, "max_error_m": max_geometry_error,
                           "compared_model_environment_count": expected_worlds, "matched_hull_pairs": expected_worlds * expected_count},
                       "physical_properties": {"passed": True, "reference_mjcf_sha256": reference_mjcf_sha256,
                           "compared_model_body_count": len(body_ids), "actual_first_body": read_physics(model, body_ids[0]),
                           "original_reference_body": original, "inertia_comparison": "link_frame_symmetric_tensor",
                           "absolute_tolerance": 1e-8, "relative_tolerance": 1e-5, **maxima}}
    return {"schema": "hoi_v1_compiled_ultra_foot_mesh_v1", "passed": True, "profile": FOOT_PROFILE,
            "manifest_schema": FOOT_SCHEMA, "foot_asset_manifest_sha256": manifest_sha256,
            "source_mesh_sha256": {side: manifest["source_meshes"][side]["sha256"] for side in sides},
            "source_mesh_transforms": {side: {key: manifest["source_meshes"][side][key] for key in (
                "collision_origin", "collision_rpy", "mesh_scale", "link")} for side in sides},
            "original_ultra_urdf_sha256": manifest["original_urdf"]["sha256"],
            "cook_sha256": manifest["cook_sha256"], "cook": manifest["cook"],
            "overlay_sha256": robot_asset_sha256, "topology": topology, "num_envs": num_envs, "sides": sides}


def physical_context(mass, com, inertia, friction, ground_friction):
    """12D context from effective native values; inertia is in object axes."""
    entries = torch.stack((inertia[..., 0, 0], inertia[..., 1, 1], inertia[..., 2, 2],
                           inertia[..., 0, 1], inertia[..., 0, 2], inertia[..., 1, 2]), -1)
    return torch.cat((mass.unsqueeze(-1), com, entries, friction.unsqueeze(-1), ground_friction.unsqueeze(-1)), -1)


def validate_physical_profile(profile):
    mass = float(profile["mass"])
    com = torch.as_tensor(profile["com"], dtype=torch.float64)
    inertia = torch.as_tensor(profile["inertia"], dtype=torch.float64).reshape(3, 3)
    friction, ground = float(profile["friction"]), float(profile["ground_friction"])
    if not torch.isfinite(torch.cat((com, inertia.flatten(), torch.tensor([mass, friction, ground])))).all():
        raise ValueError("object physical profile contains nonfinite values")
    if mass <= 0 or friction < 0 or ground < 0 or com.shape != (3,):
        raise ValueError("invalid mass/COM/friction")
    if not torch.allclose(inertia, inertia.T, rtol=0, atol=1e-10) or (torch.linalg.eigvalsh(inertia) <= 0).any():
        raise ValueError("object inertia must be symmetric positive definite")
    eig = torch.linalg.eigvalsh(inertia)
    if eig[-1] > eig[:2].sum() + 1e-8:
        raise ValueError("object inertia violates the rigid-body triangle inequality")
    return mass, com, inertia, friction, ground








def prepare_object_usd(asset, output, *, allow_deferred_inertia=False):
    """Flatten an independent copy; source USD is read only and already scaled."""
    import numpy as np
    from pxr import Gf, Usd, UsdPhysics
    from scipy.spatial.transform import Rotation
    from supermimic.physics import sha
    source = Path(asset["native_asset"])
    if "native_asset_sha256" not in asset or sha(source) != asset["native_asset_sha256"]:
        raise ValueError("native object asset requires a matching SHA256 binding")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "object.usda"
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise ValueError(f"Unable to open object USD {source}")
    stage.Flatten().Export(str(path))
    stage = Usd.Stage.Open(str(path))
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) != 1:
        raise ValueError("v1 requires exactly one object rigid body")
    from .native_fixture import DEFERRED_INERTIA_MODE, validate_cooking_mass_profile
    deferred = asset.get("inertia_mode") == DEFERRED_INERTIA_MODE
    if asset.get("inertia_mode") is not None and (not allow_deferred_inertia or not deferred):
        raise ValueError("deferred inertia is only permitted in the explicit native cooking fixture")
    mass_api = UsdPhysics.MassAPI.Apply(bodies[0])
    if deferred:
        mass, _, _ = validate_cooking_mass_profile(asset["physical_profile"])
        if mass_api.GetCenterOfMassAttr().HasAuthoredValueOpinion() or mass_api.GetDiagonalInertiaAttr().HasAuthoredValueOpinion():
            raise ValueError("deferred native cooking source must not author placeholder COM/inertia")
        mass_api.CreateMassAttr(mass)
    else:
        mass, com, inertia, _, _ = validate_physical_profile(asset["physical_profile"])
        mass_api.CreateMassAttr(mass)
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com)))
        values, axes = np.linalg.eigh(inertia.numpy())
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        q = Rotation.from_matrix(axes).as_quat()
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, values)))
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))
    rigid = UsdPhysics.RigidBodyAPI(bodies[0])
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(False)
    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    if not colliders or any(not UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() for p in colliders):
        raise ValueError("bound collision recipe must contain enabled native colliders")
    stage.GetRootLayer().Save()
    return path


def make_hoi_sim(device, args):
    from supermimic.env import make_sim
    return configure_hoi_capacity(make_sim(device, args))


from .scene_supports import (validate_native_scene_supports, NativeSceneSupportBank, omni_support_recipe,
                              SUPPORT_IDS, SUPPORT_ROLES, digest as support_digest)


class HOITrackingSim:
    def __init__(self, sim_cfg, num_envs, motion, mdp, asset, output, *, gamma=.99,
                 eval_mode=False, reset_noise=True, max_episode_steps=500, robot_asset=None, cooking_only=False):
        if asset is not None and asset.get("inertia_mode") is not None and not cooking_only:
            raise ValueError("deferred inertia cannot enter a training or evaluation environment")
        validate_native_scene_supports(motion)
        import isaaclab.sim as sim_utils
        import warp as wp
        from .native_runtime import configure_native_forward_compat, require_native_physics_driver
        require_native_physics_driver(wp, cooking_only=cooking_only)
        self.warp_compilation = configure_native_forward_compat(wp, physics_autograd=False)
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.utils import configclass
        from supermimic.env import TrackingSceneCfg, TrackingSim
        asset_id = int(asset["asset_id"])
        compatible = motion.compatible_bins(asset_id)
        if not len(compatible):
            raise ValueError("worker asset has no compatible motion bins")
        support_bank = NativeSceneSupportBank(motion, asset_id, sim_cfg.device)
        usd = prepare_object_usd(asset, Path(output) / "object_asset", allow_deferred_inertia=cooking_only)

        @configclass
        class HOISceneCfg(TrackingSceneCfg):
            object = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Object",
                spawn=sim_utils.UsdFileCfg(usd_path=str(usd), rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, linear_damping=0., angular_damping=0.)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(4., 4., 2.)))

        def scene_factory(**kwargs):
            config = HOISceneCfg(**kwargs)
            recipe = omni_support_recipe()
            for sid in support_bank.support_ids:
                setattr(config, "support_" + sid, RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/Support_" + sid,
                    spawn=sim_utils.CuboidCfg(size=tuple(recipe["size_m"]),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True,
                            linear_damping=recipe["linear_damping"], angular_damping=recipe["angular_damping"]),
                        mass_props=sim_utils.MassPropertiesCfg(mass=recipe["mass_kg"]),
                        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=recipe["source_contact_offset_m"],
                            rest_offset=recipe["source_rest_offset_m"]),
                        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=recipe["static_friction"],
                            dynamic_friction=recipe["dynamic_friction"], restitution=recipe["restitution"])),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.,0.,-10. if sid == "table1" else -5.))))
            if robot_asset is not None:
                from supermimic.physics import load_hand_asset, sha
                if sha(robot_asset["path"]) != robot_asset["sha256"]:
                    raise ValueError("Bundle's fixed robot hand overlay bytes changed")
                load_hand_asset(robot_asset["path"])
                config.robot.spawn.usd_path = str(robot_asset["path"])
            return config

        class AssetTrackingSim(TrackingSim):
            def sample_frames(this, n):
                m = this.motion
                bins = compatible
                soft = torch.softmax(-25 * this.bin_G[bins] / this.G_max, 0)
                this.bin_w_avg[bins] = .999 * this.bin_w_avg[bins] + .001 * soft
                avg = this.bin_w_avg[bins]
                probs = .5 / len(bins) + .4 * soft + .1 * avg / avg.sum()
                sampled = bins[torch.multinomial(probs, n, replacement=True)]
                start = m.bin_start[sampled]
                end = torch.minimum(start + 50, m.clip_end[m.bin_clip[sampled]])
                # Last-frame bins have no outgoing transition; use clip-1.
                start = torch.minimum(start, m.clip_end[m.bin_clip[sampled]] - 1)
                frame = start + (torch.rand(n, device=this.device) * (end - start).float()).long()
                return frame, sampled

        self.env = AssetTrackingSim(sim_cfg, num_envs, motion, mdp.human, gamma=gamma, eval_mode=eval_mode,
                                   max_episode_steps=max_episode_steps, reset_noise=reset_noise, scene_factory=scene_factory)
        self.device, self.N, self.motion, self.mdp = self.env.device, num_envs, motion, mdp
        self.asset = dict(asset)
        self.cooking_only = cooking_only
        self.output = Path(output)
        self.object_usd = usd
        self.asset_id = torch.full((num_envs,), asset_id, device=self.device, dtype=torch.long)
        self.object = self.env.scene["object"]
        self.support_bank = support_bank
        self.supports = {sid: self.env.scene["support_" + sid] for sid in support_bank.support_ids}
        self.support_reset_poses = {sid: torch.zeros(num_envs,7,device=self.device) for sid in self.supports}
        self.support_reset_present = torch.zeros(num_envs,2,dtype=torch.bool,device=self.device)
        self.support_alignment = {"reset_count": 0, "step_count": 0, "max_position_error_m": 0., "max_quaternion_error": 0.}
        self.env.G_max = mdp.max_step_reward / (1 - gamma)
        self.contact_missing_counter = torch.zeros(num_envs, len(CHANNELS), device=self.device, dtype=torch.int32)
        self.physics_report = self._apply_and_read_physics()
        self.physics_report["warp_compilation"] = self.warp_compilation
        # MuJoCo-Warp exposes this supported scheduling option independently of
        # Newton's outer graph setting. No package/runtime files are patched.
        solver = self.env.sim.backend_manager._solver
        solver.mjw_model.opt.graph_conditional = bool(wp.is_conditional_graph_supported())
        self.contacts = PairContactRecorder(self.env.sim.backend_manager, num_envs)
        self.support_geometry_audit = self._audit_support_geometry()
        self.native_robot_asset = robot_asset
        self.foot_asset_manifest = None
        self.compiled_foot_audit = {"passed": False, "reason": "no explicit ULTRA-aligned foot asset binding"}
        if robot_asset is not None and robot_asset.get("foot_asset_manifest"):
            from .foot_collision import validate_foot_asset
            import hashlib
            import mujoco
            self.foot_asset_manifest = validate_foot_asset(robot_asset)
            reference_path = Path(self.mdp.human.kin.mjcf_path)
            reference_model = mujoco.MjModel.from_xml_path(str(reference_path))
            self.compiled_foot_audit = compiled_foot_geometry_audit(self.contacts.solver.mj_model,
                num_envs=num_envs, topology=self.contacts.topology, manifest=self.foot_asset_manifest,
                manifest_sha256=robot_asset["foot_asset_manifest"]["sha256"], robot_asset_sha256=robot_asset["sha256"],
                reference_model=reference_model, reference_mjcf_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest())
        self.physics_report["compiled_foot_audit"] = self.compiled_foot_audit
        self.reset(torch.arange(num_envs, device=self.device), self.env.f.clone())
        self.physics_report["robot_fk_comparison"] = self._verify_native_fk()

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _apply_and_read_physics(self):
        import warp as wp
        from newton import ModelFlags
        from .native_acceptance import physical_profile_identity
        from .native_fixture import DEFERRED_INERTIA_MODE
        deferred_readback = None
        if self.asset.get("inertia_mode") == DEFERRED_INERTIA_MODE:
            if not self.cooking_only:
                raise ValueError("deferred inertia cannot supply native rollout physics")
            import mujoco
            mj = self.env.sim.backend_manager._solver.mj_model
            body_ids = [i for i in range(mj.nbody) if (mujoco.mj_id2name(mj,mujoco.mjtObj.mjOBJ_BODY,i) or "").endswith("Object")]
            if not body_ids:
                raise RuntimeError("cooking import has no bound target Object body")
            records=[]
            for body in body_ids:
                R=quat_to_mat(torch.tensor(mj.body_iquat[body],dtype=torch.float64))
                I=R@torch.diag(torch.tensor(mj.body_inertia[body],dtype=torch.float64))@R.T
                records.append({"mass":float(mj.body_mass[body]),"com":mj.body_ipos[body].tolist(),"inertia":I.tolist()})
            reference=records[0]
            if any(not torch.allclose(torch.tensor(r[k]),torch.tensor(reference[k]),atol=1e-7,rtol=1e-6) for r in records for k in ('mass','com','inertia')):
                raise RuntimeError("native temporary cooking inertia differs across environments")
            original=self.asset["physical_profile"]
            if abs(reference['mass']-float(original['mass']))>1e-6*max(1.,float(original['mass'])):
                raise RuntimeError("native cooking import changed the explicitly supplied source mass")
            self.asset["physical_profile"]={**original,**reference}
            deferred_readback={"mode":DEFERRED_INERTIA_MODE,"native_import_mass_properties":reference,
                "status":"temporary_import_readback_only; not_final_uniform_compound_inertia",
                "physics_steps_permitted":False,"final_profile_requires_closed_compound_hull_integration":True}
        mass, com, inertia, friction, ground = validate_physical_profile(self.asset["physical_profile"])
        obj, env = self.object, self.env
        m = torch.full_like(obj.data.body_mass.torch, mass)
        c = com.to(device=self.device, dtype=torch.float32).reshape(1, 1, 3).expand(self.N, 1, 3).contiguous()
        I = inertia.to(device=self.device, dtype=torch.float32).reshape(1, 1, 9).expand(self.N, 1, 9).contiguous()
        obj.set_masses_index(masses=m)
        obj.set_coms_index(coms=c)
        obj.set_inertias_index(inertias=I)
        manager = env.sim.backend_manager
        manager.add_model_change(ModelFlags.BODY_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES)
        model = manager.get_model()
        material = wp.to_torch(obj._root_view.get_attribute("shape_material_mu", model))
        material[:] = friction
        all_mu = wp.to_torch(model.shape_material_mu)
        shape_body = wp.to_torch(model.shape_body)
        all_mu[shape_body < 0] = ground
        for sid, support in self.supports.items():
            support_material = wp.to_torch(support._root_view.get_attribute("shape_material_mu", model))
            support_material[:] = omni_support_recipe()["dynamic_friction"]
        manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)
        read_m = obj.data.body_mass.torch[:, 0]
        read_c = obj.data.body_com_pose_b.torch[:, 0, :3]
        read_I = obj.data.body_inertia.torch.reshape(self.N, 1, 3, 3)[:, 0]
        for name, actual, expected in (("mass", read_m, m[:, 0]), ("COM", read_c, c[:, 0]), ("inertia", read_I, I[:, 0].reshape(-1, 3, 3))):
            if not torch.allclose(actual, expected, atol=1e-6, rtol=1e-5):
                raise RuntimeError(f"Object native {name} readback differs from physical profile")
        read_mu = material.reshape(self.N, -1)
        if not torch.allclose(read_mu, torch.full_like(read_mu, friction), atol=1e-7, rtol=0):
            raise RuntimeError("Object friction readback differs")
        ground_mask = shape_body < 0
        # Newton shape views retain exact entity indices; support material above
        # is checked independently by the compiled support geometry audit.
        read_ground = all_mu[ground_mask]
        if not read_ground.numel() or not torch.allclose(read_ground, torch.full_like(read_ground, ground), atol=1e-7, rtol=0):
            raise RuntimeError("Scene ground friction readback differs")
        mu = read_mu[:, 0].clone()
        self.xi_o = physical_context(read_m, read_c, read_I, mu, torch.full_like(mu, float(read_ground[0])))
        gravity = wp.to_torch(model.gravity)
        if not torch.isfinite(gravity).all() or not bool((gravity[..., 2] < 0).all()):
            raise RuntimeError("Native scene gravity must act downwards")
        import mujoco
        mj = manager._solver.mj_model
        objects = [i for i in range(mj.nbody) if "Object" in (mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, i) or "")]
        if not objects or any(float(mj.body_mass[i]) <= 0 or int(mj.body_jntnum[i]) != 1 or
                              int(mj.jnt_type[int(mj.body_jntadr[i])]) != int(mujoco.mjtJoint.mjJNT_FREE) for i in objects):
            raise RuntimeError("Object must compile as a positive-mass dynamic free rigid body")
        from supermimic.physics import sha
        fixed_native = {"cone": int(mj.opt.cone), "integrator": int(mj.opt.integrator),
                        "solver_iterations": int(mj.opt.iterations), "line_search_iterations": int(mj.opt.ls_iterations),
                        "solver_tolerance": float(mj.opt.tolerance), "geom_margin_m": mj.geom_margin.tolist(),
                        "geom_gap_m": mj.geom_gap.tolist(), "geom_contact_dimensions": mj.geom_condim.tolist(),
                        "geom_solref": mj.geom_solref.tolist(), "geom_solimp": mj.geom_solimp.tolist(),
                        "body_exclusion_signatures": mj.exclude_signature.tolist(),
                        "constraint_capacity_per_world": int(manager._solver.mjw_data.njmax),
                        "total_contact_capacity": int(manager._solver.mjw_data.naconmax)}
        return {"deferred_cooking_inertia": deferred_readback,
                "bound_physical_profile": self.asset["physical_profile"],
                "bound_physical_profile_sha256": physical_profile_identity(self.asset["physical_profile"]),
                "mass": read_m.detach().cpu().tolist(), "com": read_c.detach().cpu().tolist(),
                "inertia": read_I.detach().cpu().tolist(), "object_friction": mu.detach().cpu().tolist(),
                "ground_friction": ground, "gravity": gravity.detach().cpu().tolist(),
                "object_pose_origin": "link", "velocity_reference_point": "link_origin", "frame": "world",
                "control_dt": CONTROL_DT, "substeps": 4, "integration_dt": CONTROL_DT / 4,
                "use_cuda_graph": False, "linear_damping": 0., "angular_damping": 0.,
                "physics": env.physics_settings, "collision_hash": self.asset["collision_hash"],
                "fixed_native_parameters": fixed_native,
                "capacity_configuration": {"legacy_motion_only": LEGACY_CAPACITY, "hoi_v1": HOI_CAPACITY,
                    "scope": "HOI-only allocation; PD, integration, solver settings and physical properties preserved",
                    "reason": "Observed NEFC overflow at njmax500 during actual plasticbox080 reference-reset zero-action rollout",
                    "overflow_policy": "abort without replay insertion or successful native gate"},
                "flattened_object_usd": str(self.object_usd), "flattened_object_usd_sha256": sha(self.object_usd),
                "compiled_collision_geometry_verified_against_bundle": False}

    def support_state(self):
        """Observed support state in the same world axes/origin as robot state."""
        from .scene_context import empty_scene_state
        result=empty_scene_state((self.N,),device=self.device)
        reference=result["support_pose_xyzw"].clone()
        for sid,support in self.supports.items():
            slot=SUPPORT_IDS.index(sid)
            result["support_pose_xyzw"][:,slot]=support.data.root_link_pose_w.torch
            reference[:,slot]=self.support_reset_poses[sid]
            result["support_pose_xyzw"][:,slot,:3]-=self.scene.env_origins
            reference[:,slot,:3]-=self.scene.env_origins
        result["support_present"]=self.support_reset_present.clone()
        result["reference_support_pose_xyzw"]=reference
        return result

    def _check_support_alignment(self, env_ids=None, *, reset=False):
        ids = torch.arange(self.N,device=self.device) if env_ids is None else env_ids
        for sid, support in self.supports.items():
            pose = support.data.root_link_pose_w.torch[ids]
            expected = self.support_reset_poses[sid][ids]
            velocity = support.data.root_link_vel_w.torch[ids]
            if not bool(torch.isfinite(pose).all() and torch.isfinite(velocity).all()):
                raise RuntimeError("Nonfinite native scene support state")
            ep = float((pose[:,:3]-expected[:,:3]).abs().max())
            # q and -q encode the same rotation.
            eq = float(torch.minimum((pose[:,3:]-expected[:,3:]).abs().amax(-1),
                                     (pose[:,3:]+expected[:,3:]).abs().amax(-1)).max())
            self.support_alignment["max_position_error_m"] = max(self.support_alignment["max_position_error_m"],ep)
            self.support_alignment["max_quaternion_error"] = max(self.support_alignment["max_quaternion_error"],eq)
            if ep > 2e-5 or eq > 2e-5 or not torch.allclose(velocity,torch.zeros_like(velocity),atol=2e-5,rtol=0):
                raise RuntimeError(f"Native kinematic {sid} drifted from its sampled reference reset frame")
        self.support_alignment["reset_count" if reset else "step_count"] += 1

    def _write_support_probe_pose(self, sid, pose):
        ids = torch.arange(self.N,device=self.device)
        support = self.supports[sid]
        support.write_root_link_pose_to_sim_index(root_pose=pose.contiguous(),env_ids=ids)
        support.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros(self.N,6,device=self.device),env_ids=ids)
        self.support_reset_poses[sid][:] = pose

    def _park_supports_for_probe(self):
        # Disposable calibration only. Every normal reset restores source poses.
        for sid in self.supports:
            pose = torch.zeros(self.N,7,device=self.device);pose[:,6]=1
            pose[:,:3] = self.scene.env_origins
            pose[:,2] += -10. if sid == "table1" else -5.
            self._write_support_probe_pose(sid,pose)
        self.env.refresh()
        self._check_support_alignment()

    def _audit_support_geometry(self):
        import mujoco
        import warp as wp
        from newton import BodyFlags
        model = self.contacts.solver.mj_model
        records = []
        for sid in self.supports:
            role = SUPPORT_ROLES[SUPPORT_IDS.index(sid)]
            geoms = [i for i,r in enumerate(self.contacts.roles) if r == role]
            expected = 1 if self.contacts.topology == "one_mujoco_world_per_environment" else self.N
            if self.contacts.topology == "one_world_with_named_environments" and {self.contacts.geom_envs[g] for g in geoms} != set(range(self.N)):
                raise RuntimeError(f"{sid} collider identity does not cover every environment")
            if len(geoms) != expected:
                raise RuntimeError(f"{sid} must retain exactly one bound cuboid collider per native world")
            native_flags = wp.to_torch(self.supports[sid]._root_view.get_attribute("body_flags",self.env.sim.backend_manager.get_model()))
            if native_flags.numel() != self.N or not bool(((native_flags.long() & int(BodyFlags.KINEMATIC)) != 0).all()):
                raise RuntimeError(f"{sid} did not import as a native kinematic body")
            instances = []
            for g in geoms:
                if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                    raise RuntimeError("Native table must compile as the exact authored cuboid")
                actual_size = torch.as_tensor(model.geom_size[g],dtype=torch.float64)*2
                if not torch.allclose(actual_size,torch.tensor(omni_support_recipe()["size_m"],dtype=torch.float64),atol=1e-7,rtol=0):
                    raise RuntimeError("Native support cuboid dimensions differ from bound author recipe")
                if abs(float(model.geom_friction[g,0])-omni_support_recipe()["dynamic_friction"]) > 1e-6:
                    raise RuntimeError("Native support friction differs from bound author material")
                body = int(model.geom_bodyid[g])
                if body == 0 or self.contacts.links[g] is not None:
                    raise RuntimeError("Scene support lost its independent native entity identity")
                instances.append({"geom_id":g,"geom_name":self.contacts.names[g],"role":role,
                    "body_id":body,"body_name":mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body),
                    "body_mocap_id":int(model.body_mocapid[body]),"body_mass_kg":float(model.body_mass[body]),
                    "size_m":actual_size.tolist(),"geom_friction":model.geom_friction[g].tolist(),
                    "geom_margin_m":float(model.geom_margin[g]),"geom_gap_m":float(model.geom_gap[g])})
            records.append({"support_id":sid,"geometry_recipe_sha256":self.support_bank.manifest["geometry_recipe_sha256"],
                            "compiled_geom_ids":geoms,"native_body_flags":native_flags.cpu().tolist(),
                            "kinematic_verified":True,"instances":instances})
        return records

    def _body_support_probe_placement(self, ids):
        """Place a held legal hand pose from compiled geometry, never stale live FK."""
        from .probe_pose import CompiledProbePlanner
        planner = getattr(self, "_compiled_probe_planner", None)
        if planner is None:
            planner = CompiledProbePlanner(self.contacts.solver.mj_model, self.contacts.roles,
                joint_names=list(self.robot.joint_names), environment_ids=self.contacts.geom_envs)
            planner.search(self.env.default_q.detach().cpu().reshape(-1).tolist())
            self._compiled_probe_planner = planner
        root, rotation, joint, surface, normal, audit = planner.placement(LEFT, ground=True)
        cast = lambda value: torch.as_tensor(value, device=self.device, dtype=torch.float32)
        # The exact same rigid translation applies to root and surface. All
        # geometry was evaluated at this prescribed q/R in the compiled model;
        # env.refresh() is deliberately not used as an mjw geom-cache refresh.
        translation = cast([0., 0., 3.])
        root_world = cast(root).expand(self.N, -1) + self.scene.env_origins + translation
        rotation = cast(rotation).expand(self.N, -1, -1)
        root_q = mat_to_quat(rotation)
        joint = cast(joint).expand(self.N, -1).contiguous()
        surface_world = cast(surface).expand(self.N, -1) + self.scene.env_origins + translation
        normal = cast(normal).expand(self.N, -1)
        self.robot.write_root_link_pose_to_sim_index(root_pose=torch.cat(
            (root_world, root_q[:, 1:4], root_q[:, :1]), -1).contiguous(), env_ids=ids)
        self.robot.write_root_link_velocity_to_sim_index(
            root_velocity=torch.zeros(self.N, 6, device=self.device), env_ids=ids)
        self.robot.write_joint_position_to_sim_index(position=joint, env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint), env_ids=ids)
        actions = (joint - self.env.default_q - self.env.xi[:, 1:30]) / self.env.alpha
        self.env.refresh()
        initial, _ = self.state()
        errors = {
            "robot_root_position_m": float((initial[0] + self.scene.env_origins - root_world).abs().max()),
            "robot_root_rotation": float((initial[1] - rotation).abs().max()),
            "robot_root_linear_velocity_m_s": float(initial[2].abs().max()),
            "robot_root_angular_velocity_rad_s": float(initial[3].abs().max()),
            "robot_joint_position_rad": float((initial[4] - joint).abs().max()),
            "robot_joint_velocity_rad_s": float(initial[5].abs().max()),
            "held_pd_target_rad": float((self.env.default_q + self.env.xi[:, 1:30] + self.env.alpha * actions - joint).abs().max())}
        if max(errors.values()) > 2e-5:
            raise RuntimeError(f"native support probe initial state differs from prescribed pose/velocity: {errors}")
        support_pose = torch.zeros(self.N, 7, device=self.device)
        # Planner exposes the actual hand downward; gravity therefore pushes
        # the free robot toward the static table instead of away from it.
        if not torch.allclose(normal, cast([0., 0., -1.]).expand_as(normal), atol=1e-6, rtol=0):
            raise RuntimeError("support probe requires the compiled exposed downward hand surface")
        support_pose[:, 6] = 1.
        support_pose[:, :3] = surface_world + normal * (.005 - .002)
        definition = {"placement": "exposed_compiled_hand_surface_vs_table_face", "region": CHANNELS[0],
            "region_pose_audit": audit, "prescribed_penetration_m": .002,
            "rigid_translation_world_m": translation.cpu().tolist(),
            "surface_point_before_translation": cast(surface).cpu().tolist(),
            "surface_point_world": surface_world.cpu().tolist(), "outward_normal_world": normal.cpu().tolist(),
            "initial_state_readback": {"robot_root_position_world": (initial[0] + self.scene.env_origins).cpu().tolist(),
                "robot_root_rotation": initial[1].cpu().tolist(), "robot_joint_positions": initial[4].cpu().tolist(),
                "robot_joint_velocities": initial[5].cpu().tolist(), "maximum_errors": errors},
            "held_pd_offset_actions": actions.cpu().tolist(), "extra_physics_steps": 0}
        return support_pose, actions, definition

    def calibrate_support_probes(self, ticks=8):
        """Real isolated support collisions; these must never label the 16 head."""
        ids = torch.arange(self.N,device=self.device)
        frames = self.f.clone(); actions=torch.zeros(self.N,29,device=self.device)
        report = {"schema":"hoi_native_scene_supports_v1", **self.support_bank.manifest,
                  "supports":self.support_geometry_audit,"isolation_probes":{}}
        report["schema"] = "hoi_native_scene_supports_v1"
        vertices = torch.cat([m["vertices"] for m in self.mdp.shapes.geometries[int(self.asset["asset_id"])]["collision_meshes"]]).to(self.device,dtype=torch.float32)
        for sid in self.supports:
            support_column=SUPPORT_IDS.index(sid)
            report["isolation_probes"][sid]={}
            for case in ("object_support","body_support"):
                self.active_pair_probe=sid+"_"+case
                self.reset(ids,frames);self._park_supports_for_probe()
                actions = torch.zeros(self.N,29,device=self.device)
                self._advance(actions)
                S,O=self.state()
                # Move the entire robot above the plane to avoid body-ground
                # contacts and isolate the finite table from other entities.
                root_q=mat_to_quat(S[1]); root=S[0]+self.scene.env_origins
                root[:,2]+=3.
                self.robot.write_root_link_pose_to_sim_index(root_pose=torch.cat((root,root_q[:,1:4],root_q[:,:1]),-1).contiguous(),env_ids=ids)
                self.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.zeros(self.N,6,device=self.device),env_ids=ids)
                self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros(self.N,29,device=self.device),env_ids=ids)
                object_pose=torch.zeros(self.N,7,device=self.device);object_pose[:,6]=1
                object_pose[:,:3]=self.scene.env_origins+object_pose.new_tensor([8.,8.,5.])
                support_pose=torch.zeros(self.N,7,device=self.device);support_pose[:,6]=1
                self.env.refresh()
                if case=="object_support":
                    support_pose[:,:3]=self.scene.env_origins+support_pose.new_tensor([4.,4.,3.])
                    # Actual target object collision geometry, identity object
                    # orientation. Its lowest support rests 2 mm into table top.
                    lowest = vertices[int(vertices[:,2].argmin())]
                    object_pose[:,:3]=support_pose[:,:3]-lowest
                    object_pose[:,2]+=.005-.002
                    diagnostic_column=len(BODY_ROLES)
                    definition={"placement":"object_lowest_surface_vs_table_top","prescribed_penetration_m":.002}
                else:
                    support_pose, actions, definition = self._body_support_probe_placement(ids)
                    diagnostic_column=0
                self.object.write_root_link_pose_to_sim_index(root_pose=object_pose.contiguous(),env_ids=ids)
                self.object.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros(self.N,6,device=self.device),env_ids=ids)
                self._write_support_probe_pose(sid,support_pose);self.env.refresh()
                actual_support = self.supports[sid].data.root_link_pose_w.torch
                support_error = float((actual_support[:,:3]-support_pose[:,:3]).abs().max())
                rotation_error = float(torch.minimum((actual_support[:,3:]-support_pose[:,3:]).abs().amax(-1),
                    (actual_support[:,3:]+support_pose[:,3:]).abs().amax(-1)).max())
                if max(support_error,rotation_error) > 2e-5:
                    raise RuntimeError("native support probe table writeback differs from prescribed pose")
                definition["support_initial_readback"] = {"pose_xyzw":actual_support.cpu().tolist(),
                    "position_error_m":support_error,"quaternion_error":rotation_error}
                learned,diagnostic=[],[]
                for _ in range(ticks):
                    pair=self._advance(actions)
                    learned.append(pair["impulse"].cpu().tolist())
                    diagnostic.append(pair["support_impulse"][:,support_column,diagnostic_column].cpu().tolist())
                learned_t=torch.tensor(learned); diagnostic_t=torch.tensor(diagnostic)
                passed=bool((learned_t==0).all() and (diagnostic_t.amax(0)/CONTROL_DT>self.contacts.threshold).all())
                report["isolation_probes"][sid][case]={"ticks":ticks,"definition":definition,
                    "control_dt_s":CONTROL_DT,"substeps_per_interval":4,"interval_normal_impulse_Ns":learned,
                    "interval_support_impulse_Ns":diagnostic,"support_initial_pose_xyzw":support_pose.cpu().tolist(),
                    "object_initial_pose_xyzw":object_pose.cpu().tolist(),"passed":passed}
        self.active_pair_probe=None;self.reset(ids,frames)
        report["alignment"] = dict(self.support_alignment)
        report["reference_reset_audit"] = {
            "environment_ids":ids.cpu().tolist(),"frames":frames.cpu().tolist(),
            "clip_indices":self.motion.clip_id[frames].cpu().tolist(),
            "clip_local_frames":(frames-self.motion.clip_start[self.motion.clip_id[frames]]).cpu().tolist(),
            "supports":{sid:{"expected_pose_xyzw":self.support_bank.sample(sid,frames,self.scene.env_origins).cpu().tolist(),
                "actual_pose_xyzw":support.data.root_link_pose_w.torch.cpu().tolist(),
                "actual_velocity":support.data.root_link_vel_w.torch.cpu().tolist()}
                for sid,support in self.supports.items()}}
        report["passed"] = all(p["passed"] for cases in report["isolation_probes"].values() for p in cases.values())
        return report

    def state(self):
        S = self.env.state()
        data = self.object.data
        pose, vel = data.root_link_pose_w.torch.clone(), data.root_link_vel_w.torch.clone()
        R = quat_to_mat(torch.cat((pose[:, 6:7], pose[:, 3:6]), -1))
        O = (pose[:, :3] - self.env.scene.env_origins, R, vel[:, :3], vel[:, 3:6])
        return S, O

    def _verify_native_fk(self):
        """Compare reward/palm FK with effective native link frames."""
        kin = self.mdp.human.kin
        S, _ = self.state()
        p, R, _, _ = kin.forward(*S, all_bodies=True)
        native = self.env.robot.data.body_link_pose_w.torch
        names = list(self.env.robot.body_names)
        shared = [name for name in kin.body_names if name in names]
        if not all(side + "_wrist_yaw_link" in shared for side in ("left", "right")):
            raise RuntimeError("Actual wrist link frames unavailable for geometry alignment")
        expected_ids = [kin.body_names.index(n) for n in shared]
        native_ids = [names.index(n) for n in shared]
        q = native[:, native_ids, 3:7]
        native_R = quat_to_mat(torch.cat((q[..., 3:4], q[..., :3]), -1))
        ep = torch.linalg.vector_norm(p[:, expected_ids] - (native[:, native_ids, :3] - self.scene.env_origins[:, None]), dim=-1)
        er = (R[:, expected_ids] - native_R).abs().amax(dim=(-1, -2))
        result = {"shared_body_names": shared, "max_link_position_error_m": float(ep.max()),
                  "max_rotation_matrix_error": float(er.max()), "position_tolerance_m": 5e-5, "rotation_tolerance": 5e-5}
        if result["max_link_position_error_m"] > 5e-5 or result["max_rotation_matrix_error"] > 5e-5:
            raise RuntimeError(f"Native robot and analytic reward/palm FK differ: {result}")
        result["passed"] = True
        return result

    def reset(self, env_ids, frames=None):
        if not len(env_ids):
            return
        if frames is not None and not bool((self.motion.asset_ids[frames] == self.asset_id[env_ids]).all()):
            raise ValueError("Cannot reset a geometry group to an incompatible motion")
        self.env.reset(env_ids, frames)
        f = self.env.f[env_ids]
        p, R, v, w = self.motion.ref_object(f)
        q = mat_to_quat(R)
        pose = torch.cat((p + self.env.scene.env_origins[env_ids], q[:, 1:4], q[:, :1]), -1).contiguous()
        self.object.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        if hasattr(self.object, "write_root_link_velocity_to_sim_index"):
            self.object.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((v, w), -1).contiguous(), env_ids=env_ids)
        else:
            v_com = link_to_com_velocity(R, v, w, self.xi_o[env_ids, 1:4])
            self.object.write_root_com_velocity_to_sim_index(root_velocity=torch.cat((v_com, w), -1).contiguous(), env_ids=env_ids)
        for sid, support in self.supports.items():
            support_pose = self.support_bank.sample(sid, f, self.scene.env_origins[env_ids])
            support.write_root_link_pose_to_sim_index(root_pose=support_pose.contiguous(), env_ids=env_ids)
            support.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros(len(env_ids),6,device=self.device), env_ids=env_ids)
            self.support_reset_poses[sid][env_ids] = support_pose
            self.support_reset_present[env_ids,SUPPORT_IDS.index(sid)] = self.support_bank.present[sid][f]
        self.env.scene.reset(env_ids)
        self.env.refresh()
        self._check_support_alignment(env_ids, reset=True)
        self.contact_missing_counter[env_ids] = 0
        _, actual = self.state()
        if not torch.allclose(actual[0][env_ids], p, atol=2e-5, rtol=0):
            raise RuntimeError("Object and robot reset did not reach the same reference phase")
        if not torch.allclose(actual[2][env_ids], v, atol=2e-5, rtol=1e-5):
            raise RuntimeError("Object native reset link-origin velocity disagrees with reference")
        if not torch.allclose(actual[1][env_ids], R, atol=2e-5, rtol=0) or not torch.allclose(actual[3][env_ids], w, atol=2e-5, rtol=1e-5):
            raise RuntimeError("Object native reset orientation/angular velocity disagrees with reference")

    def _overflow_gate(self):
        import mujoco_warp
        import warp as wp
        value = wp.to_torch(self.contacts.solver.mjw_data.overflow)
        # All recorded capacity overflow bits invalidate dynamics/contact labels.
        capacity = 0
        for name in ("NEFC", "NJMAX_NNZ", "BROADPHASE", "NARROWPHASE", "CCD", "HFIELD", "NVMAX", "EPA_HORIZON"):
            capacity |= int(getattr(mujoco_warp.OverflowType, name, 0))
        if bool(((value & capacity) != 0).any()):
            failure = {"error": "MuJoCo-Warp collision/constraint capacity overflow", "bits": value.cpu().tolist(),
                       "object_asset": self.asset["object_asset"], "pair_probe": getattr(self, "active_pair_probe", None),
                       "reference_frames": self.env.f.detach().cpu().tolist(),
                       "constraint_capacity_per_world": int(self.contacts.solver.mjw_data.njmax),
                       "total_contact_capacity": int(self.contacts.solver.mjw_data.naconmax),
                       "action": "abort; invalid dynamics/contact labels must not enter replay"}
            with (self.output / "failure.json").open("x", encoding="utf-8") as stream:
                json.dump(failure, stream, indent=2)
            raise RuntimeError(f"MuJoCo-Warp collision/constraint capacity overflow: {value.cpu().tolist()}")

    def _advance(self, action):
        if self.__dict__.get("cooking_only", False) or self.asset.get("inertia_mode") is not None:
            raise RuntimeError("native cooking fixture is geometry-only; physics rollout is prohibited")
        e = self.env
        target = e.default_q + e.xi[:, 1:30] + e.alpha * action
        e.robot.set_joint_position_target_index(target=target)
        e.scene.write_data_to_sim()
        self.contacts.begin_interval()
        e.sim.step(render=False)
        e.scene.update(CONTROL_DT)
        self._overflow_gate()
        self._check_support_alignment()
        S, O = self.state()
        names = ("robot_p", "robot_R", "robot_v", "robot_w", "robot_q", "robot_qd",
                 "object_p", "object_R", "object_v", "object_w")
        bad = [name for name, value in zip(names, S + O) if not bool(torch.isfinite(value).all())]
        if bad:
            failure = {"error": "nonfinite native physical state", "fields": bad, "asset_id": int(self.asset["asset_id"]),
                       "action": "abort without reset, replay insertion, or successful native gate"}
            with (self.output / "failure.json").open("x", encoding="utf-8") as stream:
                json.dump(failure, stream, indent=2)
            torch.save({"robot": tuple(v.detach().cpu() for v in S), "object": tuple(v.detach().cpu() for v in O),
                        "action": action.detach().cpu(), "frame": self.env.f.detach().cpu()}, self.output / "failure_state.pt")
            raise RuntimeError(f"Native physics produced nonfinite state: {bad}")
        interval = self.contacts.end_interval()
        # Optional one-shot smoke evidence, after real integration and before
        # step() can reset. Never armed during cooking, training or probes.
        capture = self.__dict__.pop("_post_advance_capture", None)
        if capture is not None:
            capture(self)
        return interval

    def _role_point(self, role):
        import warp as wp
        data = wp.to_torch(self.contacts.solver.mjw_data.geom_xpos)
        ids = torch.where(self.contacts.geom_role == role)[0]
        if self.contacts.topology == "one_mujoco_world_per_environment":
            return data[:, ids[0]].clone()
        points = []
        for env_id in range(self.N):
            selected = ids[self.contacts.geom_env[ids] == env_id]
            if not len(selected):
                raise RuntimeError("Probe geom cannot be mapped to a native environment")
            points.append(data[0, selected[0]])
        return torch.stack(points).clone()

    def _region_surface(self, role, *, lowest=False):
        """Read actual live solver geometry, including welded/merged body transforms."""
        import warp as wp
        model, data = self.contacts.solver.mj_model, self.contacts.solver.mjw_data
        positions = wp.to_torch(data.geom_xpos)
        rotations = wp.to_torch(data.geom_xmat).reshape(*positions.shape[:-1], 3, 3)
        root = self.state()[0][0] + self.scene.env_origins
        surfaces, normals, records = [], [], []
        for env_id in range(self.N):
            world = env_id if self.contacts.topology == "one_mujoco_world_per_environment" else 0
            ids = [g for g, r in enumerate(self.contacts.roles) if r == role and
                   (self.contacts.topology == "one_mujoco_world_per_environment" or self.contacts.geom_envs[g] == env_id)]
            if not ids:
                raise RuntimeError("Full-body probe has no bound native collider")
            if not lowest:
                ids = [max(ids, key=lambda g: float(model.geom_rbound[g]))]
            candidates = []
            for geom in ids:
                center, rotation = positions[world, geom], rotations[world, geom]
                direction = center.new_tensor([0., 0., -1.]) if lowest else center - root[env_id]
                if float(direction.norm()) < 1e-6:
                    direction = center.new_tensor([1., 0., 0.])
                local_direction = (rotation.T @ direction).detach().cpu().double()
                local, normal, proof = compiled_geom_support(model, geom, local_direction, facet=not lowest)
                surface = center + rotation @ local.to(center)
                world_normal = rotation @ normal.to(center)
                proof.update(geom_name=self.contacts.names[geom], canonical_link=self.contacts.links[geom])
                candidates.append((surface, world_normal, proof))
            surface, normal, proof = min(candidates, key=lambda x: float(x[0][2])) if lowest else candidates[0]
            surfaces.append(surface); normals.append(normal); records.append(proof)
        return torch.stack(surfaces), torch.stack(normals), records

    def calibrate_pair_probes(self, ticks=8):
        """Shallow surface-contact probes in a disposable validation environment.

        The source dataset and legacy scene are untouched. Probe transitions
        never enter replay. All channels integrate real solver forces across
        four substeps; hand/ground and each foot/ground are independently
        measured negative controls. Placements and outcomes are saved verbatim.
        """
        from .probe_pose import CompiledProbePlanner, object_support_facet
        ids = torch.arange(self.N, device=self.device)
        frames = self.f.clone()
        baseline_actions = torch.zeros(self.N, 29, device=self.device)
        report = {}
        geometries = self.mdp.shapes.geometries[int(self.asset["asset_id"])]
        vertices = torch.cat([m["vertices"] for m in geometries["collision_meshes"]]).to(self.device, dtype=torch.float32)
        object_rotation, object_support, object_facet = object_support_facet(geometries["collision_meshes"])
        planner = CompiledProbePlanner(self.contacts.solver.mj_model, self.contacts.roles,
            joint_names=list(self.robot.joint_names), environment_ids=self.contacts.geom_envs)
        try:
            planner.search(self.env.default_q.detach().cpu().reshape(-1).tolist())
        finally:
            (self.output / "probe-pose-search.json").write_text(
                json.dumps(planner.search_evidence, indent=2) + "\n", encoding="utf-8")
        self._compiled_probe_planner = planner
        cases = tuple(zip(CHANNELS, CHANNEL_ROLES)) + tuple(
            (("hand_ground_negative" if role == LEFT else name.removesuffix("_object") + "_ground_negative"), role)
            for name, role in zip(CHANNELS[:-1], BODY_ROLES))
        for name, role in cases:
            self.active_pair_probe = name
            self.reset(ids, frames)
            self._park_supports_for_probe()
            # The source reference can have substantial joint/root velocities.
            # Probe initial conditions hold the searched legal articulation;
            # PD gains, action scales and collision filters remain unchanged.
            S, O = self.state()
            q = mat_to_quat(O[1])
            position = O[0] + self.scene.env_origins
            probe_velocity = torch.zeros(self.N, 6, device=self.device)
            actions = baseline_actions
            if role in BODY_ROLES:
                negative = name not in CHANNELS
                root, rotation, joint, surface, normal, audit = planner.placement(role, ground=negative)
                cast = lambda value: torch.as_tensor(value, device=self.device, dtype=torch.float32)
                root_world = cast(root).expand(self.N, -1) + self.scene.env_origins
                root_q = mat_to_quat(cast(rotation).expand(self.N, -1, -1))
                joint = cast(joint).expand(self.N, -1).contiguous()
                self.robot.write_root_link_pose_to_sim_index(root_pose=torch.cat(
                    (root_world, root_q[:, 1:4], root_q[:, :1]), -1).contiguous(), env_ids=ids)
                self.robot.write_root_link_velocity_to_sim_index(
                    root_velocity=torch.zeros(self.N, 6, device=self.device), env_ids=ids)
                self.robot.write_joint_position_to_sim_index(position=joint, env_ids=ids)
                self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint), env_ids=ids)
                # _advance applies exactly default_q + xi + alpha * action.
                actions = (joint - self.env.default_q - self.env.xi[:, 1:30]) / self.env.alpha
                definition = {"placement": "exposed_compiled_region_ground_negative" if negative else
                              "exposed_compiled_region_vs_object_support_surface",
                              "prescribed_penetration_m": .002, "region_pose_audit": audit}
                if negative:
                    position = position.new_tensor([8., 8., 5.]).expand(self.N, -1) + self.scene.env_origins
                else:
                    surface = cast(surface).expand(self.N, -1) + self.scene.env_origins
                    normal = cast(normal).expand(self.N, -1)
                    object_R = cast(object_rotation).expand(self.N, -1, -1)
                    q = mat_to_quat(object_R)
                    world_offsets = (object_R @ vertices.T).transpose(-1, -2)
                    position = surface - cast(object_support) - .002 * normal
                    # A hovering free robot and object share gravity. A gentle
                    # relative approach supplies a defined real contact impulse;
                    # no additional force, constraint or physics tuning is used.
                    probe_velocity[:, :3] = -.2 * normal
                    minimum_z = (position[:, None, 2] + world_offsets[..., 2]).amin(-1)
                    definition.update(surface_point_world=surface.cpu().tolist(), outward_normal_world=normal.cpu().tolist(),
                        object_support_facet=object_facet,
                        object_support_selection="center_of_one_actual_exposed_convex_component_facet",
                        prescribed_relative_approach_speed_m_s=.2,
                        ground_placement_audit={"passed": bool((minimum_z >= .002-2e-6).all()),
                            "maximum_allowed_penetration_m": .002, "minimum_required_clearance_m": .002,
                            "minimum_final_world_z_m": float(minimum_z.min()),
                            "maximum_ground_penetration_m": float((-minimum_z).clamp_min(0).max()),
                            "per_environment_ground_penetration_m": (-minimum_z).clamp_min(0).cpu().tolist(),
                            "placement_method": "all_robot_colliders_above_ground_exposed_target_up"})
                    if not definition["ground_placement_audit"]["passed"]:
                        raise RuntimeError("Exposed-region object probe failed ground clearance")
            else:
                # The object-ground control is spatially separated from the
                # robot, so a source reset cannot contaminate its identity.
                world_offsets = (O[1] @ vertices.T).transpose(-1, -2)
                position = self.scene.env_origins.clone()
                position[:, 2] = -world_offsets[..., 2].amin(-1) - .002
                root_q = mat_to_quat(S[1])
                root_world = S[0] + self.scene.env_origins
                root_world[:, 2] += (world_offsets[..., 2].amax(-1) - world_offsets[..., 2].amin(-1)) + 3.
                self.robot.write_root_link_pose_to_sim_index(root_pose=torch.cat(
                    (root_world, root_q[:, 1:4], root_q[:, :1]), -1).contiguous(), env_ids=ids)
                self.robot.write_root_link_velocity_to_sim_index(
                    root_velocity=torch.zeros(self.N, 6, device=self.device), env_ids=ids)
                self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros(self.N, 29, device=self.device), env_ids=ids)
                actions = (S[4] - self.env.default_q - self.env.xi[:, 1:30]) / self.env.alpha
                definition = {"placement": "object_ground_support_surface_robot_raised_clear",
                              "prescribed_penetration_m": .002}
            pose = torch.cat((position, q[:, 1:4], q[:, :1]), -1).contiguous()
            self.object.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=ids)
            self.object.write_root_com_velocity_to_sim_index(root_velocity=probe_velocity, env_ids=ids)
            self.env.refresh()
            initial_robot, initial_object = self.state()
            definition["initial_state_readback"] = {
                "robot_root_position_world": (initial_robot[0] + self.scene.env_origins).cpu().tolist(),
                "robot_root_rotation": initial_robot[1].cpu().tolist(),
                "robot_joint_positions": initial_robot[4].cpu().tolist(),
                "robot_joint_velocities": initial_robot[5].cpu().tolist(),
                "object_position_world": (initial_object[0] + self.scene.env_origins).cpu().tolist(),
                "object_rotation": initial_object[1].cpu().tolist(),
                "object_link_velocity_world": initial_object[2].cpu().tolist(),
                "object_angular_velocity_world": initial_object[3].cpu().tolist()}
            desired_joint = joint if role in BODY_ROLES else S[4]
            errors = {
                "robot_root_position_m": float((initial_robot[0] + self.scene.env_origins - root_world).abs().max()),
                "robot_root_rotation": float((initial_robot[1] - quat_to_mat(root_q)).abs().max()),
                "robot_joint_position_rad": float((initial_robot[4] - desired_joint).abs().max()),
                "robot_joint_velocity_rad_s": float(initial_robot[5].abs().max()),
                "held_pd_target_rad": float((self.env.default_q + self.env.xi[:, 1:30] + self.env.alpha * actions - desired_joint).abs().max()),
                "object_position_m": float((initial_object[0] + self.scene.env_origins - position).abs().max()),
                "object_rotation": float((initial_object[1] - quat_to_mat(q)).abs().max()),
                "object_linear_velocity_m_s": float((initial_object[2] - probe_velocity[:, :3]).abs().max()),
                "object_angular_velocity_rad_s": float(initial_object[3].abs().max())}
            definition["initial_state_readback"]["maximum_errors"] = errors
            if max(errors.values()) > 2e-5:
                raise RuntimeError(f"native probe initial state differs from prescribed pose/velocity: {errors}")
            max_force = torch.zeros(len(CHANNELS), device=self.device)
            hand_ground, foot_ground = torch.zeros(2, device=self.device), torch.zeros(2, device=self.device)
            body_ground = torch.zeros(len(BODY_ROLES), device=self.device)
            impulse_trace, force_trace, hand_ground_trace, foot_ground_trace, body_ground_trace = [], [], [], [], []
            try:
                for _ in range(ticks):
                    pair = self._advance(actions)
                    max_force = torch.maximum(max_force, pair["contact_force"].amax(0))
                    body_ground = torch.maximum(body_ground, (pair["body_ground_impulse"] / CONTROL_DT).amax(0))
                    body_ground_trace.append(pair["body_ground_impulse"].cpu().tolist())
                    hand_ground = torch.maximum(hand_ground, (pair["hand_ground_impulse"] / CONTROL_DT).amax(0))
                    foot_ground = torch.maximum(foot_ground, (pair["foot_ground_impulse"] / CONTROL_DT).amax(0))
                    impulse_trace.append(pair["impulse"].cpu().tolist())
                    force_trace.append(pair["contact_force"].cpu().tolist())
                    hand_ground_trace.append(pair["hand_ground_impulse"].cpu().tolist())
                    foot_ground_trace.append(pair["foot_ground_impulse"].cpu().tolist())
            except BaseException as exc:
                failure = {"probe": name, "definition": definition, "object_initial_pose_xyzw": pose.cpu().tolist(),
                           "object_initial_com_velocity_world": probe_velocity.cpu().tolist(),
                           "completed_interval_impulses_Ns": impulse_trace, "completed_interval_forces_N": force_trace,
                           "error_type": type(exc).__name__, "error": str(exc), "previous_complete_probes": report,
                           "scientific_status": "failed native calibration; no successful pair labels inferred"}
                with (self.output / "pair-probes-failure.json").open("x", encoding="utf-8") as stream:
                    json.dump(failure, stream, indent=2)
                raise
            report[name] = {"object_initial_pose_xyzw": pose.cpu().tolist(), "ticks": ticks,
                            "definition": definition, "held_pd_offset_actions": actions.cpu().tolist(),
                            "object_initial_com_velocity_world": probe_velocity.cpu().tolist(),
                            "maximum_interval_normal_force_N": max_force.cpu().tolist(),
                            "maximum_body_ground_interval_force_N": body_ground.cpu().tolist(),
                            "interval_body_ground_impulse_Ns": body_ground_trace,
                            "maximum_hand_ground_interval_force_N": hand_ground.cpu().tolist(),
                            "maximum_foot_ground_interval_force_N": foot_ground.cpu().tolist(),
                            "interval_normal_impulse_Ns": impulse_trace, "interval_mean_normal_force_N": force_trace,
                            "interval_hand_ground_impulse_Ns": hand_ground_trace,
                            "interval_foot_ground_impulse_Ns": foot_ground_trace,
                            "control_dt_s": CONTROL_DT, "substeps_per_interval": 4}
            (self.output / "pair-probes-progress.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        checks = {name: bool((torch.tensor(report[name]["interval_mean_normal_force_N"])[..., channel].amax(0) > self.contacts.threshold).all())
                  for channel, name in enumerate(CHANNELS)}
        negative = report["hand_ground_negative"]
        negative_ok = max(negative["maximum_hand_ground_interval_force_N"]) > self.contacts.threshold and max(negative["maximum_interval_normal_force_N"][:2]) == 0
        checks["hand_ground_not_mislabelled"] = negative_ok
        for side, index in (("left", 0), ("right", 1)):
            negative = report[side + "_foot_ground_negative"]
            checks[side + "_foot_ground_not_mislabelled"] = (
                negative["maximum_foot_ground_interval_force_N"][index] > self.contacts.threshold and
                max(negative["maximum_interval_normal_force_N"][:-1]) == 0)
        for index, (channel, role) in enumerate(zip(CHANNELS[:-1], BODY_ROLES)):
            name = "hand_ground_negative" if role == LEFT else channel.removesuffix("_object") + "_ground_negative"
            negative = report[name]
            checks[channel + "_ground_not_mislabelled"] = (
                negative["maximum_body_ground_interval_force_N"][index] > self.contacts.threshold and
                max(negative["maximum_interval_normal_force_N"][:-1]) == 0)
        from .native_acceptance import probe_trace_blockers
        trace_blockers = probe_trace_blockers(report, self.N, self.contacts.threshold)
        report["passed"], report["checks"] = not trace_blockers, checks
        report["every_environment_trace_blockers"] = trace_blockers
        self.active_pair_probe = None
        self.reset(ids, frames)
        return report

    def step(self, action):
        e = self.env
        scene = self.support_state()
        if action.shape != (self.N, 29) or not bool(torch.isfinite(action).all()):
            raise ValueError("Expected finite [N,29] G1 PD-offset actions")
        S, O = self.state()
        f, a_prev, ep_step = e.f.clone(), e.a_prev.clone(), e.ep_step.clone()
        pair = self._advance(action)
        S1, O1 = self.state()
        # Frames only clamp for full-duration batched evaluation; training resets
        # at clip ends. A short clip cannot read into its neighbour's reference.
        clip_end = self.motion.clip_end[self.motion.clip_id[f]]
        f1 = torch.minimum(f + 1, clip_end)
        reward, terms = self.mdp.reward(S1, O1, f1, action, a_prev, pair["contact"], pair["contact_valid"], terms=True)
        term = self.mdp.termination(S1, O1, f1)
        timeout = (f + 1 >= clip_end) | (ep_step + 1 >= e.max_episode_steps)
        done = term | timeout
        # Diagnostic-only reason bits; no additional termination is introduced.
        human = self.mdp.human
        height_fail = (self.motion.root_p[f1, 2] - S1[0][:, 2]).abs() > human.cfg.term_height
        tilt_fail = (self.motion.root_R[f1, 2, 2] - S1[1][:, 2, 2]).abs() > human.cfg.term_tilt
        object_fail = torch.linalg.vector_norm(O1[0] - self.motion.object_p[f1], dim=-1) > self.mdp.object_failure_threshold(f1)
        failure_reason = (height_fail.int() | (tilt_fail.int() << 1) | (object_fail.int() << 2) |
                          ((f + 1 >= clip_end).int() << 3) | ((ep_step + 1 >= e.max_episode_steps).int() << 4))
        expected, valid = self.motion.expected(f1)
        missing = (expected > .5) & (valid > .5) & pair["contact_valid"] & (pair["contact"] < .5)
        self.contact_missing_counter = torch.where(missing, self.contact_missing_counter + 1, 0)
        tr = {"s": flatten_state(S), "s1": flatten_state(S1), "o": flatten_object(O), "o1": flatten_object(O1),
              "a": action.clone(), "a_prev": a_prev, "xi": e.xi.clone(), "xi_o": self.xi_o.clone(), "f": f,
              "ep_step": ep_step, "done": done, "term": term, "asset_id": self.asset_id.clone(),
              "failure_reason": failure_reason,
              "r": reward, "r_human": terms["human_reward"], "r_hoi": reward - terms["human_reward"],
              "contact_missing_counter": self.contact_missing_counter.clone(), "terms": terms, **pair,
              "support_pose_xyzw":scene["support_pose_xyzw"],"support_present":scene["support_present"]}
        e.ep_disc_ret += (e.gamma ** ep_step.float()) * reward
        e.ep_ret += reward
        e.f, e.ep_step, e.a_prev = f1, ep_step + 1, action.clone()
        if e.eval_mode:
            e.failed_eval |= term & ~e.done_eval
            e.done_eval |= done
            return tr
        ids = done.nonzero(as_tuple=False).flatten()
        if len(ids):
            length = e.ep_step[ids].float()
            boot = torch.where(term[ids], 0., (e.gamma ** length) * (e.ep_ret[ids] / length) / (1 - e.gamma))
            G = e.ep_disc_ret[ids] + boot
            bins = e.start_bin[ids]
            e.bin_G.index_put_((bins,), .8 * e.bin_G[bins] + .2 * G)
            tr["ep_len"] = length
            tr["ep_ret_norm"] = e.ep_ret[ids] / (length * self.mdp.max_step_reward)
            # s1/o1/contacts above were cloned before this joint reset.
            self.reset(ids)
        return tr

    def close(self):
        self.contacts.close()

    def export_compiled_geometry(self, output):
        """Export exact compiled convex meshes in object/hand link frames.

        This creates a new explicit-hull USD; subsequent preparation does not
        need to infer PhysX cooking or recook an unbound source mesh.
        """
        import hashlib
        import mujoco
        import numpy as np
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        from scipy.spatial.transform import Rotation
        from supermimic.physics import sha
        from hoi.hand_collision import HAND_SHA, ORIGIN, RECIPE
        from .geometry import collision_distance
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        model = self.contacts.solver.mj_model
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]
        object_body = next(i for i, n in enumerate(names) if "Object" in n and int(model.body_jntnum[i]) == 1)

        links = compiled_geom_links(model)

        def extract(role, body, offset=None, canonical_link=None):
            meshes = []
            for g, r in enumerate(self.contacts.roles):
                if r != role or (canonical_link is not None and links[g] != canonical_link):
                    continue
                # Export first environment only when the CPU model encodes all.
                ge = self.contacts.geom_envs[g]
                if self.contacts.topology == "one_world_with_named_environments" and ge != 0:
                    continue
                if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                    raise RuntimeError("Compiled collision export currently requires mesh colliders")
                mesh_id = int(model.geom_dataid[g])
                vertices, faces, evidence = compiled_convex_mesh(model, mesh_id)
                world = vertices @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
                local = (world - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
                if offset is not None:
                    local -= np.asarray(offset)
                meshes.append({"vertices": torch.tensor(local), "faces": torch.tensor(faces, dtype=torch.long),
                               "source_native_geom": self.contacts.names[g], "native_convex_evidence": evidence})
            if not meshes:
                raise RuntimeError("No actual compiled collision meshes were exported")
            return meshes

        object_meshes = extract(OBJECT, object_body)
        hands = {}
        repo = Path(__file__).resolve().parents[2]
        for side, role in (("left", LEFT), ("right", RIGHT)):
            body = next(i for i, n in enumerate(names) if side + "_wrist_yaw_link" in n)
            meshes = extract(role, body, ORIGIN[side], canonical_link=side + "_rubber_hand")
            stl = repo / "data/g1_description/meshes" / f"{side}_rubber_hand.STL"
            if sha(stl) != HAND_SHA[side]:
                raise RuntimeError("Native hand source STL identity changed")
            from .prepare import read_triangle_mesh
            source_mesh, _ = read_triangle_mesh({"path": str(stl), "sha256": HAND_SHA[side]}, stl.parent)
            points, patch_definition = palm_patch_points(meshes, [source_mesh], side)
            if float(collision_distance(points, meshes).abs().max()) > 1e-5:
                raise RuntimeError("Ray-cast palm patch does not lie on the actual collider union surface")
            _write_palm_qa(output / f"{side}_palm_patch.png", source_mesh["vertices"], points)
            from .geometry import tensor_sha256
            from . import probes
            evidence = {"schema": "hoi_v1_palm_patch_evidence_v1", "source_stl_sha256": HAND_SHA[side],
                        "probes_source": {"path": str(Path(probes.__file__)), "sha256": sha(probes.__file__)},
                        "definition": patch_definition, "points_sha256": tensor_sha256(points),
                        "collision_meshes_sha256": hashlib.sha256(json.dumps(
                            [{"vertices": m["vertices"].tolist(), "faces": m["faces"].tolist()} for m in meshes],
                            sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                        "visual_qa": {"path": str(output / f"{side}_palm_patch.png"), "sha256": sha(output / f"{side}_palm_patch.png")}}
            evidence["evidence_sha256"] = hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            hands[side] = {"mount": ORIGIN[side], "source_stl": {"path": str(stl), "sha256": HAND_SHA[side]},
                           "collision_meshes": [{"vertices": m["vertices"].tolist(), "faces": m["faces"].tolist(), "frame": "hand_local"} for m in meshes],
                           "palm_points": points.tolist(), "recipe": RECIPE,
                           "native_convex_evidence": [m["native_convex_evidence"] for m in meshes],
                           "probe_definition": patch_definition, "palm_selection_verified": True,
                           "palm_selection_evidence": evidence,
                           "visual_qa": {"path": str(output / f"{side}_palm_patch.png"), "projections": ["XY", "XZ", "YZ"],
                                         "source": "blue", "palm_patch": "red"}}
        self.physics_report["native_convex_hull_export"] = {
            "source": "mjModel.mesh_graph", "all_geometries_graph_backed": True,
            "object_hulls": len(object_meshes), "hand_hulls": {side: len(hands[side]["collision_meshes"]) for side in hands},
            "object_evidence_sha256": hashlib.sha256(json.dumps([m["native_convex_evidence"] for m in object_meshes], sort_keys=True).encode()).hexdigest(),
            "hand_evidence_sha256": {side: hashlib.sha256(json.dumps(hands[side]["native_convex_evidence"], sort_keys=True).encode()).hexdigest() for side in hands}}
        target = output / "object_explicit_hulls.usda"
        stage = Usd.Stage.CreateNew(str(target))
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        root = UsdGeom.Xform.Define(stage, "/Object").GetPrim()
        stage.SetDefaultPrim(root)
        UsdPhysics.RigidBodyAPI.Apply(root)
        mass = UsdPhysics.MassAPI.Apply(root)
        p = self.asset["physical_profile"]
        mass.CreateMassAttr(float(p["mass"]))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, p["com"])))
        values, axes = np.linalg.eigh(np.asarray(p["inertia"]))
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        q = Rotation.from_matrix(axes).as_quat()
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, values)))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))
        if self.asset.get("repair_binding"):
            from hoi.data.native_repairs import read_binding, apply_hull_vertex_contract
            read_binding(self.asset["repair_binding"]["path"])
        for i, mesh in enumerate(object_meshes):
            prim = UsdGeom.Mesh.Define(stage, f"/Object/hull_{i:03d}")
            prim.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in mesh["vertices"]])
            prim.CreateFaceVertexCountsAttr([3] * len(mesh["faces"]))
            prim.CreateFaceVertexIndicesAttr(mesh["faces"].flatten().tolist())
            prim.CreateSubdivisionSchemeAttr("none")
            UsdPhysics.CollisionAPI.Apply(prim.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim()).CreateApproximationAttr("convexHull")
            if self.asset.get("repair_binding"):
                apply_hull_vertex_contract(prim.GetPrim(),len(mesh["vertices"]))
        stage.GetRootLayer().Save()
        record = {"object_asset": self.asset["object_asset"], "exact_scale": self.asset["exact_scale"],
                  "mesh_to_object": np.eye(4).tolist(), "native_asset": str(target), "native_asset_sha256": sha(target),
                  "physical_profile": p, "cooking_recipe": "compiled_Newton_CoACD_pointsets_explicit_convexHull_import_geometric_equivalence_requires_native_check",
                  "collision_meshes": [{"vertices": m["vertices"].tolist(), "faces": m["faces"].tolist(), "frame": "object_local"} for m in object_meshes],
                  "compiled_geometry_frame": "object_root_link_origin", "compiled_mesh_names": [m["source_native_geom"] for m in object_meshes],
                  "native_convex_evidence": [m["native_convex_evidence"] for m in object_meshes],
                  "source_native_asset": self.asset["native_asset"], "source_native_asset_sha256": self.asset["native_asset_sha256"]}
        if getattr(self, "cooking_source_evidence", None):
            from hoi.data.native_assets import digest as surface_digest
            evidence = self.cooking_source_evidence
            if evidence.get("repair_binding"):
                from hoi.data.native_repairs import compare_candidate_export
                record["repair_import_comparison"]=compare_candidate_export(evidence["repair_binding"]["path"],record["collision_meshes"])
                record.update(repair_binding=evidence["repair_binding"],
                    cooking_input_geometry_sha256=evidence["cooking_input_geometry_sha256"],
                    cooking_input_surface_sha256=evidence["cooking_input_surface_sha256"],
                    original_source_build=evidence["original_source_build"],
                    hull_vertex_policy="per_component_input_vertex_count_no_decimation")
            record.update(object_geometry_sha256=evidence["source_object_geometry_sha256"],
                source_exact_surface_sha256=evidence["source_exact_surface_sha256"],
                native_compiled_surface_sha256=surface_digest(record["collision_meshes"]),
                source_build_receipt=evidence["build_receipt"],physical_evidence=evidence["physical_evidence"],
                geometry_stage="native_compiled_export_requires_second_import_geometry_gate",
                training_ready=False)
        recipe = {"schema": "hoi_v1_compiled_geometry_recipe_v1", "assets": [record], "hands": hands,
                  "native_physics_readback": self.physics_report,
                  "hand_runtime_report": self.env.physics_settings["hand_asset"],
                  "native_robot_asset": getattr(self, "native_robot_asset", None) or {
                      "path": self.env.physics_settings["robot_usd"], "sha256": self.env.physics_settings["robot_usd_sha256"]},
                  "limitations": "Actual Newton CoACD meshes; does not claim original PhysX cooker equivalence"}
        if getattr(self, "reference_kinematics", None) is not None:
            recipe["reference_kinematics"] = self.reference_kinematics
        if getattr(self, "cooking_source_evidence", None):
            recipe["robot_geometry_sha256"] = self.cooking_source_evidence["robot_geometry_sha256"]
            recipe["cooking_fixture_source"] = self.cooking_source_evidence
        (output / "assets.json").write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
        return recipe, object_meshes

    def verify_compiled_geometry(self, meshes, *, compiled_hands=None, tolerance=1e-5):
        """Bidirectional union-surface and bbox check against bound recipe."""
        from .geometry import collision_distance, deterministic_surface_points
        expected = self.mdp.shapes.geometries[int(self.asset["asset_id"])]["collision_meshes"]
        errors = []
        for source, target in ((meshes, expected), (expected, meshes)):
            sample = deterministic_surface_points(source, 2048, 0).double()
            # Exclude internal faces of overlapping convex pieces: they are
            # not part of the collider union's external boundary.
            boundary = collision_distance(sample, source).abs() < 2e-6
            if not bool(boundary.any()):
                raise RuntimeError("Unable to compare collision-union external surfaces")
            errors.append(float(collision_distance(sample[boundary], target).abs().max()))
        actual_vertices = torch.cat([m["vertices"].double() for m in meshes])
        expected_vertices = torch.cat([m["vertices"].double() for m in expected])
        box_error = max(float((actual_vertices.amin(0) - expected_vertices.amin(0)).abs().max()),
                        float((actual_vertices.amax(0) - expected_vertices.amax(0)).abs().max()))
        passed = max(*errors, box_error) <= tolerance
        result = {"passed": passed, "max_bidirectional_surface_errors_m": errors, "bbox_max_abs_error_m": box_error,
                  "tolerance_m": tolerance, "surface_samples_per_direction": 2048,
                  "test": "compiled convex union vs SHA-bound geometry cache in object link frame"}
        if not passed:
            raise RuntimeError(f"Native collision recipe changed: {result}")
        self.physics_report["compiled_collision_geometry_verified_against_bundle"] = True
        self.physics_report["compiled_geometry_comparison"] = result
        if compiled_hands is None:
            raise RuntimeError("Native geometry gate must compare actual hand colliders and palm probes too")
        hand_comparisons = {}
        for side in ("left", "right"):
            hand = self.mdp.shapes.hands[side]
            actual_hand = compiled_hands[side]
            observed = [{"vertices": torch.tensor(m["vertices"], dtype=torch.float64), "faces": torch.tensor(m["faces"], dtype=torch.long)}
                        for m in actual_hand["collision_meshes"]]
            errors = []
            for a, b in ((observed, hand["collision_meshes"]), (hand["collision_meshes"], observed)):
                sample = deterministic_surface_points(a, 1024, 0).double()
                boundary = collision_distance(sample, a).abs() < 2e-6
                if not bool(boundary.any()):
                    raise RuntimeError("Cannot establish actual hand collider exterior")
                errors.append(float(collision_distance(sample[boundary], b).abs().max()))
            actual_points = torch.tensor(actual_hand["palm_points"], dtype=torch.float32)
            expected_points = torch.as_tensor(hand["palm_points"], dtype=torch.float32)
            point_error = float((actual_points - expected_points).abs().max()) if actual_points.shape == expected_points.shape else float("inf")
            if max(*errors, point_error) > tolerance:
                raise RuntimeError(f"{side} hand collider/palm recipe changed: surfaces {errors}, point error {point_error}")
            hand_comparisons[side] = {"passed": True, "bidirectional_surface_error_m": errors, "palm_points_error_m": point_error,
                                      "palm_evidence_sha256": hand.get("palm_selection_evidence", {}).get("evidence_sha256")}
        self.physics_report["compiled_hand_geometry_comparison"] = hand_comparisons
        self.physics_report["independent_native_geometry_queries"] = self._native_distance_audit()
        return result

    def _native_distance_audit(self, tolerance=5e-5):
        """Compare cached surfaces with independent native convex GJK distances.

        CPU mjData is private: no GPU state, physics step or dataset pose is
        modified. AABB lower bounds only exclude pairs which cannot improve the
        measured native distance; every remaining cache pair uses exact mesh
        distance, rather than mesh-ray queries against a visual surface.
        """
        import mujoco
        import numpy as np
        from .geometry import exact_mesh_gap
        model = self.contacts.solver.mj_model
        data = mujoco.MjData(model)
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]
        body = next(i for i, n in enumerate(names) if "Object" in n and int(model.body_jntnum[i]) == 1)
        address = int(model.jnt_qposadr[int(model.body_jntadr[body])])
        object_geoms = [g for g, role in enumerate(self.contacts.roles) if role == OBJECT and int(model.geom_bodyid[g]) == body]
        objects = self.mdp.shapes.geometries[int(self.asset["asset_id"])]["collision_meshes"]
        radius = max(float(torch.as_tensor(m["vertices"]).norm(dim=-1).max()) for m in objects)
        from hoi.hand_collision import ORIGIN
        from supermimic.rotations import so3_exp
        records = []
        links = compiled_geom_links(model)
        for side, role in (("left", LEFT), ("right", RIGHT)):
            wrist = next(i for i, name in enumerate(names) if side + "_wrist_yaw_link" in name)
            hand_geoms = [g for g, r in enumerate(self.contacts.roles) if r == role and links[g] == side + "_rubber_hand" and
                          (self.contacts.topology == "one_mujoco_world_per_environment" or self.contacts.geom_envs[g] == 0)]
            hands = self.mdp.shapes.hands[side]["collision_meshes"]
            mount = torch.tensor(ORIGIN[side], dtype=torch.float64)
            hand_radius = max(float((torch.as_tensor(m["vertices"]).double() + mount).norm(dim=-1).max()) for m in hands)
            for index, direction in enumerate((np.array([1., .3, .2]), np.array([-.2, .5, 1.]))):
                direction /= np.linalg.norm(direction)
                mujoco.mj_forward(model, data)
                data.qpos[address:address + 3] = data.xpos[wrist] + direction * (radius + hand_radius + .15)
                q = mat_to_quat(so3_exp(torch.tensor([.17, -.23, .11]) * index)).numpy()
                data.qpos[address + 3:address + 7] = q
                mujoco.mj_forward(model, data)
                native = min(float(mujoco.mj_geomDistance(model, data, h, o, 100., np.empty(6)))
                             for h in hand_geoms for o in object_geoms)
                if native <= 0 or native >= 100:
                    raise RuntimeError("Independent geometry query did not produce a separated convex distance")
                def world(mesh, rigid_body, offset=None):
                    v = torch.as_tensor(mesh["vertices"]).double()
                    if offset is not None:
                        v = v + offset
                    rotation = torch.tensor(data.xmat[rigid_body].reshape(3, 3))
                    return {"vertices": v @ rotation.T + torch.tensor(data.xpos[rigid_body]), "faces": mesh["faces"]}
                hand_world = [world(m, wrist, mount) for m in hands]
                object_world = [world(m, body) for m in objects]
                best, visited = float("inf"), 0
                for hand in hand_world:
                    for obj in object_world:
                        h, o = hand["vertices"], obj["vertices"]
                        lower = torch.maximum(torch.maximum(h.amin(0) - o.amax(0), o.amin(0) - h.amax(0)), h.new_zeros(3)).norm()
                        if float(lower) > min(best, native + tolerance):
                            continue
                        gap, penetration = exact_mesh_gap([hand], [obj])
                        if float(penetration) > tolerance:
                            raise RuntimeError("Cached collider overlaps a separated native geometry probe")
                        best, visited = min(best, float(gap)), visited + 1
                error = abs(best - native)
                records.append({"hand": side, "pose": index, "native_distance_m": native, "cached_surface_distance_m": best,
                                "abs_error_m": error, "cache_pairs_queried": visited, "cache_pairs_total": len(hands) * len(objects)})
                if not np.isfinite(error) or error > tolerance:
                    raise RuntimeError(f"Independent native convex distance differs from cached geometry: {records[-1]}")
        return {"passed": True, "api": "mujoco.mj_geomDistance convex collision distance", "tolerance_m": tolerance,
                "overlap": "excluded; actual pair force probes independently verify overlap identity", "samples": records}


def _load_legacy_fixture(path, asset_id, kin, device):
    """Read one immutable canonical case solely for the native physics probe."""
    import hashlib
    from types import SimpleNamespace
    from supermimic.mdp import MDP
    from supermimic.robot import G1_JOINT_NAMES, default_joint_pos_list
    from supermimic.physics import sha
    from hoi.data.reference import adapt_motion
    from .motion import TrainingMotionBank
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    key = list(manifest["assets"])[asset_id]
    source_asset = manifest["assets"][key]
    source_case = next(c for c in manifest["cases"] if c["object_asset"] == key)
    if sha(source_case["motion"]) != source_case["motion_sha256"]:
        raise ValueError("legacy fixture source motion changed")
    frames = torch.load(source_case["motion"], weights_only=True, map_location="cpu")
    reference = adapt_motion(frames, manifest["robot_joint_names"], G1_JOINT_NAMES)
    n = len(reference["time"])
    unknown = torch.zeros(n, len(CHANNELS))
    # Expected-contact masks are unknown here. This path never computes HOI
    # reward, updates a normalizer, creates replay or enters a learner.
    bank = TrainingMotionBank([{"name": source_case["id"], "asset_id": 0, "reference": reference,
                                "expected_contact": unknown, "expected_valid": unknown}], kin, device=device, partition="validation")
    profile = source_asset["metadata"]["object_properties"][0]
    scale = [int(x) / 100 for x in key.split("_")[-3:]]
    asset = {"asset_id": 0, "object_asset": key, "exact_scale": scale,
             "native_asset": source_asset["usd"], "native_asset_sha256": source_asset["usd_sha256"],
             "collision_hash": hashlib.sha256((source_asset["usd_sha256"] + ":Newton_CoACD_native_probe").encode()).hexdigest(),
             "physical_profile": {**profile, "friction": .6, "ground_friction": .3}}
    mdp = SimpleNamespace(human=MDP(kin, bank, torch.tensor(default_joint_pos_list(), device=device)), max_step_reward=.15)
    source = {"fixture_manifest": str(path / "manifest.json"), "fixture_manifest_sha256": sha(path / "manifest.json"),
              "case": source_case, "scope": "single_asset_physics_probe; no HOI reward or training",
              "reference_kinematics": {"urdf": {"path": str(path / "metadata/ultra_g1.urdf"), "sha256": sha(path / "metadata/ultra_g1.urdf")},
                                       "body_names": source_asset["metadata"]["body_names"], "dof_names": source_asset["metadata"]["dof_names"]}}
    return bank, mdp, asset, source


def main(argv=None):
    from isaaclab.app import add_launcher_args, launch_simulation
    from supermimic.physics import add_physics_arguments, resolve_arguments
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--bundle", type=Path)
    inputs.add_argument("--legacy-prepared-fixture", type=Path)
    inputs.add_argument("--curated-cooking-fixture", type=Path, help="First-stage source-bound native collision cook; no reward or training")
    parser.add_argument("--robot-asset-evidence", type=Path, help="Fixture-only recipe binding an already verified exact hand overlay")
    parser.add_argument("--asset-id", type=int, required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--capture-smoke-visual", action="store_true", help="Capture/render one existing smoke step before autoreset; no additional physics; unavailable for cooking")
    parser.add_argument("--audit-scene-resets", action="store_true", help="Terminal all-clip sampled support reset/readback audit; no additional physics; state discarded")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    add_physics_arguments(parser)
    add_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.capture_smoke_visual and (args.curated_cooking_fixture is not None or args.steps < 1):
        raise ValueError("--capture-smoke-visual requires standard native smoke with existing physics steps; cooking stays geometry-only")
    if args.audit_scene_resets and (args.curated_cooking_fixture is not None or args.legacy_prepared_fixture is not None or args.steps < 1):
        raise ValueError("--audit-scene-resets requires a bound bundle and completed standard smoke")
    settings, _ = resolve_arguments(args)
    if not (settings.hand_collision and settings.self_collision):
        parser.error("v1 native gate requires the specified hand ON/self ON profile")
    args.output.mkdir(parents=True, exist_ok=False)
    import hashlib
    repo = Path(__file__).resolve().parents[2]
    from .config import source_identity
    source_code = source_identity()["files"]
    (args.output / "source.json").write_text(json.dumps(source_code, indent=2) + "\n", encoding="utf-8")
    import os
    os.environ["SUPERMIMIC_ASSET_DIR"] = str(args.output / "robot_assets")
    from supermimic.kinematics import G1Kinematics
    from supermimic.robot import default_joint_pos_list
    import warp as wp
    wp.init()
    torch.manual_seed(args.seed)
    device = "cuda:0"
    kin = G1Kinematics(device)
    is_fixture = bool(args.legacy_prepared_fixture or args.curated_cooking_fixture)
    if args.legacy_prepared_fixture:
        bank, mdp, asset, source = _load_legacy_fixture(args.legacy_prepared_fixture, args.asset_id, kin, device)
        robot_asset = None
        if args.robot_asset_evidence:
            recipe = json.loads(args.robot_asset_evidence.read_text(encoding="utf-8"))
            robot_asset = recipe.get("native_robot_asset")
            if robot_asset is None and recipe.get("hand_runtime_report"):
                saved = recipe["hand_runtime_report"]
                robot_asset = {"path": saved["usd"], "sha256": saved["usd_sha256"]}
            if robot_asset is None:
                raise ValueError("Robot evidence recipe does not bind an exact verified hand overlay")
            source["robot_asset_evidence"] = {"path": str(args.robot_asset_evidence), "sha256": hashlib.sha256(args.robot_asset_evidence.read_bytes()).hexdigest()}
    elif args.curated_cooking_fixture:
        if args.robot_asset_evidence or args.asset_id != 0:
            parser.error("curated cooking fixture binds its robot internally and requires --asset-id 0")
        from .native_fixture import load_curated_cooking_fixture
        bank, mdp, asset, source, robot_asset = load_curated_cooking_fixture(args.curated_cooking_fixture,kin,device)
    else:
        if args.robot_asset_evidence:
            parser.error("--robot-asset-evidence applies only to the independent legacy physics fixture")
        from hoi.v1.prepare import load_bundle
        from hoi.v1.mdp import HOIMDP
        bank, shapes, bundle = load_bundle(args.bundle, kin, device=device, require_train=False, asset_filter=[args.asset_id])
        asset = next(a for a in bundle["assets"] if int(a["asset_id"]) == args.asset_id)
        mdp = HOIMDP(kin, bank, torch.tensor(default_joint_pos_list(), device=device), shapes)
        robot_asset = bundle.get("native_robot_asset")
        if robot_asset is None:
            raise ValueError("HOI bundle must bind an exact verified native robot hand overlay")
        source = {"bundle": str(args.bundle), "scope": "single_asset_native_bundle_gate",
                  "bundle_content_sha256": bundle["bundle_content_sha256"]}
    cfg = make_hoi_sim(device, args)
    cfg.physics.solver_cfg.save_to_mjcf = str(args.output / "compiled_model.generated.xml")
    with launch_simulation(cfg, args), torch.no_grad():
        env = HOITrackingSim(cfg, args.num_envs, bank, mdp, asset, args.output, eval_mode=False, reset_noise=False, robot_asset=robot_asset, cooking_only=bool(args.curated_cooking_fixture))
        env.reference_kinematics = source.get("reference_kinematics")
        env.cooking_source_evidence = source if args.curated_cooking_fixture else None
        from hoi.hand_collision import compiled_audit
        from supermimic.physics import compiled_self_collision_audit
        hand_audit = compiled_audit(env.contacts.solver.mj_model, True)
        self_audit = compiled_self_collision_audit(env.contacts.solver.mj_model, True)
        from .native_model_debug import export_actual_model
        export_actual_model(env.contacts.solver.mj_model, args.output,
            args.output / "compiled_model.generated.xml", recorder=env.contacts,
            scene_snapshot={**env.support_state(), "environment_origins": env.scene.env_origins,
                "reference_frames": env.f, "declared_support_ids": list(env.supports)})
        export, meshes = env.export_compiled_geometry(args.output / "compiled_geometry")
        if args.curated_cooking_fixture:
            from .native_fixture import finish_geometry_only_cooking
            finish_geometry_only_cooking(env,source,args.output,hand_audit=hand_audit,self_audit=self_audit,seed=args.seed)
            return
        if is_fixture:
            from types import SimpleNamespace
            mdp.shapes = SimpleNamespace(geometries=[{"collision_meshes": meshes}], hands=export["hands"])
            env.physics_report["independent_native_geometry_queries"] = env._native_distance_audit()
        else:
            env.verify_compiled_geometry(meshes, compiled_hands=export["hands"])
        probes = env.calibrate_pair_probes()
        support_probes = env.calibrate_support_probes()
        (args.output / "pair-probes.json").write_text(json.dumps(probes, indent=2) + "\n", encoding="utf-8")
        smoke_visual = {"enabled": False}
        if args.capture_smoke_visual:
            from .native_model_debug import enable_smoke_visual_capture
            smoke_visual = enable_smoke_visual_capture(env,args.output / "smoke_visual",args.output / "compiled_model.generated.xml")
        input_contract = {"passed": False, "reason": "independent fixture has no bound training/reference inputs"}
        if not is_fixture:
            from .privileged import configure_teacher
            configure_teacher(mdp, bundle, device)
        totals = torch.zeros(len(CHANNELS), device=device)
        maximum_force = torch.zeros(len(CHANNELS), device=device)
        for _ in range(args.steps):
            action = torch.zeros(args.num_envs, 29, device=device)
            tr = env._advance(action) if is_fixture else env.step(action)
            if not is_fixture and not input_contract.get("passed"):
                from .native_input_contract import validate_transition_inputs
                input_contract = validate_transition_inputs(tr, mdp, args.output / "first_transition_inputs.pt")
            totals += tr["contact"].sum(0)
            maximum_force = torch.maximum(maximum_force, tr["contact_force"].amax(0))
        palm_verified = False
        if not is_fixture:
            palm_verified = all(bool(mdp.shapes.hands[s].get("palm_selection_verified", False)) for s in ("left", "right"))
        from .native_acceptance import capture_native_runtime
        report = {"native_runtime": capture_native_runtime(env.contacts, compilation=env.warp_compilation),
                  "physics": env.physics_report, "contact": env.contacts.manifest(), "steps": args.steps, "source": source,
                  "source_code_sha256": source_code,
                  "asset_id": int(asset["asset_id"]), "object_asset": asset["object_asset"],
                  "seed": args.seed,
                  "bundle_content_sha256": source.get("bundle_content_sha256"), "palm_selection_verified": palm_verified,
                  "palm_selection_evidence": None if is_fixture else {s: mdp.shapes.hands[s].get("palm_selection_evidence") for s in ("left", "right")},
                  "compiled_hand_audit": hand_audit, "compiled_self_collision_audit": self_audit,
                  "compiled_foot_audit": env.compiled_foot_audit,
                  "compiled_geometry_export": str(args.output / "compiled_geometry/assets.json"),
                  "num_envs": args.num_envs, "contact_positives": totals.cpu().tolist(),
                  "maximum_interval_force_N": maximum_force.cpu().tolist(),
                  "zero_action_smoke_completed": True, "formal_training_started": False,
                  "native_pair_calibration_complete": probes["passed"], "pair_probes": probes,
                  "native_scene_supports": support_probes, "native_input_contract": input_contract, "smoke_visual_snapshot": smoke_visual}
        if args.capture_smoke_visual and not smoke_visual["captured"]:
            raise RuntimeError("Requested smoke visual was not captured by an existing control interval")
        # All ordinary acceptance results and visual/input captures are fixed.
        # This optional terminal audit only writes/reads reset state; no later
        # physics or network inference is permitted before environment close.
        if args.audit_scene_resets:
            from .native_scene_reset_audit import audit_scene_resets
            report["native_scene_reset_audit"] = audit_scene_resets(env, args.output / "scene_reset_audit.json")
        (args.output / "native.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        env.close()
        if args.capture_smoke_visual:
            # A fresh interpreter selects EGL before MuJoCo imports its GL
            # backend; rendering performs no FK, policy or integration calls.
            import subprocess
            import sys
            subprocess.run([sys.executable,"-m","hoi.v1.native_snapshot_render",
                "--manifest",str(args.output / "smoke_visual/compiled_model_manifest.json"),
                "--output",str(args.output / "smoke_visual/render")],check=True,
                env={**os.environ,"MUJOCO_GL":"glfw" if os.name == "nt" else "egl"})


if __name__ == "__main__":
    main()
