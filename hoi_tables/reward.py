"""Unchanged fourteen v1.1 reward formulae on current hw3 robot geometry."""
from __future__ import annotations

import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import torch

from hoi_residual.v11_reward import reward_terms, TERM_NAMES, MAX_STEP_REWARD
from supermimic.robot_metadata import URDF_PATH
from .geometry import (query_grid, _prepare_exact_meshes, _rigid_mesh_vertices,
                       _exact_mesh_gap, expected_schedule_from_distances, validate_mesh)


def bound_tensor(binding, parent=None):
    path = Path(binding["path"])
    if not path.is_absolute() and parent is not None:
        path = Path(parent) / path
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise ValueError(f"Geometry cache identity mismatch: {path}")
    import io
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)


def current_hands(binding, *, native_hands=None):
    """Bind current actual collision hulls; project calibrated probes onto them."""
    import numpy as np
    import trimesh
    from scipy.spatial.transform import Rotation
    from supermimic.g1_assets import SOURCE_URDF
    from supermimic.g1_physics import ASSET_DIR
    old = bound_tensor(binding)
    asset_dir = ASSET_DIR if native_hands is None else native_hands.asset_dir
    urdf = URDF_PATH if native_hands is None else asset_dir / "source" / SOURCE_URDF
    tree = ET.parse(urdf).getroot()
    hands, evidence = {}, {}
    for side in ("left", "right"):
        link = tree.find(f"link[@name='{side}_wrist_yaw_link']")
        meshes, identity = [], []
        for collider in link.findall("collision"):
            mesh = collider.find("geometry/mesh")
            if mesh is None or "rubber_hand" not in mesh.get("filename", ""):
                continue
            path = asset_dir / "source/meshes/g1" / Path(mesh.get("filename")).name
            data = trimesh.load_mesh(path, process=False)
            # STL repeats three vertices for every face. Weld coincident source
            # vertices before checking closed collision topology; keep native
            # collision assets, shape, transforms and face winding unchanged.
            data.merge_vertices(digits_vertex=12)
            origin = collider.find("origin")
            xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
            rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
            vertices = data.vertices @ Rotation.from_euler("xyz", rpy).as_matrix().T + xyz
            welded_vertices, welded_faces = validate_mesh(torch.tensor(vertices, dtype=torch.float64),
                                                         torch.tensor(data.faces.copy(), dtype=torch.long))
            meshes.append({"vertices": welded_vertices, "faces": welded_faces})
            identity.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "origin_xyz": xyz.tolist(), "origin_rpy": rpy})
        if len(meshes) != (3 if native_hands is None else 5):
            raise ValueError("Unexpected number of current hand collision hulls")
        probes = (old[side]["palm_points"] + torch.tensor(old[side]["mount"])).numpy()
        nearest, distances = [], []
        for mesh in meshes:
            m = trimesh.Trimesh(mesh["vertices"].numpy(), mesh["faces"].numpy(), process=False)
            closest, distance, _ = trimesh.proximity.closest_point_naive(m, probes)
            nearest.append(closest)
            distances.append(distance)
        best = np.argmin(np.stack(distances), axis=0)
        projected = np.stack(nearest)[best, np.arange(len(probes))]
        hands[side] = {"collision_meshes": meshes, "palm_points": torch.tensor(projected, dtype=torch.float32),
                       "mount": [0., 0., 0.]}
        evidence[side] = {"colliders": identity, "probe_count": len(probes),
                          "calibrated_to_current_projection_max_m": float(np.min(distances, axis=0).max()),
                          "collision_topology": "source STL coincident vertices welded at 12 decimal places; closed winding verified",
                          "probe_frame": "current_wrist_link_metres",
                          "probe_sha256": hashlib.sha256(projected.astype("float32").tobytes()).hexdigest()}
    return hands, evidence


class TablesReward:
    max_step_reward = MAX_STEP_REWARD

    def __init__(self, mdp, assets, hands_binding, *, reference_contacts=None, native_hands=None):
        self.mdp, self.motion, self.kin = mdp, mdp.motion, mdp.kin
        self.assets = {int(a["asset_id"]): a for a in assets}
        self.geometry = {i: bound_tensor(a["geometry_cache"]) for i, a in self.assets.items()}
        self.hands, self.hand_evidence = current_hands(hands_binding, native_hands=native_hands)
        self._object_meshes = {i: _prepare_exact_meshes(g["collision_meshes"]) for i, g in self.geometry.items()}
        self._hand_meshes = {side: _prepare_exact_meshes(h["collision_meshes"]) for side, h in self.hands.items()}
        tracked = [self.kin.body_names[i] for i in self.kin.tracked_idx]
        self.wrists = [tracked.index(side + "_wrist_yaw_link") for side in self.hands]
        self._reference_contacts = {}
        self.contract = {"source_branch": "hoi/v1.1-observation-matching", "source_commit": "66d9184",
                         "terms": list(TERM_NAMES), "dt": .02,
                         "reference_contact_schema": "reference_hand_object_geometric_intent_v2_surface_intersection_unknown",
                         "hand_geometry": self.hand_evidence}
        self._frozen_contacts = None
        if reference_contacts is not None:
            from .prepare_contacts import load_contact_cache
            expected, valid, metadata = load_contact_cache(reference_contacts, self.motion, assets, hands_binding,
                                                           native_hands=native_hands)
            self._frozen_contacts = expected.to(self.motion.device), valid.to(self.motion.device)
            self.contract["frozen_reference_contacts"] = metadata

    def _expected(self, frames):
        if getattr(self, "_frozen_contacts", None) is not None:
            expected, valid = self._frozen_contacts
            return expected[frames], valid[frames]
        # Reference geometry is constant, so cache by frame. Never use simulated
        # or imagined contact labels to construct the reference intent.
        pending = [int(f) for f in frames.unique().tolist() if int(f) not in self._reference_contacts]
        if pending:
            idx = torch.tensor(pending, device=self.motion.device)
            with torch.no_grad():
                bp, bR, _, _ = (x.cpu() for x in self.kin.forward(*self.motion.ref_state(idx)))
                op, oR, _, _ = (x.cpu() for x in self.motion.object_ref(idx))
            for j, frame in enumerate(pending):
                asset_id = int(self.motion.clip_asset_ids[self.motion.clip_id[frame]])
                gaps, depths, crossings = [], [], []
                for side, wrist in zip(self.hands, self.wrists):
                    transformed = []
                    for mesh in self._hand_meshes[side]:
                        world = bp[j, wrist].double() + mesh.vertices @ bR[j, wrist].double().T
                        local = (world - op[j].double()) @ oR[j].double()
                        transformed.append(_rigid_mesh_vertices(mesh, local))
                    gap, depth, crossing = _exact_mesh_gap(transformed, self._object_meshes[asset_id], return_intersection=True)
                    gaps.append(gap); depths.append(depth); crossings.append(crossing)
                gap, depth, crossing = torch.tensor(gaps), torch.tensor(depths), torch.tensor(crossings)
                trustworthy = ~crossing & ~((gap <= 1e-7) & (depth > 1e-7))
                e, v = expected_schedule_from_distances(gap, depth, trustworthy)
                self._reference_contacts[frame] = (e, v)
        expected = torch.stack([self._reference_contacts[int(f)][0] for f in frames.tolist()]).to(frames.device)
        valid = torch.stack([self._reference_contacts[int(f)][1] for f in frames.tolist()]).to(frames.device)
        return expected, valid

    def hand_distances(self, bodies, O, frames):
        bp, bR = bodies[:2]
        probes = torch.stack([h["palm_points"] for h in self.hands.values()]).to(bp)
        world = bp[:, self.wrists, None] + (bR[:, self.wrists, None] @ probes.unsqueeze(-1)).squeeze(-1)
        local = (O[1].transpose(-1, -2)[:, None, None] @ (world - O[0][:, None, None]).unsqueeze(-1)).squeeze(-1)
        ids = self.motion.clip_asset_ids[self.motion.clip_id[frames]]
        result = torch.zeros_like(local[..., 0])
        for aid in ids.unique().tolist():
            mask, g = ids == aid, self.geometry[int(aid)]
            result[mask] = query_grid(g["sdf_grid"], g["grid_min"], g["grid_max"], local[mask])
        return result

    def terms(self, S, O, f, a, a_prev, contact, contact_valid=None):
        expected, valid = self._expected(f)
        bodies = self.kin.forward(*S)
        contact_valid = torch.ones_like(contact) if contact_valid is None else contact_valid
        return reward_terms(self.mdp, S, O, f, a, a_prev, expected, valid,
                            contact[..., :2], contact_valid[..., :2], self.hand_distances(bodies, O, f), bodies=bodies)

    def __call__(self, S, O, f, a, a_prev, contact, contact_valid=None):
        return sum(self.terms(S, O, f, a, a_prev, contact, contact_valid).values())

    def object_failure_threshold(self, f):
        ids = self.motion.clip_asset_ids[self.motion.clip_id[f]]
        bbox = torch.stack([self.geometry[int(i)]["bbox_diagonal"] for i in ids.tolist()]).to(f.device)
        return (bbox * 2.).clamp_min(1.)

    def termination(self, S, O, f):
        return self.mdp.termination(S, f) | ((O[0] - self.motion.object_ref(f)[0]).norm(dim=-1) > self.object_failure_threshold(f))

    def soft_termination(self, S, O, f):
        human = self.mdp.soft_termination(S, f)
        obj = torch.sigmoid(((O[0] - self.motion.object_ref(f)[0]).norm(dim=-1) - self.object_failure_threshold(f)) / .1)
        return 1 - (1 - human) * (1 - obj)
