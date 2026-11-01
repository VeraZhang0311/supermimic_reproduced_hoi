"""Deterministic CPU geometry planning for disposable native contact probes.

The planner queries the actual compiled colliders. It never executes dynamics,
changes a collision filter or substitutes a visual mesh. The returned poses are
initial conditions only; success still requires live GPU interval impulses.
"""
from __future__ import annotations

import math
import re

import numpy as np

SCHEMA = "hoi_native_exposed_region_pose_v1"


def _directions(count=384):
    k = np.arange(count)
    z = 1 - 2 * (k + .5) / count
    phi = k * (math.pi * (3 - math.sqrt(5)))
    radial = np.sqrt(1 - z * z)
    return np.vstack((np.eye(3), -np.eye(3), np.column_stack((radial*np.cos(phi), radial*np.sin(phi), z))))


def rotation_between(source, target):
    """Proper rotation, including the antiparallel case."""
    a, b = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), np.dot(a, b)
    if c < -1 + 1e-12:
        axis = np.eye(3)[np.argmin(np.abs(a))]
        axis = np.cross(a, axis); axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + skew + skew @ skew / (1 + c)


def object_support_facet(meshes, *, tolerance=2e-6):
    """Choose one real exposed convex-component facet, never a union average.

    A chair's feet can share a supporting plane while their average lies in
    empty space. The selected point is a convex combination of coplanar
    vertices from one actual hull only, and its plane supports every hull.
    """
    vertices = [np.asarray(mesh["vertices"], dtype=float) for mesh in meshes]
    union = np.concatenate(vertices)
    best = None
    for mesh_id, (mesh, points) in enumerate(zip(meshes, vertices)):
        for face_id, face in enumerate(np.asarray(mesh["faces"], dtype=int)):
            triangle = points[face]
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            area2 = np.linalg.norm(normal)
            if area2 <= 1e-12:
                continue
            normal /= area2
            if np.dot(normal, triangle.mean(0) - points.mean(0)) < 0:
                normal = -normal
            plane = float(triangle[0] @ normal)
            excess = float((union @ normal).max() - plane)
            if excess > tolerance:
                continue
            ids = np.flatnonzero(np.abs(points @ normal - plane) <= tolerance)
            if len(ids) < 3:
                continue
            center = points[ids].mean(0)
            score = float(area2)
            if best is None or score > best[0]:
                best = score, mesh_id, face_id, ids, center, normal, plane, excess
    if best is None:
        raise RuntimeError("actual object collision union has no exposed component facet")
    _, mesh_id, face_id, ids, center, normal, plane, excess = best
    rotation = rotation_between(normal, [0, 0, -1])
    support = rotation @ center
    minimum = float((union @ rotation.T)[:, 2].min())
    if abs(support[2] - minimum) > tolerance:
        raise RuntimeError("selected actual facet center is not on the union support plane")
    evidence = {"schema": "hoi_native_object_support_facet_v1", "mesh_index": mesh_id,
        "source_face_index": face_id, "coplanar_component_vertex_ids": ids.tolist(),
        "point_local": center.tolist(), "normal_local": normal.tolist(),
        "rotation_matrix": rotation.tolist(), "point_rotated": support.tolist(),
        "maximum_union_support_excess_m": excess,
        "support_plane_error_m": abs(float(support[2]) - minimum),
        "point_rule": "mean_of_coplanar_vertices_in_one_actual_convex_hull",
        "all_worlds_same_orientation": True}
    return rotation, support, evidence


class CompiledProbePlanner:
    def __init__(self, model, roles, *, joint_names, environment_ids=None, environment=0):
        import mujoco
        from .contacts import BODY_ROLES
        from .native import compiled_convex_mesh
        self.model, self.roles = model, list(roles)
        self.joint_names = list(joint_names)
        self.joints = []
        for name in self.joint_names:
            matches = [j for j in range(model.njnt)
                if re.search(r"(?:^|[/_])" + re.escape(name) + r"$", model.joint(j).name or "")]
            if environment_ids is not None:
                matches = [j for j in matches if any(int(model.geom_bodyid[g]) == int(model.jnt_bodyid[j]) and
                    environment_ids[g] == environment for g in range(model.ngeom))] or matches
            # In a multi-environment CPU model the compiler orders env_0 first.
            if not matches:
                raise RuntimeError("Compiled probe joint missing: " + name)
            self.joints.append(matches[0])
        root = int(model.jnt_bodyid[self.joints[0]])
        while root > 0:
            js = range(int(model.body_jntadr[root]), int(model.body_jntadr[root] + model.body_jntnum[root]))
            free = [j for j in js if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]
            if free:
                break
            root = int(model.body_parentid[root])
        if root == 0 or len(free) != 1:
            raise RuntimeError("Compiled robot has no unambiguous floating root")
        self.root_body, self.root_joint = root, free[0]
        self.root_address = int(model.jnt_qposadr[self.root_joint])
        self.addresses = np.asarray(model.jnt_qposadr[self.joints], dtype=int)
        descendants = {root}
        for body in range(root + 1, model.nbody):
            if int(model.body_parentid[body]) in descendants:
                descendants.add(body)
        self.geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in descendants and
            (int(model.geom_contype[g]) != 0 or int(model.geom_conaffinity[g]) != 0)]
        if not set(BODY_ROLES).issubset({roles[g] for g in self.geoms}):
            raise RuntimeError("Compiled robot probe does not cover all 15 active region colliders")
        self.meshes = {g: compiled_convex_mesh(model, int(model.geom_dataid[g]))[0]
                       for g in self.geoms if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH)}
        self.data = mujoco.MjData(model)
        self.directions = _directions()
        self.plans = {}
        self.search_evidence = {}

    def support(self, directions):
        """Exact per-geom support projections for unit world directions."""
        import mujoco
        d = np.asarray(directions, dtype=float).reshape(-1, 3)
        values = []
        for geom in self.geoms:
            rotation = self.data.geom_xmat[geom].reshape(3, 3)
            local = d @ rotation
            size, kind = self.model.geom_size[geom], int(self.model.geom_type[geom])
            if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
                radius = (local @ self.meshes[geom].T).max(1)
            elif kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                radius = np.full(len(d), size[0])
            elif kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                radius = size[0] + size[1]*np.abs(local[:, 2])
            elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
                radius = np.abs(local) @ size
            elif kind == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                radius = np.linalg.norm(local * size, axis=1)
            elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                radius = size[0]*np.linalg.norm(local[:, :2], axis=1) + size[1]*np.abs(local[:, 2])
            else:
                raise RuntimeError(f"Unsupported compiled robot collider type {kind}")
            values.append(d @ self.data.geom_xpos[geom] + radius)
        return np.asarray(values)

    def _set(self, q, rotation=None, position=None):
        import mujoco
        from scipy.spatial.transform import Rotation
        self.data.qpos[:] = self.model.qpos0
        a = self.root_address
        self.data.qpos[a:a+3] = [0, 0, 3] if position is None else position
        self.data.qpos[a+3:a+7] = ([1, 0, 0, 0] if rotation is None else
            Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]])
        self.data.qpos[self.addresses] = q
        mujoco.mj_kinematics(self.model, self.data)

    def _self_contacts(self, data):
        """Actual accepted collision pairs, preserving compiled filtering."""
        robot = set(self.geoms)
        result = []
        for contact in data.contact:
            a,b = int(contact.geom1),int(contact.geom2)
            if a not in robot or b not in robot:
                continue
            ba,bb = int(self.model.geom_bodyid[a]),int(self.model.geom_bodyid[b])
            result.append({"geom_ids":[a,b], "geom_names":[self.model.geom(a).name,self.model.geom(b).name],
                "body_ids":[ba,bb], "body_names":[self.model.body(ba).name,self.model.body(bb).name],
                "body_weld_ids":[int(self.model.body_weldid[ba]),int(self.model.body_weldid[bb])],
                "roles":[self.roles[a],self.roles[b]], "signed_distance_m":float(contact.dist)})
        return sorted(result,key=lambda row:row["signed_distance_m"])

    def _local_clearance_seed(self, initial, penetration):
        """Resolve narrow legal subchain postures before whole-body sampling.

        Only joint coordinates are searched. Real ancestor/descendant collider
        pairs determine the one- or two-joint chains; no role is exempted and
        no collision/filter/model parameter changes.
        """
        import itertools
        import mujoco
        q = np.asarray(initial, dtype=float).copy()
        limits = self.model.jnt_range[self.joints]
        self._set(q);mujoco.mj_collision(self.model,self.data)
        contacts = self._self_contacts(self.data)
        chains = {}
        for contact in contacts:
            if contact["signed_distance_m"] >= -penetration:
                continue
            a,b = contact["body_ids"]
            path = []
            for ancestor,descendant in ((a,b),(b,a)):
                current=descendant;candidate=[]
                while current>0 and current!=ancestor:
                    candidate.append(current);current=int(self.model.body_parentid[current])
                if current==ancestor:
                    path=candidate
                    break
            columns=[i for i,joint in enumerate(self.joints) if int(self.model.jnt_bodyid[joint]) in path]
            if not 1 <= len(columns) <= 2:
                continue
            key=tuple(columns)
            chains.setdefault(key,set()).add(tuple(sorted((a,b))))
        records=[];fixed=[]
        for columns,pairs in chains.items():
            grids=[np.unique(np.r_[np.linspace(*limits[i],41),q[i]]) for i in columns]
            best=None;queries=0
            for values in itertools.product(*grids):
                candidate=q.copy();candidate[list(columns)]=values
                self._set(candidate);mujoco.mj_collision(self.model,self.data)
                depth=0.
                for contact in self.data.contact:
                    bodies=tuple(sorted((int(self.model.geom_bodyid[contact.geom1]),int(self.model.geom_bodyid[contact.geom2]))))
                    if bodies in pairs:
                        depth=max(depth,-float(contact.dist))
                distance=float(np.square(candidate[list(columns)]-initial[list(columns)]).sum())
                objective=(depth,distance)
                queries+=1
                if best is None or objective<best[0]:
                    best=(objective,candidate)
            q=best[1]
            accepted=best[0][0] <= penetration+1e-6
            if accepted:
                fixed.extend(columns)
            records.append({"body_pairs":[list(pair) for pair in sorted(pairs)],
                "joint_columns":list(columns),"joint_names":[self.joint_names[i] for i in columns],
                "grid_queries":queries,"selected_joint_positions":q[list(columns)].tolist(),
                "maximum_pair_penetration_m":best[0][0],"within_original_threshold":accepted})
        self._set(q);mujoco.mj_collision(self.model,self.data)
        contacts=self._self_contacts(self.data)
        depth=max(0.,-contacts[0]["signed_distance_m"]) if contacts else 0.
        return q,sorted(set(fixed)),{"method":"compiled_ancestor_subchain_grid_then_global_search",
            "grid_points_per_joint":41,"chains":records,"seed_joint_positions":q.tolist(),
            "seed_maximum_self_penetration_m":depth,"fixed_seed_columns":sorted(set(fixed)),
            "maximum_allowed_self_penetration_m":penetration,"model_or_filter_changes":False}

    def search(self, initial_q, *, max_candidates=1024, clearance=.005, penetration=.002):
        """Find an exposed legal posture per region, without deep self overlap."""
        import mujoco
        from .contacts import BODY_ROLES
        initial = np.asarray(initial_q, dtype=float)
        if initial.shape != (len(self.joints),) or not np.isfinite(initial).all():
            raise ValueError("Probe posture needs all finite canonical joint positions")
        limits = self.model.jnt_range[self.joints]
        lo, hi = limits[:, 0], limits[:, 1]
        rng = np.random.default_rng(1731)
        seed,fixed_columns,seed_evidence=self._local_clearance_seed(np.clip(initial,lo,hi),penetration)
        best, candidates, rejected = {}, 0, 0
        rejected_pairs, first_cases = {}, []
        min_self_depth,max_self_depth=float("inf"),0.
        least_penetrating = None
        self.search_evidence = {"initial_joint_positions":initial.tolist(),"joint_limits":limits.tolist(),
            "root_body_id":self.root_body,"root_joint_id":self.root_joint,"canonical_joint_ids":self.joints,
            "robot_geom_ids":self.geoms,"model_ngeom":int(self.model.ngeom),"model_nbody":int(self.model.nbody),
            "model_disableflags":int(self.model.opt.disableflags),"model_enableflags":int(self.model.opt.enableflags),
            "local_chain_seed_search":seed_evidence}
        for candidate in range(max_candidates):
            if candidate == 0:
                q = np.clip(initial, lo, hi)
            elif candidate == 1:
                q = np.clip(np.zeros_like(initial), lo, hi)
            elif candidate == 2:
                q = seed.copy()
            else:
                # Alternate the repaired seed (preserving narrowly feasible
                # subchains) with unrestricted legal full-body exploration.
                span = .3 if candidate % 4 < 2 else .9
                use_seed = candidate % 2 == 1
                center = seed if use_seed else initial
                q = np.clip(center + rng.uniform(-span, span, len(initial)) * (hi-lo), lo, hi)
                if use_seed:
                    q[fixed_columns] = seed[fixed_columns]
            self._set(q)
            mujoco.mj_collision(self.model, self.data)
            contacts = self._self_contacts(self.data)
            self_depth = max(0.,-contacts[0]["signed_distance_m"]) if contacts else 0.
            min_self_depth,max_self_depth=min(min_self_depth,self_depth),max(max_self_depth,self_depth)
            if least_penetrating is None or self_depth < least_penetrating["maximum_self_penetration_m"]:
                least_penetrating={"candidate":candidate,"joint_positions":q.tolist(),
                    "maximum_self_penetration_m":self_depth,"deepest_contacts":contacts[:10]}
            for row in contacts:
                if row["signed_distance_m"] >= -penetration-1e-6:
                    continue
                key=",".join(str(g) for g in row["geom_ids"])
                item=rejected_pairs.setdefault(key,{"contact":row,"candidate_count":0,
                    "minimum_signed_distance_m":row["signed_distance_m"]})
                item["candidate_count"]+=1
                item["minimum_signed_distance_m"]=min(item["minimum_signed_distance_m"],row["signed_distance_m"])
            if candidate < 2:
                # Compare the complete public position pipeline in an
                # independent scratch data buffer; never change solver/model.
                diagnostic=mujoco.MjData(self.model);diagnostic.qpos[:]=self.data.qpos
                mujoco.mj_fwdPosition(self.model,diagnostic)
                full_contacts=self._self_contacts(diagnostic)
                first_cases.append({"candidate":candidate,"joint_positions":q.tolist(),
                    "kinematics_then_collision":contacts[:20],"full_position_pipeline":full_contacts[:20]})
            candidates += 1
            if self_depth > penetration + 1e-6:
                rejected += 1
                continue
            supports = self.support(self.directions)
            for role in BODY_ROLES:
                target = np.array([self.roles[g] == role for g in self.geoms])
                target_max, other_max = supports[target].max(0), supports[~target].max(0)
                margin = target_max - other_max
                direction = int(margin.argmax())
                if role not in best or margin[direction] > best[role]["exposure_margin_m"]:
                    best[role] = {"joint_positions": q.tolist(), "outward_direction": self.directions[direction].tolist(),
                        "exposure_margin_m": float(margin[direction]), "candidate_index": candidate,
                        "maximum_self_penetration_m": self_depth}
            if all(role in best and best[role]["exposure_margin_m"] >= clearance for role in BODY_ROLES):
                break
        missing = [role for role in BODY_ROLES if role not in best or best[role]["exposure_margin_m"] < 0.]
        margins = {str(role): best.get(role, {}).get("exposure_margin_m") for role in BODY_ROLES}
        self.plans = best
        self.search_evidence.update(candidate_count=candidates,self_collision_rejected_count=rejected,
            maximum_candidates=max_candidates,minimum_required_exposure_margin_m=0.,
            preferred_exposure_clearance_m=clearance,physics_steps=0,collisions_disabled=False,joint_limits_preserved=True,
            minimum_candidate_self_penetration_m=min_self_depth,maximum_candidate_self_penetration_m=max_self_depth,
            least_penetrating_candidate=least_penetrating,deep_pairs=sorted(rejected_pairs.values(),key=lambda row:-row["candidate_count"]),
            first_candidate_pipeline_comparison=first_cases,exposure_margins_m=margins,missing_roles=missing)
        if missing:
            raise RuntimeError(f"No shallow exposed compiled robot posture for roles {missing}; margins={margins}; "
                f"self_rejected={rejected}/{candidates}; min_self_penetration_m={min_self_depth}")
        return best

    def placement(self, role, *, ground, penetration=.002, clearance=.002):
        """Pose target down on ground, or up with the entire robot above ground."""
        from .native import compiled_geom_support
        from .contacts import CHANNELS, BODY_ROLES
        plan = self.plans[role]
        rotation = rotation_between(plan["outward_direction"], [0, 0, -1 if ground else 1])
        q = np.asarray(plan["joint_positions"])
        self._set(q, rotation, np.zeros(3))
        low = -self.support([[0, 0, -1]])[:, 0]
        target = np.array([self.roles[g] == role for g in self.geoms])
        z = -float(low[target].min()) - penetration if ground else -float(low.min()) + clearance
        root = np.array([0., 0., z])
        self._set(q, rotation, root)
        low += z
        direction = np.array([0., 0., -1 if ground else 1])
        supports = self.support([direction])[:, 0]
        index = int(np.where(target, supports, -np.inf).argmax())
        geom = self.geoms[index]
        gr = self.data.geom_xmat[geom].reshape(3, 3)
        point, _, proof = compiled_geom_support(self.model, geom, gr.T @ direction)
        surface = self.data.geom_xpos[geom] + gr @ point.numpy()
        maximum_other_depth = float(max(0., -low[~target].min()))
        if maximum_other_depth > penetration + 2e-6:
            raise RuntimeError("Exposed probe would deeply bury a non-target collider")
        evidence = {"schema": SCHEMA, "region": CHANNELS[BODY_ROLES.index(role)], "ground_negative": ground,
            "prescribed_penetration_m": penetration, "maximum_allowed_non_target_ground_penetration_m": penetration,
            "maximum_non_target_ground_penetration_m": maximum_other_depth,
            "target_minimum_world_z_m": float(low[target].min()), "robot_minimum_world_z_m": float(low.min()),
            "robot_geom_ids": self.geoms, "robot_geom_roles": [self.roles[g] for g in self.geoms],
            "per_geom_minimum_world_z_m": low.tolist(), "root_position": root.tolist(),
            "root_rotation": rotation.tolist(), "joint_names": self.joint_names,
            "joint_positions": q.tolist(), "compiled_surface": proof,
            "surface_point_world": surface.tolist(), "outward_normal_world": direction.tolist(),
            **plan, "search": self.search_evidence}
        return root, rotation, q, surface, direction, evidence

