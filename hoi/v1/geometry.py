"""Metric collision geometry, deterministic shape sampling, and differentiable SDFs.

Meshes and grids use object link-local metres.  Shape samples are never used as
the collider, a COM estimate, or pair-contact truth.  Grid queries retain their
Jacobian with respect to robot/object coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import torch
from torch.nn import functional as F

from hoi.hand_collision import ORIGIN


def tensor_sha256(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode() + value.numpy().tobytes()).hexdigest()


def validate_mesh(vertices, faces):
    vertices, raw_faces = torch.as_tensor(vertices, dtype=torch.float64), torch.as_tensor(faces)
    if raw_faces.is_floating_point() and (not bool(torch.isfinite(raw_faces).all()) or not torch.equal(raw_faces, raw_faces.round())):
        raise ValueError("mesh face indices must be finite integers")
    faces = raw_faces.to(dtype=torch.long)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("collision meshes require vertices[V,3] and triangle faces[F,3]")
    if len(vertices) < 4 or len(faces) < 4 or not bool(torch.isfinite(vertices).all()):
        raise ValueError("invalid collision mesh")
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise ValueError("mesh face index out of range")
    tri = vertices[faces]
    if bool((torch.linalg.vector_norm(torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1), dim=-1) < 1e-14).any()):
        raise ValueError("degenerate collision triangle")
    # Signed distances need closed, consistently oriented components.
    edges = torch.cat((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    canonical, _ = edges.sort(-1)
    unique, inverse, counts = canonical.unique(dim=0, return_inverse=True, return_counts=True)
    if not bool((counts == 2).all()):
        raise ValueError("signed collision mesh must be watertight")
    direction = (edges[:, 0] < edges[:, 1]).long() * 2 - 1
    totals = torch.zeros(len(unique), dtype=torch.long).scatter_add_(0, inverse, direction)
    if bool((totals != 0).any()):
        raise ValueError("collision face winding is inconsistent")
    return vertices, faces


def point_mesh_signed_distance(points, vertices, faces, chunk=512):
    """Exact closest triangle distance with generalized-winding inside test.

    Face projection plus all three segment distances covers triangle Voronoi
    regions. Winding is orientation independent for closed components. This CPU
    audit implementation is deliberately separate from the training grid.
    """
    vertices, faces = validate_mesh(vertices, faces)
    return _point_mesh_signed_distance(points, vertices, faces, chunk)


def _point_mesh_signed_distance(points, vertices, faces, chunk=512):
    """Exact arithmetic kernel; caller must already validate this closed mesh."""
    points = torch.as_tensor(points, dtype=torch.float64)
    shape = points.shape[:-1]
    tri = vertices[faces]
    a, b, c = tri.unbind(1)
    e0, e1 = b - a, c - a
    normal = torch.cross(e0, e1, dim=-1)
    norm2 = normal.square().sum(-1)
    out = []
    chunk = min(chunk, max(1, 262144 // len(faces)))
    for p in points.reshape(-1, 3).split(chunk):
        pa = p[:, None] - a
        h = (pa * normal).sum(-1) / norm2
        projected = p[:, None] - h[..., None] * normal
        rel = projected - a
        d00, d01, d11 = (e0 * e0).sum(-1), (e0 * e1).sum(-1), (e1 * e1).sum(-1)
        d20, d21 = (rel * e0).sum(-1), (rel * e1).sum(-1)
        denominator = d00 * d11 - d01.square()
        u, v = (d11 * d20 - d01 * d21) / denominator, (d00 * d21 - d01 * d20) / denominator
        inside_face = (u >= 0) & (v >= 0) & (u + v <= 1)
        dface = h.square() * norm2
        d2 = torch.where(inside_face, dface, torch.inf)
        for start, end in ((a, b), (b, c), (c, a)):
            edge = end - start
            weight = ((p[:, None] - start) * edge).sum(-1) / edge.square().sum(-1)
            nearest = start + weight.clamp(0, 1)[..., None] * edge
            d2 = torch.minimum(d2, (p[:, None] - nearest).square().sum(-1))
        x, y, z = a - p[:, None], b - p[:, None], c - p[:, None]
        lx, ly, lz = (torch.linalg.vector_norm(t, dim=-1) for t in (x, y, z))
        numerator = (x * torch.cross(y, z, dim=-1)).sum(-1)
        denominator = lx * ly * lz + (x * y).sum(-1) * lz + (y * z).sum(-1) * lx + (z * x).sum(-1) * ly
        winding = (2 * torch.atan2(numerator, denominator)).sum(-1).abs()
        sign = torch.where(winding > 2 * math.pi, -1., 1.)
        out.append(d2.min(-1).values.clamp_min(0).sqrt() * sign)
    return torch.cat(out).reshape(shape)


def collision_distance(points, meshes):
    """Minimum component signed distances, with exact bound-based pruning.

    A component is contained in its AABB, hence phi_box <= phi_component
    everywhere (including inside/overlapping components). Once phi_box is no
    smaller than the current minimum, that component cannot improve the result.
    A 1e-10 metre conservative margin only schedules extra exact triangle work.
    No surface, distance value, sign, or contact threshold is approximated here.
    """
    if not meshes:
        raise ValueError("collision distance requires at least one explicit component")
    points = torch.as_tensor(points, dtype=torch.float64)
    if not bool(torch.isfinite(points).all()):
        raise ValueError("collision distance query contains nonfinite coordinates")
    shape = points.shape[:-1]
    flat = points.reshape(-1, 3)
    result = flat.new_full((len(flat),), torch.inf)
    for mesh in meshes:
        vertices, faces = validate_mesh(mesh["vertices"], mesh["faces"])
        low, high = vertices.min(0).values, vertices.max(0).values
        delta = torch.maximum(low - flat, flat - high)
        box_phi = torch.linalg.vector_norm(delta.clamp_min(0), dim=-1) + delta.amax(-1).clamp_max(0)
        active = box_phi - 1e-10 < result
        if bool(active.any()):
            exact = _point_mesh_signed_distance(flat[active], vertices, faces)
            result[active] = torch.minimum(result[active], exact)
    return result.reshape(shape)


def _mesh_edges(faces):
    return torch.cat((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])).sort(-1).values.unique(dim=0)


def _edge_triangle_crossing(vertices, faces, other_vertices, other_faces, *, edges=None):
    edges = _mesh_edges(faces) if edges is None else edges
    tri = other_vertices[other_faces]
    a, b, c = tri.unbind(1)
    e0, e1 = b - a, c - a
    n = torch.cross(e0, e1, dim=-1)
    for edge in edges.split(max(1, min(256, 262144 // len(other_faces)))):
        start, finish = vertices[edge[:, 0]], vertices[edge[:, 1]]
        delta = finish - start
        denominator = (delta[:, None] * n).sum(-1)
        nonparallel = denominator.abs() > 1e-14
        t = ((a - start[:, None]) * n).sum(-1) / torch.where(nonparallel, denominator, torch.ones_like(denominator))
        p = start[:, None] + t[..., None] * delta[:, None]
        rel = p - a
        d00, d01, d11 = e0.square().sum(-1), (e0 * e1).sum(-1), e1.square().sum(-1)
        d20, d21 = (rel * e0).sum(-1), (rel * e1).sum(-1)
        det = d00 * d11 - d01.square()
        u, v = (d11 * d20 - d01 * d21) / det, (d00 * d21 - d01 * d20) / det
        if bool((nonparallel & (t >= 0) & (t <= 1) & (u >= 0) & (v >= 0) & (u + v <= 1)).any()):
            return True
    return False


def _edge_edge_gap(vertices, faces, other_vertices, other_faces, *, edges=None, other_edges=None):
    pairs = _mesh_edges(faces) if edges is None else edges
    other = _mesh_edges(other_faces) if other_edges is None else other_edges
    q0, q1 = other_vertices[other[:, 0]], other_vertices[other[:, 1]]
    v = q1 - q0
    minimum = torch.tensor(torch.inf, dtype=vertices.dtype)
    for pair in pairs.split(max(1, min(256, 262144 // len(other)))):
        p0, p1 = vertices[pair[:, 0]], vertices[pair[:, 1]]
        u, w = p1 - p0, p0[:, None] - q0
        a, b, c = u.square().sum(-1)[:, None], (u[:, None] * v).sum(-1), v.square().sum(-1)
        d, e = (u[:, None] * w).sum(-1), (v * w).sum(-1)
        denom = a * c - b.square()
        safe = torch.where(denom.abs() > 1e-14, denom, torch.ones_like(denom))
        s, t = (b * e - c * d) / safe, (a * e - b * d) / safe
        interior = (denom.abs() > 1e-14) & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
        dist2 = (w + s[..., None] * u[:, None] - t[..., None] * v).square().sum(-1)
        # Endpoint-region minima are covered by vertex-to-triangle queries.
        if bool(interior.any()):
            minimum = torch.minimum(minimum, dist2[interior].min().sqrt())
    return minimum


@dataclass(frozen=True)
class _ExactMesh:
    vertices: torch.Tensor
    faces: torch.Tensor
    edges: torch.Tensor
    low: torch.Tensor
    high: torch.Tensor


def _prepare_exact_meshes(meshes):
    """Validate immutable collision topology once for an offline query batch."""
    result = []
    for mesh in meshes:
        v, f = validate_mesh(mesh["vertices"], mesh["faces"])
        result.append(_ExactMesh(v, f, _mesh_edges(f), v.amin(0), v.amax(0)))
    return result


def _rigid_mesh_vertices(mesh, vertices):
    """Reuse topology only for caller-computed rigidly transformed vertices.

    This private path is used after fixed-mount/reference FK, never to admit an
    arbitrary unvalidated mesh. Public query APIs continue to validate inputs.
    """
    vertices = torch.as_tensor(vertices, dtype=torch.float64)
    if vertices.shape != mesh.vertices.shape or not bool(torch.isfinite(vertices).all()):
        raise ValueError("rigid mesh query requires finite vertices with unchanged topology")
    return _ExactMesh(vertices, mesh.faces, mesh.edges, vertices.amin(0), vertices.amax(0))


def exact_mesh_gap(meshes, other_meshes, *, return_intersection=False):
    """Closest collider-surface gap and conservative vertex penetration audit.

    All vertex-face, interior edge-edge, and edge-face intersection cases are
    covered. Penetration reports the deepest sampled collider *vertex*; this is
    an audit bound, not the global minimum-translation penetration depth.

    A pair's AABB gap is a lower bound on its surface gap. Disjoint pairs whose
    lower bound exceeds the current exact minimum cannot affect either output:
    they cannot penetrate, and cannot improve the gap. Every overlapping or
    touching AABB pair is always queried, even once a zero gap has been found,
    so no vertex-penetration evidence is lost. The 1e-10m margin is conservative.

    ``return_intersection`` also returns the edge-face intersection flag. This
    includes boundary touches which the triangle test cannot distinguish from
    penetration; callers must treat it as uncertainty, not a penetration depth.
    """
    return _exact_mesh_gap(_prepare_exact_meshes(meshes), _prepare_exact_meshes(other_meshes),
                           return_intersection=return_intersection)


def _exact_mesh_gap(first, second, *, return_intersection=False):
    """Exact query on privately prevalidated meshes; all geometry math is shared."""
    gap, penetration = torch.tensor(torch.inf, dtype=torch.float64), torch.tensor(0., dtype=torch.float64)
    intersection = False
    pairs = []
    for i, mesh in enumerate(first):
        for j, other in enumerate(second):
            lower = torch.maximum(mesh.low - other.high, other.low - mesh.high).clamp_min(0).norm()
            pairs.append((float(lower), i, j))
    for lower, i, j in sorted(pairs):
        if lower > float(gap) + 1e-10:
            continue
        one, two = first[i], second[j]
        v, f, w, g = one.vertices, one.faces, two.vertices, two.faces
        d0, d1 = _point_mesh_signed_distance(v, w, g), _point_mesh_signed_distance(w, v, f)
        gap = torch.minimum(gap, torch.minimum(d0.abs().min(), d1.abs().min()))
        penetration = torch.maximum(penetration, torch.maximum((-d0).clamp_min(0).max(), (-d1).clamp_min(0).max()))
        if _edge_triangle_crossing(v, f, w, g, edges=one.edges) or _edge_triangle_crossing(w, g, v, f, edges=two.edges):
            gap = gap.new_zeros(())
            intersection = True
        else:
            gap = torch.minimum(gap, _edge_edge_gap(v, f, w, g, edges=one.edges, other_edges=two.edges))
    return (gap, penetration, intersection) if return_intersection else (gap, penetration)


def deterministic_surface_points(meshes, count=4096, seed=0):
    triangles = torch.cat([torch.as_tensor(m["vertices"], dtype=torch.float64)[torch.as_tensor(m["faces"], dtype=torch.long)] for m in meshes])
    areas = torch.linalg.vector_norm(torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1), dim=-1) / 2
    generator = torch.Generator(device="cpu").manual_seed(seed)
    accepted, remaining = [], count
    for _ in range(32):
        number = remaining if len(meshes) == 1 else max(remaining * 2, 128)
        index = torch.multinomial(areas, number, replacement=True, generator=generator)
        random = torch.rand((number, 2), generator=generator, dtype=torch.float64)
        u, v = random[:, :1].sqrt(), random[:, 1:]
        t = triangles[index]
        candidates = ((1 - u) * t[:, 0] + u * (1 - v) * t[:, 1] + u * v * t[:, 2]).float()
        if len(meshes) > 1:
            # Overlapping decomposition hulls contain internal faces that are
            # not exposed collider-union surface. Do not feed these to PointNet.
            candidates = candidates[collision_distance(candidates, meshes).abs() <= 1e-6]
        accepted.append(candidates[:remaining])
        remaining -= min(remaining, len(candidates))
        if not remaining:
            return torch.cat(accepted)
    raise ValueError("unable to sample enough exposed collision-union surface candidates")


def farthest_point_sample(points, count=128):
    """Deterministic FPS with a fixed lexicographic first point and index ties."""
    points = torch.as_tensor(points, dtype=torch.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < count or not bool(torch.isfinite(points).all()):
        raise ValueError("not enough finite surface candidates for FPS")
    # Python sorting is offline and gives an explicit stable lexicographic rule.
    first = min(range(len(points)), key=lambda i: tuple(points[i].tolist()))
    nearest = points.new_full((len(points),), torch.inf)
    chosen, current = [], first
    for _ in range(count):
        chosen.append(current)
        nearest = torch.minimum(nearest, (points - points[current]).square().sum(-1))
        nearest[torch.tensor(chosen)] = -1
        current = int(nearest.argmax())
    return points[chosen].clone()


def query_grid(grid, low, high, points):
    """Trilinear SDF + exterior box distance; does not flatten outside gradients."""
    low, high = low.to(points), high.to(points)
    clamped = points.maximum(low).minimum(high)
    normalized = 2 * (clamped - low) / (high - low) - 1
    original_shape = points.shape[:-1]
    query = normalized.reshape(1, 1, 1, -1, 3)
    sampled = F.grid_sample(grid.to(points).reshape(1, 1, *grid.shape), query,
                            mode="bilinear", padding_mode="border", align_corners=True).reshape(original_shape)
    # Grid domains include padding beyond the collider; outside points are outside.
    return sampled + torch.linalg.vector_norm(points - clamped, dim=-1)


def build_geometry_cache(meshes, *, grid_resolution=64, padding=.08, candidate_count=4096,
                         seed=0, max_grid_error=.01, metadata=None):
    if grid_resolution < 8 or padding <= 0 or candidate_count < 128:
        raise ValueError("invalid collision distance recipe")
    meshes = [{"vertices": validate_mesh(m["vertices"], m["faces"])[0],
               "faces": torch.as_tensor(m["faces"], dtype=torch.long)} for m in meshes]
    vertices = torch.cat([m["vertices"] for m in meshes])
    low, high = vertices.min(0).values - padding, vertices.max(0).values + padding
    z, y, x = torch.meshgrid(*(torch.linspace(float(low[i]), float(high[i]), grid_resolution, dtype=torch.float64)
                                for i in (2, 1, 0)), indexing="ij")
    xyz = torch.stack((x, y, z), -1)
    sdf = collision_distance(xyz, meshes).float()
    candidates = deterministic_surface_points(meshes, candidate_count, seed)
    points = farthest_point_sample(candidates, 128)
    generator = torch.Generator().manual_seed(seed + 1)
    probes = torch.cat((candidates[:512].double(), low + torch.rand((512, 3), generator=generator, dtype=torch.float64) * (high - low)))
    exact = collision_distance(probes, meshes)
    error = (query_grid(sdf.double(), low, high, probes) - exact).abs()
    if float(error.max()) > max_grid_error:
        raise ValueError(f"SDF grid error {float(error.max()):.6g}m exceeds {max_grid_error}m; refine the declared grid")
    return {"points": points, "sdf_grid": sdf, "grid_min": low.float(), "grid_max": high.float(),
            "bbox_diagonal": (vertices.max(0).values - vertices.min(0).values).norm().float(),
            "collision_meshes": meshes,
            "metadata": {**(metadata or {}), "frame": "object_link_local_metres", "point_count": 128,
                         "sampling": "area_weighted_surface_candidates_then_deterministic_FPS",
                         "seed": seed, "candidate_count": candidate_count, "point_hash": tensor_sha256(points),
                         "sdf_sign": "negative_inside_collider_union", "grid_axis_order": "zyx",
                         "signed_field_reference": "minimum_component_signed_triangle_distances; exact outside union, interior magnitude CSG surrogate",
                         "outside_grid_extension": "boundary_signed_value_plus_exterior_box_distance; approximate_magnitude_preserves_gradient",
                         "grid_resolution": grid_resolution, "max_abs_grid_error_m": float(error.max()),
                         "rms_grid_error_m": float(error.square().mean().sqrt()), "grid_qa_probes": len(probes)}}


def palm_world_points(kin, S, hands):
    """Use wrist FK plus the recorded fixed collider mounting transform."""
    pos, rot, _, _ = kin.forward(*S, all_bodies=True)
    points = []
    for side in ("left", "right"):
        index = kin.body_names.index(side + "_wrist_yaw_link")
        hand = hands[side]
        local = torch.as_tensor(hand["palm_points"], device=pos.device, dtype=pos.dtype)
        mount = torch.as_tensor(hand.get("mount", ORIGIN[side]), device=pos.device, dtype=pos.dtype)
        points.append(pos[..., index, None, :] + (rot[..., index, None, :, :] @ (local + mount).unsqueeze(-1)).squeeze(-1))
    return torch.stack(points, -3)


class ShapeAssetBank:
    def __init__(self, assets, geometries, hands, device="cpu"):
        if len(assets) != len(geometries) or not assets:
            raise ValueError("assets and cached collision geometries must correspond")
        self.assets, self.geometries, self.hands, self.device = assets, geometries, hands, str(device)
        self._points = torch.stack([g["points"] for g in geometries]).to(device)
        self._bbox = torch.stack([torch.as_tensor(g["bbox_diagonal"]) for g in geometries]).to(device)
        contexts = []
        for asset in assets:
            p = asset["physical_profile"]
            inertia = torch.as_tensor(p["inertia"], dtype=torch.float32)
            if inertia.shape == (3,):
                inertia = torch.diag(inertia)
            contexts.append(torch.cat((torch.tensor([p["mass"]], dtype=torch.float32), torch.tensor(p["com"], dtype=torch.float32),
                                       inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]],
                                       torch.tensor([p["friction"], p.get("ground_friction", .3)], dtype=torch.float32))))
        self._contexts = torch.stack(contexts).to(device)
        self._grid_cache = {}

    def points(self, ids):
        return self._points[ids]

    def bbox_diagonal(self, ids):
        return self._bbox[ids]

    def physical_context(self, ids):
        return self._contexts[ids]

    def query(self, ids, local_points):
        ids = torch.as_tensor(ids, device=local_points.device, dtype=torch.long)
        flat_ids = ids.reshape(-1)
        tail = local_points.shape[len(ids.shape):-1]
        points = local_points.reshape(len(flat_ids), *tail, 3)
        result = points.new_zeros(points.shape[:-1])
        for index in flat_ids.unique().tolist():
            if index < 0 or index >= len(self.geometries):
                raise ValueError("asset ID outside bound geometry registry")
            mask = flat_ids == index
            g = self.geometries[index]
            cache_key = (index, str(points.device), points.dtype)
            if cache_key not in self._grid_cache:
                self._grid_cache[cache_key] = tuple(g[name].to(device=points.device, dtype=points.dtype)
                                                   for name in ("sdf_grid", "grid_min", "grid_max"))
            result[mask] = query_grid(*self._grid_cache[cache_key], points[mask])
        return result.reshape(ids.shape + tail)

    def hand_distances(self, kin, S, O, asset_ids):
        world = palm_world_points(kin, S, self.hands)
        local = (O[1].transpose(-1, -2)[..., None, None, :, :] @ (world - O[0][..., None, None, :]).unsqueeze(-1)).squeeze(-1)
        return self.query(asset_ids, local)


def expected_schedule_from_distances(gap, penetration=None, geometry_valid=None, *, positive=.02, negative=.04, severe=.02):
    gap = torch.as_tensor(gap)
    valid = (gap <= positive) | (gap >= negative)
    if penetration is not None:
        valid &= torch.as_tensor(penetration) < severe
    if geometry_valid is not None:
        valid &= torch.as_tensor(geometry_valid).bool()
    valid &= torch.isfinite(gap)
    return ((gap <= positive) & valid).float(), valid.float()


def contact_missing_counter(previous, expected, valid, actual, actual_valid=None):
    if actual_valid is None:
        actual_valid = torch.ones_like(valid)
    missing = valid.bool() & actual_valid.bool() & expected.bool() & ~actual.bool()
    return torch.where(missing, previous + 1, torch.zeros_like(previous))
