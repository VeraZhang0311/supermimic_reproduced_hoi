"""Traceable central palm ray probes on the original hand STL and actual hulls."""
from pathlib import Path

import torch


def _ray_surface(meshes, origins, directions):
    """First triangle hit per ray, with reproducible source triangle indices."""
    triangles = torch.cat([m["vertices"].double()[m["faces"].long()] for m in meshes])
    v0, e1, e2 = triangles[:, 0], triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    points, faces = [], []
    for origin, direction in zip(origins.double(), directions.double()):
        cross = torch.linalg.cross(direction.expand_as(e2), e2)
        determinant = (e1 * cross).sum(-1)
        inverse = torch.where(determinant.abs() > 1e-12, 1 / determinant, 0.)
        delta = origin - v0
        u = inverse * (delta * cross).sum(-1)
        q = torch.linalg.cross(delta, e1)
        v = inverse * (direction * q).sum(-1)
        t = inverse * (e2 * q).sum(-1)
        valid = (determinant.abs() > 1e-12) & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t >= 0)
        distance, face = torch.where(valid, t, float("inf")).min(0)
        if not torch.isfinite(distance):
            raise RuntimeError("Specified central palm ray did not hit the actual hand collider")
        points.append(origin + distance * direction)
        faces.append(int(face))
    return torch.stack(points).float(), faces


def palm_patch_points(meshes, source_meshes, side):
    """Central inner palm patch in the verified original STL hand frame.

    Original meshes have x along wrist→fingers, z across the palm and mirrored
    y thickness. Curled fingers extend beyond x≈0.075m; the fixed x∈[.025,.055],
    z∈[-.015,.015] patch lies on the central palm. Rays approach the inner face
    from -y (left) / +y (right). Every sample is a real cooked-collider surface
    intersection, independently compared with the byte-verified source STL.
    """
    if side not in ("left", "right"):
        raise ValueError("unknown hand side")
    x = torch.tensor([.025, .035, .045, .055], dtype=torch.float64)
    z = torch.tensor([-.015, -.005, .005, .015], dtype=torch.float64)
    gx, gz = torch.meshgrid(x, z, indexing="ij")
    sign = -1 if side == "left" else 1
    origins = torch.stack((gx.flatten(), torch.full((16,), sign * .15), gz.flatten()), -1)
    direction = torch.tensor([0., -sign, 0.], dtype=torch.float64).expand_as(origins)
    actual, actual_faces = _ray_surface(meshes, origins, direction)
    source, source_faces = _ray_surface(source_meshes, origins, direction)
    error = torch.linalg.vector_norm(actual - source, dim=-1)
    if float(error.max()) > .010:
        raise RuntimeError(f"Cooked palm patch differs from source palm by >1cm: {float(error.max())}m")
    return actual, {"version": "central_inner_palm_raycast_v1", "frame": "original_hand_STL_metres",
                    "x_m": x.tolist(), "z_m": z.tolist(), "approach_axis": "-y" if side == "left" else "+y",
                    "source_triangle_indices": source_faces, "compiled_triangle_indices": actual_faces,
                    "max_source_collider_patch_gap_m": float(error.max()), "max_source_collider_patch_gap_allowed_m": .010,
                    "source_palm_points": source.tolist(), "palm_selection_verified": True}


def _write_palm_qa(path, source_vertices, points):
    """Dependency-free PNG projection: XY, XZ, YZ; blue source, red palm patch."""
    import binascii
    import numpy as np
    import struct
    import zlib
    vertices = source_vertices.detach().cpu().numpy()
    probes = points.detach().cpu().numpy()
    image = np.full((600, 1500, 3), 255, dtype=np.uint8)
    for panel, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
        low = vertices.min(0)
        scale = min(420 / np.ptp(vertices[:, a]), 480 / np.ptp(vertices[:, b]))
        x = np.rint(panel * 500 + 40 + (vertices[:, a] - low[a]) * scale).astype(int)
        y = np.rint(560 - (vertices[:, b] - low[b]) * scale).astype(int)
        image[y, x] = [30, 60, 120]
        for p in probes:
            px = round(panel * 500 + 40 + (p[a] - low[a]) * scale)
            py = round(560 - (p[b] - low[b]) * scale)
            image[py - 4:py + 5, px - 4:px + 5] = [220, 20, 30]
    def chunk(kind, data):
        return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", binascii.crc32(kind + data) & 0xffffffff)
    raw = b"".join(b"\0" + row.tobytes() for row in image)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!2I5B", 1500, 600, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    Path(path).write_bytes(png)
