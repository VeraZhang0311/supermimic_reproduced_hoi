"""Quaternion operations in ULTRA's xyzw convention (rotation columns 1, 3)."""

import torch


def rotate(q, v):
    xyz, w = q[..., :3], q[..., 3:4]
    return (
        v * (2 * w.square() - 1)
        + 2 * w * torch.cross(xyz, v, dim=-1)
        + 2 * xyz * (xyz * v).sum(-1, keepdim=True)
    )


def multiply(a, b):
    av, aw, bv, bw = a[..., :3], a[..., 3:4], b[..., :3], b[..., 3:4]
    return torch.cat(
        (
            aw * bv + bw * av + torch.cross(av, bv, dim=-1),
            aw * bw - (av * bv).sum(-1, keepdim=True),
        ),
        -1,
    )


def inverse(q):
    return torch.cat((-q[..., :3], q[..., 3:4]), -1)


def relative_rotation(reference, current):
    q = multiply(inverse(reference), current)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def rotation6(q):
    x = torch.zeros_like(q[..., :3])
    x[..., 0] = 1
    z = torch.zeros_like(x)
    z[..., 2] = 1
    return torch.cat((rotate(q, x), rotate(q, z)), -1)


def heading(q):
    x = torch.zeros_like(q[..., :3])
    x[..., 0] = 1
    direction = rotate(q, x)
    half = torch.atan2(direction[..., 1], direction[..., 0]) / 2
    zero = torch.zeros_like(half)
    return torch.stack((zero, zero, torch.sin(half), torch.cos(half)), -1)


def world_points(points, position, quaternion):
    q = quaternion[:, None, :].expand(-1, points.shape[1], -1)
    return rotate(q, points) + position[:, None, :]


def interaction_vectors(body_position, object_points):
    vectors = body_position[:, :, None, :] - object_points[:, None, :, :]
    nearest = vectors.norm(dim=-1).argmin(-1)
    return vectors.gather(2, nearest[:, :, None, None].expand(-1, -1, 1, 3)).squeeze(2)


def encode_interaction(vectors):
    norm = vectors.norm(dim=-1, keepdim=True)
    return vectors / (norm + 1e-6) * torch.exp(-5 * norm)


def slerp(a, b, alpha):
    dot = (a * b).sum(-1, keepdim=True)
    b = torch.where(dot < 0, -b, b)
    dot = dot.abs().clamp(0, 1)
    angle = torch.acos(dot)
    denominator = torch.sin(angle).clamp_min(1e-8)
    spherical = (
        torch.sin((1 - alpha) * angle) * a + torch.sin(alpha * angle) * b
    ) / denominator
    linear = (1 - alpha) * a + alpha * b
    result = torch.where(dot > 0.9995, linear, spherical)
    return result / result.norm(dim=-1, keepdim=True).clamp_min(1e-9)
