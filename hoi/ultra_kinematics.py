"""Reconstruct ULTRA link frames from its URDF, root link state and joint state."""

import math
import xml.etree.ElementTree as ET

import torch

from .ultra_math import multiply, rotate


class UltraKinematics:
    def __init__(self, urdf, body_names, dof_names, device="cpu"):
        tree = ET.parse(urdf).getroot()
        self.body_names = list(body_names)
        self.dof_names = list(dof_names)
        self.device = device
        children = {j.find("child").get("link") for j in tree.findall("joint")}
        roots = {link.get("name") for link in tree.findall("link")} - children
        if roots != {self.body_names[0]}:
            raise ValueError(f"Unexpected URDF root: {roots}")
        self.joints = []
        pending = list(tree.findall("joint"))
        reached = set(roots)
        while pending:
            ready = [j for j in pending if j.find("parent").get("link") in reached]
            if not ready:
                raise ValueError("Disconnected URDF")
            for joint in ready:
                parent, child = (
                    joint.find("parent").get("link"),
                    joint.find("child").get("link"),
                )
                origin = joint.find("origin")
                xyz = (
                    [float(x) for x in origin.get("xyz", "0 0 0").split()]
                    if origin is not None
                    else [0.0, 0.0, 0.0]
                )
                rpy = (
                    [float(x) for x in origin.get("rpy", "0 0 0").split()]
                    if origin is not None
                    else [0.0, 0.0, 0.0]
                )
                r, p, y = [v / 2 for v in rpy]
                sr, cr, sp, cp, sy, cy = (
                    math.sin(r),
                    math.cos(r),
                    math.sin(p),
                    math.cos(p),
                    math.sin(y),
                    math.cos(y),
                )
                quat = [
                    sr * cp * cy - cr * sp * sy,
                    cr * sp * cy + sr * cp * sy,
                    cr * cp * sy - sr * sp * cy,
                    cr * cp * cy + sr * sp * sy,
                ]
                axis_xml = joint.find("axis")
                axis = (
                    [float(x) for x in axis_xml.get("xyz", "1 0 0").split()]
                    if axis_xml is not None
                    else [1.0, 0.0, 0.0]
                )
                typ = joint.get("type")
                if typ not in ("fixed", "revolute", "continuous"):
                    raise ValueError(f"Unsupported joint: {joint.get('name')} ({typ})")
                dof = (
                    None if typ == "fixed" else self.dof_names.index(joint.get("name"))
                )
                tensors = [
                    torch.tensor(v, dtype=torch.float32, device=device)
                    for v in (xyz, quat, axis)
                ]
                self.joints.append(
                    (
                        self.body_names.index(parent),
                        self.body_names.index(child),
                        dof,
                        *tensors,
                    )
                )
                reached.add(child)
                pending.remove(joint)
        self.com_offsets = torch.zeros(len(body_names), 3, device=device)
        for link in tree.findall("link"):
            origin = link.find("inertial/origin")
            if origin is not None:
                self.com_offsets[self.body_names.index(link.get("name"))] = (
                    torch.tensor(
                        [float(x) for x in origin.get("xyz", "0 0 0").split()],
                        device=device,
                    )
                )

    def forward(self, root, q, qd, com_velocities=True, root_velocity_is_com=True):
        n = len(root)
        pos = [None] * len(self.body_names)
        rot = pos.copy()
        vel = pos.copy()
        ang = pos.copy()
        pos[0], rot[0], vel[0], ang[0] = (
            root[:, :3],
            root[:, 3:7],
            root[:, 7:10],
            root[:, 10:13],
        )
        if root_velocity_is_com:
            offset = rotate(rot[0], self.com_offsets[0].expand(n, -1))
            vel[0] = vel[0] - torch.cross(ang[0], offset, dim=-1)
        for parent, child, dof, xyz, origin_rot, axis in self.joints:
            offset = rotate(rot[parent], xyz.expand(n, -1))
            pos[child] = pos[parent] + offset
            pre_rot = multiply(rot[parent], origin_rot.expand(n, -1))
            vel[child] = vel[parent] + torch.cross(ang[parent], offset, dim=-1)
            if dof is None:
                rot[child], ang[child] = pre_rot, ang[parent]
            else:
                half = q[:, dof : dof + 1] / 2
                turn = torch.cat((axis.expand(n, -1) * half.sin(), half.cos()), -1)
                rot[child] = multiply(pre_rot, turn)
                ang[child] = (
                    ang[parent]
                    + rotate(pre_rot, axis.expand(n, -1)) * qd[:, dof : dof + 1]
                )
        pos, rot, vel, ang = [torch.stack(values, 1) for values in (pos, rot, vel, ang)]
        if com_velocities:
            vel = vel + torch.cross(
                ang, rotate(rot, self.com_offsets[None].expand(n, -1, -1)), dim=-1
            )
        return {"body_pos": pos, "body_rot": rot, "body_vel": vel, "body_ang_vel": ang}
