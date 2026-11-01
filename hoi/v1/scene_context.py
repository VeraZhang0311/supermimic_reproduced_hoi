"""Observed stationary support context shared by real and imagined rollouts.

Raw support world poses are frozen at the sampled episode reset. Only their
coordinates relative to the current (possibly imagined) robot are recomputed.
"""
from __future__ import annotations
import torch
from supermimic.rotations import yaw_mat, quat_to_mat, mat_to_6d
from .scene_supports import SUPPORT_IDS, omni_support_recipe

SCENE_CONTEXT_DIM = 26
SCENE_CONTEXT_SCHEMA = {
    "version":"hoi_scene_context_two_cuboids_v1", "dim":SCENE_CONTEXT_DIM,
    "slots":list(SUPPORT_IDS), "per_slot":["present1","heading_relative_position3","heading_relative_rotation6d6","size_m3"],
    "raw_pose":"world_position_xyz_quaternion_xyzw_relative_to_environment_origin",
    "time":"sampled_reset_frame_held_for_episode; imagination_freezes_world_pose",
    "absent":"all_13_values_zero_before_and_after_normalization",
    "kinematic_velocity":"zero; stationary author supports only", "material":"bound_fixed_Omni_friction_1_restitution_0"}
SCENE_FIELDS = ("support_pose_xyzw", "support_present")


def empty_scene_state(batch_shape, *, device=None, dtype=torch.float32):
    pose=torch.zeros(tuple(batch_shape)+(len(SUPPORT_IDS),7),device=device,dtype=dtype)
    pose[...,6]=1
    return {"support_pose_xyzw":pose,"support_present":torch.zeros(tuple(batch_shape)+(len(SUPPORT_IDS),),device=device,dtype=torch.bool)}


def validate_scene_state(pose, present, batch_shape=None):
    if not isinstance(pose,torch.Tensor) or not isinstance(present,torch.Tensor) or pose.shape[-2:] != (2,7) or present.shape != pose.shape[:-1]:
        raise ValueError("scene state requires world support_pose_xyzw[...,2,7] and support_present[...,2]")
    if batch_shape is not None and pose.shape[:-2] != tuple(batch_shape):
        raise ValueError("scene state batch does not align with robot state")
    if not bool(torch.isfinite(pose).all()) or not bool(((present==0)|(present==1)).all()):
        raise ValueError("scene poses must be finite and presence binary")
    qnorm=pose[...,3:].norm(dim=-1)
    if bool((present.bool() & ((qnorm-1).abs()>1e-3)).any()):
        raise ValueError("present support orientation must be unit xyzw")


def scene_context(S, pose, present):
    validate_scene_state(pose,present,S[0].shape[:-1])
    present=present.detach().to(device=S[0].device,dtype=torch.bool)
    pose=pose.detach().to(S[0])
    # Absent fields cannot introduce NaNs through quaternion conversion or
    # spurious gradients, even when a caller uses zero placeholder quaternions.
    identity=torch.zeros_like(pose);identity[...,6]=1
    pose=torch.where(present[...,None],pose,identity)
    Ht=yaw_mat(S[1]).transpose(-1,-2).unsqueeze(-3)
    position=(Ht@(pose[...,:3]-S[0].unsqueeze(-2)).unsqueeze(-1)).squeeze(-1)
    R=quat_to_mat(pose[...,[6,3,4,5]])
    rotation=mat_to_6d(Ht@R)
    size=pose.new_tensor(omni_support_recipe()["size_m"]).expand_as(position)
    values=torch.cat((present.to(pose.dtype)[...,None],position,rotation,size),-1)
    return torch.where(present[...,None],values,torch.zeros_like(values)).flatten(-2)


def scene_from_record(S, record):
    if any(key not in record for key in SCENE_FIELDS):
        raise ValueError("observed scene support state is required; old scene-less inputs are incompatible")
    return scene_context(S,record[SCENE_FIELDS[0]],record[SCENE_FIELDS[1]])


def normalize_scene_context(normalizer, context):
    if context.shape[-1] != SCENE_CONTEXT_DIM or not bool(torch.isfinite(context).all()):
        raise ValueError("scene context must be finite 26D")
    raw=context.reshape(context.shape[:-1]+(2,13));mask=raw[...,0:1]
    if not bool(((mask==0)|(mask==1)).all()) or bool(((mask==0)&(raw!=0)).any()):
        raise ValueError("scene context absent slots must be exactly zero with binary presence")
    normalized=normalizer(context).reshape_as(raw)
    normalized=torch.cat((mask,normalized[...,1:]),-1)
    return torch.where(mask.bool(),normalized,torch.zeros_like(normalized)).flatten(-2)
