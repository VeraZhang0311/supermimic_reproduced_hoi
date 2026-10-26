"""Versioned DR choices and CPU-testable sampling; no simulator on import."""
from __future__ import annotations

import torch

LEGACY_DR = "v1"
MATCHING_DR = "v1_1_mass_object"
DR_PROFILES = (LEGACY_DR, MATCHING_DR)


def randomization_enabled(profile, eval_mode):
    randomization_contract(profile)
    return profile != LEGACY_DR and not eval_mode


def validate_dr_evidence(path, profile):
    """Old nominal evidence never certifies new property-changing DR."""
    import json
    from pathlib import Path
    randomization_contract(profile)
    if profile == LEGACY_DR:
        return {"passed": True, "profile": profile, "scope": "existing_native_contract"}
    raw = json.loads(Path(path).read_text(encoding="utf-8")) if path and Path(path).is_file() else {}
    evidence = raw.get("domain_randomization", {})
    required = ("native_property_readback_passed", "subset_reset_passed", "nominal_eval_passed")
    passed = evidence.get("contract") == randomization_contract(profile) and all(evidence.get(k) is True for k in required)
    return {"passed": passed, "profile": profile, "required_checks": list(required),
            "scope": "independent_Linux_GPU_native_evidence;CPU_checks_do_not_satisfy_this_gate"}


def randomization_contract(profile):
    if profile not in DR_PROFILES:
        raise ValueError(f"unknown domain randomization profile: {profile}")
    result = {"profile": profile, "robot_friction": [.3, 1.2],
              "torso_com_halfwidth_m": [.025, .05, .05], "joint_zero_halfwidth_rad": .01,
              "robot_friction_com_resample": "environment_initialization",
              "joint_zero_resample": "episode_reset", "robot_reset_noise": "unchanged",
              "evaluation": "nominal_no_randomization"}
    if profile == MATCHING_DR:
        from .robot_context import torso_mass_contract
        result.update(torso_mass_delta_kg=[-3., 3.], torso_mass_resample="environment_initialization",
                      object_mass_delta_kg=[-1.5, 1.5], object_mass_lower_bound="max(nominal-1.5,0.1*nominal)",
                      object_com_halfwidth_m=.01, object_friction=[.4, .8],
                      object_resample="episode_reset", inertia="nominal_COM_tensor_times_mass_ratio",
                      independent_inertia_randomization=False, ground_friction="bound_asset_profile",
                      PD_randomization=False, gravity_randomization=False, action_delay=False, external_push=False,
                      robot_mass_in_WM_context=True, robot_mass_in_critic_context=True,
                      robot_mass_context=torso_mass_contract(),
                      native_DR_validation="required_before_training")
    return result


def _uniform_like(reference, lower, upper, generator=None):
    return lower + (upper - lower) * torch.rand(reference.shape, dtype=reference.dtype,
                                               device=reference.device, generator=generator)


def sample_mass(nominal, halfwidth, generator=None):
    """Truncate the *sampling support*, never draw negative masses then clamp."""
    if not bool(torch.isfinite(nominal).all()) or bool((nominal <= 0).any()):
        raise ValueError("nominal mass must be finite and positive")
    lower = torch.maximum(nominal - halfwidth, nominal * .1)
    return _uniform_like(nominal, lower, nominal + halfwidth, generator)


def sample_object_properties(mass, com, inertia, *, generator=None):
    """All values use the nominal object link/COM axes, not previous DR values."""
    sampled_mass = sample_mass(mass, 1.5, generator)
    sampled_com = com + _uniform_like(com, -.01, .01, generator)
    sampled_inertia = inertia * (sampled_mass / mass)[..., None, None]
    friction = _uniform_like(mass, .4, .8, generator)
    return sampled_mass, sampled_com, sampled_inertia, friction


class NativeRandomization:
    """Apply sampled properties with readback; full arrays preserve other envs.

    This adapter is unverified on native GPUs until a new DR-specific native
    gate has run. CPU mocks establish setter wiring, not physical correctness.
    """
    def __init__(self, host):
        self.host = host
        env, obj = host.env, host.object
        self.robot_mass = env.robot.data.body_mass.torch.clone()
        self.robot_inertia = env.robot.data.body_inertia.torch.clone()
        self.nominal_robot_mass = self.robot_mass.clone()
        self.nominal_robot_inertia = self.robot_inertia.clone()
        self.nominal_object_mass = obj.data.body_mass.torch[:, 0].clone()
        self.nominal_object_com = obj.data.body_com_pose_b.torch[:, 0, :3].clone()
        self.nominal_object_inertia = obj.data.body_inertia.torch.reshape(host.N, 1, 3, 3)[:, 0].clone()
        self.mass = self.nominal_object_mass.clone()
        self.com = self.nominal_object_com.clone()
        self.inertia = self.nominal_object_inertia.clone()
        self.friction = host.xi_o[:, 10].clone()
        torso = env.torso_id
        nominal = self.robot_mass[:, torso].clone()
        if bool((nominal <= 3.).any()):
            raise ValueError("torso nominal mass must exceed 3 kg for symmetric +/-3 kg DR")
        sampled = sample_mass(nominal, 3.)
        self.robot_mass[:, torso] = sampled
        view = self.robot_inertia.reshape(host.N, -1, 9)
        view[:, torso] *= (sampled / nominal)[:, None]
        env.robot.set_masses_index(masses=self.robot_mass)
        env.robot.set_inertias_index(inertias=self.robot_inertia)
        self._changed(body=True)
        self._close(env.robot.data.body_mass.torch, self.robot_mass, "robot mass")
        self._close(env.robot.data.body_inertia.torch, self.robot_inertia, "robot inertia")

    @staticmethod
    def _close(actual, expected, name):
        if not bool(torch.isfinite(actual).all()) or not torch.allclose(actual, expected, atol=1e-6, rtol=1e-5):
            raise RuntimeError(f"DR native {name} readback differs from sampled property")

    def _changed(self, *, body=False):
        from newton import ModelFlags
        flags = ModelFlags.SHAPE_PROPERTIES
        if body:
            flags |= ModelFlags.BODY_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES
        self.host.env.sim.backend_manager.add_model_change(flags)

    def reset(self, ids):
        if not len(ids):
            return
        import warp as wp
        from .native import physical_context
        host, obj = self.host, self.host.object
        values = sample_object_properties(self.nominal_object_mass[ids], self.nominal_object_com[ids],
                                          self.nominal_object_inertia[ids])
        for target, value in zip((self.mass, self.com, self.inertia, self.friction), values):
            target[ids] = value
        # The already-supported all-environment setters avoid guessing a runtime
        # subset-index ABI. Untouched environments retain their previous values.
        obj.set_masses_index(masses=self.mass[:, None].contiguous())
        obj.set_coms_index(coms=self.com[:, None].contiguous())
        obj.set_inertias_index(inertias=self.inertia.reshape(host.N, 1, 9).contiguous())
        model = host.env.sim.backend_manager.get_model()
        material = wp.to_torch(obj._root_view.get_attribute("shape_material_mu", model))
        if not material.is_contiguous():
            raise RuntimeError("DR requires a contiguous writable native object-friction view")
        material.reshape(host.N, -1)[:] = self.friction[:, None]
        self._changed(body=True)
        actual_mass = obj.data.body_mass.torch[:, 0]
        actual_com = obj.data.body_com_pose_b.torch[:, 0, :3]
        actual_inertia = obj.data.body_inertia.torch.reshape(host.N, 1, 3, 3)[:, 0]
        actual_friction = material.reshape(host.N, -1)
        for actual, expected, name in ((actual_mass, self.mass, "object mass"),
                                       (actual_com, self.com, "object COM"),
                                       (actual_inertia, self.inertia, "object inertia"),
                                       (actual_friction, self.friction[:, None].expand_as(actual_friction), "object friction")):
            self._close(actual, expected, name)
        host.xi_o = physical_context(actual_mass, actual_com, actual_inertia, actual_friction[:, 0], host.xi_o[:, 11])
