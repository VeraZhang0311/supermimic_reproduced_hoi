"""Sufficient hw3 actuator state at 50 Hz command boundaries (pure torch).

The policy holds one position target for four 5 ms motor ticks. A shared
delay of 2--4 ticks therefore reads only the previous held target and the
new command during the next control interval. At a valid boundary, the
last four entries of G1Motor's five-slot ring are equal. Its next-write
slot is overwritten before any delayed read and its global ring phase has
no physical effect. This is why the full FIFO need not enter the WM.

The compact state also carries the episode delay/torque multipliers and
the startup joint-friction multipliers. It fixes actuator hidden state;
it does not claim that solver warm starts or arbitrary future actuators
are represented. Changing decimation, max delay, within-interval target
holding, or the motor model requires a new state schema.
"""
from __future__ import annotations

from functools import lru_cache
import re

import torch

from .g1_physics import CONTROL_DECIMATION, DEFAULT_JOINT_POS, action_scale
from .robot_metadata import G1_JOINT_NAMES


ACTUATOR_DIM = 88
ACTUATOR_STATE_DIM = ACTUATOR_DIM
PREVIOUS_TARGET = slice(0, 29)
DELAY = slice(29, 30)
TORQUE_SCALE = slice(30, 59)
FRICTION_SCALE = slice(59, 88)
_DEFAULT_Q = tuple(next((value for pattern, value in DEFAULT_JOINT_POS.items()
                        if re.fullmatch(pattern, name)), 0.) for name in G1_JOINT_NAMES)
_ACTION_SCALE = tuple(action_scale(name) for name in G1_JOINT_NAMES)
_FRICTION_HIGH = tuple(2.5 if i < 12 else 2. for i in range(29))

ACTUATOR_SCHEMA = {
    "version": "fengqiao_hw3_control_boundary_actuator_v1",
    "dim": ACTUATOR_STATE_DIM,
    "joint_names": list(G1_JOINT_NAMES),
    "fields": [
        {"name": "previous_commanded_joint_target", "start": 0, "end": 29, "units": "rad"},
        {"name": "shared_command_delay", "start": 29, "end": 30, "units": "5ms physics ticks; exact integer 2..4"},
        {"name": "joint_torque_scale", "start": 30, "end": 59, "units": "multiplier"},
        {"name": "joint_friction_scale", "start": 59, "end": 88, "units": "multiplier"},
    ],
    "control_decimation": 4,
    "delay_range_ticks": [2, 4],
    "fifo_length": 5,
    "boundary_invariant": "slots (next_write_tick - 1..4) mod 5 are exactly the same held command; reset fills all slots with q",
    "discarded_fifo_state": "next-write slot is overwritten before delayed read; global ring phase is immaterial",
    "advance": "default_joint_position + raw_xi[1:30] + action_scale * action; the other 59 parameters are retained",
    "normalization": {
        "previous_target": "(target-default_joint_position)/action_scale",
        "delay": "ticks-3",
        "torque_scale": "(multiplier-1)/0.1",
        "friction_scale": "(multiplier-midpoint)/half_width of startup range; legs .8..2.5, other .8..2.0",
        "clipping": False,
    },
    "default_joint_position": list(_DEFAULT_Q),
    "action_scale": list(_ACTION_SCALE),
}


def _shape_context(context):
    if not isinstance(context, torch.Tensor) or context.ndim < 1 or context.shape[-1] != ACTUATOR_STATE_DIM:
        raise ValueError("Actuator context must have trailing dimension 88")
    if not context.is_floating_point():
        raise TypeError("Actuator context stores raw floating-point targets/parameters")


@lru_cache(maxsize=16)
def _constants(device, dtype):
    # Cached, non-trainable vectors avoid repeated host->GPU copies during a
    # 32-step imagination rollout. Callers never mutate these tensors.
    vectors = (_DEFAULT_Q, _ACTION_SCALE,
               tuple((.8 + hi)/2 for hi in _FRICTION_HIGH),
               tuple((hi - .8)/2 for hi in _FRICTION_HIGH))
    return tuple(torch.tensor(v, device=device, dtype=dtype) for v in vectors)


def validate_actuator_state(context):
    """Check raw replay/native state, outside the differentiable WM hot path."""
    _shape_context(context)
    if not bool(torch.isfinite(context).all()):
        raise ValueError("Nonfinite raw actuator state")
    delay = context[..., DELAY]
    if not bool(((delay >= 2) & (delay <= 4) & (delay == delay.round())).all()):
        raise ValueError("Actuator delay must be an exact 2..4 physics-tick integer")
    torque = context[..., TORQUE_SCALE]
    if not bool(((torque >= .9-1e-6) & (torque <= 1.1+1e-6)).all()):
        raise ValueError("Actuator torque multipliers are outside the hw3 reset range")
    friction = context[..., FRICTION_SCALE]
    high = context.new_tensor(_FRICTION_HIGH)
    if not bool(((friction >= .8-1e-6) & (friction <= high+1e-6)).all()):
        raise ValueError("Actuator joint-friction multipliers are outside the hw3 startup range")
    return context


def export_motor_state(motor):
    """Snapshot real G1Motor state; reject a nonuniform partial-step FIFO.

    The returned tensor owns its storage: a later native reset cannot change
    replay records retroactively. The unused fifth slot is intentionally
    neither serialized nor required to equal the preceding command.
    """
    if CONTROL_DECIMATION != 4:
        raise RuntimeError("The compact hw3 actuator schema requires four motor ticks per held command")
    queue = motor.queue
    if queue.shape != (5, motor.N, 29) or not queue.is_floating_point():
        raise ValueError("Unexpected G1Motor FIFO layout")
    if not isinstance(motor.tick, int) or not 0 <= motor.tick < 5:
        raise ValueError("Invalid G1Motor next-write FIFO phase")
    if motor.delay.shape != (motor.N,) or motor.scale.shape != (motor.N, 29) or motor.friction_scale.shape != (motor.N, 29):
        raise ValueError("Unexpected G1Motor delay/parameter layout")
    slots = [(motor.tick-i) % 5 for i in range(1, 5)]
    held = queue[slots[0]]
    if any(not torch.equal(queue[slot], held) for slot in slots[1:]):
        raise ValueError("Actuator state is not at a held-command control boundary: last four FIFO slots differ")
    state = torch.cat((held, motor.delay.to(dtype=queue.dtype).unsqueeze(-1), motor.scale, motor.friction_scale), -1)
    return validate_actuator_state(state)


def advance_actuator_state(context, action, xi):
    """Exactly advance the compact actuator state after four held-command ticks.

    ``xi`` is the raw 33D robot context, not normalize_xi(xi). Arbitrary batch
    prefixes are supported for replay sequences and imagination. Shape/type
    checks do not synchronize the device; validate exported/replayed inputs
    separately. No detach, clipping, rounding or learned actuator prediction.
    """
    _shape_context(context)
    if action.shape != (*context.shape[:-1], 29) or xi.shape != (*context.shape[:-1], 33):
        raise ValueError("Actuator advance needs matching (...,88), (...,29), raw (...,33) shapes")
    if not action.is_floating_point() or not xi.is_floating_point():
        raise TypeError("Action and raw robot context must be floating-point tensors")
    if action.device != context.device or xi.device != context.device:
        raise ValueError("Actuator state, action and raw robot context must share a device")
    default, alpha, _, _ = _constants(context.device, context.dtype)
    target = default + xi[..., 1:30] + alpha * action
    return torch.cat((target, context[..., 29:]), -1)


def normalize_actuator_state(context):
    """Fixed-scale WM/privileged input transform; gradients remain intact."""
    _shape_context(context)
    default, alpha, friction_mid, friction_half = _constants(context.device, context.dtype)
    return torch.cat(((context[..., PREVIOUS_TARGET]-default)/alpha,
                      context[..., DELAY]-3.,
                      (context[..., TORQUE_SCALE]-1.)/.1,
                      (context[..., FRICTION_SCALE]-friction_mid)/friction_half), -1)
