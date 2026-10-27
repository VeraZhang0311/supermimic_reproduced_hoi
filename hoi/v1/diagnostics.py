"""Read-only HOI evaluation diagnostics; none supplies training targets.

Source body net-contact flags are only compared as annotations. Paired reward
checks use saved simulator initial states and executed actions, never a second
policy rollout or reference transitions as dynamics truth.
"""
from __future__ import annotations

import torch

from supermimic.mdp import normalize_xi
from .contacts import CHANNELS, LABEL_SCHEMA
from .state import unflatten_object, unflatten_state


def validate_actual_contact_fields(control, rows, fields=("contact", "contact_valid", "contact_impulse_Ns")):
    """Require the current actual-pair contract, independent of hand goals."""
    binary = {"contact", "actual_contact", "contact_valid"}
    for name in fields:
        value = control.get(name)
        if not torch.is_tensor(value) or value.shape != (rows, len(CHANNELS)):
            raise ValueError(f"actual contact field {name} requires {len(CHANNELS)} channels in the {LABEL_SCHEMA} order")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"actual contact field {name} must remain finite")
        if name in binary and not bool(((value == 0) | (value == 1)).all()):
            raise ValueError(f"actual contact field {name} must remain binary")
        if name not in binary and bool((value < 0).any()):
            raise ValueError(f"actual contact field {name} must remain nonnegative")


def source_flag_comparison(source_reference, expected, valid, body_names=None):
    """Compare endpoint f+1 flags at explicitly named original rubber hands."""
    if (source_reference.ndim != 2 or source_reference.shape[-1] != 630 or
            expected.shape != (len(source_reference), 2) or valid.shape != expected.shape):
        raise ValueError("source flag comparison requires aligned endpoint630 and expected/valid LR")
    common = {"source_is_pair_truth": False, "affects_reward_or_gate": False,
              "timing": "source nearest-sampled endpoint f+1 versus geometric expected(f+1); not interval-force truth"}
    names = ("left_rubber_hand", "right_rubber_hand")
    if body_names is None or len(body_names) != 39 or any(body_names.count(name) != 1 for name in names):
        return {**common, "available": False, "reason": "original39 rubber-hand body-name mapping unavailable"}
    indices = [body_names.index(name) for name in names]
    source = source_reference[:, 591:630][:, indices].detach().cpu()
    if not bool(((source == 0) | (source == 1)).all()):
        raise ValueError("source body contact flags must remain binary annotations")
    expected, valid = expected.detach().cpu().bool(), valid.detach().cpu().bool()
    source = source.bool()
    result = {**common, "available": True, "source_body_names": list(names), "source_body_indices": indices,
              "interval_count": len(source), "source_positive_fraction": source.float().mean(0).tolist()}
    for name, selection in (("valid_intervals", valid), ("mismatch_intervals", (source != expected) & valid),
                            ("source_only_positive_intervals", source & ~expected & valid),
                            ("expected_only_positive_intervals", ~source & expected & valid),
                            ("both_positive_intervals", source & expected & valid)):
        result[name] = selection.sum(0).tolist()
    result["mismatch_fraction_valid"] = [int(((source[:, i] != expected[:, i]) & valid[:, i]).sum()) / int(valid[:, i].sum())
                                         if bool(valid[:, i].any()) else None for i in range(2)]
    return result


def contact_transition_events(labels, valid, channels, dt=.02):
    """Transitions between adjacent known intervals; unknown intervals break them."""
    if labels.ndim != 2 or valid.shape != labels.shape or labels.shape[1] != len(channels) or dt <= 0:
        raise ValueError("contact event labels/validity/channel dimensions or dt differ")
    labels, valid = labels.detach().cpu() > .5, valid.detach().cpu().bool()
    result = {"timing": "interval(t,t+1] label; event at latter interval endpoint; aggregation uncertainty <=1 control interval",
              "first_positive_interval_is_not_an_onset": True, "unknown_intervals_break_transitions": True, "channels": {}}
    for index, name in enumerate(channels):
        connected = valid[1:, index] & valid[:-1, index]
        onset = connected & labels[1:, index] & ~labels[:-1, index]
        release = connected & ~labels[1:, index] & labels[:-1, index]
        row = {"valid_adjacent_intervals": int(connected.sum()),
               "initial_observed_contact": bool(labels[0, index]) if len(labels) and valid[0, index] else None}
        for event, mask in (("onset", onset), ("release", release)):
            steps = (mask.nonzero().flatten() + 2).tolist()
            row[event + "_count"] = len(steps)
            row[event + "_endpoint_steps"] = steps
            row[event + "_endpoint_times_s"] = [step * dt for step in steps]
        result["channels"][name] = row
    return result


@torch.no_grad()
def executed_action_reward_gap(wm, mdp, states, control, frames, points, *, horizons=(1, 8, 32), batch_size=128):
    """Paired real/imagined rewards from identical real starts and real actions.

    Horizon-1 is teacher forced at every real start. Longer horizons use
    nonoverlapping complete windows and autoregressive WM states. The real
    trajectory is never reset by failure; neither comparison applies imagined
    survival weighting. Batching bounds additional memory independently of clip
    duration. No optimizer, parameter or normalizer update occurs.
    """
    T = len(control["action"])
    if (T < 1 or states["s"].shape != (T + 1, 71) or states["o"].shape != (T + 1, 13) or
            len(frames) != T + 1 or type(batch_size) is not int or batch_size < 1 or
            any(type(h) is not int or h < 1 for h in horizons)):
        raise ValueError("paired reward diagnostic needs T controls, T+1 states/frames and positive integer windows")
    for key, dim in (("action", 29), ("a_prev", 29), ("xi", 33), ("xi_o", 12)):
        if control[key].shape != (T, dim):
            raise ValueError(f"paired reward diagnostic {key} shape differs")
    validate_actual_contact_fields(control, T, ("contact_valid",))
    from .robot_context import mass_context_from
    mass_context = mass_context_from(control) if getattr(wm, "torso_mass_context", False) else None
    if mass_context is not None:
        if mass_context.shape != (T, 1):
            raise ValueError("paired reward diagnostic torso mass context shape differs")
        if "torso_mass_context" in control and not torch.equal(control["torso_mass_context"], mass_context):
            raise ValueError("saved torso mass context differs from raw kg fields")
        if not bool((mass_context == mass_context[:1]).all()):
            raise ValueError("paired reward diagnostic torso mass must be constant within a clip")
    if control["reward"].shape != (T,) or not bool(torch.isfinite(control["reward"]).all()):
        raise ValueError("paired reward diagnostic requires finite saved simulator rewards")
    device = next(wm.parameters()).device
    frames = torch.as_tensor(frames, device=device, dtype=torch.long)
    if not bool((frames[1:] == frames[:-1] + 1).all()):
        raise ValueError("paired reward windows must belong to one full-duration contiguous clip")
    points = torch.as_tensor(points, device=device)
    if points.shape != (128, 3):
        raise ValueError("paired reward diagnostic requires this clip's exact asset pointcloud")
    versions = [(value, value._version) for value in wm.state_dict().values()]
    training = wm.training
    wm.eval()
    reports = {}
    try:
        for horizon in horizons:
            starts = torch.arange(0, max(0, T - horizon + 1), horizon, device=device)
            predicted_sum = actual_sum = signed_sum = absolute_sum = square_sum = window_abs_sum = 0.
            intervals = 0
            for chunk in starts.split(batch_size):
                cpu_chunk = chunk.cpu()
                S = unflatten_state(states["s"][cpu_chunk].to(device))
                O = unflatten_object(states["o"][cpu_chunk].to(device))
                cloud = points[None].expand(len(chunk), -1, -1)
                predicted_return = torch.zeros(len(chunk), device=device)
                actual_return = torch.zeros_like(predicted_return)
                for offset in range(horizon):
                    indices, cpu_indices = chunk + offset, cpu_chunk + offset
                    action = control["action"][cpu_indices].to(device)
                    mass_kwargs = {} if mass_context is None else {"mass_context": mass_context[cpu_indices].to(device)}
                    S, O, logits = wm.step(S, O, action, normalize_xi(control["xi"][cpu_indices].to(device)),
                                             control["xi_o"][cpu_indices].to(device), cloud, **mass_kwargs)
                    if logits.shape != (len(chunk), len(CHANNELS)):
                        raise ValueError("paired reward diagnostic WM contact logits differ from the five-channel contract")
                    if not bool(torch.isfinite(logits).all()):
                        raise FloatingPointError("nonfinite paired WM contact logits")
                    predicted = mdp.reward(S, O, frames[indices + 1], action, control["a_prev"][cpu_indices].to(device),
                                           torch.sigmoid(logits), contact_valid=control["contact_valid"][cpu_indices].to(device))
                    actual = control["reward"][cpu_indices].to(device)
                    if not bool(torch.isfinite(predicted).all()):
                        raise FloatingPointError("nonfinite paired imagined reward")
                    difference = predicted - actual
                    predicted_sum += float(predicted.sum())
                    actual_sum += float(actual.sum())
                    signed_sum += float(difference.sum())
                    absolute_sum += float(difference.abs().sum())
                    square_sum += float(difference.square().sum())
                    predicted_return += predicted
                    actual_return += actual
                    intervals += len(chunk)
                window_abs_sum += float((predicted_return - actual_return).abs().sum())
            reports[str(horizon)] = {"available": bool(intervals), "sequence_count": len(starts), "valid_intervals": intervals,
                                     "tail_intervals_not_scored": T - intervals,
                                     "real_reward_mean": actual_sum / intervals if intervals else None,
                                     "imagined_reward_mean": predicted_sum / intervals if intervals else None,
                                     "signed_reward_gap_mean": signed_sum / intervals if intervals else None,
                                     "absolute_reward_gap_mean": absolute_sum / intervals if intervals else None,
                                     "reward_gap_rmse": (square_sum / intervals) ** .5 if intervals else None,
                                     "absolute_window_return_gap_mean": window_abs_sum / len(starts) if len(starts) else None,
                                     "mode": "teacher_forced_real_starts" if horizon == 1 else "autoregressive_nonoverlapping_complete_windows"}
        if any(value._version != version for value, version in versions):
            raise RuntimeError("frozen reward diagnostic modified WM parameters or normalization statistics")
    finally:
        wm.train(training)
    return {"horizons": reports, "real_executed_actions": True, "identical_real_initial_state_per_window": True,
            "contact_channels": list(CHANNELS), "contact_label_schema": LABEL_SCHEMA,
            "predicted_contact_source": "sigmoid_frozen_WM_pair_logits; actual validity mask retained",
            "reward_timing": "predicted/real state after action[t] scored at reference(f[t]+1)",
            "survival_weighting": False, "optimizer_updates": 0, "normalization_updates": 0,
            "used_as_training_data_or_gate": False, "extra_batch_size_limit": batch_size}
