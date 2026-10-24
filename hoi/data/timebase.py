"""Versioned canonical 60 Hz source / 50 Hz control clock for HOI v1.

The source rate is an interpretation protocol, not evidence of the released
dataset's generation interval. This module does not select simulator settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class CanonicalTimebase:
    """An immutable clock; alternative source rates require a new version.

    N source *states* contain N - 1 intervals. Integer arithmetic determines
    the complete control steps, including endpoints which floating floor can
    otherwise lose. The last fractional control interval is never stretched.
    """

    @property
    def source_hz(self) -> int:
        return 60

    @property
    def control_hz(self) -> int:
        return 50

    @property
    def control_dt_s(self) -> float:
        return 1.0 / self.control_hz

    def metadata(self) -> dict:
        return {
            "timebase_schema_version": "hoi_v1_canonical60_control50_v1",
            "timebase_mode": "canonical_60hz",
            "source_fps_assumed": 60.0,
            "source_timebase_status": "protocol_assumed_not_generation_verified",
            "source_timebase_evidence": "ULTRA_DATASET_AND_TIMEBASE_HANDOFF_20261023.zh-CN.md",
            "control_hz": 50,
            "control_dt_s": self.control_dt_s,
            "reference_sampling": "continuous_time_linear_and_quaternion_slerp",
            "source_contact_sampling": "nearest_source_frame",
            "source_velocity_sampling": "linear_interpolation_preserve_source_fields",
            "robot_reference_velocity_mode": "pose_diff_on_control_grid",
            "object_reference_velocity_mode": "pose_diff_on_control_grid",
            "velocity_difference_method": "central_one_sided_boundaries_so3_world",
            "source_velocity_reference_point_status": "unverified",
            "control_time_dtype": "float64",
            "frame_count_rule": "control_steps = 5 * (source_frames - 1) // 6; state_frames = control_steps + 1",
        }

    @staticmethod
    def _validate_frames(source_frames: int) -> None:
        if isinstance(source_frames, bool) or not isinstance(source_frames, int):
            raise TypeError("source_frames must be an integer state-frame count")
        if source_frames < 1:
            raise ValueError("source_frames must be at least 1")

    def control_steps(self, source_frames: int) -> int:
        self._validate_frames(source_frames)
        return 5 * (source_frames - 1) // 6

    def state_frames(self, source_frames: int) -> int:
        return self.control_steps(source_frames) + 1

    def time_grid(self, source_frames: int, *, device=None) -> torch.Tensor:
        """Return T + 1 state times in seconds, always in float64."""
        import torch

        return torch.arange(
            self.state_frames(source_frames), device=device, dtype=torch.float64
        ) / self.control_hz

    def sample_grid(self, source_frames: int, *, device=None) -> dict[str, torch.Tensor]:
        """Return one shared robot/object clock and its exact rational indices.

        Source positions equal 60 * time = 6 * tick / 5. Quotient/remainder
        construction keeps integer endpoints exact; alpha is float64. The
        canonical grid has no half-frame ties for nearest contact sampling.
        """
        import torch

        ticks = torch.arange(self.state_frames(source_frames), device=device, dtype=torch.int64)
        numerator = ticks * 6
        i0 = torch.div(numerator, 5, rounding_mode="floor")
        i1 = (i0 + 1).clamp_max(source_frames - 1)
        alpha = numerator.remainder(5).to(torch.float64) / 5
        return {
            "time": ticks.to(torch.float64) / self.control_hz,
            "source_index": i0.to(torch.float64) + alpha,
            "i0": i0,
            "i1": i1,
            "alpha": alpha,
            "nearest": torch.div(numerator + 2, 5, rounding_mode="floor"),
        }
