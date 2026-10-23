"""Time-based sampling of ULTRA's released 60 FPS [T,630] motions."""

import torch

from .ultra_math import slerp


class UltraReference:
    def __init__(self, frames, fps=60.0):
        if frames.ndim != 2 or frames.shape[1] != 630 or frames.shape[0] < 2:
            raise ValueError(f"Expected [T,630] motion, got {frames.shape}")
        if not torch.isfinite(frames).all():
            raise ValueError("Non-finite motion")
        self.frames = frames.float()
        self.fps = float(fps)
        self.duration = (len(frames) - 1) / self.fps

    def sample(self, times):
        times = torch.as_tensor(
            times, device=self.frames.device, dtype=self.frames.dtype
        ).reshape(-1)
        index = (times * self.fps).clamp(0, len(self.frames) - 1)
        lo = index.floor().long()
        hi = (lo + 1).clamp_max(len(self.frames) - 1)
        alpha = (index - lo)[:, None]
        a, b = self.frames[lo], self.frames[hi]
        result = a * (1 - alpha) + b * alpha
        for start, stop in [(3, 7), (74, 78), (201, 357)]:
            qa = a[:, start:stop].reshape(len(times), -1, 4)
            qb = b[:, start:stop].reshape(len(times), -1, 4)
            result[:, start:stop] = slerp(qa, qb, alpha[:, None, :]).flatten(1)
        result[:, 591:630] = self.frames[index.round().long(), 591:630]
        return result
