"""Per-asset Torch random streams for serial solvers sharing one process.

The process worker's seed rule stays unchanged. Only the owning CPU and one
CUDA stream are swapped; caller/learner streams and other GPUs are preserved.
Use around construction, stepping (including autoreset), and explicit reset.
These process-global swaps require serial calls, not concurrent Python threads.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError("Torch seed must be an integer in [0, 2**64)")
    return value


def asset_seed(base_seed, asset_id):
    """Same stream identity as the original one-asset process worker."""
    _seed(base_seed)
    if type(asset_id) is not int or asset_id < 0:
        raise ValueError("asset_id must be a nonnegative integer")
    return _seed(base_seed + 100003 * asset_id)


class AssetRNG:
    def __init__(self, seed, device="cpu", *, cuda=None):
        self.seed = _seed(seed)
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda") or (
                self.device.type == "cuda" and self.device.index is None):
            raise ValueError("asset RNG requires cpu or an explicit cuda:N device")
        self._active = False
        self._cpu_state = torch.Generator(device="cpu").manual_seed(self.seed).get_state().clone()
        self._cuda = (torch.cuda if cuda is None else cuda) if self.device.type == "cuda" else None
        self._cuda_state = None
        if self._cuda is not None:
            # manual_seed seeds only the selected device, unlike torch.manual_seed
            # or cuda.manual_seed_all, which would alter unrelated GPU streams.
            caller = self._cuda.get_rng_state(self.device).clone()
            with self._cuda.device(self.device):
                try:
                    self._cuda.manual_seed(self.seed)
                    self._cuda_state = self._cuda.get_rng_state(self.device).clone()
                finally:
                    self._cuda.set_rng_state(caller, self.device)

    @contextmanager
    def activate(self):
        if self._active:
            raise RuntimeError("the same asset RNG cannot be activated recursively")
        caller_cpu = torch.get_rng_state().clone()
        caller_cuda = self._cuda.get_rng_state(self.device).clone() if self._cuda is not None else None
        self._active = True
        try:
            torch.set_rng_state(self._cpu_state)
            if self._cuda is not None:
                self._cuda.set_rng_state(self._cuda_state, self.device)
            try:
                yield self
            finally:
                # Preserve draws even when the solver raises. Publish both states
                # only after successful reads; caller restoration is unconditional.
                next_cpu = torch.get_rng_state().clone()
                next_cuda = self._cuda.get_rng_state(self.device).clone() if self._cuda is not None else None
                self._cpu_state, self._cuda_state = next_cpu, next_cuda
        finally:
            try:
                torch.set_rng_state(caller_cpu)
            finally:
                try:
                    if self._cuda is not None:
                        self._cuda.set_rng_state(caller_cuda, self.device)
                finally:
                    self._active = False
