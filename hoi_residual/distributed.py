"""Synchronous tensor collectives for one-GPU/multi-process residual PPO.

Gloo transports CUDA values through a private contiguous CPU buffer so multiple
asset ranks can share one CUDA device without NCCL's duplicate-device limit.
CPU/Gloo and CUDA/NCCL use their native in-place collectives directly. These
helpers preserve the caller's reduction operator; gradient averaging remains
the PPO caller's responsibility. No asynchronous Work object escapes a call.
"""
from __future__ import annotations

import torch
import torch.distributed as dist


def _requires_cpu_staging(tensor):
    return tensor.is_cuda and dist.get_backend() == "gloo"


def _collective_buffer(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Distributed collectives require a tensor")
    if _requires_cpu_staging(tensor):
        return tensor.detach().to(device="cpu", non_blocking=False, copy=True).contiguous()
    return tensor


@torch.no_grad()
def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM):
    """Synchronously reduce in place with exactly SUM/MIN/MAX/etc. semantics."""
    buffer = _collective_buffer(tensor)
    dist.all_reduce(buffer, op=op, async_op=False)
    if buffer is not tensor:
        tensor.copy_(buffer, non_blocking=False)
    return tensor


@torch.no_grad()
def broadcast_tensor(tensor, src=0):
    """Synchronously broadcast in place; retain dtype, shape and original device."""
    buffer = _collective_buffer(tensor)
    dist.broadcast(buffer, src=src, async_op=False)
    if buffer is not tensor:
        tensor.copy_(buffer, non_blocking=False)
    return tensor
