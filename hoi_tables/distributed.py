"""Synchronous two-asset learning, including real-data RMS moments."""
from __future__ import annotations

from types import MethodType
import torch
import torch.distributed as dist
from supermimic.networks import RunningNorm


def _sum(value):
    # Gloo CPU buffers support two workers sharing one scheduler-visible GPU.
    buffer = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    dist.all_reduce(buffer)
    return buffer.to(device=value.device, dtype=value.dtype)


def synchronize_gradients(module):
    world = dist.get_world_size()
    for parameter in module.parameters():
        if parameter.grad is not None:
            parameter.grad.copy_(_sum(parameter.grad) / world)


def synchronize_contact_counts(positive, negative):
    return _sum(positive), _sum(negative)


@torch.no_grad()
def _distributed_norm_update(self, values):
    x = values.reshape(-1, values.shape[-1]).double()
    summary = torch.cat((x.sum(0), x.square().sum(0), x.new_tensor([len(x)])))
    global_summary = _sum(summary)
    count = global_summary[-1]
    if count <= 0 or self.count >= self.max_count:
        return
    d = self.mean.numel()
    mean = global_summary[:d] / count
    var = (global_summary[d:2*d] / count - mean.square()).clamp_min(0)
    total = self.count.double() + count
    delta = mean - self.mean.double()
    variance = (self.var.double() * self.count + var * count + delta.square() * self.count * count / total) / total
    self.mean.copy_((self.mean.double() + delta * count / total).to(self.mean))
    self.var.copy_(variance.to(self.var))
    self.count.copy_(total.to(self.count))


def bind_distributed_statistics(*modules):
    for module in modules:
        for child in module.modules():
            if isinstance(child, RunningNorm):
                child.update = MethodType(_distributed_norm_update, child)


def broadcast_models(*modules):
    for module in modules:
        for value in module.state_dict().values():
            cpu = value.detach().cpu().contiguous()
            dist.broadcast(cpu, src=0)
            value.copy_(cpu.to(value))


def state_hash(*modules):
    import hashlib
    result = hashlib.sha256()
    for module in modules:
        for name, value in module.state_dict().items():
            result.update(name.encode())
            result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()
