"""Synchronous asset learning, including real-data RMS moments."""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
import torch
import torch.distributed as dist
from supermimic.networks import RunningNorm


def _release_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.synchronize(torch.cuda.current_device())
        torch.cuda.empty_cache()


@contextmanager
def serialized_gpu_backward(workers_per_gpu, release=None, *, ranks_per_turn=1):
    """Run full local backwards in groups on each shared GPU.

    Ranks assigned to different GPUs run the same turn concurrently. The body
    must contain no distributed collectives: synchronize gradients and step
    optimizers only after every rank leaves this context successfully. Caught
    body/release errors are propagated after all turns and a CPU Gloo vote, so
    healthy peers never proceed to an optimizer step while a rank has failed.
    This preserves the complete local batch, graph and gradient calculation.
    """
    if type(workers_per_gpu) is not int or workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be a positive integer")
    if type(ranks_per_turn) is not int or ranks_per_turn < 1 or workers_per_gpu % ranks_per_turn:
        raise ValueError("ranks_per_turn must be a positive divisor of workers_per_gpu")
    if workers_per_gpu == 1 or not dist.is_available() or not dist.is_initialized():
        yield
        return
    world, rank = dist.get_world_size(), dist.get_rank()
    if world % workers_per_gpu:
        raise ValueError("Distributed ranks must divide equally among shared GPUs")
    if dist.get_backend() != "gloo":
        raise ValueError("Grouped GPU backward requires the CPU Gloo process group")
    release = _release_cuda_cache if release is None else release
    failure = None
    # Every process returns unused segments from the preceding WM phase
    # before the first large actor graph claims the shared device's free RAM.
    try:
        release()
    except BaseException as error:
        failure = (error, error.__traceback__)
    # No rank starts the large graph while a peer is still in the prior phase.
    dist.barrier()
    for turn in range(workers_per_gpu // ranks_per_turn):
        if (rank % workers_per_gpu) // ranks_per_turn == turn:
            try:
                yield
            except BaseException as error:
                if failure is None:
                    failure = (error, error.__traceback__)
            finally:
                try:
                    release()
                except BaseException as error:
                    if failure is None:
                        failure = (error, error.__traceback__)
        # A failed body still reaches every barrier; peers finish their own
        # turns before any rank can enter the caller's gradient collective.
        dist.barrier()
    failures = torch.tensor([int(failure is not None)], device="cpu", dtype=torch.int64)
    dist.all_reduce(failures, op=dist.ReduceOp.SUM)
    if failure is not None:
        error, traceback = failure
        raise error.with_traceback(traceback)
    if failures.item():
        raise RuntimeError(f"Serialized GPU backward failed on {failures.item()} peer rank(s)")


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
