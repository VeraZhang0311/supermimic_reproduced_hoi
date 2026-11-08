"""Two-worker CPU Gloo checks of shared-policy learning synchronization."""
from datetime import timedelta
import json
from pathlib import Path
import socket
import tempfile
import unittest

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

from supermimic.networks import RunningNorm
from hoi_tables.distributed import (bind_distributed_statistics, broadcast_models,
                                   state_hash, synchronize_contact_counts, synchronize_gradients)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.norm = RunningNorm(2)

    def forward(self, x):
        return self.linear(x)


def worker(rank, store_uri, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=store_uri, rank=rank, world_size=2,
                            timeout=timedelta(seconds=25))
    try:
        torch.manual_seed(10 + rank)
        model = TinyModel()
        broadcast_models(model)
        initial = [None, None]
        dist.all_gather_object(initial, state_hash(model))
        bind_distributed_statistics(model)
        # Rank 1 has no local samples. It must still participate, and its
        # buffers must become exactly equal to rank 0's global moments.
        first = torch.tensor([[1., 2.], [3., 4.]]) if rank == 0 else torch.empty(0, 2)
        model.norm.update(first)
        first_moments = {"mean": model.norm.mean.tolist(), "var": model.norm.var.tolist(),
                         "count": float(model.norm.count)}
        second = torch.empty(0, 2) if rank == 0 else torch.tensor([[5., 6.]])
        model.norm.update(second)
        # A globally empty update is a collective no-op, never NaN.
        before_empty = state_hash(model)
        model.norm.update(torch.empty(0, 2))
        after_empty = state_hash(model)
        x = torch.tensor([[1. + rank, 2. + rank, 3. + rank]])
        (model(x).sum() * (rank + 1)).backward()
        synchronize_gradients(model)
        gradient_weight = model.linear.weight.grad.tolist()
        gradient_bias = model.linear.bias.grad.tolist()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        optimizer.step()
        positive, negative = synchronize_contact_counts(torch.tensor([rank + 1., 0., 0., 0., 0.]),
                                                        torch.tensor([0., rank + 1., 0., 0., 0.]))
        hashes = [None, None]
        dist.all_gather_object(hashes, state_hash(model))
        report = {"initial_hashes": initial, "final_hashes": hashes, "first_moments": first_moments,
                  "final_mean": model.norm.mean.tolist(), "final_var": model.norm.var.tolist(),
                  "final_count": float(model.norm.count), "empty_unchanged": before_empty == after_empty,
                  "weight_gradient": gradient_weight, "bias_gradient": gradient_bias,
                  "positive": positive.tolist(), "negative": negative.tolist()}
        (Path(directory) / f"rank_{rank}.json").write_text(json.dumps(report), encoding="utf-8")
        dist.barrier()
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "PyTorch Gloo is required")
class DistributedTests(unittest.TestCase):
    def test_two_workers_broadcast_gradients_rms_empty_batches_and_counts(self):
        with tempfile.TemporaryDirectory(prefix="tables-gloo-") as directory:
            # TCP rendezvous avoids native FileStore's narrow-path behavior
            # on Windows profiles containing non-ASCII characters.
            with socket.socket() as address:
                address.bind(("127.0.0.1", 0))
                port = address.getsockname()[1]
            store = f"tcp://127.0.0.1:{port}?use_libuv=0"
            mp.spawn(worker, args=(store, directory), nprocs=2, join=True)
            reports = [json.loads((Path(directory) / f"rank_{i}.json").read_text()) for i in range(2)]
        self.assertEqual(reports[0], reports[1])
        report = reports[0]
        self.assertEqual(len(set(report["initial_hashes"])), 1)
        self.assertEqual(len(set(report["final_hashes"])), 1)
        self.assertNotEqual(report["initial_hashes"][0], report["final_hashes"][0])
        self.assertEqual(report["first_moments"], {"mean": [2., 3.], "var": [1., 1.], "count": 2.})
        self.assertEqual(report["final_count"], 3.)
        self.assertEqual(report["final_mean"], [3., 4.])
        self.assertTrue(torch.allclose(torch.tensor(report["final_var"]), torch.full((2,), 8./3), atol=1e-6))
        self.assertTrue(report["empty_unchanged"])
        self.assertEqual(report["weight_gradient"], [[2.5, 4., 5.5], [2.5, 4., 5.5]])
        self.assertEqual(report["bias_gradient"], [1.5, 1.5])
        self.assertEqual(report["positive"], [3., 0., 0., 0., 0.])
        self.assertEqual(report["negative"], [0., 3., 0., 0., 0.])


if __name__ == "__main__":
    unittest.main()
