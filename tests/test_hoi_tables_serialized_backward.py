"""CPU Gloo proof of shared-GPU turn scheduling; no native GPU claims."""
from datetime import timedelta
import json
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

from hoi_tables.distributed import (serialized_gpu_backward, state_hash,
                                   synchronize_gradients)
from supermimic.networks import RunningNorm


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = RunningNorm(3)
        self.net = nn.Sequential(nn.Linear(3, 12), nn.Tanh(), nn.Linear(12, 3))
        with torch.no_grad():
            self.norm.mean.copy_(torch.tensor([.1, .2, .3]))
            self.norm.var.copy_(torch.tensor([.8, 1.1, 1.4]))
            self.norm.count.fill_(100)

    def forward(self, state):
        return self.net(self.norm(state))


def full_batch_backward(model, rank):
    """A complete 17-start, H32 differentiable rollout, without RNG or RMS writes."""
    state = torch.arange(51, dtype=torch.float64).reshape(17, 3)/50 + rank*.1
    objective = state.new_zeros(())
    for _ in range(32):
        action = model(state)
        state = .97*state + .03*action
        objective = objective + state.square().mean() + .2*action.square().mean()
    loss = objective/32
    loss.backward()
    return float(loss.detach())


def enter_phase(rank, workers_per_gpu, shared):
    active, maxima, global_maximum, release_counts, lock, gpu_rendezvous = shared
    gpu = rank//workers_per_gpu
    with lock:
        active[gpu] += 1
        maxima[gpu] = max(maxima[gpu], active[gpu])
        global_maximum.value = max(global_maximum.value, sum(active))
    # The active ranks on each logical GPU meet here. This would time out
    # if the implementation serialized independent GPUs as well as sharers.
    gpu_rendezvous.wait(timeout=20)


def release_phase(rank, workers_per_gpu, shared, *, entered=True, fail=False):
    active, maxima, global_maximum, release_counts, lock, gpu_rendezvous = shared
    with lock:
        if entered:
            active[rank//workers_per_gpu] -= 1
        release_counts[rank] += 1
    if fail:
        raise RuntimeError("injected allocator release failure")


def gradient_worker(rank, world, workers_per_gpu, store, directory, shared, ranks_per_turn):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=store, rank=rank, world_size=world,
                            timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(901)
        model = Actor().double()
        initial_hash = state_hash(model)
        initial_rng = torch.random.get_rng_state().clone()
        concurrent_loss = full_batch_backward(model, rank)
        concurrent_local = [p.grad.clone() for p in model.parameters()]
        synchronize_gradients(model)
        concurrent_global = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        entered = False
        releases = lambda: release_phase(rank, workers_per_gpu, shared, entered=entered)
        with serialized_gpu_backward(workers_per_gpu, release=releases, ranks_per_turn=ranks_per_turn):
            entered = True
            enter_phase(rank, workers_per_gpu, shared)
            serial_loss = full_batch_backward(model, rank)
        local_equal = all(torch.equal(p.grad, expected) for p, expected in zip(model.parameters(), concurrent_local))
        # The full batch's gradients are synchronized only after all turns.
        synchronize_gradients(model)
        global_equal = all(torch.equal(p.grad, expected) for p, expected in zip(model.parameters(), concurrent_global))
        unchanged = state_hash(model) == initial_hash
        unchanged_rng = torch.equal(initial_rng, torch.random.get_rng_state())
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        optimizer.step()
        final_hashes = [None]*world
        dist.all_gather_object(final_hashes, state_hash(model))
        report = {"local_equal": local_equal, "global_equal": global_equal,
                  "loss_equal": concurrent_loss == serial_loss,
                  "parameters_and_rms_unchanged": unchanged, "rng_unchanged": unchanged_rng,
                  "final_model_hashes": final_hashes,
                  "model_changed_after_step": state_hash(model) != initial_hash,
                  "batch": 17, "horizon": 32}
        (Path(directory)/f"rank_{rank}.json").write_text(json.dumps(report))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def failure_worker(rank, world, workers_per_gpu, store, directory, shared, ranks_per_turn):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=store, rank=rank, world_size=world,
                            timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(902)
        model = Actor().double()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        initial_hash = state_hash(model)
        initial_rng = torch.random.get_rng_state().clone()
        reports = []
        # Both a body error and an allocator release error must reach all peers.
        # Reusing the group proves no barrier was left unmatched by the first.
        for failure_kind in ("body", "release", "pre_release"):
            model.zero_grad(set_to_none=True)
            optimizer_steps, sync_entered, error_type, message = 0, False, None, None
            entered = False
            release = lambda: release_phase(rank, workers_per_gpu, shared, entered=entered,
                fail=rank == 1 and ((failure_kind == "release" and entered)
                                   or (failure_kind == "pre_release" and not entered)))
            try:
                with serialized_gpu_backward(workers_per_gpu, release=release, ranks_per_turn=ranks_per_turn):
                    entered = True
                    enter_phase(rank, workers_per_gpu, shared)
                    full_batch_backward(model, rank)
                    if failure_kind == "body" and rank == 1:
                        raise ValueError("injected actor backward failure")
                sync_entered = True
                synchronize_gradients(model)
                optimizer.step()
                optimizer_steps += 1
            except (ValueError, RuntimeError) as error:
                error_type, message = type(error).__name__, str(error)
            reports.append({"kind": failure_kind, "error_type": error_type, "message": message,
                            "optimizer_steps": optimizer_steps, "sync_entered": sync_entered,
                            "parameters_and_rms_unchanged": state_hash(model) == initial_hash,
                            "rng_unchanged": torch.equal(initial_rng, torch.random.get_rng_state())})
            dist.barrier()
        (Path(directory)/f"rank_{rank}.json").write_text(json.dumps(reports))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def launch(worker, world, workers_per_gpu, ranks_per_turn=1):
    context = mp.get_context("spawn")
    gpus = world//workers_per_gpu
    shared = (context.Array("i", gpus), context.Array("i", gpus), context.Value("i", 0),
              context.Array("i", world), context.Lock(), context.Barrier(gpus*ranks_per_turn))
    with TemporaryDirectory(prefix="hoi-serialized-gloo-") as directory:
        with socket.socket() as address:
            address.bind(("127.0.0.1", 0)); port = address.getsockname()[1]
        # Native Windows FileStore and libuv cannot safely rendezvous under
        # every non-ASCII profile, so reuse the project's TCP Gloo convention.
        store = f"tcp://127.0.0.1:{port}?use_libuv=0"
        mp.spawn(worker, args=(world, workers_per_gpu, store, directory, shared, ranks_per_turn),
                 nprocs=world, join=True)
        reports = [json.loads((Path(directory)/f"rank_{rank}.json").read_text()) for rank in range(world)]
    return reports, {"active": list(shared[0]), "maxima": list(shared[1]),
                     "global_maximum": shared[2].value, "release_counts": list(shared[3])}


class SerializedNullContextTests(unittest.TestCase):
    def test_single_worker_and_uninitialized_modes_preserve_original_body_behavior(self):
        for workers_per_gpu, initialized in ((1, True), (8, False)):
            releases, body = [], []
            with patch("hoi_tables.distributed.dist.is_initialized", return_value=initialized), \
                 patch("hoi_tables.distributed.dist.get_world_size", side_effect=AssertionError("no collective")):
                with serialized_gpu_backward(workers_per_gpu, release=lambda: releases.append(True)):
                    body.append(True)
                self.assertEqual(body, [True])
                self.assertEqual(releases, [])
                with self.assertRaisesRegex(ValueError, "original body error"):
                    with serialized_gpu_backward(workers_per_gpu):
                        raise ValueError("original body error")


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "PyTorch Gloo is required")
class SerializedDistributedTests(unittest.TestCase):
    def test_two_and_four_ranks_keep_complete_batch_gradients_and_parallel_independent_gpus(self):
        for world, workers_per_gpu, ranks_per_turn in ((2, 2, 1), (4, 2, 1), (4, 4, 1), (4, 4, 2), (4, 4, 4)):
            with self.subTest(world=world, workers_per_gpu=workers_per_gpu, ranks_per_turn=ranks_per_turn):
                reports, shared = launch(gradient_worker, world, workers_per_gpu, ranks_per_turn)
                for report in reports:
                    for key in ("local_equal", "global_equal", "loss_equal", "parameters_and_rms_unchanged",
                                "rng_unchanged", "model_changed_after_step"):
                        self.assertTrue(report[key], key)
                    self.assertEqual((report["batch"], report["horizon"]), (17, 32))
                    self.assertEqual(len(set(report["final_model_hashes"])), 1)
                self.assertEqual(shared["active"], [0]*(world//workers_per_gpu))
                self.assertEqual(shared["maxima"], [ranks_per_turn]*(world//workers_per_gpu))
                self.assertEqual(shared["global_maximum"], (world//workers_per_gpu)*ranks_per_turn)
                self.assertEqual(shared["release_counts"], [2]*world)

    def test_body_and_release_failure_reach_all_peers_without_optimizer_step_or_hang(self):
        reports, shared = launch(failure_worker, 4, 2, ranks_per_turn=2)
        for rank, phases in enumerate(reports):
            for phase in phases:
                self.assertEqual(phase["optimizer_steps"], 0)
                self.assertFalse(phase["sync_entered"])
                self.assertTrue(phase["parameters_and_rms_unchanged"])
                self.assertTrue(phase["rng_unchanged"])
                if rank == 1:
                    self.assertEqual(phase["error_type"], "ValueError" if phase["kind"] == "body" else "RuntimeError")
                    self.assertIn("injected", phase["message"])
                else:
                    self.assertEqual(phase["error_type"], "RuntimeError")
                    self.assertIn("peer rank", phase["message"])
        self.assertEqual(shared["active"], [0, 0])
        self.assertEqual(shared["maxima"], [2, 2])
        self.assertEqual(shared["release_counts"], [6]*4)


if __name__ == "__main__":
    unittest.main()
