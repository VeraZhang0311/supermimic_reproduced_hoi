"""CPU deployment contracts; no CUDA context or simulator is created."""
import argparse
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hoi.v1.workers import (WorkerSpec, WorkerGroup, collector_device_list, assign_worker_devices,
                            isolate_worker_cuda, startup_concurrency, add_collector_arguments)
from hoi.v1.resource_probe import summarize_gpu_samples, critic_target_values


class WorkerDeviceTests(unittest.TestCase):
    def test_whole_specs_round_robin_preserves_all_quotas_and_order(self):
        specs = [WorkerSpec(i, n, n * 2, str(i)) for i, n in enumerate((8, 3, 9, 2, 5))]
        original = list(specs)
        self.assertEqual(assign_worker_devices(specs, device="cuda:0"), ["cuda:0"] * 5)
        self.assertEqual(assign_worker_devices(specs, device="cuda:0", collector_devices=["cuda:2", "cuda:5"]),
                         ["cuda:2", "cuda:5", "cuda:2", "cuda:5", "cuda:2"])
        self.assertEqual(specs, original)
        self.assertEqual(sum(s.num_envs for s in specs), 27)
        self.assertEqual(len(set(s.asset_id for s in specs)), 5)

    def test_invalid_device_lists_and_startup_limits_fail(self):
        for value in ([], "cuda:1", ["cpu"], ["cuda"], ["cuda:-1"], ["cuda:0", "cuda:00"], [1]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                collector_device_list("cuda:0", value)
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                startup_concurrency(value, 5)
        self.assertEqual(startup_concurrency(None, 5), 5)
        self.assertEqual(startup_concurrency(2, 5), 2)
        self.assertEqual(startup_concurrency(9, 5), 5)
        with self.assertRaises(ValueError):
            assign_worker_devices([WorkerSpec(0, 1, 1, "0")] * 2, device="cpu")

    def test_child_visibility_maps_parent_logical_index_before_cuda_init(self):
        for visible, assigned, expected in (("GPU-A,GPU-B", "cuda:1", "GPU-B"),
                                             ("3,7", "cuda:1", "7"),
                                             ("MIG-device-A,MIG-device-B", "cuda:0", "MIG-device-A"),
                                             (None, "cuda:3", "3")):
            environ = {} if visible is None else {"CUDA_VISIBLE_DEVICES": visible}
            events = []
            class Cuda:
                def is_initialized(self):
                    events.append("check")
                    return False
                def set_device(self, index):
                    self_index = index
                    events.append(("set", self_index, environ["CUDA_VISIBLE_DEVICES"]))
            self.assertEqual(isolate_worker_cuda(assigned, environ=environ, cuda=Cuda()), ("cuda:0", expected))
            self.assertEqual(events, ["check", ("set", 0, expected)])
        with self.assertRaisesRegex(RuntimeError, "precede CUDA"):
            isolate_worker_cuda("cuda:1", environ={}, cuda=SimpleNamespace(is_initialized=lambda: True))
        for visible, device in (("GPU-A", "cuda:1"), ("", "cuda:0"), ("-1", "cuda:0")):
            with self.assertRaises(ValueError):
                isolate_worker_cuda(device, environ={"CUDA_VISIBLE_DEVICES": visible},
                                    cuda=SimpleNamespace(is_initialized=lambda: False))

    def test_cli_defaults_and_explicit_devices(self):
        parser = argparse.ArgumentParser()
        add_collector_arguments(parser)
        args = parser.parse_args([])
        self.assertIsNone(args.collector_devices)
        self.assertIsNone(args.worker_startup_concurrency)
        args = parser.parse_args(["--collector-devices", "cuda:1", "cuda:2", "--worker-startup-concurrency", "2"])
        self.assertEqual(args.collector_devices, ["cuda:1", "cuda:2"])
        self.assertEqual(args.worker_startup_concurrency, 2)

    def test_group_bounds_only_startup_and_retains_transport_order(self):
        for concurrency, expected_events in ((2, ["start0", "start1", "ready0", "ready1", "start2", "ready2"]),
                                              (None, ["start0", "start1", "start2", "ready0", "ready1", "ready2"])):
            events, configs, parents = [], [], []
            specs = [WorkerSpec(i, n, n, str(i)) for i, n in enumerate((2, 1, 3))]
            class Pipe:
                message = None
                def send(self, message): self.message = message
                def close(self): pass
            class Context:
                def Pipe(self):
                    parent = Pipe(); parents.append(parent)
                    return parent, Pipe()
                def Process(self, *, target, args, name):
                    _, spec, config = args
                    configs.append((spec, config))
                    return SimpleNamespace(start=lambda: events.append("start" + str(spec.asset_id)),
                                           join=lambda **kw: None, is_alive=lambda: False)
            def receive(group, index, timeout=None):
                spec = group.specs[index]
                snapshot = {"asset_id": torch.full((spec.num_envs,), spec.asset_id)}
                if parents[index].message is None:
                    events.append("ready" + str(index))
                    return "ready", snapshot, {"asset_id": spec.asset_id}
                return "step", {"a": parents[index].message[1], **snapshot}, snapshot
            config = {"device": "cuda:0", "collector_devices": ["cuda:1", "cuda:2"],
                      "worker_startup_concurrency": concurrency}
            with patch("hoi.v1.workers.mp.get_context", return_value=Context()), patch.object(WorkerGroup, "_receive", receive):
                group = WorkerGroup(specs, config, device="cpu")
                self.assertEqual(events, expected_events)
                self.assertEqual([x[0] for x in configs], specs)
                self.assertEqual([x[1]["device"] for x in configs], ["cuda:1", "cuda:2", "cuda:1"])
                self.assertEqual(config["device"], "cuda:0")
                self.assertEqual(group.snapshot()["asset_id"].tolist(), [0, 0, 1, 2, 2, 2])
                action = torch.arange(6 * 29).reshape(6, 29).float()
                transition = group.step(action)
                torch.testing.assert_close(transition["a"], action)
                self.assertEqual(transition["asset_id"].tolist(), [0, 0, 1, 2, 2, 2])
                group.close()

    def test_gpu_report_includes_owned_worker_devices(self):
        sample = {"elapsed_seconds": 1., "gpus": [[u, "GPU", "40000", "5000"] for u in ("A", "B", "C")],
                  "processes": [["10", "1000", "A"], ["11", "2000", "B"], ["12", "300", "B"], ["99", "9000", "C"]]}
        report = summarize_gpu_samples([sample], [10, 11], 10)
        self.assertEqual(report["assigned_device_uuids"], ["A", "B"])
        self.assertEqual(report["owned_process_sampled_peak_MiB_by_gpu_uuid"], {"A": 1000, "B": 2000})
        self.assertEqual(report["external_process_sampled_peak_MiB_by_gpu_uuid"], {"B": 300})

    def test_critic_targets_match_production_chunk_size_and_all_row_order(self):
        # H*B=18, four minibatches => ceil(18/4)=5, including all 24 bootstrap rows.
        x = torch.arange(24).float()[:, None]
        calls = []
        def critic(a, b):
            calls.append(len(a))
            return (a + b).squeeze(-1)
        values = critic_target_values(critic, (x, 2*x), horizon=3, batch=6, critic_minibatches=4)
        self.assertEqual(calls, [5, 5, 5, 5, 4])
        torch.testing.assert_close(values, (3*x).reshape(4, 6))


if __name__ == "__main__":
    unittest.main()
