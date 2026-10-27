"""CPU checks for GPU validation guards, not GPU/native evidence."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hoi.v1.actor_fusion import FUSION_ACTOR
from hoi.v1.config import TrainConfig
from hoi.v1.observations import MATCHING
from hoi.v1.randomization import MATCHING_DR
from tools.hoi_v1_2_resource_probe import validate_production_config, fusion_gradient_summary
from tools.hoi_v1_2_validate import native_report_workers


def fake_policy():
    fusion = nn.Module()
    fusion.projections = nn.ModuleDict({name: nn.Linear(2, 2) for name in
                                      ("robot_history", "human_future", "object_history", "object_future", "shape")})
    return SimpleNamespace(fusion=fusion, enc=nn.Linear(2, 2), object_enc=nn.Linear(2, 2),
                           shape_encoder=nn.Linear(2, 2), dec=nn.Linear(2, 2))


class GPUValidationGuardTests(unittest.TestCase):
    def test_native_report_controller_uses_cpu_transport_without_changing_cuda_workers(self):
        specs, worker_config = [object()], {"device": "cuda:0", "validate_domain_randomization": True}
        with patch("hoi.v1.workers.WorkerGroup") as group:
            self.assertIs(native_report_workers(specs, worker_config), group.return_value)
        group.assert_called_once_with(specs, worker_config, device="cpu")
        self.assertEqual(worker_config["device"], "cuda:0")

    def test_capacity_guard_rejects_reduced_batches_or_wrong_actor(self):
        config = TrainConfig(actor_architecture=FUSION_ACTOR, observation_profile=MATCHING,
                             domain_randomization_profile=MATCHING_DR)
        validate_production_config(config)
        for key, value in (("pi_batch", 2048), ("horizon", 16), ("buffer_steps", 64),
                           ("num_envs", 64), ("wm_batch", 1024), ("critic_epochs", 1),
                           ("actor_architecture", "mlp"), ("wm_only_iters", 0)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_production_config(replace(config, **{key: value}))

    def test_second_actor_requires_every_modality_to_have_live_finite_gradients(self):
        policy = fake_policy()
        for module in (policy.fusion, policy.enc, policy.object_enc, policy.shape_encoder, policy.dec):
            for parameter in module.parameters():
                parameter.grad = torch.ones_like(parameter)
        summary = fusion_gradient_summary(policy, require_live=True)
        self.assertEqual(len(summary["projections"]), 5)
        for parameter in policy.fusion.projections["object_future"].parameters():
            parameter.grad.zero_()
        with self.assertRaisesRegex(AssertionError, "object_future"):
            fusion_gradient_summary(policy, require_live=True)
        for parameter in policy.fusion.parameters():
            parameter.grad.zero_()
        fusion_gradient_summary(policy, require_live=False)
        next(policy.fusion.parameters()).grad.fill_(float("nan"))
        with self.assertRaisesRegex(AssertionError, "nonfinite"):
            fusion_gradient_summary(policy, require_live=True)


if __name__ == "__main__":
    unittest.main()
