import copy
from types import SimpleNamespace
import unittest

import torch

from hoi_residual.evaluate import rollout_clip, summarize, validate_metadata
from hoi_residual.randomization import CONTRACT
from hoi_residual.contract import implementation_contract
from hoi_residual.features import ORACLE_DIM, observation_contract


class EvaluationContractTests(unittest.TestCase):
    def test_checkpoint_binding_rejects_changed_identities(self):
        metadata = {"checkpoint_sha256": "gmt-hash", "iteration": 26000,
                    "source_contract": {"version": "v3"}, "target_contract": {"dataset": "ultra"}, "source_amp": "bf16"}
        architecture = {"schema": "oracle", "oracle_dim": ORACLE_DIM}
        shape_metadata = {'bank_sha256':'shape-hash','contract':{'points':128}}
        checkpoint = {"schema": "hoi_residual_ppo_teacher_v2", "architecture": architecture,
                      "manifest_sha256": "manifest-hash", "assets": ["asset"],
                      "asset_bundle_sha256": "asset-hash", "gmt": metadata, "iteration": 2,
                      "training_assets": ["asset"], "capacity": {"nconmax": 512, "njmax": 2048},
                      "domain_randomization": CONTRACT, "residual_contract": implementation_contract(),
                      'observation_contract':observation_contract(),
                      'shape_bank_sha256':'shape-hash','shape_contract':shape_metadata['contract']}
        arguments = dict(architecture=architecture, gmt_metadata=metadata, manifest_sha256="manifest-hash",
                         object_assets=["asset"], asset_bundle_sha256="asset-hash", selected_asset="asset",
                         shape_metadata=shape_metadata)
        validate_metadata(checkpoint, **arguments)
        for key in ("schema", "architecture", "manifest_sha256", "assets", "asset_bundle_sha256", "gmt",
                    "training_assets", "capacity", "residual_contract", "observation_contract", "shape_bank_sha256", "shape_contract"):
            with self.subTest(key=key):
                modified = copy.deepcopy(checkpoint)
                modified[key] = {}
                with self.assertRaises(ValueError):
                    validate_metadata(modified, **arguments)

    def test_complete_duration_continues_after_first_step_failure(self):
        class Env:
            device = "cpu"
            asset_ids = torch.tensor([0])
            bounds = torch.zeros(6)
            xi = torch.zeros(1, 33)

            def __init__(self):
                self.f = torch.zeros(1, dtype=torch.long)
                self.a_prev = torch.zeros(1, 29)
                self.reset_count, self.step_count = 0, 0

            def reset(self, ids, frames):
                self.reset_count += 1
                self.f = frames.clone()

            def refresh(self):
                pass

            def state(self):
                return (torch.tensor([[self.step_count * .1, 0., .8]]), torch.eye(3)[None],
                        torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 29), torch.zeros(1, 29))

            def object_state(self):
                return torch.zeros(1, 3), torch.eye(3)[None], torch.zeros(1, 3), torch.zeros(1, 3)

            def object_context(self):
                return torch.zeros(1, 12)

            def actuator_state(self):
                return torch.zeros(1, 88)

            def step(self, action):
                self.f += 1
                self.step_count += 1
                self.a_prev = action.clone()
                return {"previous_q": torch.zeros(1, 29), "previous_qd": torch.zeros(1, 29),
                        "terms": {"object_surface_error_m": torch.zeros(1)}, "terminated": torch.tensor([True]),
                        "reward": torch.tensor([.1]), "truncated": torch.tensor([self.step_count == 3])}

        env = Env()
        reference = (torch.tensor([[0., 0., .8]]), *env.state()[1:])
        motion = SimpleNamespace(clip_start=torch.tensor([0]), clip_end=torch.tensor([3]), clip_len=torch.tensor([4]),
                                 clip_id=torch.zeros(4, dtype=torch.long),
                                 object_ref=lambda f: tuple(v[0].expand(*f.shape,*v.shape[1:]) for v in env.object_state()),
                                 goal=lambda f, k: torch.zeros(1, 730), ref_state=lambda f: reference,
                                 reference_bodies=lambda f: (reference[0][:, None], None, None, None),
                                 entries=[{"motion": "/fixture.pt", "motion_sha256": "sha", "source_group": "fixture",
                                           "partition": "test", "object_asset": "largebox_100_100_100"}])
        mdp = SimpleNamespace(obs=lambda *args: torch.zeros(1, 99), termination=lambda *args: torch.tensor([True]),
                              kin=SimpleNamespace(forward=lambda *state: (state[0][:, None], None, None, None)))
        privileged = lambda *args: torch.zeros(1, 3371)
        gmt = lambda *args: torch.zeros(1, 29)
        seen = []
        def act(oracle, base, points, **kwargs):
            self.assertEqual(points.shape,(128,3))
            seen.append(oracle[:,3371:3461].reshape(1,10,9).clone())
            return torch.zeros(1,29)
        policy = SimpleNamespace(act=act)
        teacher_kin = SimpleNamespace(body_names=['torso_link'],
                                     forward=lambda state:(state[0][:,None],state[1][:,None],None,None))
        result = rollout_clip(env, motion, mdp, privileged, gmt, policy, 0, teacher_kin, torch.zeros(128,3))
        self.assertEqual(env.reset_count, 1)
        self.assertEqual(env.step_count, 3)
        self.assertEqual(result["control_steps"], 3)
        self.assertEqual(result["state_frames"], 4)
        self.assertEqual(result["first_failure_step"], 1)
        self.assertEqual(result["termination_steps"], 3)
        self.assertEqual(result["resets_within_clip"], 0)
        torch.testing.assert_close(seen[0],seen[0][:,-1:].expand_as(seen[0]))
        self.assertAlmostEqual(float(seen[1][0,-1,0]),-.1,places=6)
        self.assertAlmostEqual(float(seen[2][0,-2,0]),-.1,places=6)
        self.assertAlmostEqual(float(seen[2][0,-1,0]),-.2,places=6)
        self.assertAlmostEqual(result["return"], .3, places=6)
        self.assertAlmostEqual(result["root_position_error_m"]["mean"], .2, places=6)
        report = summarize([result])
        self.assertEqual(report["failure_clips"], 1)
        self.assertAlmostEqual(report["mean_normalized_return"], result["normalized_return"])


if __name__ == "__main__":
    unittest.main()
