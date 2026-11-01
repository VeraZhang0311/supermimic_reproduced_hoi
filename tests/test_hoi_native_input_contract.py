"""Synthetic CPU contract tests; these never establish native CUDA acceptance."""
from pathlib import Path
import tempfile
import unittest
import torch
from hoi.v1.native_input_contract import validate_transition_inputs
from tests.test_hoi_privileged_teacher import teacher_learner
from tests.test_hoi_v1_learner import transition


class NativeInputContractTests(unittest.TestCase):
    def test_production_networks_keep_rng_unknown_and_absent_support_contract(self):
        learner=teacher_learner();tr=transition(learner,[10,70],0)
        tr["support_pose_xyzw"][...,6]=1
        tr["support_present"][0,0]=True
        tr["support_pose_xyzw"][0,0,:3]=torch.tensor([1.,2.,.4])
        learner.motion.expected_contact[tr["f"]+1,7]=0
        learner.motion.expected_valid[tr["f"]+1,7]=0
        original={k:v.clone() for k,v in tr.items()};rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"input.pt"
            report=validate_transition_inputs(tr,learner.mdp,path,require_cuda=False)
            saved=torch.load(path,weights_only=True)
        self.assertTrue(report["passed"]);self.assertFalse(report["cuda_verified"])
        self.assertTrue(report["rng_preserved"]);self.assertEqual(report["unknown_reference_slots"],2)
        self.assertEqual(report["dimensions"]["wm_input"],250)
        self.assertEqual(report["dimensions"]["teacher"],3962)
        self.assertEqual(report["dimensions"]["contact_logits"],16)
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        for k,v in tr.items():torch.testing.assert_close(v,original[k],rtol=0,atol=0)
        self.assertTrue(bool((saved["scene_arrival"].reshape(2,2,13)[~tr["support_present"]]==0).all()))
        torch.testing.assert_close(saved["expected_valid"],learner.motion.expected_valid[tr["f"]+1])
        self.assertEqual(report["optimizer_steps"],0);self.assertFalse(report["outputs_applied_to_simulator"])

    def test_cpu_is_rejected_for_real_native_acceptance(self):
        learner=teacher_learner();tr=transition(learner,[10,70],0)
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError,"actual CUDA"):
            validate_transition_inputs(tr,learner.mdp,Path(directory)/"input.pt")

    def test_bad_contact_and_missing_scene_rejected_before_any_model_probe(self):
        learner=teacher_learner();tr=transition(learner,[10,70],0)
        tr["contact"][0,4]=.5
        with tempfile.TemporaryDirectory() as directory,self.assertRaisesRegex(ValueError,"binary"):
            validate_transition_inputs(tr,learner.mdp,Path(directory)/"input.pt",require_cuda=False)
        tr["contact"][0,4]=1;del tr["support_present"]
        with tempfile.TemporaryDirectory() as directory,self.assertRaisesRegex(ValueError,"scene support"):
            validate_transition_inputs(tr,learner.mdp,Path(directory)/"input.pt",require_cuda=False)
