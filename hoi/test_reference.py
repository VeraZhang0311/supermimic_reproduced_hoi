"""Check physical-time sampling and quaternion interpolation at the adapter boundary."""
import unittest
import torch
from .ultra_reference import UltraReference

class ReferenceTest(unittest.TestCase):
    def test_60_to_50_preserves_human_object_clock_and_endpoint(self):
        x=torch.zeros(301,630)
        t=torch.arange(301)/60
        x[:,0]=2*t;x[:,71]=3*t
        x[:,6]=1;x[:,77]=1
        body_quat=x[:,201:357].reshape(301,39,4);body_quat[:,:,3]=1
        ref=UltraReference(x)
        sampled=ref.sample(torch.arange(251)*.02)
        torch.testing.assert_close(sampled[:,0],torch.arange(251)*.04)
        torch.testing.assert_close(sampled[:,71],torch.arange(251)*.06)
        self.assertEqual(ref.duration,5.)
        self.assertEqual(float(sampled[-1,0]),10.)
    def test_antipodal_quaternions_do_not_interpolate_through_zero(self):
        x=torch.zeros(2,630);x[0,6]=1;x[1,6]=-1;x[:,77]=1
        x[:,201:357].reshape(2,39,4)[:,:,3]=1
        y=UltraReference(x).sample(torch.tensor([1/120]))
        self.assertAlmostEqual(float(y[0,3:7].norm()),1.)
        self.assertAlmostEqual(abs(float(y[0,6])),1.)
    def test_invalid_or_nonfinite_reference_is_rejected(self):
        with self.assertRaises(ValueError):UltraReference(torch.zeros(1,630))
        x=torch.zeros(2,630);x[0,0]=float('nan')
        with self.assertRaises(ValueError):UltraReference(x)

if __name__=='__main__':unittest.main()
