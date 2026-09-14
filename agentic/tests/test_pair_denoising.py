"""Preference sign, stop-gradient, and endpoint contract regression tests."""
from pathlib import Path
import sys
import unittest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent));sys.path.insert(0,str(ROOT/'scripts'))
from pair_denoising import preference_loss


class PreferenceTests(unittest.TestCase):
    def test_reference_is_frozen_and_descent_favors_winner(self):
        p=torch.tensor([3.,7.],requires_grad=True);r=p.detach().clone().requires_grad_(True)
        loss=preference_loss(p,r);loss.backward()
        self.assertAlmostEqual(loss.item(),torch.tensor(2.).log().item())
        self.assertGreater(p.grad[0].item(),0);self.assertLess(p.grad[1].item(),0)
        self.assertIsNone(r.grad)

    def test_improving_winner_relative_to_reference_lowers_loss(self):
        r=torch.tensor([3.,7.])
        self.assertLess(preference_loss(torch.tensor([2.,8.]),r),preference_loss(r,r))
        self.assertGreater(preference_loss(torch.tensor([4.,6.]),r),preference_loss(r,r))

    def test_swapping_both_endpoints_reverses_preference(self):
        p=torch.tensor([2.,8.]);r=torch.tensor([3.,7.])
        self.assertGreater(preference_loss(p.flip(0),r.flip(0)),preference_loss(p,r))

    def test_common_energy_shift_cancels(self):
        p=torch.tensor([2.,8.]);r=torch.tensor([3.,7.])
        torch.testing.assert_close(preference_loss(p+5,r),preference_loss(p,r))

    def test_reject_scalar_batch_mean_or_invalid_beta(self):
        for p,r,b in [(torch.tensor(1.),torch.ones(2),1),(torch.ones(3),torch.ones(2),1),
                      (torch.ones(2),torch.ones(2),0)]:
            with self.assertRaises(ValueError):preference_loss(p,r,b)

if __name__=='__main__':unittest.main()
