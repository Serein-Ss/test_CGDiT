"""Uncertainty-aware comparisons must not inflate evidence or training readiness."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from stage7_feni_extraction import robust_pair

class FeNiEvidenceTests(unittest.TestCase):
    def obs(self,t):return dict(material_formula=str(t),temperature_k=t,uncertainty_k=10)
    def test_uncertainty_changes_55k_margin_decision(self):
        p=robust_pair(self.obs(695),self.obs(640))
        self.assertEqual(p['interval_gap_k'],35)
        self.assertFalse(p['passes_margin'])
    def test_72k_difference_clears_50k_after_reported_error(self):
        p=robust_pair(self.obs(767),self.obs(695))
        self.assertEqual(p['interval_gap_k'],52)
        self.assertTrue(p['passes_margin'])
    def test_pair_acceptance_does_not_imply_independent_or_trainable(self):
        p=robust_pair(self.obs(767),self.obs(640))
        self.assertTrue(p['passes_margin']);self.assertFalse(p['training_eligible'])
        self.assertEqual(p['split'],'unassigned_source_group')
        self.assertEqual(p['source_group'],'Westinghouse_FeNiN_1962_TableII')

if __name__=='__main__':unittest.main()
