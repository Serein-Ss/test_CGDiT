from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from stage3_nitride_feasibility import ordered_atom_lower_bound, pair_gate, inspect_explicit_cell
from stage3_source_pairs import reference_cif

class Stage3Tests(unittest.TestCase):
    def pair(self,**updates):
        p=dict(winner_event='curie',loser_event='curie',evidence_kind='experimental_report',same_paper=True,
            winner_temperature_k=780,loser_temperature_k=610,comparison_margin_k=50,local_structure_mapping_verified=True,
            winner_split='train',loser_split='train',minimum_ordered_atoms=5,configured_max_atoms=100)
        p.update(updates);return p
    def test_compensation_not_tc(self):
        self.assertFalse(pair_gate(self.pair(loser_event='compensation'))['literature_comparison_supported'])
    def test_theory_not_experiment(self):
        self.assertFalse(pair_gate(self.pair(evidence_kind='calculated'))['literature_comparison_supported'])
    def test_no_holdout_leakage(self):
        self.assertFalse(pair_gate(self.pair(loser_split='test'))['training_eligible'])
    def test_unmatched_structure_rejected(self):
        self.assertFalse(pair_gate(self.pair(local_structure_mapping_verified=False))['training_eligible'])
    def test_margin_blocks_close_values(self):
        self.assertFalse(pair_gate(self.pair(winner_temperature_k=750,loser_temperature_k=745))['literature_comparison_supported'])
    def test_nominal_ga_cell(self):
        self.assertEqual(ordered_atom_lower_bound('0.24'),125)
        self.assertEqual(ordered_atom_lower_bound('0.25'),20)
        self.assertFalse(pair_gate(self.pair(minimum_ordered_atoms=125))['training_eligible'])
    def test_carbon_is_not_nitride(self):
        self.assertFalse(inspect_explicit_cell(reference_cif('Mn3AlC',3.869,'Al','C'))['template_match'])
        self.assertTrue(inspect_explicit_cell(reference_cif('Mn4N',3.865,'Mn','N'))['template_match'])
    def test_partial_occupancy_not_full_cell(self):
        cif=reference_cif('Mn4N',3.865,'Mn','N').replace('N N4 1 0.5 0.5 0.5 1','N N4 1 0.5 0.5 0.5 0.75')
        self.assertFalse(inspect_explicit_cell(cif)['template_match'])
if __name__=='__main__':unittest.main()
