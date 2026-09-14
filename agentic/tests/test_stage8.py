"""Composition arithmetic must preserve nitrogen vacancies and fractional Ni."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from stage8_structure_audit import ordered_lower_bound

class VacancyFeasibilityTests(unittest.TestCase):
    def test_fractional_nickel_not_rounded_to_five_atoms(self):
        self.assertEqual(ordered_lower_bound([3.6,.4,1]),dict(formula_units=5,atoms=25))
    def test_eight_percent_vacancy_exceeds_100_atoms(self):
        self.assertEqual(ordered_lower_bound([3,1,.92]),dict(formula_units=25,atoms=123))
    def test_nickel_and_vacancy_denominators_combined(self):
        self.assertEqual(ordered_lower_bound([3.6,.4,.92]),dict(formula_units=25,atoms=123))

if __name__=='__main__':unittest.main()
