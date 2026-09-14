from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_pilot30 import check

class Pilot30Validation(unittest.TestCase):
    def event(self, **updates):
        e=dict(material_formula='X',type='neel',temperature_k=440,uncertainty_k=5,
            qualifier='reported_value',sublattice=None,sample_conditions=None,quote='TN=440(5) K')
        e.update(updates)
        return {'events':[e],'abstained':False}

    def test_uncertainty_retained(self):
        self.assertEqual(check(self.event(),dict(formula='X',excerpts=['TN=440(5) K'])),[])

    def test_uncertainty_omission_flagged(self):
        self.assertIn('reported_uncertainty_missing',check(self.event(uncertainty_k=None),dict(formula='X',excerpts=['TN=440(5) K'])))

    def test_wrong_compound_rejected(self):
        with self.assertRaises(ValueError):
            check(self.event(material_formula='Y'),dict(formula='X',excerpts=['TN=440(5) K']))

    def test_upper_bound_loss_flagged(self):
        flags=check(self.event(quote='Transition Temperature: <295 K'),dict(formula='X',excerpts=['Transition Temperature: <295 K']))
        self.assertIn('upper_bound_not_preserved',flags)

    def test_unknown_not_curie(self):
        flags=check(self.event(type='curie',quote='Tc=440 K'),dict(formula='X',excerpts=['Tc=440 K']))
        self.assertIn('curie_without_explicit_quote',flags)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            check(self.event(temperature_k=float('nan')),dict(formula='X',excerpts=['TN=440(5) K']))

    def test_near_is_not_exact(self):
        flags=check(self.event(quote='near TN=22K'),dict(formula='X',excerpts=['near TN=22K']))
        self.assertIn('approximation_not_preserved',flags)

    def test_no_evidence_abstention(self):
        self.assertEqual(check(dict(events=[],abstained=True),dict(formula='X',excerpts=[])),[])

    def test_invented_quote_rejected(self):
        with self.assertRaises(ValueError):
            check(self.event(),dict(formula='X',excerpts=['Nothing reported']))

if __name__=='__main__':
    unittest.main()
