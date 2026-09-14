import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('pilot', Path(__file__).resolve().parents[1] / 'scripts/run_evidence_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)

class EvidenceValidationTests(unittest.TestCase):
    def test_no_source_requires_abstention(self):
        self.assertEqual(pilot.validate({'events': [], 'abstained': True}, [])['events'], [])
        with self.assertRaises(ValueError):
            pilot.validate({'events': [], 'abstained': False}, [])

    def test_fabricated_quote_rejected(self):
        event = dict(type='curie', temperature_k=700, qualifier='exact', quote='Tc = 700 K')
        with self.assertRaises(ValueError):
            pilot.validate({'events': [event], 'abstained': False}, ['No temperature reported.'])

    def test_bound_preserved(self):
        event = dict(type='neel', temperature_k=1000, qualifier='lower_bound', quote='TN > 1000 K')
        result = pilot.validate({'events': [event], 'abstained': False}, [event['quote']])
        self.assertEqual(result['events'][0]['qualifier'], 'lower_bound')

    def test_nonphysical_numeric_rejected(self):
        for temperature in [-1, float('nan'), float('inf'), True, '745']:
            event = dict(type='curie', temperature_k=temperature, qualifier='exact', quote='745 K')
            with self.assertRaises(ValueError):
                pilot.validate({'events': [event], 'abstained': False}, ['745 K'])

    def test_abstention_cannot_include_events(self):
        with self.assertRaises(ValueError):
            pilot.validate({'events': [{}], 'abstained': True}, ['text'])

class ObservedSemanticErrors(unittest.TestCase):
    def test_tc_symbol_is_not_curie_evidence(self):
        result = {'events': [dict(type='curie', qualifier='exact', quote='Tc = 0.953 K')]}
        self.assertIn('event_0:curie_requires_explicit_evidence', pilot.semantic_flags(result))

    def test_below_is_not_automatically_exact(self):
        result = {'events': [dict(type='other', qualifier='exact', quote='transition occurs below 7 K')]}
        self.assertIn('event_0:exactness_requires_context_review', pilot.semantic_flags(result))

if __name__ == '__main__':
    unittest.main()
