"""Regression checks for leakage and chemical identity in reference pairing."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from stage4_pair_acceptance import assign_pair, parse
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator


class PairAcceptanceTests(unittest.TestCase):
    def test_heldout_match_blocks_pair(self):
        for split in ('val', 'test'):
            self.assertEqual(assign_pair({'a':[{'split':'train'}], 'b':[{'split':split}]}, []),
                             'blocked_heldout_overlap')

    def test_incomplete_scan_blocks_pair(self):
        self.assertEqual(assign_pair({'a':[]}, [{'error':'invalid CIF'}]), 'blocked_incomplete_audit')

    def test_cn_identity_is_not_topology_match(self):
        s,_=parse((ROOT/'data/curation/stage3_reference_structures/Mn3AlC_1962_reference.cif').read_text())
        other=s.copy();other.replace_species({'C':'N'})
        matcher=StructureMatcher(comparator=ElementComparator())
        self.assertFalse(matcher.fit(s,other))
        self.assertTrue(matcher.fit(s,s.copy()))

    def test_exported_pair_orientation_and_scope(self):
        p=json.loads((ROOT/'data/curation/stage4_pair/pairs.jsonl').read_text())
        self.assertEqual(p['winner_id'],'Mn4N_1962_reference')
        self.assertEqual(p['loser_id'],'Mn3AlC_1962_reference')
        self.assertEqual(p['winner_temperature_k']-p['loser_temperature_k'],p['delta_k'])
        self.assertTrue(p['pipeline_ready'])
        self.assertFalse(p['production_training_eligible'])
        self.assertEqual(p['structure_kind'],'source_reconstructed_nuclear_reference')

if __name__=='__main__':unittest.main()
