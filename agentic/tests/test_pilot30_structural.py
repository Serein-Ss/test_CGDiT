"""Leakage, missing-label and conditioner contracts for the bounded structural pilot."""
import json,sys,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT.parent))
from pair_denoising import per_structure_energy

class PilotContracts(unittest.TestCase):
 def test_group_and_endpoint_holdouts_do_not_leak(self):
  pairs=json.loads((ROOT/'data/pilot30_v1/pairs.json').read_text());groups={};ends={}
  for p in pairs:
   groups.setdefault(p['split'],set()).update(p['source_groups']);ends.setdefault(p['split'],set()).update([p['winner'],p['loser']])
  for a in groups:
   for b in groups:
    if a!=b:self.assertFalse(groups[a]&groups[b]);self.assertFalse(ends[a]&ends[b])
 def test_training_requires_high_temperature_anchor_and_gap_buffer(self):
  for p in json.loads((ROOT/'data/pilot30_v1/pairs.json').read_text()):
   if p['split']=='train':self.assertTrue(p['high_tc_anchor']);self.assertGreater(p['sensitivity_gap_k'],50)
 def test_known_electrical_conflicts_are_never_train_targets(self):
  rows=json.loads((ROOT/'data/pilot30_v1/materials.json').read_text())
  for r in rows:
   self.assertIsNone(r['experimental_band_gap_ev'])
   if r['formula'] in ('EuO','EuS'):
    self.assertEqual(r['split'],'electrical_control');self.assertEqual(r['electrical_class'],'insulator');self.assertEqual(r['local_mp_band_gap_ev'],0)
 def test_invalid_paired_conditions_fail_before_scoring(self):
  model=SimpleNamespace(training=False,conditioner=SimpleNamespace(embedders={'band_gap':None}))
  for values in ([0.,1.],[float('nan'),0.],[float('inf'),float('inf')],[0.]):
   with self.assertRaisesRegex(ValueError,'finite and identical'):per_structure_energy(model,SimpleNamespace(band_gap=torch.tensor(values)),{})
 def test_training_mode_rejected(self):
  with self.assertRaisesRegex(ValueError,'eval mode'):per_structure_energy(SimpleNamespace(training=True),None,{})
if __name__=='__main__':unittest.main()
