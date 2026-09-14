"""Construct and validate prototype graphs for the restricted pilot."""
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT.parent));os.environ.setdefault('PROJECT_ROOT',str(ROOT.parent))
import torch
from cgdit.pl_data.dataset import CrystDataset

def graphs():
 out=ROOT/'results/pilot30_v1';rows=json.loads((ROOT/'data/pilot30_v1/materials.json').read_text())
 ds=CrystDataset(name='prototype-level Tc pilot30',path=str(ROOT/'data/pilot30_v1/endpoints.csv'),save_path=str(out/'graphs.pt'),prop='tc',prop_list=['tc'],niggli=False,primitive=False,graph_method='crystalnn',preprocess_workers=1,lattice_scale_method='scale_length',tolerance=.01,use_space_group=True,use_pos_index=False)
 result={r['material_id']:ds[i] for i,r in enumerate(rows)}
 audit=[]
 for r in rows:
  g=result[r['material_id']];assert len(g.atom_types)<=100
  assert int(g.spacegroup)==r['spacegroup'];assert torch.equal(g.atom_types[g.anchor_index],g.atom_types)
  audit.append(dict(material_id=r['material_id'],natoms=len(g.atom_types),spacegroup=int(g.spacegroup),orbits=int(g.anchor_index.unique().numel())))
 (out/'graph_audit.json').write_text(json.dumps(audit,indent=2));return result

def alignment(a,b):
 distance=a.frac_coords[:,None,:]-b.frac_coords[None,:,:];distance=(distance-distance.round()).norm(dim=-1);mapping=distance.argmin(dim=1)
 if mapping.unique().numel()!=len(a.atom_types) or distance[torch.arange(len(mapping)),mapping].max()>1e-6:raise ValueError('Frameworks need a verified origin/basis alignment')
 return mapping.tolist()

if __name__=='__main__':
 torch.set_num_threads(1);g=graphs();pairs=json.loads((ROOT/'data/pilot30_v1/pairs.json').read_text());a={p['pair_id']:alignment(g[p['winner']],g[p['loser']]) for p in pairs};(ROOT/'results/pilot30_v1/alignments.json').write_text(json.dumps(a,indent=2));print('Aligned',len(a),'pairs')
