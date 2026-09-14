"""Small full-diffusion generation check on a fixed 5-site antiperovskite framework."""
import argparse,copy,json,time
from pathlib import Path
import torch
from torch_geometric.data import Batch
from pymatgen.core import Structure,Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pilot30_finetune import ROOT,R,load_model,sha,write
from pilot30_graphs import graphs

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['base','band_gap','both'],default='both');args=parser.parse_args()
 torch.set_num_threads(2);g=graphs();out=R/'generation';out.mkdir(exist_ok=True)
 records=json.loads((out/'samples.json').read_text()) if (out/'samples.json').exists() else []
 for arm in (('base','band_gap') if args.arm=='both' else (args.arm,)):
  records=[r for r in records if r['arm']!=arm]
  model,audit=load_model(arm);head=model.decoder.type_out;initial=copy.deepcopy(head.state_dict())
  path=R/arm/'true_labels_seed42/adapter.pt';adapter=torch.load(path,map_location='cpu',weights_only=True)
  assert adapter['base_checkpoint_sha256']==audit['sha256'];assert adapter['control']=='true_labels'
  conditions=[None,None] if arm=='base' else [0.,0.,1.,1.,2.,2.]
  batch=Batch.from_data_list([g['Mn4N'].clone() for _ in conditions])
  if arm=='band_gap':batch.band_gap=torch.tensor(conditions)
  for version in ('original','pilot_seed42'):
   head.load_state_dict(initial if version=='original' else adapter['head_state_dict'],strict=True)
   started=time.time()
   with torch.no_grad():final,_=model.sample(batch,diff_ratio=1.,guidance_scale=1.,noise_seed=91442,retain_trajectory_stack=False)
   torch.save(final,out/f'{arm}_{version}_raw.pt');offset=0
   for i,cond in enumerate(conditions):
    n=int(final['num_atoms'][i]);atoms=final['atom_types'][offset:offset+n]+1;xyz=final['frac_coords'][offset:offset+n];lat=final['lattices'][i];offset+=n
    row=dict(arm=arm,version=version,sample_index=i,target_band_gap_ev=cond,noise_seed=91442,diffusion_steps=model.beta_scheduler.timesteps-1,nominal_scheduler_steps=model.beta_scheduler.timesteps,prototype_constraint='5-site Pm-3m antiperovskite',fixed_atom_types=False,tc_k=None,validated_band_gap_ev=None)
    finite=bool(torch.isfinite(xyz).all() and torch.isfinite(lat).all());types_ok=bool(((atoms>=1)&(atoms<=100)).all());row.update(finite=finite,valid_atomic_numbers=types_ok)
    if finite and types_ok:
     s=Structure(Lattice(lat.numpy()),atoms.tolist(),xyz.numpy());distance=min(nei.nn_distance for neighbors in s.get_all_neighbors(min(5.,s.lattice.a)) for nei in neighbors) if .5<s.volume/n<1000 else None
     valid=bool(s.is_valid() and .5<s.volume/n<1000 and distance is not None and distance>.7)
     cif=out/f'{arm}_{version}_{i}.cif';s.to(filename=str(cif),fmt='cif')
     row.update(formula=s.composition.reduced_formula,volume_per_atom=s.volume/n,min_distance_angstrom=distance,geometric_validity=valid,spacegroup=SpacegroupAnalyzer(s,.1).get_space_group_number(),cif=str(cif.relative_to(ROOT)))
    else:row['geometric_validity']=False
    records.append(row)
   write(out/'samples.json',records);print(arm,version,'generated',len(conditions),'seconds',round(time.time()-started,2),flush=True)
 write(out/'summary.json',dict(samples=len(records),groups=[dict(arm=arm,version=v,n=sum(r['arm']==arm and r['version']==v for r in records),geometrically_valid=sum(r['arm']==arm and r['version']==v and r['geometric_validity'] for r in records)) for arm in ('base','band_gap') for v in ('original','pilot_seed42')],chemical_stability_validated=False,magnetic_order_validated=False,tc_validated=False,band_gap_validated=False,note='Full sampler trajectory (1000-step scheduler, 999 reverse iterations) on a supplied prototype, not unrestricted de novo search; conditions are requested values only.'))
if __name__=='__main__':main()
