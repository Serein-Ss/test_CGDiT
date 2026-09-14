"""Matched-corruption condition-response probes; not electronic-property validation."""
import copy,json
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from pilot30_finetune import ROOT,R,load_model,write
from pilot30_graphs import graphs,alignment
from pair_denoising import paired_corruption,per_structure_energy
from cgdit.rl.symmetry_quotient import representative_indices

def main():
 torch.set_num_threads(1);g=graphs();result=[]
 pairs=[('GdFe2','TbAl2'),('Co2FeSi','Co2TiSn'),('Mn4N','Mn3AlC')]
 for arm in ('base','band_gap'):
  model,audit=load_model(arm);initial=copy.deepcopy(model.decoder.type_out.state_dict());adapter=torch.load(R/arm/'true_labels_seed42/adapter.pt',map_location='cpu',weights_only=True);assert adapter['base_checkpoint_sha256']==audit['sha256']
  for winner,loser in pairs:
   b=Batch.from_data_list([g[winner],g[loser]]);c=paired_corruption(model,b,alignment(g[winner],g[loser]),500,91914);refs={};condition_vectors={}
   with torch.no_grad():
    for cond in (0.,1.,2.):
     b.band_gap=torch.tensor([cond,cond]);model.decoder.type_out.load_state_dict(initial);_,args=per_structure_energy(model,b,c);before=args['pred_atom_logits'];reps=representative_indices(b.anchor_index);refs[cond]=before.clone();embedding=model.conditioner(b).clone()
     model.decoder.type_out.load_state_dict(adapter['head_state_dict']);_,args=per_structure_energy(model,b,c);after=args['pred_atom_logits'];assert torch.equal(embedding,model.conditioner(b))
     kl=F.kl_div(after[reps].log_softmax(-1),before[reps].softmax(-1),reduction='batchmean')
     result.append(dict(arm=arm,pair=f'{winner}__{loser}',target_band_gap_ev=cond,reference_to_pilot_kl=float(kl),logit_rmse=float((after-before).square().mean().sqrt()),condition_embedding_unchanged=True))
    if arm=='base':assert all(torch.equal(refs[0.],refs[c]) for c in (1.,2.))
    else:assert all(not torch.equal(refs[0.],refs[c]) for c in (1.,2.))
 write(R/'condition_retention.json',dict(probes=result,identical_corruption_across_conditions=True,base_ignores_gap_verified=True,conditional_model_responds_to_gap_verified=True,conditioner_unchanged=True,band_gap_prediction_or_realization_validated=False))
 print('Condition response and frozen conditioner checks passed',len(result),'probes')
if __name__=='__main__':main()
