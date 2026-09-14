"""Frozen-backbone atom-head preference pilot; caches exact features to reduce CPU cost."""
import copy,hashlib,json,os,random,sys,time
from collections import defaultdict
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT.parent));os.environ.setdefault('PROJECT_ROOT',str(ROOT.parent))
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf,DictConfig,ListConfig
from omegaconf.base import Metadata,ContainerMetadata
from omegaconf.nodes import AnyNode
from torch_geometric.data import Batch
from cgdit.pl_modules.diffusion import Diffusion
from cgdit.rl.symmetry_quotient import representative_indices
from pair_denoising import paired_corruption,per_structure_energy,atom_energy_from_logits,preference_loss
from pilot30_graphs import graphs,alignment
R=ROOT/'results/pilot30_v1';D=ROOT/'data/pilot30_v1'
MODELS={'base':'2026-06-27/00-32-50-mp20_base/epoch=869-step=737760.ckpt','band_gap':'2026-06-30/11-28-44-mp20_bg/epoch=784-step=665680.ckpt'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,x):Path(path).write_text(json.dumps(x,ensure_ascii=False,indent=2))

def load_model(arm):
 path=ROOT.parent/'output/singlerun'/MODELS[arm]
 allowed=[Metadata,ContainerMetadata,ListConfig,DictConfig,AnyNode,Any,int,dict,list,defaultdict]
 with torch.serialization.safe_globals(allowed):payload=torch.load(path,map_location='cpu',weights_only=True)
 hp=OmegaConf.load(path.parent/'hparams.yaml');cfg=OmegaConf.to_container(hp.model,resolve=True);cfg.pop('_target_',None)
 model=Diffusion(**OmegaConf.create(cfg)).eval().requires_grad_(False);model.load_state_dict(payload['state_dict'],strict=True)
 audit=dict(path=str(path),sha256=sha(path),hparams_sha256=sha(path.parent/'hparams.yaml'),strict_load=True,epoch=payload.get('epoch'),global_step=payload.get('global_step'),conditions=list(model.conditioner.embedders),parameters=sum(p.numel() for p in model.parameters()))
 return model,audit

def frozen_hash(model):
 h=hashlib.sha256()
 for name,p in model.named_parameters():
  if not name.startswith('decoder.type_out.'):
   h.update(name.encode());h.update(p.detach().numpy().tobytes())
 return h.hexdigest()

def cache_case(model,batch,mapping,t,seed):
 c=paired_corruption(model,batch,mapping,t,seed);captured=[]
 hook=model.decoder.type_out.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
 with torch.no_grad():energy,args=per_structure_energy(model,batch,c)
 hook.remove();features=captured[0];logits=model.decoder.type_out(features).detach()
 cached=atom_energy_from_logits(model,batch,c,logits)
 torch.testing.assert_close(cached,energy['atom'],rtol=1e-5,atol=1e-7)
 torch.testing.assert_close(energy['total'].mean(),model.loss_fn(**args)['loss'],rtol=2e-5,atol=1e-6)
 assert all(torch.isfinite(v).all() for v in energy.values())
 return dict(features=features,reference_logits=logits,reference_atom=energy['atom'],reference_total=energy['total'],batch=batch,corruption=c,timestep=t,seed=seed)

def score(model,case):
 logits=model.decoder.type_out(case['features']);atom=atom_energy_from_logits(model,case['batch'],case['corruption'],logits)
 total=case['reference_total']+model.loss_fn.cost_atom*(atom-case['reference_atom'])
 reps=representative_indices(case['batch'].anchor_index)
 kl=F.kl_div(logits[reps].log_softmax(-1),case['reference_logits'][reps].softmax(-1),reduction='batchmean')
 return total,kl

def evaluate(model,pairs,caches,beta):
 records=[]
 with torch.no_grad():
  for p in pairs:
   values=[]
   for c in caches[p['pair_id']]:
    energy,kl=score(model,c);base=c['reference_total'];delta=energy-base
    values.append((float(energy[1]-energy[0]),float(base[1]-base[0]),float(beta*(delta[1]-delta[0])),float(kl)))
   mean=lambda i:sum(v[i] for v in values)/len(values)
   records.append(dict(pair_id=p['pair_id'],split=p['split'],source_groups=p['source_groups'],absolute_margin=mean(0),reference_absolute_margin=mean(1),relative_margin_beta=mean(2),mean_kl=mean(3),noise_realizations=len(values)))
 summaries={}
 for split in ('train','validation','test','electrical_control'):
  group=[r for r in records if r['split']==split]
  summaries[split]=dict(pairs=len(group),absolute_order_accuracy=sum(r['absolute_margin']>0 for r in group)/len(group),reference_absolute_order_accuracy=sum(r['reference_absolute_margin']>0 for r in group)/len(group),fraction_moving_toward_label=sum(r['relative_margin_beta']>1e-6 for r in group)/len(group),mean_relative_margin_beta=sum(r['relative_margin_beta'] for r in group)/len(group),mean_kl=sum(r['mean_kl'] for r in group)/len(group))
 return dict(summary=summaries,pairs=records)

def main():
 torch.set_num_threads(1);torch.manual_seed(20260914);start=time.time()
 assert json.loads((R/'intern/summary.json').read_text())['records_passed']==30,'Intern extraction checks must complete first'
 design=json.loads((R/'design.json').read_text());pairs=json.loads((D/'pairs.json').read_text());materials={r['material_id']:r for r in json.loads((D/'materials.json').read_text())};g=graphs()
 manifest=dict(design_sha256=sha(R/'design.json'),materials_sha256=sha(D/'materials.json'),pairs_sha256=sha(D/'pairs.json'),arms={})
 # A deliberately simple composition confound baseline; ties score 0.5.
 from pymatgen.core import Composition
 baseline={}
 for split in ('train','validation','test','electrical_control'):
  ps=[p for p in pairs if p['split']==split];v=[]
  for p in ps:
   a=Composition(p['winner']).get_atomic_fraction('Fe');b=Composition(p['loser']).get_atomic_fraction('Fe');v.append(1. if a>b else .5 if a==b else 0.)
  baseline[split]=dict(n=len(v),iron_fraction_pair_accuracy_with_half_credit_ties=sum(v)/len(v))
 write(R/'composition_baseline.json',baseline)
 for arm in MODELS:
  out=R/arm;out.mkdir(exist_ok=True);model,audit=load_model(arm);head=model.decoder.type_out;initial=copy.deepcopy(head.state_dict());freeze_before=frozen_hash(model);train_cache={};eval_cache={};conditions={}
  for i,p in enumerate(pairs):
   batch=Batch.from_data_list([g[p['winner']],g[p['loser']]]);mapping=alignment(g[p['winner']],g[p['loser']]);a,b=materials[p['winner']],materials[p['loser']]
   cond=None
   if arm=='band_gap' and p['split']!='electrical_control' and a['local_mp_band_gap_ev']==b['local_mp_band_gap_ev']==0.:
    batch.band_gap=torch.zeros(2);cond=0.
   conditions[p['pair_id']]=cond
   if p['split']=='train':train_cache[p['pair_id']]=[cache_case(model,batch,mapping,t,100000+100*i+10*j+k) for j,t in enumerate(design['timesteps']) for k in range(design['noise_replicates_train'])]
   eval_cache[p['pair_id']]=[cache_case(model,batch,mapping,t,200000+100*i+10*j+k) for j,t in enumerate(design['timesteps']) for k in range(design['noise_replicates_eval'])]
   if i%10==0:print(arm,'cached',i+1,'/50',flush=True)
  write(out/'checkpoint_audit.json',audit);write(out/'pair_conditions.json',conditions)
  runs=[]
  for seed,control in [(s,'true_labels') for s in design['seeds']]+[(42,'shuffled_labels'),(42,'exclude_Co2FeSi')]:
   run=out/f'{control}_seed{seed}';run.mkdir(exist_ok=True);head.load_state_dict(initial);head.requires_grad_(True);rng=random.Random(seed);torch.manual_seed(seed)
   selected=[p for p in pairs if p['split']=='train' and not(control=='exclude_Co2FeSi' and 'Co2FeSi' in (p['winner'],p['loser']))]
   group_pairs=defaultdict(list)
   for p in selected:group_pairs['+'.join(p['source_groups'])].append(p)
   # Fixed pairwise random reversals, not resampled contradictory labels every step.
   flips={p['pair_id']:(-1 if control=='shuffled_labels' and rng.random()<.5 else 1) for p in selected}
   optimizer=torch.optim.Adam(head.parameters(),lr=design['learning_rate']);trace=[]
   for step in range(design['steps_per_run']):
    optimizer.zero_grad();losses=[];prefs=[];kls=[]
    for group in sorted(group_pairs):
     p=rng.choice(group_pairs[group]);case=rng.choice(train_cache[p['pair_id']]);energy,kl=score(model,case);ref=case['reference_total']
     if flips[p['pair_id']]<0:energy=energy.flip(0);ref=ref.flip(0)
     pref=preference_loss(energy,ref,beta=design['beta']);losses.append(pref+design['reference_kl_weight']*kl);prefs.append(float(pref.detach()));kls.append(float(kl.detach()))
    loss=torch.stack(losses).mean();assert torch.isfinite(loss);loss.backward()
    assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('decoder.type_out.'))
    grad=torch.nn.utils.clip_grad_norm_(head.parameters(),design['max_gradient_norm']);assert torch.isfinite(grad);optimizer.step()
    trace.append(dict(step=step+1,loss=float(loss.detach()),preference=sum(prefs)/len(prefs),kl=sum(kls)/len(kls),gradient_norm=float(grad)))
   head.requires_grad_(False);assert frozen_hash(model)==freeze_before
   changes={k:float((v-initial[k]).norm()) for k,v in head.state_dict().items()};assert any(v>0 for v in changes.values())
   # Post-update cached scores must equal a fresh full network forward on the same corruption.
   first=eval_cache[pairs[0]['pair_id']][0]
   with torch.no_grad():fresh,_=per_structure_energy(model,first['batch'],first['corruption']);cached,_=score(model,first)
   torch.testing.assert_close(fresh['total'],cached,rtol=2e-5,atol=2e-6)
   evaluation=evaluate(model,pairs,eval_cache,design['beta']);write(run/'evaluation.json',evaluation);write(run/'training_trace.json',trace)
   # A small delta adapter references the unchanged source checkpoint; no duplicate 148 MB checkpoint.
   adapter=dict(format='cgdit_type_head_replacement_v1',base_checkpoint_sha256=audit['sha256'],base_checkpoint=audit['path'],head_state_dict={k:v.detach().clone() for k,v in head.state_dict().items()},optimizer_steps=design['steps_per_run'],seed=seed,control=control,design_sha256=manifest['design_sha256'])
   torch.save(adapter,run/'adapter.pt');write(run/'label_directions.json',flips)
   entry=dict(seed=seed,control=control,steps=design['steps_per_run'],training_pairs=len(selected),training_groups=len(group_pairs),adapter=str((run/'adapter.pt').relative_to(ROOT)),adapter_sha256=sha(run/'adapter.pt'),parameter_change_norms=changes,frozen_parameters_unchanged=True,cached_scores_equal_full_network=True,summary=evaluation['summary'])
   runs.append(entry);write(out/'runs.json',runs);print(arm,control,seed,'steps',design['steps_per_run'],'train',evaluation['summary']['train'],flush=True)
  manifest['arms'][arm]=dict(checkpoint=audit,runs=runs,trainable_parameter_count=sum(p.numel() for p in head.parameters()),frozen_parameter_sha256=freeze_before)
  write(R/'training_manifest.json',manifest)
 manifest.update(elapsed_seconds=time.time()-start,optimizer_steps_total=sum(r['steps'] for a in manifest['arms'].values() for r in a['runs']),scientific_discovery_validated=False)
 write(R/'training_manifest.json',manifest);print('Training completed',manifest['optimizer_steps_total'],'steps',flush=True)
if __name__=='__main__':main()
