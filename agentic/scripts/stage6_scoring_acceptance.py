"""Verify paired scoring, original-loss equivalence, gradients and base checkpoint."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent));os.environ.setdefault('PROJECT_ROOT',str(ROOT.parent))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    out=ROOT/'results/stage6';out.mkdir(parents=True,exist_ok=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        import torch
        from omegaconf import OmegaConf,DictConfig,ListConfig
        from omegaconf.base import Metadata,ContainerMetadata
        from omegaconf.nodes import AnyNode
        from collections import defaultdict
        from typing import Any
        from torch_geometric.data import Batch
        from cgdit.pl_data.dataset import CrystDataset
        from cgdit.pl_modules.diffusion import Diffusion
        from pair_denoising import paired_corruption,per_structure_energy,preference_loss
        torch.set_num_threads(1);torch.manual_seed(20260914)
        cache=out/'symmetry_graphs.pt'
        if cache.exists():cache.unlink()
        ds=CrystDataset(name='Stage6 score acceptance',path=str(ROOT/'data/curation/stage4_pair/endpoints.csv'),
            save_path=str(cache),prop='tc',prop_list=['tc'],niggli=False,primitive=False,
            graph_method='crystalnn',preprocess_workers=1,lattice_scale_method='scale_length',tolerance=.01,
            use_space_group=True,use_pos_index=False)
        batch=Batch.from_data_list([ds[i] for i in range(2)])
        mapping=json.loads((ROOT/'results/stage5/pair_site_alignment.json').read_text())['winner_to_loser_site']
        config=OmegaConf.create(json.loads((ROOT/'results/stage5/smoke_config.json').read_text()))
        policy=Diffusion(**config).eval();reference=copy.deepcopy(policy).eval().requires_grad_(False)
        parameters={k:v.detach().clone() for k,v in policy.named_parameters()}
        checks=[]
        for timestep in (1,10,20):
            corruption=paired_corruption(policy,batch,mapping,timestep,20260914+timestep)
            replay=paired_corruption(policy,batch,mapping,timestep,20260914+timestep)
            assert all(torch.equal(corruption[k],replay[k]) for k in corruption)
            n=5
            assert torch.equal(corruption['atoms'][:n]==100,(corruption['atoms'][n:]==100)[mapping])
            energies,args=per_structure_energy(policy,batch,corruption)
            assert all(v.shape==(2,) and torch.isfinite(v).all() for v in energies.values())
            original=policy.loss_fn(**args)
            for a,b in [('total','loss'),('lattice','loss_lattice'),('coordinate','loss_coord'),('atom','loss_atom_types')]:
                torch.testing.assert_close(energies[a].mean(),original[b],rtol=1e-5,atol=1e-6)
            with torch.no_grad(): ref,_=per_structure_energy(reference,batch,corruption)
            torch.testing.assert_close(energies['total'],ref['total'],rtol=0,atol=0)
            loss=preference_loss(energies['total'],ref['total'])
            torch.testing.assert_close(loss,torch.tensor(2.).log())
            policy.zero_grad();loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
            assert policy.decoder.type_out.weight.grad.norm()>0
            assert all(p.grad is None for p in reference.parameters())
            checks.append(dict(timestep=timestep,energy={k:v.detach().tolist() for k,v in energies.items()},
                original_loss_mean_equal=True,replay_identical=True,mask_coupling_verified=True,
                preference_loss=float(loss.detach()),atom_preference_gradient_norm=float(policy.decoder.type_out.weight.grad.norm())))
            if timestep==10:
                (out/'explicit_corruption.json').write_text(json.dumps({k:v.tolist() for k,v in corruption.items()},indent=2))
        assert all(torch.equal(parameters[k],v) for k,v in policy.named_parameters())
        # Numerical direction and label-swap checks independent of the neural network.
        values=torch.tensor([2.,3.],requires_grad=True)
        baseline=torch.tensor([2.,3.],requires_grad=True)
        preference_loss(values,baseline).backward()
        assert values.grad[0]>0 and values.grad[1]<0 and baseline.grad is None
        improved=preference_loss(torch.tensor([1.9,3.1]),torch.tensor([2.,3.]))
        assert improved < torch.tensor(2.).log()
        invalid=[]
        for bad in ([0,0,1,2,4],[0,1,2,3,4]):
            try:paired_corruption(policy,batch,bad,10,42)
            except ValueError:invalid.append(bad)
        assert len(invalid)==2
        result=dict(passed=True,checks=checks,optimizer_steps=0,weights_unchanged=True,
            reference_gradients_none=True,invalid_mappings_rejected=len(invalid),preference_direction_correct=True,
            scope='Same 5-site framework; unconditional CPU; original-loss-normalized surrogate; not exact DPO')
        (out/'scoring_acceptance.json').write_text(json.dumps(result,indent=2))
        # The runtime configuration names this original MP20 base, not an RL/property predictor.
        base=ROOT.parent/'output/singlerun/2026-06-27/00-32-50-mp20_base'
        path=base/'epoch=869-step=737760.ckpt'
        audit=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),
                   hparams_sha256=sha(base/'hparams.yaml'),pretraining_dataset='mp20',tc_trained=False,
                   static_globals=torch.serialization.get_unsafe_globals_in_checkpoint(path))
        safe_types=[Metadata,ContainerMetadata,ListConfig,DictConfig,AnyNode,Any,int,dict,list,defaultdict]
        with torch.serialization.safe_globals(safe_types):
            payload=torch.load(path,map_location='cpu',weights_only=True)
        hp=OmegaConf.load(base/'hparams.yaml')
        base_config=OmegaConf.to_container(hp.model,resolve=True)
        base_config.pop('_target_',None)
        base_model=Diffusion(**OmegaConf.create(base_config)).eval().requires_grad_(False)
        mismatch=base_model.load_state_dict(payload['state_dict'],strict=True)
        audit.update(strict_load=True,missing_keys=list(mismatch.missing_keys),unexpected_keys=list(mismatch.unexpected_keys),
                     epoch=payload.get('epoch'),global_step=payload.get('global_step'),loading='weights_only with explicit OmegaConf/builtin allowlist')
        with torch.no_grad():
            base_scores=[]
            for t in (1,500,1000):
                c=paired_corruption(base_model,batch,mapping,t,20260914+t)
                e,args=per_structure_energy(base_model,batch,c)
                assert all(torch.isfinite(v).all() for v in e.values())
                torch.testing.assert_close(e['total'].mean(),base_model.loss_fn(**args)['loss'],rtol=1e-5,atol=1e-6)
                base_scores.append(dict(timestep=t,energy={k:v.tolist() for k,v in e.items()}))
        audit.update(finite_pair_scoring=True,scores=base_scores,scientific_tc_validation=False,
            note='Compatible general crystal generator only; denoising energy is not predicted Tc or high-Tc evidence')
        (out/'checkpoint_audit.json').write_text(json.dumps(audit,indent=2))
    (out/'warnings.json').write_text(json.dumps(sorted(set(str(w.message) for w in caught)),indent=2))
    print(json.dumps(dict(scoring_passed=True,strict_checkpoint_load=True,checkpoint_finite_scores=True,
                         optimizer_steps=0,checkpoint=str(path)),indent=2))

if __name__=='__main__':main()
