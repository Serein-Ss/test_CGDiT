"""CPU-only actual diffusion interface/gradient check, with no optimizer step."""
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
os.environ.setdefault('PROJECT_ROOT',str(ROOT.parent))


def main():
    out=ROOT/'results/stage5';out.mkdir(parents=True,exist_ok=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        import torch
        from omegaconf import OmegaConf
        from torch_geometric.data import Batch
        from cgdit.pl_data.dataset import CrystDataset
        from cgdit.pl_modules.diffusion import Diffusion
        from cgdit.rl.symmetry_quotient import representative_indices
        torch.set_num_threads(1);torch.manual_seed(20260914)
        cache=out/'symmetry_graphs.pt'
        if cache.exists():cache.unlink()
        ds=CrystDataset(name='Stage5 symmetry smoke only',path=str(ROOT/'data/curation/stage4_pair/endpoints.csv'),
            save_path=str(cache),prop='tc',prop_list=['tc'],niggli=False,primitive=False,
            graph_method='crystalnn',preprocess_workers=1,lattice_scale_method='scale_length',
            tolerance=0.01,use_space_group=True,use_pos_index=False)
        graphs=[ds[i] for i in range(len(ds))];batch=Batch.from_data_list(graphs)
        assert batch.spacegroup.tolist()==[221,221]
        assert torch.equal(batch.batch[batch.anchor_index],batch.batch)
        assert torch.equal(batch.atom_types[batch.anchor_index],batch.atom_types)
        reps=representative_indices(batch.anchor_index)
        # Both cells share the same fractional framework, but pyxtal changes site order.
        delta=graphs[0].frac_coords[:,None,:]-graphs[1].frac_coords[None,:,:]
        distance=(delta-delta.round()).norm(dim=-1)
        site_map=distance.argmin(dim=1)
        assert site_map.unique().numel()==5
        assert distance[torch.arange(5),site_map].max()<1e-6
        for i in range(5):
            for j in range(5):
                assert bool(graphs[0].anchor_index[i]==graphs[0].anchor_index[j]) == bool(
                    graphs[1].anchor_index[site_map[i]]==graphs[1].anchor_index[site_map[j]])
        (out/'pair_site_alignment.json').write_text(json.dumps(dict(
            winner_to_loser_site=site_map.tolist(), basis='fractional framework position, intentionally species-independent',
            use='Noise/position alignment only; NEVER a chemical duplicate match',
            winner_atomic_numbers=graphs[0].atom_types.tolist(),
            aligned_loser_atomic_numbers=graphs[1].atom_types[site_map].tolist()),indent=2))
        config=dict(time_dim=32,cost_coord=1.,cost_lattice=1.,cost_atom=1.,conditions={},cond_dropout_prob=0.,
            decoder=dict(_target_='cgdit.pl_modules.decoder.cspnet.CSPNet',hidden_dim=32,num_layers=1,
                         max_atoms=100,num_freqs=8,edge_style='fc',ln=True,ip=True,pred_type=True),
            beta_scheduler=dict(_target_='cgdit.pl_modules.diff_utils.diff_utils.BetaScheduler',timesteps=20,scheduler_mode='cosine'),
            sigma_scheduler=dict(_target_='cgdit.pl_modules.diff_utils.diff_utils.SigmaScheduler',timesteps=20,sigma_begin=.005,sigma_end=.5),
            d3pm_scheduler=dict(kind='standard',beta_min=.0001,beta_max=.02))
        (out/'smoke_config.json').write_text(json.dumps(config,indent=2))
        model=Diffusion(**OmegaConf.create(config));model.train()
        before={k:v.detach().clone() for k,v in model.named_parameters()}
        output=model(batch)
        assert all(v.ndim==0 and torch.isfinite(v) for v in output.values())
        output['loss'].backward()
        gradients={name:float(getattr(model.decoder,name).weight.grad.norm()) for name in ['type_out','coord_out','lattice_out']}
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert gradients['type_out']>0
        assert all(torch.equal(before[k],v) for k,v in model.named_parameters())
        # Verify the existing loss returns a scalar, not a per-endpoint preference vector.
        assert output['loss_atom_types'].ndim==0
        per_graph=[]
        for g in graphs:
            anchors=g.anchor_index
            per_graph.append(dict(atom_types=g.atom_types.tolist(),spacegroup=int(g.spacegroup.item()),
                anchors=anchors.tolist(),orbit_sizes=[int((anchors==i).sum()) for i in representative_indices(anchors)],
                operations_shape=list(g.ops.shape),
                operation_linear_ranks=torch.linalg.matrix_rank(g.ops[:,:3,:3]).tolist()))
    sources=['cgdit/pl_modules/diffusion.py','cgdit/pl_modules/training_utils/diffusion_loss.py',
             'cgdit/pl_data/dataset.py','cgdit/rl/symmetry_quotient.py']
    report=dict(passed=True,scope='Random small-model CPU forward/backward only; no training or preference objective',
        seed=20260914,graphs=per_graph,batch_anchor_index=batch.anchor_index.tolist(),total_orbits=len(reps),
        loss_values_random_model={k:float(v.detach()) for k,v in output.items()},gradient_norms=gradients,
        weights_unchanged=True,optimizer_steps=0,tc_condition_used=False,api_calls=0,
        source_hashes={p:hashlib.sha256((ROOT.parent/p).read_bytes()).hexdigest() for p in sources},
        warnings=sorted(set(str(w.message) for w in caught)))
    (out/'diffusion_smoke.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='warnings'},indent=2))

if __name__=='__main__':main()
