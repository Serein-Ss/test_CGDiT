"""Same-framework CPU denoising surrogate with shared or missing conditions. Not an exact likelihood."""
import torch
import torch.nn.functional as F
from cgdit.common.data_utils import lattice_params_to_matrix_torch
from cgdit.rl.symmetry_quotient import representative_indices, broadcast_from_representatives
from cgdit.pl_modules.diff_utils.diff_utils import d_log_p_wrapped_normal
from cgdit.pl_modules.diff_utils.discrete_diff_utils import compute_kl_reverse_process
from cgdit.pl_modules.training_utils.diffusion_loss import aggregate_per_sample


def paired_corruption(model, batch, site_map, timestep, seed):
    """Explicit shared noise for two equal-framework graphs, in mapped site order."""
    if batch.num_graphs != 2 or batch.num_atoms[0] != batch.num_atoms[1]:
        raise ValueError('Requires two equal-size framework endpoints')
    if not 1 <= timestep <= model.beta_scheduler.timesteps:
        raise ValueError('Timestep outside scheduler')
    n=int(batch.num_atoms[0]); site_map=torch.as_tensor(site_map,device=batch.batch.device)
    if sorted(site_map.tolist()) != list(range(n)):
        raise ValueError('Site mapping must be a permutation')
    if batch.spacegroup[0] != batch.spacegroup[1]:
        raise ValueError('Space groups differ')
    delta=batch.frac_coords[:n]-batch.frac_coords[n:][site_map]
    if not torch.allclose(delta-delta.round(),torch.zeros_like(delta),atol=1e-6):
        raise ValueError('Site mapping does not align the fractional framework')
    a=batch.anchor_index[:n];b=batch.anchor_index[n:][site_map]
    if not torch.equal(a[:,None]==a[None,:],b[:,None]==b[None,:]):
        raise ValueError('Site mapping does not preserve orbits')
    if batch.batch.device.type != 'cpu':
        raise ValueError('This acceptance adapter is CPU-only')
    generator=torch.Generator().manual_seed(seed)
    times=torch.full((2,),timestep,dtype=torch.long)
    lattice=model.crystal_family.de_so3(lattice_params_to_matrix_torch(batch.lengths,batch.angles))
    clean=model.crystal_family.proj_k_to_spacegroup(model.crystal_family.m2v(lattice),batch.spacegroup)
    lattice_noise=torch.randn((1,6),generator=generator).repeat(2,1)
    lattice_noise=model.crystal_family.proj_k_to_spacegroup(lattice_noise,batch.spacegroup)
    alpha=model.beta_scheduler.alphas_cumprod[times,None]
    noisy_lattice=model.crystal_family.proj_k_to_spacegroup(alpha.sqrt()*clean+(1-alpha).sqrt()*lattice_noise,batch.spacegroup)
    noise=torch.randn((n,3),generator=generator);full_noise=torch.empty((2*n,3))
    full_noise[:n]=noise;full_noise[n+site_map]=noise
    anchor_noise=(batch.ops_inv[batch.anchor_index]@full_noise[batch.anchor_index].unsqueeze(-1)).squeeze(-1)
    coord_noise=(batch.ops[:,:3,:3]@anchor_noise.unsqueeze(-1)).squeeze(-1)
    sigma=model.sigma_scheduler.sigmas[times].repeat_interleave(batch.num_atoms)[:,None]
    norm=model.sigma_scheduler.sigmas_norm[times].repeat_interleave(batch.num_atoms)[:,None]
    x0=batch.atom_types.long()-1
    reps=representative_indices(batch.anchor_index)
    per_node_t=times[batch.batch]
    probs=model.d3pm.get_qt_given_q0(x0[reps],per_node_t[reps],make_one_hot=True)
    uniforms=torch.rand(n,generator=generator)[a];full_u=torch.empty(2*n)
    full_u[:n]=uniforms;full_u[n+site_map]=uniforms
    # MaskDiffusion has support only on the clean species and the mask.
    masked=full_u[reps] < probs[:,-1]
    xt_rep=torch.where(masked,model.mask_token_id,x0[reps])
    xt=broadcast_from_representatives(xt_rep,batch.anchor_index)
    return dict(times=times,lattice=noisy_lattice,coords=(batch.frac_coords+sigma*coord_noise)%1,
                atoms=xt,lattice_target=lattice_noise,anchor_noise=anchor_noise,sigma=sigma,norm=norm,
                x0=x0,discrete_t=per_node_t-1)


def per_structure_energy(model,batch,corruption):
    """Return project-normalized lattice, coordinate, and orbit-mean D3PM terms."""
    if model.training:
        raise ValueError('Scorer requires eval mode for deterministic corruption comparisons')
    for name in model.conditioner.embedders:
        if hasattr(batch,name):
            values=getattr(batch,name)
            if not torch.isfinite(values).all() or values.shape[0]!=2 or not torch.equal(values[0],values[1]):
                raise ValueError('Paired conditions must be finite and identical')
    c=corruption
    time_emb=model.time_embedding(c['times'])+model.conditioner(batch)
    lattice,coord,logits=model.decoder(time_emb,c['atoms'],c['coords'],c['lattice'],batch.num_atoms,batch.batch)
    lattice=model.crystal_family.proj_k_to_spacegroup(lattice,batch.spacegroup)
    lat=((lattice-c['lattice_target'])**2).mean(-1)
    projected=torch.einsum('bij,bj->bi',batch.ops_inv,coord)
    target=d_log_p_wrapped_normal(c['sigma']*c['anchor_noise'],c['sigma'])/c['norm'].sqrt()
    xyz=aggregate_per_sample((projected-target)**2,batch.batch,'mean',batch.num_graphs)
    atom=atom_energy_from_logits(model,batch,c,logits)
    total=model.loss_fn.cost_lattice*lat+model.loss_fn.cost_coord*xyz+model.loss_fn.cost_atom*atom
    return dict(total=total,lattice=lat,coordinate=xyz,atom=atom),dict(
        pred_crys_fam=lattice,rand_crys_fam=c['lattice_target'],pred_x=coord,batch=batch,
        rand_x_anchor=c['anchor_noise'],sigmas_per_atom=c['sigma'],sigmas_norm_per_atom=c['norm'],
        pred_atom_logits=logits,x_start_atoms=c['x0'],input_atom_types=c['atoms'],t_discrete=c['discrete_t'])


def atom_energy_from_logits(model,batch,c,logits):
    """Original orbit-normalized atom energy, also used with frozen cached features."""
    reps=representative_indices(batch.anchor_index)
    padded=F.pad(logits[reps],(0,1),value=-1e9) if logits.shape[-1]==model.d3pm.dim-1 else logits[reps]
    metrics=compute_kl_reverse_process(x_start=c['x0'][reps],t=c['discrete_t'][reps],
        x_t_plus_1=c['atoms'][reps],diffusion=model.d3pm,denoise_fn=lambda targets,timestep:padded,
        predict_x0=True,log_space=True,hybrid_lambda=model.loss_fn.hybrid_lambda,use_cached_transition=False)
    atom=aggregate_per_sample(metrics['loss'],batch.batch[reps],'mean',batch.num_graphs)
    return atom


def preference_loss(policy_energy,reference_energy,beta=1.):
    if policy_energy.shape != (2,) or reference_energy.shape != (2,):
        raise ValueError('Expected winner/loser energy vectors of length two')
    if beta <= 0:
        raise ValueError('beta must be positive')
    delta=policy_energy-reference_energy.detach()
    return F.softplus(beta*(delta[0]-delta[1]))
