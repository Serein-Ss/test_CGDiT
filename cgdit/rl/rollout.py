"""Differentiable probability replay for recorded CGDiT transitions."""

import torch
from torch_scatter import scatter

from .symmetry_quotient import lattice_active_mask, representative_indices
from .trajectory import RLTrajectory, RLTransition
from .transition_logprob import (
    categorical_orbit_log_prob,
    coordinate_orbit_log_prob,
    subspace_normal_log_prob,
)


def recompute_transition_log_probs(
    model,
    batch,
    transition: RLTransition,
    step_lr: float,
    guidance_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Recompute one recorded action under ``model`` with gradients enabled."""
    batch_size = batch.num_graphs
    if not transition.stochastic:
        zero = transition.state_t.crys_fam.new_zeros(batch_size)
        return {"lattice": zero, "coord": zero, "atom": zero}

    t = transition.timestep
    times = torch.full((batch_size,), t, device=transition.state_t.crys_fam.device)
    time_emb = model.time_embedding(times)
    sigma_norm = model.sigma_scheduler.sigmas_norm[t]

    _, pred_x, _ = model._get_model_output(
        time_emb,
        transition.state_t.atom_types,
        transition.state_t.frac_coords,
        transition.state_t.crys_fam,
        batch.num_atoms,
        batch.batch,
        batch_obj=batch,
        guidance_scale=guidance_scale,
    )
    pred_x = pred_x * torch.sqrt(sigma_norm)
    pred_x_proj = torch.einsum("bij,bj->bi", batch.ops_inv, pred_x)
    pred_x_anchor = scatter(
        pred_x_proj, batch.anchor_index, dim=0, reduce="mean"
    )[batch.anchor_index]
    pred_x = (batch.ops[:, :3, :3] @ pred_x_anchor.unsqueeze(-1)).squeeze(-1)

    corrector_step = step_lr / (
        sigma_norm * model.sigma_scheduler.sigma_begin ** 2
    )
    corrector_mean = transition.state_t.frac_coords - corrector_step * pred_x
    corrector_std = torch.sqrt(2 * corrector_step)
    coord_log_prob = coordinate_orbit_log_prob(
        transition.state_half.frac_coords,
        corrector_mean,
        corrector_std,
        batch.anchor_index,
        batch.batch,
        batch_size,
    )

    pred_crys_fam, pred_x, pred_atom_logits = model._get_model_output(
        time_emb,
        transition.state_half.atom_types,
        transition.state_half.frac_coords,
        transition.state_half.crys_fam,
        batch.num_atoms,
        batch.batch,
        batch_obj=batch,
        guidance_scale=guidance_scale,
    )
    pred_x = pred_x * torch.sqrt(sigma_norm)
    pred_x_proj = torch.einsum("bij,bj->bi", batch.ops_inv, pred_x)
    pred_x_anchor = scatter(
        pred_x_proj, batch.anchor_index, dim=0, reduce="mean"
    )[batch.anchor_index]
    pred_x = (batch.ops[:, :3, :3] @ pred_x_anchor.unsqueeze(-1)).squeeze(-1)

    sigma_x = model.sigma_scheduler.sigmas[t]
    adjacent_sigma_x = model.sigma_scheduler.sigmas[t - 1]
    predictor_step = sigma_x.square() - adjacent_sigma_x.square()
    predictor_std = torch.sqrt(
        adjacent_sigma_x.square() * predictor_step / sigma_x.square()
    )
    predictor_mean = transition.state_half.frac_coords - predictor_step * pred_x
    coord_log_prob = coord_log_prob + coordinate_orbit_log_prob(
        transition.state_next.frac_coords,
        predictor_mean,
        predictor_std,
        batch.anchor_index,
        batch.batch,
        batch_size,
    )

    alpha = model.beta_scheduler.alphas[t]
    alpha_cumprod = model.beta_scheduler.alphas_cumprod[t]
    c0 = 1.0 / torch.sqrt(alpha)
    c1 = (1 - alpha) / torch.sqrt(1 - alpha_cumprod)
    lattice_mean = c0 * (
        transition.state_t.crys_fam - c1 * pred_crys_fam
    )
    lattice_mean = model.crystal_family.proj_k_to_spacegroup(
        lattice_mean, batch.spacegroup
    )
    lattice_log_prob = subspace_normal_log_prob(
        transition.state_next.crys_fam,
        lattice_mean,
        model.beta_scheduler.sigmas[t],
        lattice_active_mask(model.crystal_family, batch.spacegroup),
    )

    representatives = representative_indices(batch.anchor_index)
    pred_x0_logits = pred_atom_logits[
        representatives, :model.num_atom_types
    ]
    full_logits = pred_x0_logits.new_full(
        (representatives.numel(), model.num_atom_types + 1), -1e9
    )
    full_logits[:, :model.num_atom_types] = pred_x0_logits
    posterior_logits, _ = model.d3pm.sample_and_compute_posterior_q(
        x_0=full_logits.softmax(dim=-1),
        t=torch.full(
            (representatives.numel(),), t - 1,
            device=representatives.device,
            dtype=torch.long,
        ),
        samples=transition.state_half.atom_types[representatives].long(),
        make_one_hot=False,
        return_logits=True,
        step_size=1,
    )
    atom_log_prob = categorical_orbit_log_prob(
        posterior_logits,
        transition.state_next.atom_types,
        batch.anchor_index,
        batch.batch,
        batch_size,
    )
    return {
        "lattice": lattice_log_prob,
        "coord": coord_log_prob,
        "atom": atom_log_prob,
    }


def recompute_trajectory_log_probs(
    model,
    batch,
    trajectory: RLTrajectory,
    step_lr: float,
    guidance_scale: float = 1.0,
    transition_indices: list[int] | torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Recompute all recorded transitions and stack them as [T, B]."""
    transitions = trajectory.transitions
    if transition_indices is not None:
        indices = torch.as_tensor(transition_indices).tolist()
        transitions = [trajectory.transitions[index] for index in indices]
    per_transition = [
        recompute_transition_log_probs(
            model, batch, transition, step_lr, guidance_scale
        )
        for transition in transitions
    ]
    if not per_transition:
        return {}
    return {
        channel: torch.stack([item[channel] for item in per_transition])
        for channel in per_transition[0]
    }
