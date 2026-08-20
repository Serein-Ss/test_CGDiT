"""Measure-correct transition probabilities on CGDiT independent variables."""

import math

import torch
import torch.nn.functional as F

from .symmetry_quotient import representative_indices


def aggregate_by_graph(
    values: torch.Tensor, batch_index: torch.Tensor, num_graphs: int
) -> torch.Tensor:
    """Sum independent-variable contributions for every crystal."""
    result = values.new_zeros(num_graphs)
    return result.scatter_add(0, batch_index, values)


def categorical_orbit_log_prob(
    logits: torch.Tensor,
    actions: torch.Tensor,
    anchor_index: torch.Tensor,
    batch_index: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    """Categorical log probability counted once per symmetry orbit."""
    representatives = representative_indices(anchor_index)
    if logits.shape[0] == anchor_index.numel():
        logits = logits[representatives]
    if actions.shape[0] == anchor_index.numel():
        actions = actions[representatives]
    if logits.shape[0] != representatives.numel():
        raise ValueError("logits must contain either full-cell or representative rows")
    row_log_prob = F.log_softmax(logits, dim=-1).gather(
        -1, actions.long().unsqueeze(-1)
    ).squeeze(-1)
    return aggregate_by_graph(
        row_log_prob, batch_index[representatives], num_graphs
    )


def subspace_normal_log_prob(
    value: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor | float,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Gaussian density with respect to the active lattice subspace measure."""
    std = torch.as_tensor(std, dtype=value.dtype, device=value.device)
    if torch.any(std <= 0):
        raise ValueError("std must be positive for a stochastic transition")
    while std.ndim < value.ndim:
        std = std.unsqueeze(-1)
    log_prob = -0.5 * ((value - mean) / std).square()
    log_prob = log_prob - torch.log(std) - 0.5 * math.log(2 * math.pi)
    return (log_prob * active_mask.to(log_prob.dtype)).sum(dim=-1)


def wrapped_normal_log_prob(
    value: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor | float,
    num_images: int = 3,
) -> torch.Tensor:
    """Diagonal wrapped-normal density on a unit torus.

    The finite image sum is stable for the noise range used by the sampler.
    ``num_images`` is exposed for convergence checks, not as a training knob.
    """
    if num_images < 0:
        raise ValueError("num_images must be non-negative")
    std = torch.as_tensor(std, dtype=value.dtype, device=value.device)
    if torch.any(std <= 0):
        raise ValueError("std must be positive for a stochastic transition")
    while std.ndim < value.ndim:
        std = std.unsqueeze(-1)
    shifts = torch.arange(
        -num_images, num_images + 1, dtype=value.dtype, device=value.device
    )
    delta = value.unsqueeze(-1) - mean.unsqueeze(-1) + shifts
    image_log_prob = -0.5 * (delta / std.unsqueeze(-1)).square()
    image_log_prob = (
        image_log_prob
        - torch.log(std).unsqueeze(-1)
        - 0.5 * math.log(2 * math.pi)
    )
    return torch.logsumexp(image_log_prob, dim=-1).sum(dim=-1)


def coordinate_orbit_log_prob(
    value: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor | float,
    anchor_index: torch.Tensor,
    batch_index: torch.Tensor,
    num_graphs: int,
    num_images: int = 3,
) -> torch.Tensor:
    """Wrapped coordinate density counted once per orbit representative."""
    representatives = representative_indices(anchor_index)
    row_log_prob = wrapped_normal_log_prob(
        value[representatives], mean[representatives], std, num_images=num_images
    )
    return aggregate_by_graph(
        row_log_prob, batch_index[representatives], num_graphs
    )


def joint_log_prob(log_prob_by_channel: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the exact independent-variable joint log probability."""
    if not log_prob_by_channel:
        raise ValueError("at least one probability channel is required")
    return torch.stack(tuple(log_prob_by_channel.values()), dim=0).sum(dim=0)


def importance_ratio(
    current_log_prob: torch.Tensor, old_log_prob: torch.Tensor, max_log_ratio: float = 20.0
) -> torch.Tensor:
    """Exponentiate a numerically bounded log importance ratio."""
    return torch.exp((current_log_prob - old_log_prob).clamp(-max_log_ratio, max_log_ratio))
