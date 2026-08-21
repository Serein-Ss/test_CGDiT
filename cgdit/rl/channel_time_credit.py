"""Explicit channel/time credit for terminal crystal rewards (H2)."""

import torch

from .transition_logprob import importance_ratio


def temporal_difference_scores(intermediate_rewards: torch.Tensor) -> torch.Tensor:
    """Convert K+1 intermediate rewards to K denoising-bucket increments."""
    if intermediate_rewards.ndim != 2 or intermediate_rewards.shape[1] < 2:
        raise ValueError("intermediate_rewards must have shape [B, K+1]")
    return intermediate_rewards[:, 1:] - intermediate_rewards[:, :-1]


def counterfactual_channel_scores(
    full_rewards: torch.Tensor, counterfactual_rewards: torch.Tensor
) -> torch.Tensor:
    """Estimate channel contributions as full minus channel-replaced reward."""
    if counterfactual_rewards.ndim != 3:
        raise ValueError("counterfactual_rewards must have shape [B, K, C]")
    if full_rewards.shape != counterfactual_rewards.shape[:2]:
        raise ValueError("full and counterfactual reward shapes are incompatible")
    return full_rewards.unsqueeze(-1) - counterfactual_rewards


def normalized_channel_time_credit(
    terminal_advantages: torch.Tensor,
    contribution_scores: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Distribute each terminal advantage without changing its overall scale."""
    if contribution_scores.ndim != 3:
        raise ValueError("contribution_scores must have shape [B, K, C]")
    if terminal_advantages.shape != contribution_scores.shape[:1]:
        raise ValueError("one terminal advantage is required per structure")
    denominator = contribution_scores.abs().sum(dim=(1, 2), keepdim=True)
    uniform = torch.full_like(
        contribution_scores,
        1.0 / (contribution_scores.shape[1] * contribution_scores.shape[2]),
    )
    weights = torch.where(
        denominator > eps,
        contribution_scores / denominator.clamp_min(eps),
        uniform,
    )
    return terminal_advantages[:, None, None] * weights


def channel_time_objective(
    current_log_prob_by_channel: dict[str, torch.Tensor],
    old_log_prob_by_channel: dict[str, torch.Tensor],
    credits: torch.Tensor,
    channels: tuple[str, ...] = ("lattice", "coord", "atom"),
    clip_epsilon: float = 0.2,
) -> torch.Tensor:
    """Apply independently clipped ratios to [T, B, C] credit values."""
    if credits.ndim != 3 or credits.shape[-1] != len(channels):
        raise ValueError("credits must have shape [T, B, number_of_channels]")
    losses = []
    for channel_index, channel in enumerate(channels):
        ratio = importance_ratio(
            current_log_prob_by_channel[channel], old_log_prob_by_channel[channel]
        )
        advantage = credits[..., channel_index]
        unclipped = ratio * advantage
        clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage
        losses.append(-torch.minimum(unclipped, clipped).mean())
    return torch.stack(losses).sum()
