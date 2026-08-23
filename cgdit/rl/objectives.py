"""PPO and GRPO objectives over recorded diffusion transition measures."""

from dataclasses import dataclass

import torch

from .transition_logprob import importance_ratio


@dataclass
class PolicyObjectiveResult:
    loss: torch.Tensor
    ratio: torch.Tensor
    advantages: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor


def group_relative_advantages(
    rewards: torch.Tensor, group_index: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Standardize rewards independently inside every prompt/condition group."""
    if rewards.ndim != 1 or group_index.shape != rewards.shape:
        raise ValueError("rewards and group_index must be one-dimensional and aligned")
    advantages = torch.empty_like(rewards)
    for group in torch.unique(group_index):
        mask = group_index == group
        group_rewards = rewards[mask]
        advantages[mask] = (
            group_rewards - group_rewards.mean()
        ) / (group_rewards.std(unbiased=False) + eps)
    return advantages


def _joint_channel_log_prob(log_prob_by_channel: dict[str, torch.Tensor]) -> torch.Tensor:
    if not log_prob_by_channel:
        raise ValueError("at least one log-probability channel is required")
    return torch.stack(tuple(log_prob_by_channel.values()), dim=0).sum(dim=0)


def clipped_policy_objective(
    current_log_prob_by_channel: dict[str, torch.Tensor],
    old_log_prob_by_channel: dict[str, torch.Tensor],
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
    pirl_scale: float = 1.0,
) -> PolicyObjectiveResult:
    """Compute the clipped objective using the exact joint policy ratio."""
    current = _joint_channel_log_prob(current_log_prob_by_channel)
    old = _joint_channel_log_prob(old_log_prob_by_channel)
    if current.shape != old.shape:
        raise ValueError("current and old log probabilities must have the same shape")
    if advantages.ndim == 1 and current.ndim == 2:
        advantages = advantages.unsqueeze(0).expand_as(current)
    if advantages.shape != current.shape:
        raise ValueError("advantages must be [B] or match the transition log-probability shape")

    advantages = advantages * pirl_scale
    ratio = importance_ratio(current, old)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantages
    loss = -torch.minimum(unclipped, clipped).mean()
    return PolicyObjectiveResult(
        loss=loss,
        ratio=ratio,
        advantages=advantages,
        approx_kl=(old - current).mean(),
        clip_fraction=((ratio - 1).abs() > clip_epsilon).float().mean(),
    )


def ppo_objective(
    current_log_prob_by_channel: dict[str, torch.Tensor],
    old_log_prob_by_channel: dict[str, torch.Tensor],
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
    pirl_scale: float = 1.0,
) -> PolicyObjectiveResult:
    """PPO objective with externally estimated advantages."""
    return clipped_policy_objective(
        current_log_prob_by_channel,
        old_log_prob_by_channel,
        advantages,
        clip_epsilon,
        pirl_scale,
    )


def grpo_objective(
    current_log_prob_by_channel: dict[str, torch.Tensor],
    old_log_prob_by_channel: dict[str, torch.Tensor],
    rewards: torch.Tensor,
    group_index: torch.Tensor,
    clip_epsilon: float = 0.2,
    pirl_scale: float = 1.0,
    advantages: torch.Tensor | None = None,
) -> PolicyObjectiveResult:
    """GRPO objective using group-relative terminal rewards."""
    if advantages is None:
        advantages = group_relative_advantages(rewards, group_index)
    return clipped_policy_objective(
        current_log_prob_by_channel,
        old_log_prob_by_channel,
        advantages,
        clip_epsilon,
        pirl_scale,
    )
