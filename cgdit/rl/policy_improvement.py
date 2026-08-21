"""Algorithm-agnostic paired policy-improvement reliability layer (PIRL)."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ImprovementDecision:
    action: str
    scale: float
    mean_delta: torch.Tensor
    lower_confidence_bound: torch.Tensor


def paired_bootstrap_lcb(
    old_metrics: torch.Tensor,
    new_metrics: torch.Tensor,
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return paired mean improvement and its bootstrap lower bound."""
    if old_metrics.shape != new_metrics.shape or old_metrics.ndim != 2:
        raise ValueError("old_metrics and new_metrics must share shape [N, M]")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1")
    delta = new_metrics - old_metrics
    generator = torch.Generator(device=delta.device).manual_seed(seed)
    indices = torch.randint(
        delta.shape[0],
        (n_bootstrap, delta.shape[0]),
        generator=generator,
        device=delta.device,
    )
    bootstrap_means = delta[indices].mean(dim=1)
    lcb = torch.quantile(bootstrap_means, 1.0 - confidence, dim=0)
    return delta.mean(dim=0), lcb


def decide_policy_improvement(
    old_metrics: torch.Tensor,
    new_metrics: torch.Tensor,
    primary_metric: int = 0,
    safety_tolerances: dict[int, float] | None = None,
    minimum_improvement: float = 0.0,
    attenuation: float = 0.25,
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> ImprovementDecision:
    """Accept, attenuate or reject an update using paired probe evidence."""
    mean_delta, lcb = paired_bootstrap_lcb(
        old_metrics, new_metrics, confidence, n_bootstrap, seed
    )
    safety_tolerances = safety_tolerances or {}
    safe = all(lcb[index] >= -tolerance for index, tolerance in safety_tolerances.items())
    if safe and lcb[primary_metric] > minimum_improvement:
        action, scale = "accept", 1.0
    elif safe and mean_delta[primary_metric] > minimum_improvement:
        action, scale = "attenuate", attenuation
    else:
        action, scale = "reject", 0.0
    return ImprovementDecision(action, scale, mean_delta, lcb)


def decide_reward_improvement(
    old_reward: torch.Tensor,
    new_reward: torch.Tensor,
    old_safety_metrics: dict[str, torch.Tensor] | None = None,
    new_safety_metrics: dict[str, torch.Tensor] | None = None,
    safety_tolerances: dict[str, float] | None = None,
    **decision_kwargs,
) -> ImprovementDecision:
    """Apply the paired policy gate to fixed-scale raw rewards and safety metrics."""
    if old_reward.ndim != 1 or old_reward.shape != new_reward.shape:
        raise ValueError("old_reward and new_reward must share shape [N]")
    old_safety_metrics = old_safety_metrics or {}
    new_safety_metrics = new_safety_metrics or {}
    safety_tolerances = safety_tolerances or {}
    if set(old_safety_metrics) != set(new_safety_metrics):
        raise ValueError("old and new safety metrics must use the same names")
    if not set(safety_tolerances).issubset(old_safety_metrics):
        raise ValueError("safety tolerances must refer to supplied safety metrics")

    names = tuple(old_safety_metrics)
    old_columns = [old_reward]
    new_columns = [new_reward]
    for name in names:
        old_metric = old_safety_metrics[name]
        new_metric = new_safety_metrics[name]
        if old_metric.shape != old_reward.shape or new_metric.shape != new_reward.shape:
            raise ValueError(f"safety metric '{name}' must align with the rewards")
        old_columns.append(old_metric)
        new_columns.append(new_metric)

    indexed_tolerances = {
        names.index(name) + 1: tolerance
        for name, tolerance in safety_tolerances.items()
    }
    return decide_policy_improvement(
        torch.stack(old_columns, dim=1),
        torch.stack(new_columns, dim=1),
        primary_metric=0,
        safety_tolerances=indexed_tolerances,
        **decision_kwargs,
    )
