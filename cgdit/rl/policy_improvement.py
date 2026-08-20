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
