"""Bounded, uncertainty-aware rewards for general scalar properties."""

import torch


def property_reward(
    prediction: torch.Tensor,
    mode: str,
    target: float | None = None,
    tolerance: float = 1.0,
    lower: float | None = None,
    upper: float | None = None,
) -> torch.Tensor:
    """Map maximize/minimize/target/range objectives to a bounded [0, 1] score."""
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if mode in {"maximize", "minimize", "target"} and target is None:
        raise ValueError(f"{mode} mode requires target")
    if mode == "maximize":
        return torch.sigmoid((prediction - target) / tolerance)
    if mode == "minimize":
        return torch.sigmoid((target - prediction) / tolerance)
    if mode == "target":
        return torch.exp(-0.5 * ((prediction - target) / tolerance).square())
    if mode == "range":
        if lower is None or upper is None or lower >= upper:
            raise ValueError("range mode requires lower < upper")
        return torch.sigmoid((prediction - lower) / tolerance) * torch.sigmoid(
            (upper - prediction) / tolerance
        )
    raise ValueError("mode must be maximize, minimize, target or range")


def robust_validity_gated_reward(
    reward: torch.Tensor,
    valid: torch.Tensor,
    uncertainty: torch.Tensor | None = None,
    uncertainty_penalty: float = 0.0,
) -> torch.Tensor:
    """Penalize predictor uncertainty and hard-gate invalid structures."""
    if valid.shape != reward.shape:
        raise ValueError("valid and reward must have the same shape")
    if uncertainty is not None:
        if uncertainty.shape != reward.shape:
            raise ValueError("uncertainty and reward must have the same shape")
        reward = reward - uncertainty_penalty * uncertainty
    return torch.where(valid, reward, torch.zeros_like(reward))
