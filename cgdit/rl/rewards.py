"""Fixed-scale rewards for closed-loop material generation."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MaterialRewardResult:
    """Raw reward and its auditable, pre-normalization components."""

    raw_reward: torch.Tensor
    property_reward: torch.Tensor
    components: dict[str, torch.Tensor]


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
    invalid_penalty: float = 0.0,
) -> torch.Tensor:
    """Penalize predictor uncertainty and hard-gate invalid structures."""
    if valid.shape != reward.shape:
        raise ValueError("valid and reward must have the same shape")
    if invalid_penalty < 0:
        raise ValueError("invalid_penalty must be non-negative")
    if uncertainty is not None:
        if uncertainty.shape != reward.shape:
            raise ValueError("uncertainty and reward must have the same shape")
        reward = reward - uncertainty_penalty * uncertainty
    invalid_reward = torch.full_like(reward, -invalid_penalty)
    return torch.where(valid, reward, invalid_reward)


def continuous_creativity_reward(
    unique: torch.Tensor,
    novel: torch.Tensor,
    boundary_distance: torch.Tensor,
) -> torch.Tensor:
    """Chemeleon2-style continuous reward for unique/novel boundary cases."""
    if unique.shape != novel.shape or unique.shape != boundary_distance.shape:
        raise ValueError("unique, novel and boundary_distance must have identical shapes")
    if unique.dtype != torch.bool or novel.dtype != torch.bool:
        raise ValueError("unique and novel must be boolean tensors")
    both = unique & novel
    neither = ~unique & ~novel
    boundary = boundary_distance.clamp(0.0, 1.0)
    return torch.where(both, torch.ones_like(boundary), torch.where(neither, 0.0, boundary))


def bounded_stability_reward(
    energy_above_hull: torch.Tensor,
    maximum_energy: float = 1.0,
) -> torch.Tensor:
    """Map lower energy above hull to a fixed [0, 1] stability reward."""
    if maximum_energy <= 0:
        raise ValueError("maximum_energy must be positive")
    safe_energy = torch.nan_to_num(
        energy_above_hull,
        nan=maximum_energy,
        posinf=maximum_energy,
        neginf=0.0,
    )
    return 1.0 - safe_energy.clamp(0.0, maximum_energy) / maximum_energy


def leave_one_out_mmd_reward(
    generated_features: torch.Tensor,
    reference_features: torch.Tensor,
    degree: int = 3,
    constant: float = 1.0,
) -> torch.Tensor:
    """Return each generated sample's marginal contribution to reference coverage."""
    if generated_features.ndim != 2 or reference_features.ndim != 2:
        raise ValueError("generated_features and reference_features must be two-dimensional")
    if generated_features.shape[1] != reference_features.shape[1]:
        raise ValueError("generated and reference features must have the same width")
    if generated_features.shape[0] < 3 or reference_features.shape[0] < 2:
        raise ValueError("MMD marginal rewards require at least 3 generated and 2 reference samples")
    if degree <= 0:
        raise ValueError("degree must be positive")

    width = generated_features.shape[1]

    def kernel(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (left @ right.T / width + constant).pow(degree)

    generated_kernel = kernel(generated_features, generated_features)
    cross_kernel = kernel(generated_features, reference_features)
    reference_kernel = kernel(reference_features, reference_features)

    generated_count = generated_features.shape[0]
    reference_count = reference_features.shape[0]
    generated_sum = generated_kernel.sum() - generated_kernel.diagonal().sum()
    cross_sum = cross_kernel.sum()
    reference_term = (
        reference_kernel.sum() - reference_kernel.diagonal().sum()
    ) / (reference_count * (reference_count - 1))
    full_mmd = (
        generated_sum / (generated_count * (generated_count - 1))
        + reference_term
        - 2.0 * cross_sum / (generated_count * reference_count)
    )

    row_generated = generated_kernel.sum(dim=1) - generated_kernel.diagonal()
    row_cross = cross_kernel.sum(dim=1)
    reduced_count = generated_count - 1
    leave_one_out_mmd = (
        (generated_sum - 2.0 * row_generated)
        / (reduced_count * (reduced_count - 1))
        + reference_term
        - 2.0 * (cross_sum - row_cross) / (reduced_count * reference_count)
    )
    return leave_one_out_mmd - full_mmd


def bounded_diversity_reward(
    marginal_utility: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Map MMD marginal utility to [0, 1] using a frozen, non-batch scale."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    return torch.sigmoid(marginal_utility / scale)


def combine_property_rewards(
    property_scores: dict[str, torch.Tensor],
    mode: str = "bottleneck",
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Combine fixed-scale property scores without batch-dependent normalization."""
    if not property_scores:
        raise ValueError("at least one property score is required")
    names = tuple(property_scores)
    scores = torch.stack(tuple(property_scores.values()))
    if not torch.isfinite(scores).all():
        raise ValueError("property scores must be finite")
    if torch.any(scores < 0) or torch.any(scores > 1):
        raise ValueError("property scores must lie in [0, 1]")
    if mode == "bottleneck":
        if weights is not None:
            raise ValueError("bottleneck combination does not use weights")
        return scores.amin(dim=0)
    if mode != "weighted_mean":
        raise ValueError("mode must be 'bottleneck' or 'weighted_mean'")
    weights = weights or {name: 1.0 for name in names}
    if set(weights) != set(names) or any(weight < 0 for weight in weights.values()):
        raise ValueError("weights must provide one non-negative value per property")
    weight_tensor = scores.new_tensor([weights[name] for name in names])
    if weight_tensor.sum() == 0:
        raise ValueError("at least one property weight must be positive")
    view_shape = (len(names),) + (1,) * (scores.ndim - 1)
    return (scores * weight_tensor.view(view_shape)).sum(dim=0) / weight_tensor.sum()


def general_material_reward(
    property_scores: dict[str, torch.Tensor],
    valid: torch.Tensor,
    property_mode: str = "bottleneck",
    property_weights: dict[str, float] | None = None,
    auxiliary_scores: dict[str, torch.Tensor] | None = None,
    auxiliary_weights: dict[str, float] | None = None,
    invalid_penalty: float = 1.0,
) -> MaterialRewardResult:
    """Compose target, validity and optional exploration rewards on a fixed scale."""
    property_score = combine_property_rewards(
        property_scores,
        mode=property_mode,
        weights=property_weights,
    )
    if valid.shape != property_score.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean tensor aligned with the property scores")

    auxiliary_scores = auxiliary_scores or {}
    auxiliary_weights = auxiliary_weights or {}
    if set(auxiliary_scores) != set(auxiliary_weights):
        raise ValueError("auxiliary weights must match the auxiliary score names")

    weighted_reward = property_score.clone()
    total_weight = 1.0
    for name, score in auxiliary_scores.items():
        if score.shape != property_score.shape:
            raise ValueError(f"auxiliary score '{name}' has an incompatible shape")
        if not torch.isfinite(score).all() or torch.any(score < 0) or torch.any(score > 1):
            raise ValueError(f"auxiliary score '{name}' must be finite and lie in [0, 1]")
        weight = auxiliary_weights[name]
        if weight < 0:
            raise ValueError("auxiliary weights must be non-negative")
        weighted_reward = weighted_reward + weight * score
        total_weight += weight

    combined = weighted_reward / total_weight
    raw_reward = robust_validity_gated_reward(
        combined,
        valid,
        invalid_penalty=invalid_penalty,
    )
    return MaterialRewardResult(
        raw_reward=raw_reward,
        property_reward=property_score,
        components={**property_scores, **auxiliary_scores},
    )
