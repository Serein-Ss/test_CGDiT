"""Budgeted predictor -> MLFF -> DFT verification orchestration."""

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import torch


@dataclass
class FidelityEvaluation:
    score: torch.Tensor
    uncertainty: torch.Tensor
    valid: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.score.ndim != 1:
            raise ValueError("score must be one-dimensional")
        if self.uncertainty.shape != self.score.shape or self.valid.shape != self.score.shape:
            raise ValueError("score, uncertainty and valid must have identical shapes")


class FidelityEvaluator(Protocol):
    def evaluate(self, candidates: Sequence[Any]) -> FidelityEvaluation:
        ...


class PredictorEnsembleEvaluator:
    """Convert independent predictor callables into mean/uncertainty rewards."""

    def __init__(self, predictors, score_transform=None):
        if not predictors:
            raise ValueError("at least one predictor is required")
        self.predictors = list(predictors)
        self.score_transform = score_transform

    def evaluate(self, candidates: Sequence[Any]) -> FidelityEvaluation:
        predictions = torch.stack([
            torch.as_tensor(predictor(candidates), dtype=torch.float)
            for predictor in self.predictors
        ])
        if predictions.ndim != 2 or predictions.shape[1] != len(candidates):
            raise ValueError("every predictor must return one scalar per candidate")
        if self.score_transform is not None:
            predictions = self.score_transform(predictions)
        valid = torch.isfinite(predictions).all(dim=0)
        safe_predictions = torch.nan_to_num(predictions)
        return FidelityEvaluation(
            score=safe_predictions.mean(dim=0),
            uncertainty=safe_predictions.std(dim=0, unbiased=False),
            valid=valid,
            metadata={"ensemble_predictions": predictions},
        )


@dataclass
class MultiFidelityTrace:
    proxy: FidelityEvaluation
    mlff_indices: torch.Tensor
    mlff: FidelityEvaluation | None
    dft_indices: torch.Tensor
    dft: FidelityEvaluation | None


def _budgeted_selection(
    evaluation: FidelityEvaluation,
    budget: int,
    exploration_fraction: float,
) -> torch.Tensor:
    """Select a deterministic mixture of high-value and high-uncertainty items."""
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if not 0.0 <= exploration_fraction <= 1.0:
        raise ValueError("exploration_fraction must be in [0, 1]")
    valid_indices = torch.nonzero(evaluation.valid, as_tuple=False).flatten()
    budget = min(budget, valid_indices.numel())
    if budget == 0:
        return valid_indices[:0]

    explore_count = min(round(budget * exploration_fraction), budget)
    exploit_count = budget - explore_count
    score_order = valid_indices[
        torch.argsort(evaluation.score[valid_indices], descending=True)
    ]
    selected = score_order[:exploit_count].tolist()
    if len(selected) >= budget:
        return torch.tensor(selected, dtype=torch.long, device=evaluation.score.device)
    uncertainty_order = valid_indices[
        torch.argsort(evaluation.uncertainty[valid_indices], descending=True)
    ]
    for index in uncertainty_order.tolist():
        if index not in selected:
            selected.append(index)
        if len(selected) >= budget:
            break
    return torch.tensor(selected, dtype=torch.long, device=evaluation.score.device)


class MultiFidelityVerifier:
    """Run available fidelity stages while enforcing explicit stage budgets."""

    def __init__(
        self,
        predictor: FidelityEvaluator,
        mlff: FidelityEvaluator | None = None,
        dft: FidelityEvaluator | None = None,
    ):
        self.predictor = predictor
        self.mlff = mlff
        self.dft = dft

    def run(
        self,
        candidates: Sequence[Any],
        mlff_budget: int,
        dft_budget: int,
        exploration_fraction: float = 0.2,
    ) -> MultiFidelityTrace:
        proxy_result = self.predictor.evaluate(candidates)
        mlff_indices = _budgeted_selection(
            proxy_result, mlff_budget, exploration_fraction
        )
        mlff_result = None
        if self.mlff is not None and mlff_indices.numel():
            mlff_result = self.mlff.evaluate([candidates[i] for i in mlff_indices.tolist()])

        dft_source = mlff_result if mlff_result is not None else proxy_result
        local_dft_indices = _budgeted_selection(
            dft_source, dft_budget, exploration_fraction
        )
        if mlff_result is not None:
            dft_indices = mlff_indices[local_dft_indices]
        else:
            dft_indices = local_dft_indices

        dft_result = None
        if self.dft is not None and dft_indices.numel():
            dft_result = self.dft.evaluate([candidates[i] for i in dft_indices.tolist()])
        return MultiFidelityTrace(
            proxy=proxy_result,
            mlff_indices=mlff_indices,
            mlff=mlff_result,
            dft_indices=dft_indices,
            dft=dft_result,
        )


def calibrated_reward(
    proxy_score: torch.Tensor, physics_score: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Interpolate proxy reward toward matched higher-fidelity evidence."""
    if proxy_score.shape != physics_score.shape:
        raise ValueError("proxy and physics scores must have the same shape")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return proxy_score + alpha * (physics_score - proxy_score)


def inverse_propensity_weights(
    selection_probability: torch.Tensor, minimum_probability: float = 1e-3
) -> torch.Tensor:
    """Weights for an externally randomized high-fidelity selection policy."""
    if torch.any(selection_probability <= 0) or torch.any(selection_probability > 1):
        raise ValueError("selection probabilities must lie in (0, 1]")
    return selection_probability.clamp_min(minimum_probability).reciprocal()
