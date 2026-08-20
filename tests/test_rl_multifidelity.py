import torch

from cgdit.rl.multifidelity import (
    FidelityEvaluation,
    MultiFidelityVerifier,
    PredictorEnsembleEvaluator,
    calibrated_reward,
    inverse_propensity_weights,
)


class FakeEvaluator:
    def __init__(self, scale=1.0):
        self.scale = scale
        self.calls = []

    def evaluate(self, candidates):
        self.calls.append(list(candidates))
        values = torch.tensor(candidates, dtype=torch.float)
        return FidelityEvaluation(
            score=self.scale * values,
            uncertainty=torch.flip(values, dims=(0,)),
            valid=torch.ones(values.shape, dtype=torch.bool),
        )


def test_multifidelity_pipeline_respects_mlff_and_dft_budgets():
    predictor = FakeEvaluator()
    mlff = FakeEvaluator(0.9)
    dft = FakeEvaluator(0.8)
    verifier = MultiFidelityVerifier(predictor, mlff, dft)

    trace = verifier.run(list(range(10)), mlff_budget=5, dft_budget=2, exploration_fraction=0.2)

    assert len(predictor.calls[0]) == 10
    assert len(mlff.calls[0]) == 5
    assert len(dft.calls[0]) == 2
    assert trace.mlff_indices.numel() == 5
    assert trace.dft_indices.numel() == 2


def test_reward_calibration_and_propensity_weights_are_explicit():
    proxy = torch.tensor([1.0, 2.0])
    physics = torch.tensor([3.0, 0.0])
    assert torch.equal(calibrated_reward(proxy, physics, 0.5), torch.tensor([2.0, 1.0]))
    assert torch.equal(
        inverse_propensity_weights(torch.tensor([0.5, 0.25])),
        torch.tensor([2.0, 4.0]),
    )


def test_predictor_ensemble_exposes_mean_and_epistemic_spread():
    ensemble = PredictorEnsembleEvaluator([
        lambda candidates: torch.tensor(candidates, dtype=torch.float),
        lambda candidates: torch.tensor(candidates, dtype=torch.float) + 2.0,
    ])
    result = ensemble.evaluate([1, 3])
    assert torch.equal(result.score, torch.tensor([2.0, 4.0]))
    assert torch.equal(result.uncertainty, torch.tensor([1.0, 1.0]))
