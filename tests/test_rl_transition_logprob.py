import math

import torch

from cgdit.rl.transition_logprob import (
    categorical_orbit_log_prob,
    coordinate_orbit_log_prob,
    importance_ratio,
    subspace_normal_log_prob,
    wrapped_normal_log_prob,
)


def test_categorical_probability_is_invariant_to_orbit_multiplicity():
    logits_small = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    actions_small = torch.tensor([0, 0])
    anchors_small = torch.tensor([0, 0])
    batch_small = torch.zeros(2, dtype=torch.long)

    logits_large = logits_small.repeat(3, 1)
    actions_large = actions_small.repeat(3)
    anchors_large = torch.zeros(6, dtype=torch.long)
    batch_large = torch.zeros(6, dtype=torch.long)

    small = categorical_orbit_log_prob(
        logits_small, actions_small, anchors_small, batch_small, 1
    )
    large = categorical_orbit_log_prob(
        logits_large, actions_large, anchors_large, batch_large, 1
    )
    assert torch.allclose(small, large)


def test_lattice_density_uses_only_active_subspace_and_has_gradients():
    mean = torch.zeros((1, 6), requires_grad=True)
    value = torch.tensor([[1.0, 50.0, 2.0, 50.0, 50.0, 50.0]])
    mask = torch.tensor([[True, False, True, False, False, False]])

    log_prob = subspace_normal_log_prob(value, mean, 2.0, mask)
    expected = -0.5 * (1.0 / 2.0) ** 2 - 0.5 * (2.0 / 2.0) ** 2
    expected -= 2 * math.log(2.0 * math.sqrt(2 * math.pi))
    assert torch.allclose(log_prob, torch.tensor([expected]))
    log_prob.sum().backward()
    assert torch.isfinite(mean.grad).all()
    assert torch.equal(mean.grad[~mask], torch.zeros(4))


def test_wrapped_density_is_periodic_and_orbit_counted_once():
    value = torch.tensor([[0.98, 0.10, 0.20], [0.98, 0.10, 0.20]])
    mean = torch.tensor([[0.02, 0.10, 0.20], [0.02, 0.10, 0.20]])
    shifted = wrapped_normal_log_prob(value + 1.0, mean, 0.1)
    original = wrapped_normal_log_prob(value, mean, 0.1)
    assert torch.allclose(original, shifted, atol=1e-5)

    orbit = coordinate_orbit_log_prob(
        value, mean, 0.1, torch.tensor([0, 0]), torch.zeros(2, dtype=torch.long), 1
    )
    assert torch.allclose(orbit, original[:1])


def test_unchanged_policy_has_unit_importance_ratio():
    log_prob = torch.tensor([-3.0, 2.0])
    assert torch.equal(importance_ratio(log_prob, log_prob), torch.ones(2))
