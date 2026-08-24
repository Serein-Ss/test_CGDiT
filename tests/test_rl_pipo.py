import pytest
import torch

from cgdit.rl.pipo import normalize_group_attributions, pipo_feedback


def test_pipo_feedback_uses_history_before_the_current_batch():
    feedback = pipo_feedback(
        current_mean=4.0,
        history=[1.0, 3.0],
        negative_scale=0.1,
    )

    assert feedback is not None
    assert feedback.history_mean == pytest.approx(2.0)
    assert feedback.history_std == pytest.approx(2.0 ** 0.5)
    assert feedback.signal == pytest.approx(2.0 / (2.0 ** 0.5 + 1.0e-6))
    assert feedback.modulation == pytest.approx(feedback.signal)


def test_pipo_negative_feedback_is_scaled_and_reverses_previous_advantage():
    feedback = pipo_feedback(
        current_mean=0.0,
        history=[1.0, 3.0],
        negative_scale=0.1,
    )

    assert feedback is not None
    assert feedback.signal < 0
    assert feedback.modulation == pytest.approx(0.1 * feedback.signal)
    advantages = torch.tensor([-1.0, 2.0])
    assert torch.allclose(
        advantages * feedback.modulation,
        torch.tensor([-feedback.modulation, 2.0 * feedback.modulation]),
    )


def test_pipo_feedback_requires_two_historical_batches():
    assert pipo_feedback(1.0, []) is None
    assert pipo_feedback(1.0, [0.5]) is None


def test_pipo_group_attributions_follow_paper_normalization():
    normalized = normalize_group_attributions(
        torch.tensor([-1.0, 1.0, -1.0, 3.0]),
        torch.tensor([0, 0, 1, 1]),
    )

    assert torch.allclose(normalized, torch.tensor([-1.0, 1.0, -0.5, 1.5]))


def test_pipo_group_attributions_keep_degenerate_group_at_zero():
    normalized = normalize_group_attributions(
        torch.zeros(4),
        torch.tensor([0, 0, 1, 1]),
    )

    assert torch.equal(normalized, torch.zeros(4))
