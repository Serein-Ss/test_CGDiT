import pytest
import torch

from cgdit.rl.channel_time_credit import (
    channel_time_objective,
    normalized_channel_time_credit,
    temporal_difference_scores,
)
from cgdit.rl.objectives import grpo_objective, group_relative_advantages, ppo_objective
from cgdit.rl.policy_improvement import (
    decide_dual_baseline_reward_improvement,
    decide_policy_improvement,
    decide_reward_improvement,
)


def make_log_probs(offset=0.0):
    return {
        "lattice": torch.tensor([[0.0, 0.0, 0.0, 0.0]]) + offset,
        "coord": torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
        "atom": torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
    }


def test_group_relative_advantages_are_centered_per_condition():
    rewards = torch.tensor([1.0, 3.0, 10.0, 14.0])
    groups = torch.tensor([0, 0, 1, 1])
    advantages = group_relative_advantages(rewards, groups)
    assert torch.allclose(advantages[:2].mean(), torch.tensor(0.0))
    assert torch.allclose(advantages[2:].mean(), torch.tensor(0.0))


def test_pirl_scale_applies_after_both_ppo_and_grpo_advantage_estimation():
    current = make_log_probs(0.1)
    old = make_log_probs()
    ppo_full = ppo_objective(current, old, torch.tensor([1.0, -1.0, 2.0, -2.0]))
    ppo_half = ppo_objective(
        current, old, torch.tensor([1.0, -1.0, 2.0, -2.0]), pirl_scale=0.5
    )
    grpo_full = grpo_objective(
        current, old, torch.tensor([1.0, 2.0, 4.0, 8.0]), torch.tensor([0, 0, 1, 1])
    )
    grpo_half = grpo_objective(
        current,
        old,
        torch.tensor([1.0, 2.0, 4.0, 8.0]),
        torch.tensor([0, 0, 1, 1]),
        pirl_scale=0.5,
    )
    assert torch.allclose(ppo_half.loss, 0.5 * ppo_full.loss)
    assert torch.allclose(grpo_half.loss, 0.5 * grpo_full.loss)

def test_grpo_accepts_precomputed_retrospective_advantages():
    old = make_log_probs()
    current = make_log_probs()
    current["lattice"] = torch.tensor([[0.2, 0.0, -0.1, 0.3]])
    rewards = torch.ones(4)
    groups = torch.tensor([0, 0, 1, 1])
    explicit_advantages = torch.tensor([1.0, -1.0, -0.5, 0.5])

    computed = grpo_objective(current, old, rewards, groups)
    explicit = grpo_objective(
        current, old, rewards, groups, advantages=explicit_advantages
    )

    assert computed.loss == pytest.approx(0.0)
    assert torch.equal(explicit.advantages, explicit_advantages.unsqueeze(0))
    assert explicit.loss != pytest.approx(0.0)


def test_channel_time_credit_and_objective_preserve_explicit_channels():
    intermediate = torch.tensor([[0.0, 1.0, 3.0], [0.0, -1.0, 1.0]])
    temporal = temporal_difference_scores(intermediate)
    contributions = temporal.unsqueeze(-1).repeat(1, 1, 3)
    credits = normalized_channel_time_credit(torch.tensor([2.0, -1.0]), contributions)
    assert credits.shape == (2, 2, 3)
    assert torch.allclose(credits[0].abs().sum(), torch.tensor(2.0))

    log_probs = {key: torch.zeros((2, 2)) for key in ("lattice", "coord", "atom")}
    loss = channel_time_objective(log_probs, log_probs, credits.permute(1, 0, 2))
    assert torch.isfinite(loss)


def test_paired_pirl_gate_accepts_safe_improvement_and_rejects_regression():
    old = torch.zeros((16, 2))
    improved = torch.tensor([1.0, 0.1]).repeat(16, 1)
    regressed = torch.tensor([-1.0, 0.1]).repeat(16, 1)
    accepted = decide_policy_improvement(
        old, improved, safety_tolerances={1: 0.0}, n_bootstrap=100, seed=7
    )
    rejected = decide_policy_improvement(
        old, regressed, safety_tolerances={1: 0.0}, n_bootstrap=100, seed=7
    )
    assert accepted.action == "accept" and accepted.scale == 1.0
    assert rejected.action == "reject" and rejected.scale == 0.0


def test_closed_loop_reward_gate_uses_raw_reward_and_named_safety_metrics():
    old_reward = torch.zeros(16)
    new_reward = torch.ones(16)
    old_safety = {"validity": torch.ones(16)}
    new_safety = {"validity": torch.full((16,), 0.8)}
    decision = decide_reward_improvement(
        old_reward,
        new_reward,
        old_safety,
        new_safety,
        safety_tolerances={"validity": 0.1},
        n_bootstrap=100,
        seed=7,
    )
    assert decision.action == "reject"


def test_dual_baseline_gate_rejects_local_gain_below_frozen_base():
    base = torch.ones(32)
    current = torch.zeros(32)
    candidate = torch.full((32,), 0.5)

    decision = decide_dual_baseline_reward_improvement(
        base_reward=base,
        current_reward=current,
        candidate_reward=candidate,
        n_bootstrap=100,
        seed=7,
    )

    assert decision.action == "reject"
    assert decision.local.lower_confidence_bound[0] > 0
    assert decision.absolute.lower_confidence_bound[0] < 0


def test_dual_baseline_gate_accepts_only_when_both_anchors_pass():
    base = torch.zeros(32)
    current = torch.full((32,), 0.25)
    candidate = torch.ones(32)
    safety = {"validity": torch.ones(32)}

    decision = decide_dual_baseline_reward_improvement(
        base_reward=base,
        current_reward=current,
        candidate_reward=candidate,
        base_safety_metrics=safety,
        current_safety_metrics=safety,
        candidate_safety_metrics=safety,
        safety_tolerances={"validity": 0.0},
        n_bootstrap=100,
        seed=7,
    )

    assert decision.action == "accept"
    assert decision.scale == 1.0
    assert not decision.requires_recheck


def test_dual_baseline_gate_allows_quality_drop_within_base_tolerance():
    base_reward = torch.zeros(32)
    current_reward = torch.full((32,), 0.25)
    candidate_reward = torch.ones(32)
    base_safety = {"diversity": torch.full((32,), 0.80)}
    current_safety = {"diversity": torch.full((32,), 0.95)}
    candidate_safety = {"diversity": torch.full((32,), 0.78)}

    decision = decide_dual_baseline_reward_improvement(
        base_reward=base_reward,
        current_reward=current_reward,
        candidate_reward=candidate_reward,
        base_safety_metrics=base_safety,
        current_safety_metrics=current_safety,
        candidate_safety_metrics=candidate_safety,
        safety_tolerances={"diversity": 0.03},
        n_bootstrap=100,
        seed=7,
    )

    assert decision.action == "accept"
    assert decision.local.lower_confidence_bound.numel() == 1
    assert decision.absolute.lower_confidence_bound[1].item() == pytest.approx(
        -0.02, abs=1.0e-6
    )


def test_dual_baseline_gate_marks_attenuation_as_unverified():
    base = torch.zeros(8)
    current = torch.zeros(8)
    candidate = torch.tensor(
        [-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0]
    )

    decision = decide_dual_baseline_reward_improvement(
        base_reward=base,
        current_reward=current,
        candidate_reward=candidate,
        attenuation=0.25,
        confidence=0.95,
        n_bootstrap=1000,
        seed=7,
    )

    assert decision.action == "attenuate"
    assert decision.scale == 0.25
    assert decision.requires_recheck


@pytest.mark.parametrize(
    "metric",
    [
        "validity",
        "stability",
        "uniqueness",
        "novelty",
        "composition_diversity",
        "structure_diversity",
    ],
)
def test_dual_baseline_gate_rejects_each_safety_regression(metric):
    reward = torch.zeros(32)
    candidate_reward = torch.ones(32)
    safe = {metric: torch.ones(32)}
    regressed = {metric: torch.zeros(32)}

    decision = decide_dual_baseline_reward_improvement(
        base_reward=reward,
        current_reward=reward,
        candidate_reward=candidate_reward,
        base_safety_metrics=safe,
        current_safety_metrics=safe,
        candidate_safety_metrics=regressed,
        safety_tolerances={metric: 0.0},
        n_bootstrap=100,
        seed=7,
    )

    assert decision.action == "reject"
