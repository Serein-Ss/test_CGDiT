import torch

import cgdit.pl_modules.training_utils.diffusion_loss as diffusion_loss_module
from cgdit.pl_modules.training_utils.diffusion_loss import DiffusionLoss
from cgdit.rl.rewards import (
    bounded_diversity_reward,
    bounded_stability_reward,
    combine_property_rewards,
    continuous_creativity_reward,
    general_material_reward,
    leave_one_out_mmd_reward,
    property_reward,
    robust_validity_gated_reward,
)


def test_general_property_rewards_support_all_required_objective_types():
    values = torch.tensor([0.0, 1.0, 2.0])
    assert property_reward(values, "maximize", target=1.0)[2] > 0.5
    assert property_reward(values, "minimize", target=1.0)[0] > 0.5
    assert property_reward(values, "target", target=1.0)[1] == 1.0
    range_score = property_reward(values, "range", lower=0.5, upper=1.5, tolerance=0.1)
    assert range_score[1] > range_score[0]
    assert range_score[1] > range_score[2]


def test_uncertainty_penalty_and_validity_gate_are_applied_after_property_score():
    reward = torch.tensor([0.8, 0.8])
    result = robust_validity_gated_reward(
        reward,
        valid=torch.tensor([True, False]),
        uncertainty=torch.tensor([0.2, 0.0]),
        uncertainty_penalty=0.5,
    )
    assert torch.allclose(result, torch.tensor([0.7, 0.0]))


def test_general_material_reward_uses_fixed_targets_and_explicit_invalid_penalty():
    formation_energy = property_reward(
        torch.tensor([-1.5, -1.4, -1.5]),
        "target",
        target=-1.5,
        tolerance=0.06,
    )
    band_gap = property_reward(
        torch.tensor([2.0, 2.0, 1.0]),
        "target",
        target=2.0,
        tolerance=0.45,
    )
    result = general_material_reward(
        {"formation_energy_per_atom": formation_energy, "band_gap": band_gap},
        valid=torch.tensor([True, False, True]),
        invalid_penalty=0.5,
    )
    assert result.raw_reward[0] == 1.0
    assert result.raw_reward[1] == -0.5
    assert torch.equal(
        result.property_reward,
        combine_property_rewards(
            {"formation_energy_per_atom": formation_energy, "band_gap": band_gap}
        ),
    )
    assert result.raw_reward[2] == band_gap[2]


def test_chemeleon_style_general_generation_components_are_bounded_and_auditable():
    creativity = continuous_creativity_reward(
        unique=torch.tensor([True, False, True]),
        novel=torch.tensor([True, False, False]),
        boundary_distance=torch.tensor([0.2, 0.8, 0.4]),
    )
    stability = bounded_stability_reward(torch.tensor([0.0, 0.1, float("nan")]))
    assert torch.equal(creativity, torch.tensor([1.0, 0.0, 0.4]))
    assert torch.allclose(stability, torch.tensor([1.0, 0.9, 0.0]))

    result = general_material_reward(
        {"target": torch.tensor([1.0, 0.5, 0.25])},
        valid=torch.ones(3, dtype=torch.bool),
        auxiliary_scores={"creativity": creativity, "stability": stability},
        auxiliary_weights={"creativity": 1.0, "stability": 1.0},
    )
    assert set(result.components) == {"target", "creativity", "stability"}
    assert torch.all((result.raw_reward >= 0) & (result.raw_reward <= 1))


def test_leave_one_out_mmd_assigns_one_reward_per_generated_structure():
    generated = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    reference = torch.tensor([[0.0], [1.0], [2.0], [4.0]])
    reward = leave_one_out_mmd_reward(generated, reference)
    assert reward.shape == (4,)
    assert torch.isfinite(reward).all()
    bounded = bounded_diversity_reward(reward, scale=0.1)
    assert torch.all((bounded >= 0) & (bounded <= 1))


def test_d3pm_training_loss_receives_only_orbit_representatives(monkeypatch):
    captured = {}

    def fake_compute_d3pm_loss(**kwargs):
        captured.update(kwargs)
        return torch.tensor(2.0)

    monkeypatch.setattr(diffusion_loss_module, "compute_d3pm_loss", fake_compute_d3pm_loss)
    loss_fn = DiffusionLoss(d3pm_diffusion=object(), num_atom_types=100)
    anchors = torch.tensor([0, 0, 2, 2])
    result = loss_fn.calc_atom_loss(
        pred_atom_logits=torch.zeros((4, 100)),
        x_start_atoms=torch.tensor([1, 1, 2, 2]),
        input_atom_types=torch.tensor([100, 100, 2, 2]),
        t_discrete=torch.tensor([3, 3, 3, 3]),
        batch_idx=torch.zeros(4, dtype=torch.long),
        num_graphs=1,
        anchor_index=anchors,
    )
    assert result == 2.0
    assert captured["score_model_logits"].shape[0] == 2
    assert torch.equal(captured["x_start"], torch.tensor([1, 2]))
