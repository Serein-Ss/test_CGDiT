import torch

import cgdit.pl_modules.training_utils.diffusion_loss as diffusion_loss_module
from cgdit.pl_modules.training_utils.diffusion_loss import DiffusionLoss
from cgdit.rl.rewards import property_reward, robust_validity_gated_reward


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
