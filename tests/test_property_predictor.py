from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cgdit.prop_models.gnn_models.m3gnet import M3GNetSurrogate


class DummyNormalizedModel(nn.Module):
    def forward(self, batch):
        return batch.normalized_predictions


def make_surrogate(
    target_prop="band_gap", target_mean=10.0, target_std=2.0, n_blocks=1, replace_model=True
):
    model = M3GNetSurrogate(
        target_prop=target_prop,
        target_mean=target_mean,
        target_std=target_std,
        n_atom_types=10,
        dim_node_embedding=4,
        dim_edge_embedding=4,
        dim_state_embedding=0,
        units=4,
        cutoff=5.0,
        threebody_cutoff=4.0,
        max_n=1,
        max_l=1,
        n_blocks=n_blocks,
        ntargets=1,
    )
    if replace_model:
        model.model = DummyNormalizedModel()
    return model


def test_surrogate_returns_predictions_in_physical_units():
    model = make_surrogate()
    batch = SimpleNamespace(normalized_predictions=torch.tensor([0.0, 1.0]))

    predictions = model(batch)

    assert torch.allclose(predictions, torch.tensor([10.0, 12.0]))


def test_loss_is_standardized_but_mae_uses_physical_units():
    model = make_surrogate()
    predictions = torch.tensor([10.0, 12.0])
    targets = torch.tensor([10.0, 14.0])

    metrics, loss = model.compute_stats(predictions, targets, prefix="val")

    assert loss.item() == pytest.approx(0.5)
    assert metrics["val_mae"].item() == pytest.approx(1.0)


def test_target_property_is_used_when_y_is_absent():
    model = make_surrogate(target_prop="band_gap")
    batch = SimpleNamespace(
        y=None,
        formation_energy_per_atom=torch.tensor([-1.5]),
        band_gap=torch.tensor([2.0]),
    )

    assert torch.equal(model._get_targets(batch), torch.tensor([2.0]))


def test_non_positive_target_std_is_rejected():
    with pytest.raises(ValueError, match="target_std"):
        make_surrogate(target_std=0.0)


def test_m3gnet_block_initialization_is_reproducible():
    torch.manual_seed(123)
    first = make_surrogate(n_blocks=3, replace_model=False)
    torch.manual_seed(123)
    second = make_surrogate(n_blocks=3, replace_model=False)

    first_state = first.model.state_dict()
    second_state = second.model.state_dict()
    assert first_state.keys() == second_state.keys()
    assert all(torch.equal(first_state[key], second_state[key]) for key in first_state)
