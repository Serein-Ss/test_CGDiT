import torch
from torch import nn
from torch_geometric.data import Data

from cgdit.rl.property_reward import M3GNetRewardAdapter, PropertyRewardSpec


class FakePredictor(nn.Module):
    def __init__(self, target_prop, values):
        super().__init__()
        self.target_prop = target_prop
        self.values = values
        self.marker = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        return self.values[: batch.num_graphs].to(batch.x.device)


def test_frozen_predictors_feed_fixed_scale_reward_and_invalid_penalty():
    predictors = {
        "formation_energy_per_atom": FakePredictor(
            "formation_energy_per_atom", torch.tensor([-1.5, -1.5])
        ),
        "band_gap": FakePredictor("band_gap", torch.tensor([2.0, 2.0])),
    }
    specs = {
        "formation_energy_per_atom": PropertyRewardSpec(
            mode="target", target=-1.5, tolerance=0.06
        ),
        "band_gap": PropertyRewardSpec(
            mode="target", target=2.0, tolerance=0.45
        ),
    }
    adapter = M3GNetRewardAdapter(
        predictors,
        specs,
        device="cpu",
        invalid_penalty=0.5,
    )
    data_list = [
        Data(x=torch.ones((1, 1)), num_nodes=1),
        Data(x=torch.ones((1, 1)), num_nodes=1),
    ]
    result = adapter.evaluate_graphs(
        data_list,
        successful_indices=[0, 1],
        validity=torch.tensor([True, False]),
        graph_errors=(None, None),
        total_count=2,
    )
    assert torch.equal(result.reward.raw_reward, torch.tensor([1.0, -0.5]))
    assert result.valid.tolist() == [True, False]
    assert all(
        not parameter.requires_grad
        for predictor in predictors.values()
        for parameter in predictor.parameters()
    )


def test_prediction_failure_is_retained_as_invalid_in_reward_denominator():
    predictor = FakePredictor("band_gap", torch.tensor([2.0]))
    adapter = M3GNetRewardAdapter(
        {"band_gap": predictor},
        {"band_gap": PropertyRewardSpec("target", target=2.0, tolerance=0.45)},
        device="cpu",
    )
    result = adapter.evaluate_graphs(
        [Data(x=torch.ones((1, 1)), num_nodes=1)],
        successful_indices=[0],
        validity=torch.tensor([True, True]),
        graph_errors=(None, "graph failed"),
        total_count=2,
    )
    assert result.valid.tolist() == [True, False]
    assert torch.isnan(result.predictions["band_gap"][1])
    assert result.reward.raw_reward.tolist() == [1.0, -1.0]

def test_stability_surrogate_is_separate_from_property_reward():
    reward_predictor = FakePredictor("band_gap", torch.tensor([2.0, 2.0]))
    stability_predictor = FakePredictor(
        "formation_energy_per_atom", torch.tensor([-1.0, 1.0])
    )
    adapter = M3GNetRewardAdapter(
        {"band_gap": reward_predictor},
        {"band_gap": PropertyRewardSpec("target", target=2.0, tolerance=0.45)},
        device="cpu",
        stability_predictor=stability_predictor,
        stability_target=0.0,
        stability_tolerance=0.3,
    )
    data_list = [
        Data(x=torch.ones((1, 1)), num_nodes=1),
        Data(x=torch.ones((1, 1)), num_nodes=1),
    ]

    result = adapter.evaluate_graphs(
        data_list,
        successful_indices=[0, 1],
        validity=torch.tensor([True, True]),
        graph_errors=(None, None),
        total_count=2,
    )

    assert torch.equal(result.reward.raw_reward, torch.ones(2))
    assert set(result.safety_metrics) == {"stability"}
    assert result.safety_metrics["stability"][0] > 0.5
    assert result.safety_metrics["stability"][1] < 0.5
    assert all(
        not parameter.requires_grad
        for parameter in stability_predictor.parameters()
    )
