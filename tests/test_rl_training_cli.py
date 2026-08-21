from types import SimpleNamespace

import pytest
import torch

from scripts.cli.training.train_crystal_rl import (
    _blend_decoder,
    _grouped_batch,
    _transition_indices,
)


class _Prompt:
    def __init__(self, value):
        self.value = value

    def clone(self):
        return _Prompt(self.value)


def test_transition_indices_cover_endpoints():
    assert _transition_indices(10, 3) == [0, 4, 9]
    assert _transition_indices(2, 5) == [0, 1]


def test_grouped_batch_repeats_each_prompt(monkeypatch):
    captured = []

    def fake_batch(data):
        captured.extend(item.value for item in data)
        return SimpleNamespace()

    monkeypatch.setattr(
        "scripts.cli.training.train_crystal_rl.Batch.from_data_list", fake_batch
    )
    _, groups = _grouped_batch([_Prompt("a"), _Prompt("b")], 2, 3)

    assert captured == ["a", "a", "a", "b", "b", "b"]
    assert groups.tolist() == [0, 0, 0, 1, 1, 1]


def test_blend_decoder_applies_decision_scale():
    model = torch.nn.Module()
    model.decoder = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.decoder.weight.fill_(3.0)
    old_state = {"weight": torch.tensor([[1.0]])}

    _blend_decoder(model, old_state, 0.25)

    assert model.decoder.weight.item() == pytest.approx(1.5)
