import pytest
import torch

from scripts.cli.training.diagnose_rl_update_scale import (
    _decoder_update_norm,
    _set_decoder_scale,
)


def _model(weight):
    model = torch.nn.Module()
    model.decoder = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.decoder.weight.fill_(weight)
    return model


def test_decoder_scale_uses_fixed_base_and_candidate_endpoints():
    model = _model(9.0)
    base = {"weight": torch.tensor([[1.0]])}
    candidate = {"weight": torch.tensor([[5.0]])}

    _set_decoder_scale(model, base, candidate, 0.25)
    assert model.decoder.weight.item() == pytest.approx(2.0)

    _set_decoder_scale(model, base, candidate, 1.0)
    assert model.decoder.weight.item() == pytest.approx(5.0)

    _set_decoder_scale(model, base, candidate, 0.05)
    assert model.decoder.weight.item() == pytest.approx(1.2)


def test_decoder_update_norm_reports_absolute_and_relative_change():
    base = {
        "first": torch.tensor([3.0, 4.0]),
        "second": torch.tensor([0.0]),
    }
    candidate = {
        "first": torch.tensor([6.0, 8.0]),
        "second": torch.tensor([0.0]),
    }

    absolute, relative = _decoder_update_norm(base, candidate)
    assert absolute == pytest.approx(5.0)
    assert relative == pytest.approx(1.0)
