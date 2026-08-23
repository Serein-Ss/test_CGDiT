from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cgdit.common.evaluation_utils import load_data
from cgdit.evaluation.metrics import GenEval


def test_generation_metric_sampling_uses_explicit_seed():
    crystals = [SimpleNamespace(valid=True, identifier=index) for index in range(20)]

    evaluator = GenEval(crystals, [], n_samples=5, calc_prop=False, seed=7)

    expected = np.random.default_rng(7).choice(20, 5, replace=False)
    assert [crystal.identifier for crystal in evaluator.valid_samples] == expected.tolist()


def test_generation_metric_sampling_clamps_to_small_valid_pool():
    crystals = [SimpleNamespace(valid=True, identifier=index) for index in range(20)]

    evaluator = GenEval(crystals, [], calc_prop=False, seed=7)

    assert len(evaluator.valid_samples) == 20


def test_load_data_accepts_path_objects(tmp_path):
    path = tmp_path / "generation.pt"
    torch.save({"value": torch.tensor([1.0])}, path)

    loaded = load_data(path)

    assert torch.equal(loaded["value"], torch.tensor([1.0]))


def test_matgl_prediction_failure_is_not_replaced_with_zero(monkeypatch):
    import matgl

    class FailingModel:
        def predict_structure(self, structure):
            raise ValueError("synthetic MatGL failure")

    monkeypatch.setattr(matgl, "load_model", lambda _: FailingModel())
    evaluator = GenEval(
        [SimpleNamespace(valid=True, structure=object())],
        [],
        n_samples=1,
        prop_model_path="matgl",
    )

    with pytest.raises(RuntimeError, match="item 0") as exc_info:
        evaluator.get_prop_wdist()

    assert isinstance(exc_info.value.__cause__, ValueError)
