from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from cgdit.rl.config import load_rl_config
from scripts.cli.training.train_crystal_rl import (
    PROPERTY_NAMES,
    _metallicity_proxy_summary,
    _property_specs,
)


def test_metal_task_reuses_band_gap_predictor_with_calibrated_upper_threshold():
    contract = OmegaConf.load(
        "conf/rl/components/rewards/contracts/mp20_metal_seed42_pilot.yaml"
    )

    assert PROPERTY_NAMES["metal"] == ("band_gap",)
    spec = _property_specs(contract, "metal")["band_gap"]
    assert spec.mode == "minimize"
    assert spec.target == 0.10
    assert spec.tolerance == 0.05


def test_metal_training_starts_from_the_frozen_base_policy():
    config = load_rl_config(
        "conf/rl/experiments/grpo_metal_seed42_pipo_probe32.yaml"
    )

    assert config.property == "metal"
    assert config.model_path.endswith("00-32-50-mp20_base")
    assert config.max_prompt_atoms == 60
    assert config.num_prompts == 4
    assert config.group_size == 16


def test_metallicity_summary_reports_valid_and_all_structure_yields():
    contract = OmegaConf.load(
        "conf/rl/components/rewards/contracts/mp20_metal_seed42_pilot.yaml"
    )
    evaluation = SimpleNamespace(
        predictions={"band_gap": torch.tensor([0.0, 0.2, 0.1, float("nan")])},
        valid=torch.tensor([True, True, False, True]),
    )

    summary = _metallicity_proxy_summary(evaluation, contract)

    assert summary == {
        "predicted_threshold_eV": 0.1,
        "predicted_metal_count": 1,
        "predicted_metal_fraction_all": 0.25,
        "predicted_metal_fraction_valid": 0.5,
    }
