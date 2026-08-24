import numpy as np
import pandas as pd

from scripts.result_figures.crystalpirl_seed42.audit_conclusions import _finite
from scripts.result_figures.crystalpirl_seed42.build_quality_source import (
    _sample_seed,
)
from scripts.result_figures.crystalpirl_seed42.build_source_data import (
    _expected_contract,
    _group_specs,
    _summary,
)


def test_method_summary_keeps_failed_predictions_in_total_count():
    rows = pd.DataFrame(
        {
            "task": ["fe", "fe"],
            "method": ["Base", "Base"],
            "method_family": ["base", "base"],
            "algorithm": [None, None],
            "generation_mode": ["abinitio", "abinitio"],
            "seed": [42, 42],
            "structure_id": ["a", "b"],
            "independent_prediction": [-1.5, np.nan],
            "absolute_target_error": [0.0, np.nan],
            "hit": [True, False],
            "valid_target_yield": [True, False],
            "valid": [True, False],
            "predictor_gap": [0.1, np.nan],
        }
    )

    summary = _summary(rows).iloc[0]

    assert summary["n"] == 2
    assert summary["finite_prediction_n"] == 1
    assert summary["finite_prediction_fraction"] == 0.5
    assert summary["valid_target_yield"] == 0.5


def test_quality_sampling_seed_is_group_deterministic():
    assert _sample_seed("open_ppo_fe_abinitio") == _sample_seed(
        "open_ppo_fe_abinitio"
    )
    assert _sample_seed("open_ppo_fe_abinitio") != _sample_seed(
        "open_ppo_fe_abinitio_replicate"
    )


def test_audit_finite_check_rejects_missing_values():
    assert _finite(pd.DataFrame({"value": [1.0, 2.0]}), ["value"])
    assert not _finite(pd.DataFrame({"value": [1.0, np.nan]}), ["value"])


def test_formal_structure_groups_are_ab_initio_only(tmp_path):
    specs = _group_specs(tmp_path, rl_samples=256)

    assert {spec["generation_mode"] for spec in specs} == {"abinitio"}
    assert len(specs) == 15


def test_quick8h_source_contract_counts_are_explicit():
    contract = _expected_contract(updates_per_run=3, rl_samples=256)

    assert contract["fig1_rows"] == 12
    assert contract["fig2_rows"] == 36
    assert contract["fig3_rows"] == 20480
    assert contract["fig3_summary_rows"] == 18
    assert contract["fig4_groups"] == 15
    assert contract["rl_final_groups"] == 12
    assert contract["fig4_group_sizes"] == [256, 4096]
