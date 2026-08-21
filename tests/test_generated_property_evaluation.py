import argparse

import numpy as np
import pytest
import torch

from cgdit.evaluation.generated_properties import (
    DEFAULT_TOLERANCES,
    build_graphs,
    compute_property_metrics,
    discover_formal_generation_files,
    extract_condition_targets,
    load_generation_payload,
)


def test_extract_condition_targets_prefers_joint_condition_dict():
    payload = {
        "conditions": {
            "formation_energy_per_atom": -1.5,
            "band_gap": 2.0,
        },
        "property_name": "formation_energy_per_atom",
        "target_value": -2.0,
    }
    assert extract_condition_targets(payload) == {
        "formation_energy_per_atom": -1.5,
        "band_gap": 2.0,
    }


def test_property_metrics_include_single_and_joint_hit_rates():
    predictions = {
        "formation_energy_per_atom": np.array([-1.50, -1.40, np.nan]),
        "band_gap": np.array([2.00, 2.30, 2.00]),
        "e_above_hull": np.array([0.01, 0.02, 0.03]),
    }
    metrics = compute_property_metrics(
        predictions=predictions,
        targets={"formation_energy_per_atom": -1.5, "band_gap": 2.0},
        tolerances=DEFAULT_TOLERANCES,
        total_count=3,
        graph_success_count=2,
        reference_predictions={
            name: np.array([0.0, 1.0]) for name in predictions
        },
        reference_targets={
            name: np.array([0.0, 1.0]) for name in predictions
        },
    )

    formation_energy = metrics["properties"]["formation_energy_per_atom"]
    assert formation_energy["n_hits"] == 1
    assert formation_energy["hit_rate_predicted"] == pytest.approx(0.5)
    assert formation_energy["hit_rate_all"] == pytest.approx(1 / 3)
    assert metrics["joint"]["n_joint_hits"] == 1
    assert metrics["joint"]["joint_hit_rate_predicted"] == pytest.approx(0.5)
    assert metrics["joint"]["joint_hit_rate_all"] == pytest.approx(1 / 3)
    assert "wdist_to_test_predictions" in formation_energy


def test_discovery_is_not_tied_to_the_removed_ehull_models(tmp_path):
    formal_template = tmp_path / "eval_gen_template_uncond_n4096_seed42.pt"
    formal_abinitio = (
        tmp_path / "eval_gen_abinitio_empirical_uncond_n4096_seed42.pt"
    )
    pilot = tmp_path / "eval_gen_pilot_template_uncond_seed42.pt"
    smoke = tmp_path / "eval_gen_abinitio_fix_smoke.pt"
    for path in (formal_template, formal_abinitio, pilot, smoke):
        path.write_bytes(b"result")

    assert discover_formal_generation_files(tmp_path) == [
        formal_abinitio,
        formal_template,
    ]


def test_generation_payload_uses_safe_tensor_loading(tmp_path):
    path = tmp_path / "generation.pt"
    torch.save(
        {
            "eval_setting": argparse.Namespace(seed=42),
            "atom_types": torch.tensor([0, 13]),
        },
        path,
    )

    payload = load_generation_payload(path)

    assert payload["eval_setting"].seed == 42
    assert torch.equal(payload["atom_types"], torch.tensor([0, 13]))


def test_graph_builder_canonicalizes_reordered_niggli_lattice():
    crystal = {
        "frac_coords": np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        "atom_types": np.array([14, 39]),
        "lengths": np.array([11.8, 11.8, 6.1]),
        "angles": np.array([90.0, 90.0, 90.0]),
    }
    data_list, indices, errors = build_graphs([crystal], num_workers=2)
    assert indices == [0]
    assert errors == [None]
    assert len(data_list) == 1
