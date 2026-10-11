import argparse

import numpy as np
import pytest
import torch

from cgdit.evaluation.generated_properties import (
    DEFAULT_TOLERANCES,
    build_graphs,
    compute_property_metrics,
    crystal_validity,
    discover_formal_generation_files,
    extract_condition_targets,
    load_generation_payload,
    predict_graphs,
    resolve_checkpoint,
)


def test_checkpoint_resolution_accepts_explicit_file_and_single_checkpoint(tmp_path):
    checkpoint = tmp_path / "epoch=4-step=10.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    assert resolve_checkpoint(checkpoint) == checkpoint
    assert resolve_checkpoint(tmp_path) == checkpoint


def test_resumed_predictor_uses_last_tested_checkpoint(tmp_path):
    tested = tmp_path / "epoch=19-step=200.ckpt"
    other = tmp_path / "epoch=29-step=300.ckpt"
    for checkpoint in (tested, other):
        checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "run.log").write_text(
        f"Testing best checkpoint: {other}\n"
        f"Testing best checkpoint: {tested}\n"
    )
    assert resolve_checkpoint(tmp_path) == tested
    assert other.exists()


def test_tested_checkpoint_survives_run_directory_relocation(tmp_path):
    tested = tmp_path / "epoch=19-step=200.ckpt"
    tested.write_bytes(b"checkpoint")
    (tmp_path / "epoch=29-step=300.ckpt").write_bytes(b"checkpoint")
    (tmp_path / "run.log").write_text(
        "Testing best checkpoint: /previous/node/predictor/epoch=19-step=200.ckpt\n"
    )
    assert resolve_checkpoint(tmp_path) == tested


@pytest.mark.parametrize("log", [None, "Starting training!\n",
                                "Testing best checkpoint: /missing/epoch=99.ckpt\n",
                                "Testing best checkpoint: epoch=4.ckpt\n"
                                "Testing best checkpoint: /missing/epoch=99.ckpt\n"])
def test_multiple_checkpoints_require_a_resolvable_test_record(tmp_path, log):
    for epoch in (4, 9):
        (tmp_path / f"epoch={epoch}.ckpt").write_bytes(b"checkpoint")
    if log is not None:
        (tmp_path / "run.log").write_text(log)
    with pytest.raises(RuntimeError, match="Expected exactly one checkpoint"):
        resolve_checkpoint(tmp_path)


def test_checkpoint_resolution_reports_missing_checkpoint(tmp_path):
    with pytest.raises(RuntimeError, match="found 0"):
        resolve_checkpoint(tmp_path)


@pytest.mark.parametrize("with_bonds", [False, True])
def test_property_prediction_supports_empty_triplet_batches(monkeypatch, with_bonds):
    from torch_geometric.data import Data
    from cgdit.prop_models.gnn_models.m3gnet import M3GNetSurrogate

    model = M3GNetSurrogate(
        target_prop="band_gap", n_atom_types=10, dim_node_embedding=4,
        dim_edge_embedding=4, dim_state_embedding=0, units=4, max_n=1,
        max_l=1, n_blocks=1, ntargets=1,
    )
    monkeypatch.setattr(M3GNetSurrogate, "load_from_checkpoint",
                        lambda *args, **kwargs: model)
    graph = Data(
        frac_coords=torch.tensor([[0., 0., 0.], [0.25, 0.25, 0.25]]),
        atom_types=torch.tensor([1, 2]), lengths=torch.tensor([[5.6, 5.6, 5.6]]),
        angles=torch.tensor([[90., 90., 90.]]),
        edge_index=(torch.tensor([[0, 1], [1, 0]]) if with_bonds
                    else torch.empty((2, 0), dtype=torch.long)),
        to_jimages=torch.zeros((2 if with_bonds else 0, 3), dtype=torch.long),
        line_edge_index=torch.empty((2, 0), dtype=torch.long),
        num_atoms=2, num_bonds=2 if with_bonds else 0, num_nodes=2,
    )
    predictions = predict_graphs("unused.ckpt", "band_gap", [graph], [0], 1, 1, "cpu")
    assert predictions.shape == (1,)
    assert np.isfinite(predictions).all()


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


def test_extract_condition_targets_accepts_evaluation_only_metadata():
    payload = {
        "conditions": {},
        "evaluation_targets": {"formation_energy_per_atom": -1.5},
    }

    assert extract_condition_targets(payload) == {
        "formation_energy_per_atom": -1.5
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


def test_property_metrics_support_fe_bg_only_scope():
    predictions = {
        "formation_energy_per_atom": np.array([-1.5, -1.0]),
        "band_gap": np.array([2.0, 1.0]),
    }

    metrics = compute_property_metrics(
        predictions=predictions,
        targets={"formation_energy_per_atom": -1.5},
        tolerances=DEFAULT_TOLERANCES,
        total_count=2,
        graph_success_count=2,
    )

    assert set(metrics["properties"]) == set(predictions)
    assert metrics["joint"]["target_properties"] == [
        "formation_energy_per_atom"
    ]


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


def test_generation_payload_supports_torch_without_safe_globals(
    tmp_path, monkeypatch
):
    path = tmp_path / "legacy_generation.pt"
    torch.save(
        {
            "eval_setting": argparse.Namespace(seed=123),
            "atom_types": torch.tensor([7, 25]),
        },
        path,
    )
    monkeypatch.delattr(torch.serialization, "add_safe_globals", raising=False)

    payload = load_generation_payload(path)

    assert payload["eval_setting"].seed == 123
    assert torch.equal(payload["atom_types"], torch.tensor([7, 25]))


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


def test_crystal_validity_returns_per_structure_components():
    crystal = {
        "frac_coords": np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]),
        "atom_types": np.array([11, 17]),
        "lengths": np.array([5.6, 5.6, 5.6]),
        "angles": np.array([90.0, 90.0, 90.0]),
    }

    composition_valid, structure_valid, valid = crystal_validity(crystal)

    assert composition_valid
    assert structure_valid
    assert valid
