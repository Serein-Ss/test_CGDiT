import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from cgdit.prop_models.diffusion_backbone import extract_decoder_state
from scripts.cli.evaluation.summarize_magndata_tc import regression_metrics, summarize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "magndata"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_packaged_splits() -> None:
    if not (DATA_ROOT / "train.csv").is_file():
        pytest.skip("Magndata CSV files are distributed separately from the code")


def test_packaged_magndata_splits_match_manifest_without_formula_leakage():
    _require_packaged_splits()
    manifest = json.loads((DATA_ROOT / "split_manifest.json").read_text())
    frames = {
        split: pd.read_csv(DATA_ROOT / f"{split}.csv")
        for split in ("train", "val", "test")
    }

    assert [len(frames[name]) for name in ("train", "val", "test")] == [1023, 122, 139]
    assert sum(map(len, frames.values())) == 1284
    assert set(frames["train"].pretty_formula).isdisjoint(frames["val"].pretty_formula)
    assert set(frames["train"].pretty_formula).isdisjoint(frames["test"].pretty_formula)
    assert set(frames["val"].pretty_formula).isdisjoint(frames["test"].pretty_formula)
    assert all(frame.natoms.max() <= 100 for frame in frames.values())
    assert all(np.isfinite(frame.tc).all() for frame in frames.values())

    for filename, expected_hash in manifest["files"].items():
        assert _sha256(DATA_ROOT / filename) == expected_hash


def test_magndata_config_uses_training_only_target_statistics():
    _require_packaged_splits()
    frame = pd.read_csv(DATA_ROOT / "train.csv")
    config = OmegaConf.load(PROJECT_ROOT / "conf" / "data" / "magndata.yaml")
    assert np.isclose(config.target_stats.tc.mean, frame.tc.mean())
    assert np.isclose(config.target_stats.tc.std, frame.tc.std(ddof=0))


def test_decoder_state_extraction_strips_diffusion_prefix_only():
    state = {
        "decoder.node_embedding.weight": np.zeros(1),
        "conditioner.foo": np.zeros(1),
    }
    extracted = extract_decoder_state(state)
    assert set(extracted) == {"node_embedding.weight"}


def test_regression_metrics_are_reported_in_kelvin():
    targets = np.array([0.0, 100.0, 200.0])
    predictions = np.array([10.0, 90.0, 220.0])
    metrics = regression_metrics(targets, predictions, bootstrap_seed=42)
    assert np.isclose(metrics["mae_k"], 40.0 / 3.0)
    assert np.isclose(metrics["rmse_k"], np.sqrt(200.0))
    assert metrics["n_test"] == 3


def test_benchmark_summary_writes_all_comparison_tables(tmp_path: Path):
    targets = np.array([10.0, 100.0, 300.0])
    offsets = {
        "m3gnet_scratch": 20.0,
        "diffusion_scratch": 15.0,
        "diffusion_base_pretrained": 10.0,
        "diffusion_joint_pretrained": 5.0,
    }
    for method, offset in offsets.items():
        run_dir = tmp_path / method / "seed_42"
        run_dir.mkdir(parents=True)
        np.save(run_dir / "test_targets.npy", targets)
        np.save(run_dir / "test_preds.npy", targets + offset)

    per_run, aggregate = summarize(tmp_path, [42])
    assert len(per_run) == 4
    assert set(aggregate["method"]) == set(offsets)
    assert (tmp_path / "aggregate_metrics.csv").is_file()
    assert (tmp_path / "paired_differences.csv").is_file()
    assert (tmp_path / "benchmark_summary.json").is_file()
    paired = pd.read_csv(tmp_path / "paired_differences.csv")
    assert len(paired) == 5
    assert "diffusion_joint_pretrained_minus_diffusion_base_pretrained" in set(
        paired["comparison"]
    )
