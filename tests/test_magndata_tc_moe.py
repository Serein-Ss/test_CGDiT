import json
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir

from scripts.cli.data.prepare_magndata_tc_moe import prepare_moe_splits
from scripts.cli.evaluation.summarize_magndata_tc_moe import (
    choose_probability_threshold,
    routed_predictions,
    summarize,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_finetune_checkpoint_override_accepts_lightning_equals_filename(
    monkeypatch,
):
    checkpoint = "/tmp/epoch=224-step=190800.ckpt"
    monkeypatch.setenv("FINETUNE_CHECKPOINT", checkpoint)
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "conf"),
        version_base=None,
    ):
        config = compose(
            config_name="default",
            overrides=[
                "data=magndata",
                "model=property_predictors/diffusion_cspnet/tc_regressor",
                "optim=magndata_tc",
                "train=magndata_tc",
                "+train.finetune_from_ckpt=${oc.env:FINETUNE_CHECKPOINT}",
            ],
        )
    assert config.train.finetune_from_ckpt == checkpoint


def test_prepare_moe_splits_preserves_full_test_for_both_experts(tmp_path: Path):
    source_root = tmp_path / "source"
    output_root = tmp_path / "moe"
    source_root.mkdir()
    frames = {
        "train": pd.DataFrame({"material_id": range(6), "tc": [10, 50, 100, 299, 300, 500]}),
        "val": pd.DataFrame({"material_id": range(2), "tc": [80, 350]}),
        "test": pd.DataFrame({"material_id": range(4), "tc": [25, 250, 325, 600]}),
    }
    for split, frame in frames.items():
        frame.to_csv(source_root / f"{split}.csv", index=False)

    manifest = prepare_moe_splits(source_root, output_root, threshold_k=300.0)

    assert manifest["counts"]["train"] == {"low": 4, "high": 2}
    assert np.isclose(manifest["gate_pos_weight"], 2.0)
    assert len(pd.read_csv(output_root / "low" / "train.csv")) == 4
    assert len(pd.read_csv(output_root / "high" / "train.csv")) == 2
    assert len(pd.read_csv(output_root / "low" / "test.csv")) == 4
    assert len(pd.read_csv(output_root / "high" / "test.csv")) == 4
    gate_test = pd.read_csv(output_root / "gate" / "test.csv")
    assert gate_test.high_tc.tolist() == [0.0, 0.0, 1.0, 1.0]
    saved_manifest = json.loads((output_root / "moe_data_manifest.json").read_text())
    assert saved_manifest["threshold_k"] == 300.0


def test_gate_threshold_and_routing_are_deterministic():
    targets = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.20, 0.75, 0.95])
    threshold, metrics = choose_probability_threshold(
        targets,
        probabilities,
        min_recall=0.80,
    )
    assert metrics["recall"] >= 0.80
    assert metrics["mcc"] == 1.0

    routes = routed_predictions(
        gate_probabilities=np.array([0.25, 0.75]),
        low_predictions=np.array([100.0, 150.0]),
        high_predictions=np.array([400.0, 500.0]),
        true_targets=np.array([200.0, 350.0]),
        tc_threshold_k=300.0,
        probability_threshold=0.5,
    )
    np.testing.assert_allclose(routes["oracle_route"], [100.0, 500.0])
    np.testing.assert_allclose(routes["hard_route"], [100.0, 500.0])
    np.testing.assert_allclose(routes["soft_route"], [175.0, 412.5])
    assert 0.20 < threshold <= 0.75


def test_moe_summary_writes_gate_and_regression_outputs(tmp_path: Path):
    moe_root = tmp_path / "moe"
    global_root = tmp_path / "global"
    gate_root = moe_root / "seed_42" / "gate"
    low_root = moe_root / "seed_42" / "low_expert"
    high_root = moe_root / "seed_42" / "high_expert"
    baseline_root = global_root / "diffusion_base_pretrained" / "seed_42"
    for path in (gate_root, low_root, high_root, baseline_root):
        path.mkdir(parents=True)

    targets = np.array([100.0, 180.0, 250.0, 310.0, 400.0, 520.0])
    classes = (targets >= 300.0).astype(float)
    np.save(gate_root / "val_targets.npy", np.array([0.0, 0.0, 1.0, 1.0]))
    np.save(gate_root / "val_probs.npy", np.array([0.10, 0.30, 0.70, 0.90]))
    np.save(gate_root / "test_targets.npy", classes)
    np.save(gate_root / "test_probs.npy", np.array([0.10, 0.20, 0.35, 0.65, 0.80, 0.95]))
    np.save(baseline_root / "test_targets.npy", targets)
    np.save(baseline_root / "test_preds.npy", targets + np.array([15, -10, 20, -20, 25, -30]))
    np.save(low_root / "test_targets.npy", targets)
    np.save(low_root / "test_preds.npy", np.array([105, 170, 260, 275, 290, 310]))
    np.save(high_root / "test_targets.npy", targets)
    np.save(high_root / "test_preds.npy", np.array([280, 290, 300, 320, 390, 510]))

    gate_metrics, regression_metrics = summarize(
        moe_root,
        global_root,
        seeds=[42],
        tc_threshold_k=300.0,
        min_gate_recall=0.80,
    )

    assert set(gate_metrics.split) == {"validation", "test"}
    assert set(regression_metrics.method) == {
        "global_regressor",
        "low_expert",
        "high_expert",
        "oracle_route",
        "hard_route",
        "soft_route",
    }
    assert set(regression_metrics.scope) == {"all", "true_low", "true_high"}
    for filename in (
        "gate_metrics_per_seed.csv",
        "gate_metrics_aggregate.csv",
        "regression_metrics_per_seed.csv",
        "regression_metrics_aggregate.csv",
        "routing_differences.csv",
        "test_predictions.csv",
        "moe_summary.json",
    ):
        assert (moe_root / filename).is_file()
