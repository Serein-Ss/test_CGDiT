"""Summarize high/low Tc gate and mixture-of-experts regression results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from scripts.cli.evaluation.summarize_magndata_tc import regression_metrics


def classification_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    probability_threshold: float,
) -> dict[str, float | int]:
    targets = np.asarray(targets, dtype=int).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(targets) != len(probabilities) or not len(targets):
        raise ValueError("Classification arrays must have the same non-zero length")
    predictions = (probabilities >= probability_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    return {
        "n": int(len(targets)),
        "n_high": int(targets.sum()),
        "probability_threshold": float(probability_threshold),
        "pr_auc": float(average_precision_score(targets, probabilities)),
        "roc_auc": float(roc_auc_score(targets, probabilities)),
        "mcc": float(matthews_corrcoef(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_probability_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
    min_recall: float = 0.80,
) -> tuple[float, dict[str, float | int]]:
    candidates = np.unique(np.concatenate([
        np.linspace(0.01, 0.99, 99),
        np.asarray(probabilities, dtype=float).reshape(-1),
    ]))
    records = [classification_metrics(targets, probabilities, value) for value in candidates]
    eligible = [record for record in records if record["recall"] >= min_recall]
    pool = eligible or records
    best = max(
        pool,
        key=lambda record: (
            record["mcc"],
            record["precision"],
            -abs(record["probability_threshold"] - 0.5),
        ),
    )
    return float(best["probability_threshold"]), best


def routed_predictions(
    gate_probabilities: np.ndarray,
    low_predictions: np.ndarray,
    high_predictions: np.ndarray,
    true_targets: np.ndarray,
    tc_threshold_k: float,
    probability_threshold: float,
) -> dict[str, np.ndarray]:
    gate_probabilities = np.asarray(gate_probabilities, dtype=float).reshape(-1)
    low_predictions = np.asarray(low_predictions, dtype=float).reshape(-1)
    high_predictions = np.asarray(high_predictions, dtype=float).reshape(-1)
    true_targets = np.asarray(true_targets, dtype=float).reshape(-1)
    sizes = {len(gate_probabilities), len(low_predictions), len(high_predictions), len(true_targets)}
    if len(sizes) != 1:
        raise ValueError("Gate, expert, and target arrays must have equal length")
    return {
        "oracle_route": np.where(
            true_targets >= tc_threshold_k,
            high_predictions,
            low_predictions,
        ),
        "hard_route": np.where(
            gate_probabilities >= probability_threshold,
            high_predictions,
            low_predictions,
        ),
        "soft_route": (
            (1.0 - gate_probabilities) * low_predictions
            + gate_probabilities * high_predictions
        ),
    }


def summarize(
    moe_root: Path,
    global_benchmark_root: Path,
    seeds: list[int],
    tc_threshold_k: float,
    min_gate_recall: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    classification_records: list[dict] = []
    regression_records: list[dict] = []
    prediction_tables: list[pd.DataFrame] = []
    difference_records: list[dict] = []

    for seed in seeds:
        seed_root = moe_root / f"seed_{seed}"
        gate_root = seed_root / "gate"
        low_root = seed_root / "low_expert"
        high_root = seed_root / "high_expert"
        global_root = global_benchmark_root / "diffusion_base_pretrained" / f"seed_{seed}"

        val_probs = np.load(gate_root / "val_probs.npy").reshape(-1)
        val_classes = np.load(gate_root / "val_targets.npy").reshape(-1).astype(int)
        probability_threshold, validation_gate = choose_probability_threshold(
            val_classes,
            val_probs,
            min_recall=min_gate_recall,
        )
        test_probs = np.load(gate_root / "test_probs.npy").reshape(-1)
        test_classes = np.load(gate_root / "test_targets.npy").reshape(-1).astype(int)
        test_gate = classification_metrics(test_classes, test_probs, probability_threshold)
        classification_records.extend([
            {"seed": seed, "split": "validation", **validation_gate},
            {"seed": seed, "split": "test", **test_gate},
        ])

        targets = np.load(global_root / "test_targets.npy").reshape(-1)
        global_predictions = np.load(global_root / "test_preds.npy").reshape(-1)
        low_predictions = np.load(low_root / "test_preds.npy").reshape(-1)
        high_predictions = np.load(high_root / "test_preds.npy").reshape(-1)
        low_targets = np.load(low_root / "test_targets.npy").reshape(-1)
        high_targets = np.load(high_root / "test_targets.npy").reshape(-1)
        expected_classes = (targets >= tc_threshold_k).astype(int)
        if not (
            np.array_equal(targets, low_targets)
            and np.array_equal(targets, high_targets)
            and np.array_equal(test_classes, expected_classes)
        ):
            raise RuntimeError(f"Test target ordering differs for seed {seed}")

        routes = routed_predictions(
            test_probs,
            low_predictions,
            high_predictions,
            targets,
            tc_threshold_k,
            probability_threshold,
        )
        methods = {
            "global_regressor": global_predictions,
            "low_expert": low_predictions,
            "high_expert": high_predictions,
            **routes,
        }
        scopes = {
            "all": np.ones(len(targets), dtype=bool),
            "true_low": targets < tc_threshold_k,
            "true_high": targets >= tc_threshold_k,
        }
        seed_metrics: dict[tuple[str, str], dict] = {}
        for method, predictions in methods.items():
            for scope, mask in scopes.items():
                metrics = regression_metrics(
                    targets[mask],
                    predictions[mask],
                    bootstrap_seed=seed,
                )
                seed_metrics[(method, scope)] = metrics
                regression_records.append({
                    "seed": seed,
                    "method": method,
                    "scope": scope,
                    **metrics,
                })

        global_all = seed_metrics[("global_regressor", "all")]
        for method in ("oracle_route", "hard_route", "soft_route"):
            candidate = seed_metrics[(method, "all")]
            difference_records.append({
                "seed": seed,
                "comparison": f"{method}_minus_global_regressor",
                "delta_mae_k": candidate["mae_k"] - global_all["mae_k"],
                "delta_rmse_k": candidate["rmse_k"] - global_all["rmse_k"],
                "delta_r2": candidate["r2"] - global_all["r2"],
            })

        table = pd.DataFrame({
            "seed": seed,
            "structure_index": np.arange(len(targets)),
            "target_tc_k": targets,
            "true_high_tc": expected_classes,
            "gate_probability": test_probs,
            "gate_hard_class": (test_probs >= probability_threshold).astype(int),
            "global_prediction_k": global_predictions,
            "low_expert_prediction_k": low_predictions,
            "high_expert_prediction_k": high_predictions,
            "oracle_prediction_k": routes["oracle_route"],
            "hard_prediction_k": routes["hard_route"],
            "soft_prediction_k": routes["soft_route"],
        })
        prediction_tables.append(table)

    classification = pd.DataFrame(classification_records)
    regression = pd.DataFrame(regression_records)
    metric_columns = [
        column for column in regression.columns
        if column not in {"seed", "method", "scope", "n_test"}
    ]
    aggregate = (
        regression.groupby(["method", "scope"])[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in aggregate.columns.to_flat_index()
    ]
    classification_metric_columns = [
        column for column in classification.columns
        if column not in {"seed", "split", "n", "n_high", "tn", "fp", "fn", "tp"}
    ]
    classification_aggregate = (
        classification.groupby("split")[classification_metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    classification_aggregate.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in classification_aggregate.columns.to_flat_index()
    ]

    moe_root.mkdir(parents=True, exist_ok=True)
    classification.to_csv(moe_root / "gate_metrics_per_seed.csv", index=False)
    classification_aggregate.to_csv(moe_root / "gate_metrics_aggregate.csv", index=False)
    regression.to_csv(moe_root / "regression_metrics_per_seed.csv", index=False)
    aggregate.to_csv(moe_root / "regression_metrics_aggregate.csv", index=False)
    pd.DataFrame(difference_records).to_csv(
        moe_root / "routing_differences.csv", index=False
    )
    pd.concat(prediction_tables, ignore_index=True).to_csv(
        moe_root / "test_predictions.csv", index=False
    )
    summary = {
        "experiment": "Magndata-Tc gated mixture of experts",
        "tc_threshold_k": float(tc_threshold_k),
        "min_gate_validation_recall": float(min_gate_recall),
        "seeds": seeds,
        "classification": classification.to_dict(orient="records"),
        "regression": regression.to_dict(orient="records"),
    }
    (moe_root / "moe_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return classification, regression


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moe-root", type=Path, required=True)
    parser.add_argument("--global-benchmark-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 3407])
    parser.add_argument("--tc-threshold-k", type=float, default=300.0)
    parser.add_argument("--min-gate-recall", type=float, default=0.80)
    return parser


def cli(argv=None) -> None:
    args = build_parser().parse_args(argv)
    classification, regression = summarize(
        args.moe_root.resolve(),
        args.global_benchmark_root.resolve(),
        args.seeds,
        args.tc_threshold_k,
        args.min_gate_recall,
    )
    print(classification.to_string(index=False))
    print(regression[regression.scope == "all"].to_string(index=False))


if __name__ == "__main__":
    cli()
