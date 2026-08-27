"""Summarize the strict four-arm Magndata-Tc benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


METHODS = (
    "m3gnet_scratch",
    "diffusion_scratch",
    "diffusion_base_pretrained",
    "diffusion_joint_pretrained",
)


def regression_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    bootstrap_seed: int,
) -> dict[str, float | int]:
    targets = np.asarray(targets, dtype=float).reshape(-1)
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    if len(targets) != len(predictions) or not len(targets):
        raise ValueError("Targets and predictions must have the same non-zero length")
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError("Targets and predictions must be finite")

    errors = predictions - targets
    absolute_errors = np.abs(errors)
    denominator = np.sum((targets - targets.mean()) ** 2)
    r2 = 1.0 - np.sum(errors ** 2) / denominator if denominator else np.nan
    pearson = np.corrcoef(targets, predictions)[0, 1]
    spearman = spearmanr(targets, predictions).statistic

    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_indices = rng.integers(0, len(targets), size=(2000, len(targets)))
    bootstrap_mae = absolute_errors[bootstrap_indices].mean(axis=1)
    return {
        "n_test": int(len(targets)),
        "mae_k": float(absolute_errors.mean()),
        "mae_ci95_low_k": float(np.quantile(bootstrap_mae, 0.025)),
        "mae_ci95_high_k": float(np.quantile(bootstrap_mae, 0.975)),
        "rmse_k": float(np.sqrt(np.mean(errors ** 2))),
        "median_ae_k": float(np.median(absolute_errors)),
        "max_ae_k": float(absolute_errors.max()),
        "bias_k": float(errors.mean()),
        "r2": float(r2),
        "pearson_r": float(pearson),
        "spearman_rho": float(spearman),
    }


def summarize(benchmark_root: Path, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict] = []
    reference_targets: np.ndarray | None = None
    predictions_table: dict[str, np.ndarray] = {}
    for method in METHODS:
        for seed in seeds:
            run_dir = benchmark_root / method / f"seed_{seed}"
            predictions_path = run_dir / "test_preds.npy"
            targets_path = run_dir / "test_targets.npy"
            if not predictions_path.is_file() or not targets_path.is_file():
                raise FileNotFoundError(f"Incomplete benchmark run: {run_dir}")
            predictions = np.load(predictions_path).reshape(-1)
            targets = np.load(targets_path).reshape(-1)
            if reference_targets is None:
                reference_targets = targets
                predictions_table["target_tc_k"] = targets
            elif not np.array_equal(reference_targets, targets):
                raise RuntimeError(f"Test targets differ in {run_dir}")
            predictions_table[f"{method}_seed_{seed}"] = predictions
            records.append(
                {
                    "method": method,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    **regression_metrics(targets, predictions, bootstrap_seed=seed),
                }
            )

    per_run = pd.DataFrame(records)
    metric_columns = [
        column
        for column in per_run.columns
        if column not in {"method", "seed", "run_dir", "n_test"}
    ]
    aggregate = (
        per_run.groupby("method")[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in aggregate.columns.to_flat_index()
    ]

    benchmark_root.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(benchmark_root / "per_run_metrics.csv", index=False)
    aggregate.to_csv(benchmark_root / "aggregate_metrics.csv", index=False)
    pd.DataFrame(predictions_table).to_csv(
        benchmark_root / "test_predictions.csv", index=False
    )

    paired_records: list[dict] = []
    for seed in seeds:
        indexed = per_run[per_run["seed"] == seed].set_index("method")
        comparisons = (
            ("diffusion_base_pretrained", "m3gnet_scratch"),
            ("diffusion_base_pretrained", "diffusion_scratch"),
            ("diffusion_joint_pretrained", "m3gnet_scratch"),
            ("diffusion_joint_pretrained", "diffusion_scratch"),
            ("diffusion_joint_pretrained", "diffusion_base_pretrained"),
        )
        for candidate, comparator in comparisons:
            paired_records.append(
                {
                    "seed": seed,
                    "comparison": f"{candidate}_minus_{comparator}",
                    "delta_mae_k": float(
                        indexed.loc[candidate, "mae_k"]
                        - indexed.loc[comparator, "mae_k"]
                    ),
                    "delta_rmse_k": float(
                        indexed.loc[candidate, "rmse_k"]
                        - indexed.loc[comparator, "rmse_k"]
                    ),
                    "delta_r2": float(
                        indexed.loc[candidate, "r2"]
                        - indexed.loc[comparator, "r2"]
                    ),
                }
            )
    pd.DataFrame(paired_records).to_csv(
        benchmark_root / "paired_differences.csv", index=False
    )

    payload = {
        "benchmark": "Magndata-Tc strict four-arm comparison",
        "target": "Curie temperature",
        "unit": "K",
        "seeds": seeds,
        "primary_metric": "test MAE (K)",
        "lower_is_better": ["mae_k", "rmse_k", "median_ae_k"],
        "higher_is_better": ["r2", "pearson_r", "spearman_rho"],
        "per_run": per_run.to_dict(orient="records"),
        "aggregate": aggregate.to_dict(orient="records"),
    }
    (benchmark_root / "benchmark_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return per_run, aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 3407])
    return parser


def cli(argv=None) -> None:
    args = build_parser().parse_args(argv)
    per_run, aggregate = summarize(args.benchmark_root.resolve(), args.seeds)
    print(per_run.to_string(index=False))
    print("\nAggregate metrics:\n" + aggregate.to_string(index=False))


if __name__ == "__main__":
    cli()
