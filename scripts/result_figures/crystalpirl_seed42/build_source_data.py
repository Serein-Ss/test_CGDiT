"""Freeze update- and structure-level Source Data for Fig. 1–Fig. 4."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_ROOT = Path("output/singlerun/2026-06-27/00-32-50-mp20_base")
CFG_ROOTS = {
    "fe": Path("output/singlerun/2026-06-28/16-46-08-mp20_fe"),
    "bg": Path("output/singlerun/2026-06-30/11-28-44-mp20_bg"),
}
TARGETS = {"fe": -1.5, "bg": 2.0}
TOLERANCES = {"fe": 0.30, "bg": 0.45}
PROPERTY_COLUMNS = {
    "fe": "formation_energy_per_atom",
    "bg": "band_gap",
}
BASE_LABELS = {
    "abinitio": "abinitio_empirical_uncond_n4096_seed42",
}
CFG_LABELS = {
    "fe": {
        "abinitio": "abinitio_empirical_fe_m1p5_n4096_seed42",
    },
    "bg": {
        "abinitio": "abinitio_empirical_bg_2_n4096_seed42",
    },
}


def _method_info(run_name: str) -> dict[str, str]:
    algorithm, task, _ = run_name.split("_", 2)
    if run_name.endswith("_pirl"):
        verifier = "crystalpirl"
        display = f"CrystalPIRL-{algorithm.upper()}"
    elif run_name.endswith("_pipo"):
        verifier = "pipo"
        display = f"PIPO-{algorithm.upper()}"
    else:
        verifier = "open"
        display = f"Open-{algorithm.upper()}"
    return {
        "algorithm": algorithm,
        "task": task,
        "verifier": verifier,
        "method": display,
    }


def _run_names() -> list[str]:
    names = []
    for task in ("fe", "bg"):
        for algorithm in ("ppo", "grpo"):
            names.extend(
                [
                    f"{algorithm}_{task}_seed42",
                    f"{algorithm}_{task}_seed42_pipo",
                    f"{algorithm}_{task}_seed42_pirl",
                ]
            )
    return names


def _dual_value(decision: dict[str, Any] | None, anchor: str, field: str) -> float:
    if not decision or anchor not in decision:
        return float("nan")
    values = decision[anchor].get(field)
    if not values:
        return float("nan")
    return float(values[0])


def _probe_row(probe: dict[str, Any] | None, prefix: str) -> dict[str, Any]:
    if probe is None:
        return {
            f"{prefix}_action": None,
            f"{prefix}_local_mean": np.nan,
            f"{prefix}_local_lcb": np.nan,
            f"{prefix}_absolute_mean": np.nan,
            f"{prefix}_absolute_lcb": np.nan,
            f"{prefix}_candidate_reward": np.nan,
            f"{prefix}_current_reward": np.nan,
            f"{prefix}_base_reward": np.nan,
            f"{prefix}_safety_complete": False,
        }
    decision = probe["decision"]
    return {
        f"{prefix}_action": decision["action"],
        f"{prefix}_local_mean": _dual_value(decision, "local", "mean_delta"),
        f"{prefix}_local_lcb": _dual_value(
            decision, "local", "lower_confidence_bound"
        ),
        f"{prefix}_absolute_mean": _dual_value(
            decision, "absolute", "mean_delta"
        ),
        f"{prefix}_absolute_lcb": _dual_value(
            decision, "absolute", "lower_confidence_bound"
        ),
        f"{prefix}_candidate_reward": float(
            probe["candidate"]["reward_mean"]
        ),
        f"{prefix}_current_reward": float(probe["current"]["reward_mean"]),
        f"{prefix}_base_reward": float(probe["base"]["reward_mean"]),
        f"{prefix}_safety_complete": bool(
            probe["safety_audit"]["formal_complete"]
        ),
    }


def build_update_tables(
    train_root: Path, expected_updates: int = 8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    update_rows = []
    framework_rows = []
    for run_name in _run_names():
        info = _method_info(run_name)
        path = train_root / run_name / "train_results" / "metrics.jsonl"
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(records) != expected_updates:
            raise RuntimeError(
                f"Expected {expected_updates} updates in {path}, got {len(records)}"
            )
        for record in records:
            audit = record.get("audit") or {}
            fixed = audit.get("fixed")
            holdout = audit.get("holdout")
            update_applied = (
                bool(record["decision"]["verified_checkpoint_update"])
                if info["verifier"] == "crystalpirl"
                else True
            )
            row = {
                "run_id": run_name,
                "seed": 42,
                "update_id": int(record["step"]),
                **info,
                "update_applied": update_applied,
                "training_reward_mean": float(record["reward"]["reward_mean"]),
                "training_valid_fraction": float(
                    record["reward"]["valid_fraction"]
                ),
                "loss": float(record["loss"]),
                "gradient_norm": float(record["gradient_norm"]),
                "approx_kl": float(record["approx_kl"]),
                "clip_fraction": float(record["clip_fraction"]),
                "ratio_mean": float(record["ratio_mean"]),
                "checkpoint": record["checkpoint"],
                "pipo_modulation": (
                    record["pipo_feedback"]["feedback"]["modulation"]
                    if record.get("pipo_feedback")
                    and record["pipo_feedback"].get("feedback")
                    else np.nan
                ),
                "pipo_retrospective_update": bool(
                    record.get("pipo_feedback")
                    and record["pipo_feedback"].get("retrospective_update")
                ),
                "trajectory_transitions": record["trajectory"]["num_transitions"],
                "trajectory_stochastic_transitions": record["trajectory"][
                    "num_stochastic_transitions"
                ],
                "mean_orbits_per_structure": float(
                    np.mean(record["trajectory"]["num_orbits_per_structure"])
                ),
            }
            for channel in ("lattice", "coord", "atom"):
                channel_data = record["trajectory"]["channel_log_prob"][channel]
                row[f"{channel}_log_prob_mean"] = channel_data["mean"]
                row[f"{channel}_log_prob_std"] = channel_data["std"]
            row.update(_probe_row(fixed, "fixed"))
            row.update(_probe_row(holdout, "holdout"))
            row["bad_applied_update"] = bool(
                update_applied
                and (
                    not np.isfinite(row["holdout_absolute_lcb"])
                    or row["holdout_local_lcb"] <= 0
                    or row["holdout_absolute_lcb"] <= 0
                    or not row["holdout_safety_complete"]
                )
            )
            update_rows.append(row)
            if info["verifier"] == "crystalpirl":
                framework_rows.append(row.copy())
    return pd.DataFrame(framework_rows), pd.DataFrame(update_rows)


def _artifacts(model_root: Path, label: str) -> dict[str, Path]:
    evaluation = model_root / "evaluations"
    generation = list(model_root.rglob(f"eval_gen_{label}.pt"))
    if len(generation) != 1:
        raise RuntimeError(
            f"Expected one generation file for {model_root}/{label}, got {generation}"
        )
    paths = {
        "generation": generation[0],
        "independent": evaluation
        / "property_predictions"
        / f"eval_properties_gen_{label}_predictor_seed123.csv",
        "reward": evaluation
        / "property_predictions"
        / f"eval_properties_gen_{label}_predictor_seed42_reward.csv",
        "property_metrics": evaluation
        / "property_metrics"
        / f"eval_property_metrics_gen_{label}_predictor_seed123.json",
        "structural_metrics": evaluation
        / "structural_metrics"
        / f"eval_metrics_gen_{label}.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))
    return paths


def _group_specs(train_root: Path, rl_samples: int = 4096) -> list[dict[str, Any]]:
    specs = []
    for mode, label in BASE_LABELS.items():
        specs.append(
            {
                "group_id": f"base_{mode}",
                "method": "Base",
                "method_family": "base",
                "task": None,
                "generation_mode": mode,
                "model_root": BASE_ROOT,
                "label": label,
            }
        )
    for task in ("fe", "bg"):
        for mode, label in CFG_LABELS[task].items():
            specs.append(
                {
                    "group_id": f"cfg_{task}_{mode}",
                    "method": f"CFG-{task.upper()}",
                    "method_family": "cfg",
                    "task": task,
                    "generation_mode": mode,
                    "model_root": CFG_ROOTS[task],
                    "label": label,
                }
            )
    for run_name in _run_names():
        info = _method_info(run_name)
        target_label = "fe_m1p5" if info["task"] == "fe" else "bg_2"
        specs.append(
            {
                "group_id": f"{run_name}_abinitio",
                "method": info["method"],
                "method_family": info["verifier"],
                "algorithm": info["algorithm"],
                "task": info["task"],
                "generation_mode": "abinitio",
                "model_root": train_root / run_name / "model",
                "label": f"abinitio_empirical_uncond_policy_{target_label}_n{rl_samples}_seed42",
            }
        )
    return specs


def _structure_table(spec: dict[str, Any], task: str) -> pd.DataFrame:
    artifacts = _artifacts(spec["model_root"], spec["label"])
    independent = pd.read_csv(artifacts["independent"])
    reward = pd.read_csv(artifacts["reward"])
    property_name = PROPERTY_COLUMNS[task]
    column = f"predicted_{property_name}"
    required = {"structure_index", "valid", column}
    if not required.issubset(independent.columns):
        raise RuntimeError(f"Missing {required - set(independent.columns)} in {artifacts['independent']}")
    merged = independent.merge(
        reward[["structure_index", column]].rename(
            columns={column: "reward_prediction"}
        ),
        on="structure_index",
        validate="one_to_one",
    )
    merged = merged.rename(columns={column: "independent_prediction"})
    merged["task"] = task
    merged["method"] = spec["method"]
    merged["method_family"] = spec["method_family"]
    merged["algorithm"] = spec.get("algorithm")
    merged["generation_mode"] = spec["generation_mode"]
    merged["seed"] = 42
    merged["target"] = TARGETS[task]
    merged["tolerance"] = TOLERANCES[task]
    merged["absolute_target_error"] = (
        merged["independent_prediction"] - TARGETS[task]
    ).abs()
    merged["hit"] = merged["absolute_target_error"] <= TOLERANCES[task]
    merged["valid_target_yield"] = merged["valid"].astype(bool) & merged["hit"]
    merged["predictor_gap"] = (
        merged["independent_prediction"] - merged["reward_prediction"]
    )
    merged["group_id"] = spec["group_id"]
    merged["structure_id"] = (
        spec["group_id"] + ":" + merged["structure_index"].astype(str)
    )
    merged["source_generation"] = str(artifacts["generation"])
    return merged


def _best_of_eight(base: pd.DataFrame, task: str) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    selected = []
    for mode, frame in base.groupby("generation_mode", sort=False):
        frame = frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)
        usable = len(frame) - len(frame) % 8
        for start in range(0, usable, 8):
            group = frame.iloc[start : start + 8]
            score = (
                group["reward_prediction"]
                if task == "fe"
                else (group["reward_prediction"] - TARGETS[task]).abs()
            )
            index = score.idxmin()
            selected.append(group.loc[index].copy())
    result = pd.DataFrame(selected).reset_index(drop=True)
    result["method"] = "Best-of-8"
    result["method_family"] = "best_of_n"
    result["algorithm"] = None
    result["group_id"] = "best8_" + task + "_" + result["generation_mode"]
    result["structure_id"] = (
        result["group_id"] + ":" + result["structure_index"].astype(str)
    )
    result["selection_queries"] = 8
    return result


def build_structure_tables(
    train_root: Path, rl_samples: int = 4096
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = _group_specs(train_root, rl_samples)
    rows = []
    quality_rows = []
    for task in ("fe", "bg"):
        task_specs = [
            spec
            for spec in specs
            if spec["task"] in (None, task)
            and not (
                spec["method_family"] == "cfg" and spec["task"] != task
            )
        ]
        task_tables = [_structure_table(spec, task) for spec in task_specs]
        task_data = pd.concat(task_tables, ignore_index=True)
        base = task_data[task_data["method_family"] == "base"]
        task_data = pd.concat(
            [task_data, _best_of_eight(base, task)], ignore_index=True
        )
        rows.append(task_data)

    for spec in specs:
        task = spec["task"] or "fe"
        table = _structure_table(spec, task)
        artifacts = _artifacts(spec["model_root"], spec["label"])
        structural = json.loads(
            artifacts["structural_metrics"].read_text(encoding="utf-8")
        )
        prop = json.loads(
            artifacts["property_metrics"].read_text(encoding="utf-8")
        )
        finite_fe = table["independent_prediction"].to_numpy(dtype=float)
        if task != "fe":
            independent = pd.read_csv(artifacts["independent"])
            finite_fe = independent[
                "predicted_formation_energy_per_atom"
            ].to_numpy(dtype=float)
        valid = table["valid"].to_numpy(dtype=bool)
        stability = 1.0 / (1.0 + np.exp(finite_fe / 0.30))
        stability = np.where(valid & np.isfinite(stability), stability, 0.0)
        primary = prop["properties"][PROPERTY_COLUMNS[task]]
        quality_rows.append(
            {
                "group_id": spec["group_id"],
                "method": spec["method"],
                "method_family": spec["method_family"],
                "algorithm": spec.get("algorithm"),
                "task": spec["task"],
                "generation_mode": spec["generation_mode"],
                "seed": 42,
                "n": int(prop["n_total"]),
                "validity": float(structural["valid"]),
                "composition_validity": float(structural["comp_valid"]),
                "structure_validity": float(structural["struct_valid"]),
                "stability_proxy_mean": float(np.mean(stability)),
                "coverage_precision": float(structural["cov_precision"]),
                "coverage_recall": float(structural["cov_recall"]),
                "density_wasserstein": float(structural["wdist_density"]),
                "num_elements_wasserstein": float(structural["wdist_num_elems"]),
                "target_mae": primary.get("target_mae"),
                "target_hit_rate": primary.get("hit_rate_all"),
                "predictor_gap_mae": float(table["predictor_gap"].abs().mean()),
                "source_generation": str(artifacts["generation"]),
                "structural_metrics_file": str(artifacts["structural_metrics"]),
                "property_metrics_file": str(artifacts["property_metrics"]),
            }
        )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(quality_rows)


def _summary(fig3: pd.DataFrame) -> pd.DataFrame:
    summary = (
        fig3.groupby(
            [
                "task",
                "method",
                "method_family",
                "algorithm",
                "generation_mode",
                "seed",
            ],
            dropna=False,
        )
        .agg(
            n=("structure_id", "size"),
            finite_prediction_n=("independent_prediction", "count"),
            independent_mean=("independent_prediction", "mean"),
            independent_std=("independent_prediction", "std"),
            target_mae=("absolute_target_error", "mean"),
            hit_rate=("hit", "mean"),
            valid_target_yield=("valid_target_yield", "mean"),
            validity=("valid", "mean"),
            predictor_gap_mae=("predictor_gap", lambda values: values.abs().mean()),
        )
        .reset_index()
    )
    summary["finite_prediction_fraction"] = (
        summary["finite_prediction_n"] / summary["n"]
    )
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_contract(updates_per_run: int, rl_samples: int) -> dict[str, Any]:
    baseline_samples = 4096
    best_of_eight_samples = baseline_samples // 8
    return {
        "fig1_rows": 4 * updates_per_run,
        "fig2_rows": 12 * updates_per_run,
        "fig2_runs": 12,
        "updates_per_run": updates_per_run,
        "fig3_rows": 2
        * (2 * baseline_samples + 6 * rl_samples + best_of_eight_samples),
        "fig3_summary_rows": 18,
        "fig4_groups": 15,
        "rl_final_groups": 12,
        "baseline_samples_per_group": baseline_samples,
        "rl_samples_per_group": rl_samples,
        "best_of_eight_samples_per_group": best_of_eight_samples,
        "fig4_group_sizes": sorted({baseline_samples, rl_samples}),
        "formal_generation_modes": ["abinitio"],
        "seed": 42,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates-per-run", type=int, default=8)
    parser.add_argument("--rl-samples", type=int, default=4096)
    parser.add_argument(
        "--profile", choices=("formal", "quick8h"), default="formal"
    )
    args = parser.parse_args()
    if args.updates_per_run < 1 or args.rl_samples < 8:
        raise ValueError("updates-per-run must be positive and rl-samples must be >= 8")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig1, fig2 = build_update_tables(args.train_root, args.updates_per_run)
    fig3, fig4 = build_structure_tables(args.train_root, args.rl_samples)
    outputs = {
        "fig1_framework_evidence.csv": fig1,
        "fig2_update_audit.csv": fig2,
        "fig3_structure_properties.csv": fig3,
        "fig3_method_summary.csv": _summary(fig3),
        "fig4_quality_summary.csv": fig4,
    }
    for filename, dataframe in outputs.items():
        dataframe.to_csv(args.output_dir / filename, index=False)

    expected = _expected_contract(args.updates_per_run, args.rl_samples)
    fig3_summary = outputs["fig3_method_summary.csv"]
    observed = {
        "fig1_rows": len(fig1),
        "fig2_rows": len(fig2),
        "fig3_rows": len(fig3),
        "fig3_summary_rows": len(fig3_summary),
        "fig4_groups": len(fig4),
        "rl_final_groups": int((fig4["method_family"].isin(["open", "pipo", "crystalpirl"])).sum()),
        "fig4_group_sizes": sorted(fig4["n"].astype(int).unique().tolist()),
        "seed_values": sorted(fig3["seed"].unique().tolist()),
    }
    complete = (
        observed["fig1_rows"] == expected["fig1_rows"]
        and observed["fig2_rows"] == expected["fig2_rows"]
        and observed["fig3_rows"] == expected["fig3_rows"]
        and observed["fig3_summary_rows"] == expected["fig3_summary_rows"]
        and observed["fig4_groups"] == expected["fig4_groups"]
        and observed["rl_final_groups"] == expected["rl_final_groups"]
        and observed["fig4_group_sizes"] == expected["fig4_group_sizes"]
        and observed["seed_values"] == [42]
    )
    completeness = {
        "status": "complete" if complete else "incomplete",
        "scope": f"exploratory_seed42_{args.profile}",
        "expected": expected,
        "observed": observed,
    }
    (args.output_dir / "data_completeness.json").write_text(
        json.dumps(completeness, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "scope": f"exploratory_seed42_{args.profile}",
        "train_root": str(args.train_root),
        "files": {
            filename: {
                "rows": len(dataframe),
                "sha256": _sha256(args.output_dir / filename),
            }
            for filename, dataframe in outputs.items()
        },
    }
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if not complete:
        raise RuntimeError(f"Source Data completeness failed: {observed}")


if __name__ == "__main__":
    main()
