"""Diagnose why step-39 GRPO+PIPO increases the Fig. 3a S.U.N. rate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "assets" / "crystalpirl_article_blueprint" / "source"
OUTPUT = SOURCE / "sun_gain_diagnostic"
QUALITY = SOURCE / "fig3_quality_per_structure.csv"
QUALITY_SUMMARY = SOURCE / "fig3_quality_metrics.csv"
PROFILES = SOURCE / "fig4c_structure_profiles.csv"

METHODS = {
    "Base": ROOT
    / "output/singlerun/2026-06-27/00-32-50-mp20_base/evaluations/property_predictions/eval_properties_gen_abinitio_empirical_uncond_n4096_seed42_predictor_seed42.csv",
    "GRPO+PIPO-FE (step 39)": ROOT
    / "output/reinforcement_learning/2026-08-24/05-11-36-rl-target-generation/grpo_fe_seed42_pipo/model/evaluations/property_predictions/eval_properties_gen_abinitio_empirical_rl_fe_m1p5_n4096_seed42_predictor_seed42.csv",
    "GRPO+PIPO-BG (step 39)": ROOT
    / "output/reinforcement_learning/2026-08-24/05-11-36-rl-target-generation/grpo_bg_seed42_pipo/model/evaluations/property_predictions/eval_properties_gen_abinitio_empirical_rl_bg_2_n4096_seed42_predictor_seed42.csv",
}

TRAINING_LOGS = {
    "FE": ROOT
    / "logs/reinforcement_learning/slurm/20260823-214735-grpo-pipo-probe32/task_0/training.log",
    "BG": ROOT
    / "logs/reinforcement_learning/slurm/20260823-214735-grpo-pipo-probe32/task_1/training.log",
}


def _joined_quality() -> pd.DataFrame:
    quality = pd.read_csv(QUALITY)
    profiles = pd.read_csv(PROFILES).rename(
        columns={"source_index": "structure_index"}
    )
    selected = quality[quality["method"].isin(METHODS)].merge(
        profiles[["method", "structure_index", "n_elements", "formula"]],
        on=["method", "structure_index"],
        how="left",
        validate="one_to_one",
    )
    predictions = []
    for method, path in METHODS.items():
        frame = pd.read_csv(path)[["structure_index", "predicted_band_gap"]]
        frame.insert(0, "method", method)
        predictions.append(frame)
    return selected.merge(
        pd.concat(predictions, ignore_index=True),
        on=["method", "structure_index"],
        how="left",
        validate="one_to_one",
    )


def _method_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        frame = data[data["method"] == method]
        valid = frame[frame["valid"]]
        rows.append(
            {
                "method": method,
                "n_total": len(frame),
                "n_valid": int(frame["valid"].sum()),
                "n_stable": int(frame["stable"].sum()),
                "n_sun": int(frame["sun"].sum()),
                "validity": float(frame["valid"].mean()),
                "stability_all": float(frame["stable"].mean()),
                "stability_given_valid": float(valid["stable"].mean()),
                "uniqueness_given_valid": float(
                    frame["unique_representative"].sum()
                    / frame["valid"].sum()
                ),
                "novelty_given_unique": float(
                    frame["novel_vs_mp20_train"].sum()
                    / frame["unique_representative"].sum()
                ),
                "sun": float(frame["sun"].mean()),
                "sun_given_stable": float(
                    frame["sun"].sum() / frame["stable"].sum()
                ),
                "unary_binary_fraction_given_valid": float(
                    (valid["n_elements"] <= 2).mean()
                ),
                "shallow_mp_hull_fraction_given_valid": float(
                    (valid["mp_hull_formation_energy_per_atom"] > -0.5).mean()
                ),
                "formation_energy_mean_given_valid": float(
                    valid["predicted_formation_energy_per_atom"].mean()
                ),
                "mp_hull_energy_mean_given_valid": float(
                    valid["mp_hull_formation_energy_per_atom"].mean()
                ),
                "e_above_hull_median_given_valid": float(
                    valid["predicted_e_above_hull"].median()
                ),
                "fe_target_hit_fraction_all": float(
                    (
                        frame["valid"]
                        & frame["predicted_formation_energy_per_atom"].between(
                            -1.8, -1.2
                        )
                    ).mean()
                ),
                "bg_target_hit_fraction_all": float(
                    (
                        frame["valid"]
                        & frame["predicted_band_gap"].between(1.55, 2.45)
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _by_n_elements(data: pd.DataFrame) -> pd.DataFrame:
    table = (
        data.groupby(["method", "n_elements"], as_index=False)
        .agg(
            n_total=("sun", "size"),
            n_valid=("valid", "sum"),
            n_stable=("stable", "sum"),
            n_sun=("sun", "sum"),
        )
    )
    base = table[table["method"] == "Base"].set_index("n_elements")
    table["stable_delta_n_vs_base"] = table.apply(
        lambda row: row.n_stable
        - int(base.n_stable.get(row.n_elements, 0)),
        axis=1,
    )
    table["sun_delta_n_vs_base"] = table.apply(
        lambda row: row.n_sun - int(base.n_sun.get(row.n_elements, 0)),
        axis=1,
    )
    return table


def _training_steps() -> pd.DataFrame:
    rows = []
    for task, path in TRAINING_LOGS.items():
        for line in path.read_text(errors="replace").splitlines():
            try:
                record = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if "step" not in record or int(record["step"]) > 39:
                continue
            reward = record["reward"]
            validity = float(reward["valid_fraction"])
            raw_reward = float(reward["reward_mean"])
            property_mean = next(iter(reward["property_means"].values()))
            rows.append(
                {
                    "task": task,
                    "step": int(record["step"]),
                    "raw_reward_mean": raw_reward,
                    "valid_fraction": validity,
                    "inferred_valid_property_score": (
                        raw_reward + 1.0 - validity
                    )
                    / validity,
                    "property_prediction_mean": float(property_mean),
                }
            )
    return pd.DataFrame(rows).sort_values(["task", "step"])


def _training_windows(steps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, frame in steps.groupby("task", sort=False):
        for window, selected in (
            ("steps_0_7", frame[frame["step"] <= 7]),
            ("steps_32_39", frame[frame["step"] >= 32]),
        ):
            rows.append(
                {
                    "task": task,
                    "window": window,
                    "n_updates": len(selected),
                    "raw_reward_mean": selected["raw_reward_mean"].mean(),
                    "valid_fraction_mean": selected["valid_fraction"].mean(),
                    "valid_property_score_mean": selected[
                        "inferred_valid_property_score"
                    ].mean(),
                    "property_prediction_mean": selected[
                        "property_prediction_mean"
                    ].mean(),
                }
            )
    return pd.DataFrame(rows)


def _validate_counts(summary: pd.DataFrame) -> None:
    expected = pd.read_csv(QUALITY_SUMMARY).set_index("method")
    actual = summary.set_index("method")
    for method in METHODS:
        for column in ("n_total", "n_valid", "n_stable", "n_sun"):
            if int(actual.loc[method, column]) != int(expected.loc[method, column]):
                raise RuntimeError(f"{method} {column} does not match Fig. 3a")


def _write_report(summary: pd.DataFrame, by_elements: pd.DataFrame) -> None:
    indexed = summary.set_index("method")
    base = indexed.loc["Base"]
    lines = [
        "# Step-39 GRPO+PIPO S.U.N. gain diagnostic",
        "",
        "All values use the same seed=42, 4,096-structure Ab initio sets as Fig. 3a.",
        "This is a within-seed descriptive decomposition, not a multi-seed uncertainty claim.",
        "",
        "## Main finding",
        "",
        "The S.U.N. increase is almost entirely a post-hoc stability increase caused by a shift toward unary and binary compositions with shallow MP formation-energy hulls. It is not evidence of a staged general-material reward phase, because stability, uniqueness and novelty were not part of the step-39 GRPO+PIPO training reward.",
        "",
        "## Exact decomposition",
        "",
        "| Method | S.U.N. | Stability | Unary+binary among valid | Median predicted E above hull | FE hit | BG hit |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in indexed.iterrows():
        lines.append(
            f"| {method} | {100*row.sun:.2f}% | {100*row.stability_all:.2f}% | "
            f"{100*row.unary_binary_fraction_given_valid:.2f}% | "
            f"{row.e_above_hull_median_given_valid:.4f} eV/atom | "
            f"{100*row.fe_target_hit_fraction_all:.2f}% | "
            f"{100*row.bg_target_hit_fraction_all:.2f}% |"
        )
    lines.extend(["", "## Low-order composition contribution", ""])
    for method in list(METHODS)[1:]:
        row = indexed.loc[method]
        subset = by_elements[
            (by_elements["method"] == method)
            & (by_elements["n_elements"] <= 2)
        ]
        low_order_sun_delta = int(subset["sun_delta_n_vs_base"].sum())
        total_sun_delta = int(row.n_sun - base.n_sun)
        lines.append(
            f"- {method}: S.U.N. count increased by {total_sun_delta}; unary+binary strata alone contributed {low_order_sun_delta} additional S.U.N. structures, while losses in higher-order compositions offset part of that gain."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Unary elemental compositions have an MP formation-energy hull near 0 eV/atom and are unusually easy to pass the 0.1 eV/atom stability threshold.",
            "- The PIPO sets moved to shallower composition-dependent hulls; their absolute formation energies did not move toward the FE target.",
            "- Validity, uniqueness and novelty did not improve enough to explain the S.U.N. increase.",
            "- The result is consistent with a reward/distribution shortcut and must not be described as successful target-property learning.",
            "- Causality at the model-training level still requires checkpoint-matched, common-random-number comparisons and additional seeds.",
            "",
        ]
    )
    (OUTPUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = _joined_quality()
    summary = _method_summary(data)
    by_elements = _by_n_elements(data)
    training_steps = _training_steps()
    training_windows = _training_windows(training_steps)
    _validate_counts(summary)
    summary.to_csv(OUTPUT / "method_summary.csv", index=False)
    by_elements.to_csv(OUTPUT / "sun_gain_by_n_elements.csv", index=False)
    training_steps.to_csv(OUTPUT / "training_reward_steps_0_39.csv", index=False)
    training_windows.to_csv(
        OUTPUT / "training_reward_window_summary.csv", index=False
    )
    _write_report(summary, by_elements)
    print(OUTPUT)


if __name__ == "__main__":
    main()
