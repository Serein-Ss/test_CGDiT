"""Create publication-style figures for CGDiT generation and predictors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import yaml

from scripts.result_figures.plot_conditioning_effect import (
    plot_conditioned_structures,
    plot_conditioning_distributions,
)


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.dpi": 600,
    }
)

DEFAULT_DPI = 600


BLUE = "#0F4D92"
BLUE_LIGHT = "#7FAAD4"
TEAL = "#33B5A5"
VIOLET = "#7C6CCF"
GRAY = "#767676"
GRAY_LIGHT = "#D9D9D9"
RED = "#B64342"
BLACK = "#272727"

MODEL_ORDER = [
    ("00-32-50-mp20_base", "Base", True),
    ("16-46-08-mp20_fe", "FE", True),
    ("11-28-44-mp20_bg", "BG", True),
    ("07-54-15-mp20_fe_bg", "FE+BG", True),
    ("04-07-02-mp20_eh", "Ehull", False),
    ("18-02-14-mp20_fe_eh", "FE+Ehull", False),
    ("02-53-51-mp20_bg_eh", "BG+Ehull", False),
    ("13-12-29-mp20_fe_bg_eh", "FE+BG+Ehull", False),
]
MODEL_META = {directory: (label, active) for directory, label, active in MODEL_ORDER}
MODEL_RANK = {directory: index for index, (directory, _, _) in enumerate(MODEL_ORDER)}

PROPERTY_NAMES = {
    "formation_energy_per_atom": "FE",
    "band_gap": "BG",
    "e_above_hull": "Ehull",
}


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.08,
        1.015,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def load_generation_groups(project_root: Path) -> list[dict]:
    path = project_root / "assets/model_results/source_data/property_evaluation_seed42/seed42_property_summary.json"
    payload = json.loads(path.read_text())
    groups = payload.get("groups")
    if not isinstance(groups, list) or len(groups) != 23:
        raise ValueError(f"Expected 23 formal generation groups in {path}, found {len(groups or [])}")
    if any(group.get("n_total") != 4096 for group in groups):
        raise ValueError("Every formal generation group must retain n_total=4096")
    return sorted(groups, key=generation_sort_key)


def generation_sort_key(group: dict) -> tuple[int, int, str]:
    label = str(group["generation_label"])
    if label.startswith("abinitio"):
        mode_rank = 0
    elif "uncond" not in label:
        mode_rank = 1
    else:
        mode_rank = 2
    return MODEL_RANK[group["generation_model_dir"]], mode_rank, label


def condition_text(group: dict) -> str:
    targets = []
    for prop in ("formation_energy_per_atom", "band_gap", "e_above_hull"):
        value = group.get(f"condition_targets.{prop}")
        if value is not None:
            targets.append(f"{PROPERTY_NAMES[prop]}={float(value):g}")
    return ", ".join(targets) if targets else "uncond"


def generation_display_label(group: dict) -> str:
    model, _ = MODEL_META[group["generation_model_dir"]]
    mode = "Ab initio" if str(group["generation_label"]).startswith("abinitio") else "Template"
    return f"{model} | {mode} | {condition_text(group)}"


def active_count(groups: list[dict]) -> int:
    return sum(MODEL_META[group["generation_model_dir"]][1] for group in groups)


def active_generation_groups(groups: list[dict]) -> list[dict]:
    return [group for group in groups if MODEL_META[group["generation_model_dir"]][1]]


def compact_generation_label(group: dict) -> str:
    model, _ = MODEL_META[group["generation_model_dir"]]
    route = "Ab initio" if str(group["generation_label"]).startswith("abinitio") else "Template"
    if condition_text(group) == "uncond" and model != "Base":
        route += " (uncond.)"
    return f"{model} · {route}"


def plot_generation_structural(groups: list[dict], output_path: Path, dpi: int) -> None:
    shown = active_generation_groups(groups)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 5.2),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.05, 1.0]},
    )
    y = np.arange(len(shown))

    validity = np.asarray([group["structural_metrics.valid"] for group in shown])
    bars = axes[0].barh(y, validity, height=0.58, color=BLUE, alpha=0.88)
    axes[0].set_xlim(0.70, 0.88)
    axes[0].set_xlabel("Overall validity")
    axes[0].set_yticks(y, [compact_generation_label(group) for group in shown])
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", color="#E6E6E6", lw=0.5)
    axes[0].set_axisbelow(True)
    for bar, value in zip(bars, validity):
        axes[0].text(
            value + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{100 * value:.1f}%",
            va="center",
            fontsize=5.5,
        )

    precision = np.asarray([group["structural_metrics.cov_precision"] for group in shown])
    recall = np.asarray([group["structural_metrics.cov_recall"] for group in shown])
    axes[1].barh(y - 0.16, precision, height=0.28, color=VIOLET, label="Precision")
    axes[1].barh(y + 0.16, recall, height=0.28, color=TEAL, label="Recall")
    axes[1].set_xlim(0.58, 1.02)
    axes[1].set_xlabel("Coverage fraction")
    axes[1].set_yticks(y, [])
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", color="#E6E6E6", lw=0.5)
    axes[1].set_axisbelow(True)
    axes[1].legend(loc="lower left")

    for label, ax in zip("ab", axes):
        add_panel_label(ax, label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def axis_range(values: list[float], fraction: float = 0.04) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    span = max(float(array.max() - array.min()), 1e-6)
    return float(array.min() - fraction * span), float(array.max() + fraction * span)


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    residuals = predictions - targets
    sum_squared = float(np.sum(residuals**2))
    total_squared = float(np.sum((targets - targets.mean()) ** 2))
    return {
        "n": int(targets.size),
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "r2": float(1.0 - sum_squared / total_squared),
    }


def calibration_points(targets: np.ndarray, predictions: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.unique(np.quantile(predictions, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        raise ValueError("Too few distinct prediction quantiles for calibration")
    assignments = np.digitize(predictions, edges[1:-1], right=True)
    rows = []
    for index in range(edges.size - 1):
        mask = assignments == index
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin": index + 1,
                "n": int(mask.sum()),
                "mean_prediction": float(predictions[mask].mean()),
                "mean_target": float(targets[mask].mean()),
                "target_sem": float(targets[mask].std(ddof=1) / np.sqrt(mask.sum())),
            }
        )
    return rows


def load_predictor_runs(project_root: Path) -> dict[str, list[dict]]:
    config_path = (
        project_root / "conf/rl/components/rewards/registries/mp20.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    output = {}
    for prop in ("formation_energy_per_atom", "band_gap"):
        specs = [("Seed 42 (active)", config["models"][prop], True)]
        specs.extend(
            (f"Seed {item['seed']} (audit)", item, False)
            for item in config["non_active_seed_runs"][prop]
        )
        runs = []
        reference_targets = None
        for label, spec, active in specs:
            targets = np.load(project_root / spec["test_targets"], allow_pickle=False).reshape(-1)
            predictions = np.load(project_root / spec["test_predictions"], allow_pickle=False).reshape(-1)
            if targets.shape != predictions.shape or not np.isfinite(targets).all() or not np.isfinite(predictions).all():
                raise ValueError(f"Invalid predictor arrays for {prop}, {label}")
            if reference_targets is not None and not np.array_equal(targets, reference_targets):
                raise ValueError(f"Predictor seeds do not share identical test targets for {prop}")
            reference_targets = targets
            runs.append(
                {
                    "property": prop,
                    "label": label,
                    "seed": int(spec["seed"]),
                    "active": active,
                    "targets": targets,
                    "predictions": predictions,
                    "metrics": regression_metrics(targets, predictions),
                    "calibration": calibration_points(targets, predictions),
                }
            )
        output[prop] = runs
    return output


def parity_panel(ax, run: dict, unit: str) -> None:
    targets = run["targets"]
    predictions = run["predictions"]
    low = float(min(targets.min(), predictions.min()))
    high = float(max(targets.max(), predictions.max()))
    pad = 0.035 * max(high - low, 1e-6)
    low -= pad
    high += pad
    cmap = LinearSegmentedColormap.from_list("parity_density", ["#F2F2F2", "#77D7D1", BLUE])
    hexes = ax.hexbin(targets, predictions, gridsize=55, mincnt=1, bins="log", cmap=cmap)
    ax.plot([low, high], [low, high], color=BLACK, lw=0.9, ls="--")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Target ({unit})")
    ax.set_ylabel(f"Prediction ({unit})")
    metrics = run["metrics"]
    ax.text(
        0.04,
        0.96,
        f"n={metrics['n']:,}\nMAE={metrics['mae']:.3f}\nRMSE={metrics['rmse']:.3f}\nR²={metrics['r2']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 2.0},
    )
    colorbar = ax.figure.colorbar(hexes, ax=ax, fraction=0.045, pad=0.025)
    colorbar.set_label("log count", fontsize=5.5)
    colorbar.ax.tick_params(labelsize=5)


def residual_panel(ax, runs: list[dict], unit: str) -> None:
    residual_sets = [run["predictions"] - run["targets"] for run in runs]
    low = min(float(values.min()) for values in residual_sets)
    high = max(float(values.max()) for values in residual_sets)
    bins = np.linspace(low, high, 81)
    colors = [BLUE, GRAY]
    linestyles = ["-", "--"]
    for run, residuals, color, linestyle in zip(runs, residual_sets, colors, linestyles):
        ax.hist(
            residuals,
            bins=bins,
            density=True,
            histtype="step",
            lw=1.35,
            color=color,
            ls=linestyle,
            label=f"{run['label']}; MAE={run['metrics']['mae']:.3f}",
        )
    ax.axvline(0, color=BLACK, lw=0.8, ls=":")
    ax.set_xlim(low, high)
    ax.set_xlabel(f"Prediction − target ({unit})")
    ax.set_ylabel("Density")
    ax.legend()


def calibration_panel(ax, runs: list[dict], unit: str) -> None:
    colors = [BLUE, GRAY]
    markers = ["o", "s"]
    all_values = []
    for run, color, marker in zip(runs, colors, markers):
        points = run["calibration"]
        x = np.asarray([point["mean_prediction"] for point in points])
        y = np.asarray([point["mean_target"] for point in points])
        sem = np.asarray([point["target_sem"] for point in points])
        ax.errorbar(
            x,
            y,
            yerr=sem,
            color=color,
            marker=marker,
            ms=3.1,
            lw=1.0,
            capsize=1.5,
            label=run["label"],
        )
        all_values.extend(x.tolist())
        all_values.extend(y.tolist())
    low, high = axis_range(all_values, fraction=0.05)
    ax.plot([low, high], [low, high], color=BLACK, lw=0.8, ls="--")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel(f"Mean prediction ({unit})")
    ax.set_ylabel(f"Mean target ({unit})")
    ax.legend()


def plot_predictor_performance(runs: dict[str, list[dict]], output_path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.8), constrained_layout=True)
    rows = [
        ("formation_energy_per_atom", "eV atom⁻¹"),
        ("band_gap", "eV"),
    ]
    for row_index, (prop, unit) in enumerate(rows):
        prop_runs = runs[prop]
        active_run = next(run for run in prop_runs if run["active"])
        parity_panel(axes[row_index, 0], active_run, unit)
        residual_panel(axes[row_index, 1], prop_runs, unit)
        calibration_panel(axes[row_index, 2], prop_runs, unit)
    for label, ax in zip("abcdef", axes.flat):
        add_panel_label(ax, label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_source_data(
    groups: list[dict], runs: dict[str, list[dict]], output_dir: Path, dpi: int
) -> None:
    source_dir = output_dir / "source_data"
    generation_columns = [
        "display_label",
        "active_scope",
        "generation_model_dir",
        "generation_label",
        "n_total",
        "n_graph_success",
        "graph_success_rate",
        "structural_metrics.comp_valid",
        "structural_metrics.struct_valid",
        "structural_metrics.valid",
        "structural_metrics.cov_precision",
        "structural_metrics.cov_recall",
        "structural_metrics.wdist_density",
        "structural_metrics.wdist_num_elems",
        "structural_metrics.amsd_recall",
        "structural_metrics.amsd_precision",
        "structural_metrics.amcd_recall",
        "structural_metrics.amcd_precision",
        "properties.formation_energy_per_atom.mean",
        "properties.formation_energy_per_atom.median",
        "properties.formation_energy_per_atom.p05",
        "properties.formation_energy_per_atom.p95",
        "properties.formation_energy_per_atom.hit_rate_all",
        "condition_targets.formation_energy_per_atom",
        "properties.band_gap.mean",
        "properties.band_gap.median",
        "properties.band_gap.p05",
        "properties.band_gap.p95",
        "properties.band_gap.hit_rate_all",
        "condition_targets.band_gap",
        "joint.joint_hit_rate_all",
    ]
    generation_rows = []
    for group in groups:
        row = dict(group)
        row["display_label"] = generation_display_label(group)
        row["active_scope"] = MODEL_META[group["generation_model_dir"]][1]
        generation_rows.append(row)
    write_csv(source_dir / "generation_summary_plot_data.csv", generation_rows, generation_columns)

    metric_rows = []
    calibration_rows = []
    for prop_runs in runs.values():
        for run in prop_runs:
            metric_rows.append(
                {
                    "property": run["property"],
                    "seed": run["seed"],
                    "active": run["active"],
                    **run["metrics"],
                }
            )
            for point in run["calibration"]:
                calibration_rows.append(
                    {
                        "property": run["property"],
                        "seed": run["seed"],
                        "active": run["active"],
                        **point,
                    }
                )
    write_csv(
        source_dir / "predictor_metrics.csv",
        metric_rows,
        ["property", "seed", "active", "n", "mae", "rmse", "r2"],
    )
    write_csv(
        source_dir / "predictor_calibration_plot_data.csv",
        calibration_rows,
        ["property", "seed", "active", "bin", "n", "mean_prediction", "mean_target", "target_sem"],
    )

    manifest = {
        "backend": "python_matplotlib",
        "export": {"formats": ["png"], "dpi": dpi},
        "generation": {
            "source_groups": len(groups),
            "samples_per_group": sorted({group["n_total"] for group in groups}),
            "active_groups": active_count(groups),
            "archived_groups": len(groups) - active_count(groups),
            "structural_figure_groups": len(active_generation_groups(groups)),
            "property_figure_groups": 9,
            "representative_structures": 4,
            "display_rule": (
                "Figure 1 shows the 11 active FE/BG-scope groups. Figure 2 compares "
                "full property distributions for matched unconditioned and conditioned "
                "groups. Figure 4 shows four valid structures selected by target proximity. "
                "All 23 source groups remain in CSV."
            ),
        },
        "predictors": {
            "properties": ["formation_energy_per_atom", "band_gap"],
            "active_seed": 42,
            "audit_seed": 123,
            "samples_per_run": sorted(
                {run["metrics"]["n"] for prop_runs in runs.values() for run in prop_runs}
            ),
            "excluded": "Ehull predictors are outside the current FE/BG research scope",
        },
        "limitations": [
            "Generation properties are surrogate predictions.",
            "Generation groups have one formal seed, so no replicate uncertainty is inferred.",
            "A final independent FE/BG evaluator has not yet been frozen.",
            "Seed 123 predictor runs are displayed for audit and are not active rewards.",
        ],
    }
    (source_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="CGDiT repository root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/model_results"),
        help="Output directory, relative to project root unless absolute",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="PNG resolution")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir = output_dir.resolve()
    if args.dpi < 300:
        raise ValueError("Publication figures require --dpi >= 300")

    apply_publication_style()
    groups = load_generation_groups(project_root)
    runs = load_predictor_runs(project_root)
    plot_generation_structural(groups, output_dir / "01_generation_structural_quality.png", args.dpi)
    plot_conditioning_distributions(
        project_root,
        output_dir / "02_generation_fe_bg_properties.png",
        args.dpi,
    )
    plot_predictor_performance(runs, output_dir / "03_predictor_fe_bg_performance.png", args.dpi)
    plot_conditioned_structures(
        project_root,
        output_dir / "04_conditioned_structure_examples.png",
        args.dpi,
    )
    export_source_data(groups, runs, output_dir, args.dpi)
    print(f"Saved 4 PNG figures and source data to {output_dir}")


if __name__ == "__main__":
    main()
