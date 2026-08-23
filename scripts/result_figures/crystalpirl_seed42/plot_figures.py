"""Render the audited seed-42 CrystalPIRL pilot figures as PNG files."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


MM = 1 / 25.4
COLORS = {
    "base": "#777777",
    "best_of_n": "#B7B7B7",
    "cfg": "#4C78A8",
    "open": "#F28E2B",
    "pipo": "#9467BD",
    "crystalpirl": "#2A9D8F",
}
MARKERS = {"template": "o", "abinitio": "s"}
LINESTYLES = {"ppo": "-", "grpo": "--"}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def _panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
    )


def _pilot_note(fig: plt.Figure, source: Path) -> None:
    completeness = json.loads(
        (source / "data_completeness.json").read_text(encoding="utf-8")
    )
    note = (
        "quick8h, seed=42"
        if completeness.get("scope", "").endswith("quick8h")
        else "pilot, seed=42"
    )
    fig.text(
        0.995,
        0.005,
        note,
        ha="right",
        va="bottom",
        fontsize=6,
        color="#555555",
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _copy_source_data(source_dir: Path, asset_dir: Path) -> None:
    target = asset_dir / "source_data"
    target.mkdir(parents=True, exist_ok=True)
    for path in source_dir.iterdir():
        if path.suffix.lower() in {".csv", ".json", ".md"}:
            shutil.copy2(path, target / path.name)


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=0.8,
        edgecolor=color,
        facecolor=mpl.colors.to_rgba(color, 0.11),
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=6.5,
    )


def _fig1(source: Path, output: Path) -> None:
    data = pd.read_csv(source / "fig1_framework_evidence.csv")
    fig = plt.figure(figsize=(183 * MM, 108 * MM), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.3, 1, 1])
    ax = fig.add_subplot(grid[:, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _panel(ax, "a")
    _box(ax, (0.05, 0.78), 0.36, 0.12, "Base crystal\ndiffusion policy", COLORS["base"])
    _box(ax, (0.58, 0.78), 0.36, 0.12, "OrbitPO\ntrajectory update", COLORS["open"])
    _box(ax, (0.58, 0.51), 0.36, 0.15, "Reward + validity\n+ safety proxy", COLORS["cfg"])
    _box(ax, (0.05, 0.50), 0.36, 0.17, "Fixed + holdout probes\nlocal and absolute LCB", COLORS["crystalpirl"])
    _box(ax, (0.31, 0.20), 0.40, 0.14, "Accept / scale / reject\nverified checkpoint", COLORS["crystalpirl"])
    arrows = [
        ((0.41, 0.84), (0.58, 0.84)),
        ((0.76, 0.78), (0.76, 0.66)),
        ((0.58, 0.58), (0.41, 0.58)),
        ((0.23, 0.50), (0.38, 0.34)),
        ((0.71, 0.27), (0.92, 0.84)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 0.9})

    ax = fig.add_subplot(grid[0, 1])
    _panel(ax, "b")
    channel_values = [
        data[f"{channel}_log_prob_mean"].to_numpy()
        for channel in ("lattice", "coord", "atom")
    ]
    box = ax.boxplot(
        channel_values,
        labels=["Lattice", "Coordinate", "Atom"],
        patch_artist=True,
        widths=0.55,
        showfliers=False,
    )
    for artist, color in zip(
        box["boxes"], ["#4C78A8", "#F28E2B", "#2A9D8F"]
    ):
        artist.set_facecolor(mpl.colors.to_rgba(color, 0.35))
        artist.set_edgecolor(color)
    ax.set_ylabel("Trajectory channel log probability")

    ax = fig.add_subplot(grid[0, 2])
    _panel(ax, "c")
    orbit_values = data["mean_orbits_per_structure"].to_numpy()
    orbit_bins = np.histogram_bin_edges(orbit_values, bins="auto")
    ax.hist(
        orbit_values,
        bins=orbit_bins,
        color=COLORS["open"],
        alpha=0.75,
        edgecolor="white",
    )
    ax.set_xlabel("Independent orbit count per structure")
    ax.set_ylabel("Verified update records")

    ax = fig.add_subplot(grid[1, 1])
    _panel(ax, "d")
    actions = (
        data["fixed_action"].replace({"attenuate": "scale"})
        .fillna("missing")
        .value_counts()
        .reindex(["accept", "scale", "reject", "missing"], fill_value=0)
    )
    actions = actions[actions > 0]
    ax.bar(
        actions.index,
        actions.values,
        color=[
            {
                "accept": COLORS["crystalpirl"],
                "scale": "#E9C46A",
                "reject": "#D55E00",
                "missing": "#AAAAAA",
            }[action]
            for action in actions.index
        ],
    )
    ax.set_ylabel("Fixed-probe decisions")
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_rotation_mode("anchor")

    ax = fig.add_subplot(grid[1, 2])
    _panel(ax, "e")
    for algorithm, frame in data.groupby("algorithm"):
        style = LINESTYLES[algorithm]
        grouped = frame.groupby("update_id")[
            ["holdout_local_lcb", "holdout_absolute_lcb"]
        ].mean()
        ax.plot(
            grouped.index,
            grouped["holdout_local_lcb"],
            linestyle=style,
            color=COLORS["open"],
            label=f"{algorithm.upper()} local",
        )
        ax.plot(
            grouped.index,
            grouped["holdout_absolute_lcb"],
            linestyle=style,
            color=COLORS["crystalpirl"],
            label=f"{algorithm.upper()} absolute",
        )
    ax.axhline(0, color="#222222", linewidth=0.7)
    ax.set_xlabel("Policy update")
    ax.set_ylabel("Holdout reward LCB")
    ax.legend(ncol=2, loc="best")
    _pilot_note(fig, source)
    _save(fig, output / "fig1_crystalpirl_framework.png")


def _run_label(row: pd.Series) -> str:
    prefix = {
        "open": "Open",
        "pipo": "PIPO",
        "crystalpirl": "CrystalPIRL",
    }[row["verifier"]]
    return f"{prefix}-{row['algorithm'].upper()}"


def _fig2(source: Path, output: Path) -> None:
    data = pd.read_csv(source / "fig2_update_audit.csv")
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(183 * MM, 112 * MM),
        constrained_layout=True,
    )
    for panel, task, ax in zip(("a", "b"), ("fe", "bg"), axes[0, :2]):
        _panel(ax, panel)
        frame = data[data["task"] == task]
        for (_, algorithm, verifier), run in frame.groupby(
            ["run_id", "algorithm", "verifier"]
        ):
            ax.plot(
                run["update_id"],
                run["holdout_absolute_lcb"],
                color=COLORS[verifier],
                linestyle=LINESTYLES[algorithm],
                linewidth=1.2,
                label=_run_label(run.iloc[0]),
            )
        ax.axhline(0, color="#222222", linewidth=0.7)
        ax.set_xlabel("Policy update")
        ax.set_ylabel(f"{task.upper()} holdout absolute LCB")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0, 1].legend(
        by_label.values(), by_label.keys(), ncol=2, loc="best"
    )

    ax = axes[0, 2]
    _panel(ax, "c")
    applied = data[data["update_applied"].astype(bool)]
    rates = (
        applied.groupby(["verifier", "algorithm"])["bad_applied_update"]
        .mean()
        .reset_index()
    )
    display = {
        "open": "Open",
        "pipo": "PIPO",
        "crystalpirl": "CrystalPIRL",
    }
    labels = [
        f"{display[row.verifier]}\n{row.algorithm.upper()}"
        for row in rates.itertuples()
    ]
    ax.bar(
        np.arange(len(rates)),
        rates["bad_applied_update"],
        color=[COLORS[value] for value in rates["verifier"]],
    )
    ax.set_xticks(np.arange(len(rates)), labels, rotation=25, ha="right")
    for label in ax.get_xticklabels():
        label.set_rotation_mode("anchor")
    ax.set_ylabel("Bad applied-update fraction")
    ax.set_ylim(0, 1.02)

    ax = axes[1, 0]
    _panel(ax, "d")
    for (verifier, algorithm), frame in data.groupby(["verifier", "algorithm"]):
        ax.scatter(
            frame["holdout_local_lcb"],
            frame["holdout_absolute_lcb"],
            s=13,
            alpha=0.7,
            marker="o" if algorithm == "ppo" else "s",
            color=COLORS[verifier],
            label=f"{verifier.capitalize()}-{algorithm.upper()}",
        )
    ax.axhline(0, color="#222222", linewidth=0.6)
    ax.axvline(0, color="#222222", linewidth=0.6)
    ax.set_xlabel("Holdout local LCB")
    ax.set_ylabel("Holdout absolute LCB")

    ax = axes[1, 1]
    _panel(ax, "e")
    final = data.sort_values("update_id").groupby("run_id").tail(1)
    x = np.arange(len(final))
    ax.scatter(
        x,
        final["holdout_absolute_lcb"],
        c=[COLORS[value] for value in final["verifier"]],
        marker="o",
        s=24,
    )
    ax.axhline(0, color="#222222", linewidth=0.7)
    ax.set_xticks(
        x,
        [
            f"{row.task.upper()}\n{row.algorithm.upper()}\n{row.verifier}"
            for row in final.itertuples()
        ],
        rotation=55,
        ha="right",
    )
    for label in ax.get_xticklabels():
        label.set_rotation_mode("anchor")
    ax.set_ylabel("Final holdout absolute LCB")

    ax = axes[1, 2]
    _panel(ax, "f")
    pipo = data[data["verifier"] == "pipo"]
    for algorithm, frame in pipo.groupby("algorithm"):
        grouped = frame.groupby("update_id")["pipo_modulation"].mean()
        ax.plot(
            grouped.index,
            grouped.values,
            color=COLORS["pipo"],
            linestyle=LINESTYLES[algorithm],
            marker="o",
            markersize=3,
            label=algorithm.upper(),
        )
    ax.axhline(1, color="#777777", linewidth=0.7)
    ax.set_xlabel("Policy update")
    ax.set_ylabel("PIPO historical modulation")
    ax.legend()
    _pilot_note(fig, source)
    _save(fig, output / "fig2_verified_update_reliability.png")


def _hist_density(
    ax: plt.Axes,
    frame: pd.DataFrame,
    target: float,
    tolerance: float,
    xlabel: str,
) -> None:
    values = frame["independent_prediction"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    lo, hi = float(values.min()), float(values.max())
    bins = np.linspace(lo, hi, 90)
    for (method, family, algorithm), group in frame.groupby(
        ["method", "method_family", "algorithm"], dropna=False
    ):
        group_values = group["independent_prediction"].to_numpy(dtype=float)
        group_values = group_values[np.isfinite(group_values)]
        hist, edges = np.histogram(group_values, bins=bins, density=True)
        ax.plot(
            (edges[:-1] + edges[1:]) / 2,
            hist,
            color=COLORS[family],
            linestyle=LINESTYLES.get(algorithm, "-"),
            linewidth=1,
            alpha=0.9,
            label=method,
        )
    ax.axvspan(
        target - tolerance,
        target + tolerance,
        color="#E9C46A",
        alpha=0.22,
        linewidth=0,
    )
    ax.axvline(target, color="#222222", linewidth=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density (finite predictions)")


def _outcome_scatter(ax: plt.Axes, frame: pd.DataFrame, task: str) -> None:
    for row in frame.itertuples():
        ax.scatter(
            row.target_mae,
            row.valid_target_yield,
            color=COLORS[row.method_family],
            marker=MARKERS[row.generation_mode],
            s=30,
            edgecolor="white",
            linewidth=0.3,
        )
    ax.set_xlabel(f"{task.upper()} independent target MAE")
    ax.set_ylabel("Valid-target yield")
    ax.legend(
        handles=[
            Line2D(
                [], [], marker=MARKERS[mode], linestyle="none", color="#555555",
                label=label,
            )
            for mode, label in (("template", "Template"), ("abinitio", "Ab initio"))
        ],
        loc="best",
    )


def _fig3(source: Path, output: Path) -> None:
    raw = pd.read_csv(source / "fig3_structure_properties.csv")
    summary = pd.read_csv(source / "fig3_method_summary.csv")
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(183 * MM, 122 * MM),
        constrained_layout=True,
    )
    panel = iter("abcdef")
    for row, task, target, tolerance, xlabel in [
        (0, "fe", -1.5, 0.30, "Formation energy (eV atom⁻¹)"),
        (1, "bg", 2.0, 0.45, "Band gap (eV)"),
    ]:
        for col, mode in enumerate(("template", "abinitio")):
            ax = axes[row, col]
            _panel(ax, next(panel))
            frame = raw[(raw["task"] == task) & (raw["generation_mode"] == mode)]
            _hist_density(ax, frame, target, tolerance, xlabel)
        ax = axes[row, 2]
        _panel(ax, next(panel))
        _outcome_scatter(ax, summary[summary["task"] == task], task)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0, 1].legend(
        by_label.values(),
        by_label.keys(),
        ncol=2,
        loc="upper right",
        fontsize=5.3,
    )
    for row, mode in enumerate(("Template", "Ab initio")):
        axes[0, row].text(
            0.02,
            0.97,
            mode,
            transform=axes[0, row].transAxes,
            va="top",
            fontsize=6,
            color="#444444",
        )
        axes[1, row].text(
            0.02,
            0.97,
            mode,
            transform=axes[1, row].transAxes,
            va="top",
            fontsize=6,
            color="#444444",
        )
    _pilot_note(fig, source)
    _save(fig, output / "fig3_independent_fe_bg_control.png")


def _quality_points(
    ax: plt.Axes,
    quality: pd.DataFrame,
    column: str,
    ylabel: str,
) -> None:
    methods = list(dict.fromkeys(quality["method"].tolist()))
    positions = {method: index for index, method in enumerate(methods)}
    for row in quality.itertuples():
        task = row.task if isinstance(row.task, str) else None
        offset = {"fe": -0.12, "bg": 0.12}.get(task, 0.0)
        filled = task != "bg"
        ax.scatter(
            positions[row.method] + offset,
            getattr(row, column),
            facecolor=COLORS[row.method_family] if filled else "none",
            edgecolor=COLORS[row.method_family],
            marker=MARKERS[row.generation_mode],
            s=25,
            alpha=0.8,
        )
    ax.set_xticks(
        list(positions.values()),
        list(positions.keys()),
        rotation=45,
        ha="right",
    )
    for label in ax.get_xticklabels():
        label.set_rotation_mode("anchor")
    ax.set_ylabel(ylabel)


def _coverage_panel(ax: plt.Axes, quality: pd.DataFrame) -> None:
    for row in quality.itertuples():
        task = row.task if isinstance(row.task, str) else None
        filled = task != "bg"
        ax.scatter(
            row.coverage_recall,
            row.coverage_precision,
            facecolor=COLORS[row.method_family] if filled else "none",
            edgecolor=COLORS[row.method_family],
            marker=MARKERS[row.generation_mode],
            s=28,
            alpha=0.8,
        )
    ax.set_xlabel("Coverage recall")
    ax.set_ylabel("Coverage precision")


def _fig4(source: Path, output: Path) -> None:
    quality = pd.read_csv(source / "fig4_quality_summary.csv")
    sample = pd.read_csv(source / "fig4_structure_sample.csv")
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(183 * MM, 120 * MM),
        constrained_layout=True,
    )
    for ax, label, column, ylabel in [
        (axes[0, 0], "a", "validity", "Structural validity"),
        (axes[0, 1], "b", "stability_proxy_mean", "FE stability proxy"),
        (axes[1, 0], "d", "uniqueness", "Sampled uniqueness"),
        (axes[1, 1], "e", "novelty_among_unique", "Novelty among unique"),
    ]:
        _panel(ax, label)
        _quality_points(ax, quality, column, ylabel)
        if column in {
            "validity",
            "stability_proxy_mean",
            "uniqueness",
            "novelty_among_unique",
        }:
            ax.set_ylim(0, 1.02)

    ax = axes[0, 2]
    _panel(ax, "c")
    _coverage_panel(ax, quality)
    ax.legend(
        handles=[
            Line2D(
                [], [], marker="o", linestyle="none", markerfacecolor="#777777",
                markeredgecolor="#777777", label="FE / task-neutral",
            ),
            Line2D(
                [], [], marker="o", linestyle="none", markerfacecolor="none",
                markeredgecolor="#777777", label="BG",
            ),
            Line2D(
                [], [], marker="o", linestyle="none", color="#555555",
                label="Template",
            ),
            Line2D(
                [], [], marker="s", linestyle="none", color="#555555",
                label="Ab initio",
            ),
        ],
        ncol=2,
        loc="best",
    )

    ax = axes[1, 2]
    _panel(ax, "f")
    for family, frame in sample.groupby("method_family"):
        ax.scatter(
            frame["embedding_1"],
            frame["embedding_2"],
            s=3,
            alpha=0.22,
            color=COLORS[family],
            rasterized=True,
            label=family.capitalize(),
        )
    ax.set_xlabel("Composition–lattice PC1")
    ax.set_ylabel("Composition–lattice PC2")
    ax.legend(markerscale=2.5, ncol=2, loc="best")
    _pilot_note(fig, source)
    _save(fig, output / "fig4_quality_and_search_space.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    args = parser.parse_args()
    audit_path = args.source_dir / "conclusion_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError("Run audit_conclusions.py before plotting")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("plot_authorized"):
        raise RuntimeError("Plotting refused: conclusion audit is not authorized")
    args.asset_dir.mkdir(parents=True, exist_ok=True)
    _copy_source_data(args.source_dir, args.asset_dir)
    _fig1(args.source_dir, args.asset_dir)
    _fig2(args.source_dir, args.asset_dir)
    _fig3(args.source_dir, args.asset_dir)
    _fig4(args.source_dir, args.asset_dir)


if __name__ == "__main__":
    main()
