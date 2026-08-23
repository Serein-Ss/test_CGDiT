"""Render honest PNG-only blueprints for the CrystalPIRL main figures.

Existing pilot and CFG results are plotted where they are available. Panels that
need the formal 300-update experiment or independent evaluation are explicitly
marked as pending; the script never fabricates replacement measurements.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "assets" / "model_results" / "source_data"
PILOT = ROOT / "assets" / "tmp"
OUTPUT = ROOT / "assets" / "crystalpirl_article_blueprint"

COLORS = {
    "base": "#6F6F6F",
    "cfg": "#4C78A8",
    "open": "#F28E2B",
    "pipo": "#8E6CBB",
    "pirl": "#269C8C",
    "safe": "#269C8C",
    "warn": "#D9A441",
    "reject": "#C44E52",
    "ink": "#222222",
    "muted": "#767676",
    "pending": "#D9D9D9",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
    )


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str,
    fontsize: float = 6.5,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
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
        fontsize=fontsize,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = COLORS["ink"],
    style: str = "-|>",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=8,
            linewidth=0.8,
            color=color,
        )
    )


def data_tag(ax: plt.Axes, text: str, color: str) -> None:
    ax.text(
        0.99,
        0.99,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
        color=color,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": mpl.colors.to_rgba(color, 0.09),
            "edgecolor": mpl.colors.to_rgba(color, 0.45),
            "linewidth": 0.5,
        },
    )


def pending(ax: plt.Axes, evidence: str, fields: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#BDBDBD")
        spine.set_linestyle((0, (3, 2)))
    ax.text(
        0.5,
        0.62,
        "DATA PENDING",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color=COLORS["muted"],
    )
    ax.text(
        0.5,
        0.45,
        textwrap.fill(evidence, width=31),
        ha="center",
        va="center",
        fontsize=6.5,
    )
    ax.text(
        0.5,
        0.23,
        textwrap.fill(fields, width=37),
        ha="center",
        va="center",
        fontsize=5.7,
        color=COLORS["muted"],
    )
    data_tag(ax, "placeholder", COLORS["muted"])


def footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.995, -0.025, text, ha="right", va="bottom", fontsize=5.5, color="#666666")


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / name, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig1() -> None:
    fig = plt.figure(figsize=(7.2047, 5.5906), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.34, 1, 1], height_ratios=[1, 1])

    ax = fig.add_subplot(grid[:, 0])
    panel(ax, "a")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, (0.07, 0.81), 0.34, 0.11, "Frozen base\nπ₀", COLORS["base"])
    box(ax, (0.58, 0.81), 0.34, 0.11, "Verified policy\nπₖ", COLORS["pirl"])
    box(ax, (0.58, 0.58), 0.34, 0.11, "OrbitPO update\n→ π′", COLORS["open"])
    box(ax, (0.07, 0.57), 0.34, 0.13, "Full trajectories\n+ terminal reward", COLORS["cfg"])
    box(ax, (0.07, 0.32), 0.34, 0.14, "Paired fixed +\nholdout probes", COLORS["pirl"])
    box(ax, (0.58, 0.31), 0.34, 0.15, "Local + absolute\nLCB and safeguards", COLORS["pirl"])
    box(ax, (0.32, 0.08), 0.38, 0.12, "accept / attenuate\n/ reject / rollback", COLORS["safe"])
    arrow(ax, (0.75, 0.81), (0.75, 0.69))
    arrow(ax, (0.58, 0.635), (0.41, 0.635))
    arrow(ax, (0.24, 0.57), (0.24, 0.46))
    arrow(ax, (0.41, 0.39), (0.58, 0.39))
    arrow(ax, (0.75, 0.31), (0.62, 0.20))
    arrow(ax, (0.32, 0.14), (0.13, 0.81), COLORS["base"])
    arrow(ax, (0.70, 0.14), (0.88, 0.81), COLORS["pirl"])
    data_tag(ax, "code-grounded schematic", COLORS["pirl"])

    ax = fig.add_subplot(grid[0, 1])
    panel(ax, "b")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    channels = [
        (0.01, "Lattice", "valid-subspace\nGaussian", COLORS["cfg"]),
        (0.345, "Coordinates", "wrapped\nlikelihood", COLORS["open"]),
        (0.68, "Elements", "orbit-level\nD3PM", COLORS["pirl"]),
    ]
    for x, name, detail, color in channels:
        box(ax, (x, 0.48), 0.30, 0.27, f"{name}\n{detail}", color, 5.3)
        arrow(ax, (x + 0.15, 0.48), (0.5, 0.24), color)
    ax.text(0.5, 0.13, "joint trajectory log probability", ha="center", fontsize=6.5)
    data_tag(ax, "schematic; trajectory audit pending", COLORS["muted"])

    ax = fig.add_subplot(grid[0, 2])
    panel(ax, "c")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    labels = ["target property", "validity", "stability", "uniqueness", "diversity"]
    colors = [COLORS["cfg"], COLORS["safe"], "#59A14F", "#EDC948", "#B07AA1"]
    for i, (label, color) in enumerate(zip(labels, colors)):
        y = 0.81 - i * 0.14
        ax.add_patch(Rectangle((0.08, y), 0.16, 0.07, facecolor=color, edgecolor="none", alpha=0.75))
        ax.text(0.28, y + 0.035, label, va="center", fontsize=6.3)
    ax.text(0.08, 0.10, "property gain subject to preregistered\nquality tolerances (not rigid equality)", fontsize=6.2)
    data_tag(ax, "reward contract", COLORS["cfg"])

    ax = fig.add_subplot(grid[1, 1])
    panel(ax, "d")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for x, label, color in [(0.05, "π₀", COLORS["base"]), (0.37, "πₖ", COLORS["pirl"]), (0.69, "π′", COLORS["open"])]:
        box(ax, (x, 0.63), 0.25, 0.15, label, color, 8)
        arrow(ax, (x + 0.125, 0.63), (x + 0.125, 0.43), color)
    ax.text(0.5, 0.34, "same prompts + same initial noise + same random stream", ha="center", fontsize=6.2)
    ax.text(0.5, 0.18, "Δlocal = R(π′) − R(πₖ)    Δabsolute = R(π′) − R(π₀)", ha="center", fontsize=6.2)
    data_tag(ax, "paired-probe schematic", COLORS["pirl"])

    ax = fig.add_subplot(grid[1, 2])
    panel(ax, "e")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, (0.04, 0.67), 0.29, 0.12, "dual-positive\nquality safe", COLORS["safe"], 5.1)
    box(ax, (0.355, 0.67), 0.29, 0.12, "uncertain\nre-test scaled", COLORS["warn"], 5.1)
    box(ax, (0.67, 0.67), 0.29, 0.12, "negative\nor unsafe", COLORS["reject"], 5.1)
    for x, label, color in [(0.20, "accept", COLORS["safe"]), (0.50, "attenuate", COLORS["warn"]), (0.80, "reject", COLORS["reject"])]:
        arrow(ax, (x, 0.67), (x, 0.43), color)
        ax.text(x, 0.35, label, ha="center", fontweight="bold", color=color, fontsize=6.5)
    ax.text(0.5, 0.16, "holdout failure after acceptance → rollback", ha="center", fontsize=6.2)
    data_tag(ax, "real examples pending", COLORS["muted"])

    footer(fig, "PNG blueprint • no quantitative claims")
    save(fig, "fig1_crystalpirl_method_blueprint.png")


def _pilot_lines(ax: plt.Axes, task: str) -> None:
    data = pd.read_csv(PILOT / "rl_fig2_seed42_updates.csv")
    data = data[data["property"] == task]
    styles = {"ppo": "-", "grpo": "--"}
    colors = {"open": COLORS["open"], "pipo": COLORS["pipo"], "pirl": COLORS["pirl"]}
    for (algorithm, mode), frame in data.groupby(["algorithm", "mode"]):
        frame = frame.sort_values("step")
        ax.plot(
            frame["step"],
            frame["holdout_reward_lcb"],
            color=colors[mode],
            linestyle=styles[algorithm],
            linewidth=1.0,
            marker="o",
            markersize=2,
            label=f"{mode.upper()}-{algorithm.upper()}",
        )
    ax.axhline(0, color=COLORS["ink"], linewidth=0.7)
    ax.set_xlabel("Pilot update")
    ax.set_ylabel("Holdout reward LCB")
    ax.set_xticks(range(1, 9))
    data_tag(ax, "real: old 8-step pilot, seed=42", COLORS["warn"])


def fig2() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2047, 5.0394), constrained_layout=True)

    ax = axes[0, 0]
    panel(ax, "a")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5, 2.5], ["Open", "PIPO", "CrystalPIRL"])
    ax.set_yticks([1.5, 0.5], ["PPO", "GRPO"])
    matrix = [["P0", "P1", "P2"], ["G0", "G1", "G2"]]
    matrix_colors = [COLORS["open"], COLORS["pipo"], COLORS["pirl"]]
    for row in range(2):
        for col in range(3):
            ax.add_patch(Rectangle((col, 1 - row), 1, 1, facecolor=mpl.colors.to_rgba(matrix_colors[col], 0.15), edgecolor="white"))
            ax.text(col + 0.5, 1.5 - row, matrix[row][col], ha="center", va="center", fontsize=8, fontweight="bold")
    ax.tick_params(length=0)
    data_tag(ax, "formal design: 12 arms incl. FE/BG", COLORS["pirl"])

    ax = axes[0, 1]
    panel(ax, "b")
    _pilot_lines(ax, "fe")

    ax = axes[0, 2]
    panel(ax, "c")
    _pilot_lines(ax, "bg")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=2, loc="lower left", fontsize=5.2)

    ax = axes[1, 0]
    panel(ax, "d")
    pending(ax, "Bad-update and false-accept rates", "needs per-update applied flag, local/absolute holdout deltas, action, seed")

    ax = axes[1, 1]
    panel(ax, "e")
    pending(ax, "Cumulative absolute gain over 300 updates", "needs reward vs frozen base, 95% paired CI, accepted checkpoint history")

    ax = axes[1, 2]
    panel(ax, "f")
    pending(ax, "Verified gain versus evaluation cost", "needs GPU-hours, NFE, oracle queries, accepted-update efficiency")

    footer(fig, "Existing curves are diagnostic only; formal inference awaits 300-update paired evaluations")
    save(fig, "fig2_verified_policy_improvement_blueprint.png")


def _conditioning_frame() -> pd.DataFrame:
    return pd.read_csv(SOURCE / "conditioning_distribution_metrics.csv")


def _summary_frame() -> pd.DataFrame:
    return pd.read_csv(SOURCE / "generation_summary_plot_data.csv")


def _property_summary(ax: plt.Axes, property_name: str) -> None:
    data = _conditioning_frame()
    data = data[data["property"] == property_name].copy()
    order = ["unconditioned", "template_conditioned", "ab_initio_conditioned"]
    data["group"] = pd.Categorical(data["group"], categories=order, ordered=True)
    data = data.sort_values("group")
    labels = ["Base", "CFG\ntemplate", "CFG\nab initio"]
    colors = [COLORS["base"], COLORS["cfg"], COLORS["cfg"]]
    x = np.arange(len(data))
    ax.bar(x, data["mean_absolute_target_error"], color=colors, alpha=0.82, width=0.62)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean absolute target error")
    unit = "eV atom⁻¹" if property_name.startswith("formation") else "eV"
    ax.text(0.02, 0.96, unit, transform=ax.transAxes, va="top", fontsize=5.5, color=COLORS["muted"])
    data_tag(ax, "real: n≈4,096 per group", COLORS["cfg"])


def fig3() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2047, 4.9606), constrained_layout=True)

    ax = axes[0, 0]
    panel(ax, "a")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    paths = [
        (0.70, "Frozen base", COLORS["base"], "unconditional"),
        (0.44, "CFG", COLORS["cfg"], "condition at sampling"),
        (0.18, "CrystalPIRL", COLORS["pirl"], "policy-weight update"),
    ]
    for y, name, color, detail in paths:
        box(ax, (0.05, y), 0.31, 0.12, name, color, 6)
        arrow(ax, (0.36, y + 0.06), (0.58, y + 0.06), color)
        box(ax, (0.60, y), 0.34, 0.12, detail, color, 5.8)
    data_tag(ax, "comparison logic", COLORS["muted"])

    ax = axes[0, 1]
    panel(ax, "b")
    _property_summary(ax, "formation_energy_per_atom")

    ax = axes[0, 2]
    panel(ax, "c")
    _property_summary(ax, "band_gap")

    ax = axes[1, 0]
    panel(ax, "d")
    data = _conditioning_frame()
    groups = ["unconditioned", "template_conditioned", "ab_initio_conditioned"]
    labels = ["Base", "CFG\ntemplate", "CFG\nab initio"]
    width = 0.34
    x = np.arange(3)
    for offset, property_name, color, marker_label in [
        (-width / 2, "formation_energy_per_atom", COLORS["open"], "FE target"),
        (width / 2, "band_gap", COLORS["cfg"], "BG target"),
    ]:
        frame = data[data["property"] == property_name].set_index("group").loc[groups]
        ax.bar(x + offset, frame["hit_rate"] * 100, width, color=color, alpha=0.82, label=marker_label)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Target hit rate (%)")
    ax.legend(loc="upper left")
    data_tag(ax, "real: predictor-defined hits", COLORS["cfg"])

    ax = axes[1, 1]
    panel(ax, "e")
    summary = _summary_frame().set_index("display_label")
    points = [
        ("FE | Template | uncond", 0.030029, COLORS["base"], "FE base"),
        ("FE | Template | FE=-1.5", 0.100586, COLORS["open"], "FE CFG"),
        ("BG | Template | uncond", 0.089600, COLORS["base"], "BG base"),
        ("BG | Template | BG=2", 0.278320, COLORS["cfg"], "BG CFG"),
    ]
    for label, hit, color, text_label in points:
        valid = float(summary.loc[label, "structural_metrics.valid"])
        ax.scatter(hit * 100, valid * 100, s=26, color=color, edgecolor="white", linewidth=0.5)
        ax.annotate(text_label, (hit * 100, valid * 100), xytext=(3, 3), textcoords="offset points", fontsize=5.5)
    ax.set_xlabel("Target hit rate (%)")
    ax.set_ylabel("Structural validity (%)")
    ax.text(0.98, 0.06, "RL points pending", transform=ax.transAxes, ha="right", color=COLORS["muted"], fontsize=6)
    data_tag(ax, "real baseline/CFG trade-off", COLORS["cfg"])

    ax = axes[1, 2]
    panel(ax, "f")
    pending(ax, "Quality non-degradation forest plot", "needs paired Δ validity, stability, uniqueness, novelty, diversity vs base")

    footer(fig, "Existing FE/BG results use seed=42 predictor summaries; RL and independent-evaluator results pending")
    save(fig, "fig3_fe_bg_control_blueprint.png")


def fig4() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2047, 4.9606), constrained_layout=True)
    summary = _summary_frame().set_index("display_label")
    rows = [
        ("Base | Template | uncond", "Base\ntemplate", COLORS["base"]),
        ("Base | Ab initio | uncond", "Base\nab initio", COLORS["base"]),
        ("FE | Template | FE=-1.5", "FE-CFG\ntemplate", COLORS["open"]),
        ("FE | Ab initio | FE=-1.5", "FE-CFG\nab initio", COLORS["open"]),
        ("BG | Template | BG=2", "BG-CFG\ntemplate", COLORS["cfg"]),
        ("BG | Ab initio | BG=2", "BG-CFG\nab initio", COLORS["cfg"]),
    ]

    ax = axes[0, 0]
    panel(ax, "a")
    values = [100 * float(summary.loc[key, "structural_metrics.valid"]) for key, _, _ in rows]
    labels = [
        "Base template",
        "Base ab initio",
        "FE-CFG template",
        "FE-CFG ab initio",
        "BG-CFG template",
        "BG-CFG ab initio",
    ]
    colors = [color for _, _, color in rows]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, values, color=colors, alpha=0.82)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Structural validity (%)")
    data_tag(ax, "real: n=4,096 per set", COLORS["cfg"])

    ax = axes[0, 1]
    panel(ax, "b")
    for key, label, color in rows:
        precision = float(summary.loc[key, "structural_metrics.cov_precision"])
        recall = float(summary.loc[key, "structural_metrics.cov_recall"])
        marker = "o" if "template" in label else "s"
        ax.scatter(100 * precision, 100 * recall, s=27, color=color, marker=marker, edgecolor="white", linewidth=0.5)
        ax.annotate(label.replace("\n", " "), (100 * precision, 100 * recall), xytext=(3, 2), textcoords="offset points", fontsize=5.1)
    ax.set_xlabel("Coverage precision (%)")
    ax.set_ylabel("Coverage recall (%)")
    data_tag(ax, "real baseline/CFG coverage", COLORS["cfg"])

    ax = axes[0, 2]
    panel(ax, "c")
    pending(ax, "Target-yield–diversity Pareto front", "needs independent target yield, diversity metric, seed CI for every frozen policy")

    ax = axes[1, 0]
    panel(ax, "d")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    regions = [
        ((0.30, 0.62), 0.32, 0.22, COLORS["base"], "train/base"),
        ((0.61, 0.61), 0.28, 0.20, COLORS["cfg"], "CFG"),
        ((0.49, 0.31), 0.31, 0.23, COLORS["pirl"], "CrystalPIRL"),
    ]
    for center, width, height, color, label in regions:
        ax.add_patch(
            Ellipse(
                center,
                width,
                height,
                facecolor=mpl.colors.to_rgba(color, 0.14),
                edgecolor=color,
                linewidth=0.9,
            )
        )
        ax.text(center[0], center[1], label, ha="center", va="center", fontsize=6, color=color)
    ax.text(
        0.02,
        0.03,
        "layout only — no measured coordinates shown",
        fontsize=5.5,
        color=COLORS["muted"],
    )
    data_tag(ax, "schematic embedding", COLORS["muted"])

    ax = axes[1, 1]
    panel(ax, "e")
    pending(ax, "Composition and symmetry shift", "needs element counts, n-ary composition, space group, atom count, density")

    ax = axes[1, 2]
    panel(ax, "f")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    cards = [
        (0.04, 0.56, "high reward", COLORS["cfg"]),
        (0.52, 0.56, "high novelty", COLORS["pirl"]),
        (0.04, 0.10, "high stability", "#59A14F"),
        (0.52, 0.10, "failure / hacking", COLORS["reject"]),
    ]
    for x0, y0, label, color in cards:
        ax.add_patch(Rectangle((x0, y0), 0.42, 0.34, facecolor=mpl.colors.to_rgba(color, 0.07), edgecolor=color, linewidth=0.7))
        for i in range(5):
            angle = 2 * np.pi * i / 5
            x = x0 + 0.21 + 0.095 * np.cos(angle)
            y = y0 + 0.19 + 0.095 * np.sin(angle)
            ax.plot([x0 + 0.21, x], [y0 + 0.19, y], color="#AAAAAA", linewidth=0.45)
            ax.scatter(x, y, s=18, color=color, edgecolor="white", linewidth=0.4)
        ax.scatter(x0 + 0.21, y0 + 0.19, s=22, color="#444444", edgecolor="white", linewidth=0.4)
        ax.text(x0 + 0.21, y0 + 0.035, label, ha="center", fontsize=5.5)
    data_tag(ax, "structure placeholders", COLORS["muted"])

    footer(fig, "Quantitative RL population, embedding, composition and representative structures pending")
    save(fig, "fig4_quality_search_space_blueprint.png")


def main() -> None:
    required = [
        SOURCE / "conditioning_distribution_metrics.csv",
        SOURCE / "generation_summary_plot_data.csv",
        PILOT / "rl_fig2_seed42_updates.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required blueprint source data: " + ", ".join(missing))
    fig1()
    fig2()
    fig3()
    fig4()
    print(f"Saved PNG blueprints to {OUTPUT}")


if __name__ == "__main__":
    main()
