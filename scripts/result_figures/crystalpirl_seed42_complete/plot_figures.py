"""Render audited complete seed-42 CrystalPIRL figures as PNG files."""

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
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

from scripts.result_figures.crystalpirl_pipo_seed42 import plot_figures as common


ROOT = Path(__file__).resolve().parents[3]
FIG1_SOURCE = ROOT / "assets/crystalpirl_article_blueprint/source/fig1_architecture_overview.png"
BAND_GAP_MAX = 10.0
COLORS = {
    "training": "#4D4D4D",
    "base": "#E69F00",
    "cfg": "#0072B2",
    "pipo": "#009E73",
    "crystalpirl": "#CC79A7",
    "fe": "#D55E00",
    "bg": "#0072B2",
    "joint": "#7B3294",
    "target": "#C44E52",
    "ink": "#222222",
}
TASK_LABELS = {"fe": "FE", "bg": "BG", "joint": "FE+BG"}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5.6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _fig1(asset_dir: Path) -> None:
    if not FIG1_SOURCE.is_file():
        raise FileNotFoundError(FIG1_SOURCE)
    shutil.copy2(FIG1_SOURCE, asset_dir / "fig1_crystalpirl_method.png")


def _rolling(values: np.ndarray, window: int = 15) -> tuple[np.ndarray, np.ndarray]:
    width = min(window, len(values))
    if width % 2 == 0:
        width -= 1
    series = pd.Series(values, dtype=float)
    minimum = max(1, width // 3)
    mean = series.rolling(width, center=True, min_periods=minimum).mean()
    std = series.rolling(width, center=True, min_periods=minimum).std(ddof=0).fillna(0)
    return mean.to_numpy(), std.to_numpy()


def _training_trace(
    ax: plt.Axes,
    pipo: pd.DataFrame,
    crystal: pd.DataFrame,
    column: str,
    ylabel: str,
) -> None:
    for family, data, linestyle, alpha in (
        ("PIPO", pipo, "--", 0.72),
        ("CrystalPIRL", crystal, "-", 1.0),
    ):
        for task in ("fe", "bg", "joint"):
            frame = data[data["task"] == task].sort_values("update_id")
            x = frame["update_id"].to_numpy(dtype=int)
            y = frame[column].to_numpy(dtype=float)
            mean, std = _rolling(y)
            color = COLORS[task]
            ax.plot(x, y, color=color, alpha=0.10, linewidth=0.45)
            ax.plot(
                x,
                mean,
                color=color,
                linestyle=linestyle,
                alpha=alpha,
                linewidth=1.05,
                label=f"{TASK_LABELS[task]} {family}",
            )
            if family == "CrystalPIRL":
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.08, linewidth=0)
    ax.set_xlabel("Policy update")
    ax.set_ylabel(ylabel)


def _fig2(source: Path, asset_dir: Path) -> None:
    pipo = pd.read_csv(source / "fig2_pipo_training_dynamics.csv")
    crystal = pd.read_csv(source / "fig1_fig2_crystalpirl_dynamics.csv")
    verification = pd.read_csv(source / "fig1_fig2_crystalpirl_verification.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2047, 4.7244), constrained_layout=True)

    common._panel(axes[0, 0], "a")
    _training_trace(axes[0, 0], pipo, crystal, "reward_mean", "Raw reward")
    axes[0, 0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=5.0)

    common._panel(axes[0, 1], "b")
    _training_trace(axes[0, 1], pipo, crystal, "advantage_mean", "Mean GRPO advantage")
    axes[0, 1].axhline(0, color=COLORS["ink"], linewidth=0.6)

    common._panel(axes[1, 0], "c")
    _training_trace(axes[1, 0], pipo, crystal, "gradient_norm", "Policy gradient norm")

    ax = axes[1, 1]
    common._panel(ax, "d")
    marker_by_action = {"accept": "o", "reject": "x", "rollback": "s"}
    for task in ("fe", "bg", "joint"):
        frame = verification[verification["task"] == task].sort_values("update_id")
        color = COLORS[task]
        ax.plot(frame["update_id"], frame["local_reward_lcb"], color=color, linewidth=0.85)
        ax.plot(frame["update_id"], frame["absolute_reward_lcb"], color=color, linewidth=0.85, linestyle="--")
        for action, marker in marker_by_action.items():
            selected = frame[frame["block_action"] == action]
            if not selected.empty:
                ax.scatter(
                    selected["update_id"],
                    selected["absolute_reward_lcb"],
                    color=color,
                    marker=marker,
                    s=18,
                    linewidth=0.8,
                    zorder=3,
                )
    ax.axhline(0, color=COLORS["ink"], linewidth=0.7)
    ax.set_xlabel("Policy update")
    ax.set_ylabel("Paired reward LCB")
    task_handles = [Line2D([0], [0], color=COLORS[t], label=TASK_LABELS[t]) for t in ("fe", "bg", "joint")]
    style_handles = [
        Line2D([0], [0], color="#555555", linestyle="-", label="vs current"),
        Line2D([0], [0], color="#555555", linestyle="--", label="vs Base"),
    ]
    action_handles = [
        Line2D([0], [0], color="#555555", marker=marker, linestyle="none", label=action.capitalize())
        for action, marker in marker_by_action.items()
    ]
    ax.legend(handles=task_handles + style_handles + action_handles, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, fontsize=5.0)
    _save(fig, asset_dir / "fig2_verified_policy_improvement.png")


def _quality_panel(ax: plt.Axes, source: Path) -> None:
    data = pd.read_csv(source / "fig3_quality_metrics.csv").set_index("method")
    methods = ["Base", "CFG-FE+BG", "CrystalPIRL-FE+BG"]
    labels = {"Base": "Base", "CFG-FE+BG": "CFG", "CrystalPIRL-FE+BG": "CrystalPIRL"}
    colors = {"Base": COLORS["base"], "CFG-FE+BG": COLORS["cfg"], "CrystalPIRL-FE+BG": COLORS["crystalpirl"]}
    columns = [
        "sun",
        "uniqueness",
        "novelty_among_unique",
        "mp_hull_stability",
        "compositional_validity",
        "structural_validity",
    ]
    axis_labels = ["S.U.N.", "Uniqueness", "Novelty", "Stability", "Comp.\nvalidity", "Struct.\nvalidity"]
    positions = np.arange(len(columns), dtype=float)
    width = 0.23
    for index, method in enumerate(methods):
        ax.bar(
            positions + (index - 1) * width,
            100.0 * data.loc[method, columns].to_numpy(dtype=float),
            width=width,
            color=colors[method],
            label=labels[method],
        )
    ax.set_xticks(positions, axis_labels, rotation=18, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 108)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3)


def _kde_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    column: str,
    methods: list[tuple[str, str, str]],
    target: float,
    xlabel: str,
) -> dict[str, int]:
    datasets = []
    excluded = {}
    for method, label, color in methods:
        values = pd.to_numeric(data.loc[data["method"] == method, column], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(values)
        if column == "predicted_band_gap":
            keep &= values <= BAND_GAP_MAX
        excluded[method] = int((~keep).sum())
        datasets.append((label, values[keep], color))
    lower = min(values.min() for _, values, _ in datasets)
    upper = BAND_GAP_MAX if column == "predicted_band_gap" else max(values.max() for _, values, _ in datasets)
    margin = 0.03 * max(upper - lower, 1.0)
    grid = np.linspace(lower - margin, upper, 450)
    for label, values, color in datasets:
        density = gaussian_kde(values)(grid)
        ax.plot(grid, density, color=color, linewidth=1.1, label=label)
        ax.fill_between(grid, 0, density, color=color, alpha=0.13, linewidth=0)
    ax.axvline(target, color=COLORS["target"], linestyle="--", linewidth=1.0)
    ax.text(target, 0.78, "Target", transform=ax.get_xaxis_transform(), rotation=90, rotation_mode="anchor", va="top", ha="right", fontsize=5.5, color=COLORS["target"])
    ax.set_xlim(lower - margin, upper)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density")
    ax.legend(loc="upper left", fontsize=5.0)
    return excluded


def _joint_density(ax: plt.Axes, data: pd.DataFrame) -> dict[str, int]:
    methods = [
        ("Training set", "Training set", COLORS["training"]),
        ("Base", "Base", COLORS["base"]),
        ("CFG-FE+BG", "CFG", COLORS["cfg"]),
        ("CrystalPIRL-FE+BG", "CrystalPIRL", COLORS["crystalpirl"]),
    ]
    x_edges = np.linspace(-3.0, 0.0, 120)
    y_edges = np.linspace(0.0, 6.0, 120)
    extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
    excluded = {}
    handles = []
    for method, label, color in methods:
        frame = data[data["method"] == method]
        fe = pd.to_numeric(frame["predicted_formation_energy_per_atom"], errors="coerce").to_numpy(dtype=float)
        bg = pd.to_numeric(frame["predicted_band_gap"], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(fe) & np.isfinite(bg) & (bg <= BAND_GAP_MAX)
        excluded[method] = int((~keep).sum())
        density, _, _ = np.histogram2d(fe[keep], bg[keep], bins=(x_edges, y_edges))
        density = gaussian_filter(density.T, sigma=1.5)
        if density.max() > 0:
            density /= density.max()
        density[density < 0.06] = 0
        rgba = np.empty((*density.shape, 4), dtype=float)
        rgba[..., :3] = mpl.colors.to_rgb(color)
        rgba[..., 3] = 0.38 * np.sqrt(density)
        ax.imshow(rgba, origin="lower", extent=extent, aspect="auto", interpolation="bilinear", rasterized=True)
        handles.append(Patch(facecolor=mpl.colors.to_rgba(color, 0.32), edgecolor="none", label=label))
    target = ax.scatter([-1.5], [2.0], marker="*", s=34, color=COLORS["target"], edgecolor="white", linewidth=0.5, label="Target", zorder=4)
    ax.set_xlim(-3.0, 0.0)
    ax.set_ylim(0.0, 6.0)
    ax.set_xlabel("Formation energy (eV atom⁻¹)")
    ax.set_ylabel("Band gap (eV)")
    ax.legend(handles=[*handles, target], loc="upper left", fontsize=5.0)
    return excluded


def _fig3(source: Path, asset_dir: Path) -> dict[str, dict[str, int]]:
    data = pd.read_csv(source / "fig3_property_distributions.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2047, 5.5118), constrained_layout=True)
    common._panel(axes[0, 0], "a")
    _quality_panel(axes[0, 0], source)
    common._panel(axes[0, 1], "b")
    fe = _kde_panel(
        axes[0, 1], data, "predicted_formation_energy_per_atom",
        [("Training set", "Training set", COLORS["training"]), ("Base", "Base", COLORS["base"]), ("CFG-FE", "CFG", COLORS["cfg"]), ("CrystalPIRL-FE", "CrystalPIRL", COLORS["crystalpirl"])],
        -1.5, "Formation energy (eV atom⁻¹)",
    )
    common._panel(axes[1, 0], "c")
    bg = _kde_panel(
        axes[1, 0], data, "predicted_band_gap",
        [("Training set", "Training set", COLORS["training"]), ("Base", "Base", COLORS["base"]), ("CFG-BG", "CFG", COLORS["cfg"]), ("CrystalPIRL-BG", "CrystalPIRL", COLORS["crystalpirl"])],
        2.0, "Band gap (eV)",
    )
    common._panel(axes[1, 1], "d")
    joint = _joint_density(axes[1, 1], data)
    _save(fig, asset_dir / "fig3_fe_bg_control.png")
    return {"formation_energy_kde": fe, "band_gap_kde": bg, "joint_density": joint}


def _tradeoff(ax: plt.Axes, source: Path) -> None:
    data = pd.read_csv(source / "fig4_target_yield_diversity.csv")
    selected = data[data["method"].isin(["Base", "CFG-FE", "CFG-BG", "CFG-FE+BG", "GRPO+PIPO-FE+BG", "CrystalPIRL-FE+BG"])]
    colors = {"base": COLORS["base"], "cfg": COLORS["cfg"], "pipo": COLORS["pipo"], "crystalpirl": COLORS["crystalpirl"]}
    markers = {"Base": "o", "CFG-FE": "^", "CFG-BG": "v", "CFG-FE+BG": "s", "GRPO+PIPO-FE+BG": "D", "CrystalPIRL-FE+BG": "P"}
    labels = {"GRPO+PIPO-FE+BG": "GRPO+PIPO", "CrystalPIRL-FE+BG": "CrystalPIRL"}
    for row in selected.itertuples():
        ax.scatter(
            row.structure_diversity,
            100.0 * row.valid_joint_target_yield,
            s=40,
            color=colors[row.method_family],
            marker=markers[row.method],
            edgecolor="white",
            linewidth=0.5,
            label=labels.get(row.method, row.method),
        )
    ax.set_xlabel("Structural diversity (mean CrystalNN distance)")
    ax.set_ylabel("Valid joint-target yield (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=5.0)


def _tsne(ax: plt.Axes, source: Path) -> None:
    data = pd.read_csv(source / "fig4_tsne.csv")
    training = data[data["method"] == "Training set"]
    density, x_edges, y_edges = np.histogram2d(training["tsne_1"], training["tsne_2"], bins=160)
    density = gaussian_filter(density.T, sigma=2.0)
    density /= density.max()
    alpha = np.where(density >= 0.01, 0.08 + 0.70 * density**0.65, 0.0)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("training_density", ["#F2F2F2", "#4D4D4D"])
    ax.imshow(density, origin="lower", extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]), aspect="auto", cmap=cmap, alpha=alpha, interpolation="bilinear", rasterized=True)
    handles = [Patch(facecolor="#4D4D4D", alpha=0.65, label="Training set density")]
    for method, color in (("Base", COLORS["base"]), ("CFG", COLORS["cfg"]), ("GRPO+PIPO", COLORS["pipo"]), ("CrystalPIRL", COLORS["crystalpirl"])):
        frame = data[data["method"] == method]
        handles.append(ax.scatter(frame["tsne_1"], frame["tsne_2"], s=2.6, alpha=0.28, color=color, linewidth=0, rasterized=True, label=method))
    ax.set_xlim(data["tsne_1"].min(), data["tsne_1"].max())
    ax.set_ylim(data["tsne_2"].min(), data["tsne_2"].max())
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, markerscale=2.5, fontsize=5.0)


def _fractions(data: pd.DataFrame, column: str, categories: list) -> pd.DataFrame:
    grouped = data.assign(category=data[column]).groupby(["method", "category"]).size().rename("count").reset_index()
    grouped["fraction"] = grouped["count"] / grouped.groupby("method")["count"].transform("sum")
    return grouped.pivot(index="category", columns="method", values="fraction").reindex(categories).fillna(0.0)


def _distribution_panels(fig: plt.Figure, spec, source: Path) -> None:
    grid = spec.subgridspec(2, 3, wspace=0.48, hspace=0.62)
    profiles = pd.read_csv(source / "fig4_structure_profiles.csv")
    elements = pd.read_csv(source / "fig4_element_frequencies.csv")
    methods = ["Training set", "Base", "CFG", "GRPO+PIPO", "CrystalPIRL"]
    colors = {"Training set": COLORS["training"], "Base": COLORS["base"], "CFG": COLORS["cfg"], "GRPO+PIPO": COLORS["pipo"], "CrystalPIRL": COLORS["crystalpirl"]}
    width = 0.16

    ax = fig.add_subplot(grid[0, 0])
    common._panel(ax, "c1")
    element_order = elements.groupby("element")["atom_fraction"].sum().nlargest(12).index.tolist()
    matrix = elements.pivot(index="method", columns="element", values="atom_fraction").reindex(index=methods, columns=element_order).fillna(0.0)
    ax.imshow(matrix.to_numpy(), aspect="auto", cmap="Blues")
    ax.set_xticks(np.arange(len(element_order)), element_order, rotation=90, rotation_mode="anchor", ha="right", fontsize=5)
    ax.set_yticks(np.arange(len(methods)), methods, fontsize=5.0)
    ax.set_xlabel("Element frequency")

    ax = fig.add_subplot(grid[0, 1])
    common._panel(ax, "c2")
    categories = [1, 2, 3, 4, 5]
    fractions = _fractions(profiles.assign(nary_group=profiles["n_elements"].clip(upper=5)), "nary_group", categories)
    positions = np.arange(len(categories), dtype=float)
    for index, method in enumerate(methods):
        ax.bar(positions + (index - 2) * width, 100.0 * fractions[method].to_numpy(), width=width, color=colors[method])
    ax.set_xticks(positions, ["1", "2", "3", "4", "5+"])
    ax.set_xlabel("Number of elements")
    ax.set_ylabel("Fraction (%)")

    ax = fig.add_subplot(grid[0, 2])
    common._panel(ax, "c3")
    all_groups = sorted(profiles["space_group"].unique())
    space = _fractions(profiles, "space_group", all_groups)
    top_groups = space.sum(axis=1).nlargest(8).index.astype(int).tolist()
    space = space.reindex(top_groups)
    positions = np.arange(len(top_groups), dtype=float)
    for index, method in enumerate(methods):
        ax.bar(positions + (index - 2) * width, 100.0 * space[method].to_numpy(), width=width, color=colors[method])
    ax.set_xticks(positions, ["?" if group == 0 else str(group) for group in top_groups], rotation=90, rotation_mode="anchor", ha="right")
    ax.set_xlabel("Space group")
    ax.set_ylabel("Fraction (%)")

    ax = fig.add_subplot(grid[1, 0])
    common._panel(ax, "c4")
    bins = np.arange(0.5, int(profiles["num_atoms"].max()) + 1.5, 1.0)
    for method in methods:
        ax.hist(profiles.loc[profiles["method"] == method, "num_atoms"], bins=bins, density=True, histtype="step", linewidth=0.95, color=colors[method])
    ax.set_xlabel("Atoms per cell")
    ax.set_ylabel("Probability")

    ax = fig.add_subplot(grid[1, 1:])
    common._panel(ax, "c5")
    finite = profiles[np.isfinite(profiles["density_g_cm3"]) & (profiles["density_g_cm3"] > 0)]
    upper = float(finite["density_g_cm3"].quantile(0.995))
    density_grid = np.linspace(0, upper, 240)
    for method in methods:
        values = finite.loc[finite["method"] == method, "density_g_cm3"].to_numpy(dtype=float)
        density = gaussian_kde(values)(density_grid)
        ax.plot(density_grid, density, color=colors[method], linewidth=1.0, label=method)
        ax.fill_between(density_grid, 0, density, color=colors[method], alpha=0.09, linewidth=0)
    ax.set_xlim(0, upper)
    ax.set_xlabel("Density (g cm⁻³)")
    ax.set_ylabel("Probability density")
    ax.legend(loc="best", ncol=3, fontsize=5.0)


def _fig4(source: Path, asset_dir: Path) -> None:
    fig = plt.figure(figsize=(7.2047, 6.3780))
    grid = fig.add_gridspec(2, 10, height_ratios=[1.0, 1.55], left=0.08, right=0.98, bottom=0.07, top=0.93, hspace=0.28, wspace=0.38)
    ax = fig.add_subplot(grid[0, :5])
    common._panel(ax, "a")
    _tradeoff(ax, source)
    ax = fig.add_subplot(grid[0, 5:])
    common._panel(ax, "b")
    _tsne(ax, source)
    _distribution_panels(fig, grid[1, :], source)
    _save(fig, asset_dir / "fig4_quality_search_space.png")


def _copy_source(source: Path, asset_dir: Path) -> None:
    target = asset_dir / "source_data"
    target.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md"}:
            shutil.copy2(path, target / path.name)


def plot(source: Path, asset_dir: Path) -> None:
    audit = json.loads((source / "conclusion_audit.json").read_text())
    if not audit.get("plot_authorized"):
        raise RuntimeError("Conclusion audit did not authorize plotting")
    if asset_dir.exists():
        raise FileExistsError(f"Asset directory already exists: {asset_dir}")
    asset_dir.mkdir(parents=True)
    _fig1(asset_dir)
    _fig2(source, asset_dir)
    exclusions = _fig3(source, asset_dir)
    _fig4(source, asset_dir)
    _copy_source(source, asset_dir)
    qa = {
        "seed": 42,
        "main_method": "GRPO+CrystalPIRL blockwise dual-anchor gate",
        "comparator": "GRPO+PIPO",
        "exports": "PNG only",
        "figure_titles": "none",
        "band_gap_filter": "finite values <= 10 eV for plotting only",
        "excluded_counts": exclusions,
        "claim_boundary": audit["claim_boundaries"],
    }
    (asset_dir / "figure_qa.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    (asset_dir / "README.md").write_text(
        "# Seed=42 CrystalPIRL Fig. 1–Fig. 4\n\n"
        "本目录保存通过完整性审计后的单种子探索性结果。主方法为分块双锚点 "
        "GRPO+CrystalPIRL，GRPO+PIPO 仅作为独立对照。所有正文统计只使用 "
        "seed=42 的从头生成结构，性质统一由 seed123 独立预测器评估。"
        "当前仅导出 PNG；单 seed 结果不能替代多 seed 重复、MLFF 或 DFT 验证。\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    args = parser.parse_args()
    plot(args.source_dir.resolve(), args.asset_dir.resolve())


if __name__ == "__main__":
    main()
