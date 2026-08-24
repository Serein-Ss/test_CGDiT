"""Render honest PNG-only blueprints for the CrystalPIRL main figures.

Existing pilot and CFG results are plotted where they are available. Panels that
need the formal 200-update experiment or independent evaluation are explicitly
marked as pending; the script never fabricates replacement measurements.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "assets" / "crystalpirl_article_blueprint"
FIG1_OVERVIEW = OUTPUT / "source" / "fig1_architecture_overview.png"
TRAIN_MP20 = ROOT / "data" / "mp_20" / "train.csv"
BASE_ABINITIO = (
    ROOT / "output" / "singlerun" / "2026-06-27" / "00-32-50-mp20_base"
    / "evaluations" / "property_predictions"
    / "eval_properties_gen_abinitio_empirical_uncond_n4096_seed42_predictor_seed123.csv"
)
BASE_STRUCTURAL_METRICS = (
    BASE_ABINITIO.parents[1]
    / "structural_metrics"
    / "eval_metrics_gen_abinitio_empirical_uncond_n4096_seed42.json"
)
CFG_ABINITIO = {
    "formation_energy_per_atom": ROOT / "output" / "singlerun" / "2026-06-28"
    / "16-46-08-mp20_fe" / "evaluations" / "property_predictions"
    / "eval_properties_gen_abinitio_empirical_fe_m1p5_n4096_seed42_predictor_seed123.csv",
    "band_gap": ROOT / "output" / "singlerun" / "2026-06-30" / "11-28-44-mp20_bg"
    / "evaluations" / "property_predictions"
    / "eval_properties_gen_abinitio_empirical_bg_2_n4096_seed42_predictor_seed123.csv",
}
JOINT_CFG_ABINITIO = (
    ROOT / "output" / "singlerun" / "2026-08-07" / "07-54-15-mp20_fe_bg"
    / "evaluations" / "property_predictions"
    / "eval_properties_gen_abinitio_empirical_fe_m1p5_bg_2_n4096_seed42_predictor_seed42.csv"
)
JOINT_CFG_STRUCTURAL_METRICS = (
    JOINT_CFG_ABINITIO.parents[1]
    / "structural_metrics"
    / "eval_metrics_gen_abinitio_empirical_fe_m1p5_bg_2_n4096_seed42.json"
)
FIG3_QUALITY_METRICS = OUTPUT / "source" / "fig3_quality_metrics.csv"
FIG4_TRADEOFF = OUTPUT / "source" / "fig4a_target_yield_diversity.csv"
FIG4_TSNE = OUTPUT / "source" / "fig4b_tsne.csv"
FIG4_PROFILES = OUTPUT / "source" / "fig4c_structure_profiles.csv"
FIG4_ELEMENTS = OUTPUT / "source" / "fig4c_element_frequencies.csv"
RL_LOGS = {
    "fe": ROOT / "output" / "rl_diagnostics" / "668001-seed42-grpo-pipo-probe32" / "task_0" / "training.log",
    "bg": ROOT / "output" / "rl_diagnostics" / "668001-seed42-grpo-pipo-probe32" / "task_1" / "training.log",
}
INTERIM_PAIRED = {
    task: ROOT / "output" / "rl_diagnostics"
    / "668001-seed42-grpo-pipo-probe32" / "interim_paired_step39"
    / f"{task}.json"
    for task in ("fe", "bg")
}
BAND_GAP_MAX = 10.0

COLORS = {
    "base": "#E69F00",
    "cfg": "#0072B2",
    "train": "#666666",
    "open": "#F28E2B",
    "pipo": "#8E6CBB",
    "pirl": "#CC79A7",
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


def cropped_image(path: Path, white_threshold: float = 0.985) -> np.ndarray:
    image = plt.imread(path)
    content = np.any(image[..., :3] < white_threshold, axis=2)
    if not content.any():
        return image
    rows, columns = np.where(content)
    return image[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1]


def fig1() -> None:
    fig = plt.figure(figsize=(7.2047, 6.6535), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, height_ratios=[3.5, 0.85, 0.85])

    ax = fig.add_subplot(grid[0, :])
    panel(ax, "a")
    if not FIG1_OVERVIEW.is_file():
        raise FileNotFoundError(f"Missing Fig. 1 overview placeholder: {FIG1_OVERVIEW}")
    ax.imshow(cropped_image(FIG1_OVERVIEW))
    ax.axis("off")
    data_tag(ax, "uploaded architecture placeholder", COLORS["muted"])

    ax = fig.add_subplot(grid[1, :2])
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

    ax = fig.add_subplot(grid[1, 2:])
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

    ax = fig.add_subplot(grid[2, :2])
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

    ax = fig.add_subplot(grid[2, 2:])
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

    footer(fig, "Panel a is an uploaded placeholder; quantitative method evidence remains pending")
    save(fig, "fig1_crystalpirl_method_blueprint.png")


def _rl_records(task: str) -> list[dict]:
    path = RL_LOGS[task]
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "step" in payload and payload.get("algorithm") == "grpo":
            records.append(payload)
    return records


def _rolling_stats(values: np.ndarray, max_window: int = 15) -> tuple[np.ndarray, np.ndarray]:
    window = min(max_window, len(values))
    if window > 1 and window % 2 == 0:
        window -= 1
    series = pd.Series(values, dtype=float)
    minimum = max(1, window // 3)
    mean = series.rolling(window, center=True, min_periods=minimum).mean()
    std = series.rolling(window, center=True, min_periods=minimum).std(ddof=0).fillna(0)
    return mean.to_numpy(), std.to_numpy()


def _reward_trace(ax: plt.Axes) -> None:
    maximum = 0
    for task, label, color in [("fe", "FE", COLORS["open"]), ("bg", "BG", COLORS["cfg"])]:
        records = _rl_records(task)
        if not records:
            continue
        steps = np.asarray([record["step"] for record in records])
        values = np.asarray([record["reward"]["reward_mean"] for record in records])
        mean, std = _rolling_stats(values)
        ax.plot(steps, values, color=color, alpha=0.22, linewidth=0.55)
        ax.plot(steps, mean, color=color, linewidth=1.2, label=label)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)
        maximum = max(maximum, len(records))
    if maximum == 0:
        pending(ax, "Reward trajectory", "needs reward_mean for FE/BG across 200 updates")
        return
    ax.axhline(0, color="#B0B0B0", linewidth=0.6)
    ax.set_xlabel("RL update")
    ax.set_ylabel("Raw reward")
    ax.legend(loc="best")


def _gradient_trace(ax: plt.Axes) -> None:
    maximum = 0
    for task, label, color in [("fe", "FE", COLORS["open"]), ("bg", "BG", COLORS["cfg"])]:
        records = _rl_records(task)
        if not records:
            continue
        steps = np.asarray([record["step"] for record in records])
        values = np.asarray([record["gradient_norm"] for record in records], dtype=float)
        mean, std = _rolling_stats(values)
        ax.plot(steps, values, color=color, alpha=0.22, linewidth=0.55)
        ax.plot(steps, mean, color=color, linewidth=1.2, label=label)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
        maximum = max(maximum, len(records))
    if maximum == 0:
        pending(ax, "Policy gradient norm", "needs gradient_norm across 200 updates")
        return
    ax.set_xlabel("RL update")
    ax.set_ylabel("Policy gradient norm")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")


def _pipo_trace(ax: plt.Axes) -> None:
    found = False
    for task, label, color in [("fe", "FE", COLORS["open"]), ("bg", "BG", COLORS["cfg"])]:
        rows = []
        for record in _rl_records(task):
            feedback = record.get("pipo_feedback") or {}
            if feedback.get("feedback") is not None:
                rows.append((record["step"], feedback["feedback"]))
        if not rows:
            continue
        found = True
        steps = np.asarray([step for step, _ in rows])
        signal = np.asarray([item["signal"] for _, item in rows])
        modulation = np.asarray([item["modulation"] for _, item in rows])
        ax.plot(steps, signal, color=color, linewidth=0.8, alpha=0.45, marker="o", markersize=2.5, label=f"{label} ξ")
        ax.plot(steps, modulation, color=color, linewidth=1.2, marker="o", markersize=2.5, label=f"{label} φ(ξ)")
    if not found:
        pending(ax, "PIPO historical feedback", "starts after K=8 warm-up updates; needs ξ and φλ(ξ)")
        return
    ax.axhline(0, color=COLORS["ink"], linewidth=0.7)
    ax.set_xlabel("RL update")
    ax.set_ylabel("Historical feedback")
    ax.legend(loc="best", ncol=2, fontsize=5.2)


def _final_paired_evaluations() -> dict[str, dict]:
    final = {}
    for task, path in RL_LOGS.items():
        if not path.is_file():
            continue
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "final_paired_evaluation" in payload:
                final[task] = payload["final_paired_evaluation"]
    return final


def _interim_paired_evaluations() -> tuple[dict[str, dict], int | None]:
    evaluations = {}
    completed_updates = []
    for task, path in INTERIM_PAIRED.items():
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        if payload.get("status") != "interim_checkpoint_paired_evaluation":
            continue
        evaluations[task] = payload["evaluation"]
        completed_updates.append(int(payload["completed_updates"]))
    if len(evaluations) != len(INTERIM_PAIRED):
        return {}, None
    return evaluations, min(completed_updates)


def _interim_reward_shift(
    records: list[dict], window: int = 8, resamples: int = 20000
) -> tuple[float, float, float] | None:
    if len(records) < 2 * window:
        return None
    early = np.asarray(
        [record["reward"]["reward_mean"] for record in records[:window]],
        dtype=float,
    )
    late = np.asarray(
        [record["reward"]["reward_mean"] for record in records[-window:]],
        dtype=float,
    )
    rng = np.random.default_rng(42)
    early_indices = rng.integers(0, window, size=(resamples, window))
    late_indices = rng.integers(0, window, size=(resamples, window))
    bootstrap = late[late_indices].mean(axis=1) - early[early_indices].mean(axis=1)
    return (
        float(late.mean() - early.mean()),
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    )


def _interim_reward_panel(ax: plt.Axes) -> None:
    rows = []
    for y, task, label, color in [
        (1, "fe", "FE", COLORS["open"]),
        (0, "bg", "BG", COLORS["cfg"]),
    ]:
        records = _rl_records(task)
        interval = _interim_reward_shift(records)
        if interval is None:
            pending(
                ax,
                "Interim training reward shift",
                "needs at least 16 updates for an 8-update early/late comparison",
            )
            return
        mean, lower, upper = interval
        ax.plot([lower, upper], [y, y], color=color, linewidth=2)
        ax.scatter([mean], [y], color=color, s=25, zorder=3, label=label)
        rows.append(
            {
                "property": task,
                "updates_available": len(records),
                "early_window": "first_8_updates",
                "late_window": "last_8_updates",
                "mean_reward_shift": mean,
                "bootstrap_95_lower": lower,
                "bootstrap_95_upper": upper,
                "paired": False,
                "independent_evaluation": False,
            }
        )
    pd.DataFrame(rows).to_csv(
        OUTPUT / "source" / "fig2d_interim_reward_shift.csv", index=False
    )
    ax.axvline(0, color=COLORS["ink"], linewidth=0.7)
    ax.set_yticks([1, 0], ["FE", "BG"])
    ax.set_xlabel("Training reward shift\n(last 8 - first 8 updates)")
    data_tag(
        ax,
        f"diagnostic only: 95% bootstrap interval, <= {max(row['updates_available'] for row in rows)} updates",
        COLORS["warn"],
    )


def _final_lcb(ax: plt.Axes) -> None:
    evaluations = _final_paired_evaluations()
    stage = "final"
    completed_updates = None
    if len(evaluations) != 2:
        evaluations, completed_updates = _interim_paired_evaluations()
        stage = "interim"
        if len(evaluations) != 2:
            _interim_reward_panel(ax)
            return
    rows = []
    for y, task, label, color in [
        (1, "fe", "FE", COLORS["open"]),
        (0, "bg", "BG", COLORS["cfg"]),
    ]:
        decision = evaluations[task]["decision"]
        mean = float(decision["mean_delta"][0])
        lcb = float(decision["lower_confidence_bound"][0])
        ax.plot([lcb, mean], [y, y], color=color, linewidth=2)
        ax.scatter([mean], [y], color=color, s=25, zorder=3, label=label)
        ax.text(
            mean,
            y - 0.16 if y == 1 else y + 0.16,
            decision["action"],
            ha="center",
            fontsize=5.5,
            color=color,
        )
        rows.append(
            {
                "property": task,
                "stage": stage,
                "completed_updates": completed_updates,
                "mean_reward_delta_vs_base": mean,
                "bootstrap_lcb": lcb,
                "decision": decision["action"],
                "paired_prompts": 32,
            }
        )
    pd.DataFrame(rows).to_csv(
        OUTPUT / "source" / "fig2d_paired_policy_evaluation.csv", index=False
    )
    ax.axvline(0, color=COLORS["ink"], linewidth=0.7)
    ax.set_yticks([1, 0], ["FE", "BG"])
    ax.set_ylim(-0.35, 1.55)
    ax.set_xlabel("Paired reward delta vs Base")
    if stage == "final":
        data_tag(ax, "real final: 32 paired prompts", COLORS["pipo"])
    else:
        data_tag(
            ax,
            f"real interim: {completed_updates} updates, 32 paired prompts",
            COLORS["warn"],
        )

def fig2() -> None:
    fig = plt.figure(figsize=(7.2047, 2.95), constrained_layout=True)
    grid = fig.add_gridspec(1, 3)

    ax = fig.add_subplot(grid[0, 0])
    panel(ax, "a")
    ax.set_box_aspect(1)
    _reward_trace(ax)

    ax = fig.add_subplot(grid[0, 1])
    panel(ax, "b")
    ax.set_box_aspect(1)
    _gradient_trace(ax)

    ax = fig.add_subplot(grid[0, 2])
    panel(ax, "c")
    ax.set_box_aspect(1)
    _pipo_trace(ax)

    save(fig, "fig2_verified_policy_improvement_blueprint.png")

def _plotting_values(values: np.ndarray, property_name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(values)
    if property_name == "band_gap":
        keep &= values <= BAND_GAP_MAX
    return values[keep]


def _property_series(property_name: str) -> list[tuple[str, np.ndarray, str]]:
    train = pd.read_csv(TRAIN_MP20)
    datasets = [("Training set", train[property_name].to_numpy(dtype=float), COLORS["train"])]
    for label, path, color in [
        ("Base", BASE_ABINITIO, COLORS["base"]),
        ("CFG", CFG_ABINITIO[property_name], COLORS["cfg"]),
    ]:
        frame = pd.read_csv(path)
        datasets.append((label, frame[f"predicted_{property_name}"].to_numpy(dtype=float), color))
    return [
        (label, _plotting_values(values, property_name), color)
        for label, values, color in datasets
    ]

def _property_kde(ax: plt.Axes, property_name: str) -> None:
    target, xlabel = {
        "formation_energy_per_atom": (-1.5, "Formation energy (eV atom⁻¹)"),
        "band_gap": (2.0, "Band gap (eV)"),
    }[property_name]
    datasets = _property_series(property_name)
    lower = min(values.min() for _, values, _ in datasets)
    upper = (
        BAND_GAP_MAX
        if property_name == "band_gap"
        else max(values.max() for _, values, _ in datasets)
    )
    margin = 0.03 * (upper - lower)
    grid = np.linspace(lower - margin, upper, 450)
    for label, values, color in datasets:
        density = gaussian_kde(values)(grid)
        ax.plot(grid, density, color=color, linewidth=1.15, label=f"{label} (n={len(values):,})")
        ax.fill_between(grid, 0, density, color=color, alpha=0.12, linewidth=0)
    ax.axvline(target, color=COLORS["reject"], linestyle="--", linewidth=1.0)
    ax.text(target, 0.78, "Target", transform=ax.get_xaxis_transform(), rotation=90, rotation_mode="anchor", va="top", ha="right", fontsize=5.5, color=COLORS["reject"])
    ax.plot([], [], color=COLORS["pirl"], linewidth=1.2, label="CrystalPIRL (pending)")
    ax.set_xlim(lower - margin, upper)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density")
    legend_location = "upper left" if property_name == "formation_energy_per_atom" else "upper right"
    ax.legend(loc=legend_location, fontsize=5.0)

def _available_quality_metrics() -> pd.DataFrame:
    evaluated = (
        pd.read_csv(FIG3_QUALITY_METRICS)
        if FIG3_QUALITY_METRICS.is_file()
        else pd.DataFrame()
    )
    rows = []
    for method, metrics_path in [
        ("Base", BASE_STRUCTURAL_METRICS),
        ("CFG", JOINT_CFG_STRUCTURAL_METRICS),
    ]:
        metrics = json.loads(metrics_path.read_text())
        row = {
            "method": method,
            "compositional_validity": 100.0 * float(metrics["comp_valid"]),
            "structural_validity": 100.0 * float(metrics["struct_valid"]),
            "mp_hull_stability": np.nan,
            "sun": np.nan,
            "uniqueness": np.nan,
            "novelty_among_unique": np.nan,
            "novelty_reference": None,
        }
        if not evaluated.empty:
            match = evaluated[evaluated["method"] == method]
            if len(match) == 1:
                for name in (
                    "sun",
                    "uniqueness",
                    "novelty_among_unique",
                    "mp_hull_stability",
                ):
                    row[name] = 100.0 * float(match.iloc[0][name])
                row["novelty_reference"] = str(
                    match.iloc[0]["novelty_reference"]
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _quality_metrics_frame(ax: plt.Axes) -> None:
    metrics = ["S.U.N.", "Uniqueness", "Novelty", "Stability", "Comp.\nvalidity", "Struct.\nvalidity"]
    ax.set_xlim(-0.5, len(metrics) - 0.5)
    ax.set_ylim(0, 110)
    ax.set_xticks(np.arange(len(metrics)), metrics, rotation=18, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Rate (%)")
    values = _available_quality_metrics()
    positions = np.arange(len(metrics), dtype=float)
    width = 0.32
    for offset, row in zip((-width / 2, width / 2), values.itertuples()):
        color = COLORS["base"] if row.method == "Base" else COLORS["cfg"]
        bar_values = [
            row.sun,
            row.uniqueness,
            row.novelty_among_unique,
            row.mp_hull_stability,
            row.compositional_validity,
            row.structural_validity,
        ]
        ax.bar(positions + offset, bar_values, width=width, color=color, label=row.method)
    ax.plot([], [], marker="s", linestyle="", color=COLORS["pirl"], label="CrystalPIRL (pending)")
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(label) for label in ["Base", "CFG", "CrystalPIRL (pending)"]]
    ax.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        fontsize=5.0,
    )

def _joint_property_data() -> list[tuple[str, np.ndarray, np.ndarray, str]]:
    specifications = [
        ("Training set", TRAIN_MP20, "formation_energy_per_atom", "band_gap", COLORS["train"]),
        ("Base", BASE_ABINITIO, "predicted_formation_energy_per_atom", "predicted_band_gap", COLORS["base"]),
        ("CFG", JOINT_CFG_ABINITIO, "predicted_formation_energy_per_atom", "predicted_band_gap", COLORS["cfg"]),
    ]
    datasets = []
    for label, path, fe_column, bg_column, color in specifications:
        frame = pd.read_csv(path)
        fe = pd.to_numeric(frame[fe_column], errors="coerce").to_numpy(dtype=float)
        bg = pd.to_numeric(frame[bg_column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(fe) & np.isfinite(bg) & (bg <= BAND_GAP_MAX)
        datasets.append((label, fe[valid], bg[valid], color))
    return datasets

def _joint_property_density(ax: plt.Axes) -> None:
    datasets = _joint_property_data()
    x_min = min(values.min() for _, values, _, _ in datasets)
    x_max = max(values.max() for _, values, _, _ in datasets)
    y_min = min(values.min() for _, _, values, _ in datasets)
    y_max = BAND_GAP_MAX
    x_edges = np.linspace(x_min, x_max, 100)
    y_edges = np.linspace(y_min, y_max, 100)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    extent = (x_centers.min(), x_centers.max(), y_centers.min(), y_centers.max())
    handles = []
    for label, fe, bg, color in datasets:
        density, _, _ = np.histogram2d(fe, bg, bins=(x_edges, y_edges), density=True)
        density = gaussian_filter(density, sigma=1.5)
        density = density / density.max()
        density[density < 0.08] = 0.0
        rgba = np.empty((*density.T.shape, 4), dtype=float)
        rgba[..., :3] = mpl.colors.to_rgb(color)
        rgba[..., 3] = 0.34 * np.sqrt(density.T)
        ax.imshow(rgba, extent=extent, origin="lower", aspect="auto", interpolation="bilinear")
        handles.append(
            Patch(facecolor=mpl.colors.to_rgba(color, 0.28), edgecolor="none", label=f"{label} (n={len(fe):,})")
        )
    handles.append(
        Patch(facecolor="none", edgecolor=COLORS["pirl"], linestyle="--", label="CrystalPIRL (pending)")
    )
    target = ax.scatter([-1.5], [2.0], marker="*", s=34, color=COLORS["reject"], edgecolor="white", linewidth=0.5, zorder=5, label="Target")
    handles.append(target)
    # Focus the joint-property view on the populated target region.  The
    # underlying samples remain unchanged; values outside this window are
    # simply outside the displayed plotting extent.
    ax.set_xlim(-3.0, 0.0)
    ax.set_ylim(0.0, 6.0)
    ax.set_xlabel("Formation energy (eV atom⁻¹)")
    ax.set_ylabel("Band gap (eV)")
    ax.legend(handles=handles, loc="upper left", fontsize=5.0)

def fig3() -> None:
    fig = plt.figure(figsize=(7.2047, 5.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(grid[0, 0])
    panel(ax, "a")
    _quality_metrics_frame(ax)

    ax = fig.add_subplot(grid[0, 1])
    panel(ax, "b")
    _property_kde(ax, "formation_energy_per_atom")

    ax = fig.add_subplot(grid[1, 0])
    panel(ax, "c")
    _property_kde(ax, "band_gap")

    ax = fig.add_subplot(grid[1, 1])
    panel(ax, "d")
    _joint_property_density(ax)

    save(fig, "fig3_fe_bg_control_blueprint.png")

def _fig4_tradeoff(ax: plt.Axes) -> None:
    data = pd.read_csv(FIG4_TRADEOFF)
    colors = {
        "Base": COLORS["base"],
        "CFG-FE": "#009E73",
        "CFG-BG": "#56B4E9",
        "CFG-FE+BG": COLORS["cfg"],
    }
    offsets = {
        "Base": (4, 4),
        "CFG-FE": (4, -10),
        "CFG-BG": (4, 4),
        "CFG-FE+BG": (4, 4),
    }
    x = data["structure_diversity"].to_numpy(dtype=float)
    y = 100.0 * data["valid_joint_target_yield"].to_numpy(dtype=float)
    for row, x_value, y_value in zip(data.itertuples(), x, y):
        color = colors[row.method]
        ax.scatter(
            x_value,
            y_value,
            s=38,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax.annotate(
            row.method,
            (x_value, y_value),
            xytext=offsets[row.method],
            textcoords="offset points",
            fontsize=5.5,
            color=color,
        )
    nondominated = [
        index
        for index in range(len(data))
        if not any(
            x[other] >= x[index]
            and y[other] >= y[index]
            and (x[other] > x[index] or y[other] > y[index])
            for other in range(len(data))
        )
    ]
    order = sorted(nondominated, key=lambda index: x[index])
    if len(order) > 1:
        ax.plot(
            x[order],
            y[order],
            color=COLORS["ink"],
            linestyle="--",
            linewidth=0.7,
            alpha=0.55,
        )
    ax.set_xlabel("Structural diversity (mean CrystalNN distance)")
    ax.set_ylabel("Valid joint-target yield (%)")
    data_tag(ax, "real: seed=42; n=1,000; no CI", COLORS["warn"])


def _fig4_tsne(ax: plt.Axes) -> None:
    data = pd.read_csv(FIG4_TSNE)
    colors = {
        "Base": COLORS["base"],
        "CFG": COLORS["cfg"],
    }
    training = data[data["method"] == "Training set"]
    x_min, x_max = data["tsne_1"].min(), data["tsne_1"].max()
    y_min, y_max = data["tsne_2"].min(), data["tsne_2"].max()
    density, x_edges, y_edges = np.histogram2d(
        training["tsne_1"],
        training["tsne_2"],
        bins=160,
        range=((x_min, x_max), (y_min, y_max)),
    )
    density = gaussian_filter(density.T, sigma=2.0)
    density /= density.max()
    alpha = np.where(density >= 0.01, 0.08 + 0.70 * density**0.65, 0.0)
    training_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "training_density", ["#F7F2FA", "#6A3D9A"]
    )
    ax.imshow(
        density,
        origin="lower",
        extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
        aspect="auto",
        cmap=training_cmap,
        alpha=alpha,
        interpolation="bilinear",
        rasterized=True,
        zorder=0,
    )
    handles = [
        Patch(
            facecolor="#6A3D9A",
            edgecolor="none",
            alpha=0.65,
            label=f"Training set density (n={len(training):,})",
        )
    ]
    for method in ["Base", "CFG"]:
        frame = data[data["method"] == method]
        handle = ax.scatter(
            frame["tsne_1"],
            frame["tsne_2"],
            s=3.0,
            alpha=0.34,
            color=colors[method],
            linewidth=0,
            rasterized=True,
            label=f"{method} (n={len(frame):,})",
            zorder=2,
        )
        handles.append(handle)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        markerscale=2.5,
        fontsize=5.0,
        borderaxespad=0,
    )


def _group_fraction(
    data: pd.DataFrame, column: str, categories: list
) -> pd.DataFrame:
    grouped = (
        data.assign(category=data[column])
        .groupby(["method", "category"])
        .size()
        .rename("count")
        .reset_index()
    )
    grouped["fraction"] = grouped["count"] / grouped.groupby("method")[
        "count"
    ].transform("sum")
    return (
        grouped.pivot(index="category", columns="method", values="fraction")
        .reindex(categories)
        .fillna(0.0)
    )


def _fig4_distributions(fig: plt.Figure, spec) -> None:
    grid = spec.subgridspec(2, 3, wspace=0.45, hspace=0.60)
    profiles = pd.read_csv(FIG4_PROFILES)
    elements = pd.read_csv(FIG4_ELEMENTS)
    methods = ["Training set", "Base", "CFG"]
    colors = {
        "Training set": COLORS["train"],
        "Base": COLORS["base"],
        "CFG": COLORS["cfg"],
    }

    ax = fig.add_subplot(grid[0, 0])
    panel(ax, "c1")
    element_order = (
        elements.groupby("element")["atom_fraction"]
        .sum()
        .nlargest(12)
        .index.tolist()
    )
    matrix = (
        elements.pivot(index="method", columns="element", values="atom_fraction")
        .reindex(index=methods, columns=element_order)
        .fillna(0.0)
    )
    ax.imshow(matrix.to_numpy(), aspect="auto", cmap="Blues")
    ax.set_xticks(
        np.arange(len(element_order)),
        element_order,
        rotation=90,
        rotation_mode="anchor",
        ha="right",
        fontsize=5,
    )
    ax.set_yticks(np.arange(len(methods)), methods, fontsize=5)
    ax.set_xlabel("Element frequency")

    ax = fig.add_subplot(grid[0, 1])
    panel(ax, "c2")
    nary = profiles.copy()
    nary["nary_group"] = nary["n_elements"].clip(upper=5)
    categories = [1, 2, 3, 4, 5]
    fractions = _group_fraction(nary, "nary_group", categories)
    positions = np.arange(len(categories), dtype=float)
    width = 0.24
    for index, method in enumerate(methods):
        ax.bar(
            positions + (index - 1) * width,
            100.0 * fractions[method].to_numpy(),
            width=width,
            color=colors[method],
        )
    ax.set_xticks(positions, ["1", "2", "3", "4", "5+"])
    ax.set_xlabel("Number of elements")
    ax.set_ylabel("Fraction (%)")

    ax = fig.add_subplot(grid[0, 2])
    panel(ax, "c3")
    space_fractions = _group_fraction(
        profiles,
        "space_group",
        sorted(profiles["space_group"].unique()),
    )
    top_groups = (
        space_fractions.sum(axis=1).nlargest(8).index.astype(int).tolist()
    )
    space_fractions = space_fractions.reindex(top_groups)
    positions = np.arange(len(top_groups), dtype=float)
    for index, method in enumerate(methods):
        ax.bar(
            positions + (index - 1) * width,
            100.0 * space_fractions[method].to_numpy(),
            width=width,
            color=colors[method],
        )
    ax.set_xticks(
        positions,
        ["?" if group == 0 else str(group) for group in top_groups],
        rotation=90,
        rotation_mode="anchor",
        ha="right",
    )
    ax.set_xlabel("Space group")
    ax.set_ylabel("Fraction (%)")

    ax = fig.add_subplot(grid[1, 0])
    panel(ax, "c4")
    maximum_atoms = int(profiles["num_atoms"].max())
    bins = np.arange(0.5, maximum_atoms + 1.5, 1.0)
    for method in methods:
        values = profiles.loc[profiles["method"] == method, "num_atoms"]
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.0,
            color=colors[method],
        )
    ax.set_xlabel("Atoms per cell")
    ax.set_ylabel("Probability")

    ax = fig.add_subplot(grid[1, 1:])
    panel(ax, "c5")
    finite_density = profiles[
        np.isfinite(profiles["density_g_cm3"])
        & (profiles["density_g_cm3"] > 0)
    ]
    upper = float(finite_density["density_g_cm3"].quantile(0.995))
    density_grid = np.linspace(0, upper, 240)
    for method in methods:
        values = finite_density.loc[
            finite_density["method"] == method, "density_g_cm3"
        ].to_numpy(dtype=float)
        density = gaussian_kde(values)(density_grid)
        ax.plot(
            density_grid,
            density,
            color=colors[method],
            linewidth=1.0,
            label=method,
        )
        ax.fill_between(
            density_grid,
            0,
            density,
            color=colors[method],
            alpha=0.10,
            linewidth=0,
        )
    ax.set_xlim(0, upper)
    ax.set_xlabel("Density (g cm^-3)")
    ax.set_ylabel("Probability density")
    ax.legend(loc="best", fontsize=5.0)
    ax.text(
        0.98,
        0.03,
        "x-range: 99.5%\n(all rows retained)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=COLORS["muted"],
    )


def fig4() -> None:
    fig = plt.figure(figsize=(7.2047, 6.35))
    grid = fig.add_gridspec(
        2,
        10,
        height_ratios=[1.05, 1.8],
        left=0.08,
        right=0.98,
        bottom=0.07,
        top=0.94,
        hspace=0.46,
        wspace=0.35,
    )

    ax = fig.add_subplot(grid[0, :5])
    panel(ax, "a")
    _fig4_tradeoff(ax)

    ax = fig.add_subplot(grid[0, 5:])
    panel(ax, "b")
    _fig4_tsne(ax)

    _fig4_distributions(fig, grid[1, :])

    footer(
        fig,
        "Panels a-c use seed=42 real data; CrystalPIRL, MLFF and DFT validation remain pending",
    )
    save(fig, "fig4_quality_search_space_blueprint.png")

def main() -> None:
    required = [
        FIG1_OVERVIEW,
        TRAIN_MP20,
        BASE_ABINITIO,
        BASE_STRUCTURAL_METRICS,
        *CFG_ABINITIO.values(),
        JOINT_CFG_ABINITIO,
        JOINT_CFG_STRUCTURAL_METRICS,
        FIG4_TRADEOFF,
        FIG4_TSNE,
        FIG4_PROFILES,
        FIG4_ELEMENTS,
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
