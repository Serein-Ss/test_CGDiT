"""Plot a single-seed RL diagnostic analogous to Chemeleon2 Fig. 2b,c."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


RUN_ORDER = [
    ("ppo", "fe", "open"),
    ("ppo", "fe", "pipo"),
    ("ppo", "fe", "pirl"),
    ("grpo", "fe", "open"),
    ("grpo", "fe", "pipo"),
    ("grpo", "fe", "pirl"),
    ("ppo", "bg", "open"),
    ("ppo", "bg", "pipo"),
    ("ppo", "bg", "pirl"),
    ("grpo", "bg", "open"),
    ("grpo", "bg", "pipo"),
    ("grpo", "bg", "pirl"),
]

MODE_LABEL = {"open": "Open", "pipo": "PIPO", "pirl": "CrystalPIRL"}
MODE_COLOR = {"open": "#6B7280", "pipo": "#D97706", "pirl": "#008B72"}
ALGORITHM_STYLE = {
    "ppo": {"linestyle": "-", "marker": "o"},
    "grpo": {"linestyle": "--", "marker": "s"},
}
ACTION_VALUE = {"rollback": 0, "reject": 1, "attenuate": 2, "accept": 3}
ACTION_COLOR = {
    "rollback": "#7E57C2",
    "reject": "#D55E5E",
    "attenuate": "#E6AB02",
    "accept": "#3A923A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("assets/tmp"))
    return parser.parse_args()


def run_name(algorithm: str, prop: str, mode: str) -> str:
    suffix = "" if mode == "open" else f"_{mode}"
    return f"{algorithm}_{prop}_seed42{suffix}"


def audit_fields(record: dict) -> tuple[str, float, float | None, bool]:
    if record["pirl"]:
        decision = record["decision"]
        fixed_lcb = decision["initial"]["absolute"]["lower_confidence_bound"][0]
        holdout = decision.get("holdout")
        holdout_lcb = None
        if holdout is not None:
            holdout_lcb = holdout["decision"]["absolute"][
                "lower_confidence_bound"
            ][0]
        return (
            decision["action"],
            fixed_lcb,
            holdout_lcb,
            bool(decision["verified_checkpoint_update"]),
        )

    fixed = record["audit"]["fixed"]["decision"]
    holdout = record["audit"]["holdout"]["decision"]
    return (
        fixed["action"],
        fixed["absolute"]["lower_confidence_bound"][0],
        holdout["absolute"]["lower_confidence_bound"][0],
        True,
    )


def load_data(train_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    updates = []
    finals = []
    for algorithm, prop, mode in RUN_ORDER:
        root = train_root / run_name(algorithm, prop, mode)
        metrics_path = root / "train_results" / "metrics.jsonl"
        records = [
            json.loads(line)
            for line in metrics_path.read_text().splitlines()
            if line.strip()
        ]
        if len(records) != 8:
            raise RuntimeError(f"Expected 8 updates in {metrics_path}, got {len(records)}")

        for record in records:
            action, fixed_lcb, holdout_lcb, applied = audit_fields(record)
            updates.append(
                {
                    "algorithm": algorithm,
                    "property": prop,
                    "mode": mode,
                    "step": int(record["step"]) + 1,
                    "reward_mean": record["reward"]["reward_mean"],
                    "valid_fraction": record["reward"]["valid_fraction"],
                    "audit_action": action,
                    "fixed_reward_lcb": fixed_lcb,
                    "holdout_reward_lcb": holdout_lcb,
                    "checkpoint_updated": applied,
                }
            )

        final = json.loads(
            (root / "train_results" / "final_paired_evaluation.json").read_text()
        )
        finals.append(
            {
                "algorithm": algorithm,
                "property": prop,
                "mode": mode,
                "reward_delta": final["final"]["reward_mean"]
                - final["baseline"]["reward_mean"],
                "reward_lcb": final["decision"]["lower_confidence_bound"][0],
                "validity_delta": final["final"]["valid_fraction"]
                - final["baseline"]["valid_fraction"],
                "final_action": final["decision"]["action"],
            }
        )

    return pd.DataFrame(updates), pd.DataFrame(finals)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
    )


def plot_reward_panel(ax: plt.Axes, updates: pd.DataFrame, prop: str) -> None:
    frame = updates[updates["property"] == prop]
    for algorithm in ("ppo", "grpo"):
        for mode in ("open", "pipo", "pirl"):
            data = frame[
                (frame["algorithm"] == algorithm) & (frame["mode"] == mode)
            ].sort_values("step")
            style = ALGORITHM_STYLE[algorithm]
            ax.plot(
                data["step"],
                data["reward_mean"],
                color=MODE_COLOR[mode],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=3.2,
                linewidth=1.15,
                alpha=0.9,
            )
    ax.axhline(0, color="#B8B8B8", linewidth=0.7, zorder=0)
    ax.set_xlim(0.7, 8.3)
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("RL update")
    ax.set_ylabel(f"{'Formation-energy' if prop == 'fe' else 'Band-gap'} rollout reward")


def plot_final_delta(ax: plt.Axes, finals: pd.DataFrame) -> None:
    combinations = [(a, m) for a in ("ppo", "grpo") for m in ("open", "pipo", "pirl")]
    x = np.arange(len(combinations))
    offsets = {"fe": -0.11, "bg": 0.11}
    markers = {"fe": "o", "bg": "^"}
    for prop in ("fe", "bg"):
        for index, (algorithm, mode) in enumerate(combinations):
            row = finals[
                (finals["algorithm"] == algorithm)
                & (finals["mode"] == mode)
                & (finals["property"] == prop)
            ].iloc[0]
            xpos = index + offsets[prop]
            ax.vlines(
                xpos,
                row["reward_lcb"],
                row["reward_delta"],
                color=MODE_COLOR[mode],
                linewidth=1.1,
            )
            ax.scatter(
                xpos,
                row["reward_delta"],
                marker=markers[prop],
                s=24,
                color=MODE_COLOR[mode],
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
    ax.axhspan(0, 0.32, color="#E8F4EA", alpha=0.55, zorder=0)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{algorithm.upper()}\n{MODE_LABEL[mode]}" for algorithm, mode in combinations],
        fontsize=5.8,
    )
    ax.set_ylabel("Final paired Δ reward\nvs frozen baseline")
    ax.text(
        0.02,
        0.03,
        "Point: mean; line: one-sided LCB (n = 8)",
        transform=ax.transAxes,
        fontsize=5.5,
        color="#4B5563",
        va="bottom",
    )


def plot_action_matrix(ax: plt.Axes, updates: pd.DataFrame) -> None:
    matrix = []
    applied = []
    labels = []
    for algorithm, prop, mode in RUN_ORDER:
        data = updates[
            (updates["algorithm"] == algorithm)
            & (updates["property"] == prop)
            & (updates["mode"] == mode)
        ].sort_values("step")
        matrix.append([ACTION_VALUE[action] for action in data["audit_action"]])
        applied.append(data["checkpoint_updated"].to_numpy(dtype=bool))
        labels.append(
            f"{algorithm.upper()}-{prop.upper()} {MODE_LABEL[mode]}"
            + (" gate" if mode == "pirl" else " audit")
        )

    cmap = ListedColormap([ACTION_COLOR[name] for name in ("rollback", "reject", "attenuate", "accept")])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    ax.imshow(np.asarray(matrix), cmap=cmap, norm=norm, aspect="auto")
    yy, xx = np.where(np.asarray(applied))
    ax.scatter(xx, yy, s=5, color="#111111", marker=".", zorder=3)
    ax.set_xticks(range(8), labels=range(1, 9))
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=5.1)
    ax.set_xlabel("RL update")
    ax.tick_params(axis="both", length=0)
    for boundary in (2.5, 5.5, 8.5):
        ax.axhline(boundary, color="white", linewidth=1.3)


def main() -> None:
    args = parse_args()
    updates, finals = load_data(args.train_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    updates.to_csv(args.output_dir / "rl_fig2_seed42_updates.csv", index=False)
    finals.to_csv(args.output_dir / "rl_fig2_seed42_final_pairs.csv", index=False)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    plot_reward_panel(axes[0, 0], updates, "fe")
    plot_reward_panel(axes[0, 1], updates, "bg")
    plot_final_delta(axes[1, 0], finals)
    plot_action_matrix(axes[1, 1], updates)

    for ax, label in zip(axes.flat, "abcd", strict=True):
        panel_label(ax, label)

    mode_handles = [
        Line2D([0], [0], color=MODE_COLOR[mode], linewidth=2, label=MODE_LABEL[mode])
        for mode in ("open", "pipo", "pirl")
    ]
    algorithm_handles = [
        Line2D(
            [0],
            [0],
            color="#222222",
            linestyle=ALGORITHM_STYLE[algorithm]["linestyle"],
            marker=ALGORITHM_STYLE[algorithm]["marker"],
            markersize=3.5,
            label=algorithm.upper(),
        )
        for algorithm in ("ppo", "grpo")
    ]
    property_handles = [
        Line2D([0], [0], color="#333333", marker="o", linestyle="none", label="FE"),
        Line2D([0], [0], color="#333333", marker="^", linestyle="none", label="BG"),
    ]
    fig.legend(
        handles=mode_handles + algorithm_handles + property_handles,
        loc="upper center",
        ncol=7,
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.3,
        handlelength=2.0,
    )

    action_handles = [
        Patch(facecolor=ACTION_COLOR[name], label=name.capitalize())
        for name in ("accept", "attenuate", "reject", "rollback")
    ]
    action_handles.append(
        Line2D([0], [0], color="#111111", marker=".", linestyle="none", label="Checkpoint updated")
    )
    axes[1, 1].legend(
        handles=action_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=3,
        fontsize=5.5,
        columnspacing=1.0,
        handlelength=1.2,
    )

    fig.text(
        0.985,
        0.012,
        "Seed 42 only; exploratory diagnostic",
        ha="right",
        va="bottom",
        fontsize=5.5,
        color="#6B7280",
    )
    fig.subplots_adjust(left=0.13, right=0.985, top=0.91, bottom=0.12, wspace=0.34, hspace=0.42)
    output = args.output_dir / "rl_fig2_like_diagnostic_seed42.png"
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
