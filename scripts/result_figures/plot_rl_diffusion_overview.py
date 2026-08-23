"""Draw the CGDiT diffusion and CrystalPIRL reinforcement-learning overview."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon


NAVY = "#24465B"
BLUE = "#4C78A8"
BLUE_LIGHT = "#E8F1F7"
TEAL = "#4E9F91"
TEAL_LIGHT = "#E5F3F0"
GOLD = "#D99A3D"
GOLD_LIGHT = "#FAF0DD"
VIOLET = "#7868B2"
VIOLET_LIGHT = "#EEEAF8"
GREEN = "#3C8B69"
GREEN_LIGHT = "#E7F3EC"
RED = "#C65353"
RED_LIGHT = "#F8E9E9"
GRAY = "#69747C"
GRAY_LIGHT = "#E4E8EB"
INK = "#263238"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    subtitle: str = "",
    *,
    facecolor: str = "white",
    edgecolor: str = NAVY,
    title_color: str = INK,
    linestyle: str = "-",
    linewidth: float = 1.0,
    title_size: float = 7.2,
    subtitle_size: float = 5.8,
    zorder: int = 3,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    center_x = x + width / 2
    if subtitle:
        ax.text(
            center_x,
            y + height * 0.64,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=title_color,
            zorder=zorder + 1,
        )
        ax.text(
            center_x,
            y + height * 0.29,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=INK,
            linespacing=1.15,
            zorder=zorder + 1,
        )
    else:
        ax.text(
            center_x,
            y + height / 2,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=title_color,
            zorder=zorder + 1,
        )
    return patch


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = NAVY,
    linewidth: float = 1.2,
    linestyle: str = "-",
    connectionstyle: str = "arc3",
    mutation_scale: float = 9,
    zorder: int = 6,
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def draw_crystal(
    ax,
    center: tuple[float, float],
    width: float,
    height: float,
    atoms: list[tuple[float, float, str, float]],
    label: str,
) -> None:
    cx, cy = center
    dx = width * 0.18
    dy = height * 0.17
    front = [
        (cx - width / 2, cy - height / 2),
        (cx + width / 2, cy - height / 2),
        (cx + width / 2, cy + height / 2),
        (cx - width / 2, cy + height / 2),
    ]
    back = [(x + dx, y + dy) for x, y in front]
    ax.add_patch(
        Polygon(front, closed=True, fill=False, edgecolor=GRAY, linewidth=0.75, zorder=3)
    )
    ax.add_patch(
        Polygon(
            back,
            closed=True,
            fill=False,
            edgecolor=GRAY,
            linewidth=0.65,
            linestyle="--",
            zorder=2,
        )
    )
    for point_a, point_b in zip(front, back):
        ax.plot(
            [point_a[0], point_b[0]],
            [point_a[1], point_b[1]],
            color=GRAY,
            linewidth=0.65,
            linestyle="--",
            zorder=2,
        )
    for rel_x, rel_y, color, size in atoms:
        atom = Circle(
            (cx + rel_x * width, cy + rel_y * height),
            radius=size * min(width, height),
            facecolor=color,
            edgecolor="white",
            linewidth=0.65,
            zorder=5,
        )
        ax.add_patch(atom)
    ax.text(
        cx + dx * 0.35,
        cy - height * 0.72,
        label,
        ha="center",
        va="top",
        fontsize=6,
        color=INK,
    )


def add_panel_label(ax, label: str) -> None:
    ax.text(
        0.006,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=INK,
    )


def draw_rl_loop(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_facecolor("#F8FAFC")
    add_panel_label(ax, "a")
    ax.text(
        0.035,
        0.955,
        "CrystalPIRL reinforcement fine-tuning loop",
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=NAVY,
    )

    box(
        ax,
        0.025,
        0.60,
        0.12,
        0.19,
        "Verified policy",
        "Base CGDiT  πᵥ\nCFG = 0",
        facecolor=BLUE_LIGHT,
        edgecolor=BLUE,
    )

    diffusion_card = FancyBboxPatch(
        (0.18, 0.54),
        0.39,
        0.31,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        facecolor="white",
        edgecolor=BLUE,
        linewidth=1.0,
        zorder=1,
    )
    ax.add_patch(diffusion_card)
    ax.text(
        0.375,
        0.815,
        "CGDiT reverse diffusion",
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color=BLUE,
    )
    crystal_atoms = [
        [(-0.25, 0.18, VIOLET, 0.095), (0.18, -0.18, RED, 0.09), (0.2, 0.23, BLUE, 0.085)],
        [(-0.2, 0.18, GOLD, 0.09), (0.18, -0.18, TEAL, 0.085), (0.2, 0.23, VIOLET, 0.08)],
        [(-0.19, 0.12, GOLD, 0.09), (0.16, -0.16, TEAL, 0.085), (0.14, 0.22, TEAL, 0.075)],
        [(-0.18, 0.15, GOLD, 0.09), (0.15, -0.15, TEAL, 0.085), (0.16, 0.22, TEAL, 0.075)],
    ]
    crystal_x = [0.225, 0.325, 0.425, 0.525]
    crystal_labels = ["Mₜ", "Mₜ₋ₖ", "M₁", "M₀"]
    for index, (x, atoms, label) in enumerate(
        zip(crystal_x, crystal_atoms, crystal_labels)
    ):
        draw_crystal(ax, (x, 0.675), 0.055, 0.085, atoms, label)
        if index < len(crystal_x) - 1:
            arrow(
                ax,
                (x + 0.04, 0.675),
                (crystal_x[index + 1] - 0.04, 0.675),
                color=BLUE,
                linewidth=0.9,
                mutation_scale=7,
            )
    ax.text(
        0.375,
        0.565,
        "lattice  K  ·  periodic coordinates  X  ·  atom types  A",
        ha="center",
        va="center",
        fontsize=5.7,
        color=GRAY,
    )

    box(
        ax,
        0.61,
        0.60,
        0.15,
        0.19,
        "Material reward",
        "Fixed-scale FE / BG\nvalidity gate",
        facecolor=GOLD_LIGHT,
        edgecolor=GOLD,
    )
    box(
        ax,
        0.80,
        0.60,
        0.17,
        0.19,
        "PPO / GRPO",
        "OrbitPO ratio + KL\ncandidate update",
        facecolor=VIOLET_LIGHT,
        edgecolor=VIOLET,
    )
    box(
        ax,
        0.80,
        0.39,
        0.17,
        0.12,
        "Candidate policy  πcand",
        facecolor="white",
        edgecolor=VIOLET,
        title_size=6.8,
    )
    arrow(ax, (0.145, 0.695), (0.18, 0.695), color=BLUE)
    arrow(ax, (0.57, 0.695), (0.61, 0.695), color=GOLD)
    arrow(ax, (0.76, 0.695), (0.80, 0.695), color=VIOLET)
    arrow(ax, (0.885, 0.60), (0.885, 0.51), color=VIOLET)

    box(
        ax,
        0.43,
        0.16,
        0.36,
        0.19,
        "Paired-PIRL verification",
        "common targets + random streams\npaired ΔRraw + safety · bootstrap LCB",
        facecolor=TEAL_LIGHT,
        edgecolor=TEAL,
        title_color=NAVY,
        title_size=7.4,
    )
    arrow(
        ax,
        (0.085, 0.60),
        (0.43, 0.255),
        color=BLUE,
        linewidth=1.0,
        linestyle="--",
        connectionstyle="angle3,angleA=-90,angleB=0",
    )
    ax.text(0.19, 0.285, "verified old", fontsize=5.5, color=BLUE, ha="center")
    arrow(
        ax,
        (0.885, 0.39),
        (0.79, 0.255),
        color=VIOLET,
        linewidth=1.0,
        linestyle="--",
        connectionstyle="angle3,angleA=-90,angleB=180",
    )
    ax.text(0.87, 0.285, "candidate new", fontsize=5.5, color=VIOLET, ha="center")
    arrow(ax, (0.685, 0.60), (0.685, 0.35), color=GOLD, linewidth=0.95)
    ax.text(
        0.702,
        0.455,
        "raw reward",
        fontsize=5.4,
        color=GOLD,
        rotation=90,
        rotation_mode="anchor",
        ha="center",
        va="center",
    )

    decision_specs = [
        (0.435, "Accept", GREEN_LIGHT, GREEN, "update verified"),
        (0.555, "Attenuate", GOLD_LIGHT, GOLD, "scale + recheck"),
        (0.675, "Reject", RED_LIGHT, RED, "rollback"),
    ]
    for x, title, face, edge, subtitle in decision_specs:
        box(
            ax,
            x,
            0.035,
            0.105,
            0.075,
            title,
            subtitle,
            facecolor=face,
            edgecolor=edge,
            title_color=edge,
            title_size=6.1,
            subtitle_size=5.0,
            linewidth=0.9,
        )
        arrow(ax, (0.61, 0.16), (x + 0.0525, 0.11), color=edge, linewidth=0.8, mutation_scale=6)

    arrow(
        ax,
        (0.435, 0.073),
        (0.085, 0.60),
        color=GREEN,
        linewidth=1.15,
        connectionstyle="arc3,rad=-0.34",
        mutation_scale=8,
    )
    ax.text(
        0.20,
        0.055,
        "verified checkpoint / rollback",
        fontsize=5.5,
        color=GREEN,
        ha="center",
    )


def draw_channel_icon(ax, x: float, y: float, channel: str, color: str) -> None:
    if channel == "lattice":
        ax.add_patch(
            Polygon(
                [(x - 0.025, y - 0.035), (x + 0.02, y - 0.035), (x + 0.03, y + 0.03), (x - 0.015, y + 0.03)],
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=1.1,
            )
        )
        ax.plot([x - 0.01, x + 0.035], [y - 0.015, y - 0.015], color=color, lw=0.8)
    elif channel == "coordinate":
        for dx, dy in [(-0.022, -0.02), (0.022, -0.01), (0.0, 0.025)]:
            ax.add_patch(Circle((x + dx, y + dy), 0.011, facecolor=color, edgecolor="white", lw=0.5))
        ax.add_patch(Circle((x, y), 0.043, fill=False, edgecolor=color, linewidth=0.7, linestyle="--"))
    else:
        for index, atom_color in enumerate((GOLD, TEAL, VIOLET)):
            ax.add_patch(
                Circle(
                    (x - 0.025 + index * 0.025, y),
                    0.013,
                    facecolor=atom_color,
                    edgecolor="white",
                    linewidth=0.5,
                )
            )


def draw_diffusion_step(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_facecolor("#FCFCFB")
    add_panel_label(ax, "b")
    ax.text(
        0.035,
        0.955,
        "One symmetry-reduced reverse transition",
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=NAVY,
    )

    rows = [
        (0.63, "Lattice  kₜ", "active symmetry subspace", "lattice", VIOLET, VIOLET_LIGHT),
        (0.38, "Coordinates  Xₜ", "orbit representatives on T³", "coordinate", TEAL, TEAL_LIGHT),
        (0.13, "Atom types  Aₜ", "orbit-level D3PM state", "atom", GOLD, GOLD_LIGHT),
    ]
    output_titles = [
        ("pᴷθ(kₜ₋₁ | Mₜ)", "subspace Gaussian"),
        ("pˣθ(Xₜ₋₁ | Mₜ)", "wrapped Gaussian"),
        ("pᴬθ(Aₜ₋₁ | Mₜ)", "categorical posterior"),
    ]

    for (y, title, subtitle, channel, color, face), (out_title, out_subtitle) in zip(
        rows, output_titles
    ):
        box(
            ax,
            0.035,
            y,
            0.22,
            0.16,
            title,
            subtitle,
            facecolor=face,
            edgecolor=color,
            title_color=color,
            title_size=6.8,
            subtitle_size=5.4,
        )
        draw_channel_icon(ax, 0.075, y + 0.08, channel, color)
        box(
            ax,
            0.62,
            y,
            0.255,
            0.16,
            out_title,
            out_subtitle,
            facecolor="white",
            edgecolor=color,
            title_color=color,
            title_size=6.7,
            subtitle_size=5.4,
        )
        arrow(ax, (0.255, y + 0.08), (0.35, y + 0.08), color=color, linewidth=1.0)
        arrow(ax, (0.55, y + 0.08), (0.62, y + 0.08), color=color, linewidth=1.0)

    box(
        ax,
        0.35,
        0.29,
        0.20,
        0.43,
        "Symmetry-aware\nCSPNet denoiser",
        "φθ(Mₜ, t)\npredicts K / X / A updates",
        facecolor=BLUE_LIGHT,
        edgecolor=BLUE,
        title_color=NAVY,
        title_size=7.3,
        subtitle_size=5.7,
        linewidth=1.15,
    )
    box(
        ax,
        0.35,
        0.78,
        0.20,
        0.105,
        "Mₜ  ·  t  ·  space group  ·  occupied orbits",
        facecolor="white",
        edgecolor=GRAY,
        title_color=GRAY,
        title_size=5.8,
        linewidth=0.8,
    )
    arrow(ax, (0.45, 0.78), (0.45, 0.72), color=BLUE, linewidth=0.9)


    joint = FancyBboxPatch(
        (0.39, 0.025),
        0.50,
        0.105,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        facecolor=NAVY,
        edgecolor=NAVY,
        linewidth=1.0,
        zorder=3,
    )
    ax.add_patch(joint)
    ax.text(
        0.64,
        0.090,
        "log πθ(aₜ | sₜ) = log pᴷ + log pˣ + log pᴬ",
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color="white",
        zorder=4,
    )
    ax.text(
        0.64,
        0.052,
        "transition replay  →  ratio / KL  →  PPO or GRPO",
        ha="center",
        va="center",
        fontsize=5.5,
        color="#DDE8EE",
        zorder=4,
    )
    draw_crystal(
        ax,
        (0.94, 0.48),
        0.065,
        0.11,
        [(-0.18, 0.15, GOLD, 0.085), (0.14, -0.15, TEAL, 0.08), (0.16, 0.2, TEAL, 0.07)],
        "Mₜ₋₁",
    )
    for y, _, _, _, color, _ in rows:
        arrow(ax, (0.875, y + 0.08), (0.905, 0.48), color=color, linewidth=0.75, mutation_scale=6)


def build_figure() -> plt.Figure:
    configure_style()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.2),
        gridspec_kw={"height_ratios": [1.03, 1.0], "hspace": 0.075},
    )
    draw_rl_loop(axes[0])
    draw_diffusion_step(axes[1])
    return fig


def save_figure(fig: plt.Figure, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "cgdit_crystalpirl_overview"
    fig.savefig(prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "assets/tmp",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig = build_figure()
    save_figure(fig, args.output_dir.resolve())
    plt.close(fig)


if __name__ == "__main__":
    main()
