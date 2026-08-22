"""Plot full FE/BG conditioning effects and representative generated crystals."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path

from ase.visualize.plot import plot_atoms
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
import torch

from cgdit.common.evaluation_utils import smact_validity, structure_validity
from cgdit.evaluation.generated_properties import _crystal_array_list


BLUE = "#0F4D92"
TEAL = "#33B5A5"
GRAY = "#767676"
GRAY_LIGHT = "#D9D9D9"
RED = "#B64342"
BLACK = "#272727"

FE_TARGET = -1.5
FE_TOLERANCE = 0.06
BG_TARGET = 2.0
BG_TOLERANCE = 0.45

FE_DIR = Path("output/singlerun/2026-06-28/16-46-08-mp20_fe")
BG_DIR = Path("output/singlerun/2026-06-30/11-28-44-mp20_bg")
JOINT_DIR = Path("output/singlerun/2026-08-07/07-54-15-mp20_fe_bg")


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


def read_values(path: Path, column: str) -> np.ndarray:
    values = pd.read_csv(path, usecols=[column])[column].to_numpy(dtype=float)
    return values[np.isfinite(values)]


def density_curve(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if np.unique(values).size < 3:
        raise ValueError("Density estimation requires at least three distinct values")
    return gaussian_kde(values)(grid)


def target_metrics(
    values: np.ndarray,
    target: float,
    tolerance: float,
) -> dict[str, float]:
    errors = np.abs(values - target)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "mean_absolute_target_error": float(errors.mean()),
        "median_absolute_target_error": float(np.median(errors)),
        "hit_rate": float(np.mean(errors <= tolerance)),
    }


def plot_marginal_density(
    ax,
    training: np.ndarray,
    unconditional: np.ndarray,
    template: np.ndarray,
    ab_initio: np.ndarray,
    target: float,
    tolerance: float,
    xlim: tuple[float, float],
    xlabel: str,
) -> None:
    grid = np.linspace(*xlim, 500)
    ax.fill_between(
        grid,
        density_curve(training, grid),
        color=GRAY_LIGHT,
        alpha=0.65,
        lw=0,
        label=f"MP20 train (n={training.size:,})",
    )
    ax.plot(
        grid,
        density_curve(unconditional, grid),
        color=GRAY,
        lw=1.25,
        ls="--",
        label=f"Unconditioned (n={unconditional.size:,})",
    )
    ax.plot(
        grid,
        density_curve(template, grid),
        color=BLUE,
        lw=1.6,
        label=f"Template conditioned (n={template.size:,})",
    )
    ax.plot(
        grid,
        density_curve(ab_initio, grid),
        color=TEAL,
        lw=1.45,
        label=f"Ab initio conditioned (n={ab_initio.size:,})",
    )
    ax.axvspan(target - tolerance, target + tolerance, color=RED, alpha=0.10, lw=0)
    ax.axvline(target, color=RED, lw=1.0, ls=":", label="Target")
    ax.set_xlim(*xlim)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")

    before = target_metrics(unconditional, target, tolerance)
    after = target_metrics(template, target, tolerance)
    ax.text(
        0.98,
        0.95,
        "Template conditioning\n"
        f"median |delta|: {before['median_absolute_target_error']:.3f} -> "
        f"{after['median_absolute_target_error']:.3f}\n"
        f"hit: {100 * before['hit_rate']:.1f}% -> "
        f"{100 * after['hit_rate']:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color=BLACK,
    )


def highest_density_levels(
    histogram: np.ndarray,
    masses: tuple[float, ...],
) -> list[float]:
    ordered = np.sort(histogram.ravel())[::-1]
    cumulative = np.cumsum(ordered)
    if cumulative[-1] <= 0:
        raise ValueError("Cannot contour an empty density")
    cumulative /= cumulative[-1]
    levels = [
        ordered[min(np.searchsorted(cumulative, mass), ordered.size - 1)]
        for mass in masses
    ]
    return sorted(set(float(level) for level in levels))


def joint_density(
    fe: np.ndarray,
    bg: np.ndarray,
    fe_range: tuple[float, float],
    bg_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if fe.shape != bg.shape:
        raise ValueError("FE and BG arrays must be aligned")
    histogram, fe_edges, bg_edges = np.histogram2d(
        fe,
        bg,
        bins=(80, 80),
        range=(fe_range, bg_range),
    )
    histogram = gaussian_filter(histogram, sigma=1.8)
    fe_centers = (fe_edges[:-1] + fe_edges[1:]) / 2
    bg_centers = (bg_edges[:-1] + bg_edges[1:]) / 2
    return fe_centers, bg_centers, histogram.T


def plot_joint_contours(
    ax,
    datasets: list[tuple[str, np.ndarray, np.ndarray, str, str]],
    fe_range: tuple[float, float],
    bg_range: tuple[float, float],
) -> None:
    legend_handles = []
    for label, fe, bg, color, linestyle in datasets:
        x, y, density = joint_density(fe, bg, fe_range, bg_range)
        levels = highest_density_levels(density, (0.90, 0.50))
        ax.scatter(
            fe,
            bg,
            s=2.0,
            color=color,
            alpha=0.075,
            edgecolors="none",
            rasterized=True,
            zorder=1,
        )
        ax.contour(
            x,
            y,
            density,
            levels=levels,
            colors=[color],
            linewidths=[0.9, 1.5],
            linestyles=linestyle,
            zorder=2,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=1.5,
                ls=linestyle,
                marker="o",
                markersize=2.8,
                markerfacecolor=color,
                markeredgewidth=0,
                label=f"{label} (n={fe.size:,})",
            )
        )
    ax.add_patch(
        Rectangle(
            (FE_TARGET - FE_TOLERANCE, BG_TARGET - BG_TOLERANCE),
            2 * FE_TOLERANCE,
            2 * BG_TOLERANCE,
            facecolor=RED,
            edgecolor=RED,
            alpha=0.12,
            lw=0.8,
            zorder=3,
        )
    )
    ax.plot(
        FE_TARGET,
        BG_TARGET,
        marker="x",
        ms=6,
        mew=1.2,
        color=RED,
        label="Target",
        zorder=4,
    )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=RED,
            marker="x",
            lw=0,
            markersize=6,
            label="Target",
        )
    )
    ax.set_xlim(*fe_range)
    ax.set_ylim(*bg_range)
    ax.set_xlabel("Formation energy (eV atom⁻¹)")
    ax.set_ylabel("Band gap (eV)")
    ax.legend(handles=legend_handles, loc="upper left", ncol=2)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_joint(path: Path) -> pd.DataFrame:
    columns = ["predicted_formation_energy_per_atom", "predicted_band_gap"]
    return pd.read_csv(path, usecols=columns).dropna()


def plot_conditioning_distributions(
    project_root: Path,
    output_path: Path,
    dpi: int,
) -> None:
    train = pd.read_csv(
        project_root / "data/mp_20/train.csv",
        usecols=["formation_energy_per_atom", "band_gap"],
    )
    training_fe = train["formation_energy_per_atom"].dropna().to_numpy(dtype=float)
    training_bg = train["band_gap"].dropna().to_numpy(dtype=float)

    fe_uncond = read_values(
        project_root / FE_DIR
        / "eval_properties_gen_template_uncond_n4096_seed42_predictor_seed42.csv",
        "predicted_formation_energy_per_atom",
    )
    fe_template = read_values(
        project_root / FE_DIR
        / "eval_properties_gen_template_fe_m1p5_n4096_seed42_predictor_seed42.csv",
        "predicted_formation_energy_per_atom",
    )
    fe_ab_initio = read_values(
        project_root / FE_DIR
        / "eval_properties_gen_abinitio_empirical_fe_m1p5_n4096_seed42_predictor_seed42.csv",
        "predicted_formation_energy_per_atom",
    )
    bg_uncond = read_values(
        project_root / BG_DIR
        / "eval_properties_gen_template_uncond_n4096_seed42_predictor_seed42.csv",
        "predicted_band_gap",
    )
    bg_template = read_values(
        project_root / BG_DIR
        / "eval_properties_gen_template_bg_2_n4096_seed42_predictor_seed42.csv",
        "predicted_band_gap",
    )
    bg_ab_initio = read_values(
        project_root / BG_DIR
        / "eval_properties_gen_abinitio_empirical_bg_2_n4096_seed42_predictor_seed42.csv",
        "predicted_band_gap",
    )

    joint_uncond = load_joint(
        project_root / JOINT_DIR
        / "eval_properties_gen_template_uncond_n4096_seed42_predictor_seed42.csv"
    )
    joint_template = load_joint(
        project_root / JOINT_DIR
        / "eval_properties_gen_template_fe_m1p5_bg_2_n4096_seed42_predictor_seed42.csv"
    )
    joint_ab_initio = load_joint(
        project_root / JOINT_DIR
        / "eval_properties_gen_abinitio_empirical_fe_m1p5_bg_2_n4096_seed42_predictor_seed42.csv"
    )

    fig = plt.figure(figsize=(7.2, 5.8), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[0.12, 1.0, 1.18])
    ax_legend = fig.add_subplot(grid[0, :])
    ax_fe = fig.add_subplot(grid[1, 0])
    ax_bg = fig.add_subplot(grid[1, 1])
    ax_joint = fig.add_subplot(grid[2, :])

    plot_marginal_density(
        ax_fe,
        training_fe,
        fe_uncond,
        fe_template,
        fe_ab_initio,
        FE_TARGET,
        FE_TOLERANCE,
        (-4.5, 0.25),
        "Formation energy (eV atom⁻¹)",
    )
    plot_marginal_density(
        ax_bg,
        training_bg,
        bg_uncond,
        bg_template,
        bg_ab_initio,
        BG_TARGET,
        BG_TOLERANCE,
        (-0.3, 8.0),
        "Band gap (eV)",
    )
    handles, labels = ax_fe.get_legend_handles_labels()
    ax_legend.legend(handles, labels, loc="center", ncol=3)
    ax_legend.set_axis_off()

    columns = ["predicted_formation_energy_per_atom", "predicted_band_gap"]
    joint_sets = [
        (
            "Unconditioned",
            joint_uncond[columns[0]].to_numpy(),
            joint_uncond[columns[1]].to_numpy(),
            GRAY,
            "--",
        ),
        (
            "Template conditioned",
            joint_template[columns[0]].to_numpy(),
            joint_template[columns[1]].to_numpy(),
            BLUE,
            "-",
        ),
        (
            "Ab initio conditioned",
            joint_ab_initio[columns[0]].to_numpy(),
            joint_ab_initio[columns[1]].to_numpy(),
            TEAL,
            "-.",
        ),
    ]
    plot_joint_contours(ax_joint, joint_sets, (-4.5, 0.25), (-0.3, 8.0))
    joint_hits = []
    for label, fe, bg, _, _ in joint_sets:
        hit = np.mean(
            (np.abs(fe - FE_TARGET) <= FE_TOLERANCE)
            & (np.abs(bg - BG_TARGET) <= BG_TOLERANCE)
        )
        joint_hits.append(f"{label}: {100 * hit:.2f}%")
    ax_joint.text(
        0.99,
        0.97,
        "Joint target hit\n" + "\n".join(joint_hits),
        transform=ax_joint.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
    )

    for label, ax in zip("abc", (ax_fe, ax_bg, ax_joint)):
        add_panel_label(ax, label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    rows = []
    specs = [
        (
            "formation_energy_per_atom",
            FE_TARGET,
            FE_TOLERANCE,
            [
                ("unconditioned", fe_uncond),
                ("template_conditioned", fe_template),
                ("ab_initio_conditioned", fe_ab_initio),
            ],
        ),
        (
            "band_gap",
            BG_TARGET,
            BG_TOLERANCE,
            [
                ("unconditioned", bg_uncond),
                ("template_conditioned", bg_template),
                ("ab_initio_conditioned", bg_ab_initio),
            ],
        ),
    ]
    for prop, target, tolerance, datasets in specs:
        for group, values in datasets:
            rows.append(
                {
                    "property": prop,
                    "group": group,
                    "target": target,
                    "tolerance": tolerance,
                    **target_metrics(values, target, tolerance),
                }
            )
    write_rows(
        output_path.parent / "source_data/conditioning_distribution_metrics.csv",
        rows,
    )


def structure_from_array(crystal_array: dict[str, np.ndarray]) -> Structure | None:
    try:
        return Structure(
            lattice=Lattice.from_parameters(
                *(
                    crystal_array["lengths"].tolist()
                    + crystal_array["angles"].tolist()
                )
            ),
            species=crystal_array["atom_types"],
            coords=crystal_array["frac_coords"],
            coords_are_cartesian=False,
        )
    except Exception:
        return None


def array_is_valid(crystal_array: dict[str, np.ndarray]) -> bool:
    structure = structure_from_array(crystal_array)
    if structure is None or not structure_validity(structure):
        return False
    counts = Counter(int(value) for value in crystal_array["atom_types"])
    divisor = np.gcd.reduce(np.asarray(list(counts.values()), dtype=int))
    elems = tuple(sorted(counts))
    composition = tuple(counts[elem] // divisor for elem in elems)
    return bool(smact_validity(elems, composition))


def load_safe_generation_payload(path: Path) -> dict:
    torch.serialization.add_safe_globals([argparse.Namespace])
    return torch.load(path, map_location="cpu", weights_only=True)


def select_structure(
    project_root: Path,
    pt_path: Path,
    csv_path: Path,
    score,
) -> tuple[object, dict]:
    frame = pd.read_csv(project_root / csv_path)
    finite = frame[
        np.isfinite(frame["predicted_formation_energy_per_atom"])
        & np.isfinite(frame["predicted_band_gap"])
    ].copy()
    finite["selection_score"] = score(finite)
    arrays = _crystal_array_list(load_safe_generation_payload(project_root / pt_path))
    for row in finite.sort_values("selection_score").itertuples(index=False):
        index = int(row.structure_index)
        if array_is_valid(arrays[index]):
            structure = structure_from_array(arrays[index])
            if len(structure.composition.elements) > 4:
                continue
            return structure, {
                "structure_index": index,
                "formation_energy_per_atom": float(
                    row.predicted_formation_energy_per_atom
                ),
                "band_gap": float(row.predicted_band_gap),
                "selection_score": float(row.selection_score),
            }
    raise ValueError(f"No valid representative structure found in {pt_path}")


def plot_conditioned_structures(
    project_root: Path,
    output_path: Path,
    dpi: int,
) -> None:
    specs = [
        (
            "FE - Template",
            FE_DIR / "eval_gen_template_fe_m1p5_n4096_seed42.pt",
            FE_DIR
            / "eval_properties_gen_template_fe_m1p5_n4096_seed42_predictor_seed42.csv",
            lambda frame: np.abs(
                frame["predicted_formation_energy_per_atom"] - FE_TARGET
            ),
        ),
        (
            "BG - Template",
            BG_DIR / "eval_gen_template_bg_2_n4096_seed42.pt",
            BG_DIR
            / "eval_properties_gen_template_bg_2_n4096_seed42_predictor_seed42.csv",
            lambda frame: np.abs(frame["predicted_band_gap"] - BG_TARGET),
        ),
        (
            "FE+BG - Template",
            JOINT_DIR / "eval_gen_template_fe_m1p5_bg_2_n4096_seed42.pt",
            JOINT_DIR
            / "eval_properties_gen_template_fe_m1p5_bg_2_n4096_seed42_predictor_seed42.csv",
            lambda frame: np.sqrt(
                (
                    (frame["predicted_formation_energy_per_atom"] - FE_TARGET)
                    / FE_TOLERANCE
                )
                ** 2
                + ((frame["predicted_band_gap"] - BG_TARGET) / BG_TOLERANCE)
                ** 2
            ),
        ),
        (
            "FE+BG - Ab initio",
            JOINT_DIR / "eval_gen_abinitio_empirical_fe_m1p5_bg_2_n4096_seed42.pt",
            JOINT_DIR
            / "eval_properties_gen_abinitio_empirical_fe_m1p5_bg_2_n4096_seed42_predictor_seed42.csv",
            lambda frame: np.sqrt(
                (
                    (frame["predicted_formation_energy_per_atom"] - FE_TARGET)
                    / FE_TOLERANCE
                )
                ** 2
                + ((frame["predicted_band_gap"] - BG_TARGET) / BG_TOLERANCE)
                ** 2
            ),
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    rows = []
    for panel, ax, (label, pt_path, csv_path, score) in zip(
        "abcd",
        axes.flat,
        specs,
    ):
        structure, metadata = select_structure(
            project_root,
            pt_path,
            csv_path,
            score,
        )
        atoms = AseAtomsAdaptor.get_atoms(structure)
        plot_atoms(
            atoms,
            ax,
            rotation="10x,20y,0z",
            radii=0.36,
            show_unit_cell=2,
        )
        analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
        space_group = (
            f"{analyzer.get_space_group_symbol()} "
            f"({analyzer.get_space_group_number()})"
        )
        formula = structure.composition.reduced_formula
        ax.set_axis_off()
        ax.text(
            0.01,
            0.98,
            panel,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="top",
        )
        rows.append(
            {
                "panel": panel,
                "label": label,
                "source_pt": str(pt_path),
                "structure_index": metadata["structure_index"],
                "formula": formula,
                "space_group": space_group,
                "predicted_formation_energy_per_atom": metadata[
                    "formation_energy_per_atom"
                ],
                "predicted_band_gap": metadata["band_gap"],
                "selection_score": metadata["selection_score"],
                "selection_rule": (
                    "closest target among composition- and geometry-valid "
                    "generated structures containing at most four elements"
                ),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    write_rows(
        output_path.parent / "source_data/conditioned_structure_examples.csv",
        rows,
    )
