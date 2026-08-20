"""Generate publication-grade figures for every CGDiT dataset split.

The descriptive figures use all available observations. They do not impute
missing targets or silently trim outliers. Train/validation/test comparisons
use empirical distributions and effect-size summaries rather than smoothed
curves whose appearance depends on an arbitrary bandwidth.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.stats import ks_2samp
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Mandatory publication settings: keep text editable in vector exports.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})


SPLITS = ("train", "val", "test")
SPLIT_LABELS = {"train": "Train", "val": "Validation", "test": "Test"}
SPLIT_COLORS = {"train": "#0F4D92", "val": "#42949E", "test": "#B64342"}
SPLIT_LINESTYLES = {"train": "-", "val": "--", "test": ":"}
SPLIT_HATCHES = {"train": "", "val": "//", "test": "xx"}
CRYSTAL_SYSTEM_ORDER = (
    "Triclinic",
    "Monoclinic",
    "Orthorhombic",
    "Tetragonal",
    "Trigonal",
    "Hexagonal",
    "Cubic",
    "Unknown",
)


@dataclass(frozen=True)
class TargetSpec:
    column: str
    label: str
    unit: str

    @property
    def axis_label(self) -> str:
        return f"{self.label} ({self.unit})" if self.unit else self.label


@dataclass(frozen=True)
class DatasetSpec:
    display_name: str
    targets: tuple[TargetSpec, ...]
    symmetry_kind: str = "crystal_system"


DATASET_SPECS = {
    "c2db_51": DatasetSpec(
        "C2DB-51",
        (
            TargetSpec("gap", "Band gap", "eV"),
            TargetSpec("energy", "Total energy", "eV cell⁻¹"),
            TargetSpec("ehull", "Energy above hull", "eV atom⁻¹"),
            TargetSpec("hform", "Formation energy", "eV atom⁻¹"),
        ),
        symmetry_kind="layer_group",
    ),
    "carbon_24": DatasetSpec(
        "Carbon-24",
        (TargetSpec("energy_per_atom", "Energy", "eV atom⁻¹"),),
    ),
    "mp_20": DatasetSpec(
        "MP-20",
        (
            TargetSpec("formation_energy_per_atom", "Formation energy", "eV atom⁻¹"),
            TargetSpec("band_gap", "Band gap", "eV"),
            TargetSpec("e_above_hull", "Energy above hull", "eV atom⁻¹"),
        ),
    ),
    "mpts_52": DatasetSpec(
        "MPTS-52",
        (
            TargetSpec("energy_above_hull", "Energy above hull", "eV atom⁻¹"),
            TargetSpec("formation_energy_per_atom", "Formation energy", "eV atom⁻¹"),
        ),
    ),
    "perov_5": DatasetSpec(
        "Perov-5",
        (
            TargetSpec("heat_all", "Decomposition energy (all phases)", "eV atom⁻¹"),
            TargetSpec("heat_ref", "Decomposition energy (reference phases)", "eV atom⁻¹"),
            TargetSpec("dir_gap", "Direct band gap", "eV"),
            TargetSpec("ind_gap", "Indirect band gap", "eV"),
        ),
    ),
    "test_data": DatasetSpec(
        "Test data",
        (
            TargetSpec("formation_energy_per_atom", "Formation energy", "eV atom⁻¹"),
            TargetSpec("band_gap", "Band gap", "eV"),
            TargetSpec("e_above_hull", "Energy above hull", "eV atom⁻¹"),
        ),
    ),
}

GENERAL_CORRELATION_FEATURES = (
    "natoms",
    "n_elements",
    "volume_per_atom",
    "density",
    "lattice_a",
    "lattice_b",
    "lattice_c",
    "alpha",
    "beta",
    "gamma",
)
C2DB_CORRELATION_FEATURES = (
    "natoms",
    "n_elements",
    "area_per_atom",
    "thickness",
    "lattice_a",
    "lattice_b",
    "gamma",
)
FEATURE_LABELS = {
    "natoms": "Atoms per cell",
    "n_elements": "Element count",
    "volume_per_atom": "Volume per atom",
    "area_per_atom": "In-plane area per atom",
    "density": "Density",
    "thickness": "Layer thickness",
    "lattice_a": "Lattice a",
    "lattice_b": "Lattice b",
    "lattice_c": "Lattice c",
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
}


def apply_publication_style() -> None:
    """Apply compact Nature-style defaults before any figure is created."""
    mpl.rcParams.update(
        {
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _finite(values: pd.Series | np.ndarray) -> np.ndarray:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array)]


def empirical_cdf(values: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted finite observations and their empirical cumulative fraction."""
    x = np.sort(_finite(values))
    if len(x) == 0:
        return x, np.array([], dtype=float)
    return x, np.arange(1, len(x) + 1, dtype=float) / len(x)


def crystal_system(spacegroup_number: object) -> str:
    """Map an international space-group number to a crystal system."""
    try:
        sg = int(float(spacegroup_number))
    except (TypeError, ValueError):
        return "Unknown"
    if 1 <= sg <= 2:
        return "Triclinic"
    if 3 <= sg <= 15:
        return "Monoclinic"
    if 16 <= sg <= 74:
        return "Orthorhombic"
    if 75 <= sg <= 142:
        return "Tetragonal"
    if 143 <= sg <= 167:
        return "Trigonal"
    if 168 <= sg <= 194:
        return "Hexagonal"
    if 195 <= sg <= 230:
        return "Cubic"
    return "Unknown"


def discover_datasets(data_root: Path) -> tuple[list[Path], list[Path]]:
    """Return complete split directories and incomplete dataset-like directories."""
    complete: list[Path] = []
    incomplete: list[Path] = []
    for path in sorted(p for p in data_root.iterdir() if p.is_dir()):
        present = [(path / f"{split}.csv").exists() for split in SPLITS]
        if all(present):
            complete.append(path)
        elif any(present) or not any(path.iterdir()):
            incomplete.append(path)
    return complete, incomplete


def _parse_cif(payload: tuple[str, bool]) -> dict[str, object]:
    cif, infer_symmetry = payload
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structure = Structure.from_str(cif, fmt="cif")
            sg_number = (
                SpacegroupAnalyzer(structure, symprec=0.1).get_space_group_number()
                if infer_symmetry
                else np.nan
            )
        lattice = structure.lattice
        natoms = len(structure)
        elements = sorted(str(element) for element in structure.composition.elements)
        cell_area = float(np.linalg.norm(np.cross(lattice.matrix[0], lattice.matrix[1])))
        return {
            "parsed": True,
            "parse_error": "",
            "natoms": natoms,
            "n_elements": len(elements),
            "elements_parsed": ";".join(elements),
            "volume": float(structure.volume),
            "volume_per_atom": float(structure.volume / natoms),
            "area_per_atom": float(cell_area / natoms),
            "density": float(structure.density),
            "lattice_a": float(lattice.a),
            "lattice_b": float(lattice.b),
            "lattice_c": float(lattice.c),
            "alpha": float(lattice.alpha),
            "beta": float(lattice.beta),
            "gamma": float(lattice.gamma),
            "spacegroup_number": sg_number,
        }
    except Exception as exc:  # one failed structure must not erase the full audit
        result: dict[str, object] = {
            "parsed": False,
            "parse_error": type(exc).__name__,
            "elements_parsed": "",
        }
        for name in (
            "natoms",
            "n_elements",
            "volume",
            "volume_per_atom",
            "area_per_atom",
            "density",
            "lattice_a",
            "lattice_b",
            "lattice_c",
            "alpha",
            "beta",
            "gamma",
            "spacegroup_number",
        ):
            result[name] = np.nan
        return result


def _input_fingerprint(dataset_dir: Path) -> dict[str, dict[str, int]]:
    result = {}
    for split in SPLITS:
        path = dataset_dir / f"{split}.csv"
        stat = path.stat()
        result[path.name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def _load_splits(dataset_dir: Path) -> pd.DataFrame:
    frames = []
    for split in SPLITS:
        frame = pd.read_csv(dataset_dir / f"{split}.csv")
        frame.insert(0, "_row", np.arange(len(frame), dtype=int))
        frame.insert(0, "_split", split)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _structural_features(
    source: pd.DataFrame,
    infer_symmetry: bool,
    jobs: int,
) -> pd.DataFrame:
    payloads = [(str(cif), infer_symmetry) for cif in source["cif"]]
    if jobs == 1:
        rows = list(map(_parse_cif, payloads))
    else:
        chunk_size = max(1, len(payloads) // (jobs * 24))
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            rows = list(executor.map(_parse_cif, payloads, chunksize=chunk_size))
    features = pd.DataFrame(rows)
    features.insert(0, "_row", source["_row"].to_numpy())
    features.insert(0, "_split", source["_split"].to_numpy())
    return features


def load_or_extract_features(
    dataset_dir: Path,
    source: pd.DataFrame,
    output_dir: Path,
    jobs: int,
    refresh: bool,
) -> tuple[pd.DataFrame, bool]:
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    cache_path = source_dir / "structural_features.csv.gz"
    metadata_path = source_dir / "structural_features.meta.json"
    fingerprint = _input_fingerprint(dataset_dir)

    if not refresh and cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("input_fingerprint") == fingerprint:
            features = pd.read_csv(cache_path)
            if len(features) == len(source):
                return features, True

    infer_symmetry = dataset_dir.name not in {"c2db_51", "mp_20", "test_data"}
    features = _structural_features(source, infer_symmetry=infer_symmetry, jobs=jobs)
    features.to_csv(cache_path, index=False, compression="gzip")
    metadata_path.write_text(
        json.dumps(
            {
                "input_fingerprint": fingerprint,
                "rows": len(features),
                "symmetry_inferred_from_structure": infer_symmetry,
                "symmetry_tolerance_angstrom": 0.1 if infer_symmetry else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return features, False


def _combine_source_and_features(source: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    derived_columns = [column for column in features if column not in {"_split", "_row"}]
    clean_source = source.drop(columns=[c for c in derived_columns if c in source], errors="ignore")
    clean_features = features.drop(columns=["_split", "_row"])
    frame = pd.concat(
        [clean_source.reset_index(drop=True), clean_features.reset_index(drop=True)],
        axis=1,
    )
    if "spacegroup.number" in source:
        supplied = pd.to_numeric(source["spacegroup.number"], errors="coerce").reset_index(drop=True)
        frame["spacegroup_number"] = supplied.fillna(frame["spacegroup_number"])
    if "thickness" in source:
        frame["thickness"] = pd.to_numeric(source["thickness"], errors="coerce").to_numpy()
    return frame


def _dataset_spec(dataset_name: str, frame: pd.DataFrame) -> DatasetSpec:
    if dataset_name in DATASET_SPECS:
        return DATASET_SPECS[dataset_name]
    excluded = {
        "_row",
        "index",
        "Unnamed: 0",
        "material_id",
        "spacegroup.number",
        "year",
        "lgnum",
    }
    targets = []
    for column in frame.columns:
        if column in excluded or column.startswith("_"):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]) and frame[column].nunique(dropna=True) > 1:
            targets.append(TargetSpec(column, column.replace("_", " ").title(), ""))
    return DatasetSpec(dataset_name, tuple(targets))


def _style_axis(ax: plt.Axes) -> None:
    ax.tick_params(direction="out", length=3, pad=2)
    ax.spines["left"].set_color("#4D4D4D")
    ax.spines["bottom"].set_color("#4D4D4D")


def save_figure(
    fig: plt.Figure,
    base_path: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Save the requested figure derivatives."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.8)
    export_paths = {
        "svg": base_path.with_suffix(".svg"),
        "pdf": base_path.with_suffix(".pdf"),
        "tiff": base_path.with_suffix(".tiff"),
        "png": base_path.with_suffix(".png"),
    }
    saved = []
    common = {
        "bbox_inches": "tight",
        "pad_inches": 0.04,
    }
    for fmt in ("svg", "pdf", "tiff", "png"):
        if fmt not in formats:
            continue
        output = export_paths[fmt]
        if fmt == "tiff":
            fig.savefig(output, dpi=600, pil_kwargs={"compression": "tiff_lzw"}, **common)
        elif fmt == "png":
            fig.savefig(output, dpi=300, **common)
        else:
            fig.savefig(output, **common)
        saved.append(output)
    plt.close(fig)
    return saved


def _plot_ecdf(
    ax: plt.Axes,
    frame: pd.DataFrame,
    column: str,
    include_missing: bool = False,
) -> None:
    for split in SPLITS:
        subset = frame.loc[frame["_split"] == split, column]
        x, y = empirical_cdf(subset)
        if len(x) == 0:
            continue
        missing = int(pd.to_numeric(subset, errors="coerce").isna().sum())
        label = f"{SPLIT_LABELS[split]} (n={len(x):,}"
        if include_missing and missing:
            label += f", missing={missing:,}"
        label += ")"
        ax.step(
            x,
            y,
            where="post",
            color=SPLIT_COLORS[split],
            linestyle=SPLIT_LINESTYLES[split],
            label=label,
        )
    ax.set_ylabel("Cumulative fraction")
    ax.set_ylim(0, 1.02)
    _style_axis(ax)


def _ks_against_train(frame: pd.DataFrame, column: str) -> dict[str, float | None]:
    train = _finite(frame.loc[frame["_split"] == "train", column])
    result: dict[str, float | None] = {"train": None}
    for split in ("val", "test"):
        values = _finite(frame.loc[frame["_split"] == split, column])
        result[split] = (
            float(ks_2samp(train, values, method="asymp").statistic)
            if len(train) and len(values)
            else None
        )
    return result


def plot_target_distributions(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    output_dir: Path,
    formats: Sequence[str],
) -> list[Path]:
    saved = []
    for target in spec.targets:
        if target.column not in frame:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        _plot_ecdf(ax, frame, target.column, include_missing=True)
        ax.set_xlabel(target.axis_label)
        ax.set_title(f"{spec.display_name}: {target.label} across fixed data splits")
        ks = _ks_against_train(frame, target.column)
        details = []
        for split in ("val", "test"):
            value = ks[split]
            details.append(f"Dₖₛ(train, {split}) = {value:.3f}" if value is not None else f"Dₖₛ(train, {split}) = NA")
        ax.text(
            0.98,
            0.04,
            "\n".join(details),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color="#4D4D4D",
            fontsize=7,
        )
        ax.legend(loc="best")
        saved.extend(
            save_figure(
                fig,
                output_dir / f"01_target_{target.column}_distribution",
                formats,
            )
        )
    return saved


def _category_percentages(
    frame: pd.DataFrame,
    column: str,
    categories: Sequence[object],
) -> dict[str, np.ndarray]:
    result = {}
    for split in SPLITS:
        values = frame.loc[frame["_split"] == split, column]
        counts = values.value_counts(normalize=True, dropna=False)
        result[split] = np.asarray([100 * counts.get(category, 0.0) for category in categories])
    return result


def _grouped_percentage_bars(
    ax: plt.Axes,
    categories: Sequence[object],
    percentages: dict[str, np.ndarray],
) -> None:
    x = np.arange(len(categories), dtype=float)
    width = 0.24
    for index, split in enumerate(SPLITS):
        ax.bar(
            x + (index - 1) * width,
            percentages[split],
            width=width,
            color=SPLIT_COLORS[split],
            edgecolor="#272727",
            linewidth=0.45,
            hatch=SPLIT_HATCHES[split],
            label=SPLIT_LABELS[split],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Structures (%)")
    ax.legend(ncols=3, loc="upper right")
    _style_axis(ax)


def plot_compositional_diversity(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    output_dir: Path,
    formats: Sequence[str],
) -> list[Path]:
    categories = sorted(int(value) for value in frame["n_elements"].dropna().unique())
    percentages = _category_percentages(frame, "n_elements", categories)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    _grouped_percentage_bars(ax, categories, percentages)
    ax.set_xlabel("Unique elements per structure")

    train = percentages["train"] / 100
    distances = []
    for split in ("val", "test"):
        distance = 0.5 * np.abs(train - percentages[split] / 100).sum()
        distances.append(f"TV(train, {split}) = {distance:.3f}")
    ax.set_title(
        f"{spec.display_name}: compositional complexity\n"
        + ", ".join(distances)
    )
    return save_figure(fig, output_dir / "02_compositional_diversity", formats)


def _symmetry_categories(frame: pd.DataFrame, spec: DatasetSpec) -> tuple[pd.Series, list[str], str]:
    if spec.symmetry_kind == "layer_group" and "layergroup" in frame:
        values = frame["layergroup"].fillna("Unknown").astype(str)
        top = values.value_counts().head(12).index.tolist()
        collapsed = values.where(values.isin(top), "Other")
        categories = top + (["Other"] if (collapsed == "Other").any() else [])
        return collapsed, categories, "Layer group"

    values = frame["spacegroup_number"].map(crystal_system)
    categories = [name for name in CRYSTAL_SYSTEM_ORDER if (values == name).any()]
    return values, categories, "Crystal system"


def plot_symmetry_diversity(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    output_dir: Path,
    formats: Sequence[str],
) -> tuple[list[Path], pd.Series]:
    values, categories, axis_label = _symmetry_categories(frame, spec)
    plotting = frame.assign(_symmetry=values)
    percentages = _category_percentages(plotting, "_symmetry", categories)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _grouped_percentage_bars(ax, categories, percentages)
    ax.set_xlabel(axis_label)
    ax.set_title(f"{spec.display_name}: symmetry coverage")
    ax.tick_params(axis="x", labelrotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    return save_figure(fig, output_dir / "03_crystal_system_diversity", formats), values


def plot_physical_properties(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    output_dir: Path,
    formats: Sequence[str],
) -> list[Path]:
    if spec.symmetry_kind == "layer_group":
        panels = (
            ("area_per_atom", "In-plane cell area (Å² atom⁻¹)"),
            ("thickness", "Layer thickness (Å)"),
        )
    else:
        panels = (
            ("volume_per_atom", "Cell volume (Å³ atom⁻¹)"),
            ("density", "Density (g cm⁻³)"),
        )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharey=True)
    for ax, (column, label) in zip(axes, panels):
        _plot_ecdf(ax, frame, column)
        ax.set_xlabel(label)
        ax.set_title(label.split(" (")[0])
    axes[0].legend(loc="best")
    axes[1].legend().remove()
    fig.suptitle(f"{spec.display_name}: physical scale", fontsize=8.5, fontweight="semibold")
    return save_figure(fig, output_dir / "04_physical_properties_distribution", formats)


def plot_lattice_parameters(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    output_dir: Path,
    formats: Sequence[str],
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    colors = {"lattice_a": "#0F4D92", "lattice_b": "#42949E", "lattice_c": "#9A4D8E"}
    labels = {"lattice_a": "a", "lattice_b": "b", "lattice_c": "c"}
    for column in ("lattice_a", "lattice_b", "lattice_c"):
        if spec.symmetry_kind == "layer_group" and column == "lattice_c":
            continue
        x, y = empirical_cdf(frame[column])
        ax.step(x, y, where="post", color=colors[column], label=f"Lattice {labels[column]} (n={len(x):,})")
    ax.set_xlabel("Lattice length (Å)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{spec.display_name}: lattice-length distribution")
    ax.legend(loc="best")
    _style_axis(ax)
    return save_figure(fig, output_dir / "05_lattice_parameters", formats)


def _element_frequency_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        subset = frame.loc[frame["_split"] == split, "elements_parsed"].fillna("")
        counter: Counter[str] = Counter()
        for text in subset:
            counter.update(set(filter(None, str(text).split(";"))))
        for element, count in counter.items():
            rows.append(
                {
                    "split": split,
                    "element": element,
                    "structures": count,
                    "structures_percent": 100 * count / len(subset) if len(subset) else np.nan,
                }
            )
    overall = Counter()
    for text in frame["elements_parsed"].fillna(""):
        overall.update(set(filter(None, str(text).split(";"))))
    for element, count in overall.items():
        rows.append(
            {
                "split": "all",
                "element": element,
                "structures": count,
                "structures_percent": 100 * count / len(frame) if len(frame) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_element_frequency(
    element_table: pd.DataFrame,
    spec: DatasetSpec,
    output_dir: Path,
    formats: Sequence[str],
) -> list[Path]:
    overall = (
        element_table.loc[element_table["split"] == "all"]
        .nlargest(20, "structures")
        .sort_values("structures_percent")
    )
    if overall.empty:
        return []
    height = max(2.5, 0.22 * len(overall) + 1.3)
    fig, ax = plt.subplots(figsize=(7.2, height))
    bars = ax.barh(
        overall["element"],
        overall["structures_percent"],
        color="#3775BA",
        edgecolor="#272727",
        linewidth=0.45,
    )
    for bar, value in zip(bars, overall["structures_percent"]):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.1f}%", va="center", fontsize=6.5)
    ax.set_xlabel("Structures containing element (%)")
    ax.set_ylabel("Element")
    ax.set_title(f"{spec.display_name}: most prevalent elements (all splits)")
    ax.set_xlim(0, max(overall["structures_percent"].max() * 1.16, 1))
    _style_axis(ax)
    return save_figure(fig, output_dir / "06_top_elements_frequency", formats)


def correlation_table(frame: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    features = C2DB_CORRELATION_FEATURES if spec.symmetry_kind == "layer_group" else GENERAL_CORRELATION_FEATURES
    features = tuple(name for name in features if name in frame)
    targets = tuple(target.column for target in spec.targets if target.column in frame)
    numeric = frame[list(features) + list(targets)].apply(pd.to_numeric, errors="coerce")
    return numeric.corr(method="spearman").loc[list(features), list(targets)]


def plot_feature_target_correlation(
    correlations: pd.DataFrame,
    spec: DatasetSpec,
    output_dir: Path,
    formats: Sequence[str],
) -> list[Path]:
    if correlations.empty:
        return []
    width = max(4.2, 1.2 * len(correlations.columns) + 2.0)
    height = max(3.7, 0.36 * len(correlations.index) + 1.5)
    fig, ax = plt.subplots(figsize=(width, height))
    masked = np.ma.masked_invalid(correlations.to_numpy(dtype=float))
    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad("white")
    image = ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    for row in range(masked.shape[0]):
        for column in range(masked.shape[1]):
            value = correlations.iat[row, column]
            if np.isfinite(value):
                color = "white" if abs(value) >= 0.58 else "#272727"
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=6.5)
    target_labels = {target.column: target.label for target in spec.targets}
    ax.set_xticks(range(len(correlations.columns)))
    ax.set_xticklabels(
        [target_labels.get(name, name) for name in correlations.columns],
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(range(len(correlations.index)))
    ax.set_yticklabels([FEATURE_LABELS.get(name, name) for name in correlations.index])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Spearman ρ")
    ax.set_title(f"{spec.display_name}: structure–property associations")
    return save_figure(fig, output_dir / "07_feature_target_correlation", formats)


def _target_summary(frame: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    rows = []
    for target in spec.targets:
        if target.column not in frame:
            continue
        ks = _ks_against_train(frame, target.column)
        for split in SPLITS:
            raw = frame.loc[frame["_split"] == split, target.column]
            values = _finite(raw)
            rows.append(
                {
                    "target": target.column,
                    "label": target.label,
                    "unit": target.unit,
                    "split": split,
                    "n_total": len(raw),
                    "n_valid": len(values),
                    "n_missing": len(raw) - len(values),
                    "mean": np.mean(values) if len(values) else np.nan,
                    "std": np.std(values, ddof=1) if len(values) > 1 else np.nan,
                    "minimum": np.min(values) if len(values) else np.nan,
                    "q25": np.quantile(values, 0.25) if len(values) else np.nan,
                    "median": np.median(values) if len(values) else np.nan,
                    "q75": np.quantile(values, 0.75) if len(values) else np.nan,
                    "maximum": np.max(values) if len(values) else np.nan,
                    "ks_vs_train": ks[split],
                }
            )
    return pd.DataFrame(rows)


def _split_overlap(frame: pd.DataFrame) -> dict[str, object]:
    id_candidates = ("unique_id", "material_id", "uid", "index", "Unnamed: 0")
    id_column = next((column for column in id_candidates if column in frame), None)
    if id_column is None:
        return {"id_column": None, "pairwise_overlap": {}}
    ids = {
        split: set(frame.loc[frame["_split"] == split, id_column].dropna().astype(str))
        for split in SPLITS
    }
    return {
        "id_column": id_column,
        "pairwise_overlap": {
            "train_val": len(ids["train"] & ids["val"]),
            "train_test": len(ids["train"] & ids["test"]),
            "val_test": len(ids["val"] & ids["test"]),
        },
    }


def write_source_data(
    frame: pd.DataFrame,
    symmetry_values: pd.Series,
    target_summary: pd.DataFrame,
    element_table: pd.DataFrame,
    correlations: pd.DataFrame,
    spec: DatasetSpec,
    dataset_dir: Path,
    output_dir: Path,
    cache_reused: bool,
) -> None:
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)

    split_summary = (
        frame.groupby("_split", sort=False)
        .agg(rows=("_row", "size"), cif_parsed=("parsed", "sum"))
        .reindex(SPLITS)
        .reset_index(names="split")
    )
    split_summary["cif_parse_failures"] = split_summary["rows"] - split_summary["cif_parsed"]
    split_summary.to_csv(source_dir / "split_summary.csv", index=False)
    target_summary.to_csv(source_dir / "target_summary.csv", index=False)

    composition = (
        frame.groupby(["_split", "n_elements"], dropna=False)
        .size()
        .rename("structures")
        .reset_index()
        .rename(columns={"_split": "split"})
    )
    totals = composition.groupby("split")["structures"].transform("sum")
    composition["structures_percent"] = 100 * composition["structures"] / totals
    composition.to_csv(source_dir / "composition_summary.csv", index=False)

    symmetry = pd.DataFrame({"split": frame["_split"], "category": symmetry_values})
    symmetry = symmetry.groupby(["split", "category"], dropna=False).size().rename("structures").reset_index()
    symmetry_totals = symmetry.groupby("split")["structures"].transform("sum")
    symmetry["structures_percent"] = 100 * symmetry["structures"] / symmetry_totals
    symmetry.to_csv(source_dir / "symmetry_summary.csv", index=False)
    element_table.to_csv(source_dir / "element_frequency.csv", index=False)
    correlations.to_csv(source_dir / "spearman_structure_target.csv", index_label="feature")

    parse_errors = frame.loc[~frame["parsed"], "parse_error"].value_counts().to_dict()
    png_files = sorted((dataset_dir / "analysis").glob("**/*.png"))
    audit = {
        "dataset": dataset_dir.name,
        "display_name": spec.display_name,
        "input_fingerprint": _input_fingerprint(dataset_dir),
        "rows": int(len(frame)),
        "split_rows": {split: int((frame["_split"] == split).sum()) for split in SPLITS},
        "cif_parsed": int(frame["parsed"].sum()),
        "cif_parse_failures": int((~frame["parsed"]).sum()),
        "parse_error_types": {str(key): int(value) for key, value in parse_errors.items()},
        "feature_cache_reused": cache_reused,
        "missing_values_not_imputed": True,
        "display_clipping": None,
        "split_overlap": _split_overlap(frame),
        "generated_files": [str(path.relative_to(dataset_dir)) for path in png_files],
    }
    (source_dir / "figure_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def plot_model_parity(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    analysis_dir: Path,
    formats: Sequence[str],
    jobs: int,
) -> list[Path]:
    """Rebuild model-parity panels against the untouched fixed test split."""
    features = C2DB_CORRELATION_FEATURES if spec.symmetry_kind == "layer_group" else GENERAL_CORRELATION_FEATURES
    features = tuple(name for name in features if name in frame)
    saved = []
    for target in spec.targets:
        if target.column not in frame:
            continue
        needed = list(features) + [target.column]
        train_candidates = frame.loc[frame["_split"].isin(("train", "val")), needed].apply(
            pd.to_numeric, errors="coerce"
        )
        test_candidates = frame.loc[frame["_split"] == "test", needed].apply(pd.to_numeric, errors="coerce")
        n_before = {"train_validation": len(train_candidates), "test": len(test_candidates)}
        train = train_candidates.dropna()
        test = test_candidates.dropna()
        n_after = {"train_validation": len(train), "test": len(test)}
        excluded_count = {name: n_before[name] - n_after[name] for name in n_before}
        if len(train) < 5 or len(test) < 3:
            continue
        x_train, y_train = train[list(features)], train[target.column]
        x_test, y_test = test[list(features)], test[target.column]
        models = {
            "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            "Random forest": RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=2,
                n_jobs=jobs,
                random_state=42,
            ),
            "Extra trees": ExtraTreesRegressor(
                n_estimators=200,
                min_samples_leaf=2,
                n_jobs=jobs,
                random_state=42,
            ),
            "Histogram gradient boosting": HistGradientBoostingRegressor(
                max_iter=200,
                l2_regularization=0.1,
                random_state=42,
            ),
        }
        predictions = {}
        metrics = []
        for name, model in models.items():
            model.fit(x_train, y_train)
            prediction = model.predict(x_test)
            predictions[name] = prediction
            metrics.append(
                {
                    "model": name,
                    "target": target.column,
                    "n_train_validation": len(train),
                    "n_test": len(test),
                    "excluded_train_validation": excluded_count["train_validation"],
                    "excluded_test": excluded_count["test"],
                    "r2": r2_score(y_test, prediction),
                    "mae": mean_absolute_error(y_test, prediction),
                    "rmse": mean_squared_error(y_test, prediction) ** 0.5,
                }
            )

        model_dir = analysis_dir / f"ml_figs_{target.column}"
        model_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(metrics).to_csv(model_dir / "metrics.csv", index=False)
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0), sharex=True, sharey=True)
        observed = y_test.to_numpy(dtype=float)
        finite_predictions = np.concatenate([np.asarray(value) for value in predictions.values()])
        lower = float(min(observed.min(), finite_predictions.min()))
        upper = float(max(observed.max(), finite_predictions.max()))
        if lower == upper:
            lower, upper = lower - 0.5, upper + 0.5
        pad = 0.03 * (upper - lower)
        lower, upper = lower - pad, upper + pad
        for ax, metric in zip(axes.flat, metrics):
            prediction = predictions[metric["model"]]
            ax.scatter(
                observed,
                prediction,
                s=7,
                color="#3775BA",
                edgecolors="none",
                alpha=0.25,
                rasterized=True,
            )
            ax.plot([lower, upper], [lower, upper], color="#272727", linestyle="--", linewidth=1)
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
            ax.set_title(metric["model"])
            ax.text(
                0.04,
                0.96,
                f"R² = {metric['r2']:.3f}\nMAE = {metric['mae']:.3g}\nRMSE = {metric['rmse']:.3g}\nn = {metric['n_test']:,}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
            )
            _style_axis(ax)
        for ax in axes[:, 0]:
            ax.set_ylabel(f"Predicted {target.axis_label}")
        for ax in axes[-1, :]:
            ax.set_xlabel(f"Observed {target.axis_label}")
        fig.suptitle(
            f"{spec.display_name}: fixed-test parity for {target.label}",
            fontsize=8.5,
            fontweight="semibold",
        )
        saved.extend(save_figure(fig, model_dir / "parity_plots", formats))
    return saved


def process_dataset(
    dataset_dir: Path,
    formats: Sequence[str],
    jobs: int,
    refresh_features: bool,
    include_model_parity: bool,
) -> list[Path]:
    analysis_dir = dataset_dir / "analysis"
    output_dir = analysis_dir / "diversity_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    source = _load_splits(dataset_dir)
    features, cache_reused = load_or_extract_features(
        dataset_dir,
        source,
        output_dir,
        jobs=jobs,
        refresh=refresh_features,
    )
    frame = _combine_source_and_features(source, features)
    spec = _dataset_spec(dataset_dir.name, frame)

    generated: list[Path] = []
    generated.extend(plot_target_distributions(frame, spec, output_dir, formats))
    generated.extend(plot_compositional_diversity(frame, spec, output_dir, formats))
    symmetry_files, symmetry_values = plot_symmetry_diversity(frame, spec, output_dir, formats)
    generated.extend(symmetry_files)
    generated.extend(plot_physical_properties(frame, spec, output_dir, formats))
    generated.extend(plot_lattice_parameters(frame, spec, output_dir, formats))
    element_table = _element_frequency_table(frame)
    generated.extend(plot_element_frequency(element_table, spec, output_dir, formats))
    correlations = correlation_table(frame, spec)
    generated.extend(plot_feature_target_correlation(correlations, spec, output_dir, formats))
    if include_model_parity:
        generated.extend(plot_model_parity(frame, spec, analysis_dir, formats, jobs=jobs))

    target_summary = _target_summary(frame, spec)
    write_source_data(
        frame,
        symmetry_values,
        target_summary,
        element_table,
        correlations,
        spec,
        dataset_dir,
        output_dir,
        cache_reused,
    )
    return generated


def build_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=repository_root / "data")
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset directory name. Repeat to select several; omit to process all complete datasets.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Processes for CIF parsing and workers for tree models (default: min(8, CPU count)).",
    )
    parser.add_argument("--refresh-features", action="store_true", help="Ignore valid structural-feature caches.")
    parser.add_argument(
        "--include-model-parity",
        action="store_true",
        help="Also rebuild model-parity panels using train+validation and the fixed test split.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    data_root = args.data_root.resolve()
    complete, incomplete = discover_datasets(data_root)
    requested = set(args.dataset or [])
    if requested:
        available = {path.name: path for path in complete}
        missing = sorted(requested - available.keys())
        if missing:
            raise SystemExit(f"Datasets are missing complete train/val/test CSVs: {', '.join(missing)}")
        complete = [available[name] for name in sorted(requested)]
    if not complete:
        raise SystemExit(f"No complete train/val/test datasets found below {data_root}")

    apply_publication_style()
    if incomplete:
        print("Skipped incomplete dataset directories: " + ", ".join(path.name for path in incomplete))
    total = 0
    for dataset_dir in complete:
        print(f"[{dataset_dir.name}] extracting features and rendering figures")
        generated = process_dataset(
            dataset_dir,
            formats=("png",),
            jobs=args.jobs,
            refresh_features=args.refresh_features,
            include_model_parity=args.include_model_parity,
        )
        total += len(generated)
        print(f"[{dataset_dir.name}] wrote {len(generated)} figure files")
    print(f"Completed {len(complete)} datasets; wrote {total} figure files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
