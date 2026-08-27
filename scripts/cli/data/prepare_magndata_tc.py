"""Prepare leakage-controlled Magndata Curie-temperature regression splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

warnings.filterwarnings("ignore", message="Issues encountered while parsing CIF")


REQUIRED_COLUMNS = {
    "material_id",
    "tc",
    "pretty_formula",
    "elements",
    "cif",
}
SPLIT_NAMES = ("train", "val", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_group_splits(
    frame: pd.DataFrame,
    seed: int,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> dict[str, str]:
    """Stratify formula groups by median Tc and assign whole groups to splits."""
    group_targets = frame.groupby("pretty_formula", sort=True)["tc"].median()
    bins = pd.qcut(
        group_targets.rank(method="first"),
        q=min(10, len(group_targets)),
        labels=False,
        duplicates="drop",
    )
    assignments: dict[str, str] = {}
    for bin_id in sorted(bins.unique()):
        formulas = bins.index[bins == bin_id].tolist()
        formulas.sort(
            key=lambda formula: hashlib.sha256(
                f"{seed}:{formula}".encode("utf-8")
            ).hexdigest()
        )
        n_groups = len(formulas)
        n_train = int(round(train_fraction * n_groups))
        n_val = int(round(val_fraction * n_groups))
        n_train = min(max(n_train, 1), n_groups)
        n_val = min(max(n_val, 1), max(n_groups - n_train, 0))
        for index, formula in enumerate(formulas):
            if index < n_train:
                split = "train"
            elif index < n_train + n_val:
                split = "val"
            else:
                split = "test"
            assignments[formula] = split
    return assignments


def standardize_structures(
    source: pd.DataFrame,
    max_atoms: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    rejected: list[dict] = []
    for _, row in source.iterrows():
        material_id = str(row["material_id"])
        try:
            structure = Structure.from_str(row["cif"], fmt="cif")
            source_natoms = len(structure)
            structure = structure.get_primitive_structure().get_reduced_structure()
            primitive_natoms = len(structure)
            if primitive_natoms > max_atoms:
                rejected.append(
                    {
                        "material_id": material_id,
                        "reason": "primitive_natoms_exceeds_limit",
                        "source_natoms": source_natoms,
                        "primitive_natoms": primitive_natoms,
                    }
                )
                continue

            output = row.to_dict()
            output.update(
                {
                    "material_id": material_id,
                    "pretty_formula": structure.composition.reduced_formula,
                    "natoms": primitive_natoms,
                    "source_natoms": source_natoms,
                    "cif": str(CifWriter(structure)),
                    "spacegroup.number": int(
                        SpacegroupAnalyzer(structure, symprec=0.1).get_space_group_number()
                    ),
                }
            )
            rows.append(output)
        except Exception as exc:
            rejected.append(
                {
                    "material_id": material_id,
                    "reason": f"parse_error:{type(exc).__name__}",
                    "source_natoms": row.get("natoms"),
                    "primitive_natoms": None,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(rejected)


def summary(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "rows": int(len(frame)),
        "formula_groups": int(frame["pretty_formula"].nunique()),
        "tc_mean_k": float(frame["tc"].mean()),
        "tc_std_k_ddof0": float(frame["tc"].std(ddof=0)),
        "tc_min_k": float(frame["tc"].min()),
        "tc_max_k": float(frame["tc"].max()),
        "max_atoms": int(frame["natoms"].max()),
    }


def prepare(source_path: Path, output_dir: Path, seed: int, max_atoms: int) -> dict:
    source_path = source_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(source_path)
    missing = REQUIRED_COLUMNS.difference(source.columns)
    if missing:
        raise ValueError(f"Source dataset is missing columns: {sorted(missing)}")
    if source["material_id"].duplicated().any():
        raise ValueError("material_id must be unique in the deduplicated source")
    if not np.isfinite(source["tc"].to_numpy(dtype=float)).all():
        raise ValueError("tc contains non-finite values")

    source_copy = output_dir / "source_magndata_tc.csv"
    if source_path != source_copy.resolve():
        shutil.copyfile(source_path, source_copy)

    standardized, rejected = standardize_structures(source, max_atoms=max_atoms)
    if standardized.empty:
        raise RuntimeError("No Magndata structures survived standardization")

    assignments = stable_group_splits(standardized, seed=seed)
    standardized["split"] = standardized["pretty_formula"].map(assignments)
    if standardized["split"].isna().any():
        raise RuntimeError("At least one formula group was not assigned to a split")

    all_path = output_dir / "all.csv"
    standardized.to_csv(all_path, index=False)
    split_frames: dict[str, pd.DataFrame] = {}
    for split in SPLIT_NAMES:
        split_frame = standardized[standardized["split"] == split].copy()
        split_frame.drop(columns="split").to_csv(output_dir / f"{split}.csv", index=False)
        split_frames[split] = split_frame
    rejected.to_csv(output_dir / "rejected.csv", index=False)

    formula_sets = {
        split: set(frame["pretty_formula"])
        for split, frame in split_frames.items()
    }
    overlaps = {
        "train_val": len(formula_sets["train"] & formula_sets["val"]),
        "train_test": len(formula_sets["train"] & formula_sets["test"]),
        "val_test": len(formula_sets["val"] & formula_sets["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Formula leakage detected: {overlaps}")

    train_tc = split_frames["train"]["tc"].to_numpy(dtype=float)
    manifest = {
        "dataset": "Magndata-Tc",
        "target": "tc",
        "target_unit": "K",
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "source_rows": int(len(source)),
        "standardization": "primitive cell followed by Niggli reduction",
        "deduplication": "source already removes equivalent structures with identical Tc",
        "max_primitive_atoms": max_atoms,
        "rejected_rows": int(len(rejected)),
        "split_seed": seed,
        "split_policy": "formula-grouped, group-median-Tc decile stratified, 8:1:1",
        "formula_overlap": overlaps,
        "target_stats_from_train_only": {
            "mean": float(train_tc.mean()),
            "std_ddof0": float(train_tc.std(ddof=0)),
        },
        "splits": {split: summary(frame) for split, frame in split_frames.items()},
    }
    manifest["files"] = {
        path.name: sha256(path)
        for path in sorted(output_dir.glob("*.csv"))
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-atoms", type=int, default=100)
    return parser


def cli(argv=None) -> None:
    args = build_parser().parse_args(argv)
    manifest = prepare(args.source, args.output_dir, args.seed, args.max_atoms)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    cli()
