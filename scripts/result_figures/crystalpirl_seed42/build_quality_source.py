"""Add deterministic quality, novelty and embedding evidence for Fig. 4."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cgdit.evaluation.generated_properties import (
    _crystal_array_list,
    load_generation_payload,
)


def _model_root(generation: Path) -> Path:
    for parent in generation.parents:
        if parent.name == "generated_structures":
            return parent.parent
    raise ValueError(f"Not inside generated_structures/: {generation}")


def _prediction_path(generation: Path) -> Path:
    label = generation.stem.removeprefix("eval_gen_")
    return (
        _model_root(generation)
        / "evaluations"
        / "property_predictions"
        / f"eval_properties_gen_{label}_predictor_seed123.csv"
    )


def _structure(crystal: dict[str, np.ndarray]) -> Structure:
    return Structure(
        lattice=Lattice.from_parameters(
            *(crystal["lengths"].tolist() + crystal["angles"].tolist())
        ),
        species=crystal["atom_types"],
        coords=np.mod(crystal["frac_coords"], 1.0),
        coords_are_cartesian=False,
    )


def _sample_seed(group_id: str) -> int:
    digest = hashlib.sha256(f"seed42:{group_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _safe_match(
    matcher: StructureMatcher, left: Structure, right: Structure
) -> bool:
    try:
        return bool(matcher.fit(left, right))
    except Exception:
        return False


def _features(structure: Structure) -> np.ndarray:
    composition = np.zeros(100, dtype=float)
    total = float(structure.composition.num_atoms)
    for element, amount in structure.composition.items():
        if 1 <= element.Z <= 100:
            composition[element.Z - 1] = float(amount) / total
    lengths = np.asarray(structure.lattice.abc, dtype=float)
    angles = np.asarray(structure.lattice.angles, dtype=float)
    shape = np.concatenate(
        [
            np.log1p(lengths),
            angles / 180.0,
            [
                np.log1p(structure.volume / len(structure)),
                np.log1p(float(structure.density)),
                len(structure) / 20.0,
            ],
        ]
    )
    return np.concatenate([composition, shape])


def _load_samples(
    summary: pd.DataFrame, sample_size: int
) -> tuple[pd.DataFrame, list[Structure], np.ndarray]:
    rows = []
    structures = []
    features = []
    for item in summary.to_dict("records"):
        generation = Path(item["source_generation"])
        predictions = pd.read_csv(_prediction_path(generation))
        valid_indices = predictions.loc[
            predictions["valid"].astype(bool), "structure_index"
        ].to_numpy(dtype=int)
        if valid_indices.size == 0:
            raise RuntimeError(f"No valid structures for {item['group_id']}")
        rng = np.random.default_rng(_sample_seed(item["group_id"]))
        chosen = rng.choice(
            valid_indices,
            size=min(sample_size, valid_indices.size),
            replace=False,
        )
        payload = load_generation_payload(generation)
        crystals = _crystal_array_list(payload)
        for structure_index in chosen:
            structure = _structure(crystals[int(structure_index)])
            formula = structure.composition.reduced_formula
            try:
                spacegroup = int(
                    SpacegroupAnalyzer(structure, symprec=0.1)
                    .get_space_group_number()
                )
            except Exception:
                spacegroup = 0
            rows.append(
                {
                    "group_id": item["group_id"],
                    "method": item["method"],
                    "method_family": item["method_family"],
                    "algorithm": item.get("algorithm"),
                    "task": item.get("task"),
                    "generation_mode": item["generation_mode"],
                    "seed": 42,
                    "valid_pool_n": int(valid_indices.size),
                    "structure_index": int(structure_index),
                    "structure_id": f"{item['group_id']}:{structure_index}",
                    "formula": formula,
                    "num_atoms": len(structure),
                    "num_elements": len(structure.composition.elements),
                    "volume_per_atom": structure.volume / len(structure),
                    "density": float(structure.density),
                    "spacegroup": spacegroup,
                    "source_generation": str(generation),
                }
            )
            structures.append(structure)
            features.append(_features(structure))
    return pd.DataFrame(rows), structures, np.asarray(features)


def _training_index(formulas: set[str], train_csv: Path) -> dict[str, list[Structure]]:
    table = pd.read_csv(train_csv, usecols=["pretty_formula", "cif"])
    table["reduced_formula"] = table["pretty_formula"].map(
        lambda value: Composition(value).reduced_formula
    )
    table = table[table["reduced_formula"].isin(formulas)]
    index: dict[str, list[Structure]] = defaultdict(list)
    for row in table.itertuples(index=False):
        try:
            index[row.reduced_formula].append(Structure.from_str(row.cif, fmt="cif"))
        except Exception:
            continue
    return index


def _mark_unique_and_novel(
    table: pd.DataFrame,
    structures: list[Structure],
    train_index: dict[str, list[Structure]],
) -> pd.DataFrame:
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0)
    unique = np.ones(len(table), dtype=bool)
    novel = np.ones(len(table), dtype=bool)
    for _, indices in table.groupby(["group_id", "formula"], sort=False).groups.items():
        representatives: list[Structure] = []
        for index in indices:
            structure = structures[int(index)]
            if any(_safe_match(matcher, structure, other) for other in representatives):
                unique[int(index)] = False
            else:
                representatives.append(structure)
    for index, row in table.iterrows():
        references = train_index.get(row["formula"], [])
        novel[index] = not any(
            _safe_match(matcher, structures[index], reference)
            for reference in references
        )
    result = table.copy()
    result["unique"] = unique
    result["novel"] = novel
    return result


def _add_embedding(table: pd.DataFrame, features: np.ndarray) -> pd.DataFrame:
    scaled = StandardScaler().fit_transform(features)
    embedding = PCA(n_components=2, random_state=42).fit_transform(scaled)
    result = table.copy()
    result["embedding_1"] = embedding[:, 0]
    result["embedding_2"] = embedding[:, 1]
    return result


def _shannon(values: pd.Series) -> float:
    probabilities = values.value_counts(normalize=True).to_numpy(dtype=float)
    return float(-(probabilities * np.log(probabilities)).sum())


def _sample_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_id, frame in table.groupby("group_id", sort=False):
        unique_frame = frame[frame["unique"]]
        rows.append(
            {
                "group_id": group_id,
                "sample_n": len(frame),
                "uniqueness": float(frame["unique"].mean()),
                "novelty_among_unique": (
                    float(unique_frame["novel"].mean())
                    if len(unique_frame)
                    else np.nan
                ),
                "formula_shannon": _shannon(frame["formula"]),
                "mean_num_elements": float(frame["num_elements"].mean()),
                "mean_num_atoms": float(frame["num_atoms"].mean()),
                "embedding_dispersion": float(
                    np.sqrt(
                        frame[["embedding_1", "embedding_2"]]
                        .var(ddof=0)
                        .sum()
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, default=Path("data/mp_20/train.csv"))
    parser.add_argument("--sample-size", type=int, default=256)
    args = parser.parse_args()
    if args.sample_size < 2:
        raise ValueError("sample-size must be at least 2")

    quality_path = args.source_dir / "fig4_quality_summary.csv"
    quality = pd.read_csv(quality_path)
    table, structures, features = _load_samples(quality, args.sample_size)
    train_index = _training_index(set(table["formula"]), args.train_csv)
    table = _mark_unique_and_novel(table, structures, train_index)
    table = _add_embedding(table, features)
    sample_summary = _sample_summary(table)
    enriched = quality.merge(sample_summary, on="group_id", validate="one_to_one")

    sample_path = args.source_dir / "fig4_structure_sample.csv"
    table.to_csv(sample_path, index=False)
    enriched.to_csv(quality_path, index=False)
    contract_path = args.source_dir / "fig4_sampling_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "seed": 42,
                "sample_size_per_group": args.sample_size,
                "sampling": "uniform_without_replacement_from_valid_structures",
                "uniqueness_matcher": {
                    "ltol": 0.2,
                    "stol": 0.3,
                    "angle_tol": 5.0,
                },
                "novelty_reference": str(args.train_csv),
                "embedding": "standardized composition-lattice PCA",
                "claim_boundary": "pilot proxy; no MLFF/DFT stability claim",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = args.source_dir / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for path in (quality_path, sample_path, contract_path):
        rows = len(enriched) if path == quality_path else (
            len(table) if path == sample_path else None
        )
        manifest["files"][path.name] = {
            "rows": rows,
            "sha256": _sha256(path),
        }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
