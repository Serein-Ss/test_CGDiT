"""Build full-set Base and CFG quality metrics used by Fig. 3a."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymatgen.analysis.phase_diagram import PDEntry
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Lattice, Structure
from scipy.optimize import linprog
from tqdm import tqdm

from cgdit.evaluation.generated_properties import (
    _crystal_array_list,
    crystal_validity,
    load_generation_payload,
)


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "assets" / "crystalpirl_article_blueprint" / "source"
TRAIN_CSV = ROOT / "data" / "mp_20" / "train.csv"
MP_FORMATION_REFERENCE = OUTPUT / "fig3_mp_formation_reference.csv"
STABLE_E_ABOVE_HULL_MAX = 0.1


@dataclass(frozen=True)
class MethodFiles:
    method: str
    generation: Path
    predictions: Path


METHODS = (
    MethodFiles(
        method="Base",
        generation=ROOT
        / "output/singlerun/2026-06-27/00-32-50-mp20_base/generated_structures/formal/abinitio_empirical/unconditional/eval_gen_abinitio_empirical_uncond_n4096_seed42.pt",
        predictions=ROOT
        / "output/singlerun/2026-06-27/00-32-50-mp20_base/evaluations/property_predictions/eval_properties_gen_abinitio_empirical_uncond_n4096_seed42_predictor_seed42.csv",
    ),
    MethodFiles(
        method="CFG-FE",
        generation=ROOT
        / "output/singlerun/2026-06-28/16-46-08-mp20_fe/generated_structures/formal/abinitio_empirical/conditional/eval_gen_abinitio_empirical_fe_m1p5_n4096_seed42.pt",
        predictions=ROOT
        / "output/singlerun/2026-06-28/16-46-08-mp20_fe/evaluations/property_predictions/eval_properties_gen_abinitio_empirical_fe_m1p5_n4096_seed42_predictor_seed42.csv",
    ),
    MethodFiles(
        method="CFG-BG",
        generation=ROOT
        / "output/singlerun/2026-06-30/11-28-44-mp20_bg/generated_structures/formal/abinitio_empirical/conditional/eval_gen_abinitio_empirical_bg_2_n4096_seed42.pt",
        predictions=ROOT
        / "output/singlerun/2026-06-30/11-28-44-mp20_bg/evaluations/property_predictions/eval_properties_gen_abinitio_empirical_bg_2_n4096_seed42_predictor_seed42.csv",
    ),
    MethodFiles(
        method="CFG",
        generation=ROOT
        / "output/singlerun/2026-08-07/07-54-15-mp20_fe_bg/generated_structures/formal/abinitio_empirical/conditional/eval_gen_abinitio_empirical_fe_m1p5_bg_2_n4096_seed42.pt",
        predictions=ROOT
        / "output/singlerun/2026-08-07/07-54-15-mp20_fe_bg/evaluations/property_predictions/eval_properties_gen_abinitio_empirical_fe_m1p5_bg_2_n4096_seed42_predictor_seed42.csv",
    ),
    MethodFiles(
        method="GRPO+PIPO-FE (step 39)",
        generation=ROOT
        / "output/rl_generation/20260824-051136/fe/model/generated_structures/formal/abinitio_empirical/conditional/eval_gen_abinitio_empirical_rl_fe_m1p5_n4096_seed42.pt",
        predictions=ROOT
        / "output/rl_generation/20260824-051136/fe/model/evaluations/property_predictions/eval_properties_gen_abinitio_empirical_rl_fe_m1p5_n4096_seed42_predictor_seed42.csv",
    ),
    MethodFiles(
        method="GRPO+PIPO-BG (step 39)",
        generation=ROOT
        / "output/rl_generation/20260824-051136/bg/model/generated_structures/formal/abinitio_empirical/conditional/eval_gen_abinitio_empirical_rl_bg_2_n4096_seed42.pt",
        predictions=ROOT
        / "output/rl_generation/20260824-051136/bg/model/evaluations/property_predictions/eval_properties_gen_abinitio_empirical_rl_bg_2_n4096_seed42_predictor_seed42.csv",
    ),
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


def _safe_match(matcher: StructureMatcher, left: Structure, right: Structure) -> bool:
    try:
        return bool(matcher.fit(left, right))
    except Exception:
        return False


def _unique_representatives(
    structures: list[Structure], valid: np.ndarray, matcher: StructureMatcher
) -> np.ndarray:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in np.flatnonzero(valid):
        grouped[structures[int(index)].composition.reduced_formula].append(int(index))

    representatives = []
    for indices in tqdm(grouped.values(), desc="Uniqueness", leave=False):
        formula_representatives: list[int] = []
        for index in indices:
            if not any(
                _safe_match(matcher, structures[index], structures[other])
                for other in formula_representatives
            ):
                formula_representatives.append(index)
        representatives.extend(formula_representatives)
    return np.asarray(representatives, dtype=int)


def _training_index(formulas: set[str], train_csv: Path) -> dict[str, list[Structure]]:
    table = pd.read_csv(train_csv, usecols=["pretty_formula", "cif"])
    table["formula"] = table["pretty_formula"].map(
        lambda value: Composition(value).reduced_formula
    )
    index: dict[str, list[Structure]] = defaultdict(list)
    selected = table[table["formula"].isin(formulas)]
    for row in tqdm(selected.itertuples(index=False), desc="MP-20 reference", leave=False):
        try:
            index[row.formula].append(Structure.from_str(row.cif, fmt="cif"))
        except Exception:
            continue
    return index


def _novel_mask(
    structures: list[Structure],
    representatives: np.ndarray,
    reference: dict[str, list[Structure]],
    matcher: StructureMatcher,
) -> np.ndarray:
    novel = np.zeros(len(structures), dtype=bool)
    for index in tqdm(representatives, desc="Novelty", leave=False):
        structure = structures[int(index)]
        formula = structure.composition.reduced_formula
        novel[int(index)] = not any(
            _safe_match(matcher, structure, candidate)
            for candidate in reference.get(formula, [])
        )
    return novel


def _mp_formation_reference() -> tuple[dict[tuple[str, ...], list[PDEntry]], str]:
    if MP_FORMATION_REFERENCE.is_file():
        table = pd.read_csv(MP_FORMATION_REFERENCE)
        database_version = str(table["mp_database_version"].iloc[0])
    else:
        if not os.getenv("MP_API_KEY"):
            raise RuntimeError(
                "MP_API_KEY is not set. Put it in the shell environment or an ignored .env file."
            )
        try:
            from mp_api.client import MPRester
        except ImportError as error:
            raise RuntimeError(
                "mp-api is missing; run `uv sync --extra stability`."
            ) from error
        rows = []
        with MPRester(mute_progress_bars=True) as rester:
            database_version = rester.get_database_version()
            documents = rester.materials.thermo.search(
                is_stable=True,
                thermo_types=["GGA_GGA+U"],
                fields=["composition", "formation_energy_per_atom"],
                chunk_size=1000,
            )
            for document in tqdm(documents, desc="MP stable formation entries"):
                composition = document.composition.reduced_composition
                rows.append(
                    {
                        "chemsys": "-".join(
                            sorted(
                                element.symbol
                                for element in composition.elements
                            )
                        ),
                        "formula": composition.formula.replace(" ", ""),
                        "formation_energy_per_atom": float(
                            document.formation_energy_per_atom
                        ),
                        "mp_database_version": database_version,
                    }
                )
        table = pd.DataFrame(rows).drop_duplicates()
        table.to_csv(MP_FORMATION_REFERENCE, index=False)

    reference: dict[tuple[str, ...], list[PDEntry]] = defaultdict(list)
    for row in table.itertuples(index=False):
        composition = Composition(row.formula)
        system = tuple(str(row.chemsys).split("-"))
        reference[system].append(
            PDEntry(
                composition,
                float(row.formation_energy_per_atom) * composition.num_atoms,
            )
        )
    return reference, database_version


def _mp_hull_distances(
    structures: list[Structure],
    valid: np.ndarray,
    formation_energy: np.ndarray,
    reference: dict[tuple[str, ...], list[PDEntry]],
) -> tuple[np.ndarray, np.ndarray]:
    hull_energy = np.full(len(structures), np.nan)
    e_above_hull = np.full(len(structures), np.nan)
    linear_programs: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray]] = {}
    for index in tqdm(
        np.flatnonzero(valid & np.isfinite(formation_energy)),
        desc="MP formation-energy hull",
    ):
        structure = structures[int(index)]
        elements = tuple(
            sorted(element.symbol for element in structure.composition.elements)
        )
        if elements not in linear_programs:
            element_set = set(elements)
            entries = [
                entry
                for subsystem, subsystem_entries in reference.items()
                if set(subsystem).issubset(element_set)
                for entry in subsystem_entries
            ]
            present_elements = {
                element.symbol
                for entry in entries
                for element in entry.composition.elements
            }
            entries.extend(
                PDEntry(Composition(element), 0.0)
                for element in elements
                if element not in present_elements
            )
            objective = np.asarray(
                [entry.energy_per_atom for entry in entries], dtype=float
            )
            fractions = np.asarray(
                [
                    [entry.composition.get_atomic_fraction(element) for entry in entries]
                    for element in elements
                ],
                dtype=float,
            )
            linear_programs[elements] = objective, fractions
        objective, fractions = linear_programs[elements]
        target = np.asarray(
            [structure.composition.get_atomic_fraction(element) for element in elements],
            dtype=float,
        )
        solution = linprog(
            objective,
            A_eq=fractions,
            b_eq=target,
            bounds=(0.0, None),
            method="highs",
        )
        if not solution.success:
            continue
        hull = float(solution.fun)
        hull_energy[int(index)] = hull
        e_above_hull[int(index)] = max(
            0.0, formation_energy[int(index)] - hull
        )
    return hull_energy, e_above_hull


def _load_method(files: MethodFiles) -> dict[str, object]:
    crystals = _crystal_array_list(load_generation_payload(files.generation))
    structures = [_structure(crystal) for crystal in crystals]
    validity = np.asarray([crystal_validity(crystal) for crystal in crystals])

    predictions = pd.read_csv(files.predictions).set_index("structure_index")
    formation_energy = predictions["predicted_formation_energy_per_atom"].reindex(
        range(len(structures))
    ).to_numpy(dtype=float)
    return {
        "files": files,
        "structures": structures,
        "composition_valid": validity[:, 0],
        "structure_valid": validity[:, 1],
        "valid": validity[:, 2],
        "formation_energy": formation_energy,
    }


def _metric_row(
    method: str,
    valid: np.ndarray,
    stable: np.ndarray,
    representatives: np.ndarray,
    novel: np.ndarray,
    composition_valid: np.ndarray,
    structure_valid: np.ndarray,
    novelty_reference: str,
) -> dict[str, object]:
    total = len(valid)
    unique = np.zeros(total, dtype=bool)
    unique[representatives] = True
    sun = stable & unique & novel
    return {
        "method": method,
        "n_total": total,
        "n_valid": int(valid.sum()),
        "n_unique": int(unique.sum()),
        "n_novel_unique": int((unique & novel).sum()),
        "n_stable": int(stable.sum()),
        "n_sun": int(sun.sum()),
        "compositional_validity": float(composition_valid.mean()),
        "structural_validity": float(structure_valid.mean()),
        "uniqueness": float(unique.sum() / valid.sum()) if valid.any() else np.nan,
        "novelty_among_unique": (
            float((unique & novel).sum() / unique.sum()) if unique.any() else np.nan
        ),
        "sun": float(sun.sum() / total) if total else np.nan,
        "mp_hull_stability": float(stable.sum() / total) if total else np.nan,
        "novelty_reference": novelty_reference,
    }


def main() -> None:
    load_dotenv(ROOT / ".env")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    matcher = StructureMatcher(
        ltol=0.2,
        stol=0.3,
        angle_tol=5.0,
        primitive_cell=True,
        scale=True,
    )
    loaded = [_load_method(files) for files in METHODS]
    mp_reference, mp_database_version = _mp_formation_reference()

    for item in loaded:
        item["mp_hull_formation_energy"], item["e_above_hull"] = (
            _mp_hull_distances(
                item["structures"],
                item["valid"],
                item["formation_energy"],
                mp_reference,
            )
        )
        item["stable"] = (
            item["valid"]
            & np.isfinite(item["e_above_hull"])
            & (item["e_above_hull"] <= STABLE_E_ABOVE_HULL_MAX)
        )
        item["representatives"] = _unique_representatives(
            item["structures"], item["valid"], matcher
        )

    formulas = {
        item["structures"][int(index)].composition.reduced_formula
        for item in loaded
        for index in item["representatives"]
    }
    train_reference = _training_index(formulas, TRAIN_CSV)
    for item in loaded:
        item["novel"] = _novel_mask(
            item["structures"], item["representatives"], train_reference, matcher
        )

    novelty_reference = "mp_20_train"
    rows = [
        _metric_row(
            method=item["files"].method,
            valid=item["valid"],
            stable=item["stable"],
            representatives=item["representatives"],
            novel=item["novel"],
            composition_valid=item["composition_valid"],
            structure_valid=item["structure_valid"],
            novelty_reference=novelty_reference,
        )
        for item in loaded
    ]
    results = pd.DataFrame(rows)
    results["stable_e_above_hull_max_eV_per_atom"] = STABLE_E_ABOVE_HULL_MAX
    results["mp_database_version"] = mp_database_version
    output = OUTPUT / "fig3_quality_metrics.csv"
    results.to_csv(output, index=False)

    rate_columns = [
        "compositional_validity",
        "structural_validity",
        "mp_hull_stability",
        "sun",
        "uniqueness",
        "novelty_among_unique",
    ]
    percent = results[["method", *rate_columns, "novelty_reference", "mp_database_version"]].copy()
    percent[rate_columns] *= 100.0
    percent.to_csv(OUTPUT / "fig3a_metrics_percent.csv", index=False)

    per_structure = []
    for item in loaded:
        unique = np.zeros(len(item["valid"]), dtype=bool)
        unique[item["representatives"]] = True
        sun = item["stable"] & unique & item["novel"]
        per_structure.append(
            pd.DataFrame(
                {
                    "method": item["files"].method,
                    "structure_index": np.arange(len(item["valid"])),
                    "composition_valid": item["composition_valid"],
                    "structure_valid": item["structure_valid"],
                    "valid": item["valid"],
                    "predicted_formation_energy_per_atom": item["formation_energy"],
                    "mp_hull_formation_energy_per_atom": item["mp_hull_formation_energy"],
                    "predicted_e_above_hull": item["e_above_hull"],
                    "stable": item["stable"],
                    "unique_representative": unique,
                    "novel_vs_mp20_train": item["novel"],
                    "sun": sun,
                }
            )
        )
    pd.concat(per_structure, ignore_index=True).to_csv(
        OUTPUT / "fig3_quality_per_structure.csv", index=False
    )

    (OUTPUT / "fig3_quality_contract.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "generation_mode": "ab_initio_from_scratch",
                "policy_status": "GRPO+PIPO checkpoints after 40 completed updates (step 39); not final CrystalPIRL",
                "property_evaluator": "seed42 reward predictors for all generated methods",
                "sample_scope": "all_4096_generated_structures_per_method",
                "mp_reference": "stable GGA/GGA+U Materials Project entries on a formation-energy-per-atom scale",
                "predicted_e_above_hull": "max(0, predicted formation energy per atom - MP formation-energy hull at the generated composition)",
                "stable": "overall valid AND finite predicted E_hull AND predicted E_hull <= 0.1 eV/atom",
                "sun": "stable AND unique representative AND novel versus MP-20 train, divided by n_total",
                "uniqueness_denominator": "overall valid generated structures",
                "novelty_denominator": "unique overall-valid generated structures",
                "novelty_reference": novelty_reference,
                "mp_database_version": mp_database_version,
                "structure_matcher": {
                    "ltol": 0.2,
                    "stol": 0.3,
                    "angle_tol": 5.0,
                    "primitive_cell": True,
                    "scale": True,
                },
                "claim_boundary": "formation-energy-predictor estimate against an MP reference hull; not relaxed MLFF/DFT stability",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
