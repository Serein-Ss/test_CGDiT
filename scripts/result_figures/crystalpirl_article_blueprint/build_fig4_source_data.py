"""Build reproducible source data for the current Fig. 4 snapshot."""

from __future__ import annotations

import json
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from p_tqdm import p_map
from tqdm import tqdm

from cgdit.common.evaluation_utils import CompScaler, get_fp_pdist
from cgdit.evaluation.generated_properties import (
    _crystal_array_list,
    crystal_validity,
    load_generation_payload,
)
from cgdit.evaluation.metrics import Crystal
from scripts.result_figures.crystalpirl_article_blueprint.build_fig3_quality_metrics import (
    _structure,
)


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "assets" / "crystalpirl_article_blueprint" / "source"
TRAIN_CSV = ROOT / "data" / "mp_20" / "train.csv"
SEED = 42
TSNE_SAMPLES_PER_METHOD = 3000
DIVERSITY_SAMPLES_PER_METHOD = 1000
DIVERSITY_CANDIDATES_PER_METHOD = 1250
DIVERSITY_WORKERS = 16
FE_TARGET = -1.5
BG_TARGET = 2.0
FE_TOLERANCE = 0.3
BG_TOLERANCE = 0.3


@dataclass(frozen=True)
class MethodFiles:
    method: str
    generation: Path
    predictions: Path
    structural_metrics: Path


METHODS = (
    MethodFiles(
        "Base",
        ROOT / "output/singlerun/2026-06-27/00-32-50-mp20_base/generated_structures/formal/abinitio_empirical/unconditional/eval_gen_abinitio_empirical_uncond_n4096_seed42.pt",
        ROOT / "output/singlerun/2026-06-27/00-32-50-mp20_base/evaluations/property_predictions/eval_properties_gen_abinitio_empirical_uncond_n4096_seed42_predictor_seed42.csv",
        ROOT / "output/singlerun/2026-06-27/00-32-50-mp20_base/evaluations/structural_metrics/eval_metrics_gen_abinitio_empirical_uncond_n4096_seed42.json",
    ),
    MethodFiles(
        "CFG-FE",
        ROOT / "output/singlerun/2026-06-28/16-46-08-mp20_fe/generated_structures/formal/abinitio_empirical/conditional/eval_gen_abinitio_empirical_fe_m1p5_n4096_seed42.pt",
        ROOT / "output/singlerun/2026-06-28/16-46-08-mp20_fe/evaluations/property_predictions/eval_properties_gen_abinitio_empirical_fe_m1p5_n4096_seed42_predictor_seed42.csv",
        ROOT / "output/singlerun/2026-06-28/16-46-08-mp20_fe/evaluations/structural_metrics/eval_metrics_gen_abinitio_empirical_fe_m1p5_n4096_seed42.json",
    ),
    MethodFiles(
        "CFG-BG",
        ROOT / "output/singlerun/2026-06-30/11-28-44-mp20_bg/generated_structures/formal/abinitio_empirical/conditional/eval_gen_abinitio_empirical_bg_2_n4096_seed42.pt",
        ROOT / "output/singlerun/2026-06-30/11-28-44-mp20_bg/evaluations/property_predictions/eval_properties_gen_abinitio_empirical_bg_2_n4096_seed42_predictor_seed42.csv",
        ROOT / "output/singlerun/2026-06-30/11-28-44-mp20_bg/evaluations/structural_metrics/eval_metrics_gen_abinitio_empirical_bg_2_n4096_seed42.json",
    ),
    MethodFiles(
        "CFG-FE+BG",
        ROOT / "output/singlerun/2026-08-07/07-54-15-mp20_fe_bg/generated_structures/formal/abinitio_empirical/conditional/eval_gen_abinitio_empirical_fe_m1p5_bg_2_n4096_seed42.pt",
        ROOT / "output/singlerun/2026-08-07/07-54-15-mp20_fe_bg/evaluations/property_predictions/eval_properties_gen_abinitio_empirical_fe_m1p5_bg_2_n4096_seed42_predictor_seed42.csv",
        ROOT / "output/singlerun/2026-08-07/07-54-15-mp20_fe_bg/evaluations/structural_metrics/eval_metrics_gen_abinitio_empirical_fe_m1p5_bg_2_n4096_seed42.json",
    ),
)


def _load_crystals(path: Path) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
    crystals = _crystal_array_list(load_generation_payload(path))
    valid = np.asarray([crystal_validity(crystal)[2] for crystal in crystals])
    return crystals, valid


def _valid_joint_target_mask(valid: np.ndarray, predictions: pd.DataFrame) -> np.ndarray:
    formation_energy = pd.to_numeric(
        predictions["predicted_formation_energy_per_atom"], errors="coerce"
    ).to_numpy(dtype=float)
    band_gap = pd.to_numeric(
        predictions["predicted_band_gap"], errors="coerce"
    ).to_numpy(dtype=float)
    return (
        valid
        & np.isfinite(formation_energy)
        & np.isfinite(band_gap)
        & (np.abs(formation_energy - FE_TARGET) <= FE_TOLERANCE)
        & (np.abs(band_gap - BG_TARGET) <= BG_TOLERANCE)
    )


def _crystal_fingerprints(
    payload: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        crystal = Crystal(payload)
    except Exception:
        return None
    if crystal.struct_fp is None or crystal.comp_fp is None:
        return None
    return crystal.struct_fp, crystal.comp_fp


def _fingerprint_diversity(
    crystals: list[dict[str, np.ndarray]], valid: np.ndarray, method: str
) -> tuple[float, float, int]:
    rng = np.random.default_rng(SEED)
    candidates = rng.permutation(np.flatnonzero(valid))[
        :DIVERSITY_CANDIDATES_PER_METHOD
    ]
    payloads = [
        {
            name: np.array(value, copy=True)
            for name, value in crystals[int(index)].items()
        }
        for index in candidates
    ]
    fingerprints = p_map(
        _crystal_fingerprints,
        payloads,
        num_cpus=DIVERSITY_WORKERS,
        desc=f"{method} diversity",
    )
    usable = [item for item in fingerprints if item is not None][
        :DIVERSITY_SAMPLES_PER_METHOD
    ]
    structure_fingerprints = [item[0] for item in usable]
    composition_fingerprints = [item[1] for item in usable]
    if len(structure_fingerprints) != DIVERSITY_SAMPLES_PER_METHOD:
        raise RuntimeError(
            f"{method} produced only {len(structure_fingerprints)} usable "
            "fingerprints for diversity"
        )
    structure_diversity = get_fp_pdist(structure_fingerprints)
    composition_diversity = get_fp_pdist(
        CompScaler.transform(composition_fingerprints)
    )
    return (
        float(structure_diversity),
        float(composition_diversity),
        len(structure_fingerprints),
    )


def _tradeoff_data() -> pd.DataFrame:
    rows = []
    for files in METHODS:
        crystals, valid = _load_crystals(files.generation)
        predictions = (
            pd.read_csv(files.predictions)
            .set_index("structure_index")
            .reindex(range(len(crystals)))
        )
        hits = _valid_joint_target_mask(valid, predictions)
        structure_diversity, composition_diversity, diversity_n = (
            _fingerprint_diversity(crystals, valid, files.method)
        )
        rows.append(
            {
                "method": files.method,
                "n_total": len(crystals),
                "n_valid": int(valid.sum()),
                "n_valid_joint_target_hit": int(hits.sum()),
                "valid_joint_target_yield": float(hits.mean()),
                "structure_diversity": structure_diversity,
                "composition_diversity": composition_diversity,
                "n_diversity_fingerprints": diversity_n,
                "diversity_sample_seed": SEED,
                "seed": SEED,
                "formation_energy_target_eV_per_atom": FE_TARGET,
                "formation_energy_tolerance_eV_per_atom": FE_TOLERANCE,
                "band_gap_target_eV": BG_TARGET,
                "band_gap_tolerance_eV": BG_TOLERANCE,
            }
        )
    return pd.DataFrame(rows)


def _descriptor(structure: Structure) -> np.ndarray:
    composition = np.zeros(118, dtype=float)
    for element, amount in structure.composition.fractional_composition.items():
        composition[element.Z - 1] = float(amount)
    volume = max(float(structure.volume), 1.0e-8)
    lengths = np.asarray(structure.lattice.abc, dtype=float)
    shape = lengths / np.cbrt(volume)
    scalars = np.asarray(
        [
            len(structure),
            len(structure.composition.elements),
            float(structure.density),
            volume / max(len(structure), 1),
            *shape,
            *(np.asarray(structure.lattice.angles, dtype=float) / 180.0),
        ],
        dtype=float,
    )
    return np.concatenate([composition, scalars])


def _profile(
    method: str,
    source_index: int,
    structure: Structure,
    valid: bool,
    space_group: int,
) -> dict[str, object]:
    return {
        "method": method,
        "source_index": source_index,
        "valid": bool(valid),
        "formula": structure.composition.reduced_formula,
        "n_elements": len(structure.composition.elements),
        "num_atoms": len(structure),
        "density_g_cm3": float(structure.density),
        "space_group": int(space_group),
    }


def _space_group(structure: Structure) -> int:
    try:
        return int(
            SpacegroupAnalyzer(
                structure, symprec=0.1, angle_tolerance=5.0
            ).get_space_group_number()
        )
    except Exception:
        return 0


def _training_profiles() -> tuple[list[dict[str, object]], np.ndarray, Counter[str]]:
    table = pd.read_csv(TRAIN_CSV, usecols=["cif", "spacegroup.number"])
    profiles = []
    features = []
    elements: Counter[str] = Counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for index, row in tqdm(
            table.iterrows(), total=len(table), desc="MP-20 structure profiles"
        ):
            try:
                structure = Structure.from_str(row["cif"], fmt="cif")
            except Exception:
                continue
            profiles.append(
                _profile(
                    "Training set",
                    int(index),
                    structure,
                    True,
                    int(row["spacegroup.number"]),
                )
            )
            features.append(_descriptor(structure))
            for element, amount in structure.composition.items():
                elements[element.symbol] += float(amount)
    return profiles, np.asarray(features), elements


def _generated_profiles(
    method: str, generation: Path
) -> tuple[list[dict[str, object]], np.ndarray, Counter[str]]:
    crystals, valid = _load_crystals(generation)
    profiles = []
    valid_features = []
    elements: Counter[str] = Counter()
    for index, crystal in enumerate(
        tqdm(crystals, desc=f"{method} structure profiles")
    ):
        structure = _structure(crystal)
        profiles.append(
            _profile(
                method,
                index,
                structure,
                bool(valid[index]),
                _space_group(structure),
            )
        )
        if valid[index]:
            valid_features.append(_descriptor(structure))
        for element, amount in structure.composition.items():
            elements[element.symbol] += float(amount)
    return profiles, np.asarray(valid_features), elements


def _equal_tsne(
    features_by_method: dict[str, np.ndarray],
) -> pd.DataFrame:
    sample_size = min(
        TSNE_SAMPLES_PER_METHOD,
        *(len(features) for features in features_by_method.values()),
    )
    rng = np.random.default_rng(SEED)
    sampled = []
    labels = []
    source_rows = []
    for method, features in features_by_method.items():
        indices = rng.choice(len(features), size=sample_size, replace=False)
        sampled.append(features[indices])
        labels.extend([method] * sample_size)
        source_rows.extend(indices.tolist())
    feature_matrix = StandardScaler().fit_transform(np.vstack(sampled))
    components = min(40, feature_matrix.shape[1], feature_matrix.shape[0] - 1)
    reduced = PCA(n_components=components, random_state=SEED).fit_transform(
        feature_matrix
    )
    coordinates = TSNE(
        n_components=2,
        perplexity=40,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=SEED,
    ).fit_transform(reduced)
    return pd.DataFrame(
        {
            "method": labels,
            "sample_row": source_rows,
            "tsne_1": coordinates[:, 0],
            "tsne_2": coordinates[:, 1],
            "sample_size_per_method": sample_size,
            "seed": SEED,
        }
    )


def _element_frequencies(
    counters: dict[str, Counter[str]],
) -> pd.DataFrame:
    rows = []
    for method, counts in counters.items():
        total = float(sum(counts.values()))
        for element, count in sorted(counts.items()):
            rows.append(
                {
                    "method": method,
                    "element": element,
                    "atom_count": count,
                    "atom_fraction": count / total if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tradeoff = _tradeoff_data()
    tradeoff.to_csv(OUTPUT / "fig4a_target_yield_diversity.csv", index=False)

    training_profiles, training_features, training_elements = _training_profiles()
    base = METHODS[0]
    joint_cfg = METHODS[-1]
    base_profiles, base_features, base_elements = _generated_profiles(
        "Base", base.generation
    )
    cfg_profiles, cfg_features, cfg_elements = _generated_profiles(
        "CFG", joint_cfg.generation
    )

    profiles = pd.DataFrame(
        training_profiles + base_profiles + cfg_profiles
    )
    profiles.to_csv(OUTPUT / "fig4c_structure_profiles.csv", index=False)
    _element_frequencies(
        {
            "Training set": training_elements,
            "Base": base_elements,
            "CFG": cfg_elements,
        }
    ).to_csv(OUTPUT / "fig4c_element_frequencies.csv", index=False)

    coordinates = _equal_tsne(
        {
            "Training set": training_features,
            "Base": base_features,
            "CFG": cfg_features,
        }
    )
    coordinates.to_csv(OUTPUT / "fig4b_tsne.csv", index=False)

    contract = {
        "seed": SEED,
        "generation_mode": "ab_initio_from_scratch",
        "crystalpirl_status": "pending",
        "fig4a": {
            "y": "valid joint target hits divided by all generated structures",
            "x": "mean pairwise Euclidean distance among CrystalNN structure fingerprints",
            "diversity_sampling": "first 1000 usable fingerprints from a seed=42 permutation of at most 1250 valid candidates per method",
            "formation_energy_target_eV_per_atom": FE_TARGET,
            "formation_energy_tolerance_eV_per_atom": FE_TOLERANCE,
            "band_gap_target_eV": BG_TARGET,
            "band_gap_tolerance_eV": BG_TOLERANCE,
            "uncertainty": "none; seed=42 only",
        },
        "fig4b": {
            "classes": ["Training set", "Base", "CFG-FE+BG"],
            "sampling": "equal-size random sample of valid structures per class",
            "sample_size_per_class": int(
                coordinates["sample_size_per_method"].iloc[0]
            ),
            "descriptor": "118-element atomic-fraction vector plus atom count, number of elements, density, volume per atom, normalized lattice lengths and lattice angles",
            "pipeline": "standardization -> PCA(40) -> t-SNE(perplexity=40, seed=42)",
            "plot_encoding": "Training set shown as peak-normalized Gaussian-smoothed 2D histogram density; Base and CFG-FE+BG shown as points",
            "claim_boundary": "visual embedding only; no quantitative coverage or divergence claim",
        },
        "fig4c": {
            "classes": ["Training set", "Base", "CFG-FE+BG"],
            "sample_scope": "full available set for each class",
            "normalization": "element frequencies and categorical histograms normalized within each class; density curves normalized as probability density",
            "space_group": "training labels from MP-20; generated labels recomputed with pymatgen symprec=0.1 and angle_tolerance=5 degrees; 0 means unresolved",
            "panel_labels": ["c1 elements", "c2 n-ary composition", "c3 space group", "c4 atoms per cell", "c5 density"],
        },
        "deferred": "representative structures will be shown with MLFF and DFT validation in a later figure",
    }
    (OUTPUT / "fig4_data_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n"
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
