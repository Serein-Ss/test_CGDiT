"""Freeze seed-42 GRPO+PIPO Source Data for the current Fig. 1--Fig. 4.

This module intentionally labels the trained policies as GRPO+PIPO.  It does
not relabel them as CrystalPIRL because the upstream runs do not apply the
paired policy gate during optimization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher

from scripts.result_figures.crystalpirl_article_blueprint.build_fig3_quality_metrics import (
    MethodFiles as QualityMethodFiles,
    STABLE_E_ABOVE_HULL_MAX,
    _load_method,
    _metric_row,
    _mp_formation_reference,
    _mp_hull_distances,
    _novel_mask,
    _training_index,
    _unique_representatives,
)
from scripts.result_figures.crystalpirl_article_blueprint.build_fig4_source_data import (
    MethodFiles as SearchMethodFiles,
    _element_frequencies,
    _equal_tsne,
    _fingerprint_diversity,
    _generated_profiles,
    _load_crystals,
    _training_profiles,
    _valid_joint_target_mask,
)


ROOT = Path(__file__).resolve().parents[3]
TRAIN_CSV = ROOT / "data/mp_20/train.csv"
BASE_ROOT = ROOT / "output/singlerun/2026-06-27/00-32-50-mp20_base"
CFG_FE_ROOT = ROOT / "output/singlerun/2026-06-28/16-46-08-mp20_fe"
CFG_BG_ROOT = ROOT / "output/singlerun/2026-06-30/11-28-44-mp20_bg"
CFG_JOINT_ROOT = ROOT / "output/singlerun/2026-08-07/07-54-15-mp20_fe_bg"
BASELINE_DIVERSITY = (
    ROOT
    / "assets/crystalpirl_article_blueprint/source/fig4a_target_yield_diversity.csv"
)
SEED = 42
EXPECTED_UPDATES = 200
EXPECTED_SAMPLES = 4096
FE_TARGET = -1.5
BG_TARGET = 2.0
FE_TOLERANCE = 0.30
BG_TOLERANCE = 0.30


def _generation(model_root: Path, label: str, conditional: bool) -> Path:
    branch = "conditional" if conditional else "unconditional"
    return (
        model_root
        / "generated_structures/formal/abinitio_empirical"
        / branch
        / f"eval_gen_{label}.pt"
    )


def _predictions(model_root: Path, label: str) -> Path:
    return (
        model_root
        / "evaluations/property_predictions"
        / f"eval_properties_gen_{label}_predictor_seed123.csv"
    )


def _structural_metrics(model_root: Path, label: str) -> Path:
    return (
        model_root
        / "evaluations/structural_metrics"
        / f"eval_metrics_gen_{label}.json"
    )


def _baseline_specs() -> list[dict[str, Any]]:
    definitions = [
        (
            "Base",
            "base",
            BASE_ROOT,
            "abinitio_empirical_uncond_n4096_seed42",
            False,
        ),
        (
            "CFG-FE",
            "cfg",
            CFG_FE_ROOT,
            "abinitio_empirical_fe_m1p5_n4096_seed42",
            True,
        ),
        (
            "CFG-BG",
            "cfg",
            CFG_BG_ROOT,
            "abinitio_empirical_bg_2_n4096_seed42",
            True,
        ),
        (
            "CFG-FE+BG",
            "cfg",
            CFG_JOINT_ROOT,
            "abinitio_empirical_fe_m1p5_bg_2_n4096_seed42",
            True,
        ),
    ]
    return [
        {
            "method": method,
            "family": family,
            "model_root": model_root,
            "label": label,
            "generation": _generation(model_root, label, conditional),
            "predictions": _predictions(model_root, label),
            "structural_metrics": _structural_metrics(model_root, label),
        }
        for method, family, model_root, label, conditional in definitions
    ]


def _rl_specs(fe_bg_root: Path, joint_root: Path) -> list[dict[str, Any]]:
    definitions = [
        (
            "GRPO+PIPO-FE",
            "fe",
            fe_bg_root,
            "grpo_fe_seed42_pipo",
            "abinitio_empirical_uncond_policy_fe_m1p5_n4096_seed42",
        ),
        (
            "GRPO+PIPO-BG",
            "bg",
            fe_bg_root,
            "grpo_bg_seed42_pipo",
            "abinitio_empirical_uncond_policy_bg_2_n4096_seed42",
        ),
        (
            "GRPO+PIPO-FE+BG",
            "joint",
            joint_root,
            "grpo_joint_seed42_pipo",
            "abinitio_empirical_uncond_policy_fe_m1p5_bg_2_n4096_seed42",
        ),
    ]
    specs = []
    for method, task, train_root, run_name, label in definitions:
        model_root = train_root / run_name / "model"
        results_root = train_root / run_name / "train_results"
        specs.append(
            {
                "method": method,
                "family": "pipo",
                "task": task,
                "run_name": run_name,
                "model_root": model_root,
                "results_root": results_root,
                "label": label,
                "generation": _generation(model_root, label, False),
                "predictions": _predictions(model_root, label),
                "structural_metrics": _structural_metrics(model_root, label),
            }
        )
    return specs


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    steps = [int(record["step"]) for record in records]
    if steps != list(range(EXPECTED_UPDATES)):
        raise RuntimeError(f"Expected unique steps 0..199 in {path}, got {steps[:3]}...{steps[-3:]}")
    return records


def _build_training_dynamics(rl_specs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for spec in rl_specs:
        records = _read_metrics(spec["results_root"] / "metrics.jsonl")
        for record in records:
            feedback = (record.get("pipo_feedback") or {}).get("feedback") or {}
            row = {
                "run_id": spec["run_name"],
                "method": spec["method"],
                "method_family": "pipo",
                "algorithm": "grpo",
                "task": spec["task"],
                "seed": SEED,
                "update_id": int(record["step"]),
                "reward_mean": float(record["reward"]["reward_mean"]),
                "reward_std": float(record["reward"]["reward_std"]),
                "valid_fraction": float(record["reward"]["valid_fraction"]),
                "advantage_mean": float(record["advantage"]["mean"]),
                "advantage_std": float(record["advantage"]["std"]),
                "loss": float(record["loss"]),
                "gradient_norm": float(record["gradient_norm"]),
                "approx_kl": float(record["approx_kl"]),
                "clip_fraction": float(record["clip_fraction"]),
                "ratio_mean": float(record["ratio_mean"]),
                "learning_rate": float(record["learning_rate"]),
                "pipo_history_signal": feedback.get("signal", np.nan),
                "pipo_modulation": feedback.get("modulation", np.nan),
                "pipo_retrospective_update": bool(
                    (record.get("pipo_feedback") or {}).get(
                        "retrospective_update", False
                    )
                ),
                "training_prompts": int(
                    record["sampling_contract"]["training_prompts_per_update"]
                ),
                "trajectories_per_prompt": int(
                    record["sampling_contract"]["training_trajectories_per_prompt"]
                ),
                "policy_microbatch_prompts": int(
                    record["sampling_contract"]["policy_microbatch_prompts"]
                ),
                "max_prompt_atoms": int(
                    record["sampling_contract"]["max_prompt_atoms"]
                ),
                "mean_orbits_per_structure": float(
                    np.mean(record["trajectory"]["num_orbits_per_structure"])
                ),
            }
            for channel in ("lattice", "coord", "atom"):
                channel_data = record["trajectory"]["channel_log_prob"][channel]
                row[f"{channel}_log_prob_mean"] = float(channel_data["mean"])
                row[f"{channel}_log_prob_std"] = float(channel_data["std"])
            rows.append(row)
    return pd.DataFrame(rows)


def _build_final_paired(rl_specs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for spec in rl_specs:
        payload = json.loads(
            (spec["results_root"] / "final_paired_evaluation.json").read_text(
                encoding="utf-8"
            )
        )
        decision = payload["decision"]
        baseline = payload["baseline"]
        final = payload["final"]
        row = {
            "run_id": spec["run_name"],
            "method": spec["method"],
            "task": spec["task"],
            "seed": SEED,
            "noise_seed": int(payload["noise_seed"]),
            "decision": decision["action"],
            "reward_mean_delta_vs_base": float(decision["mean_delta"][0]),
            "reward_lcb_vs_base": float(decision["lower_confidence_bound"][0]),
            "validity_mean_delta_vs_base": float(decision["mean_delta"][1]),
            "validity_lcb_vs_base": float(decision["lower_confidence_bound"][1]),
            "base_reward_mean": float(baseline["reward_mean"]),
            "final_reward_mean": float(final["reward_mean"]),
            "base_valid_fraction": float(baseline["valid_fraction"]),
            "final_valid_fraction": float(final["valid_fraction"]),
        }
        for name in ("formation_energy_per_atom", "band_gap"):
            row[f"base_{name}"] = baseline["property_means"].get(name, np.nan)
            row[f"final_{name}"] = final["property_means"].get(name, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def _property_frame(spec: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(spec["predictions"])
    if len(frame) != EXPECTED_SAMPLES:
        raise RuntimeError(f"Expected 4096 predictions in {spec['predictions']}")
    required = {
        "structure_index",
        "composition_valid",
        "structure_valid",
        "valid",
        "predicted_formation_energy_per_atom",
        "predicted_band_gap",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError(
            f"Missing columns {sorted(required - set(frame.columns))} in {spec['predictions']}"
        )
    result = frame[list(required)].copy()
    result["method"] = spec["method"]
    result["method_family"] = spec["family"]
    result["seed"] = SEED
    result["source_generation"] = str(spec["generation"].relative_to(ROOT))
    result["source_predictions"] = str(spec["predictions"].relative_to(ROOT))
    return result


def _build_property_data(
    baseline_specs: list[dict[str, Any]], rl_specs: list[dict[str, Any]]
) -> pd.DataFrame:
    train = pd.read_csv(
        TRAIN_CSV, usecols=["formation_energy_per_atom", "band_gap"]
    ).reset_index(names="structure_index")
    train["composition_valid"] = True
    train["structure_valid"] = True
    train["valid"] = True
    train = train.rename(
        columns={
            "formation_energy_per_atom": "predicted_formation_energy_per_atom",
            "band_gap": "predicted_band_gap",
        }
    )
    train["method"] = "Training set"
    train["method_family"] = "training"
    train["seed"] = SEED
    train["source_generation"] = str(TRAIN_CSV.relative_to(ROOT))
    train["source_predictions"] = "ground_truth"
    generated = [_property_frame(spec) for spec in baseline_specs + rl_specs]
    return pd.concat([train, *generated], ignore_index=True)


def _quality_tables(
    baseline_specs: list[dict[str, Any]],
    rl_specs: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = [
        next(spec for spec in baseline_specs if spec["method"] == "Base"),
        next(spec for spec in baseline_specs if spec["method"] == "CFG-FE+BG"),
        next(spec for spec in rl_specs if spec["task"] == "joint"),
    ]
    files = [
        QualityMethodFiles(
            method=spec["method"],
            generation=spec["generation"],
            predictions=spec["predictions"],
        )
        for spec in selected
    ]
    loaded = [_load_method(item) for item in files]
    reference, database_version = _mp_formation_reference()
    matcher = StructureMatcher(
        ltol=0.2,
        stol=0.3,
        angle_tol=5.0,
        primitive_cell=True,
        scale=True,
    )
    for item in loaded:
        item["mp_hull_formation_energy"], item["e_above_hull"] = (
            _mp_hull_distances(
                item["structures"],
                item["valid"],
                item["formation_energy"],
                reference,
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
    training_reference = _training_index(formulas, TRAIN_CSV)
    for item in loaded:
        item["novel"] = _novel_mask(
            item["structures"],
            item["representatives"],
            training_reference,
            matcher,
        )

    summary = pd.DataFrame(
        [
            _metric_row(
                method=item["files"].method,
                valid=item["valid"],
                stable=item["stable"],
                representatives=item["representatives"],
                novel=item["novel"],
                composition_valid=item["composition_valid"],
                structure_valid=item["structure_valid"],
                novelty_reference="mp_20_train",
            )
            for item in loaded
        ]
    )
    summary["stable_e_above_hull_max_eV_per_atom"] = STABLE_E_ABOVE_HULL_MAX
    summary["mp_database_version"] = database_version
    summary["formation_energy_evaluator"] = "independent_seed123"

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
                    "predicted_formation_energy_per_atom": item[
                        "formation_energy"
                    ],
                    "mp_hull_formation_energy_per_atom": item[
                        "mp_hull_formation_energy"
                    ],
                    "predicted_e_above_hull": item["e_above_hull"],
                    "stable": item["stable"],
                    "unique_representative": unique,
                    "novel_vs_mp20_train": item["novel"],
                    "sun": sun,
                }
            )
        )
    return summary, pd.concat(per_structure, ignore_index=True)


def _search_tradeoff(
    baseline_specs: list[dict[str, Any]], rl_specs: list[dict[str, Any]]
) -> pd.DataFrame:
    cached = pd.read_csv(BASELINE_DIVERSITY).set_index("method")
    rows = []
    for spec in baseline_specs + rl_specs:
        crystals, valid = _load_crystals(spec["generation"])
        predictions = (
            pd.read_csv(spec["predictions"])
            .set_index("structure_index")
            .reindex(range(len(crystals)))
        )
        hits = _valid_joint_target_mask(valid, predictions)
        if spec["method"] in cached.index:
            structure_diversity = float(
                cached.loc[spec["method"], "structure_diversity"]
            )
            composition_diversity = float(
                cached.loc[spec["method"], "composition_diversity"]
            )
            diversity_n = int(
                cached.loc[spec["method"], "n_diversity_fingerprints"]
            )
        else:
            (
                structure_diversity,
                composition_diversity,
                diversity_n,
            ) = _fingerprint_diversity(crystals, valid, spec["method"])
        rows.append(
            {
                "method": spec["method"],
                "method_family": spec["family"],
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
                "property_evaluator": "independent_seed123",
            }
        )
    return pd.DataFrame(rows)


def _search_space_tables(
    baseline_specs: list[dict[str, Any]], rl_specs: list[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = next(spec for spec in baseline_specs if spec["method"] == "Base")
    cfg = next(spec for spec in baseline_specs if spec["method"] == "CFG-FE+BG")
    pipo = next(spec for spec in rl_specs if spec["task"] == "joint")
    training_profiles, training_features, training_elements = _training_profiles()
    base_profiles, base_features, base_elements = _generated_profiles(
        "Base", base["generation"]
    )
    cfg_profiles, cfg_features, cfg_elements = _generated_profiles(
        "CFG", cfg["generation"]
    )
    pipo_profiles, pipo_features, pipo_elements = _generated_profiles(
        "GRPO+PIPO", pipo["generation"]
    )
    profiles = pd.DataFrame(
        training_profiles + base_profiles + cfg_profiles + pipo_profiles
    )
    elements = _element_frequencies(
        {
            "Training set": training_elements,
            "Base": base_elements,
            "CFG": cfg_elements,
            "GRPO+PIPO": pipo_elements,
        }
    )
    coordinates = _equal_tsne(
        {
            "Training set": training_features,
            "Base": base_features,
            "CFG": cfg_features,
            "GRPO+PIPO": pipo_features,
        }
    )
    return coordinates, profiles, elements


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(
    output_dir: Path,
    baseline_specs: list[dict[str, Any]],
    rl_specs: list[dict[str, Any]],
) -> None:
    paths = [TRAIN_CSV, BASELINE_DIVERSITY]
    for spec in baseline_specs + rl_specs:
        paths.extend(
            [
                spec["generation"],
                spec["predictions"],
                spec["structural_metrics"],
            ]
        )
    for spec in rl_specs:
        paths.extend(
            [
                spec["results_root"] / "metrics.jsonl",
                spec["results_root"] / "final_paired_evaluation.json",
            ]
        )
    records = [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(set(paths))
    ]
    (output_dir / "source_manifest.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )


def build(fe_bg_root: Path, joint_root: Path, output_dir: Path) -> None:
    baseline_specs = _baseline_specs()
    rl_specs = _rl_specs(fe_bg_root, joint_root)
    required = []
    for spec in baseline_specs + rl_specs:
        required.extend(
            [spec["generation"], spec["predictions"], spec["structural_metrics"]]
        )
    for spec in rl_specs:
        required.extend(
            [
                spec["results_root"] / "metrics.jsonl",
                spec["results_root"] / "final_paired_evaluation.json",
            ]
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required seed42 artifacts:\n" + "\n".join(missing))

    output_dir.mkdir(parents=True, exist_ok=False)
    dynamics = _build_training_dynamics(rl_specs)
    dynamics.to_csv(output_dir / "fig1_fig2_training_dynamics.csv", index=False)
    _build_final_paired(rl_specs).to_csv(
        output_dir / "fig2_final_paired_evaluation.csv", index=False
    )
    _build_property_data(baseline_specs, rl_specs).to_csv(
        output_dir / "fig3_property_distributions.csv", index=False
    )
    quality, quality_rows = _quality_tables(baseline_specs, rl_specs)
    quality.to_csv(output_dir / "fig3_quality_metrics.csv", index=False)
    quality_rows.to_csv(
        output_dir / "fig3_quality_per_structure.csv", index=False
    )
    _search_tradeoff(baseline_specs, rl_specs).to_csv(
        output_dir / "fig4_target_yield_diversity.csv", index=False
    )
    tsne, profiles, elements = _search_space_tables(baseline_specs, rl_specs)
    tsne.to_csv(output_dir / "fig4_tsne.csv", index=False)
    profiles.to_csv(output_dir / "fig4_structure_profiles.csv", index=False)
    elements.to_csv(output_dir / "fig4_element_frequencies.csv", index=False)
    contract = {
        "scope": "exploratory_seed42_single_seed",
        "method": "GRPO+PIPO",
        "method_is_crystalpirl": False,
        "seed": SEED,
        "updates_per_policy": EXPECTED_UPDATES,
        "formal_generation": "4096 ab_initio_empirical structures per policy",
        "policy_tasks": ["formation_energy", "band_gap", "joint"],
        "property_evaluator": "independent M3GNet seed123",
        "reward_predictor": "M3GNet seed42 used only during RL training",
        "band_gap_plot_filter": "finite values <= 10 eV; excluded counts must be reported",
        "stability": (
            "predicted formation energy compared with the MP 2026.04.13 "
            "GGA/GGA+U stable-entry formation-energy hull; stable if "
            "predicted E_hull <= 0.1 eV/atom"
        ),
        "claim_boundary": (
            "This package evaluates the original PIPO comparator. It cannot "
            "support CrystalPIRL paired-gate or multi-seed robustness claims."
        ),
    }
    (output_dir / "data_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    _write_manifest(output_dir, baseline_specs, rl_specs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fe-bg-root", type=Path, required=True)
    parser.add_argument("--joint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(
        args.fe_bg_root.resolve(),
        args.joint_root.resolve(),
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
