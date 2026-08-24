"""Freeze complete seed-42 Source Data for CrystalPIRL Fig. 1--Fig. 4."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.result_figures.crystalpirl_pipo_seed42 import build_source_data as pipo


ROOT = Path(__file__).resolve().parents[3]
SEED = 42
EXPECTED_UPDATES = 200
EXPECTED_SAMPLES = 4096


def _crystalpirl_specs(train_root: Path) -> list[dict[str, Any]]:
    definitions = [
        (
            "CrystalPIRL-FE",
            "fe",
            "grpo_fe_seed42_pirl",
            "abinitio_empirical_uncond_crystalpirl_fe_m1p5_n4096_seed42",
        ),
        (
            "CrystalPIRL-BG",
            "bg",
            "grpo_bg_seed42_pirl",
            "abinitio_empirical_uncond_crystalpirl_bg_2_n4096_seed42",
        ),
        (
            "CrystalPIRL-FE+BG",
            "joint",
            "grpo_joint_seed42_pirl",
            "abinitio_empirical_uncond_crystalpirl_fe_m1p5_bg_2_n4096_seed42",
        ),
    ]
    specs = []
    for method, task, run_name, label in definitions:
        model_root = train_root / run_name / "model"
        results_root = train_root / run_name / "train_results"
        specs.append(
            {
                "method": method,
                "family": "crystalpirl",
                "task": task,
                "run_name": run_name,
                "model_root": model_root,
                "results_root": results_root,
                "label": label,
                "generation": pipo._generation(model_root, label, False),
                "predictions": pipo._predictions(model_root, label),
                "structural_metrics": pipo._structural_metrics(model_root, label),
            }
        )
    return specs


def _crystalpirl_dynamics(specs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        records = pipo._read_metrics(spec["results_root"] / "metrics.jsonl")
        for record in records:
            decision = record["decision"]
            due = bool(decision["verification_due"])
            initial = decision.get("initial") or {}
            local = initial.get("local") or {}
            absolute = initial.get("absolute") or {}
            holdout = decision.get("holdout") or {}
            holdout_decision = holdout.get("decision") or {}
            holdout_local = holdout_decision.get("local") or {}
            holdout_absolute = holdout_decision.get("absolute") or {}
            row = {
                "run_id": spec["run_name"],
                "method": spec["method"],
                "method_family": "crystalpirl",
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
                "training_prompts": int(record["sampling_contract"]["training_prompts_per_update"]),
                "trajectories_per_prompt": int(record["sampling_contract"]["training_trajectories_per_prompt"]),
                "policy_microbatch_prompts": int(record["sampling_contract"]["policy_microbatch_prompts"]),
                "max_prompt_atoms": int(record["sampling_contract"]["max_prompt_atoms"]),
                "verification_interval": int(record["sampling_contract"]["pirl_verification_interval"]),
                "verification_due": due,
                "block_start_step": decision.get("block_start_step"),
                "block_end_step": decision.get("block_end_step"),
                "block_action": decision["action"],
                "verified_checkpoint_update": bool(decision["verified_checkpoint_update"]),
                "local_reward_delta": (local.get("mean_delta") or [np.nan])[0],
                "local_reward_lcb": (local.get("lower_confidence_bound") or [np.nan])[0],
                "absolute_reward_delta": (absolute.get("mean_delta") or [np.nan])[0],
                "absolute_reward_lcb": (absolute.get("lower_confidence_bound") or [np.nan])[0],
                "holdout_local_reward_delta": (holdout_local.get("mean_delta") or [np.nan])[0],
                "holdout_local_reward_lcb": (holdout_local.get("lower_confidence_bound") or [np.nan])[0],
                "holdout_absolute_reward_delta": (holdout_absolute.get("mean_delta") or [np.nan])[0],
                "holdout_absolute_reward_lcb": (holdout_absolute.get("lower_confidence_bound") or [np.nan])[0],
                "mean_orbits_per_structure": float(np.mean(record["trajectory"]["num_orbits_per_structure"])),
            }
            for channel in ("lattice", "coord", "atom"):
                channel_data = record["trajectory"]["channel_log_prob"][channel]
                row[f"{channel}_log_prob_mean"] = float(channel_data["mean"])
                row[f"{channel}_log_prob_std"] = float(channel_data["std"])
            rows.append(row)
    return pd.DataFrame(rows)


def _search_space_tables(
    baseline_specs: list[dict[str, Any]],
    pipo_specs: list[dict[str, Any]],
    crystalpirl_specs: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = next(spec for spec in baseline_specs if spec["method"] == "Base")
    cfg = next(spec for spec in baseline_specs if spec["method"] == "CFG-FE+BG")
    pipo_joint = next(spec for spec in pipo_specs if spec["task"] == "joint")
    crystal_joint = next(spec for spec in crystalpirl_specs if spec["task"] == "joint")
    training_profiles, training_features, training_elements = pipo._training_profiles()
    generated = {}
    for method, spec in (
        ("Base", base),
        ("CFG", cfg),
        ("GRPO+PIPO", pipo_joint),
        ("CrystalPIRL", crystal_joint),
    ):
        generated[method] = pipo._generated_profiles(method, spec["generation"])
    profiles = pd.DataFrame(
        training_profiles
        + [row for method in generated.values() for row in method[0]]
    )
    elements = pipo._element_frequencies(
        {
            "Training set": training_elements,
            **{name: values[2] for name, values in generated.items()},
        }
    )
    coordinates = pipo._equal_tsne(
        {
            "Training set": training_features,
            **{name: values[1] for name, values in generated.items()},
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
    policy_specs: list[dict[str, Any]],
) -> None:
    paths = [pipo.TRAIN_CSV, pipo.BASELINE_DIVERSITY]
    for spec in baseline_specs + policy_specs:
        paths.extend([spec["generation"], spec["predictions"], spec["structural_metrics"]])
    for spec in policy_specs:
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


def build(
    pipo_fe_bg_root: Path,
    pipo_joint_root: Path,
    crystalpirl_root: Path,
    output_dir: Path,
) -> None:
    baseline_specs = pipo._baseline_specs()
    pipo_specs = pipo._rl_specs(pipo_fe_bg_root, pipo_joint_root)
    crystalpirl_specs = _crystalpirl_specs(crystalpirl_root)
    policy_specs = pipo_specs + crystalpirl_specs
    required = []
    for spec in baseline_specs + policy_specs:
        required.extend([spec["generation"], spec["predictions"], spec["structural_metrics"]])
    for spec in policy_specs:
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
    pipo_dynamics = pipo._build_training_dynamics(pipo_specs)
    pipo_dynamics.to_csv(output_dir / "fig2_pipo_training_dynamics.csv", index=False)
    crystal_dynamics = _crystalpirl_dynamics(crystalpirl_specs)
    crystal_dynamics.to_csv(output_dir / "fig1_fig2_crystalpirl_dynamics.csv", index=False)
    crystal_dynamics[crystal_dynamics["verification_due"]].to_csv(
        output_dir / "fig1_fig2_crystalpirl_verification.csv", index=False
    )
    pd.concat(
        [
            pipo._build_final_paired(pipo_specs),
            pipo._build_final_paired(crystalpirl_specs),
        ],
        ignore_index=True,
    ).to_csv(output_dir / "fig2_final_paired_evaluation.csv", index=False)
    pipo._build_property_data(baseline_specs, policy_specs).to_csv(
        output_dir / "fig3_property_distributions.csv", index=False
    )
    quality, quality_rows = pipo._quality_tables(baseline_specs, crystalpirl_specs)
    quality.to_csv(output_dir / "fig3_quality_metrics.csv", index=False)
    quality_rows.to_csv(output_dir / "fig3_quality_per_structure.csv", index=False)
    pipo._search_tradeoff(baseline_specs, policy_specs).to_csv(
        output_dir / "fig4_target_yield_diversity.csv", index=False
    )
    tsne, profiles, elements = _search_space_tables(
        baseline_specs, pipo_specs, crystalpirl_specs
    )
    tsne.to_csv(output_dir / "fig4_tsne.csv", index=False)
    profiles.to_csv(output_dir / "fig4_structure_profiles.csv", index=False)
    elements.to_csv(output_dir / "fig4_element_frequencies.csv", index=False)
    contract = {
        "scope": "exploratory_seed42_single_seed",
        "main_method": "GRPO+CrystalPIRL blockwise paired dual-anchor gate",
        "comparator": "GRPO+PIPO sliding-history feedback",
        "seed": SEED,
        "updates_per_policy": EXPECTED_UPDATES,
        "crystalpirl_verification_interval": 10,
        "crystalpirl_probe_contract": "32 fixed plus 32 disjoint holdout prompts",
        "formal_generation": "4096 ab_initio_empirical structures per policy",
        "policy_tasks": ["formation_energy", "band_gap", "joint"],
        "property_evaluator": "independent M3GNet seed123",
        "reward_predictor": "M3GNet seed42 used only during RL training",
        "band_gap_plot_filter": "finite values <= 10 eV; excluded counts reported",
        "claim_boundary": (
            "Single-seed proxy-model evidence only. Accepted, rejected, attenuated, "
            "and rolled-back blocks are retained; no multi-seed, MLFF, or DFT claim."
        ),
    }
    (output_dir / "data_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    _write_manifest(output_dir, baseline_specs, policy_specs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipo-fe-bg-root", type=Path, required=True)
    parser.add_argument("--pipo-joint-root", type=Path, required=True)
    parser.add_argument("--crystalpirl-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(
        args.pipo_fe_bg_root.resolve(),
        args.pipo_joint_root.resolve(),
        args.crystalpirl_root.resolve(),
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
