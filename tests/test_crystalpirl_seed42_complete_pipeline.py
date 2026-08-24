import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.result_figures.crystalpirl_seed42_complete.audit_conclusions import (
    audit,
)
from scripts.result_figures.crystalpirl_seed42_complete.build_source_data import (
    _crystalpirl_dynamics,
    _crystalpirl_specs,
)
from scripts.result_figures.crystalpirl_seed42_complete.plot_figures import plot


def test_crystalpirl_specs_keep_method_identity_and_three_tasks():
    specs = _crystalpirl_specs(Path("/tmp/crystal"))

    assert [spec["task"] for spec in specs] == ["fe", "bg", "joint"]
    assert [spec["method"] for spec in specs] == [
        "CrystalPIRL-FE",
        "CrystalPIRL-BG",
        "CrystalPIRL-FE+BG",
    ]
    assert all(spec["family"] == "crystalpirl" for spec in specs)
    assert specs[2]["model_root"] == Path(
        "/tmp/crystal/grpo_joint_seed42_pirl/model"
    )


def test_crystalpirl_dynamics_extracts_dual_anchor_boundary(monkeypatch):
    record = {
        "step": 9,
        "reward": {"reward_mean": 0.2, "reward_std": 0.1, "valid_fraction": 0.9},
        "advantage": {"mean": 0.0, "std": 1.0},
        "loss": -0.01,
        "gradient_norm": 1.2,
        "approx_kl": 0.001,
        "clip_fraction": 0.0,
        "ratio_mean": 1.0,
        "learning_rate": 1.0e-6,
        "sampling_contract": {
            "training_prompts_per_update": 4,
            "training_trajectories_per_prompt": 16,
            "policy_microbatch_prompts": 1,
            "max_prompt_atoms": 52,
            "pirl_verification_interval": 10,
        },
        "trajectory": {
            "num_orbits_per_structure": [2, 3],
            "channel_log_prob": {
                channel: {"mean": 0.1, "std": 0.2}
                for channel in ("lattice", "coord", "atom")
            },
        },
        "decision": {
            "verification_due": True,
            "block_start_step": 0,
            "block_end_step": 9,
            "action": "accept",
            "verified_checkpoint_update": True,
            "initial": {
                "local": {
                    "mean_delta": [0.2],
                    "lower_confidence_bound": [0.1],
                },
                "absolute": {
                    "mean_delta": [0.3],
                    "lower_confidence_bound": [0.15],
                },
            },
            "holdout": {
                "decision": {
                    "local": {
                        "mean_delta": [0.18],
                        "lower_confidence_bound": [0.08],
                    },
                    "absolute": {
                        "mean_delta": [0.28],
                        "lower_confidence_bound": [0.12],
                    },
                }
            },
        },
    }
    monkeypatch.setattr(
        "scripts.result_figures.crystalpirl_seed42_complete.build_source_data.pipo._read_metrics",
        lambda _: [record],
    )

    row = _crystalpirl_dynamics(_crystalpirl_specs(Path("/tmp/crystal"))[:1]).iloc[0]

    assert row["local_reward_lcb"] == 0.1
    assert row["absolute_reward_lcb"] == 0.15
    assert row["holdout_local_reward_lcb"] == 0.08
    assert row["holdout_absolute_reward_lcb"] == 0.12
    assert bool(row["verified_checkpoint_update"])


def test_complete_audit_authorizes_full_single_seed_matrix(tmp_path):
    pipo_methods = ["GRPO+PIPO-FE", "GRPO+PIPO-BG", "GRPO+PIPO-FE+BG"]
    crystal_methods = ["CrystalPIRL-FE", "CrystalPIRL-BG", "CrystalPIRL-FE+BG"]
    tasks = ["fe", "bg", "joint"]
    common = {
        "training_prompts": 4,
        "trajectories_per_prompt": 16,
        "policy_microbatch_prompts": 1,
        "max_prompt_atoms": 52,
        "reward_mean": 0.1,
        "advantage_mean": 0.0,
        "gradient_norm": 1.0,
        "loss": 0.0,
    }
    pipo_rows = [
        {"method": method, "task": task, "update_id": step, **common}
        for method, task in zip(pipo_methods, tasks)
        for step in range(200)
    ]
    pd.DataFrame(pipo_rows).to_csv(
        tmp_path / "fig2_pipo_training_dynamics.csv", index=False
    )
    crystal_rows = [
        {
            "method": method,
            "task": task,
            "update_id": step,
            "verification_due": (step + 1) % 10 == 0,
            "block_action": "reject" if (step + 1) % 10 == 0 else "pending",
            "local_reward_delta": 0.1 if (step + 1) % 10 == 0 else np.nan,
            "local_reward_lcb": 0.01 if (step + 1) % 10 == 0 else np.nan,
            "absolute_reward_delta": 0.1 if (step + 1) % 10 == 0 else np.nan,
            "absolute_reward_lcb": 0.01 if (step + 1) % 10 == 0 else np.nan,
            **common,
        }
        for method, task in zip(crystal_methods, tasks)
        for step in range(200)
    ]
    crystal = pd.DataFrame(crystal_rows)
    crystal.to_csv(tmp_path / "fig1_fig2_crystalpirl_dynamics.csv", index=False)
    crystal[crystal["verification_due"]].to_csv(
        tmp_path / "fig1_fig2_crystalpirl_verification.csv", index=False
    )

    all_policy_methods = pipo_methods + crystal_methods
    pd.DataFrame(
        {
            "method": all_policy_methods,
            "decision": ["accept", "reject", "attenuate"] * 2,
            "reward_mean_delta_vs_base": [0.1] * 6,
            "reward_lcb_vs_base": [0.01] * 6,
            "validity_mean_delta_vs_base": [0.0] * 6,
            "validity_lcb_vs_base": [0.0] * 6,
        }
    ).to_csv(tmp_path / "fig2_final_paired_evaluation.csv", index=False)

    generated_methods = [
        "Base",
        "CFG-FE",
        "CFG-BG",
        "CFG-FE+BG",
        *all_policy_methods,
    ]
    property_rows = [
        {"method": "Training set", "source_predictions": "ground_truth"}
    ]
    for method in generated_methods:
        property_rows.extend(
            {"method": method, "source_predictions": "predictor_seed123.csv"}
            for _ in range(4096)
        )
    pd.DataFrame(property_rows).to_csv(
        tmp_path / "fig3_property_distributions.csv", index=False
    )

    quality_methods = ["Base", "CFG-FE+BG", "CrystalPIRL-FE+BG"]
    pd.DataFrame(
        {
            "method": quality_methods,
            "sun": [0.3] * 3,
            "uniqueness": [0.95] * 3,
            "novelty_among_unique": [0.96] * 3,
            "mp_hull_stability": [0.32] * 3,
            "compositional_validity": [0.8] * 3,
            "structural_validity": [0.9] * 3,
        }
    ).to_csv(tmp_path / "fig3_quality_metrics.csv", index=False)
    pd.DataFrame(
        {
            "method": np.repeat(quality_methods, 4096),
            "structure_index": np.tile(np.arange(4096), 3),
        }
    ).to_csv(tmp_path / "fig3_quality_per_structure.csv", index=False)

    pd.DataFrame(
        {
            "method": generated_methods,
            "valid_joint_target_yield": [0.1] * len(generated_methods),
            "structure_diversity": [0.6] * len(generated_methods),
        }
    ).to_csv(tmp_path / "fig4_target_yield_diversity.csv", index=False)
    search_methods = ["Training set", "Base", "CFG", "GRPO+PIPO", "CrystalPIRL"]
    pd.DataFrame(
        {
            "method": np.repeat(search_methods, 3),
            "tsne_1": np.arange(15),
            "tsne_2": np.arange(15),
        }
    ).to_csv(tmp_path / "fig4_tsne.csv", index=False)
    pd.DataFrame({"method": search_methods}).to_csv(
        tmp_path / "fig4_structure_profiles.csv", index=False
    )
    pd.DataFrame({"method": search_methods}).to_csv(
        tmp_path / "fig4_element_frequencies.csv", index=False
    )
    (tmp_path / "source_manifest.json").write_text("[]\n")
    (tmp_path / "data_contract.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "main_method": "GRPO+CrystalPIRL blockwise gate",
                "comparator": "GRPO+PIPO",
            }
        )
    )

    report = audit(tmp_path)

    assert report["plot_authorized"] is True
    assert len(report["crystalpirl_boundary_actions"]) == 3


def test_complete_plot_renders_png_bundle(tmp_path):
    source = tmp_path / "source"
    assets = tmp_path / "assets"
    source.mkdir()
    tasks = ["fe", "bg", "joint"]
    pipo_rows = []
    crystal_rows = []
    verification_rows = []
    for task_index, task in enumerate(tasks):
        for step in range(20):
            common = {
                "task": task,
                "update_id": step,
                "reward_mean": 0.01 * step + 0.03 * task_index,
                "advantage_mean": 1.0e-9 * np.sin(step),
                "gradient_norm": 0.5 + 0.02 * step,
            }
            pipo_rows.append(common)
            crystal_rows.append(common)
        for step, action in ((9, "reject"), (19, "accept")):
            verification_rows.append(
                {
                    "task": task,
                    "update_id": step,
                    "block_action": action,
                    "local_reward_lcb": -0.02 if action == "reject" else 0.02,
                    "absolute_reward_lcb": -0.03 if action == "reject" else 0.01,
                }
            )
    pd.DataFrame(pipo_rows).to_csv(source / "fig2_pipo_training_dynamics.csv", index=False)
    pd.DataFrame(crystal_rows).to_csv(source / "fig1_fig2_crystalpirl_dynamics.csv", index=False)
    pd.DataFrame(verification_rows).to_csv(source / "fig1_fig2_crystalpirl_verification.csv", index=False)

    rng = np.random.default_rng(42)
    property_rows = []
    for method, fe_center, bg_center in (
        ("Training set", -1.0, 1.0),
        ("Base", -0.7, 0.7),
        ("CFG-FE", -1.3, 0.8),
        ("CFG-BG", -0.8, 1.8),
        ("CFG-FE+BG", -1.2, 1.7),
        ("CrystalPIRL-FE", -1.4, 0.9),
        ("CrystalPIRL-BG", -0.9, 1.9),
        ("CrystalPIRL-FE+BG", -1.4, 1.9),
    ):
        for fe, bg in zip(
            rng.normal(fe_center, 0.25, 80), rng.normal(bg_center, 0.35, 80)
        ):
            property_rows.append(
                {
                    "method": method,
                    "predicted_formation_energy_per_atom": fe,
                    "predicted_band_gap": max(0.0, bg),
                }
            )
    pd.DataFrame(property_rows).to_csv(source / "fig3_property_distributions.csv", index=False)
    pd.DataFrame(
        {
            "method": ["Base", "CFG-FE+BG", "CrystalPIRL-FE+BG"],
            "sun": [0.3, 0.25, 0.34],
            "uniqueness": [0.98, 0.97, 0.98],
            "novelty_among_unique": [0.96, 0.97, 0.98],
            "mp_hull_stability": [0.32, 0.27, 0.35],
            "compositional_validity": [0.8, 0.81, 0.82],
            "structural_validity": [0.95, 0.94, 0.95],
        }
    ).to_csv(source / "fig3_quality_metrics.csv", index=False)

    tradeoff = [
        ("Base", "base"),
        ("CFG-FE", "cfg"),
        ("CFG-BG", "cfg"),
        ("CFG-FE+BG", "cfg"),
        ("GRPO+PIPO-FE+BG", "pipo"),
        ("CrystalPIRL-FE+BG", "crystalpirl"),
    ]
    pd.DataFrame(
        {
            "method": [item[0] for item in tradeoff],
            "method_family": [item[1] for item in tradeoff],
            "structure_diversity": np.linspace(0.60, 0.70, len(tradeoff)),
            "valid_joint_target_yield": np.linspace(0.03, 0.12, len(tradeoff)),
        }
    ).to_csv(source / "fig4_target_yield_diversity.csv", index=False)

    methods = ["Training set", "Base", "CFG", "GRPO+PIPO", "CrystalPIRL"]
    tsne_rows = []
    profile_rows = []
    element_rows = []
    for method_index, method in enumerate(methods):
        for index in range(50):
            tsne_rows.append(
                {
                    "method": method,
                    "tsne_1": rng.normal(method_index, 0.8),
                    "tsne_2": rng.normal(-method_index, 0.8),
                }
            )
            profile_rows.append(
                {
                    "method": method,
                    "n_elements": 1 + index % 5,
                    "space_group": 1 + index % 10,
                    "num_atoms": 2 + index % 24,
                    "density_g_cm3": 1.0 + 0.1 * method_index + rng.random() * 5,
                }
            )
        fractions = rng.random(12)
        fractions /= fractions.sum()
        for index, fraction in enumerate(fractions):
            element_rows.append(
                {
                    "method": method,
                    "element": f"E{index}",
                    "atom_fraction": fraction,
                }
            )
    pd.DataFrame(tsne_rows).to_csv(source / "fig4_tsne.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(source / "fig4_structure_profiles.csv", index=False)
    pd.DataFrame(element_rows).to_csv(source / "fig4_element_frequencies.csv", index=False)
    (source / "conclusion_audit.json").write_text(
        json.dumps({"plot_authorized": True, "claim_boundaries": ["seed42 only"]})
    )

    plot(source, assets)

    assert {path.name for path in assets.glob("*.png")} == {
        "fig1_crystalpirl_method.png",
        "fig2_verified_policy_improvement.png",
        "fig3_fe_bg_control.png",
        "fig4_quality_search_space.png",
    }
    assert all(path.stat().st_size > 0 for path in assets.glob("*.png"))
