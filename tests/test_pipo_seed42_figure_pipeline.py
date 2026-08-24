import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.result_figures.crystalpirl_pipo_seed42.audit_conclusions import audit
from scripts.result_figures.crystalpirl_pipo_seed42.build_source_data import _rl_specs


def test_rl_specs_keep_pipo_identity_and_resolve_three_tasks():
    specs = _rl_specs(Path("/tmp/fe_bg"), Path("/tmp/joint"))

    assert [spec["task"] for spec in specs] == ["fe", "bg", "joint"]
    assert [spec["method"] for spec in specs] == [
        "GRPO+PIPO-FE",
        "GRPO+PIPO-BG",
        "GRPO+PIPO-FE+BG",
    ]
    assert all("CrystalPIRL" not in spec["method"] for spec in specs)
    assert specs[0]["model_root"] == Path("/tmp/fe_bg/grpo_fe_seed42_pipo/model")
    assert specs[2]["model_root"] == Path(
        "/tmp/joint/grpo_joint_seed42_pipo/model"
    )


def test_conclusion_audit_authorizes_complete_pipo_data(tmp_path):
    methods = ["GRPO+PIPO-FE", "GRPO+PIPO-BG", "GRPO+PIPO-FE+BG"]
    dynamics = []
    for method, task in zip(methods, ["fe", "bg", "joint"]):
        for step in range(200):
            dynamics.append(
                {
                    "method": method,
                    "task": task,
                    "update_id": step,
                    "training_prompts": 4,
                    "trajectories_per_prompt": 16,
                    "policy_microbatch_prompts": 1,
                    "max_prompt_atoms": 60,
                    "reward_mean": 0.01 * step,
                    "reward_std": 0.2,
                    "advantage_mean": 0.0,
                    "advantage_std": 1.0,
                    "gradient_norm": 1.0,
                    "loss": 0.0,
                    "pipo_history_signal": np.nan if step < 8 else 0.1,
                }
            )
    pd.DataFrame(dynamics).to_csv(
        tmp_path / "fig1_fig2_training_dynamics.csv", index=False
    )
    pd.DataFrame(
        {
            "method": methods,
            "decision": ["accept", "reject", "attenuate"],
            "reward_mean_delta_vs_base": [0.1, -0.1, 0.05],
            "reward_lcb_vs_base": [0.01, -0.2, -0.01],
            "validity_mean_delta_vs_base": [0.0, 0.0, 0.0],
            "validity_lcb_vs_base": [0.0, 0.0, 0.0],
        }
    ).to_csv(tmp_path / "fig2_final_paired_evaluation.csv", index=False)

    property_methods = [
        "Training set",
        "Base",
        "CFG-FE",
        "CFG-BG",
        "CFG-FE+BG",
        *methods,
    ]
    property_rows = []
    for method in property_methods:
        count = 4 if method == "Training set" else 4096
        property_rows.extend(
            {
                "method": method,
                "source_predictions": (
                    "ground_truth" if method == "Training set" else "predictor_seed123.csv"
                ),
            }
            for _ in range(count)
        )
    pd.DataFrame(property_rows).to_csv(
        tmp_path / "fig3_property_distributions.csv", index=False
    )

    quality_methods = ["Base", "CFG-FE+BG", "GRPO+PIPO-FE+BG"]
    pd.DataFrame(
        {
            "method": quality_methods,
            "compositional_validity": [0.8] * 3,
            "structural_validity": [0.9] * 3,
            "uniqueness": [0.95] * 3,
            "novelty_among_unique": [0.96] * 3,
            "sun": [0.3] * 3,
            "mp_hull_stability": [0.32] * 3,
        }
    ).to_csv(tmp_path / "fig3_quality_metrics.csv", index=False)
    pd.DataFrame(
        {
            "method": np.repeat(quality_methods, 4096),
            "structure_index": np.tile(np.arange(4096), 3),
        }
    ).to_csv(tmp_path / "fig3_quality_per_structure.csv", index=False)

    tradeoff_methods = [
        "Base",
        "CFG-FE",
        "CFG-BG",
        "CFG-FE+BG",
        *methods,
    ]
    pd.DataFrame(
        {
            "method": tradeoff_methods,
            "valid_joint_target_yield": [0.1] * 7,
            "structure_diversity": [0.6] * 7,
        }
    ).to_csv(tmp_path / "fig4_target_yield_diversity.csv", index=False)

    search_methods = ["Training set", "Base", "CFG", "GRPO+PIPO"]
    pd.DataFrame(
        {
            "method": np.repeat(search_methods, 3),
            "tsne_1": np.arange(12),
            "tsne_2": np.arange(12),
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
                "method": "GRPO+PIPO",
                "method_is_crystalpirl": False,
                "claim_boundary": "PIPO comparator only",
            }
        )
    )

    report = audit(tmp_path)

    assert report["plot_authorized"] is True
    assert report["scope"] == "exploratory_seed42_single_seed"
