"""Audit complete seed-42 CrystalPIRL Source Data before plotting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_UPDATES = 200
EXPECTED_SAMPLES = 4096
PIPO_METHODS = {"GRPO+PIPO-FE", "GRPO+PIPO-BG", "GRPO+PIPO-FE+BG"}
CRYSTAL_METHODS = {"CrystalPIRL-FE", "CrystalPIRL-BG", "CrystalPIRL-FE+BG"}
BASELINE_METHODS = {"Base", "CFG-FE", "CFG-BG", "CFG-FE+BG"}
PROPERTY_METHODS = {"Training set", *BASELINE_METHODS, *PIPO_METHODS, *CRYSTAL_METHODS}
QUALITY_METHODS = {"Base", "CFG-FE+BG", "CrystalPIRL-FE+BG"}
SEARCH_METHODS = {"Training set", "Base", "CFG", "GRPO+PIPO", "CrystalPIRL"}


def _check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "pass": bool(passed), "detail": detail}


def _finite(frame: pd.DataFrame, columns: list[str]) -> bool:
    return bool(
        not frame.empty
        and np.isfinite(frame[columns].to_numpy(dtype=float)).all()
    )


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def _update_contract(frame: pd.DataFrame, methods: set[str]) -> tuple[bool, dict[str, int]]:
    counts = frame.groupby("method")["update_id"].nunique().to_dict()
    sequences = all(
        group.sort_values("update_id")["update_id"].tolist()
        == list(range(EXPECTED_UPDATES))
        for _, group in frame.groupby("method")
    )
    return set(frame["method"]) == methods and sequences and all(
        count == EXPECTED_UPDATES for count in counts.values()
    ), counts


def audit(source_dir: Path) -> dict[str, Any]:
    contract = json.loads((source_dir / "data_contract.json").read_text())
    pipo = pd.read_csv(source_dir / "fig2_pipo_training_dynamics.csv")
    crystal = pd.read_csv(source_dir / "fig1_fig2_crystalpirl_dynamics.csv")
    verification = pd.read_csv(source_dir / "fig1_fig2_crystalpirl_verification.csv")
    paired = pd.read_csv(source_dir / "fig2_final_paired_evaluation.csv")
    properties = pd.read_csv(source_dir / "fig3_property_distributions.csv")
    quality = pd.read_csv(source_dir / "fig3_quality_metrics.csv")
    quality_rows = pd.read_csv(source_dir / "fig3_quality_per_structure.csv")
    tradeoff = pd.read_csv(source_dir / "fig4_target_yield_diversity.csv")
    tsne = pd.read_csv(source_dir / "fig4_tsne.csv")
    profiles = pd.read_csv(source_dir / "fig4_structure_profiles.csv")
    elements = pd.read_csv(source_dir / "fig4_element_frequencies.csv")

    pipo_updates_ok, pipo_counts = _update_contract(pipo, PIPO_METHODS)
    crystal_updates_ok, crystal_counts = _update_contract(crystal, CRYSTAL_METHODS)
    boundary_counts = verification.groupby("method")["update_id"].nunique().to_dict()
    boundary_action_counts = {
        f"{method}|{action}": int(count)
        for (method, action), count in verification.groupby(
            ["method", "block_action"]
        ).size().items()
    }
    boundary_sequences = all(
        group.sort_values("update_id")["update_id"].tolist() == list(range(9, 200, 10))
        for _, group in verification.groupby("method")
    )
    generated_counts = properties.groupby("method").size().drop("Training set")
    quality_counts = quality_rows.groupby("method").size()
    tsne_counts = tsne.groupby("method").size()
    source_predictions = properties.loc[
        properties["method"] != "Training set", "source_predictions"
    ].astype(str)

    checks = [
        _check("seed contract", contract.get("seed") == 42, contract.get("seed")),
        _check(
            "method identity",
            "CrystalPIRL" in contract.get("main_method", "")
            and "PIPO" in contract.get("comparator", ""),
            {"main": contract.get("main_method"), "comparator": contract.get("comparator")},
        ),
        _check("PIPO 200-update matrix", pipo_updates_ok, pipo_counts),
        _check("CrystalPIRL 200-update matrix", crystal_updates_ok, crystal_counts),
        _check(
            "training sampling contract",
            bool(
                (pipo["training_prompts"] == 4).all()
                and (pipo["trajectories_per_prompt"] == 16).all()
                and (pipo["policy_microbatch_prompts"] == 1).all()
                and (pipo["max_prompt_atoms"] <= 60).all()
                and (crystal["training_prompts"] == 4).all()
                and (crystal["trajectories_per_prompt"] == 16).all()
                and (crystal["policy_microbatch_prompts"] == 1).all()
                and (crystal["max_prompt_atoms"] <= 60).all()
            ),
            "4 prompts x 16 trajectories; prompt microbatch 1; <=60 atoms",
        ),
        _check(
            "finite policy signals",
            _finite(pipo, ["reward_mean", "advantage_mean", "gradient_norm", "loss"])
            and _finite(crystal, ["reward_mean", "advantage_mean", "gradient_norm", "loss"]),
            "reward, GRPO advantage, gradient norm and loss",
        ),
        _check(
            "20 CrystalPIRL block verifications per task",
            set(verification["method"]) == CRYSTAL_METHODS
            and boundary_sequences
            and all(value == 20 for value in boundary_counts.values()),
            boundary_counts,
        ),
        _check(
            "finite dual-anchor LCB evidence",
            _finite(
                verification,
                [
                    "local_reward_delta",
                    "local_reward_lcb",
                    "absolute_reward_delta",
                    "absolute_reward_lcb",
                ],
            ),
            "local and frozen-base paired deltas and LCBs",
        ),
        _check(
            "auditable boundary actions",
            set(verification["block_action"]).issubset({"accept", "reject", "rollback"}),
            boundary_action_counts,
        ),
        _check(
            "six final paired evaluations",
            len(paired) == 6
            and set(paired["method"]) == PIPO_METHODS | CRYSTAL_METHODS
            and _finite(
                paired,
                [
                    "reward_mean_delta_vs_base",
                    "reward_lcb_vs_base",
                    "validity_mean_delta_vs_base",
                    "validity_lcb_vs_base",
                ],
            ),
            paired[["method", "decision"]].to_dict("records"),
        ),
        _check("property method matrix", set(properties["method"]) == PROPERTY_METHODS, sorted(properties["method"].unique())),
        _check("4096 generated structures per method", bool((generated_counts == EXPECTED_SAMPLES).all()), generated_counts.to_dict()),
        _check("independent seed123 property evaluation", bool(source_predictions.str.contains("seed123").all()), "all generated structures"),
        _check("quality method matrix", set(quality["method"]) == QUALITY_METHODS, sorted(quality["method"].unique())),
        _check("quality per-structure coverage", bool((quality_counts == EXPECTED_SAMPLES).all()), quality_counts.to_dict()),
        _check(
            "finite quality summaries",
            _finite(
                quality,
                [
                    "sun",
                    "uniqueness",
                    "novelty_among_unique",
                    "mp_hull_stability",
                    "compositional_validity",
                    "structural_validity",
                ],
            ),
            "Base, joint CFG and joint CrystalPIRL",
        ),
        _check(
            "target-yield diversity matrix",
            set(tradeoff["method"]) == BASELINE_METHODS | PIPO_METHODS | CRYSTAL_METHODS,
            sorted(tradeoff["method"].unique()),
        ),
        _check(
            "finite target-yield diversity",
            _finite(tradeoff, ["valid_joint_target_yield", "structure_diversity"]),
            "all generated policy families",
        ),
        _check(
            "equal t-SNE groups",
            set(tsne["method"]) == SEARCH_METHODS and tsne_counts.nunique() == 1,
            tsne_counts.to_dict(),
        ),
        _check(
            "search-space profile groups",
            set(profiles["method"]) == SEARCH_METHODS and set(elements["method"]) == SEARCH_METHODS,
            {"profiles": sorted(profiles["method"].unique()), "elements": sorted(elements["method"].unique())},
        ),
        _check("source manifest present", (source_dir / "source_manifest.json").is_file(), "SHA-256 provenance"),
    ]
    authorized = all(item["pass"] for item in checks)
    action_counts = (
        verification.groupby(["method", "block_action"])
        .size()
        .rename("count")
        .reset_index()
        .to_dict("records")
    )
    reward_shifts = []
    for family, frame in (("PIPO", pipo), ("CrystalPIRL", crystal)):
        for method, group in frame.groupby("method"):
            ordered = group.sort_values("update_id")
            reward_shifts.append(
                {
                    "family": family,
                    "method": method,
                    "first_20_reward_mean": float(ordered.head(20)["reward_mean"].mean()),
                    "last_20_reward_mean": float(ordered.tail(20)["reward_mean"].mean()),
                    "last_minus_first": float(
                        ordered.tail(20)["reward_mean"].mean()
                        - ordered.head(20)["reward_mean"].mean()
                    ),
                }
            )
    claim_boundaries = [
        "Seed 42 is exploratory evidence and cannot support robustness or statistical-significance claims.",
        "PIPO and CrystalPIRL remain separate method families; PIPO results are never relabeled as CrystalPIRL.",
        "A rejected or rolled-back block is a valid result and is retained rather than hidden.",
        "Independent seed123 M3GNet predictions are proxy-model evaluation, not MLFF-relaxed or DFT ground truth.",
        "Stability is a predicted formation-energy hull metric and not relaxed thermodynamic stability.",
        "t-SNE is qualitative; target yield and deterministic fingerprint diversity carry the quantitative trade-off evidence.",
    ]
    report = {
        "scope": "exploratory_seed42_single_seed",
        "plot_authorized": authorized,
        "checks": checks,
        "crystalpirl_boundary_actions": action_counts,
        "training_reward_shifts": reward_shifts,
        "final_paired_evaluation": _records(paired),
        "quality_summary": _records(quality),
        "target_yield_diversity": _records(tradeoff),
        "claim_boundaries": claim_boundaries,
    }
    (source_dir / "conclusion_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Seed=42 CrystalPIRL 数据与结论审计",
        "",
        f"- 绘图授权：**{'PASS' if authorized else 'FAIL'}**",
        "- 主方法：GRPO+CrystalPIRL 分块双锚点验证。",
        "- 对照：GRPO+PIPO；不得重标为 CrystalPIRL。",
        "- 统计范围：单一训练与生成种子 42。",
        "",
        "## 完整性检查",
        "",
        *[
            f"- [{'x' if item['pass'] else ' '}] {item['name']}: {item['detail']}"
            for item in checks
        ],
        "",
        "## 结论边界",
        "",
        *[f"- {item}" for item in claim_boundaries],
    ]
    (source_dir / "conclusion_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not authorized:
        raise RuntimeError(
            "Conclusion audit failed: "
            + ", ".join(item["name"] for item in checks if not item["pass"])
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    audit(args.source_dir.resolve())


if __name__ == "__main__":
    main()
