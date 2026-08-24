"""Audit completeness and claim boundaries before seed-42 plotting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_UPDATES = 200
EXPECTED_SAMPLES = 4096
TRAINING_METHODS = {
    "GRPO+PIPO-FE",
    "GRPO+PIPO-BG",
    "GRPO+PIPO-FE+BG",
}
PROPERTY_METHODS = {
    "Training set",
    "Base",
    "CFG-FE",
    "CFG-BG",
    "CFG-FE+BG",
    *TRAINING_METHODS,
}
QUALITY_METHODS = {"Base", "CFG-FE+BG", "GRPO+PIPO-FE+BG"}
SEARCH_METHODS = {"Training set", "Base", "CFG", "GRPO+PIPO"}


def _finite(frame: pd.DataFrame, columns: list[str]) -> bool:
    if frame.empty:
        return False
    return bool(np.isfinite(frame[columns].to_numpy(dtype=float)).all())


def _check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "pass": bool(passed), "detail": detail}


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def audit(source_dir: Path) -> dict[str, Any]:
    contract = json.loads((source_dir / "data_contract.json").read_text())
    dynamics = pd.read_csv(source_dir / "fig1_fig2_training_dynamics.csv")
    paired = pd.read_csv(source_dir / "fig2_final_paired_evaluation.csv")
    properties = pd.read_csv(source_dir / "fig3_property_distributions.csv")
    quality = pd.read_csv(source_dir / "fig3_quality_metrics.csv")
    quality_rows = pd.read_csv(source_dir / "fig3_quality_per_structure.csv")
    tradeoff = pd.read_csv(source_dir / "fig4_target_yield_diversity.csv")
    tsne = pd.read_csv(source_dir / "fig4_tsne.csv")
    profiles = pd.read_csv(source_dir / "fig4_structure_profiles.csv")
    elements = pd.read_csv(source_dir / "fig4_element_frequencies.csv")

    update_counts = dynamics.groupby("method")["update_id"].nunique()
    update_sequences = all(
        frame["update_id"].tolist() == list(range(EXPECTED_UPDATES))
        for _, frame in dynamics.sort_values("update_id").groupby("method")
    )
    property_counts = properties.groupby("method").size()
    generated_counts = property_counts.drop("Training set")
    quality_counts = quality_rows.groupby("method").size()
    tsne_counts = tsne.groupby("method").size()
    profile_methods = set(profiles["method"])
    element_methods = set(elements["method"])
    forbidden_labels = [
        value
        for frame, column in (
            (dynamics, "method"),
            (properties, "method"),
            (quality, "method"),
            (tradeoff, "method"),
        )
        for value in frame[column].astype(str).unique()
        if "crystalpirl" in value.lower()
    ]

    checks = [
        _check("seed contract", contract.get("seed") == 42, contract.get("seed")),
        _check(
            "method boundary",
            contract.get("method") == "GRPO+PIPO"
            and contract.get("method_is_crystalpirl") is False,
            contract.get("claim_boundary"),
        ),
        _check(
            "no false CrystalPIRL labels",
            not forbidden_labels,
            forbidden_labels,
        ),
        _check(
            "three training policies",
            set(dynamics["method"]) == TRAINING_METHODS,
            sorted(dynamics["method"].unique()),
        ),
        _check(
            "200 unique updates per policy",
            bool((update_counts == EXPECTED_UPDATES).all() and update_sequences),
            update_counts.to_dict(),
        ),
        _check(
            "training sampling contract",
            bool(
                (dynamics["training_prompts"] == 4).all()
                and (dynamics["trajectories_per_prompt"] == 16).all()
                and (dynamics["policy_microbatch_prompts"] == 1).all()
                and (dynamics["max_prompt_atoms"] <= 60).all()
            ),
            {
                "prompts": sorted(dynamics["training_prompts"].unique().tolist()),
                "trajectories": sorted(
                    dynamics["trajectories_per_prompt"].unique().tolist()
                ),
                "microbatch_prompts": sorted(
                    dynamics["policy_microbatch_prompts"].unique().tolist()
                ),
                "max_prompt_atoms": int(dynamics["max_prompt_atoms"].max()),
            },
        ),
        _check(
            "finite training signals",
            _finite(
                dynamics,
                [
                    "reward_mean",
                    "reward_std",
                    "advantage_mean",
                    "advantage_std",
                    "gradient_norm",
                    "loss",
                ],
            ),
            "reward, advantage, gradient and loss",
        ),
        _check(
            "PIPO feedback after warm-up",
            bool(
                dynamics.groupby("method")["pipo_history_signal"]
                .apply(lambda values: values.notna().sum() >= 192)
                .all()
            ),
            dynamics.groupby("method")["pipo_history_signal"]
            .apply(lambda values: int(values.notna().sum()))
            .to_dict(),
        ),
        _check(
            "three final paired evaluations",
            len(paired) == 3
            and set(paired["method"]) == TRAINING_METHODS
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
        _check(
            "property method matrix",
            set(properties["method"]) == PROPERTY_METHODS,
            sorted(properties["method"].unique()),
        ),
        _check(
            "4096 generated structures per method",
            bool((generated_counts == EXPECTED_SAMPLES).all()),
            generated_counts.to_dict(),
        ),
        _check(
            "independent property predictions",
            properties.loc[
                properties["method"] != "Training set", "source_predictions"
            ]
            .astype(str)
            .str.contains("seed123")
            .all(),
            "all generated methods use seed123 prediction files",
        ),
        _check(
            "quality method matrix",
            set(quality["method"]) == QUALITY_METHODS,
            sorted(quality["method"].unique()),
        ),
        _check(
            "quality per-structure coverage",
            bool((quality_counts == EXPECTED_SAMPLES).all()),
            quality_counts.to_dict(),
        ),
        _check(
            "finite quality summaries",
            _finite(
                quality,
                [
                    "compositional_validity",
                    "structural_validity",
                    "uniqueness",
                    "novelty_among_unique",
                    "sun",
                    "mp_hull_stability",
                ],
            ),
            "S.U.N., validity, stability, uniqueness and novelty",
        ),
        _check(
            "target-yield diversity matrix",
            set(tradeoff["method"])
            == {
                "Base",
                "CFG-FE",
                "CFG-BG",
                "CFG-FE+BG",
                *TRAINING_METHODS,
            },
            sorted(tradeoff["method"].unique()),
        ),
        _check(
            "finite target-yield and diversity",
            _finite(
                tradeoff,
                ["valid_joint_target_yield", "structure_diversity"],
            ),
            "independent seed123 target yield and deterministic diversity",
        ),
        _check(
            "equal t-SNE groups",
            set(tsne["method"]) == SEARCH_METHODS
            and tsne_counts.nunique() == 1,
            tsne_counts.to_dict(),
        ),
        _check(
            "search-space profile groups",
            profile_methods == SEARCH_METHODS and element_methods == SEARCH_METHODS,
            {
                "profiles": sorted(profile_methods),
                "elements": sorted(element_methods),
            },
        ),
        _check(
            "source manifest present",
            (source_dir / "source_manifest.json").is_file(),
            "sha256 provenance",
        ),
    ]
    authorized = all(item["pass"] for item in checks)

    reward_shifts = []
    for method, frame in dynamics.groupby("method"):
        ordered = frame.sort_values("update_id")
        reward_shifts.append(
            {
                "method": method,
                "first_20_reward_mean": float(ordered.head(20)["reward_mean"].mean()),
                "last_20_reward_mean": float(ordered.tail(20)["reward_mean"].mean()),
                "last_minus_first": float(
                    ordered.tail(20)["reward_mean"].mean()
                    - ordered.head(20)["reward_mean"].mean()
                ),
            }
        )
    quality_view = quality[
        [
            "method",
            "sun",
            "uniqueness",
            "novelty_among_unique",
            "mp_hull_stability",
            "compositional_validity",
            "structural_validity",
        ]
    ]
    report = {
        "scope": "exploratory_seed42_single_seed",
        "plot_authorized": authorized,
        "checks": checks,
        "training_reward_shifts": reward_shifts,
        "final_paired_evaluation": _records(paired),
        "quality_summary": _records(quality_view),
        "target_yield_diversity": _records(tradeoff),
        "claim_boundaries": [
            "The trained method is GRPO+PIPO, not paired-gated CrystalPIRL.",
            "One random seed is exploratory evidence and cannot support robustness or significance claims.",
            "Independent seed123 M3GNet predictions are model-based evaluation, not DFT ground truth.",
            "Stability uses a predicted formation-energy hull and is not relaxed MLFF/DFT stability.",
            "t-SNE is qualitative; target yield and fingerprint diversity provide the quantitative trade-off evidence.",
            "A negative or rejected final policy remains in the figures and must not be hidden.",
        ],
    }
    (source_dir / "conclusion_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Seed=42 GRPO+PIPO 数据与结论审计",
        "",
        f"- 绘图授权：**{'PASS' if authorized else 'FAIL'}**",
        "- 方法身份：GRPO+PIPO 对照；不是 CrystalPIRL 主方法。",
        "- 统计范围：单一训练/生成种子 42；不得报告跨 seed 显著性。",
        "",
        "## 完整性检查",
        "",
    ]
    lines.extend(
        f"- [{'x' if item['pass'] else ' '}] {item['name']}: {item['detail']}"
        for item in checks
    )
    lines.extend(["", "## 结论边界", ""])
    lines.extend(f"- {item}" for item in report["claim_boundaries"])
    (source_dir / "conclusion_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    if not authorized:
        failed = [item["name"] for item in checks if not item["pass"]]
        raise RuntimeError("Conclusion audit failed: " + ", ".join(failed))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    audit(args.source_dir.resolve())


if __name__ == "__main__":
    main()
