"""Audit seed-42 Source Data and write conclusion boundaries before plotting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _finite(table: pd.DataFrame, columns: list[str]) -> bool:
    values = table[columns].apply(pd.to_numeric, errors="coerce").to_numpy()
    return bool(np.isfinite(values).all())


def _records(table: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(table.to_json(orient="records"))


def _update_summary(updates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["task", "algorithm", "verifier"]
    for key, frame in updates.groupby(keys, dropna=False):
        final = frame.loc[frame["update_id"].idxmax()]
        applied = frame[frame["update_applied"].astype(bool)]
        rows.append(
            {
                **dict(zip(keys, key)),
                "updates": len(frame),
                "applied_updates": len(applied),
                "bad_applied_updates": int(
                    applied["bad_applied_update"].astype(bool).sum()
                ),
                "bad_applied_rate": (
                    float(applied["bad_applied_update"].astype(bool).mean())
                    if len(applied)
                    else None
                ),
                "final_training_reward": float(final["training_reward_mean"]),
                "final_fixed_absolute_lcb": float(final["fixed_absolute_lcb"]),
                "final_holdout_absolute_lcb": float(
                    final["holdout_absolute_lcb"]
                ),
                "final_holdout_local_lcb": float(final["holdout_local_lcb"]),
            }
        )
    return pd.DataFrame(rows)


def _property_summary(summary: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "task",
        "method",
        "method_family",
        "algorithm",
        "generation_mode",
    ]
    return (
        summary.groupby(keys, dropna=False)
        .agg(
            n=("n", "sum"),
            finite_prediction_fraction=("finite_prediction_fraction", "mean"),
            independent_target_mae=("target_mae", "mean"),
            independent_hit_rate=("hit_rate", "mean"),
            valid_target_yield=("valid_target_yield", "mean"),
            validity=("validity", "mean"),
            predictor_gap_mae=("predictor_gap_mae", "mean"),
        )
        .reset_index()
        .sort_values(
            ["task", "generation_mode", "independent_target_mae"],
            kind="stable",
        )
    )


def _quality_summary(quality: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "method_family", "algorithm", "task", "generation_mode"]
    return (
        quality.groupby(keys, dropna=False)
        .agg(
            n=("n", "sum"),
            validity=("validity", "mean"),
            stability_proxy_mean=("stability_proxy_mean", "mean"),
            coverage_precision=("coverage_precision", "mean"),
            coverage_recall=("coverage_recall", "mean"),
            uniqueness=("uniqueness", "mean"),
            novelty_among_unique=("novelty_among_unique", "mean"),
            formula_shannon=("formula_shannon", "mean"),
            embedding_dispersion=("embedding_dispersion", "mean"),
        )
        .reset_index()
    )


def _ranked_lines(properties: pd.DataFrame) -> list[str]:
    lines = []
    for (task, mode), frame in properties.groupby(
        ["task", "generation_mode"], sort=True
    ):
        best = frame.sort_values(
            ["valid_target_yield", "independent_target_mae"],
            ascending=[False, True],
        ).iloc[0]
        lines.append(
            f"- {task.upper()} / {mode}: highest independent valid-target "
            f"yield is {best['method']} ({best['valid_target_yield']:.4f}); "
            f"its independent target MAE is "
            f"{best['independent_target_mae']:.4f}."
        )
    return lines


def _check(name: str, condition: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "pass": bool(condition), "detail": detail}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()

    required = {
        "fig1": "fig1_framework_evidence.csv",
        "fig2": "fig2_update_audit.csv",
        "fig3": "fig3_method_summary.csv",
        "fig3_raw": "fig3_structure_properties.csv",
        "fig4": "fig4_quality_summary.csv",
        "fig4_sample": "fig4_structure_sample.csv",
        "completeness": "data_completeness.json",
        "sampling": "fig4_sampling_contract.json",
    }
    missing = [
        filename
        for filename in required.values()
        if not (args.source_dir / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing Source Data files: " + ", ".join(missing))

    fig1 = pd.read_csv(args.source_dir / required["fig1"])
    fig2 = pd.read_csv(args.source_dir / required["fig2"])
    fig3 = pd.read_csv(args.source_dir / required["fig3"])
    fig3_raw = pd.read_csv(args.source_dir / required["fig3_raw"])
    fig4 = pd.read_csv(args.source_dir / required["fig4"])
    sample = pd.read_csv(args.source_dir / required["fig4_sample"])
    completeness = json.loads(
        (args.source_dir / required["completeness"]).read_text(encoding="utf-8")
    )
    sampling = json.loads(
        (args.source_dir / required["sampling"]).read_text(encoding="utf-8")
    )
    expected = completeness["expected"]
    expected_sample_size = int(sampling["sample_size_per_group"])

    update_sizes = fig2.groupby("run_id").size()
    fig3_group_sizes = fig3.set_index(
        ["task", "method_family", "algorithm", "generation_mode"]
    )["n"]
    summary_families = fig3.reset_index(drop=True)["method_family"]
    expected_fig3_group_sizes = pd.Series(
        np.where(
            summary_families == "best_of_n",
            expected["best_of_eight_samples_per_group"],
            np.where(
                summary_families.isin(["open", "pipo", "crystalpirl"]),
                expected["rl_samples_per_group"],
                expected["baseline_samples_per_group"],
            ),
        ),
        index=fig3_group_sizes.index,
    )
    expected_family_counts = {
        f"{task}:{family}": 2 if family in {"base", "cfg", "best_of_n"} else 4
        for task in ("fe", "bg")
        for family in ("base", "cfg", "best_of_n", "open", "pipo", "crystalpirl")
    }
    observed_family_counts = {
        f"{task}:{family}": int(count)
        for (task, family), count in fig3.groupby(
            ["task", "method_family"], dropna=False
        ).size().items()
    }
    sample_sizes = sample.groupby("group_id").size()
    expected_sample_sizes = (
        sample.groupby("group_id")["valid_pool_n"]
        .first()
        .clip(upper=expected_sample_size)
        .astype(int)
    )
    expected_fig4_sizes = np.where(
        fig4["method_family"].isin(["open", "pipo", "crystalpirl"]),
        expected["rl_samples_per_group"],
        expected["baseline_samples_per_group"],
    )
    seed_values = sorted(
        {
            *fig1["seed"].dropna().astype(int).tolist(),
            *fig2["seed"].dropna().astype(int).tolist(),
            *fig3["seed"].dropna().astype(int).tolist(),
            *fig4["seed"].dropna().astype(int).tolist(),
            *sample["seed"].dropna().astype(int).tolist(),
        }
    )
    checks = [
        _check(
            "upstream completeness",
            completeness.get("status") == "complete",
            completeness,
        ),
        _check("single seed", seed_values == [42], seed_values),
        _check("Fig. 1 update rows", len(fig1) == expected["fig1_rows"], len(fig1)),
        _check("Fig. 2 update rows", len(fig2) == expected["fig2_rows"], len(fig2)),
        _check(
            "Fig. 2 run count",
            fig2["run_id"].nunique() == expected["fig2_runs"],
            int(fig2["run_id"].nunique()),
        ),
        _check(
            f"{expected['updates_per_run']} updates per run",
            bool(
                (update_sizes == expected["updates_per_run"]).all()
                and (
                    fig2.groupby("run_id")["update_id"].nunique()
                    == expected["updates_per_run"]
                ).all()
            ),
            update_sizes.to_dict(),
        ),
        _check(
            "Fig. 3 summary groups",
            len(fig3) == expected["fig3_summary_rows"],
            len(fig3),
        ),
        _check(
            "Fig. 3 structure rows",
            len(fig3_raw) == expected["fig3_rows"],
            len(fig3_raw),
        ),
        _check(
            "Fig. 3 method matrix",
            observed_family_counts == expected_family_counts,
            observed_family_counts,
        ),
        _check(
            "Fig. 3 per-group sample counts",
            bool(
                np.array_equal(
                    fig3_group_sizes.to_numpy(dtype=int),
                    expected_fig3_group_sizes.to_numpy(dtype=int),
                )
            ),
            {
                "baseline_groups": expected["baseline_samples_per_group"],
                "rl_groups": expected["rl_samples_per_group"],
                "best_of_8_groups": expected["best_of_eight_samples_per_group"],
            },
        ),
        _check(
            "Fig. 3 structure identity coverage",
            not fig3_raw.duplicated(["task", "structure_id"]).any(),
            "unique within task and method group",
        ),
        _check("Fig. 4 groups", len(fig4) == expected["fig4_groups"], len(fig4)),
        _check(
            "Fig. 4 primary sample counts",
            bool(
                np.array_equal(
                    fig4["n"].to_numpy(dtype=int), expected_fig4_sizes
                )
            ),
            sorted(fig4["n"].astype(int).unique().tolist()),
        ),
        _check(
            "Fig. 4 sampled groups",
            sample["group_id"].nunique() == expected["fig4_groups"],
            int(sample["group_id"].nunique()),
        ),
        _check(
            "deterministic sample size",
            bool(
                sample_sizes.equals(expected_sample_sizes)
                and sampling.get("seed") == 42
            ),
            {
                "observed": sample_sizes.to_dict(),
                "expected_min_sample_size_or_valid_pool": expected_sample_sizes.to_dict(),
            },
        ),
        _check(
            "finite fixed-probe evidence",
            _finite(
                fig2,
                [
                    "training_reward_mean",
                    "fixed_local_lcb",
                    "fixed_absolute_lcb",
                ],
            ),
            "required reward and fixed-anchor LCB fields",
        ),
        _check(
            "finite required holdout evidence",
            _finite(
                fig2[
                    (fig2["verifier"] != "crystalpirl")
                    | fig2["update_applied"].astype(bool)
                ],
                ["holdout_local_lcb", "holdout_absolute_lcb"],
            ),
            "all Open/PIPO updates and applied CrystalPIRL updates",
        ),
        _check(
            "finite property evidence",
            _finite(
                fig3,
                [
                    "target_mae",
                    "finite_prediction_fraction",
                    "hit_rate",
                    "valid_target_yield",
                    "validity",
                    "predictor_gap_mae",
                ],
            ),
            "independent predictor summaries",
        ),
        _check(
            "finite quality evidence",
            _finite(
                fig4,
                [
                    "validity",
                    "stability_proxy_mean",
                    "coverage_precision",
                    "coverage_recall",
                    "uniqueness",
                    "novelty_among_unique",
                ],
            ),
            "quality and sampled diversity summaries",
        ),
    ]
    authorized = all(item["pass"] for item in checks)
    updates = _update_summary(fig2)
    properties = _property_summary(fig3)
    quality = _quality_summary(fig4)

    report = {
        "scope": completeness.get("scope", "exploratory_seed42_only"),
        "plot_authorized": authorized,
        "checks": checks,
        "update_conclusions": _records(updates),
        "property_conclusions": _records(properties),
        "quality_conclusions": _records(quality),
        "claim_boundaries": [
            "One random seed is exploratory evidence, not robustness evidence.",
            "No p values, confidence intervals across seeds, or significance claims.",
            "Stability is a formation-energy proxy, not MLFF or DFT validation.",
            "Independent seed-123 predictors test reward-model transfer but are not DFT ground truth.",
            "Property distributions include all finite predictions; invalid or failed predictions remain counted as failures in valid-target yield.",
            "The audit authorizes plotting whether the result is positive or negative.",
            "A CrystalPIRL candidate rejected by the fixed probe has no required holdout evaluation by design.",
        ],
    }
    (args.source_dir / "conclusion_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    status = "PASS" if authorized else "FAIL"
    lines = [
        "# CrystalPIRL seed=42 结论审计",
        "",
        f"- 绘图授权：**{status}**",
        "- 范围：pilot, seed=42；仅用于确认实现、效应方向和资源配置。",
        "- 判定原则：数据完整即可授权绘图，不要求 CrystalPIRL 必须优于基线。",
        "",
        "## 完整性检查",
        "",
    ]
    lines.extend(
        f"- [{'x' if item['pass'] else ' '}] {item['name']}: {item['detail']}"
        for item in checks
    )
    lines.extend(["", "## 当前可复核的性质控制结论", ""])
    lines.extend(_ranked_lines(properties))
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "- 当前只有一个训练/生成种子，不报告跨种子误差条和统计显著性。",
            "- 稳定性仅为形成能代理，不等价于 MLFF 弛豫或 DFT 稳定性。",
            "- seed=123 独立预测器用于检查奖励模型迁移，不代表第一性原理真值。",
            "- 性质分布展示全部有限预测；无效或预测失败结构仍作为失败计入 valid-target yield。",
            "- 若结果为负，图中如实呈现，不删除失败方法臂或不利生成模式。",
            "- 固定探针已拒绝的 CrystalPIRL 候选按设计不要求 holdout，该空值不视为数据丢失。",
            "",
        ]
    )
    (args.source_dir / "conclusion_audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    if not authorized:
        failed = [item["name"] for item in checks if not item["pass"]]
        raise RuntimeError("Conclusion audit failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
