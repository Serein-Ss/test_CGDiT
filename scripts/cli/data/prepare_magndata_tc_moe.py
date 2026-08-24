"""Prepare fixed gate/low-expert/high-expert splits for Magndata-Tc MoE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ("train", "val", "test")


def _target_stats(frame: pd.DataFrame) -> dict[str, float | int]:
    values = frame["tc"].to_numpy(dtype=float)
    if not len(values):
        raise ValueError("An expert training split is empty")
    std = float(np.std(values))
    if std <= 0:
        raise ValueError("An expert training split has zero target variance")
    return {
        "rows": int(len(values)),
        "mean_k": float(np.mean(values)),
        "std_k_ddof0": std,
        "min_k": float(np.min(values)),
        "max_k": float(np.max(values)),
    }


def prepare_moe_splits(
    source_root: Path,
    output_root: Path,
    threshold_k: float,
) -> dict:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if not np.isfinite(threshold_k):
        raise ValueError("threshold_k must be finite")

    frames: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        path = source_root / f"{split}.csv"
        frame = pd.read_csv(path)
        if "tc" not in frame:
            raise KeyError(f"Missing tc column in {path}")
        if not np.isfinite(frame["tc"].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite tc values in {path}")
        frame = frame.copy()
        frame["high_tc"] = (frame["tc"] >= threshold_k).astype(float)
        frames[split] = frame

    output_root.mkdir(parents=True, exist_ok=True)
    for branch in ("gate", "low", "high"):
        (output_root / branch).mkdir(parents=True, exist_ok=True)

    counts: dict[str, dict[str, int]] = {}
    for split, frame in frames.items():
        low = frame[frame["high_tc"] == 0].reset_index(drop=True)
        high = frame[frame["high_tc"] == 1].reset_index(drop=True)
        counts[split] = {"low": int(len(low)), "high": int(len(high))}
        frame.to_csv(output_root / "gate" / f"{split}.csv", index=False)
        if split == "test":
            # Both experts predict every test structure; routing is applied later.
            frame.to_csv(output_root / "low" / f"{split}.csv", index=False)
            frame.to_csv(output_root / "high" / f"{split}.csv", index=False)
        else:
            low.to_csv(output_root / "low" / f"{split}.csv", index=False)
            high.to_csv(output_root / "high" / f"{split}.csv", index=False)

    low_stats = _target_stats(frames["train"][frames["train"].high_tc == 0])
    high_stats = _target_stats(frames["train"][frames["train"].high_tc == 1])
    train_low = counts["train"]["low"]
    train_high = counts["train"]["high"]
    if train_high == 0:
        raise ValueError("No high-Tc training examples")

    manifest = {
        "threshold_k": float(threshold_k),
        "label_rule": f"high_tc = 1 if tc >= {threshold_k:g} K else 0",
        "source_root": str(source_root),
        "counts": counts,
        "gate_pos_weight": float(train_low / train_high),
        "low_expert_target_stats": low_stats,
        "high_expert_target_stats": high_stats,
    }
    (output_root / "moe_data_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threshold-k", type=float, default=300.0)
    return parser


def cli(argv=None) -> None:
    args = build_parser().parse_args(argv)
    manifest = prepare_moe_splits(
        args.source_root,
        args.output_root,
        args.threshold_k,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    cli()
