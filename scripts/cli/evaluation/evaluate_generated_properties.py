"""CLI for property evaluation of formal generated structures."""

import argparse
from pathlib import Path

import torch

from cgdit.evaluation.generated_properties import (
    DEFAULT_TOLERANCES,
    PROPERTY_NAMES,
    discover_formal_generation_files,
    evaluate_generation_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--fe_run", required=True)
    parser.add_argument("--bg_run", required=True)
    parser.add_argument("--eh_run", required=True)
    parser.add_argument(
        "--output_dir", default="assets/model_results/source_data/property_evaluation_seed42"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--fe_tolerance", type=float,
        default=DEFAULT_TOLERANCES[PROPERTY_NAMES[0]],
    )
    parser.add_argument(
        "--bg_tolerance", type=float,
        default=DEFAULT_TOLERANCES[PROPERTY_NAMES[1]],
    )
    parser.add_argument(
        "--eh_tolerance", type=float,
        default=DEFAULT_TOLERANCES[PROPERTY_NAMES[2]],
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def cli(argv=None) -> None:
    args = build_parser().parse_args(argv)
    files = discover_formal_generation_files(args.root_path)
    if not files:
        raise RuntimeError(
            f"No formal seed-42 generation files found under {args.root_path}"
        )
    print(f"Discovered {len(files)} formal generation files.")
    evaluate_generation_files(
        generation_files=files,
        predictor_runs={
            "formation_energy_per_atom": Path(args.fe_run),
            "band_gap": Path(args.bg_run),
            "e_above_hull": Path(args.eh_run),
        },
        output_dir=Path(args.output_dir),
        tolerances={
            "formation_energy_per_atom": args.fe_tolerance,
            "band_gap": args.bg_tolerance,
            "e_above_hull": args.eh_tolerance,
        },
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    cli()
