"""Create a parity plot from property prediction arrays."""

import argparse
from pathlib import Path

from cgdit.evaluation.visualization import plot_property_parity_from_files


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="", help="Directory containing test_preds.npy and test_targets.npy")
    parser.add_argument("--predictions", default="", help="Explicit prediction .npy path")
    parser.add_argument("--targets", default="", help="Explicit target .npy path")
    parser.add_argument("--output", default="", help="Output PNG; defaults to <run-dir>/parity_plot.png")
    return parser


def cli(argv=None):
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    if run_dir is None and (not args.predictions or not args.targets or not args.output):
        raise ValueError("Provide --run-dir, or provide --predictions, --targets, and --output together")
    predictions = Path(args.predictions).resolve() if args.predictions else run_dir / "test_preds.npy"
    targets = Path(args.targets).resolve() if args.targets else run_dir / "test_targets.npy"
    output = Path(args.output).resolve() if args.output else run_dir / "parity_plot.png"
    metrics = plot_property_parity_from_files(predictions, targets, output)
    print(f"MAE={metrics['mae']:.6f}, R2={metrics['r2']:.6f}")
    print(f"Saved parity plot to {metrics['output']}")


if __name__ == "__main__":
    cli()
