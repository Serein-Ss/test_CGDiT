"""Run a trained Tc gate checkpoint on its validation split."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from cgdit.common.evaluation_utils import load_model


def predict_validation(run_dir: Path, output_dir: Path, device: str) -> None:
    run_dir = run_dir.resolve()
    model, loaders, _ = load_model(run_dir, load_data=True, testing=False)
    _, val_loader = loaders
    target_device = torch.device(
        device if device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    model = model.to(target_device).eval()
    logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Predicting gate validation split"):
            batch = batch.to(target_device)
            logits.append(model(batch).detach().cpu())
            targets.append(model._targets(batch).detach().cpu())

    all_logits = torch.cat(logits).numpy()
    all_targets = torch.cat(targets).numpy()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "val_logits.npy", all_logits)
    np.save(output_dir / "val_probs.npy", torch.sigmoid(torch.from_numpy(all_logits)).numpy())
    np.save(output_dir / "val_targets.npy", all_targets)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def cli(argv=None) -> None:
    args = build_parser().parse_args(argv)
    predict_validation(args.run_dir, args.output_dir, args.device)


if __name__ == "__main__":
    cli()
