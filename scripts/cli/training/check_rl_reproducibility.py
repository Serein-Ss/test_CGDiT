"""Fingerprint one seeded full RL trajectory for cross-process comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from cgdit.common.evaluation_utils import load_model
from scripts.cli.training.train_crystal_rl import _grouped_batch, _set_seed


def _update_tensor_hash(
    digest: Any, name: str, tensor: torch.Tensor
) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.numpy().tobytes())


def _trajectory_fingerprint(trajectory: Any) -> str:
    digest = hashlib.sha256()
    _update_tensor_hash(digest, "num_atoms", trajectory.num_atoms)
    _update_tensor_hash(
        digest, "atom_types", trajectory.final_state.atom_types
    )
    _update_tensor_hash(
        digest, "frac_coords", trajectory.final_state.frac_coords
    )
    _update_tensor_hash(
        digest, "crys_fam", trajectory.final_state.crys_fam
    )
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.train_config)
    required = (
        "model_path",
        "num_prompts",
        "group_size",
        "seed",
        "short_diff_ratio",
        "step_lr",
    )
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(
            "Training config is missing: " + ", ".join(missing)
        )
    model_path = Path(config.model_path).resolve()
    _set_seed(int(config.seed), deterministic=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Reproducibility check requires a CUDA allocation")

    model, test_loader, _ = load_model(
        model_path,
        load_data=True,
        testing=True,
    )
    if test_loader is None:
        raise RuntimeError("The base model did not provide a test dataset")
    model = model.to("cuda").eval()
    batch, _ = _grouped_batch(
        test_loader.dataset,
        int(config.num_prompts),
        int(config.group_size),
    )
    batch = batch.to("cuda")
    _, _, trajectory = model.sample_rl(
        batch,
        step_lr=float(config.step_lr),
        guidance_scale=0.0,
        noise_seed=int(config.seed),
        diff_ratio=float(config.short_diff_ratio),
    )
    print(
        json.dumps(
            {
                "fingerprint": _trajectory_fingerprint(trajectory),
                "seed": int(config.seed),
                "batch_size": int(config.num_prompts)
                * int(config.group_size),
                "diff_ratio": float(config.short_diff_ratio),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
