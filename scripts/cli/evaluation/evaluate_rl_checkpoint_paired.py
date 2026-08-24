"""Evaluate one intermediate RL checkpoint against the frozen Base policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from cgdit.common.evaluation_utils import load_model
from scripts.cli.training.train_crystal_rl import (
    _adapter,
    _evaluate_final_policy,
    _grouped_batch,
    _prompt_indices,
    _required_paths,
    _set_seed,
    _validate_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    config = OmegaConf.to_container(
        OmegaConf.load(cli.train_config), resolve=True
    )
    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a mapping")
    settings = SimpleNamespace(**config)
    settings.model_path = Path(settings.model_path).resolve()
    settings.reward_contract = Path(settings.reward_contract)
    settings.reward_registry = Path(settings.reward_registry)

    _set_seed(int(settings.seed), deterministic=bool(settings.deterministic))
    contract = OmegaConf.load(settings.reward_contract)
    registry = OmegaConf.load(settings.reward_registry)
    _validate_paths([*_required_paths(settings, registry), cli.checkpoint])
    if not torch.cuda.is_available():
        raise RuntimeError("Intermediate paired evaluation requires CUDA")

    model, prompt_loaders, _ = load_model(
        settings.model_path, load_data=True, testing=False
    )
    if prompt_loaders is None:
        raise RuntimeError("The Base model did not provide validation data")
    _, validation_loader = prompt_loaders
    validation_dataset = validation_loader.dataset
    base_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }

    payload = torch.load(
        cli.checkpoint, map_location="cpu", weights_only=False
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    device = torch.device("cuda")
    model = model.to(device).eval()
    adapter = _adapter(settings, contract, registry, device)

    probe_indices = _prompt_indices(
        len(validation_dataset),
        int(settings.probe_num_prompts),
        int(settings.probe_seed),
    )
    holdout_indices = _prompt_indices(
        len(validation_dataset),
        int(settings.probe_num_prompts),
        int(settings.holdout_probe_seed),
        excluded=probe_indices,
    )
    final_indices = _prompt_indices(
        len(validation_dataset),
        int(settings.probe_num_prompts),
        int(settings.final_probe_seed),
        excluded=probe_indices + holdout_indices,
    )
    final_batch, _ = _grouped_batch(
        validation_dataset,
        int(settings.probe_num_prompts),
        int(settings.probe_group_size),
        prompt_indices=final_indices,
    )
    final_batch = final_batch.to(device)

    evaluation = _evaluate_final_policy(
        model=model,
        initial_state=base_state,
        batch=final_batch,
        adapter=adapter,
        contract=contract,
        noise_seed=int(settings.final_probe_seed),
        diff_ratio=float(settings.short_diff_ratio),
        run_kind="test",
        step_lr=float(settings.step_lr),
    )
    result = {
        "status": "interim_checkpoint_paired_evaluation",
        "property": str(settings.property),
        "checkpoint": str(cli.checkpoint),
        "checkpoint_step": int(payload.get("rl_step", -1)),
        "completed_updates": int(payload.get("rl_step", -1)) + 1,
        "prompt_split": "validation",
        "prompt_indices": final_indices,
        "num_prompts": int(settings.probe_num_prompts),
        "trajectories_per_prompt": int(settings.probe_group_size),
        "evaluation": evaluation,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
