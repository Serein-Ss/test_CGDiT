"""Diagnose whether an RL candidate update has the wrong direction or scale."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from cgdit.common.evaluation_utils import load_model
from cgdit.rl.objectives import group_relative_advantages
from cgdit.rl.policy_improvement import decide_reward_improvement
from cgdit.rl.trainer import DiffusionPolicyObjective
from scripts.cli.training.train_crystal_rl import (
    PROPERTY_NAMES,
    _adapter,
    _grouped_batch,
    _optimize_policy_epochs,
    _reward_summary,
    _set_seed,
    _stochastic_transition_indices,
    _trainable_decoder,
    _validate_paths,
)


def _set_decoder_scale(
    model: torch.nn.Module,
    base_state: dict[str, torch.Tensor],
    candidate_state: dict[str, torch.Tensor],
    scale: float,
) -> None:
    with torch.no_grad():
        for name, parameter in model.decoder.named_parameters():
            parameter.copy_(
                base_state[name]
                + scale * (candidate_state[name] - base_state[name])
            )


def _decoder_update_norm(
    base_state: dict[str, torch.Tensor],
    candidate_state: dict[str, torch.Tensor],
) -> tuple[float, float]:
    update_squared = 0.0
    base_squared = 0.0
    for name in base_state:
        update_squared += float(
            (candidate_state[name] - base_state[name]).square().sum().item()
        )
        base_squared += float(base_state[name].square().sum().item())
    update_norm = math.sqrt(update_squared)
    relative_norm = update_norm / max(math.sqrt(base_squared), 1.0e-12)
    return update_norm, relative_norm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=(0.05, 0.1, 0.25, 0.5, 1.0),
    )
    args = parser.parse_args()
    if any(scale <= 0.0 or scale > 1.0 for scale in args.scales):
        parser.error("all scales must be in (0, 1]")
    if len(set(args.scales)) != len(args.scales):
        parser.error("scales must be unique")
    return args


def _required_config(config: Any) -> None:
    required = (
        "model_path",
        "reward_contract",
        "reward_registry",
        "property",
        "algorithm",
        "num_prompts",
        "group_size",
        "short_diff_ratio",
        "step_lr",
        "learning_rate",
        "policy_epochs",
        "clip_epsilon",
        "replay_transitions",
        "predictor_batch_size",
        "seed",
        "probe_seed",
        "reference_weight",
    )
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError("Training config is missing: " + ", ".join(missing))


def main() -> None:
    cli = parse_args()
    config = OmegaConf.load(cli.train_config)
    _required_config(config)
    property_key = str(config.property)
    algorithm = str(config.algorithm)
    if property_key not in PROPERTY_NAMES:
        raise ValueError(f"Unknown property: {property_key}")
    if algorithm not in {"ppo", "grpo"}:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    model_path = Path(config.model_path).resolve()
    contract = OmegaConf.load(config.reward_contract)
    registry = OmegaConf.load(config.reward_registry)
    required_paths = [
        model_path,
        model_path / "hparams.yaml",
        Path("data/mp_20/test.csv"),
        Path("data/mp_20/test_sym.pt"),
    ]
    required_paths.extend(
        Path(registry.models[name].checkpoint)
        for name in PROPERTY_NAMES[property_key]
    )
    _validate_paths(required_paths)
    _set_seed(int(config.seed), deterministic=True)
    if not torch.cuda.is_available():
        raise RuntimeError("RL scale diagnosis requires a CUDA allocation")

    device = torch.device("cuda")
    model, test_loader, _ = load_model(
        model_path,
        load_data=True,
        testing=True,
    )
    if test_loader is None:
        raise RuntimeError("The base model did not provide a test dataset")
    model = model.to(device).eval()
    baseline = copy.deepcopy(model).eval()
    baseline.requires_grad_(False)
    trainable_parameters = _trainable_decoder(model)
    adapter_args = argparse.Namespace(
        property=property_key,
        predictor_batch_size=int(config.predictor_batch_size),
    )
    adapter = _adapter(adapter_args, contract, registry, device)
    batch, group_index = _grouped_batch(
        test_loader.dataset,
        int(config.num_prompts),
        int(config.group_size),
    )
    batch = batch.to(device)
    group_index = group_index.to(device)

    base_decoder_state = {
        name: parameter.detach().clone()
        for name, parameter in model.decoder.named_parameters()
    }
    _, _, training_trajectory = model.sample_rl(
        batch,
        step_lr=float(config.step_lr),
        guidance_scale=0.0,
        noise_seed=int(config.seed),
        diff_ratio=float(config.short_diff_ratio),
    )
    training_evaluation = adapter.evaluate_state(
        final_state=training_trajectory.final_state,
        num_atoms=training_trajectory.num_atoms,
        crystal_family=model.crystal_family,
    )
    rewards = training_evaluation.reward.raw_reward.detach()
    if algorithm == "ppo":
        advantages = (rewards - rewards.mean()) / (
            rewards.std(unbiased=False) + 1.0e-8
        )
    else:
        advantages = group_relative_advantages(rewards, group_index)
    selected = _stochastic_transition_indices(
        training_trajectory,
        int(config.replay_transitions),
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config.learning_rate),
    )
    objective = DiffusionPolicyObjective(
        algorithm=algorithm,
        clip_epsilon=float(config.clip_epsilon),
    )
    _, _, _, policy_epochs = _optimize_policy_epochs(
        objective=objective,
        model=model,
        batch=batch,
        trajectory=training_trajectory,
        step_lr=float(config.step_lr),
        rewards=rewards,
        advantages=advantages,
        group_index=group_index,
        transition_indices=selected,
        reference_weight=float(config.reference_weight),
        optimizer=optimizer,
        trainable_parameters=trainable_parameters,
        policy_epochs=int(config.policy_epochs),
    )
    candidate_decoder_state = {
        name: parameter.detach().clone()
        for name, parameter in model.decoder.named_parameters()
    }
    update_norm, relative_update_norm = _decoder_update_norm(
        base_decoder_state,
        candidate_decoder_state,
    )
    del training_trajectory

    probe_seed = int(config.probe_seed)
    _, _, baseline_trajectory = baseline.sample_rl(
        batch,
        step_lr=float(config.step_lr),
        guidance_scale=0.0,
        noise_seed=probe_seed,
        diff_ratio=float(config.short_diff_ratio),
    )
    baseline_evaluation = adapter.evaluate_state(
        final_state=baseline_trajectory.final_state,
        num_atoms=baseline_trajectory.num_atoms,
        crystal_family=model.crystal_family,
    )
    del baseline, baseline_trajectory

    scale_records = []
    for scale in cli.scales:
        _set_decoder_scale(
            model,
            base_decoder_state,
            candidate_decoder_state,
            scale,
        )
        _, _, scaled_trajectory = model.sample_rl(
            batch,
            step_lr=float(config.step_lr),
            guidance_scale=0.0,
            noise_seed=probe_seed,
            diff_ratio=float(config.short_diff_ratio),
        )
        scaled_evaluation = adapter.evaluate_state(
            final_state=scaled_trajectory.final_state,
            num_atoms=scaled_trajectory.num_atoms,
            crystal_family=model.crystal_family,
        )
        decision = decide_reward_improvement(
            old_reward=baseline_evaluation.reward.raw_reward,
            new_reward=scaled_evaluation.reward.raw_reward,
            old_safety_metrics={
                "validity": baseline_evaluation.valid.float()
            },
            new_safety_metrics={
                "validity": scaled_evaluation.valid.float()
            },
            safety_tolerances={
                "validity": float(
                    contract.closed_loop.safety_drop_tolerance
                )
            },
            confidence=float(contract.closed_loop.confidence),
            n_bootstrap=int(contract.closed_loop.bootstrap_samples),
            attenuation=float(contract.closed_loop.attenuation_scale),
            seed=probe_seed,
        )
        scale_records.append(
            {
                "scale": scale,
                "evaluation": _reward_summary(scaled_evaluation),
                "decision": {
                    "action": decision.action,
                    "scale": float(decision.scale),
                    "mean_delta": decision.mean_delta.cpu().tolist(),
                    "lower_confidence_bound": (
                        decision.lower_confidence_bound.cpu().tolist()
                    ),
                },
            }
        )
        del scaled_trajectory

    report = {
        "algorithm": algorithm,
        "property": property_key,
        "training_seed": int(config.seed),
        "probe_seed": probe_seed,
        "batch_size": int(config.num_prompts) * int(config.group_size),
        "selected_transitions": selected,
        "policy_epochs": policy_epochs,
        "training_evaluation": _reward_summary(training_evaluation),
        "decoder_update_norm": update_norm,
        "decoder_relative_update_norm": relative_update_norm,
        "baseline_evaluation": _reward_summary(baseline_evaluation),
        "scale_evaluations": scale_records,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
