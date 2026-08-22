"""Run a minimal PPO/GRPO update with real crystal-property rewards."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from cgdit.common.evaluation_utils import load_model
from cgdit.evaluation.generated_properties import resolve_checkpoint
from cgdit.rl.paired_probe import paired_policy_rollout
from cgdit.rl.policy_improvement import decide_reward_improvement
from cgdit.rl.property_reward import M3GNetRewardAdapter, PropertyRewardSpec
from cgdit.rl.trainer import DiffusionPolicyObjective


PROPERTY_NAMES = {
    "fe": ("formation_energy_per_atom",),
    "bg": ("band_gap",),
    "joint": ("formation_energy_per_atom", "band_gap"),
}


def _config_defaults(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"Training config must contain a mapping: {path}")
    return config


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--train-config", type=Path)
    known, _ = pre_parser.parse_known_args()
    defaults = _config_defaults(known.train_config)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", type=Path)
    parser.add_argument("--model-path", type=Path, required="model_path" not in defaults)
    parser.add_argument(
        "--reward-contract",
        type=Path,
        default=Path("conf/rl/reward_closed_loop_mp20.yaml"),
    )
    parser.add_argument(
        "--reward-registry",
        type=Path,
        default=Path("conf/rl/reward_models_mp20.yaml"),
    )
    parser.add_argument("--property", choices=PROPERTY_NAMES, default="fe")
    parser.add_argument("--algorithm", choices=("ppo", "grpo"), default="grpo")
    parser.add_argument("--pirl", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-root", type=Path, default=Path("output/rl_finetune"))
    parser.add_argument("--run-kind", choices=("train", "test"), default="train")
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--short-diff-ratio", type=float, default=0.02)
    parser.add_argument("--step-lr", type=float, default=1.0e-5)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--lr-decay", type=float, default=1.0)
    parser.add_argument("--replay-transitions", type=int, default=2)
    parser.add_argument("--predictor-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-seed", type=int, default=4242)
    parser.add_argument("--reference-weight", type=float, default=0.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    if args.num_prompts < 1 or args.group_size < 2:
        parser.error("num-prompts must be >= 1 and group-size must be >= 2")
    if args.updates < 1 or args.replay_transitions < 1:
        parser.error("updates and replay-transitions must be >= 1")
    if not 0.0 < args.short_diff_ratio <= 1.0:
        parser.error("short-diff-ratio must be in (0, 1]")
    return args


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _property_specs(
    contract: Any, property_key: str
) -> dict[str, PropertyRewardSpec]:
    specs = {}
    for name in PROPERTY_NAMES[property_key]:
        item = contract.properties[name]
        specs[name] = PropertyRewardSpec(
            mode=str(item.mode),
            target=float(item.target) if "target" in item else None,
            tolerance=float(item.tolerance),
            lower=float(item.lower) if "lower" in item else None,
            upper=float(item.upper) if "upper" in item else None,
        )
    return specs


def _adapter(
    args: argparse.Namespace,
    contract: Any,
    registry: Any,
    device: torch.device,
) -> M3GNetRewardAdapter:
    specs = _property_specs(contract, args.property)
    checkpoints = {
        name: str(Path(registry.models[name].checkpoint)) for name in specs
    }
    primary = contract.joint_property.primary
    mode = str(primary.mode) if args.property == "joint" else "weighted_mean"
    weights = None
    if args.property == "joint" and mode == "weighted_mean":
        weights = {
            name: float(contract.joint_property.ablation.weights[name])
            for name in specs
        }
    return M3GNetRewardAdapter.from_checkpoints(
        checkpoints=checkpoints,
        specs=specs,
        device=device,
        batch_size=args.predictor_batch_size,
        property_mode=mode,
        property_weights=weights,
        invalid_penalty=float(contract.validity.invalid_penalty),
    )


def _required_paths(
    args: argparse.Namespace,
    registry: Any,
) -> list[Path]:
    paths = [
        args.model_path,
        args.model_path / "hparams.yaml",
        Path("data/mp_20/test.csv"),
        Path("data/mp_20/test_sym.pt"),
    ]
    for name in PROPERTY_NAMES[args.property]:
        paths.append(Path(registry.models[name].checkpoint))
    return paths


def _validate_paths(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def _grouped_batch(dataset: Any, num_prompts: int, group_size: int) -> tuple[Batch, torch.Tensor]:
    if len(dataset) < num_prompts:
        raise ValueError(
            f"Dataset has {len(dataset)} entries, fewer than num-prompts={num_prompts}"
        )
    data = []
    group_index = []
    for prompt_index in range(num_prompts):
        prompt = dataset[prompt_index]
        for _ in range(group_size):
            data.append(prompt.clone())
            group_index.append(prompt_index)
    return Batch.from_data_list(data), torch.tensor(group_index, dtype=torch.long)


def _transition_indices(num_transitions: int, requested: int) -> list[int]:
    if num_transitions < 1:
        raise ValueError("Trajectory contains no stochastic transitions")
    count = min(num_transitions, requested)
    return (
        torch.linspace(0, num_transitions - 1, steps=count)
        .round()
        .long()
        .unique()
        .tolist()
    )


def _stochastic_transition_indices(trajectory: Any, requested: int) -> list[int]:
    available = [
        index
        for index, transition in enumerate(trajectory.transitions)
        if transition.stochastic
    ]
    selected = _transition_indices(len(available), requested)
    return [available[index] for index in selected]


def _trainable_decoder(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    model.requires_grad_(False)
    model.decoder.requires_grad_(True)
    parameters = [parameter for parameter in model.decoder.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("The decoder has no trainable parameters")
    return parameters


def _blend_decoder(
    model: torch.nn.Module,
    old_state: dict[str, torch.Tensor],
    scale: float,
) -> None:
    with torch.no_grad():
        for name, parameter in model.decoder.named_parameters():
            parameter.copy_(old_state[name] + scale * (parameter - old_state[name]))


def _reward_summary(evaluation: Any) -> dict[str, Any]:
    raw_reward = evaluation.reward.raw_reward
    return {
        "reward_mean": float(raw_reward.mean().item()),
        "reward_std": float(raw_reward.std(unbiased=False).item()),
        "valid_fraction": float(evaluation.valid.float().mean().item()),
        "property_means": {
            name: float(value[torch.isfinite(value)].mean().item())
            if torch.isfinite(value).any()
            else None
            for name, value in evaluation.predictions.items()
        },
        "graph_errors": evaluation.graph_errors,
    }


def _decision_summary(decision: Any) -> dict[str, Any]:
    return {
        "action": decision.action,
        "scale": float(decision.scale),
        "mean_delta": decision.mean_delta.detach().cpu().tolist(),
        "lower_confidence_bound": (
            decision.lower_confidence_bound.detach().cpu().tolist()
        ),
    }


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_output_paths(
    output_root: Path, run_name: str, run_kind: str
) -> tuple[Path, Path]:
    run_root = output_root / run_name
    return run_root / "model", run_root / f"{run_kind}_results"


def _save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    source_checkpoint: Path,
    output_dir: Path,
    step: int,
    metadata: dict[str, Any],
) -> Path:
    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    payload["state_dict"] = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    payload.pop("optimizer_states", None)
    payload.pop("lr_schedulers", None)
    payload["rl_optimizer_state"] = optimizer.state_dict()
    payload["rl_scheduler_state"] = scheduler.state_dict()
    payload["rl_step"] = step
    payload["rl_metadata"] = metadata

    output_path = output_dir / f"epoch={step}-step={step}.ckpt"
    temporary_path = output_path.with_suffix(".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, output_path)
    return output_path


def _restore_training_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["rl_optimizer_state"])
    scheduler.load_state_dict(payload["rl_scheduler_state"])
    return int(payload["rl_step"]) + 1


def main() -> None:
    args = parse_args()
    args.model_path = args.model_path.resolve()
    _set_seed(args.seed)
    contract = OmegaConf.load(args.reward_contract)
    registry = OmegaConf.load(args.reward_registry)
    _validate_paths(_required_paths(args, registry))

    if args.preflight_only:
        _adapter(args, contract, registry, torch.device("cpu"))
        load_model(args.model_path, load_data=False, testing=True)
        print(
            json.dumps(
                {
                    "status": "ready",
                    "algorithm": args.algorithm,
                    "property": args.property,
                    "predictors": list(PROPERTY_NAMES[args.property]),
                },
                ensure_ascii=False,
            )
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("RL training requires a CUDA allocation")

    device = torch.device("cuda")
    model, test_loader, _ = load_model(
        args.model_path,
        load_data=True,
        testing=True,
    )
    if test_loader is None:
        raise RuntimeError("The base model did not provide a test dataset")
    model = model.to(device).eval()
    trainable_parameters = _trainable_decoder(model)
    adapter = _adapter(args, contract, registry, device)

    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    start_step = 0
    if args.resume is not None:
        start_step = _restore_training_state(args.resume, model, optimizer, scheduler)

    dataset = test_loader.dataset
    batch, group_index = _grouped_batch(dataset, args.num_prompts, args.group_size)
    batch = batch.to(device)
    group_index = group_index.to(device)
    objective = DiffusionPolicyObjective(algorithm=args.algorithm)

    run_name = f"{args.algorithm}_{args.property}_seed{args.seed}"
    if args.pirl:
        run_name += "_pirl"
    model_dir, results_dir = _run_output_paths(
        args.output_root, run_name, args.run_kind
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.model_path / "hparams.yaml", model_dir / "hparams.yaml")
    source_checkpoint = resolve_checkpoint(args.model_path)

    metrics_path = results_dir / "metrics.jsonl"

    for step in range(start_step, start_step + args.updates):
        old_policy = copy.deepcopy(model).eval() if args.pirl else None
        old_decoder_state = None
        old_optimizer_state = None
        if args.pirl:
            old_decoder_state = {
                name: parameter.detach().clone()
                for name, parameter in model.decoder.named_parameters()
            }
            old_optimizer_state = copy.deepcopy(optimizer.state_dict())

        _, _, trajectory = model.sample_rl(
            batch,
            step_lr=args.step_lr,
            guidance_scale=0.0,
            noise_seed=args.seed + step,
            diff_ratio=args.short_diff_ratio,
        )
        evaluation = adapter.evaluate_state(
            final_state=trajectory.final_state,
            num_atoms=trajectory.num_atoms,
            crystal_family=model.crystal_family,
        )
        rewards = evaluation.reward.raw_reward.detach()
        advantages = rewards
        if args.algorithm == "ppo":
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1.0e-8
            )

        optimizer.zero_grad(set_to_none=True)
        selected = _stochastic_transition_indices(
            trajectory, args.replay_transitions
        )
        result = objective(
            model=model,
            batch=batch,
            trajectory=trajectory,
            step_lr=args.step_lr,
            rewards=rewards if args.algorithm == "grpo" else None,
            advantages=advantages if args.algorithm == "ppo" else None,
            group_index=group_index if args.algorithm == "grpo" else None,
            guidance_scale=0.0,
            transition_indices=selected,
            reference_weight=args.reference_weight,
        )
        loss = result.loss
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        if not torch.isfinite(gradient_norm) or float(gradient_norm) == 0.0:
            raise RuntimeError(f"Invalid decoder gradient norm: {float(gradient_norm)}")
        optimizer.step()

        decision_record = None
        if args.pirl and old_policy is not None:
            paired = paired_policy_rollout(
                old_policy=old_policy,
                new_policy=model,
                batch=batch,
                noise_seed=args.probe_seed,
                diff_ratio=args.short_diff_ratio,
                step_lr=args.step_lr,
                guidance_scale=0.0,
            )
            old_eval = adapter.evaluate_state(
                paired.old_trajectory.final_state,
                paired.old_trajectory.num_atoms,
                model.crystal_family,
            )
            new_eval = adapter.evaluate_state(
                paired.new_trajectory.final_state,
                paired.new_trajectory.num_atoms,
                model.crystal_family,
            )
            safety_tolerance = float(contract.closed_loop.safety_drop_tolerance)
            decision = decide_reward_improvement(
                old_reward=old_eval.reward.raw_reward,
                new_reward=new_eval.reward.raw_reward,
                old_safety_metrics={
                    "validity": old_eval.valid.float()
                },
                new_safety_metrics={
                    "validity": new_eval.valid.float()
                },
                safety_tolerances={"validity": safety_tolerance},
                confidence=float(contract.closed_loop.confidence),
                n_bootstrap=int(contract.closed_loop.bootstrap_samples),
                attenuation=float(contract.closed_loop.attenuation_scale),
                seed=args.probe_seed + step,
            )
            if decision.update_scale < 1.0:
                assert old_decoder_state is not None
                assert old_optimizer_state is not None
                _blend_decoder(model, old_decoder_state, decision.update_scale)
                optimizer.load_state_dict(old_optimizer_state)
            decision_record = _decision_summary(decision)
            del old_policy

        scheduler.step()
        record = {
            "step": step,
            "algorithm": args.algorithm,
            "property": args.property,
            "pirl": args.pirl,
            "loss": float(loss.detach().item()),
            "gradient_norm": float(gradient_norm),
            "ratio_mean": float(result.metrics["ratio_mean"].item()),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "selected_transitions": selected,
            "reward": _reward_summary(evaluation),
            "decision": decision_record,
        }
        checkpoint = _save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            source_checkpoint=source_checkpoint,
            output_dir=model_dir,
            step=step,
            metadata=record,
        )
        record["checkpoint"] = str(checkpoint)
        _append_jsonl(metrics_path, record)
        print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
