"""Run a minimal PPO/GRPO update with real crystal-property rewards."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from cgdit.common.evaluation_utils import load_model
from cgdit.evaluation.generated_properties import resolve_checkpoint
from cgdit.rl.config import load_rl_config
from cgdit.rl.objectives import group_relative_advantages
from cgdit.rl.pipo import normalize_group_attributions, pipo_feedback
from cgdit.rl.paired_probe import paired_policy_rollout
from cgdit.rl.policy_improvement import (
    decide_dual_baseline_reward_improvement,
    decide_reward_improvement,
)
from cgdit.rl.property_reward import M3GNetRewardAdapter, PropertyRewardSpec
from cgdit.rl.symmetry_quotient import representative_indices
from cgdit.rl.trainer import DiffusionPolicyObjective
from cgdit.rl.trajectory import CrystalState, RLTrajectory, RLTransition


PROPERTY_NAMES = {
    "fe": ("formation_energy_per_atom",),
    "bg": ("band_gap",),
    "metal": ("band_gap",),
    "joint": ("formation_energy_per_atom", "band_gap"),
}


def _config_defaults(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config = OmegaConf.to_container(load_rl_config(path), resolve=True)
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
        default=Path("conf/rl/components/rewards/contracts/mp20.yaml"),
    )
    parser.add_argument(
        "--reward-registry",
        type=Path,
        default=Path("conf/rl/components/rewards/registries/mp20.yaml"),
    )
    parser.add_argument("--property", choices=PROPERTY_NAMES, default="fe")
    parser.add_argument("--algorithm", choices=("ppo", "grpo"), default="grpo")
    parser.add_argument("--pirl", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pirl-verification-interval", type=int, default=1)
    parser.add_argument("--pipo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pipo-history-window", type=int, default=8)
    parser.add_argument("--pipo-negative-scale", type=float, default=0.1)
    parser.add_argument(
        "--audit-policy-updates",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--output-root", type=Path, default=Path("output/reinforcement_learning"))
    parser.add_argument("--run-kind", choices=("train", "test"), default="train")
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--probe-num-prompts", type=int)
    parser.add_argument("--probe-group-size", type=int)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--short-diff-ratio", type=float, default=0.02)
    parser.add_argument("--step-lr", type=float, default=1.0e-5)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--lr-decay", type=float, default=1.0)
    parser.add_argument("--policy-epochs", type=int, default=1)
    parser.add_argument("--policy-microbatch-prompts", type=int, default=0)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--replay-transitions", type=int, default=2)
    parser.add_argument("--predictor-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-seed", type=int, default=4242)
    parser.add_argument("--reference-weight", type=float, default=0.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-metrics", type=Path)
    parser.add_argument("--max-prompt-atoms", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--final-paired-evaluation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--final-probe-seed", type=int, default=14242)
    parser.add_argument("--holdout-probe-seed", type=int, default=24242)
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    if args.probe_num_prompts is None:
        args.probe_num_prompts = args.num_prompts
    if args.probe_group_size is None:
        args.probe_group_size = args.group_size

    if args.num_prompts < 1 or args.group_size < 2:
        parser.error("num-prompts must be >= 1 and group-size must be >= 2")
    if args.probe_num_prompts < 1 or args.probe_group_size < 1:
        parser.error(
            "probe-num-prompts and probe-group-size must both be >= 1"
        )
    if args.updates < 1 or args.replay_transitions < 1:
        parser.error("updates and replay-transitions must be >= 1")
    if args.checkpoint_every < 1:
        parser.error("checkpoint-every must be >= 1")
    if args.policy_epochs < 1:
        parser.error("policy-epochs must be >= 1")
    if args.policy_microbatch_prompts < 0:
        parser.error("policy-microbatch-prompts must be >= 0")
    if not 0.0 < args.short_diff_ratio <= 1.0:
        parser.error("short-diff-ratio must be in (0, 1]")
    if not 0.0 < args.clip_epsilon < 1.0:
        parser.error("clip-epsilon must be in (0, 1)")
    if args.pipo and args.pirl:
        parser.error("PIPO and PIRL are separate methods and cannot be enabled together")
    if args.resume_metrics is not None and args.resume is None:
        parser.error("resume-metrics requires resume")
    if args.pipo and args.resume is not None and args.resume_metrics is None:
        parser.error("PIPO resume requires resume-metrics")
    if args.max_prompt_atoms is not None and args.max_prompt_atoms < 1:
        parser.error("max-prompt-atoms must be >= 1")
    if args.pipo_history_window < 2:
        parser.error("pipo-history-window must be >= 2")
    if args.pirl_verification_interval < 1:
        parser.error("pirl-verification-interval must be >= 1")
    if not args.pirl and args.pirl_verification_interval != 1:
        parser.error("pirl-verification-interval requires --pirl")
    if not 0.0 <= args.pipo_negative_scale <= 1.0:
        parser.error("pipo-negative-scale must be in [0, 1]")
    return args


def _set_seed(seed: int, deterministic: bool = True) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = not deterministic
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = not deterministic
    torch.use_deterministic_algorithms(deterministic)


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
    required_safety = {
        str(name) for name in contract.closed_loop.safety_metrics
    }
    pilot_safety = registry.get("pilot_safety", {})
    stability = (
        pilot_safety.get("stability")
        if "stability" in required_safety
        else None
    )
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
        stability_checkpoint=(
            str(Path(stability.checkpoint)) if stability is not None else None
        ),
        stability_target=(
            float(stability.target) if stability is not None else 0.0
        ),
        stability_tolerance=(
            float(stability.tolerance) if stability is not None else 0.3
        ),
    )


def _required_paths(
    args: argparse.Namespace,
    registry: Any,
) -> list[Path]:
    paths = [
        args.model_path,
        args.model_path / "hparams.yaml",
        Path("data/mp_20/train.csv"),
        Path("data/mp_20/train_sym.pt"),
        Path("data/mp_20/val.csv"),
        Path("data/mp_20/val_sym.pt"),
    ]
    for name in PROPERTY_NAMES[args.property]:
        paths.append(Path(registry.models[name].checkpoint))
    pilot_safety = registry.get("pilot_safety", {})
    if pilot_safety.get("stability") is not None:
        paths.append(Path(pilot_safety.stability.checkpoint))
    if args.resume is not None:
        paths.append(Path(args.resume))
    if args.resume_metrics is not None:
        paths.append(Path(args.resume_metrics))
    return paths


def _validate_paths(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def _prompt_indices(
    dataset_size: int,
    num_prompts: int,
    seed: int,
    excluded: list[int] | tuple[int, ...] = (),
) -> list[int]:
    excluded_set = set(excluded)
    available = [
        index for index in range(dataset_size) if index not in excluded_set
    ]
    if len(available) < num_prompts:
        raise ValueError(
            f"Dataset has {len(available)} available entries, "
            f"fewer than num-prompts={num_prompts}"
        )
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(available), generator=generator)[:num_prompts]
    return [available[index] for index in order.tolist()]


def _oversized_prompt_indices(
    dataset: Any, max_prompt_atoms: int | None
) -> list[int]:
    if max_prompt_atoms is None:
        return []
    oversized = []
    for index in range(len(dataset)):
        num_atoms = dataset[index].num_atoms
        count = (
            int(num_atoms.item())
            if hasattr(num_atoms, "item")
            else int(num_atoms)
        )
        if count > max_prompt_atoms:
            oversized.append(index)
    return oversized


def _resume_records(
    path: Path | None, start_step: int
) -> list[dict[str, Any]]:
    if path is None:
        return []
    records_by_step = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        step = int(record["step"])
        if step < start_step:
            if step in records_by_step:
                raise ValueError(f"Duplicate resume metric for step {step}")
            records_by_step[step] = record
    expected = list(range(start_step))
    if sorted(records_by_step) != expected:
        raise ValueError(
            f"Resume metrics must contain exactly steps 0..{start_step - 1}"
        )
    return [records_by_step[step] for step in expected]


def _grouped_batch(
    dataset: Any,
    num_prompts: int,
    group_size: int,
    prompt_indices: list[int] | None = None,
) -> tuple[Batch, torch.Tensor]:
    if prompt_indices is None:
        prompt_indices = list(range(num_prompts))
    if len(prompt_indices) != num_prompts:
        raise ValueError("prompt_indices must contain num-prompts entries")
    if len(set(prompt_indices)) != len(prompt_indices):
        raise ValueError("prompt_indices must be unique")
    if any(index < 0 or index >= len(dataset) for index in prompt_indices):
        raise IndexError("prompt index is outside the dataset")
    data = []
    group_index = []
    for group, prompt_index in enumerate(prompt_indices):
        prompt = dataset[prompt_index]
        for _ in range(group_size):
            data.append(prompt.clone())
            group_index.append(group)
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


def _pirl_verification_due(
    step: int,
    block_start_step: int,
    interval: int,
    final_step: int,
) -> bool:
    """Return whether an accumulated PIRL candidate must be verified now."""
    if step < block_start_step:
        raise ValueError("step must not precede the PIRL block start")
    if interval < 1:
        raise ValueError("PIRL verification interval must be positive")
    return step == final_step or step - block_start_step + 1 >= interval


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


def _metallicity_proxy_summary(
    evaluation: Any, contract: Any
) -> dict[str, Any] | None:
    proxy = contract.get("metallicity_proxy")
    if proxy is None:
        return None
    threshold = float(proxy.predicted_threshold_eV)
    values = evaluation.predictions["band_gap"]
    valid = evaluation.valid & torch.isfinite(values)
    hits = valid & (values <= threshold)
    valid_count = int(valid.sum().item())
    return {
        "predicted_threshold_eV": threshold,
        "predicted_metal_count": int(hits.sum().item()),
        "predicted_metal_fraction_all": float(hits.float().mean().item()),
        "predicted_metal_fraction_valid": (
            float(hits.sum().item() / valid_count) if valid_count else None
        ),
    }


def _advantage_summary(
    advantages: torch.Tensor, group_index: torch.Tensor
) -> dict[str, Any]:
    groups = {}
    for group in torch.unique(group_index):
        values = advantages[group_index == group]
        groups[str(int(group.item()))] = {
            "mean": float(values.mean().item()),
            "std": float(values.std(unbiased=False).item()),
        }
    return {
        "mean": float(advantages.mean().item()),
        "std": float(advantages.std(unbiased=False).item()),
        "groups": groups,
    }


def _trajectory_summary(trajectory: Any) -> dict[str, Any]:
    stacked = trajectory.stacked_old_log_probs()
    batch_index = trajectory.metadata["batch"].long()
    anchor_index = trajectory.metadata["anchor_index"].long()
    orbit_pairs = torch.unique(
        torch.stack((batch_index, anchor_index), dim=1), dim=0
    )
    num_graphs = int(trajectory.num_atoms.numel())
    orbit_counts = torch.bincount(
        orbit_pairs[:, 0], minlength=num_graphs
    )
    timesteps = [transition.timestep for transition in trajectory.transitions]
    return {
        "num_transitions": len(trajectory.transitions),
        "num_stochastic_transitions": sum(
            transition.stochastic for transition in trajectory.transitions
        ),
        "timestep_min": min(timesteps),
        "timestep_max": max(timesteps),
        "num_atoms_per_structure": trajectory.num_atoms.detach().cpu().tolist(),
        "num_orbits_per_structure": orbit_counts.detach().cpu().tolist(),
        "channel_log_prob": {
            name: {
                "mean": float(values.mean().item()),
                "std": float(values.std(unbiased=False).item()),
                "min": float(values.min().item()),
                "max": float(values.max().item()),
            }
            for name, values in stacked.items()
        },
    }


def _decision_summary(decision: Any) -> dict[str, Any]:
    summary = {
        "action": decision.action,
        "scale": float(decision.scale),
        "requires_recheck": bool(decision.requires_recheck),
    }
    if hasattr(decision, "local"):
        summary["local"] = _decision_summary(decision.local)
        summary["absolute"] = _decision_summary(decision.absolute)
    else:
        summary["mean_delta"] = decision.mean_delta.detach().cpu().tolist()
        summary["lower_confidence_bound"] = (
            decision.lower_confidence_bound.detach().cpu().tolist()
        )
    return summary


def _safety_audit(
    required: list[str], evaluated: list[str], run_kind: str
) -> dict[str, Any]:
    missing = [name for name in required if name not in evaluated]
    formal_complete = not missing
    if run_kind == "train" and not formal_complete:
        raise RuntimeError(
            "Formal PIRL training requires all safety metrics; missing: "
            + ", ".join(missing)
        )
    return {
        "evaluated": evaluated,
        "missing": missing,
        "formal_complete": formal_complete,
        "verification_status": "verified" if formal_complete else "diagnostic_only",
    }


def _evaluation_safety_metrics(
    evaluation: Any, required: list[str]
) -> dict[str, torch.Tensor]:
    available = dict(getattr(evaluation, "safety_metrics", {}))
    available["validity"] = evaluation.valid.float()
    expected_shape = evaluation.valid.shape
    metrics = {}
    for name in required:
        if name not in available:
            continue
        value = available[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"safety metric '{name}' must be a tensor")
        if value.shape != expected_shape:
            raise ValueError(
                f"safety metric '{name}' must align with generated structures"
            )
        value = value.float()
        if not torch.isfinite(value).all():
            raise ValueError(f"safety metric '{name}' must be finite")
        metrics[name] = value
    return metrics


def _shared_safety_metrics(
    base_evaluation: Any,
    current_evaluation: Any,
    candidate_evaluation: Any,
    required: list[str],
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    list[str],
]:
    base = _evaluation_safety_metrics(base_evaluation, required)
    current = _evaluation_safety_metrics(current_evaluation, required)
    candidate = _evaluation_safety_metrics(candidate_evaluation, required)
    evaluated = [
        name
        for name in required
        if name in base and name in current and name in candidate
    ]
    return (
        {name: base[name] for name in evaluated},
        {name: current[name] for name in evaluated},
        {name: candidate[name] for name in evaluated},
        evaluated,
    )


def _configured_safety_tolerances(
    contract: Any, evaluated: list[str]
) -> dict[str, float]:
    """Return pre-registered base-anchored anti-collapse tolerances."""
    closed_loop = contract.closed_loop
    configured = closed_loop.get("safety_drop_tolerances")
    if configured is None:
        tolerance = float(closed_loop.safety_drop_tolerance)
        return {name: tolerance for name in evaluated}
    missing = [name for name in evaluated if name not in configured]
    if missing:
        raise ValueError(
            "Missing safety drop tolerances for: " + ", ".join(missing)
        )
    return {name: float(configured[name]) for name in evaluated}


def _verified_update_outcome(
    *,
    probe_action: str,
    safety_complete: bool,
    holdout_action: str | None,
) -> tuple[str, bool, str | None]:
    if probe_action != "accept":
        return "reject", False, f"probe_{probe_action}"
    if not safety_complete:
        return "reject", False, "incomplete_safety"
    if holdout_action is None:
        return "reject", False, "missing_holdout"
    if holdout_action != "accept":
        return "rollback", False, f"holdout_{holdout_action}"
    return "accept", True, None


def _evaluate_dual_baseline_candidate(
    *,
    frozen_base_state: dict[str, torch.Tensor],
    current_policy: torch.nn.Module,
    candidate_policy: torch.nn.Module,
    batch: Batch,
    adapter: M3GNetRewardAdapter,
    contract: Any,
    noise_seed: int,
    diff_ratio: float,
    step_lr: float,
) -> tuple[Any, Any, Any, Any]:
    local_pair = paired_policy_rollout(
        old_policy=current_policy,
        new_policy=candidate_policy,
        batch=batch,
        noise_seed=noise_seed,
        diff_ratio=diff_ratio,
        step_lr=step_lr,
        guidance_scale=0.0,
    )
    current_eval = adapter.evaluate_state(
        local_pair.old_trajectory.final_state,
        local_pair.old_trajectory.num_atoms,
        candidate_policy.crystal_family,
    )
    candidate_eval = adapter.evaluate_state(
        local_pair.new_trajectory.final_state,
        local_pair.new_trajectory.num_atoms,
        candidate_policy.crystal_family,
    )

    current_policy.load_state_dict(frozen_base_state, strict=True)
    absolute_pair = paired_policy_rollout(
        old_policy=current_policy,
        new_policy=candidate_policy,
        batch=batch,
        noise_seed=noise_seed,
        diff_ratio=diff_ratio,
        step_lr=step_lr,
        guidance_scale=0.0,
    )
    base_eval = adapter.evaluate_state(
        absolute_pair.old_trajectory.final_state,
        absolute_pair.old_trajectory.num_atoms,
        candidate_policy.crystal_family,
    )
    required_safety = [
        str(name) for name in contract.closed_loop.safety_metrics
    ]
    base_safety, current_safety, candidate_safety, evaluated_safety = (
        _shared_safety_metrics(
            base_eval, current_eval, candidate_eval, required_safety
        )
    )
    decision = decide_dual_baseline_reward_improvement(
        base_reward=base_eval.reward.raw_reward,
        current_reward=current_eval.reward.raw_reward,
        candidate_reward=candidate_eval.reward.raw_reward,
        base_safety_metrics=base_safety,
        current_safety_metrics=current_safety,
        candidate_safety_metrics=candidate_safety,
        safety_tolerances=_configured_safety_tolerances(
            contract, evaluated_safety
        ),
        confidence=float(contract.closed_loop.confidence),
        n_bootstrap=int(contract.closed_loop.bootstrap_samples),
        attenuation=float(contract.closed_loop.attenuation_scale),
        seed=noise_seed,
    )
    return decision, base_eval, current_eval, candidate_eval


def _audit_policy_candidate(
    *,
    frozen_base_state: dict[str, torch.Tensor],
    current_policy: torch.nn.Module,
    candidate_policy: torch.nn.Module,
    probe_batch: Batch,
    holdout_batch: Batch,
    adapter: M3GNetRewardAdapter,
    contract: Any,
    probe_seed: int,
    holdout_seed: int,
    diff_ratio: float,
    step_lr: float,
    run_kind: str,
) -> dict[str, Any]:
    records = {}
    required_safety = [
        str(name) for name in contract.closed_loop.safety_metrics
    ]
    for name, batch, noise_seed in (
        ("fixed", probe_batch, probe_seed),
        ("holdout", holdout_batch, holdout_seed),
    ):
        current_copy = copy.deepcopy(current_policy).eval()
        decision, base_eval, current_eval, candidate_eval = (
            _evaluate_dual_baseline_candidate(
                frozen_base_state=frozen_base_state,
                current_policy=current_copy,
                candidate_policy=candidate_policy,
                batch=batch,
                adapter=adapter,
                contract=contract,
                noise_seed=noise_seed,
                diff_ratio=diff_ratio,
                step_lr=step_lr,
            )
        )
        _, _, _, evaluated_safety = _shared_safety_metrics(
            base_eval,
            current_eval,
            candidate_eval,
            required_safety,
        )
        records[name] = {
            "decision": _decision_summary(decision),
            "base": _reward_summary(base_eval),
            "current": _reward_summary(current_eval),
            "candidate": _reward_summary(candidate_eval),
            "safety_audit": _safety_audit(
                required_safety, evaluated_safety, run_kind
            ),
        }
        del current_copy
    return records


def _slice_rl_trajectory(
    trajectory: RLTrajectory,
    batch: Batch,
    micro_batch: Batch,
    graph_indices: torch.Tensor,
) -> RLTrajectory:
    """Select a graph-aligned trajectory view for policy replay."""
    graph_mask = torch.zeros(
        batch.num_graphs, dtype=torch.bool, device=batch.batch.device
    )
    graph_mask[graph_indices] = True
    node_mask = graph_mask[batch.batch]
    representatives = representative_indices(batch.anchor_index)
    representative_mask = graph_mask[batch.batch[representatives]]

    def select_state(state: CrystalState) -> CrystalState:
        return CrystalState(
            atom_types=state.atom_types[node_mask],
            frac_coords=state.frac_coords[node_mask],
            crys_fam=state.crys_fam[graph_indices],
        )

    def select_record(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        selected = {}
        for name, value in values.items():
            if value.shape[0] == batch.num_graphs:
                selected[name] = value[graph_indices]
            elif value.shape[0] == batch.num_nodes:
                selected[name] = value[node_mask]
            elif value.shape[0] == representatives.numel():
                selected[name] = value[representative_mask]
            else:
                raise ValueError(
                    f"Cannot align trajectory field '{name}' with graph microbatch"
                )
        return selected

    transitions = [
        RLTransition(
            timestep=transition.timestep,
            state_t=select_state(transition.state_t),
            state_half=select_state(transition.state_half),
            state_next=select_state(transition.state_next),
            old_log_prob_by_channel={
                name: value[graph_indices]
                for name, value in transition.old_log_prob_by_channel.items()
            },
            actions_by_channel=select_record(transition.actions_by_channel),
            noise_by_channel=select_record(transition.noise_by_channel),
            stochastic=transition.stochastic,
        )
        for transition in trajectory.transitions
    ]
    return RLTrajectory(
        transitions=transitions,
        final_state=select_state(trajectory.final_state),
        num_atoms=trajectory.num_atoms[graph_indices],
        metadata={
            "spacegroup": micro_batch.spacegroup,
            "anchor_index": micro_batch.anchor_index,
            "batch": micro_batch.batch,
        },
    )


def _policy_microbatch_count(
    group_index: torch.Tensor, prompts_per_microbatch: int
) -> int:
    prompt_count = int(torch.unique(group_index).numel())
    if prompts_per_microbatch <= 0 or prompts_per_microbatch >= prompt_count:
        return 1
    return (prompt_count + prompts_per_microbatch - 1) // prompts_per_microbatch


def _policy_microbatches(
    *,
    batch: Batch,
    trajectory: RLTrajectory,
    rewards: torch.Tensor,
    advantages: torch.Tensor,
    group_index: torch.Tensor,
    prompts_per_microbatch: int,
):
    groups = torch.unique(group_index, sorted=True)
    if prompts_per_microbatch <= 0 or prompts_per_microbatch >= groups.numel():
        yield batch, trajectory, rewards, advantages, group_index
        return

    for group_chunk in groups.split(prompts_per_microbatch):
        graph_mask = (group_index[:, None] == group_chunk[None, :]).any(dim=1)
        graph_indices = torch.nonzero(graph_mask, as_tuple=False).flatten()
        micro_batch = Batch.from_data_list(
            batch.index_select(graph_indices.detach().cpu())
        ).to(batch.batch.device)
        yield (
            micro_batch,
            _slice_rl_trajectory(trajectory, batch, micro_batch, graph_indices),
            rewards[graph_indices],
            advantages[graph_indices],
            group_index[graph_indices],
        )


def _optimize_policy_epochs(
    *,
    objective: DiffusionPolicyObjective,
    model: torch.nn.Module,
    batch: Batch,
    trajectory: Any,
    step_lr: float,
    rewards: torch.Tensor,
    advantages: torch.Tensor,
    group_index: torch.Tensor,
    transition_indices: list[int],
    reference_weight: float,
    optimizer: torch.optim.Optimizer,
    trainable_parameters: list[torch.nn.Parameter],
    policy_epochs: int,
    policy_microbatch_prompts: int = 0,
) -> tuple[Any, torch.Tensor, torch.Tensor, list[dict[str, float | int]]]:
    epoch_records = []
    result = None
    loss = None
    gradient_norm = None
    for policy_epoch in range(policy_epochs):
        optimizer.zero_grad(set_to_none=True)
        transition_weight = 1.0 / len(transition_indices)
        loss = rewards.new_zeros(())
        metrics: dict[str, torch.Tensor] = {}
        for transition_index in transition_indices:
            for microbatch in _policy_microbatches(
                batch=batch,
                trajectory=trajectory,
                rewards=rewards,
                advantages=advantages,
                group_index=group_index,
                prompts_per_microbatch=policy_microbatch_prompts,
            ):
                (
                    micro_batch,
                    micro_trajectory,
                    micro_rewards,
                    micro_advantages,
                    micro_group_index,
                ) = microbatch
                batch_weight = micro_rewards.numel() / rewards.numel()
                weight = transition_weight * batch_weight
                chunk = objective(
                    model=model,
                    batch=micro_batch,
                    trajectory=micro_trajectory,
                    step_lr=step_lr,
                    rewards=(
                        micro_rewards if objective.algorithm == "grpo" else None
                    ),
                    advantages=micro_advantages,
                    group_index=(
                        micro_group_index if objective.algorithm == "grpo" else None
                    ),
                    guidance_scale=0.0,
                    transition_indices=[transition_index],
                    reference_weight=reference_weight,
                )
                (chunk.loss * weight).backward()
                loss = loss + chunk.loss.detach() * weight
                for name, value in chunk.metrics.items():
                    metrics[name] = metrics.get(
                        name, torch.zeros_like(value.detach())
                    ) + value.detach() * weight
                del chunk, microbatch
                del micro_batch, micro_trajectory
                del micro_rewards, micro_advantages, micro_group_index
        result = SimpleNamespace(metrics=metrics)
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        if not torch.isfinite(gradient_norm) or float(gradient_norm) == 0.0:
            raise RuntimeError(
                f"Invalid decoder gradient norm: {float(gradient_norm)}"
            )
        optimizer.step()
        epoch_records.append(
            {
                "epoch": policy_epoch,
                "loss": float(loss.detach().item()),
                "gradient_norm": float(gradient_norm),
                "approx_kl": float(result.metrics["approx_kl"].item()),
                "clip_fraction": float(result.metrics["clip_fraction"].item()),
                "ratio_mean": float(result.metrics["ratio_mean"].item()),
                "microbatches": _policy_microbatch_count(
                    group_index, policy_microbatch_prompts
                ),
            }
        )
    assert result is not None
    assert loss is not None
    assert gradient_norm is not None
    return result, loss, gradient_norm, epoch_records


def _evaluate_final_policy(
    *,
    model: torch.nn.Module,
    initial_state: dict[str, torch.Tensor],
    batch: Batch,
    adapter: M3GNetRewardAdapter,
    contract: Any,
    noise_seed: int,
    diff_ratio: float,
    step_lr: float,
    run_kind: str,
) -> dict[str, Any]:
    baseline = copy.deepcopy(model).eval()
    baseline.load_state_dict(initial_state, strict=True)
    paired = paired_policy_rollout(
        old_policy=baseline,
        new_policy=model,
        batch=batch,
        noise_seed=noise_seed,
        diff_ratio=diff_ratio,
        step_lr=step_lr,
        guidance_scale=0.0,
    )
    old_eval = adapter.evaluate_state(
        final_state=paired.old_trajectory.final_state,
        num_atoms=paired.old_trajectory.num_atoms,
        crystal_family=model.crystal_family,
    )
    new_eval = adapter.evaluate_state(
        final_state=paired.new_trajectory.final_state,
        num_atoms=paired.new_trajectory.num_atoms,
        crystal_family=model.crystal_family,
    )
    required_safety = [str(name) for name in contract.closed_loop.safety_metrics]
    old_safety = _evaluation_safety_metrics(old_eval, required_safety)
    new_safety = _evaluation_safety_metrics(new_eval, required_safety)
    evaluated_safety = [
        name for name in required_safety if name in old_safety and name in new_safety
    ]
    safety_audit = _safety_audit(required_safety, evaluated_safety, run_kind)
    decision = decide_reward_improvement(
        old_reward=old_eval.reward.raw_reward,
        new_reward=new_eval.reward.raw_reward,
        old_safety_metrics={name: old_safety[name] for name in evaluated_safety},
        new_safety_metrics={name: new_safety[name] for name in evaluated_safety},
        safety_tolerances=_configured_safety_tolerances(
            contract, evaluated_safety
        ),
        confidence=float(contract.closed_loop.confidence),
        n_bootstrap=int(contract.closed_loop.bootstrap_samples),
        attenuation=float(contract.closed_loop.attenuation_scale),
        seed=noise_seed,
    )
    del baseline
    return {
        "noise_seed": noise_seed,
        "baseline": _reward_summary(old_eval),
        "final": _reward_summary(new_eval),
        "baseline_metallicity_proxy": _metallicity_proxy_summary(
            old_eval, contract
        ),
        "final_metallicity_proxy": _metallicity_proxy_summary(new_eval, contract),
        "safety_audit": safety_audit,
        "decision": _decision_summary(decision),
    }


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_output_paths(
    output_root: Path, run_name: str, run_kind: str
) -> tuple[Path, Path]:
    if output_root == Path("output/reinforcement_learning"):
        if run_kind == "test":
            output_root = Path("output/test/reinforcement_learning")
        stamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        run_root = output_root / f"{stamp}-{run_name}"
    else:
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
    _set_seed(args.seed, deterministic=args.deterministic)
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
    model, prompt_loaders, _ = load_model(
        args.model_path,
        load_data=True,
        testing=False,
    )
    if prompt_loaders is None:
        raise RuntimeError("The base model did not provide train/validation datasets")
    train_loader, val_loader = prompt_loaders
    model = model.to(device).eval()
    initial_state = (
        {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        if args.final_paired_evaluation
        else None
    )
    frozen_base_state = (
        {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        if args.pirl or args.audit_policy_updates
        else None
    )
    trainable_parameters = _trainable_decoder(model)
    adapter = _adapter(args, contract, registry, device)

    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    start_step = 0
    if args.resume is not None:
        start_step = _restore_training_state(args.resume, model, optimizer, scheduler)
    prior_records = _resume_records(args.resume_metrics, start_step)

    training_dataset = train_loader.dataset
    validation_dataset = val_loader.dataset
    oversized_training_prompts = _oversized_prompt_indices(
        training_dataset, args.max_prompt_atoms
    )
    oversized_validation_prompts = _oversized_prompt_indices(
        validation_dataset, args.max_prompt_atoms
    )
    probe_prompt_indices = _prompt_indices(
        len(validation_dataset),
        args.probe_num_prompts,
        args.probe_seed,
        excluded=oversized_validation_prompts,
    )
    holdout_prompt_indices = _prompt_indices(
        len(validation_dataset),
        args.probe_num_prompts,
        args.holdout_probe_seed,
        excluded=oversized_validation_prompts + probe_prompt_indices,
    )
    final_prompt_indices = _prompt_indices(
        len(validation_dataset),
        args.probe_num_prompts,
        args.final_probe_seed,
        excluded=(
            oversized_validation_prompts
            + probe_prompt_indices
            + holdout_prompt_indices
        ),
    )
    probe_batch, _ = _grouped_batch(
        validation_dataset,
        args.probe_num_prompts,
        args.probe_group_size,
        prompt_indices=probe_prompt_indices,
    )
    holdout_batch, _ = _grouped_batch(
        validation_dataset,
        args.probe_num_prompts,
        args.probe_group_size,
        prompt_indices=holdout_prompt_indices,
    )
    final_batch, _ = _grouped_batch(
        validation_dataset,
        args.probe_num_prompts,
        args.probe_group_size,
        prompt_indices=final_prompt_indices,
    )
    if args.pirl or args.audit_policy_updates:
        probe_batch = probe_batch.to(device)
        holdout_batch = holdout_batch.to(device)
    objective = DiffusionPolicyObjective(
        algorithm=args.algorithm,
        clip_epsilon=args.clip_epsilon,
    )
    run_name = f"{args.algorithm}_{args.property}_seed{args.seed}"
    if args.pirl:
        run_name += "_pirl"
    if args.pipo:
        run_name += "_pipo"
    model_dir, results_dir = _run_output_paths(
        args.output_root, run_name, args.run_kind
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.model_path / "hparams.yaml", model_dir / "hparams.yaml")
    for scaler_name in ("lattice_scaler.pt", "prop_scaler.pt"):
        scaler_path = args.model_path / scaler_name
        if scaler_path.exists():
            shutil.copy2(scaler_path, model_dir / scaler_name)
    source_checkpoint = resolve_checkpoint(args.model_path)

    metrics_path = results_dir / "metrics.jsonl"
    if prior_records:
        if metrics_path.exists() and metrics_path.stat().st_size:
            raise FileExistsError(
                f"Resume output metrics already exist: {metrics_path}"
            )
        for record in prior_records:
            _append_jsonl(metrics_path, record)

    prior_training_prompts = [
        int(index)
        for record in prior_records
        for index in record["training_prompt_indices"]
    ]
    scheduled_prompt_count = args.num_prompts * args.updates
    training_prompt_schedule = _prompt_indices(
        len(training_dataset),
        scheduled_prompt_count,
        args.seed,
        excluded=oversized_training_prompts + prior_training_prompts,
    )

    reward_history = [
        float(record["reward"]["reward_mean"])
        for record in prior_records[-args.pipo_history_window :]
    ]
    previous_replay: dict[str, Any] | None = None
    final_step = start_step + args.updates - 1
    if args.pirl and start_step % args.pirl_verification_interval != 0:
        raise ValueError(
            "PIRL resume must start at a completed verification-block boundary"
        )
    pirl_block_start_step: int | None = None
    pirl_anchor_policy: torch.nn.Module | None = None
    pirl_anchor_decoder_state: dict[str, torch.Tensor] | None = None
    pirl_anchor_optimizer_state: dict[str, Any] | None = None
    pirl_anchor_scheduler_state: dict[str, Any] | None = None
    for step in range(start_step, start_step + args.updates):
        prompt_start = (step - start_step) * args.num_prompts
        training_prompt_indices = training_prompt_schedule[
            prompt_start : prompt_start + args.num_prompts
        ]
        batch, group_index = _grouped_batch(
            training_dataset,
            args.num_prompts,
            args.group_size,
            training_prompt_indices,
        )
        batch = batch.to(device)
        group_index = group_index.to(device)
        old_policy = (
            copy.deepcopy(model).eval()
            if args.audit_policy_updates and not args.pirl
            else None
        )
        if args.pirl and pirl_anchor_policy is None:
            pirl_block_start_step = step
            pirl_anchor_policy = copy.deepcopy(model).eval()
            pirl_anchor_decoder_state = {
                name: parameter.detach().clone()
                for name, parameter in model.decoder.named_parameters()
            }
            pirl_anchor_optimizer_state = copy.deepcopy(optimizer.state_dict())
            pirl_anchor_scheduler_state = copy.deepcopy(scheduler.state_dict())
        assert pirl_block_start_step is not None or not args.pirl
        pirl_verification_due = bool(
            args.pirl
            and _pirl_verification_due(
                step,
                pirl_block_start_step,
                args.pirl_verification_interval,
                final_step,
            )
        )

        _, _, trajectory = model.sample_rl(
            batch,
            step_lr=args.step_lr,
            guidance_scale=0.0,
            noise_seed=args.seed + step,
            diff_ratio=args.short_diff_ratio,
            replay_transitions=args.replay_transitions,
            retain_trajectory_stack=False,
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
        else:
            advantages = group_relative_advantages(rewards, group_index)

        selected = _stochastic_transition_indices(
            trajectory, args.replay_transitions
        )
        pipo_record = None
        pipo_attributions = None
        if args.pipo:
            pipo_attributions = (
                normalize_group_attributions(advantages, group_index)
                if args.algorithm == "grpo"
                else advantages
            )
            history_window = reward_history[-args.pipo_history_window :]
            feedback = None
            if len(history_window) == args.pipo_history_window:
                feedback = pipo_feedback(
                    current_mean=float(rewards.mean().item()),
                    history=history_window,
                    negative_scale=args.pipo_negative_scale,
                )
            pipo_record = {
                "window_size": args.pipo_history_window,
                "window_ready": len(history_window) == args.pipo_history_window,
                "history": list(history_window),
                "feedback": None,
                "retrospective_update": False,
            }
            if feedback is not None:
                pipo_record["feedback"] = {
                    "history_mean": feedback.history_mean,
                    "history_std": feedback.history_std,
                    "signal": feedback.signal,
                    "modulation": feedback.modulation,
                }
                if (
                    previous_replay is not None
                    and abs(feedback.modulation) > 1.0e-12
                ):
                    retrospective_advantages = (
                        previous_replay["attributions"] * feedback.modulation
                    )
                    (
                        _,
                        retrospective_loss,
                        retrospective_gradient_norm,
                        retrospective_epochs,
                    ) = _optimize_policy_epochs(
                        objective=objective,
                        model=model,
                        batch=previous_replay["batch"],
                        trajectory=previous_replay["trajectory"],
                        step_lr=args.step_lr,
                        rewards=previous_replay["rewards"],
                        advantages=retrospective_advantages,
                        group_index=previous_replay["group_index"],
                        transition_indices=previous_replay["transition_indices"],
                        reference_weight=args.reference_weight,
                        optimizer=optimizer,
                        trainable_parameters=trainable_parameters,
                        policy_epochs=args.policy_epochs,
                        policy_microbatch_prompts=args.policy_microbatch_prompts,
                    )
                    pipo_record.update(
                        {
                            "retrospective_update": True,
                            "loss": float(retrospective_loss.detach().item()),
                            "gradient_norm": float(retrospective_gradient_norm),
                            "policy_epochs": retrospective_epochs,
                        }
                    )
            previous_replay = None
            torch.cuda.empty_cache()
        result, loss, gradient_norm, policy_epoch_records = _optimize_policy_epochs(
            objective=objective,
            model=model,
            batch=batch,
            trajectory=trajectory,
            step_lr=args.step_lr,
            rewards=rewards,
            advantages=advantages,
            group_index=group_index,
            transition_indices=selected,
            reference_weight=args.reference_weight,
            optimizer=optimizer,
            trainable_parameters=trainable_parameters,
            policy_epochs=args.policy_epochs,
            policy_microbatch_prompts=args.policy_microbatch_prompts,
        )

        decision_record = None
        audit_record = None
        if args.audit_policy_updates and not args.pirl:
            assert frozen_base_state is not None
            assert old_policy is not None
            audit_record = _audit_policy_candidate(
                frozen_base_state=frozen_base_state,
                current_policy=old_policy,
                candidate_policy=model,
                probe_batch=probe_batch,
                holdout_batch=holdout_batch,
                adapter=adapter,
                contract=contract,
                probe_seed=args.probe_seed,
                holdout_seed=args.holdout_probe_seed,
                diff_ratio=args.short_diff_ratio,
                step_lr=args.step_lr,
                run_kind=args.run_kind,
            )
            del old_policy
        verified_checkpoint_update = not args.pirl
        if args.pirl and not pirl_verification_due:
            assert pirl_block_start_step is not None
            decision_record = {
                "action": "pending",
                "probe_action": None,
                "scale": None,
                "initial": None,
                "recheck": None,
                "base": None,
                "current": None,
                "candidate": None,
                "recheck_candidate": None,
                "safety_audit": None,
                "holdout": None,
                "rollback_reason": None,
                "verified_checkpoint_update": False,
                "block_start_step": pirl_block_start_step,
                "block_end_step": None,
                "updates_in_block": step - pirl_block_start_step + 1,
                "verification_due": False,
            }
        if args.pirl and pirl_verification_due:
            assert pirl_block_start_step is not None
            assert frozen_base_state is not None
            assert pirl_anchor_policy is not None
            assert pirl_anchor_decoder_state is not None
            assert pirl_anchor_optimizer_state is not None
            assert pirl_anchor_scheduler_state is not None
            old_policy = pirl_anchor_policy
            old_decoder_state = pirl_anchor_decoder_state
            old_optimizer_state = pirl_anchor_optimizer_state
            old_scheduler_state = pirl_anchor_scheduler_state
            decision, base_eval, current_eval, candidate_eval = (
                _evaluate_dual_baseline_candidate(
                    frozen_base_state=frozen_base_state,
                    current_policy=old_policy,
                    candidate_policy=model,
                    batch=probe_batch,
                    adapter=adapter,
                    contract=contract,
                    noise_seed=args.probe_seed,
                    diff_ratio=args.short_diff_ratio,
                    step_lr=args.step_lr,
                )
            )
            initial_summary = _decision_summary(decision)
            recheck_summary = None
            recheck_candidate_summary = None
            final_action = decision.action
            applied_scale = decision.scale
            gate_base_eval = base_eval
            gate_current_eval = current_eval
            gate_candidate_eval = candidate_eval

            if decision.action == "attenuate":
                _blend_decoder(model, old_decoder_state, decision.scale)
                optimizer.load_state_dict(old_optimizer_state)
                recheck_current = copy.deepcopy(model).eval()
                _blend_decoder(recheck_current, old_decoder_state, 0.0)
                recheck, recheck_base_eval, recheck_current_eval, recheck_candidate = (
                    _evaluate_dual_baseline_candidate(
                        frozen_base_state=frozen_base_state,
                        current_policy=recheck_current,
                        candidate_policy=model,
                        batch=probe_batch,
                        adapter=adapter,
                        contract=contract,
                        noise_seed=args.probe_seed,
                        diff_ratio=args.short_diff_ratio,
                        step_lr=args.step_lr,
                    )
                )
                recheck_summary = _decision_summary(recheck)
                recheck_candidate_summary = _reward_summary(recheck_candidate)
                gate_base_eval = recheck_base_eval
                gate_current_eval = recheck_current_eval
                gate_candidate_eval = recheck_candidate
                if recheck.action != "accept":
                    _blend_decoder(model, old_decoder_state, 0.0)
                    final_action = "reject"
                    applied_scale = 0.0
                else:
                    final_action = "accept"
                del recheck_current
            elif decision.action == "reject":
                _blend_decoder(model, old_decoder_state, 0.0)
                optimizer.load_state_dict(old_optimizer_state)

            required_safety = [
                str(name) for name in contract.closed_loop.safety_metrics
            ]
            _, _, _, evaluated_safety = _shared_safety_metrics(
                gate_base_eval,
                gate_current_eval,
                gate_candidate_eval,
                required_safety,
            )
            safety_audit = _safety_audit(
                required_safety, evaluated_safety, args.run_kind
            )

            holdout_record = None
            holdout_action = None
            if final_action == "accept" and safety_audit["formal_complete"]:
                holdout_current = copy.deepcopy(model).eval()
                _blend_decoder(holdout_current, old_decoder_state, 0.0)
                (
                    holdout_decision,
                    holdout_base_eval,
                    holdout_current_eval,
                    holdout_candidate_eval,
                ) = _evaluate_dual_baseline_candidate(
                    frozen_base_state=frozen_base_state,
                    current_policy=holdout_current,
                    candidate_policy=model,
                    batch=holdout_batch,
                    adapter=adapter,
                    contract=contract,
                    noise_seed=args.holdout_probe_seed,
                    diff_ratio=args.short_diff_ratio,
                    step_lr=args.step_lr,
                )
                _, _, _, holdout_evaluated = _shared_safety_metrics(
                    holdout_base_eval,
                    holdout_current_eval,
                    holdout_candidate_eval,
                    required_safety,
                )
                holdout_audit = _safety_audit(
                    required_safety, holdout_evaluated, args.run_kind
                )
                if holdout_audit["formal_complete"]:
                    holdout_action = holdout_decision.action
                holdout_record = {
                    "decision": _decision_summary(holdout_decision),
                    "base": _reward_summary(holdout_base_eval),
                    "current": _reward_summary(holdout_current_eval),
                    "candidate": _reward_summary(holdout_candidate_eval),
                    "safety_audit": holdout_audit,
                }
                del holdout_current

            (
                verified_action,
                verified_checkpoint_update,
                rollback_reason,
            ) = _verified_update_outcome(
                probe_action=final_action,
                safety_complete=safety_audit["formal_complete"],
                holdout_action=holdout_action,
            )
            if not verified_checkpoint_update:
                _blend_decoder(model, old_decoder_state, 0.0)
                optimizer.load_state_dict(old_optimizer_state)
                scheduler.load_state_dict(old_scheduler_state)
                applied_scale = 0.0

            decision_record = {
                "action": verified_action,
                "probe_action": final_action,
                "scale": float(applied_scale),
                "initial": initial_summary,
                "recheck": recheck_summary,
                "base": _reward_summary(base_eval),
                "current": _reward_summary(current_eval),
                "candidate": _reward_summary(candidate_eval),
                "recheck_candidate": recheck_candidate_summary,
                "safety_audit": safety_audit,
                "holdout": holdout_record,
                "rollback_reason": rollback_reason,
                "verified_checkpoint_update": verified_checkpoint_update,
                "block_start_step": pirl_block_start_step,
                "block_end_step": step,
                "updates_in_block": step - pirl_block_start_step + 1,
                "verification_due": True,
            }
            audit_record = {
                "fixed": {
                    "decision": initial_summary,
                    "base": _reward_summary(base_eval),
                    "current": _reward_summary(current_eval),
                    "candidate": _reward_summary(candidate_eval),
                    "safety_audit": safety_audit,
                },
                "holdout": holdout_record,
            }
            del old_policy
            pirl_anchor_policy = None
            pirl_anchor_decoder_state = None
            pirl_anchor_optimizer_state = None
            pirl_anchor_scheduler_state = None
            pirl_block_start_step = None

        if args.pipo:
            assert pipo_attributions is not None
            reward_history.append(float(rewards.mean().item()))
            reward_history = reward_history[-args.pipo_history_window :]
            previous_replay = {
                "batch": batch,
                "trajectory": trajectory,
                "rewards": rewards,
                "attributions": pipo_attributions,
                "group_index": group_index,
                "transition_indices": selected,
            }

        if verified_checkpoint_update or (args.pirl and not pirl_verification_due):
            scheduler.step()
        record = {
            "step": step,
            "algorithm": args.algorithm,
            "property": args.property,
            "pirl": args.pirl,
            "pipo": args.pipo,
            "training_prompt_indices": training_prompt_indices,
            "prompt_splits": {
                "training": "train",
                "probe": "validation",
                "holdout": "validation",
                "final_paired": "validation",
                "test": "reserved_for_post_training_evaluation",
            },
            "sampling_contract": {
                "training_prompts_per_update": args.num_prompts,
                "training_trajectories_per_prompt": args.group_size,
                "policy_microbatch_prompts": args.policy_microbatch_prompts,
                "policy_microbatch_count": _policy_microbatch_count(
                    group_index, args.policy_microbatch_prompts
                ),
                "training_prompt_pool_size": scheduled_prompt_count,
                "training_prompts_unique_across_updates": True,
                "max_prompt_atoms": args.max_prompt_atoms,
                "oversized_training_prompts_excluded": len(
                    oversized_training_prompts
                ),
                "oversized_validation_prompts_excluded": len(
                    oversized_validation_prompts
                ),
                "probe_prompts": args.probe_num_prompts,
                "probe_trajectories_per_prompt": args.probe_group_size,
                "pirl_verification_interval": args.pirl_verification_interval,
            },
            "probe_prompt_indices": probe_prompt_indices,
            "holdout_prompt_indices": holdout_prompt_indices,
            "final_prompt_indices": final_prompt_indices,
            "loss": float(loss.detach().item()),
            "gradient_norm": float(gradient_norm),
            "approx_kl": float(result.metrics["approx_kl"].item()),
            "clip_fraction": float(result.metrics["clip_fraction"].item()),
            "ratio_mean": float(result.metrics["ratio_mean"].item()),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "selected_transitions": selected,
            "advantage": _advantage_summary(advantages, group_index),
            "trajectory": _trajectory_summary(trajectory),
            "policy_epochs": policy_epoch_records,
            "reward": _reward_summary(evaluation),
            "metallicity_proxy": _metallicity_proxy_summary(evaluation, contract),
            "decision": decision_record,
            "audit": audit_record,
            "pipo_feedback": pipo_record,
            "resume_contract": (
                {
                    "start_step": start_step,
                    "prior_metric_records": len(prior_records),
                    "reward_history_restored": True,
                    "previous_replay_restored": False,
                    "first_resume_step_skips_retrospective_update": True,
                }
                if prior_records
                else None
            ),
        }
        checkpoint = None
        checkpoint_due = (step + 1) % args.checkpoint_every == 0 or step == final_step
        if checkpoint_due and (not args.pirl or pirl_verification_due):
            checkpoint = _save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                source_checkpoint=source_checkpoint,
                output_dir=model_dir,
                step=step,
                metadata=record,
            )
        record["checkpoint"] = str(checkpoint) if checkpoint is not None else None
        _append_jsonl(metrics_path, record)
        print(json.dumps(record, ensure_ascii=False))

    if args.final_paired_evaluation:
        assert initial_state is not None
        final_batch = final_batch.to(device)
        final_evaluation = _evaluate_final_policy(
            model=model,
            initial_state=initial_state,
            batch=final_batch,
            adapter=adapter,
            contract=contract,
            noise_seed=args.final_probe_seed,
            diff_ratio=args.short_diff_ratio,
            run_kind=args.run_kind,
            step_lr=args.step_lr,
        )
        final_path = results_dir / "final_paired_evaluation.json"
        final_path.write_text(
            json.dumps(final_evaluation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"final_paired_evaluation": final_evaluation}, ensure_ascii=False))


if __name__ == "__main__":
    main()
