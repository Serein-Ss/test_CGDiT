"""GPU integration test for old CGDiT weights with the new RL pipeline."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch_geometric.data import Batch

from cgdit.common.evaluation_utils import load_model
from cgdit.rl.channel_time_credit import (
    normalized_channel_time_credit,
    temporal_difference_scores,
)
from cgdit.rl.multifidelity import (
    MultiFidelityVerifier,
    PredictorEnsembleEvaluator,
)
from cgdit.rl.paired_probe import paired_policy_rollout
from cgdit.rl.policy_improvement import decide_policy_improvement
from cgdit.rl.trainer import DiffusionPolicyObjective


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-root", default="output/rl_legacy_smoke", type=Path)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--short-diff-ratio", default=0.02, type=float)
    parser.add_argument("--step-lr", default=1e-5, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-full-sample", action="store_true")
    return parser.parse_args()


def require_inputs(args):
    model_path = args.model_path.resolve()
    required = [
        model_path / "hparams.yaml",
        Path("data/mp_20/test.csv"),
        Path("data/mp_20/test_sym.pt"),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if not list(model_path.glob("*.ckpt")):
        missing.append(f"{model_path}/*.ckpt")
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))
    return model_path


def orbit_consistency(atom_types, anchor_index):
    return bool(torch.equal(atom_types, atom_types[anchor_index]))


def synthetic_rewards(trajectory):
    reward = -trajectory.final_state.crys_fam.square().mean(dim=-1)
    return reward + torch.arange(
        reward.numel(), device=reward.device, dtype=reward.dtype
    ) * 1e-3


def optimizer_step(model, result, learning_rate=1e-6):
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=learning_rate)
    optimizer.zero_grad(set_to_none=True)
    result.loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
    if not torch.isfinite(grad_norm):
        raise RuntimeError("Non-finite decoder gradient")
    changed_parameter = None
    before = None
    for parameter in model.decoder.parameters():
        if parameter.grad is not None and torch.count_nonzero(parameter.grad):
            changed_parameter = parameter
            before = parameter.detach().clone()
            break
    if changed_parameter is None:
        raise RuntimeError("No decoder parameter received a non-zero gradient")
    optimizer.step()
    parameter_delta = (changed_parameter.detach() - before).norm()
    if not torch.isfinite(parameter_delta) or parameter_delta <= 0:
        raise RuntimeError("Optimizer step did not change a decoder parameter")
    return float(grad_norm), float(parameter_delta)


def scalar_metrics(result):
    metrics = {"loss": float(result.loss.detach())}
    metrics.update({key: float(value) for key, value in result.metrics.items()})
    return metrics


def run_synthetic_multifidelity(rewards):
    values = rewards.detach().cpu()
    predictors = [
        lambda candidates: torch.tensor(candidates, dtype=torch.float),
        lambda candidates: torch.tensor(candidates, dtype=torch.float) + 0.01,
    ]
    proxy = PredictorEnsembleEvaluator(predictors)
    verifier = MultiFidelityVerifier(proxy)
    trace = verifier.run(
        values.tolist(),
        mlff_budget=min(2, values.numel()),
        dft_budget=min(1, values.numel()),
    )
    return {
        "proxy_count": int(trace.proxy.score.numel()),
        "mlff_selected": trace.mlff_indices.tolist(),
        "dft_selected": trace.dft_indices.tolist(),
        "status": "interface_only_no_real_predictor_mlff_dft",
    }


def main():
    args = parse_args()
    model_path = require_inputs(args)
    if args.preflight_only:
        print(f"Preflight OK: {model_path}")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("This integration test requires a scheduled CUDA GPU")

    job_id = os.environ.get("SLURM_JOB_ID", "manual")
    run_dir = (args.output_root / job_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "job_id": job_id,
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "model_path": str(model_path),
        "seed": args.seed,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "checks": {},
        "blocked": [
            "real property RL awaits uploaded reward predictor and independent evaluator",
            "real MLFF/DFT awaits scheduler adapters and physics settings",
        ],
    }
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    model, test_loader, _ = load_model(model_path, load_data=True, testing=True)
    model = model.cuda().eval()
    dataset = test_loader.dataset
    if args.batch_size < 4 or args.batch_size % 2:
        raise ValueError("batch-size must be an even integer of at least 4")
    batch = Batch.from_data_list([
        dataset[index] for index in range(args.batch_size)
    ]).cuda()
    report["checks"]["checkpoint_load"] = {
        "structures": args.batch_size,
        "nodes": int(batch.num_nodes),
    }

    short_started = time.perf_counter()
    final, _, trajectory = model.sample_rl(
        batch,
        diff_ratio=args.short_diff_ratio,
        step_lr=args.step_lr,
        noise_seed=args.seed,
    )
    if not orbit_consistency(final["atom_types"], batch.anchor_index):
        raise RuntimeError("Orbit element consistency failed")
    stochastic_indices = [
        index for index, transition in enumerate(trajectory.transitions)
        if transition.stochastic
    ]
    selected_indices = [
        stochastic_indices[0], stochastic_indices[len(stochastic_indices) // 2]
    ]
    report["checks"]["short_rollout"] = {
        "seconds": time.perf_counter() - short_started,
        "transitions": len(trajectory.transitions),
        "selected_transition_indices": selected_indices,
        "orbit_consistent": True,
    }

    if not args.skip_full_sample:
        full_batch = Batch.from_data_list([dataset[0]]).cuda()
        full_started = time.perf_counter()
        full_final, _ = model.sample(
            full_batch,
            diff_ratio=1.0,
            step_lr=args.step_lr,
            noise_seed=args.seed + 2,
        )
        full_seconds = time.perf_counter() - full_started
        full_consistent = orbit_consistency(
            full_final["atom_types"], full_batch.anchor_index
        )
        if not full_consistent:
            raise RuntimeError("Full rollout orbit element consistency failed")
        torch.save(
            {key: value.detach().cpu() for key, value in full_final.items()},
            run_dir / "full_sample.pt",
        )
        report["checks"]["full_1000_step_sample"] = {
            "seconds": full_seconds,
            "nodes": int(full_batch.num_nodes),
            "orbit_consistent": full_consistent,
        }

    replay_started = time.perf_counter()
    rewards = synthetic_rewards(trajectory)
    advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-6)
    ppo = DiffusionPolicyObjective("ppo")
    ppo_result = ppo(
        model,
        batch,
        trajectory,
        step_lr=args.step_lr,
        advantages=advantages,
        transition_indices=selected_indices,
    )
    ppo_metrics = scalar_metrics(ppo_result)
    ppo_metrics["grad_norm"], ppo_metrics["parameter_delta"] = optimizer_step(
        model, ppo_result
    )
    ppo_metrics["seconds"] = time.perf_counter() - replay_started
    report["checks"]["ppo_step"] = ppo_metrics

    old_model, _, _ = load_model(model_path, load_data=False, testing=True)
    old_model = old_model.cuda().eval()
    paired_started = time.perf_counter()
    paired = paired_policy_rollout(
        old_model,
        model,
        batch,
        noise_seed=args.seed + 1,
        diff_ratio=args.short_diff_ratio,
        step_lr=args.step_lr,
    )
    old_reward = -paired.old_trajectory.final_state.crys_fam.square().mean(dim=-1)
    new_reward = -paired.new_trajectory.final_state.crys_fam.square().mean(dim=-1)
    old_metrics = torch.stack([old_reward, torch.ones_like(old_reward)], dim=-1)
    new_metrics = torch.stack([new_reward, torch.ones_like(new_reward)], dim=-1)
    decision = decide_policy_improvement(
        old_metrics,
        new_metrics,
        safety_tolerances={1: 0.0},
        n_bootstrap=200,
        seed=args.seed,
    )
    report["checks"]["pirl_pair"] = {
        "seconds": time.perf_counter() - paired_started,
        "action": decision.action,
        "scale": decision.scale,
        "mean_delta": decision.mean_delta.detach().cpu().tolist(),
        "lcb": decision.lower_confidence_bound.detach().cpu().tolist(),
        "metric": "synthetic_interface_check_only",
    }

    grpo_started = time.perf_counter()
    grpo = DiffusionPolicyObjective("grpo")
    group_index = torch.arange(args.batch_size, device=batch.batch.device) // 2
    grpo_result = grpo(
        old_model,
        batch,
        trajectory,
        step_lr=args.step_lr,
        rewards=rewards,
        group_index=group_index,
        transition_indices=selected_indices,
    )
    grpo_metrics = scalar_metrics(grpo_result)
    grpo_metrics["grad_norm"], grpo_metrics["parameter_delta"] = optimizer_step(
        old_model, grpo_result
    )
    grpo_metrics["seconds"] = time.perf_counter() - grpo_started
    report["checks"]["grpo_step"] = grpo_metrics

    intermediate_rewards = torch.stack(
        [torch.zeros_like(rewards), 0.5 * rewards, rewards], dim=-1
    )
    contribution = temporal_difference_scores(intermediate_rewards).unsqueeze(-1)
    contribution = contribution.repeat(1, 1, 3)
    credits = normalized_channel_time_credit(advantages, contribution)
    h2_result = DiffusionPolicyObjective("grpo")(
        old_model,
        batch,
        trajectory,
        step_lr=args.step_lr,
        channel_time_credits=credits.permute(1, 0, 2),
        transition_indices=selected_indices,
    )
    if not torch.isfinite(h2_result.loss):
        raise RuntimeError("H2 channel-time objective is not finite")
    report["checks"]["h2"] = {
        "loss": float(h2_result.loss.detach()),
        "status": "synthetic_credit_interface_check_only",
    }
    report["checks"]["multifidelity"] = run_synthetic_multifidelity(rewards)

    report["elapsed_seconds"] = time.perf_counter() - started
    report["peak_cuda_memory_gib"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    report["status"] = "completed"
    with (run_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
