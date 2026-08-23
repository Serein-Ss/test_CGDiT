"""Algorithm-neutral loss assembly for PPO/GRPO crystal-policy updates."""

from dataclasses import dataclass

import torch

from .channel_time_credit import channel_time_objective
from .objectives import PolicyObjectiveResult, grpo_objective, ppo_objective
from .rollout import recompute_trajectory_log_probs
from .trajectory import RLTrajectory


@dataclass
class RLTrainingResult:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    log_prob_by_channel: dict[str, torch.Tensor]


class DiffusionPolicyObjective:
    """Build PPO or GRPO loss from the same symmetry-quotient trajectory."""

    def __init__(self, algorithm: str, clip_epsilon: float = 0.2):
        if algorithm not in {"ppo", "grpo"}:
            raise ValueError("algorithm must be 'ppo' or 'grpo'")
        self.algorithm = algorithm
        self.clip_epsilon = clip_epsilon

    def __call__(
        self,
        model,
        batch,
        trajectory: RLTrajectory,
        step_lr: float,
        rewards: torch.Tensor | None = None,
        group_index: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        pirl_scale: float = 1.0,
        channel_time_credits: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        transition_indices: list[int] | torch.Tensor | None = None,
        reference_model=None,
        reference_weight: float = 0.0,
    ) -> RLTrainingResult:
        current = recompute_trajectory_log_probs(
            model,
            batch,
            trajectory,
            step_lr,
            guidance_scale,
            transition_indices,
        )
        old = trajectory.stacked_old_log_probs(transition_indices)
        reference_penalty = None
        if reference_model is not None and reference_weight:
            with torch.no_grad():
                reference = recompute_trajectory_log_probs(
                    reference_model,
                    batch,
                    trajectory,
                    step_lr,
                    guidance_scale,
                    transition_indices,
                )
            current_joint = torch.stack(tuple(current.values())).sum(dim=0)
            reference_joint = torch.stack(tuple(reference.values())).sum(dim=0)
            reference_penalty = 0.5 * (current_joint - reference_joint).square().mean()
        if channel_time_credits is not None:
            loss = channel_time_objective(
                current,
                old,
                channel_time_credits * pirl_scale,
                clip_epsilon=self.clip_epsilon,
            )
            if reference_penalty is not None:
                loss = loss + reference_weight * reference_penalty
            metrics = {"pirl_scale": loss.new_tensor(pirl_scale)}
            if reference_penalty is not None:
                metrics["reference_penalty"] = reference_penalty.detach()
            return RLTrainingResult(
                loss=loss,
                metrics=metrics,
                log_prob_by_channel=current,
            )

        result: PolicyObjectiveResult
        if self.algorithm == "ppo":
            if advantages is None:
                raise ValueError("PPO requires externally estimated advantages")
            result = ppo_objective(
                current,
                old,
                advantages,
                self.clip_epsilon,
                pirl_scale,
            )
        else:
            if rewards is None or group_index is None:
                raise ValueError("GRPO requires rewards and group_index")
            result = grpo_objective(
                current,
                old,
                rewards,
                group_index,
                self.clip_epsilon,
                pirl_scale,
                advantages,
            )
        loss = result.loss
        if reference_penalty is not None:
            loss = loss + reference_weight * reference_penalty
        metrics = {
                "approx_kl": result.approx_kl.detach(),
                "clip_fraction": result.clip_fraction.detach(),
                "ratio_mean": result.ratio.mean().detach(),
                "pirl_scale": result.loss.new_tensor(pirl_scale),
        }
        if reference_penalty is not None:
            metrics["reference_penalty"] = reference_penalty.detach()
        return RLTrainingResult(
            loss=loss,
            metrics=metrics,
            log_prob_by_channel=current,
        )
