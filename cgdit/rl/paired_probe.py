"""Common-random-number rollout pairing for PIRL probe sets."""

from dataclasses import dataclass
from typing import Any


@dataclass
class PairedRollout:
    old_final: dict[str, Any]
    new_final: dict[str, Any]
    old_trajectory: Any
    new_trajectory: Any
    noise_seed: int


def paired_policy_rollout(
    old_policy,
    new_policy,
    batch,
    noise_seed: int,
    diff_ratio: float = 1.0,
    step_lr: float = 1e-5,
    guidance_scale: float = 1.0,
) -> PairedRollout:
    """Generate old/new structures under the identical random-number stream."""
    old_final, _, old_trajectory = old_policy.sample_rl(
        batch,
        diff_ratio=diff_ratio,
        step_lr=step_lr,
        guidance_scale=guidance_scale,
        noise_seed=noise_seed,
        replay_transitions=0,
        retain_trajectory_stack=False,
    )
    new_final, _, new_trajectory = new_policy.sample_rl(
        batch,
        diff_ratio=diff_ratio,
        step_lr=step_lr,
        guidance_scale=guidance_scale,
        noise_seed=noise_seed,
        replay_transitions=0,
        retain_trajectory_stack=False,
    )
    return PairedRollout(
        old_final=old_final,
        new_final=new_final,
        old_trajectory=old_trajectory,
        new_trajectory=new_trajectory,
        noise_seed=noise_seed,
    )
