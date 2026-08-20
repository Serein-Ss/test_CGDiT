"""Tensor-only trajectory records for diffusion policy optimization."""

from dataclasses import dataclass, field

import torch


@dataclass
class CrystalState:
    atom_types: torch.Tensor
    frac_coords: torch.Tensor
    crys_fam: torch.Tensor


@dataclass
class RLTransition:
    timestep: int
    state_t: CrystalState
    state_half: CrystalState
    state_next: CrystalState
    old_log_prob_by_channel: dict[str, torch.Tensor]
    actions_by_channel: dict[str, torch.Tensor]
    noise_by_channel: dict[str, torch.Tensor]
    stochastic: bool


@dataclass
class RLTrajectory:
    transitions: list[RLTransition]
    final_state: CrystalState
    num_atoms: torch.Tensor
    metadata: dict[str, torch.Tensor] = field(default_factory=dict)

    def stacked_old_log_probs(
        self, transition_indices: list[int] | torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Stack per-transition, per-structure log probabilities as [T, B]."""
        if not self.transitions:
            return {}
        transitions = self.transitions
        if transition_indices is not None:
            indices = torch.as_tensor(transition_indices).tolist()
            transitions = [self.transitions[index] for index in indices]
        channels = self.transitions[0].old_log_prob_by_channel
        return {
            channel: torch.stack([
                transition.old_log_prob_by_channel[channel]
                for transition in transitions
            ])
            for channel in channels
        }
