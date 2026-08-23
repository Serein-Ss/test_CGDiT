from types import SimpleNamespace

import torch

from scripts.cli.training.check_rl_reproducibility import (
    _trajectory_fingerprint,
)


def _trajectory(frac_coords):
    return SimpleNamespace(
        num_atoms=torch.tensor([2]),
        final_state=SimpleNamespace(
            atom_types=torch.tensor([6, 8]),
            frac_coords=torch.tensor(frac_coords),
            crys_fam=torch.eye(2),
        ),
    )


def test_trajectory_fingerprint_is_stable_for_identical_terminal_state():
    first = _trajectory([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]])
    second = _trajectory([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]])

    assert _trajectory_fingerprint(first) == _trajectory_fingerprint(second)


def test_trajectory_fingerprint_detects_terminal_state_change():
    first = _trajectory([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]])
    changed = _trajectory([[0.0, 0.1, 0.2], [0.3, 0.4, 0.6]])

    assert _trajectory_fingerprint(first) != _trajectory_fingerprint(
        changed
    )
