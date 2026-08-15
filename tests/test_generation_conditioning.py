from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cgdit.generation.conditioning import (
    apply_condition_values,
    condition_label,
    condition_values_from_args,
    parse_condition_assignments,
    validate_condition_values,
)
from cgdit.generation.symmetry import get_data_from_syminfo, get_pymatgen


def test_parse_multiple_condition_assignments():
    assert parse_condition_assignments([
        "formation_energy_per_atom=-1.5",
        "band_gap=2.0",
        "e_above_hull=0",
    ]) == {
        "formation_energy_per_atom": -1.5,
        "band_gap": 2.0,
        "e_above_hull": 0.0,
    }


def test_legacy_single_condition_is_preserved():
    args = SimpleNamespace(
        condition=["band_gap=2.0"],
        property_name="formation_energy_per_atom",
        target_value=-1.5,
    )
    assert condition_values_from_args(args) == {
        "band_gap": 2.0,
        "formation_energy_per_atom": -1.5,
    }


def test_unknown_condition_is_rejected():
    with pytest.raises(ValueError, match="was not trained"):
        validate_condition_values({"band_gap": 2.0}, {})


def test_apply_joint_scalar_and_categorical_conditions():
    batch = SimpleNamespace(num_graphs=2, batch=torch.tensor([0, 1]))
    configs = {
        "band_gap": {"type": "scalar"},
        "spacegroup": {"type": "categorical"},
    }

    apply_condition_values(batch, {"band_gap": 2.0, "spacegroup": 225}, configs)

    assert batch.band_gap.shape == (2, 1)
    assert batch.band_gap.dtype == torch.float
    assert torch.all(batch.band_gap == 2.0)
    assert batch.spacegroup.shape == (2,)
    assert batch.spacegroup.dtype == torch.long
    assert torch.all(batch.spacegroup == 225)


def test_condition_label_is_filename_safe():
    assert condition_label({"formation_energy_per_atom": -1.5, "band_gap": 2.0}) == (
        "joint_formation_energy_per_atom-m1p5_band_gap-2"
    )


def test_symmetry_query_preserves_fixed_elements_and_restores_atomic_numbers():
    data = get_data_from_syminfo(1, ["1a"], ["Si"])
    assert torch.equal(data.atom_types, torch.tensor([14]))

    structure = get_pymatgen({
        "frac_coords": np.array([[0.0, 0.0, 0.0]]),
        "atom_types": np.array([13]),
        "lengths": np.array([3.0, 3.0, 3.0]),
        "angles": np.array([90.0, 90.0, 90.0]),
    })
    assert structure[0].specie.Z == 14
