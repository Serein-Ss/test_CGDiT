"""Small checks protecting cohort identity and paired DFT input semantics."""

import json
import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Incar, Kpoints, Poscar

from scripts.basinguide_diagnostic import atomic_numbers, prepare_vasp, save_structure, read_structure


def test_zero_based_generation_elements():
    import pytest
    assert atomic_numbers([0, 2, 8, 25, 99]) == [1, 3, 9, 26, 100]
    with pytest.raises(ValueError):
        atomic_numbers([100])


def test_structure_roundtrip(tmp_path):
    structure = Structure(Lattice.cubic(4.1), ["Li", "F"], [[0, 0, 0], [.5, .5, .5]])
    save_structure(tmp_path, "generated", structure)
    restored = read_structure(tmp_path / "generated.json")
    assert restored.species == structure.species
    assert np.allclose(restored.frac_coords, structure.frac_coords)


def test_paired_vasp_protocol(tmp_path):
    structure = Structure(Lattice.cubic(4.1), ["F", "Li"], [[.5, .5, .5], [0, 0, 0]])
    for stage in ["raw_scf", "relax", "scf", "bands"]:
        prepare_vasp(structure, tmp_path / stage, stage)
    raw = Incar.from_file(tmp_path / "raw_scf/INCAR")
    relaxed = Incar.from_file(tmp_path / "scf/INCAR")
    for key in ["ENCUT", "ISPIN", "ISMEAR", "SIGMA", "EDIFF", "ISYM"]:
        assert raw[key] == relaxed[key]
    assert Incar.from_file(tmp_path / "relax/INCAR")["ISIF"] == 3
    assert Incar.from_file(tmp_path / "bands/INCAR")["ICHARG"] == 11
    assert relaxed["LCHARG"]
    assert Kpoints.from_file(tmp_path / "bands/KPOINTS").num_kpts > 1
    assert Poscar.from_file(tmp_path / "raw_scf/POSCAR").site_symbols == ["Li", "F"]
    manifest = json.loads((tmp_path / "raw_scf/input_manifest.json").read_text())
    assert [p["symbol"] for p in manifest["pseudopotentials"]] == ["Li_sv", "F"]
