"""One campaign: 500 BG-conditioned samples, seed-42 prediction, MLIP and DFT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "output/singlerun/2026-06-30/11-28-44-mp20_bg"
PREDICTOR = ROOT / "output/singlerun/2026-08-16/15-18-32-mp20_predictor_bg"
POTENTIALS = Path("/share/home/xlzou/psp/POT_GGA_PAW_PBE")
N_SAMPLES = 500
TARGET = 3.0
TOLERANCE = 0.5


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_dir(campaign, index):
    return campaign / "samples" / f"{index:04d}"


def read_structure(path):
    from pymatgen.core import Structure
    return Structure.from_dict(json.loads(Path(path).read_text()))


def save_structure(directory, name, structure):
    from pymatgen.io.cif import CifWriter
    from pymatgen.io.vasp import Poscar
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / f"{name}.json", structure.as_dict())
    CifWriter(structure, symprec=None, significant_figures=12).write_file(directory / f"{name}.cif")
    Poscar(structure).write_file(directory / f"{name}.vasp", significant_figures=16)
    restored = Poscar.from_file(directory / f"{name}.vasp", check_for_potcar=False).structure
    assert restored.species == structure.species
    assert np.allclose(restored.lattice.matrix, structure.lattice.matrix)
    assert np.allclose(restored.frac_coords, structure.frac_coords)


def crystal_array(structure):
    return dict(frac_coords=np.asarray(structure.frac_coords),
                atom_types=np.asarray(structure.atomic_numbers),
                lengths=np.asarray(structure.lattice.abc),
                angles=np.asarray(structure.lattice.angles))


def atomic_numbers(classes):
    classes = np.asarray(classes, dtype=int)
    if not np.all((classes >= 0) & (classes < 100)):
        raise ValueError("Unresolved mask or invalid generated element class")
    return (classes + 1).tolist()


def predict(structures, device):
    from cgdit.evaluation.generated_properties import build_graphs, predict_graphs, resolve_checkpoint
    graphs, indices, errors = build_graphs([crystal_array(s) for s in structures], num_workers=1)
    values = predict_graphs(resolve_checkpoint(PREDICTOR), "band_gap", graphs, indices,
                            len(structures), batch_size=16, device=device)
    return [float(v) if np.isfinite(v) else None for v in values], errors


def generate(campaign):
    import torch
    from pymatgen.core import Lattice, Structure
    from torch_geometric.loader import DataLoader
    from cgdit.common.evaluation_utils import load_model
    from cgdit.generation.general import AbInitioDataset, diffusion
    from cgdit.generation.conditioning import seed_generation

    if (campaign / "generation_complete.json").exists():
        raise RuntimeError("Generation already completed; refuse to replace this cohort")
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    seed_generation(42)
    model, _, cfg = load_model(GENERATOR, load_data=False)
    assert "band_gap" in cfg.model.conditions
    model.eval().to("cuda")
    cached = torch.load(ROOT / "data/mp_20/train_sym.pt", map_location="cpu", weights_only=False)
    counts = [SimpleNamespace(num_atoms=int(row["graph_arrays"][-1])) for row in cached]
    del cached
    dataset = AbInitioDataset(N_SAMPLES, train_set=counts, use_empirical_prior=True,
                              max_atoms=20, seed=42)
    loader = DataLoader(dataset, batch_size=20, shuffle=False, num_workers=0)
    started = time.time()
    x, a, lattice, lengths, angles, n = diffusion(
        loader, model, 1e-5, {"band_gap": TARGET}, cfg.model.conditions, guidance_scale=1.0)
    assert len(n) == N_SAMPLES
    torch.save(dict(frac_coords=x, atom_types=a, lattices=lattice, lengths=lengths,
                    angles=angles, num_atoms=n, conditions={"band_gap": TARGET}, seed=42),
               campaign / "generation.pt")
    del model
    torch.cuda.empty_cache()
    offset = 0
    manifest, structures, valid_indices = [], [], []
    for index, count in enumerate(n.tolist()):
        record = dict(sample_id=index, num_atoms=count, status="generated")
        try:
            species = atomic_numbers(a[offset:offset + count].numpy())
            structure = Structure(Lattice(lattice[index].numpy()), species,
                                  x[offset:offset + count].numpy(), to_unit_cell=True)
            assert np.isfinite(structure.cart_coords).all() and structure.volume > 0
            assert 1 <= len(structure) <= 20
            save_structure(sample_dir(campaign, index), "generated", structure)
            record.update(formula=structure.composition.reduced_formula, volume_A3=structure.volume)
            structures.append(structure)
            valid_indices.append(index)
        except Exception as exc:
            record.update(status="export_failed", error=f"{type(exc).__name__}: {exc}")
        manifest.append(record)
        offset += count
    # Publish the full cohort before prediction; no score-based replacement.
    write_json(campaign / "cohort.json", manifest)
    values, errors = predict(structures, "cuda")
    for index, value, error in zip(valid_indices, values, errors):
        manifest[index].update(predicted_bg_raw_eV=value, prediction_error=error)
    write_json(campaign / "cohort.json", manifest)
    write_json(campaign / "generation_complete.json", dict(
        n_total=N_SAMPLES, n_exported=len(structures), duration_s=time.time() - started,
        generator_checkpoint_sha256=checksum(next(GENERATOR.glob("*.ckpt"))),
        predictor_checkpoint_sha256=checksum(next(PREDICTOR.glob("*.ckpt")))))


def mlip(campaign, checkpoint):
    import torch
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    from pymatgen.io.ase import AseAtomsAdaptor
    from sevenn.calculator import SevenNetCalculator

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    calculator = SevenNetCalculator(str(checkpoint), device="cuda", modal="mpa")
    cohort = json.loads((campaign / "cohort.json").read_text())
    for row in cohort:
        index = row["sample_id"]
        directory = sample_dir(campaign, index) / "mlip"
        directory.mkdir(parents=True, exist_ok=True)
        result = dict(sample_id=index, model="SevenNet-Omni-i12", modal="mpa",
                      checkpoint_sha256=checksum(checkpoint), status="failed")
        try:
            structure = read_structure(sample_dir(campaign, index) / "generated.json")
            atoms = AseAtomsAdaptor.get_atoms(structure)
            atoms.calc = calculator
            result["initial_energy_eV"] = float(atoms.get_potential_energy())
            optimizer = FIRE(FrechetCellFilter(atoms), maxstep=0.1,
                             trajectory=str(directory / "relax.traj"),
                             logfile=str(directory / "relax.log"))
            converged = optimizer.run(fmax=0.03, steps=1000)
            final = AseAtomsAdaptor.get_structure(atoms)
            save_structure(directory, "final", final)
            result.update(status="converged" if converged else "not_converged",
                          nsteps=optimizer.nsteps, final_energy_eV=float(atoms.get_potential_energy()),
                          max_force_eV_A=float(np.linalg.norm(atoms.get_forces(), axis=1).max()))
            values, errors = predict([final], "cuda")
            result.update(predicted_bg_final_eV=values[0], prediction_error=errors[0])
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        write_json(directory / "result.json", result)
        print(index, result["status"], flush=True)
    write_json(campaign / "mlip_complete.json", dict(n_attempted=len(cohort)))


def prepare_vasp(structure, directory, stage):
    from pymatgen.io.vasp import Incar, Kpoints, Poscar, PotcarSingle
    from pymatgen.io.vasp.sets import MPRelaxSet

    directory.mkdir(parents=True, exist_ok=False)
    structure = structure.get_sorted_structure()
    settings = dict(ENCUT=520, PREC="Accurate", EDIFF=1e-6, EDIFFG=-0.02,
                    ISPIN=2, ISYM=0, ISMEAR=0, SIGMA=0.05, LREAL=False,
                    LASPH=True, ADDGRID=True, ALGO="Normal", NELM=200,
                    NCORE=4, LORBIT=11, LWAVE=False, LCHARG=False,
                    IBRION=2, ISIF=3, NSW=200, POTIM=0.3)
    if stage != "relax":
        settings.update(NSW=0, IBRION=-1, LCHARG=stage == "scf")
    inputs = MPRelaxSet(structure, user_incar_settings=settings)
    incar = inputs.incar
    symbols = inputs.potcar_symbols
    pieces, metadata = [], []
    for symbol in symbols:
        path = POTENTIALS / symbol / "POTCAR"
        content = path.read_text()
        pot = PotcarSingle(content)
        pieces.append(content)
        metadata.append(dict(symbol=symbol, sha256=checksum(path), enmax=float(pot.enmax)))
    incar["ENCUT"] = max(520, int(np.ceil(1.3 * max(p["enmax"] for p in metadata))))
    if stage == "bands":
        import seekpath
        path = seekpath.get_explicit_k_path_orig_cell(
            (structure.lattice.matrix, structure.frac_coords, structure.atomic_numbers),
            reference_distance=0.03, symprec=0.01)
        points = path["explicit_kpoints_rel"]
        kpoints = Kpoints(comment="Seekpath original-cell path; frozen SCF density",
                          num_kpts=len(points), style=Kpoints.supported_modes.Reciprocal,
                          kpts=points.tolist(), kpts_weights=[1.0] * len(points),
                          labels=path["explicit_kpoints_labels"])
        incar.update(ICHARG=11, ISMEAR=0, LCHARG=False)
        write_json(directory / "kpath.json", dict(
            segments=path["explicit_segments"], labels=path["explicit_kpoints_labels"]))
    else:
        kpoints = Kpoints.automatic_density(structure, 1000 if stage == "relax" else 3000,
                                            force_gamma=True)
    Incar(incar).write_file(directory / "INCAR")
    kpoints.write_file(directory / "KPOINTS")
    Poscar(structure).write_file(directory / "POSCAR")
    (directory / "POTCAR").write_text("".join(pieces))
    (directory / "POTCAR.spec").write_text("\n".join(symbols) + "\n")
    write_json(directory / "input_manifest.json", dict(stage=stage, pseudopotentials=metadata,
                                                       structure=structure.as_dict(),
                                                       incar=dict(incar)))
    return structure


def execute_vasp(directory):
    from pymatgen.io.vasp.outputs import Vasprun
    command = ["mpirun", "-np", os.environ["SLURM_NTASKS"], shutil.which("vasp_std")]
    started = time.time()
    with (directory / "vasp.stdout").open("w") as out, (directory / "vasp.stderr").open("w") as err:
        process = subprocess.run(command, cwd=directory, stdout=out, stderr=err, timeout=20 * 3600)
    run = Vasprun(directory / "vasprun.xml", parse_dos=False, parse_eigen=True,
                  parse_projected_eigen=False, parse_potcar_file=False)
    gap, cbm, vbm, direct = run.eigenvalue_band_properties
    finite_value = lambda value: float(value) if np.isfinite(value) else None
    result = dict(returncode=process.returncode, electronic_converged=bool(run.converged_electronic),
                  ionic_converged=bool(run.converged_ionic), gap_eV=float(gap),
                  cbm_eV=finite_value(cbm), vbm_eV=finite_value(vbm), direct=bool(direct),
                  energy_eV=float(run.final_energy), duration_s=time.time() - started)
    write_json(directory / "result.json", result)
    if process.returncode or not run.converged_electronic:
        raise RuntimeError(f"VASP failed or SCF not converged: {directory.name}")
    return run, result


def dft(campaign, index):
    directory = sample_dir(campaign, index) / "dft"
    directory.mkdir(parents=True, exist_ok=True)
    result = dict(sample_id=index, status="failed", start_structure="generated", stages={})
    try:
        initial = read_structure(sample_dir(campaign, index) / "generated.json")
        # Initial single point and direct DFT relaxation are independent branches.
        try:
            prepare_vasp(initial, directory / "raw_scf", "raw_scf")
            _, raw = execute_vasp(directory / "raw_scf")
            result["stages"]["raw_scf"] = raw
        except Exception as exc:
            result["stages"]["raw_scf"] = dict(error=str(exc))
        current = initial
        relaxed = False
        for segment in range(2):
            relax_dir = directory / f"relax_{segment}"
            prepare_vasp(current, relax_dir, "relax")
            run, segment_result = execute_vasp(relax_dir)
            result["stages"][f"relax_{segment}"] = segment_result
            current = run.final_structure
            if run.converged_ionic:
                relaxed = True
                break
        save_structure(directory, "final", current)
        if not relaxed:
            raise RuntimeError("Ionic relaxation did not converge within 400 steps")
        prepare_vasp(current, directory / "scf", "scf")
        _, scf_result = execute_vasp(directory / "scf")
        result["stages"]["scf"] = scf_result
        result["relaxed_gap_eV"] = scf_result["gap_eV"]
        result["target_hit"] = abs(scf_result["gap_eV"] - TARGET) <= TOLERANCE
        try:
            values, errors = predict([current], "cpu")
            result.update(predicted_bg_final_eV=values[0], prediction_error=errors[0])
        except Exception as exc:
            result["prediction_error"] = str(exc)
        write_json(directory / "result.json", result)
        try:
            prepare_vasp(current, directory / "bands", "bands")
            shutil.copyfile(directory / "scf/CHGCAR", directory / "bands/CHGCAR")
            _, band_result = execute_vasp(directory / "bands")
            result["stages"]["bands"] = band_result
            result["status"] = "complete"
        except Exception as exc:
            result.update(status="scf_complete_bands_failed", bands_error=str(exc))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    write_json(directory / "result.json", result)
    print(json.dumps(result), flush=True)
    if result["status"] != "complete":
        raise SystemExit(1)


def collect(campaign):
    cohort = json.loads((campaign / "cohort.json").read_text())
    rows = []
    for item in cohort:
        row = dict(item)
        for method in ["mlip", "dft"]:
            path = sample_dir(campaign, item["sample_id"]) / method / "result.json"
            result = json.loads(path.read_text()) if path.exists() else {"status": "missing"}
            row[f"{method}_status"] = result["status"]
            row[f"{method}_predicted_bg_eV"] = result.get("predicted_bg_final_eV")
            if method == "dft":
                row["dft_raw_gap_eV"] = result.get("stages", {}).get("raw_scf", {}).get("gap_eV")
                row["dft_relaxed_gap_eV"] = result.get("relaxed_gap_eV")
            row[f"{method}_error"] = result.get("error")
        rows.append(row)
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with (campaign / "paired_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    paired, categories = 0, {"hit_hit": 0, "hit_miss": 0, "miss_hit": 0, "miss_miss": 0}
    n_success = 0
    for row in rows:
        raw, final = row.get("predicted_bg_raw_eV"), row.get("dft_relaxed_gap_eV")
        if final is not None and abs(final - TARGET) <= TOLERANCE:
            n_success += 1
        if raw is not None and final is not None:
            paired += 1
            a = "hit" if abs(raw - TARGET) <= TOLERANCE else "miss"
            b = "hit" if abs(final - TARGET) <= TOLERANCE else "miss"
            categories[f"{a}_{b}"] += 1
    write_json(campaign / "summary.json", dict(n_total=N_SAMPLES, n_dft_target_hits=n_success,
               unconditional_dft_hit_rate=n_success / N_SAMPLES,
               n_predictor_dft_pairs=paired, predictor_to_dft_categories=categories,
               note="Failures remain in denominator; k-mesh PBE(+U) gaps, not experimental gaps"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["generate", "mlip", "dft", "collect"])
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    args.campaign.mkdir(parents=True, exist_ok=True)
    if args.action == "generate":
        generate(args.campaign)
    elif args.action == "mlip":
        mlip(args.campaign, args.checkpoint)
    elif args.action == "dft":
        dft(args.campaign, args.index)
    else:
        collect(args.campaign)


if __name__ == "__main__":
    main()
