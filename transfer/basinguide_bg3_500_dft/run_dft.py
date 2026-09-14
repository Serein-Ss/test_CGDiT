"""Portable adapter around the frozen, original DFT protocol."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts import basinguide_diagnostic as workflow

workflow.POTENTIALS = ROOT / "potentials"


def execute_vasp(directory):
    from pymatgen.io.vasp.outputs import Vasprun
    ranks = os.environ.get("SLURM_NTASKS", os.environ.get("NPROCS", "20"))
    command = shlex.split(os.environ.get("VASP_COMMAND", f"mpirun -np {ranks} vasp_std"))
    if not command or shutil.which(command[0]) is None:
        raise RuntimeError(f"VASP/MPI launcher unavailable: {command}")
    started = time.time()
    with (directory / "vasp.stdout").open("w") as out, (directory / "vasp.stderr").open("w") as err:
        process = subprocess.run(command, cwd=directory, stdout=out, stderr=err,
                                 timeout=int(os.environ.get("VASP_STAGE_TIMEOUT", "72000")))
    run = Vasprun(directory / "vasprun.xml", parse_dos=False, parse_eigen=True,
                  parse_projected_eigen=False, parse_potcar_file=False)
    gap, cbm, vbm, direct = run.eigenvalue_band_properties
    finite = lambda value: float(value) if np.isfinite(value) else None
    result = dict(returncode=process.returncode, electronic_converged=bool(run.converged_electronic),
                  ionic_converged=bool(run.converged_ionic), gap_eV=float(gap),
                  cbm_eV=finite(cbm), vbm_eV=finite(vbm), direct=bool(direct),
                  energy_eV=float(run.final_energy), duration_s=time.time() - started,
                  command=command)
    workflow.write_json(directory / "result.json", result)
    if process.returncode or not run.converged_electronic:
        raise RuntimeError(f"VASP failed or SCF did not converge: {directory}")
    return run, result


def check():
    manifest = json.loads((ROOT / "transfer_manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        path = ROOT / name
        if not path.is_file() or workflow.checksum(path) != expected:
            raise RuntimeError(f"Missing or modified packaged file: {name}")
    cohort = json.loads((ROOT / "campaign/cohort.json").read_text())
    assert len(cohort) == 500
    for row in cohort:
        structure = workflow.read_structure(ROOT / "campaign/samples" / f'{row["sample_id"]:04d}' / "generated.json")
        assert 1 <= len(structure) <= 20
    print(f"Verified 500 structures and {len(manifest['sha256'])} file checksums")
    missing = [symbol for symbol in manifest["potentials"]
               if not (ROOT / "potentials" / symbol / "POTCAR").is_file()]
    if missing:
        raise RuntimeError(f"Supply licensed PBE POTCAR files in potentials/<symbol>/POTCAR: {missing}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    campaign = ROOT / "campaign"
    if args.check:
        check()
    elif args.collect:
        workflow.collect(campaign)
    else:
        if args.index is None or not 0 <= args.index < 500:
            parser.error("--index must be between 0 and 499")
        directory = workflow.sample_dir(campaign, args.index)
        if args.prepare_only:
            initial = workflow.read_structure(directory / "generated.json")
            for stage in ["raw_scf", "relax"]:
                workflow.prepare_vasp(initial, ROOT / "prepared_inputs" / f"{args.index:04d}" / stage, stage)
            return
        if (directory / "dft").exists():
            result_path = directory / "dft/result.json"
            if result_path.exists() and json.loads(result_path.read_text()).get("status") == "complete":
                print("Already complete; no computation repeated")
                return
            raise RuntimeError(f"Existing incomplete run: {directory / 'dft'}. Move it to a backup before restarting.")
        workflow.execute_vasp = execute_vasp
        # The original seed-42 initial predictions are already packaged. Final
        # DFT-structure ML predictions can be added on the original GPU server.
        workflow.predict = lambda structures, device: (
            [None] * len(structures), ["deferred_to_original_seed42_predictor"] * len(structures))
        workflow.dft(campaign, args.index)


if __name__ == "__main__":
    main()
