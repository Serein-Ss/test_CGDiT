"""Create a checksummed, portable DFT transfer snapshot."""
import datetime
import json
from pathlib import Path
import shutil
import tarfile
from pymatgen.io.vasp.sets import MPRelaxSet
from scripts.basinguide_diagnostic import ROOT, POTENTIALS, checksum, read_structure, write_json


def main():
    source = ROOT / "output/basinguide/bg3_500_seed42_20260914"
    bundle = ROOT / "transfer/basinguide_bg3_500_dft"
    bundle.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT / "scripts/basinguide_transfer", bundle, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    (bundle / "scripts").mkdir()
    (bundle / "scripts/__init__.py").write_text("")
    shutil.copy2(ROOT / "scripts/basinguide_diagnostic.py", bundle / "scripts/basinguide_diagnostic.py")
    campaign = bundle / "campaign"
    campaign.mkdir()
    for name in ["cohort.json", "generation.pt", "generation_complete.json"]:
        shutil.copy2(source / name, campaign / name)
    shutil.copy2(source / "submission.json", bundle / "original_submission.json")
    shutil.copy2(source / "environment_freeze.txt", bundle / "original_environment_freeze.txt")
    shutil.copy2(ROOT / "docs/BasinGuide_bg3_500_execution.md", bundle / "original_execution.md")
    required, snapshots, affected = set(), [], {}
    cohort = json.loads((source / "cohort.json").read_text())
    assert len(cohort) == 500
    for item in cohort:
        index = item["sample_id"]
        old = source / "samples" / f"{index:04d}"
        new = campaign / "samples" / f"{index:04d}"
        new.mkdir(parents=True)
        for suffix in ["json", "cif", "vasp"]:
            shutil.copy2(old / f"generated.{suffix}", new / f"generated.{suffix}")
        structure = read_structure(old / "generated.json").get_sorted_structure()
        symbols = MPRelaxSet(structure).potcar_symbols
        required.update(symbols)
        for symbol in symbols:
            affected.setdefault(symbol, []).append(index)
        for kind in ["dft", "mlip"]:
            if (old / kind).is_dir():
                snapshot = bundle / "reference_results" / f"{index:04d}" / kind
                shutil.copytree(old / kind, snapshot)
                snapshots.append(dict(sample_id=index, kind=kind,
                                      stage_record_present=(old / kind / "result.json").exists()))
    for symbol in sorted(required):
        destination = bundle / "potentials" / symbol
        destination.mkdir(parents=True)
        if (POTENTIALS / symbol / "POTCAR").is_file():
            shutil.copy2(POTENTIALS / symbol / "POTCAR", destination / "POTCAR")
    missing = {symbol: affected[symbol] for symbol in sorted(required)
               if not (bundle / "potentials" / symbol / "POTCAR").is_file()}
    files = {str(p.relative_to(bundle)): checksum(p) for p in sorted(bundle.rglob("*"))
             if p.is_file() and p.name != "site_env.sh"}
    write_json(bundle / "transfer_manifest.json", dict(
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), n_samples=500,
        potentials=sorted(required), missing_potentials=missing,
        mutable_files=["site_env.sh"], sha256=files, result_snapshots=snapshots,
        stopped_original_jobs=["669415", "669416", "669417"],
        return_directory=str(ROOT / "output/basinguide/returned/basinguide_bg3_500_dft"),
        note="Reference snapshots include cancelled partial calculations; not necessarily converged"))
    archive = bundle.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(bundle, arcname=bundle.name)
    digest = checksum(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(f"{digest}  {archive.name}\n")
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size,
                         sha256=digest, n_structures=500, n_potentials=len(required)), indent=2))


if __name__ == "__main__":
    main()
