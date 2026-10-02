"""Extract bounded audit evidence, never execute or unpack archive code wholesale."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "tmp/basinguide_bg3_500_dft_stopped_20260914_155007.tar.gz"
OUTPUT = ROOT / "output/basinguide/returned/stopped_20260914_audit"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    inventory, saved, outcars = [], {}, {}
    last = time.monotonic()
    total = 0
    with tarfile.open(ARCHIVE, "r|gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Unsafe member: {name}")
            if not member.isfile():
                continue
            inventory.append(dict(name=str(name), bytes=member.size))
            if time.monotonic() - last > 30:
                print(f"Scanned {len(inventory)} files; saved {total / 2**20:.1f} MiB", flush=True)
                last = time.monotonic()
            sample = re.search(r"/campaign/samples/\d+/", str(name))
            keep = (sample and (name.suffix == ".json" or name.name in
                    {"INCAR", "KPOINTS", "POSCAR", "CONTCAR", "OSZICAR", "POTCAR.spec"}))
            keep = keep or (sample and name.name == "EIGENVAL" and name.parent.name in {"raw_scf", "scf", "bands"})
            keep = keep or (len(name.parts) <= 3 and name.suffix in {".json", ".txt", ".py", ".sh", ".md", ".slurm"})
            if sample and name.name == "OUTCAR":
                stream = archive.extractfile(member)
                tail, first, accurate, ended = b"", b"", False, False
                while chunk := stream.read(1024 * 1024):
                    if not first:
                        first = chunk[:2048]
                    combined = tail + chunk
                    accurate |= b"reached required accuracy" in combined
                    ended |= b"General timing and accounting" in combined
                    tail = combined[-65536:]
                outcars[str(name)] = dict(ionic_accuracy_message=accurate, normal_footer=ended,
                                         header=first.decode(errors="replace"))
                data = tail
                relative = Path(*name.parts[1:]).with_name("OUTCAR.audit_tail.txt")
            elif keep:
                if member.size > 32 * 2**20:
                    raise ValueError(f"Oversized audit input: {name}")
                data = archive.extractfile(member).read()
                relative = Path(*name.parts[1:])
            else:
                continue
            total += len(data)
            if total > 2 * 2**30:
                raise ValueError("Audit extraction exceeded 2 GiB")
            path = OUTPUT / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(data)
            saved[str(relative)] = hashlib.sha256(data).hexdigest()
    manifest = dict(source=str(ARCHIVE), source_bytes=ARCHIVE.stat().st_size,
                    inventory=inventory, extracted_sha256=saved, outcar_checks=outcars,
                    note="Selected audit evidence only; not a full calculation directory")
    (OUTPUT / "audit_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(output=str(OUTPUT), files=len(saved), bytes=total)), flush=True)


if __name__ == "__main__":
    main()
