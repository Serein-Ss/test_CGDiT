"""Migrate legacy evaluation files to the organized Newton output layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from cgdit.common.output_paths import (
    evaluation_output_path,
    generation_output_path,
    provenance_path,
    record_evaluation_metric,
    record_generated_structure,
)


@dataclass(frozen=True)
class MigrationItem:
    source: Path
    destination: Path
    backup: Path
    kind: str
    label: str
    size: int
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classify(name: str) -> tuple[str, str, str] | None:
    if name.startswith("eval_gen_") and name.endswith(".pt"):
        return "generated_structure", name[len("eval_gen_"):-3], ""
    if name.startswith("eval_metrics_gen_") and name.endswith(".json"):
        return (
            "structural_metrics",
            name[len("eval_metrics_gen_"):-5],
            "structural_metrics",
        )
    if name.startswith("eval_properties_gen_") and name.endswith(".csv"):
        return (
            "per_structure_property_predictions",
            name[len("eval_properties_gen_"):-4],
            "property_predictions",
        )
    if name.startswith("eval_property_metrics_gen_") and name.endswith(".json"):
        return (
            "aggregate_property_metrics",
            name[len("eval_property_metrics_gen_"):-5],
            "property_metrics",
        )
    return None


def build_migration_plan(
    root_path: str | Path,
    backup_root: str | Path,
) -> list[MigrationItem]:
    root_path = Path(root_path)
    backup_root = Path(backup_root)
    items: list[MigrationItem] = []

    for hparams in sorted(root_path.rglob("hparams.yaml")):
        model_root = hparams.parent
        for source in sorted(path for path in model_root.iterdir() if path.is_file()):
            classified = _classify(source.name)
            if classified is None:
                continue
            kind, label, category = classified
            if kind == "generated_structure":
                destination = generation_output_path(model_root, label)
            else:
                generation = generation_output_path(model_root, label)
                destination = evaluation_output_path(
                    generation,
                    category,
                    source.name,
                )
            relative = source.relative_to(root_path)
            backup = backup_root / relative
            items.append(MigrationItem(
                source=source,
                destination=destination,
                backup=backup,
                kind=kind,
                label=label,
                size=source.stat().st_size,
                sha256=_sha256(source),
            ))

    destinations: dict[Path, Path] = {}
    for item in items:
        previous = destinations.setdefault(item.destination, item.source)
        if previous != item.source:
            raise RuntimeError(
                f"Multiple sources map to {item.destination}: {previous}, {item.source}"
            )
        for candidate, role in ((item.destination, "destination"), (item.backup, "backup")):
            if candidate.exists() and _sha256(candidate) != item.sha256:
                raise RuntimeError(
                    f"Conflicting {role} file for {item.source}: {candidate}"
                )
    return items


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _manifest_payload(items: list[MigrationItem], status: str) -> dict:
    return {
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(items),
        "total_bytes": sum(item.size for item in items),
        "operations": [
            {
                **asdict(item),
                "source": provenance_path(item.source),
                "destination": provenance_path(item.destination),
                "backup": provenance_path(item.backup),
            }
            for item in items
        ],
    }


def apply_migration(items: list[MigrationItem], manifest_path: str | Path) -> None:
    manifest_path = Path(manifest_path)
    _atomic_json(manifest_path, _manifest_payload(items, "planned"))

    for item in items:
        item.backup.parent.mkdir(parents=True, exist_ok=True)
        if not item.backup.exists():
            shutil.copy2(item.source, item.backup)
        if _sha256(item.backup) != item.sha256:
            raise RuntimeError(f"Backup verification failed: {item.backup}")

        item.destination.parent.mkdir(parents=True, exist_ok=True)
        if item.destination.exists():
            if _sha256(item.destination) != item.sha256:
                raise RuntimeError(f"Destination verification failed: {item.destination}")
            item.source.unlink()
        else:
            os.replace(item.source, item.destination)
        if _sha256(item.destination) != item.sha256:
            raise RuntimeError(f"Migration verification failed: {item.destination}")

    generations = {
        item.destination for item in items if item.kind == "generated_structure"
    }
    for generation in sorted(generations):
        record_generated_structure(generation)
    for item in items:
        if item.kind == "generated_structure":
            continue
        generation = generation_output_path(item.destination.parents[2], item.label)
        if not generation.exists():
            raise RuntimeError(
                f"Evaluation has no matching generated structure: {item.destination}"
            )
        record_evaluation_metric(
            item.kind,
            item.destination,
            generation,
        )

    _atomic_json(manifest_path, _manifest_payload(items, "complete"))


def verify_migration(manifest_path: str | Path) -> tuple[int, int]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Migration is not complete: {manifest_path}")
    total_bytes = 0
    operations = manifest.get("operations", [])
    for operation in operations:
        expected_hash = operation["sha256"]
        expected_size = int(operation["size"])
        for role in ("destination", "backup"):
            path = Path(operation[role])
            if not path.is_file():
                raise RuntimeError(f"Missing {role}: {path}")
            if path.stat().st_size != expected_size or _sha256(path) != expected_hash:
                raise RuntimeError(f"Invalid {role}: {path}")
        source = Path(operation["source"])
        if source.exists():
            raise RuntimeError(f"Legacy source still exists: {source}")
        total_bytes += expected_size
    return len(operations), total_bytes


def cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Move legacy eval files into generated_structures/evaluations."
    )
    parser.add_argument("--root_path", default="output/singlerun")
    parser.add_argument(
        "--backup_root",
        default="output/_legacy_layout_backup_20260822/singlerun",
    )
    parser.add_argument(
        "--manifest",
        default="output/migration_manifests/newton_output_layout_20260822.json",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the migration. Without this flag only print the plan.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify destination and backup hashes recorded in --manifest.",
    )
    args = parser.parse_args(argv)

    if args.apply and args.verify:
        parser.error("--apply and --verify cannot be used together")
    if args.verify:
        count, total_bytes = verify_migration(args.manifest)
        print(f"Verified files: {count}; total bytes: {total_bytes}")
        return

    items = build_migration_plan(args.root_path, args.backup_root)
    print(
        f"Planned files: {len(items)}; "
        f"total bytes: {sum(item.size for item in items)}"
    )
    for item in items:
        print(f"{item.source} -> {item.destination}")
    if not args.apply:
        print("Dry run only. Add --apply to migrate files.")
        return
    apply_migration(items, args.manifest)
    print(f"Migration complete. Manifest: {args.manifest}")


if __name__ == "__main__":
    cli()
