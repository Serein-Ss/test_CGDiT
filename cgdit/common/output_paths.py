"""Canonical paths for generated structures and their evaluations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping


def _generation_parts(label: str) -> tuple[str, str, str]:
    normalized = label.lower()
    stage = "pilot" if normalized.startswith("pilot_") else "formal"
    if "smoke" in normalized or normalized.startswith("test_"):
        stage = "test"

    if "abinitio_empirical" in normalized or "ab_initio_empirical" in normalized:
        method = "abinitio_empirical"
    elif "abinitio" in normalized or "ab_initio" in normalized:
        method = "abinitio"
    else:
        method = "template"

    unconditional = (
        not normalized
        or "uncond" in normalized
        or "unconditional" in normalized
        or stage == "test"
    )
    conditioning = "unconditional" if unconditional else "conditional"
    return stage, method, conditioning


def generation_output_path(model_root: str | Path, label: str) -> Path:
    stage, method, conditioning = _generation_parts(label)
    return (
        Path(model_root)
        / "generated_structures"
        / stage
        / method
        / conditioning
        / (f"eval_gen_{label}.pt" if label else "eval_gen.pt")
    )


def resolve_generation_output_path(model_root: str | Path, label: str) -> Path:
    """Return an existing generation result, preferring the organized layout.

    Historical runs stored ``eval_gen_*.pt`` directly in the model directory.
    This read-only fallback avoids recomputing completed experiments while all
    newly written results continue to use ``generated_structures/``.
    """
    canonical = generation_output_path(model_root, label)
    if canonical.exists():
        return canonical
    legacy_name = f"eval_gen_{label}.pt" if label else "eval_gen.pt"
    legacy = Path(model_root) / legacy_name
    return legacy if legacy.exists() else canonical


def model_root_from_generation(generation_path: str | Path) -> Path:
    path = Path(generation_path)
    for parent in path.parents:
        if parent.name == "generated_structures":
            return parent.parent
    if path.suffix == ".pt" and path.name.startswith("eval_gen"):
        return path.parent
    raise ValueError(f"Generation file is outside generated_structures/: {path}")


def evaluation_output_path(
    generation_path: str | Path,
    category: str,
    filename: str,
) -> Path:
    model_root = model_root_from_generation(generation_path)
    return model_root / "evaluations" / category / filename


def resolve_evaluation_output_path(
    generation_path: str | Path,
    category: str,
    filename: str,
) -> Path:
    """Return an existing evaluation result from either output layout."""
    canonical = evaluation_output_path(generation_path, category, filename)
    if canonical.exists():
        return canonical
    legacy = model_root_from_generation(generation_path) / filename
    return legacy if legacy.exists() else canonical


def provenance_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _manifest_path(model_root: Path) -> Path:
    return model_root / "evaluations/source_manifest.json"


def _load_manifest(model_root: Path) -> dict:
    path = _manifest_path(model_root)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "model_directory": provenance_path(model_root),
        "generated_structures": [],
        "evaluation_metrics": [],
    }


def _write_manifest(model_root: Path, manifest: dict) -> None:
    path = _manifest_path(model_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def record_generated_structure(generation_path: str | Path) -> None:
    generation_path = Path(generation_path)
    model_root = model_root_from_generation(generation_path)
    stage, method, conditioning = _generation_parts(
        generation_path.stem.removeprefix("eval_gen_")
    )
    entry = {
        "file": provenance_path(generation_path),
        "stage": stage,
        "generation_method": method,
        "conditioning": conditioning,
    }
    manifest = _load_manifest(model_root)
    entries = manifest["generated_structures"]
    entries[:] = [item for item in entries if item.get("file") != entry["file"]]
    entries.append(entry)
    entries.sort(key=lambda item: item["file"])
    _write_manifest(model_root, manifest)


def record_evaluation_metric(
    metric_type: str,
    metric_path: str | Path,
    source_generation: str | Path,
    source_predictors: Mapping[str, str | Path] | None = None,
) -> None:
    source_generation = Path(source_generation)
    model_root = model_root_from_generation(source_generation)
    record_generated_structure(source_generation)
    entry = {
        "metric_type": metric_type,
        "metric_file": provenance_path(metric_path),
        "source_generation": provenance_path(source_generation),
    }
    if source_predictors:
        entry["source_predictors"] = {
            name: provenance_path(path) for name, path in source_predictors.items()
        }
    manifest = _load_manifest(model_root)
    entries = manifest["evaluation_metrics"]
    entries[:] = [
        item for item in entries if item.get("metric_file") != entry["metric_file"]
    ]
    entries.append(entry)
    entries.sort(key=lambda item: (item["metric_type"], item["metric_file"]))
    _write_manifest(model_root, manifest)
