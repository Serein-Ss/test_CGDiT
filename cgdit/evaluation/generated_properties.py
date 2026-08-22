"""Property evaluation for generated ``eval_gen_*.pt`` files."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from cgdit.common.output_paths import (
    evaluation_output_path,
    model_root_from_generation,
    provenance_path,
    record_evaluation_metric,
)
from scipy.stats import wasserstein_distance
from tqdm import tqdm


PROPERTY_NAMES = (
    "formation_energy_per_atom",
    "band_gap",
    "e_above_hull",
)

DEFAULT_TOLERANCES = {
    "formation_energy_per_atom": 0.06,
    "band_gap": 0.45,
    "e_above_hull": 0.03,
}


def discover_formal_generation_files(root_path: str | Path) -> list[Path]:
    """Return formal seed-42 template and ab-initio generation outputs."""
    root = Path(root_path)
    paths = set(root.rglob("eval_gen_template_*_n4096_seed42.pt"))
    paths.update(root.rglob("eval_gen_abinitio_empirical_*_n4096_seed42.pt"))
    return sorted(path for path in paths if path.is_file() and path.stat().st_size > 0)


def extract_condition_targets(payload: dict[str, Any]) -> dict[str, float]:
    raw = payload.get("conditions") or {}
    targets: dict[str, float] = {}
    if isinstance(raw, dict):
        for name, value in raw.items():
            if name in PROPERTY_NAMES:
                targets[name] = float(value.item() if hasattr(value, "item") else value)

    legacy_name = payload.get("property_name")
    legacy_value = payload.get("target_value")
    if legacy_name in PROPERTY_NAMES and legacy_value is not None:
        targets.setdefault(legacy_name, float(legacy_value))
    return targets


def compute_property_metrics(
    predictions: dict[str, np.ndarray],
    targets: dict[str, float],
    tolerances: dict[str, float],
    total_count: int,
    graph_success_count: int,
    reference_predictions: dict[str, np.ndarray] | None = None,
    reference_targets: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Calculate distribution, target-adherence, and joint-hit metrics."""
    metrics: dict[str, Any] = {
        "n_total": int(total_count),
        "n_graph_success": int(graph_success_count),
        "graph_success_rate": float(graph_success_count / total_count) if total_count else 0.0,
        "condition_targets": dict(targets),
        "tolerances": {name: float(tolerances[name]) for name in PROPERTY_NAMES},
        "properties": {},
    }

    finite_masks: dict[str, np.ndarray] = {}
    hit_masks: dict[str, np.ndarray] = {}
    for name in PROPERTY_NAMES:
        values = np.asarray(predictions[name], dtype=float).reshape(-1)
        finite = np.isfinite(values)
        finite_masks[name] = finite
        valid_values = values[finite]
        prop_metrics: dict[str, Any] = {
            "n_predicted": int(finite.sum()),
            "prediction_rate": float(finite.mean()) if total_count else 0.0,
        }

        if valid_values.size:
            prop_metrics.update({
                "mean": float(np.mean(valid_values)),
                "std": float(np.std(valid_values)),
                "median": float(np.median(valid_values)),
                "p05": float(np.quantile(valid_values, 0.05)),
                "p95": float(np.quantile(valid_values, 0.95)),
                "min": float(np.min(valid_values)),
                "max": float(np.max(valid_values)),
            })

            if reference_predictions and name in reference_predictions:
                reference = np.asarray(reference_predictions[name], dtype=float)
                reference = reference[np.isfinite(reference)]
                if reference.size:
                    prop_metrics["wdist_to_test_predictions"] = float(
                        wasserstein_distance(valid_values, reference)
                    )
            if reference_targets and name in reference_targets:
                reference = np.asarray(reference_targets[name], dtype=float)
                reference = reference[np.isfinite(reference)]
                if reference.size:
                    prop_metrics["wdist_to_test_targets"] = float(
                        wasserstein_distance(valid_values, reference)
                    )

        if name in targets:
            target = float(targets[name])
            errors = np.abs(values - target)
            hits = finite & (errors <= float(tolerances[name]))
            hit_masks[name] = hits
            finite_errors = errors[finite]
            prop_metrics.update({
                "target": target,
                "target_tolerance": float(tolerances[name]),
                "target_mae": float(np.mean(finite_errors)) if finite_errors.size else None,
                "target_rmse": (
                    float(np.sqrt(np.mean(finite_errors ** 2)))
                    if finite_errors.size else None
                ),
                "target_bias": (
                    float(np.mean(values[finite] - target))
                    if finite_errors.size else None
                ),
                "n_hits": int(hits.sum()),
                "hit_rate_predicted": (
                    float(hits.sum() / finite.sum()) if finite.any() else 0.0
                ),
                "hit_rate_all": float(hits.sum() / total_count) if total_count else 0.0,
            })

        metrics["properties"][name] = prop_metrics

    if targets:
        target_names = [name for name in PROPERTY_NAMES if name in targets]
        joint_predictable = np.logical_and.reduce([
            finite_masks[name] for name in target_names
        ])
        joint_hits = np.logical_and.reduce([hit_masks[name] for name in target_names])
        metrics["joint"] = {
            "target_properties": target_names,
            "n_joint_predictable": int(joint_predictable.sum()),
            "n_joint_hits": int(joint_hits.sum()),
            "joint_hit_rate_predicted": (
                float(joint_hits.sum() / joint_predictable.sum())
                if joint_predictable.any() else 0.0
            ),
            "joint_hit_rate_all": (
                float(joint_hits.sum() / total_count) if total_count else 0.0
            ),
        }
    else:
        metrics["joint"] = None

    return metrics


def load_generation_payload(path: str | Path) -> dict[str, Any]:
    """Load a generated tensor payload without enabling arbitrary pickle code."""
    import torch

    torch.serialization.add_safe_globals([argparse.Namespace])
    return torch.load(path, map_location="cpu", weights_only=True)


def _crystal_array_list(payload: dict[str, Any]) -> list[dict[str, np.ndarray]]:
    from cgdit.common.evaluation_utils import get_crystals_list

    arrays = get_crystals_list(
        payload["frac_coords"], payload["atom_types"], payload["lengths"],
        payload["angles"], payload["num_atoms"],
    )
    # Diffusion outputs zero-based element classes; predictor graphs use Z=1...N.
    for crystal in arrays:
        crystal["atom_types"] = np.asarray(crystal["atom_types"], dtype=np.int64) + 1
    return arrays


def _build_graph(task: tuple[int, dict[str, np.ndarray], bool, bool, str]):
    index, crystal_array, niggli, primitive, graph_method = task
    try:
        import torch
        from pymatgen.core import Lattice, Structure
        from torch_geometric.data import Data

        from cgdit.common.data_utils import build_crystal_graph

        structure = Structure(
            lattice=Lattice.from_parameters(*(
                crystal_array["lengths"].tolist()
                + crystal_array["angles"].tolist()
            )),
            species=crystal_array["atom_types"],
            coords=crystal_array["frac_coords"],
            coords_are_cartesian=False,
        )
        if primitive:
            structure = structure.get_primitive_structure()
        if niggli:
            structure = structure.get_reduced_structure()
        structure = Structure(
            lattice=Lattice.from_parameters(*structure.lattice.parameters),
            species=structure.species,
            coords=structure.frac_coords,
            coords_are_cartesian=False,
        )

        (
            frac_coords, atom_types, lengths, angles,
            edge_indices, to_jimages, num_atoms,
        ) = build_crystal_graph(structure, graph_method)
        edge_indices = np.asarray(edge_indices)
        to_jimages = np.asarray(to_jimages)
        if edge_indices.size == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            to_jimages_tensor = torch.empty((0, 3), dtype=torch.long)
            num_bonds = 0
        else:
            edge_index = torch.as_tensor(
                edge_indices.T, dtype=torch.long
            ).contiguous()
            to_jimages_tensor = torch.as_tensor(to_jimages, dtype=torch.long)
            num_bonds = int(edge_indices.shape[0])

        data = Data(
            frac_coords=torch.as_tensor(frac_coords, dtype=torch.float),
            atom_types=torch.as_tensor(atom_types, dtype=torch.long),
            lengths=torch.as_tensor(lengths, dtype=torch.float).view(1, -1),
            angles=torch.as_tensor(angles, dtype=torch.float).view(1, -1),
            edge_index=edge_index,
            to_jimages=to_jimages_tensor,
            num_atoms=int(num_atoms),
            num_bonds=num_bonds,
            num_nodes=int(num_atoms),
        )
        return index, data, None
    except Exception as exc:
        return index, None, f"{type(exc).__name__}: {exc}"


def build_graphs(
    crystal_arrays: list[dict[str, np.ndarray]],
    num_workers: int,
    niggli: bool = True,
    primitive: bool = False,
    graph_method: str = "crystalnn",
) -> tuple[list[Any], list[int], list[str | None]]:
    tasks = [
        (index, crystal, niggli, primitive, graph_method)
        for index, crystal in enumerate(crystal_arrays)
    ]
    if num_workers > 1:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context("spawn"),
        ) as executor:
            results = list(tqdm(
                executor.map(_build_graph, tasks, chunksize=8),
                total=len(tasks),
                desc="Building graphs",
            ))
    else:
        results = [_build_graph(task) for task in tqdm(tasks, desc="Building graphs")]

    data_list: list[Any] = []
    successful_indices: list[int] = []
    errors: list[str | None] = [None] * len(tasks)
    for index, data, error in results:
        if data is not None:
            successful_indices.append(index)
            data_list.append(data)
        else:
            errors[index] = error
    return data_list, successful_indices, errors


def resolve_checkpoint(run_path: str | Path) -> Path:
    path = Path(run_path)
    if path.is_file() and path.suffix == ".ckpt":
        return path
    checkpoints = sorted(path.glob("*.ckpt"))
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"Expected exactly one checkpoint in {path}, found {len(checkpoints)}"
        )
    return checkpoints[0]


def predict_graphs(
    checkpoint: str | Path,
    expected_property: str,
    data_list: list[Any],
    successful_indices: list[int],
    total_count: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    import torch
    from torch_geometric.loader import DataLoader

    from cgdit.prop_models.gnn_models.m3gnet import M3GNetSurrogate

    output = np.full(total_count, np.nan, dtype=np.float64)
    if not data_list:
        return output

    model = M3GNetSurrogate.load_from_checkpoint(
        str(checkpoint), map_location=device
    )
    if model.target_prop != expected_property:
        raise RuntimeError(
            f"Checkpoint {checkpoint} predicts {model.target_prop}, "
            f"expected {expected_property}"
        )
    model.eval().to(device)
    predictions: list[np.ndarray] = []
    loader = DataLoader(
        data_list, batch_size=batch_size, shuffle=False, num_workers=0
    )
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Predicting {model.target_prop}"):
            values = model(batch.to(device)).detach().cpu().numpy().reshape(-1)
            predictions.append(values)
    output[np.asarray(successful_indices)] = np.concatenate(predictions)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return output


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, dataframe: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    dataframe.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                visit(f"{prefix}.{key}" if prefix else key, nested)
        elif isinstance(value, list):
            flat[prefix] = ",".join(map(str, value))
        else:
            flat[prefix] = value

    visit("", metrics)
    return flat


def evaluate_generation_files(
    generation_files: Iterable[Path],
    predictor_runs: dict[str, Path],
    output_dir: Path,
    tolerances: dict[str, float],
    batch_size: int = 64,
    num_workers: int = 8,
    device: str = "cuda",
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    import pandas as pd

    generation_files = list(generation_files)
    if not generation_files:
        raise ValueError("No formal generation files were provided")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        name: resolve_checkpoint(predictor_runs[name]) for name in PROPERTY_NAMES
    }
    reference_predictions = {
        name: np.load(Path(predictor_runs[name]) / "test_preds.npy")
        for name in PROPERTY_NAMES
    }
    reference_targets = {
        name: np.load(Path(predictor_runs[name]) / "test_targets.npy")
        for name in PROPERTY_NAMES
    }

    summaries: list[dict[str, Any]] = []
    for position, generation_path in enumerate(generation_files, start=1):
        label = generation_path.stem.removeprefix("eval_gen_")
        predictions_path = evaluation_output_path(
            generation_path,
            "property_predictions",
            f"eval_properties_gen_{label}_predictor_seed42.csv",
        )
        metrics_path = evaluation_output_path(
            generation_path,
            "property_metrics",
            f"eval_property_metrics_gen_{label}_predictor_seed42.json",
        )
        print(f"\n[{position}/{len(generation_files)}] {generation_path}")

        predictions_exist = predictions_path.exists() and predictions_path.stat().st_size > 0
        metrics_exist = metrics_path.exists() and metrics_path.stat().st_size > 0
        if not overwrite and predictions_exist and metrics_exist:
            print(f"[SKIP] Existing property results: {metrics_path}")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            record_evaluation_metric(
                "per_structure_property_predictions",
                predictions_path,
                generation_path,
                checkpoints,
            )
            record_evaluation_metric(
                "aggregate_property_metrics",
                metrics_path,
                generation_path,
                checkpoints,
            )
            summaries.append(_flatten_metrics(metrics))
            continue

        payload = load_generation_payload(generation_path)
        crystal_arrays = _crystal_array_list(payload)
        data_list, successful_indices, graph_errors = build_graphs(
            crystal_arrays, num_workers=num_workers
        )
        predictions = {
            name: predict_graphs(
                checkpoints[name], name, data_list, successful_indices,
                len(crystal_arrays), batch_size, device,
            )
            for name in PROPERTY_NAMES
        }
        condition_targets = extract_condition_targets(payload)
        metrics = compute_property_metrics(
            predictions=predictions,
            targets=condition_targets,
            tolerances=tolerances,
            total_count=len(crystal_arrays),
            graph_success_count=len(successful_indices),
            reference_predictions=reference_predictions,
            reference_targets=reference_targets,
        )
        structural_metrics_path = evaluation_output_path(
            generation_path,
            "structural_metrics",
            f"eval_metrics_gen_{label}.json",
        )
        if not structural_metrics_path.exists():
            raise RuntimeError(
                f"Missing structural metrics: {structural_metrics_path}"
            )
        structural_metrics = json.loads(
            structural_metrics_path.read_text(encoding="utf-8")
        )
        metrics.update({
            "source_file": provenance_path(generation_path),
            "generation_label": label,
            "generation_model_dir": model_root_from_generation(
                generation_path
            ).name,
            "structural_metrics_file": provenance_path(structural_metrics_path),
            "structural_metrics": structural_metrics,
            "predictor_checkpoints": {
                name: provenance_path(checkpoints[name]) for name in PROPERTY_NAMES
            },
        })

        table: dict[str, Any] = {
            "structure_index": np.arange(len(crystal_arrays)),
            "graph_success": [error is None for error in graph_errors],
            "graph_error": [error or "" for error in graph_errors],
        }
        for name in PROPERTY_NAMES:
            values = predictions[name]
            table[f"predicted_{name}"] = values
            if name in condition_targets:
                errors = np.abs(values - condition_targets[name])
                table[f"target_{name}"] = condition_targets[name]
                table[f"absolute_error_{name}"] = errors
                table[f"hit_{name}"] = (
                    np.isfinite(values) & (errors <= tolerances[name])
                )
        if condition_targets:
            target_names = [
                name for name in PROPERTY_NAMES if name in condition_targets
            ]
            table["joint_hit"] = np.logical_and.reduce([
                np.asarray(table[f"hit_{name}"], dtype=bool)
                for name in target_names
            ])

        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_csv(predictions_path, pd.DataFrame(table))
        _atomic_json(metrics_path, metrics)
        record_evaluation_metric(
            "per_structure_property_predictions",
            predictions_path,
            generation_path,
            checkpoints,
        )
        record_evaluation_metric(
            "aggregate_property_metrics",
            metrics_path,
            generation_path,
            checkpoints,
        )
        summaries.append(_flatten_metrics(metrics))
        del payload, crystal_arrays, data_list, predictions

    summary_table = pd.DataFrame(summaries)
    _atomic_csv(output_dir / "seed42_property_summary.csv", summary_table)
    _atomic_json(
        output_dir / "seed42_property_summary.json", {"groups": summaries}
    )
    source_manifests = sorted({
        model_root_from_generation(path) / "evaluations/source_manifest.json"
        for path in generation_files
    })
    _atomic_json(
        output_dir / "source_manifest.json",
        {
            "summary_directory": provenance_path(output_dir),
            "summary_files": [
                provenance_path(output_dir / "seed42_property_summary.json"),
                provenance_path(output_dir / "seed42_property_summary.csv"),
            ],
            "source_model_manifests": [
                provenance_path(path) for path in source_manifests
            ],
            "source_groups": len(summaries),
            "source_predictors": {
                name: provenance_path(path) for name, path in checkpoints.items()
            },
        },
    )
    return summaries
