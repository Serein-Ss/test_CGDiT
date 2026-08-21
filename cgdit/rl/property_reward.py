"""Frozen M3GNet property rewards for generated crystal states."""

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from pymatgen.core import Lattice, Structure
from torch_geometric.loader import DataLoader

from cgdit.common.evaluation_utils import (
    get_crystals_list,
    lattices_to_params_shape,
    smact_validity,
    structure_validity,
)
from cgdit.evaluation.generated_properties import build_graphs
from cgdit.prop_models.gnn_models.m3gnet import M3GNetSurrogate

from .rewards import MaterialRewardResult, general_material_reward, property_reward


@dataclass(frozen=True)
class PropertyRewardSpec:
    mode: str
    target: float | None = None
    tolerance: float = 1.0
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True)
class PropertyRewardEvaluation:
    reward: MaterialRewardResult
    predictions: dict[str, torch.Tensor]
    valid: torch.Tensor
    graph_errors: tuple[str | None, ...]


def _crystal_is_valid(crystal: dict[str, np.ndarray]) -> bool:
    try:
        structure = Structure(
            lattice=Lattice.from_parameters(
                *(crystal["lengths"].tolist() + crystal["angles"].tolist())
            ),
            species=crystal["atom_types"],
            coords=crystal["frac_coords"],
            coords_are_cartesian=False,
        )
        counts = Counter(int(value) for value in crystal["atom_types"])
        composition = sorted(counts.items())
        elements, amounts = zip(*composition)
        divisor = np.gcd.reduce(amounts)
        reduced_amounts = tuple(int(amount // divisor) for amount in amounts)
        return bool(
            smact_validity(tuple(elements), reduced_amounts)
            and structure_validity(structure)
        )
    except Exception:
        return False


class M3GNetRewardAdapter:
    """Evaluate fixed-scale rewards with frozen property predictors."""

    def __init__(
        self,
        predictors: dict[str, torch.nn.Module],
        specs: dict[str, PropertyRewardSpec],
        device: str | torch.device,
        batch_size: int = 32,
        invalid_penalty: float = 1.0,
        property_mode: str = "bottleneck",
        property_weights: dict[str, float] | None = None,
    ):
        if set(predictors) != set(specs):
            raise ValueError("predictors and specs must use the same property names")
        self.predictors = predictors
        self.specs = specs
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.invalid_penalty = invalid_penalty
        self.property_mode = property_mode
        self.property_weights = property_weights
        for name, predictor in self.predictors.items():
            target_prop = getattr(predictor, "target_prop", name)
            if target_prop != name:
                raise ValueError(
                    f"Predictor for {name} reports target property {target_prop}"
                )
            predictor.eval().to(self.device)
            for parameter in predictor.parameters():
                parameter.requires_grad_(False)

    @classmethod
    def from_checkpoints(
        cls,
        checkpoints: dict[str, str],
        specs: dict[str, PropertyRewardSpec],
        device: str | torch.device,
        **kwargs,
    ) -> "M3GNetRewardAdapter":
        predictors = {
            name: M3GNetSurrogate.load_from_checkpoint(
                checkpoint,
                map_location=device,
            )
            for name, checkpoint in checkpoints.items()
        }
        return cls(predictors, specs, device, **kwargs)

    def evaluate_state(
        self,
        final_state: Any,
        num_atoms: torch.Tensor,
        crystal_family: torch.nn.Module,
        num_workers: int = 0,
    ) -> PropertyRewardEvaluation:
        with torch.inference_mode():
            lattices = crystal_family.v2m(final_state.crys_fam)
            lengths, angles = lattices_to_params_shape(lattices)
        crystal_arrays = get_crystals_list(
            final_state.frac_coords,
            final_state.atom_types,
            lengths,
            angles,
            num_atoms,
        )
        for crystal in crystal_arrays:
            crystal["atom_types"] = np.asarray(
                crystal["atom_types"], dtype=np.int64
            ) + 1
        return self.evaluate_arrays(crystal_arrays, num_workers=num_workers)

    def evaluate_arrays(
        self,
        crystal_arrays: list[dict[str, np.ndarray]],
        num_workers: int = 0,
    ) -> PropertyRewardEvaluation:
        validity = torch.tensor(
            [_crystal_is_valid(crystal) for crystal in crystal_arrays],
            dtype=torch.bool,
            device=self.device,
        )
        data_list, successful_indices, errors = build_graphs(
            crystal_arrays,
            num_workers=num_workers,
        )
        return self.evaluate_graphs(
            data_list,
            successful_indices,
            validity,
            tuple(errors),
            total_count=len(crystal_arrays),
        )

    def evaluate_graphs(
        self,
        data_list: list[Any],
        successful_indices: list[int],
        validity: torch.Tensor,
        graph_errors: tuple[str | None, ...],
        total_count: int,
    ) -> PropertyRewardEvaluation:
        if validity.shape != (total_count,):
            raise ValueError("validity must provide one value per generated structure")
        predictions = {
            name: torch.full(
                (total_count,),
                torch.nan,
                dtype=torch.float,
                device=self.device,
            )
            for name in self.predictors
        }
        if data_list:
            loader = DataLoader(
                data_list,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            )
            collected = {name: [] for name in self.predictors}
            with torch.inference_mode():
                for batch in loader:
                    batch = batch.to(self.device)
                    for name, predictor in self.predictors.items():
                        collected[name].append(predictor(batch).reshape(-1))
            indices = torch.as_tensor(
                successful_indices,
                dtype=torch.long,
                device=self.device,
            )
            for name in predictions:
                predictions[name][indices] = torch.cat(collected[name])

        finite = torch.stack([
            torch.isfinite(value) for value in predictions.values()
        ]).all(dim=0)
        valid = validity.to(self.device) & finite
        property_scores = {}
        for name, values in predictions.items():
            spec = self.specs[name]
            safe_values = torch.nan_to_num(values)
            property_scores[name] = property_reward(
                safe_values,
                mode=spec.mode,
                target=spec.target,
                tolerance=spec.tolerance,
                lower=spec.lower,
                upper=spec.upper,
            )
        reward = general_material_reward(
            property_scores,
            valid,
            property_mode=self.property_mode,
            property_weights=self.property_weights,
            invalid_penalty=self.invalid_penalty,
        )
        return PropertyRewardEvaluation(
            reward=reward,
            predictions=predictions,
            valid=valid,
            graph_errors=graph_errors,
        )
