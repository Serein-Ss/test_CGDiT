"""Property regression with the CGDiT diffusion decoder as a crystal backbone."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn as nn

from cgdit.common.data_utils import lattice_params_to_matrix_torch
from cgdit.pl_modules.diffusion import BaseModule, SinusoidalTimeEmbeddings
from cgdit.pl_modules.lattice.crystal_family import CrystalFamily


def resolve_checkpoint(path: str | Path) -> Path:
    """Resolve a checkpoint file or a model directory containing checkpoints."""
    path = Path(path).expanduser().resolve()
    if path.is_file():
        if path.suffix != ".ckpt":
            raise ValueError(f"Expected a .ckpt file, got: {path}")
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Pretrained model path does not exist: {path}")

    candidates = list(path.glob("*.ckpt"))
    if not candidates:
        candidates = list(path.glob("model/*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under: {path}")

    def checkpoint_step(candidate: Path) -> tuple[int, float]:
        match = re.search(r"step=(\d+)", candidate.name)
        return (int(match.group(1)) if match else -1, candidate.stat().st_mtime)

    return max(candidates, key=checkpoint_step)


def extract_decoder_state(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Strip known Lightning prefixes from diffusion decoder parameters."""
    prefixes = ("decoder.", "model.decoder.", "backbone.")
    extracted: dict[str, torch.Tensor] = {}
    for name, value in state_dict.items():
        for prefix in prefixes:
            if name.startswith(prefix):
                extracted[name[len(prefix):]] = value
                break
    return extracted


class DiffusionBackboneRegressor(BaseModule):
    """Use the denoising CSPNet at t=0 and add a scalar regression readout."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.target_prop = self.hparams.get("target_prop", "tc")
        target_mean = float(self.hparams.get("target_mean", 0.0))
        target_std = float(self.hparams.get("target_std", 1.0))
        if target_std <= 0:
            raise ValueError(f"target_std must be positive, got {target_std}")
        self.register_buffer("target_mean", torch.tensor(target_mean), persistent=False)
        self.register_buffer("target_std", torch.tensor(target_std), persistent=False)

        self.time_dim = int(self.hparams.time_dim)
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.crystal_family = CrystalFamily()
        self.backbone = hydra.utils.instantiate(
            self.hparams.decoder,
            _recursive_=False,
        )
        if not getattr(self.backbone, "pred_scalar", False):
            raise ValueError("The CSPNet decoder must set pred_scalar=true")

        self.loss_fn = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.pretrained_load_report: dict[str, Any] = {
            "mode": "scratch",
            "checkpoint": None,
            "matched_tensors": 0,
        }

        pretrained_path = self.hparams.get("pretrained_model_path")
        if pretrained_path:
            self._load_pretrained_decoder(pretrained_path)

        # The diffusion output heads are not used by scalar regression.
        for head_name in ("coord_out", "lattice_out", "type_out"):
            head = getattr(self.backbone, head_name, None)
            if head is not None:
                head.requires_grad_(False)

    def _load_pretrained_decoder(self, model_path: str | Path) -> None:
        checkpoint = resolve_checkpoint(model_path)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        source_state = payload.get("state_dict", payload)
        decoder_state = extract_decoder_state(source_state)
        if not decoder_state:
            raise RuntimeError(f"No decoder parameters found in {checkpoint}")

        current_state = self.backbone.state_dict()
        reusable = {
            name: tensor
            for name, tensor in decoder_state.items()
            if name in current_state
            and current_state[name].shape == tensor.shape
            and not name.startswith("scalar_out.")
        }
        expected = [
            name
            for name in current_state
            if not name.startswith("scalar_out.")
        ]
        matched_fraction = len(reusable) / max(len(expected), 1)
        if matched_fraction < 0.90:
            raise RuntimeError(
                "Pretrained decoder is incompatible: "
                f"matched {len(reusable)}/{len(expected)} tensors from {checkpoint}"
            )

        load_result = self.backbone.load_state_dict(reusable, strict=False)
        self.backbone.scalar_out.reset_parameters()
        self.pretrained_load_report = {
            "mode": "pretrained_finetune",
            "checkpoint": str(checkpoint),
            "matched_tensors": len(reusable),
            "expected_backbone_tensors": len(expected),
            "matched_fraction": matched_fraction,
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
            "scalar_head_reinitialized": True,
        }

    def on_fit_start(self) -> None:
        output_path = Path(self.trainer.default_root_dir) / "pretrained_load_report.json"
        output_path.write_text(
            json.dumps(self.pretrained_load_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def forward(self, batch) -> torch.Tensor:
        num_graphs = int(batch.num_graphs)
        times = torch.zeros(num_graphs, device=batch.frac_coords.device)
        time_emb = self.time_embedding(times)

        lattice_matrices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        lattice_representation = self.crystal_family.m2v(
            self.crystal_family.de_so3(lattice_matrices)
        )
        atom_types = (batch.atom_types.long() - 1).clamp(min=0, max=99)
        predictions = self.backbone(
            time_emb,
            atom_types,
            batch.frac_coords,
            lattice_representation,
            batch.num_atoms,
            batch.batch,
        ).reshape(-1)
        return predictions * self.target_std + self.target_mean

    def _targets(self, batch) -> torch.Tensor:
        targets = getattr(batch, "y", None)
        if targets is None:
            targets = getattr(batch, self.target_prop, None)
        if targets is None:
            raise KeyError(f"Target {self.target_prop!r} is absent from the batch")
        return targets.reshape(-1)

    def _stats(
        self, predictions: torch.Tensor, targets: torch.Tensor, prefix: str
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        normalized_predictions = (predictions - self.target_mean) / self.target_std
        normalized_targets = (targets - self.target_mean) / self.target_std
        loss = self.loss_fn(normalized_predictions, normalized_targets)
        metrics = {
            f"{prefix}_loss": loss,
            f"{prefix}_mae": self.mae(predictions, targets),
            f"{prefix}_rmse": torch.sqrt(torch.mean((predictions - targets) ** 2)),
        }
        return metrics, loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        predictions = self(batch)
        targets = self._targets(batch)
        metrics, loss = self._stats(predictions, targets, "train")
        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=int(batch.num_graphs),
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        predictions = self(batch)
        targets = self._targets(batch)
        metrics, loss = self._stats(predictions, targets, "val")
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=int(batch.num_graphs),
        )
        return loss

    def on_test_epoch_start(self) -> None:
        self.test_predictions: list[torch.Tensor] = []
        self.test_targets: list[torch.Tensor] = []

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        predictions = self(batch)
        targets = self._targets(batch)
        self.test_predictions.append(predictions.detach().cpu())
        self.test_targets.append(targets.detach().cpu())
        metrics, loss = self._stats(predictions, targets, "test")
        self.log_dict(metrics, batch_size=int(batch.num_graphs))
        return loss

    def on_test_epoch_end(self) -> None:
        output_dir = Path(self.trainer.default_root_dir)
        np.save(
            output_dir / "test_preds.npy",
            torch.cat(self.test_predictions).numpy(),
        )
        np.save(
            output_dir / "test_targets.npy",
            torch.cat(self.test_targets).numpy(),
        )
        self.test_predictions.clear()
        self.test_targets.clear()
