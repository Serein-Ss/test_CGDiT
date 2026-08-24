"""Binary high-Tc gate built on the CGDiT diffusion backbone."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryMatthewsCorrCoef,
)

from cgdit.prop_models.diffusion_backbone import DiffusionBackboneRegressor


class DiffusionBackboneBinaryClassifier(DiffusionBackboneRegressor):
    """Predict the probability that a crystal belongs to the high-Tc class."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "classification_pos_weight",
            torch.tensor(float(self.hparams.get("pos_weight", 1.0))),
            persistent=False,
        )
        self.train_ap = BinaryAveragePrecision()
        self.val_ap = BinaryAveragePrecision()
        self.test_ap = BinaryAveragePrecision()
        self.test_auroc = BinaryAUROC()
        self.test_f1 = BinaryF1Score()
        self.test_mcc = BinaryMatthewsCorrCoef()

    def _classification_step(
        self,
        batch: Any,
        prefix: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self(batch)
        targets = self._targets(batch).float()
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.classification_pos_weight,
        )
        probabilities = torch.sigmoid(logits)
        batch_size = int(batch.num_graphs)
        self.log(
            f"{prefix}_loss",
            loss,
            on_step=prefix == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        if prefix == "train":
            self.train_ap.update(probabilities, targets.long())
            self.log(
                "train_ap", self.train_ap, on_step=False, on_epoch=True,
                prog_bar=True, batch_size=batch_size,
            )
        elif prefix == "val":
            self.val_ap.update(probabilities, targets.long())
            self.log(
                "val_ap", self.val_ap, on_step=False, on_epoch=True,
                prog_bar=True, batch_size=batch_size,
            )
        return logits, targets, loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        _, _, loss = self._classification_step(batch, "train")
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        _, _, loss = self._classification_step(batch, "val")
        return loss

    def on_test_epoch_start(self) -> None:
        self.test_logits: list[torch.Tensor] = []
        self.test_targets: list[torch.Tensor] = []

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        logits, targets, loss = self._classification_step(batch, "test")
        probabilities = torch.sigmoid(logits)
        target_classes = targets.long()
        self.test_ap.update(probabilities, target_classes)
        self.test_auroc.update(probabilities, target_classes)
        self.test_f1.update(probabilities, target_classes)
        self.test_mcc.update(probabilities, target_classes)
        batch_size = int(batch.num_graphs)
        self.log_dict(
            {
                "test_ap": self.test_ap,
                "test_auroc": self.test_auroc,
                "test_f1_0p5": self.test_f1,
                "test_mcc_0p5": self.test_mcc,
            },
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.test_logits.append(logits.detach().cpu())
        self.test_targets.append(targets.detach().cpu())
        return loss

    def on_test_epoch_end(self) -> None:
        output_dir = Path(self.trainer.default_root_dir)
        logits_tensor = torch.cat(self.test_logits)
        logits = logits_tensor.numpy()
        targets = torch.cat(self.test_targets).numpy()
        np.save(output_dir / "test_logits.npy", logits)
        np.save(output_dir / "test_probs.npy", torch.sigmoid(logits_tensor).numpy())
        np.save(output_dir / "test_targets.npy", targets)
        self.test_logits.clear()
        self.test_targets.clear()
