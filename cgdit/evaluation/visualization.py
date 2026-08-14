"""Plotting helpers for evaluation outputs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def plot_property_parity(predictions, targets, output_path):
    """Create a parity plot and return its MAE and R-squared metrics."""
    predictions = np.asarray(predictions).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {predictions.shape} != {targets.shape}"
        )

    mae = float(mean_absolute_error(targets, predictions))
    r2 = float(r2_score(targets, predictions))
    minimum = min(np.min(targets), np.min(predictions)) - 0.5
    maximum = max(np.max(targets), np.max(predictions)) + 0.5

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(targets, predictions, alpha=0.5, color="#1f77b4", edgecolors="white", s=40)
    axis.plot([minimum, maximum], [minimum, maximum], "r--", linewidth=2)
    axis.set(xlim=(minimum, maximum), ylim=(minimum, maximum),
             xlabel="True Value", ylabel="Predicted Value", title="M3GNet Parity Plot")
    axis.text(
        0.05,
        0.95,
        f"MAE = {mae:.3f}\n$R^2$ = {r2:.3f}",
        transform=axis.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
    )
    axis.grid(True, linestyle=":", alpha=0.6)
    figure.tight_layout()

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return {"mae": mae, "r2": r2, "output": str(output_path)}


def plot_property_parity_from_files(predictions_path, targets_path, output_path):
    return plot_property_parity(
        np.load(predictions_path),
        np.load(targets_path),
        output_path,
    )
