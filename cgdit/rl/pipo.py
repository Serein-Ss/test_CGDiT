"""Original scalar PIPO historical policy-improvement feedback."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PIPOFeedback:
    history_mean: float
    history_std: float
    signal: float
    modulation: float


def pipo_feedback(
    current_mean: float,
    history: list[float],
    negative_scale: float = 0.1,
    eps: float = 1.0e-6,
) -> PIPOFeedback | None:
    if not 0.0 <= negative_scale <= 1.0:
        raise ValueError("negative_scale must be in [0, 1]")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if len(history) < 2:
        return None

    values = torch.tensor(history, dtype=torch.float64)
    history_mean = float(values.mean().item())
    history_std = float(values.std(unbiased=True).item())
    signal = (current_mean - history_mean) / (history_std + eps)
    modulation = signal if signal >= 0.0 else negative_scale * signal
    return PIPOFeedback(
        history_mean=history_mean,
        history_std=history_std,
        signal=signal,
        modulation=modulation,
    )
