"""Composition helpers for file-based RL experiment configurations."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_rl_config(path: str | Path) -> DictConfig:
    """Load an RL experiment and recursively merge its relative includes."""
    return _load_rl_config(Path(path).resolve(), ())


def _load_rl_config(path: Path, stack: tuple[Path, ...]) -> DictConfig:
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Cyclic RL config include: {chain}")

    config = OmegaConf.load(path)
    includes = config.pop("includes", [])
    if isinstance(includes, str):
        includes = [includes]

    merged = OmegaConf.create()
    for include in includes:
        include_path = Path(include)
        if not include_path.is_absolute():
            include_path = path.parent / include_path
        merged = OmegaConf.merge(
            merged,
            _load_rl_config(include_path.resolve(), (*stack, path)),
        )

    return OmegaConf.merge(merged, config)
