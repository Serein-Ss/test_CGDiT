from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.dataset_figures.plot_datasets import (
    apply_publication_style,
    crystal_system,
    discover_datasets,
    empirical_cdf,
    save_figure,
)


def test_crystal_system_boundaries():
    assert crystal_system(1) == "Triclinic"
    assert crystal_system(15) == "Monoclinic"
    assert crystal_system(74) == "Orthorhombic"
    assert crystal_system(142) == "Tetragonal"
    assert crystal_system(167) == "Trigonal"
    assert crystal_system(194) == "Hexagonal"
    assert crystal_system(230) == "Cubic"
    assert crystal_system(0) == "Unknown"
    assert crystal_system(np.nan) == "Unknown"


def test_empirical_cdf_drops_only_non_finite_values():
    x, y = empirical_cdf(pd.Series([3.0, np.nan, 1.0, np.inf, 2.0]))
    np.testing.assert_array_equal(x, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(y, [1 / 3, 2 / 3, 1.0])


def test_discover_datasets_requires_all_three_splits(tmp_path: Path):
    complete = tmp_path / "complete"
    complete.mkdir()
    for split in ("train", "val", "test"):
        (complete / f"{split}.csv").write_text("value\n1\n", encoding="utf-8")
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "train.csv").write_text("value\n1\n", encoding="utf-8")
    empty = tmp_path / "empty"
    empty.mkdir()

    found, incomplete = discover_datasets(tmp_path)

    assert found == [complete]
    assert incomplete == [empty, partial]


def test_png_export(tmp_path: Path):
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("PNG label")

    paths = save_figure(fig, tmp_path / "figure", formats=("png",))

    assert paths == [tmp_path / "figure.png"]
    assert paths[0].stat().st_size > 0
