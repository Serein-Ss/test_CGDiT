# Unified dataset figures

This directory replaces the plotting logic that was embedded in
`pre_processing/data_analysis.ipynb`. The command discovers every directory
below `data/` that contains a complete `train.csv`, `val.csv`, and `test.csv`
set, then applies the same plotting and export policy to each dataset.

## Figure contract

- **Core conclusion:** determine whether the fixed train, validation, and test
  splits have comparable target-property, compositional, and structural
  coverage.
- **Archetype:** compact quantitative figures; one scientific question per
  file.
- **Backend:** Python with matplotlib.
- **Exports:** 300 dpi PNG only for the current development stage.
- **Integrity:** all finite observations are shown; target values are not
  imputed, KDE smoothing is not used, and no display-range clipping is applied.
- **Statistics:** target curves report the two-sample Kolmogorov–Smirnov
  statistic against the training split; categorical plots report normalized
  percentages and source tables retain counts.

C2DB-51 is treated as a two-dimensional dataset: its physical-scale figure
uses in-plane area per atom and layer thickness, and its symmetry figure uses
layer groups. The other datasets use volume per atom, density, and 3D crystal
systems. Empty or incomplete directories, currently including `magndata/`, are
reported and skipped rather than populated with synthetic data.

## Usage

From the repository root:

```bash
uv run python -m scripts.dataset_figures.plot_datasets
```

Useful focused runs:

```bash
# One dataset
uv run python -m scripts.dataset_figures.plot_datasets \
  --dataset mp_20

# Force CIF feature extraction instead of reusing a valid cache
uv run python -m scripts.dataset_figures.plot_datasets \
  --refresh-features

# Rebuild the model-parity panels as a separate, optional analysis
uv run python -m scripts.dataset_figures.plot_datasets \
  --include-model-parity
```

`--jobs` controls CIF-parsing processes and tree-model workers. Its default is
the smaller of eight and the available CPU count.

## Outputs

Each complete dataset receives:

```text
data/<dataset>/analysis/
├── diversity_analysis/
│   ├── 01_target_<property>_distribution.png
│   ├── 02_compositional_diversity.png
│   ├── 03_crystal_system_diversity.png
│   ├── 04_physical_properties_distribution.png
│   ├── 05_lattice_parameters.png
│   ├── 06_top_elements_frequency.png
│   ├── 07_feature_target_correlation.png
│   └── source_data/
│       ├── figure_audit.json
│       ├── split_summary.csv
│       ├── target_summary.csv
│       ├── composition_summary.csv
│       ├── symmetry_summary.csv
│       ├── element_frequency.csv
│       └── spearman_structure_target.csv
└── ml_figs_<property>/              # only with --include-model-parity
    ├── metrics.csv
    └── parity_plots.png
```

The structural-feature cache is reused only while the size and nanosecond
modification time of all three source CSVs still match its recorded input
fingerprint.
