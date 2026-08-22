# Generation and predictor result figures

This directory contains the unified Python plotting entry point for the trained
CGDiT generation checkpoints and the current FE/BG property predictors.

## Figure contract

- **Core conclusion:** FE/BG conditioning shifts the full generated-property
  distributions toward the requested targets, including the joint FE-BG
  distribution; the frozen seed-42 FE/BG predictors have measurable held-out
  performance, but proxy predictions are not a substitute for a final
  independent evaluator.
- **Archetype:** compact quantitative comparison.
- **Backend:** Python with matplotlib only.
- **Exports:** 600 dpi PNG only, following the current project-stage request.
- **Structural evidence:** Figure 1 shows the 11 formal groups from the four
  active Base/FE/BG/FE+BG checkpoints.
- **Property evidence:** Figure 2 compares MP20 training data, matched
  unconditioned samples, template-conditioned samples, and ab-initio-conditioned
  samples using FE/BG marginal densities and all joint FE-BG points overlaid
  with 50%/90% density regions.
- **Structure evidence:** Figure 4 shows four unrelaxed generated unit cells
  selected by target proximity after composition and geometry validity checks.
- **Predictor evidence:** Figure 3 retains FE/BG parity, residual, and
  decile-calibration panels on 9046 held-out structures per completed run.
- **Source-data integrity:** all 23 formal groups and all original metrics remain
  in the plot-ready CSV. The compact figures omit archived Ehull rows and
  non-targeted diagnostic rows according to the documented display rule.
- **Statistics:** descriptive summaries only. One formal generation seed is
  available, so no replicate uncertainty or significance test is inferred.
- **Reviewer risk:** generation properties are surrogate predictions, the final
  evaluator is still missing, and seed 123 remains an audit comparison rather
  than an active reward ensemble.

## Run

From the repository root:

```bash
uv run --locked python -m scripts.result_figures.plot_results
```

An alternative output directory can be selected with `--output-dir`. The
default output is `assets/model_results/`.

## Outputs

```text
assets/model_results/
├── 01_generation_structural_quality.png
├── 02_generation_fe_bg_properties.png
├── 03_predictor_fe_bg_performance.png
├── 04_conditioned_structure_examples.png
└── source_data/
    ├── conditioned_structure_examples.csv
    ├── conditioning_distribution_metrics.csv
    ├── figure_manifest.json
    ├── generation_summary_plot_data.csv
    ├── predictor_calibration_plot_data.csv
    └── predictor_metrics.csv
```

The plot-ready CSV files are derived from the unmodified result summary,
per-structure property tables, generation tensors, and prediction arrays. They
do not replace the original files under output/.
