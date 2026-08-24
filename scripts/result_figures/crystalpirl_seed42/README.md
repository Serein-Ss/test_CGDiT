# CrystalPIRL seed=42 figure pipeline

This directory builds the exploratory, single-seed evidence for Fig. 1–Fig. 4.
It does not authorize a multi-seed robustness claim.

## Evidence contract

- Scope: MP-20, formation energy and band gap, seed=42 only.
- Methods: frozen base, FE/BG CFG, Best-of-8, open PPO/GRPO, original PIPO,
  and CrystalPIRL.
- RL matrix: `2 algorithms × 3 verifiers × 2 properties × 300 updates`.
- Each update uses `5 prompts × 16 trajectories` for every method arm.
  Every trajectory executes all 999 denoising steps. Training retains only the two replayed transitions and omits the unused full state stack to fit 24 GB GPU memory.
- Formal generation: 4,096 empirical-prior ab initio structures per final RL
  model. Template generation is reserved for functional tests and excluded from figures.
- Training reward predictor: M3GNet seed=42.
- Independent endpoint evaluator: the separately trained M3GNet seed=123.
- FE objective: minimize relative to the frozen threshold -1.5 eV/atom;
  target-yield reporting uses a ±0.30 eV/atom interval.
- BG objective: target 2.0 eV with a ±0.45 eV interval.
- Stability in this single-seed experiment is only the bounded seed123 formation-energy proxy
  `sigmoid((0 - FE) / 0.30)`. It is not an MLFF/DFT stability claim.
- Every plot must display `exploratory, seed=42`; no significance test or cross-seed
  error bar is permitted.
- Plotting reads frozen CSV files only. It must not reselect structures or tune
  thresholds.
- PNG is the only current export format, and plot titles are forbidden.

## Execution order

1. `build_source_data.py` validates and freezes update- and structure-level data.
2. `build_quality_source.py` adds deterministic uniqueness, novelty, composition,
   symmetry and embedding evidence.
3. `audit_conclusions.py` tests data completeness and claim boundaries.
4. `plot_figures.py` refuses to create `assets/` outputs unless the audit passes.

Quality matching samples `min(256, number of valid structures)` from each
group. A CrystalPIRL update rejected by the fixed probe has no required holdout
record by design; all Open/PIPO updates and every applied CrystalPIRL update
must have complete holdout evidence.

The output directory is supplied explicitly and should remain under the
corresponding `output/rl_finetune/seed42_300step_fig1_fig4/<job-id>/analysis/` tree until
the conclusion audit passes.

Example after every upstream job is complete:

```bash
TRAIN_ROOT=output/rl_finetune/seed42_300step_fig1_fig4/<training-job-id>
SOURCE_DIR="${TRAIN_ROOT}/analysis/source_data"

python scripts/result_figures/crystalpirl_seed42/build_source_data.py \
  --train-root "${TRAIN_ROOT}" --output-dir "${SOURCE_DIR}" --updates-per-run 300
python scripts/result_figures/crystalpirl_seed42/build_quality_source.py \
  --source-dir "${SOURCE_DIR}"
python scripts/result_figures/crystalpirl_seed42/audit_conclusions.py \
  --source-dir "${SOURCE_DIR}"
python scripts/result_figures/crystalpirl_seed42/plot_figures.py \
  --source-dir "${SOURCE_DIR}" \
  --asset-dir assets/crystalpirl_seed42_300step
```
