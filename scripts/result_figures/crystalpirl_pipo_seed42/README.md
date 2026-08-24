# Seed=42 GRPO+PIPO figure pipeline

This package builds the single-seed PIPO comparator evidence used while
developing the CrystalPIRL article. It never relabels PIPO as CrystalPIRL.

The pipeline requires three completed GRPO+PIPO policies: formation energy,
band gap, and joint formation-energy/band-gap. Each policy must contain 200
updates and 4,096 seed-42 empirical-prior ab initio structures evaluated by the
same independent seed-123 FE/BG predictors.

Execution order:

1. `build_source_data.py` freezes training, paired-evaluation, property,
   quality, diversity, t-SNE, composition, and symmetry data.
2. `audit_conclusions.py` verifies every expected method, update, structure,
   evaluator, and claim boundary.
3. `plot_figures.py` creates a new asset directory only after the audit passes.

The current export contract is PNG only. Fig. 1 remains the user-approved
architecture placeholder; Fig. 2--Fig. 4 use measured seed-42 data.
