# Complete seed=42 CrystalPIRL figure pipeline

This package freezes the exploratory single-seed evidence for Fig. 1--Fig. 4.
It keeps GRPO+PIPO as a comparator and labels only the blockwise, paired,
dual-anchor gated runs as CrystalPIRL.

Required policy families are Base, three CFG models, three 200-update
GRPO+PIPO models, and three 200-update GRPO+CrystalPIRL models. Every generated
policy contributes 4,096 empirical-prior ab initio structures generated with
seed 42 and evaluated by the same independent seed-123 FE/BG predictors.

The pipeline does not require a positive result. Rejected or rolled-back
CrystalPIRL blocks remain in Source Data and in the conclusion audit. One seed
is exploratory evidence only; multi-seed robustness, MLFF relaxation, and DFT
validation remain outside this package.
