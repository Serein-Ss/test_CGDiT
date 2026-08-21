# Crystal Graph Diffusion Transformer (CGDiT)

---

CGDiT (Crystal Graph Diffusion Transformer) is a **graph-based diffusion model** designed for **crystal structure generation**.  
It integrates the strengths of **graph neural networks (GNNs)** for atomic-level representation and **Transformer-based denoising** for efficient diffusion in the discrete lattice space.

---

## Overview

Traditional diffusion models excel in image and text generation but struggle to handle the **symmetry, discreteness, and periodicity** of crystalline materials.  
CGDiT addresses this challenge by:
- Representing atomic systems as **crystal graphs**
- Modeling diffusion with **discrete noise schedules**
- Employing a **Transformer-based denoiser** to predict clean graph states
- Supporting **conditional generation** (e.g., composition, symmetry, or property constraints)

---



## Table of Contents
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#usage)
- [Citation](#citation)

---

## Installation

### Setup

```bash
bash setup/setup.sh
```
Rename the `.env.template` file into `.env` and specify the following variables:
```
PROJECT_ROOT=/absolute/path/to/this/repo
HYDRA_JOBS=/absolute/path/to/save/hydra/outputs
WABDB_DIR=/absolute/path/to/save/wandb/outputs
```

## Datasets

## Generation and evaluation workflows

Reusable Python code is separated from command-line entry points:

- `cgdit/generation`: sampling, reconstruction, and symmetry generation APIs.
- `cgdit/evaluation`: generation metrics, stability, novelty, and property evaluation APIs.
- `scripts/cli/generation`: structure-generation commands.
- `scripts/cli/evaluation`: metric and evaluation commands.
- `scripts/cli/visualization`: evaluation plotting commands.

Run commands from the repository root in the `cgdit` environment:

```bash
python -m scripts.cli.generation.generate --model_path <model_path>
python -m scripts.cli.generation.generate_reconstruction --model_path <model_path> --dataset <dataset>
python -m scripts.cli.evaluation.evaluate_metrics --root_path <model_path> --tasks gen --gt_file data/<dataset>/test.csv
python -m scripts.cli.evaluation.evaluate_stability --input <eval_gen.pt> --train-csv data/<dataset>/train.csv
```

Only the module-based commands under `scripts/cli` are supported. Historical
`scripts/*.py` compatibility entry points have been removed.
