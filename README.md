# Crystal Graph Diffusion Transformer (CGDiT)

---

CGDiT (Crystal Graph Diffusion Transformer) is a **graph-based diffusion model** designed for **crystal structure generation**.  
It combines graph neural networks with continuous coordinate/lattice diffusion and D3PM atom-type diffusion.

---

## Overview

Traditional diffusion models excel in image and text generation but struggle to handle the **symmetry, discreteness, and periodicity** of crystalline materials.  
CGDiT addresses this challenge by:
- Representing atomic systems as **crystal graphs**
- Modeling diffusion with **discrete noise schedules**
- Employing a symmetry-aware message-passing denoiser to predict clean graph states
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
uv sync
cp .env.template .env
```

Set the paths in `.env`. The locked environment uses Python 3.10, PyTorch 2.4.1
with CUDA 12.4, and matching PyG CUDA extensions. Run project commands through
`uv run`:

```bash
uv run pytest
uv run --locked python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

For stability evaluation, install the optional MatGL dependency:

```bash
uv sync --extra stability
```

The login node does not expose a GPU. Submit CUDA commands through SLURM with a
GPU allocation (for example, `--partition=rtx4090 --gres=gpu:1`); a false
`torch.cuda.is_available()` result on the login node is therefore expected.

## Datasets

## Generation and evaluation workflows

Reusable Python code is separated from command-line entry points:

- `cgdit/generation`: sampling, reconstruction, and symmetry generation APIs.
- `cgdit/evaluation`: generation metrics, stability, novelty, and property evaluation APIs.
- `scripts/cli/generation`: structure-generation commands.
- `scripts/cli/evaluation`: metric and evaluation commands.
- `scripts/cli/visualization`: evaluation plotting commands.

Run commands from the repository root through `uv run`:

```bash
uv run python -m scripts.cli.generation.generate --model_path <model_path>
uv run python -m scripts.cli.generation.generate_reconstruction --model_path <model_path> --dataset <dataset>
uv run python -m scripts.cli.evaluation.evaluate_metrics --root_path <model_path> --tasks gen --gt_file data/<dataset>/test.csv
uv run --extra stability python -m scripts.cli.evaluation.evaluate_stability --input <eval_gen.pt> --train-csv data/<dataset>/train.csv
```
