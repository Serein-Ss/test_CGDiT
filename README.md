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