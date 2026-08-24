# Crystal Graph Diffusion Transformer

> 本文档对应当前 `test_CGDiT` 工作区，最后核对日期：2026-08-21。
> 文档中的命令默认在项目根目录执行，并优先使用重构后的模块化入口。


## Project Description

本项目是一个基于 DiffCSP 思路继续开发的晶体结构生成框架。模型使用具有周期性与对称性感知能力的图神经网络作为去噪骨干，在分数坐标、晶格和原子类型空间中执行扩散采样，并支持空间群、Wyckoff 位点以及材料性质条件控制。

当前项目的主要功能包括：

- Crystal Structure Prediction（CSP）与结构重构；
- 基于训练集模板的结构生成；
- Ab Initio 从头结构生成；
- 空间群和 Wyckoff 位点约束生成；
- Formation Energy、Band Gap、Energy Above Hull 等性质条件生成；
- 生成有效性、覆盖率、结构匹配、稳定性、新颖性和性质分布评估；
- M3GNetSurrogate 性质预测；
- 多条件与多模态条件注入实验。


## Project Status

当前主线模型是 `cgdit.pl_modules.diffusion.Diffusion`，主要训练入口是：

```shell
python cgdit/run.py data=<data_config> model=<model_config> expname=<experiment_name>
```

注意：项目根目录的 `main.py` 只是 PyCharm 自动生成的示例文件，不是训练入口。

当前本机保留 8 组 MP-20 生成模型训练结果：

| 实验 | 条件 | 模型目录 |
|---|---|---|
| mp20_base | 无性质条件 | `output/singlerun/2026-06-27/00-32-50-mp20_base` |
| mp20_fe | Formation Energy | `output/singlerun/2026-06-28/16-46-08-mp20_fe` |
| mp20_bg | Band Gap | `output/singlerun/2026-06-30/11-28-44-mp20_bg` |
| mp20_eh | Energy Above Hull | `output/singlerun/2026-07-02/04-07-02-mp20_eh` |
| mp20_fe_bg_eh | FE + BG + Ehull 三条件 | `output/singlerun/2026-08-05/13-12-29-mp20_fe_bg_eh` |
| mp20_fe_bg | FE + BG 双条件 | `output/singlerun/2026-08-07/07-54-15-mp20_fe_bg` |
| mp20_fe_eh | FE + Ehull 双条件 | `output/singlerun/2026-08-08/18-02-14-mp20_fe_eh` |
| mp20_bg_eh | BG + Ehull 双条件 | `output/singlerun/2026-08-10/02-53-51-mp20_bg_eh` |

当前强化学习研究范围暂时只启用 FE 和 BG。训练奖励与固定 probe 均冻结为 seed=42：FE 使用 `mp20_predictor_fe`，BG 使用 `mp20_predictor_bg`。FE/BG 的 seed=123 虽然已有测试输出，但多 seed 预测器体系尚未完整冻结，因此当前不组成 ensemble，也不参与 reward 或 checkpoint 选择。Ehull predictor 和所有含 Ehull 的生成模型只作为历史资产保留，不进入当前研究主线。

8 组生成模型仍完整保留以保证可追溯性；当前 FE/BG 主线实际使用 `mp20_base`、`mp20_fe`、`mp20_bg` 和 `mp20_fe_bg` 四个基线，对应 11 组正式 seed=42 生成/性质评估，每组 4096 个结构。完整 23 组结果不删除。具体状态见 `output/README.md` 和 `conf/rl/reward_models_mp20.yaml`。汇总文件中的 `/public/home/...` 是原服务器来源记录，本机执行必须使用当前项目相对路径。


## Project Structure

```text
test_CGDiT/
├── cgdit/
│   ├── common/
│   │   ├── constants.py
│   │   ├── data_utils.py
│   │   ├── evaluation_utils.py       # 模型、数据和评估公共工具
│   │   └── utils.py                  # 环境变量与项目根目录
│   ├── generation/
│   │   ├── general.py                # 模板与 Ab Initio 生成
│   │   ├── reconstruction.py         # CSP/重构采样
│   │   └── symmetry.py               # 空间群/Wyckoff 约束生成 API
│   ├── evaluation/
│   │   ├── metrics.py                # 通用生成与重构指标
│   │   ├── stability.py              # MLIP、E_hull、RDF、新颖性评估
│   │   ├── property.py               # M3GNetSurrogate 性质预测
│   │   └── visualization.py          # 评估结果绘图
│   ├── pl_data/                      # 数据集、DataModule、图构建
│   ├── pl_modules/
│   │   ├── cfg_utils/                # 条件嵌入与多模态 Adapter
│   │   ├── decoder/cspnet.py         # CSPNet 去噪骨干
│   │   ├── diff_utils/               # 连续/离散扩散工具
│   │   ├── lattice/                  # 晶格参数化
│   │   ├── diffusion.py              # 当前主线扩散模型
│   │   ├── diffusion_multimodal.py   # 多模态实验模型
│   │   └── diff2flow.py              # 实验性 Diff2Flow 模块
│   ├── prop_models/                  # 性质预测模型
│   └── run.py                        # Hydra + Lightning 训练入口
├── conf/
│   ├── data/                         # 数据集配置
│   ├── model/                        # 模型与条件组合配置
│   ├── optim/                        # 优化器配置
│   ├── train/                        # 训练和微调配置
│   ├── logging/                      # W&B 与日志配置
│   └── default.yaml                  # Hydra 总配置
├── data/                             # train/val/test CSV 与缓存
├── scripts/
│   ├── cli/
│   │   ├── generation/               # 结构生成命令
│   │   ├── evaluation/               # 评估命令
│   │   ├── visualization/            # 绘图命令
│   │   └── tools/                    # 查询与数据准备工具
│   └── dataset_figures/              # 统一数据集科学绘图
├── pre_processing/                   # 数据预处理与分析
├── post_processing/                  # 旧版结果后处理脚本
├── submit_python/                    # 服务器批量训练脚本
├── output/                           # Hydra 训练与评估结果
├── wandb/                            # W&B 本地记录
├── logs/                             # 批量训练日志
├── .env                              # 本机环境变量，不上传 GitHub
├── .env.template                     # 环境变量模板
├── requirements.txt                  # 既有 Conda/Python 环境快照
├── setup/                            # 既有安装与环境辅助文件
└── environment.yml                   # Conda 环境配置
```


## Project Application

- 晶体结构生成与 CSP；
- 基于目标性质的逆向材料设计；
- 晶体对称性约束生成；
- 生成模型有效性、覆盖率和多样性评估；
- 机器学习势辅助的结构弛豫与热力学稳定性筛选；
- “生成模型初筛—性质模型预测—DFT 精算”的材料发现流程。


## Model Architecture

### Backbone

当前主干网络是 `CSPNet`：

```text
Noisy Crystal Graph
    ├── atom types
    ├── fractional coordinates
    ├── lattice
    ├── space group / symmetry operations
    └── optional property conditions
            ↓
        CSPNet Decoder
            ↓
Predicted atom, coordinate and lattice denoising targets
```

对应配置位于：

```text
conf/model/decoder/cspnet.yaml
```

### Diffusion Model

`cgdit/pl_modules/diffusion.py` 是当前稳定主线，统一处理：

- 原子类型离散扩散；
- 分数坐标周期扩散；
- 晶格连续扩散；
- 条件嵌入；
- Classifier-Free Guidance（CFG）；
- 空间群与对称操作约束。

### Conditional Generation

模型配置中的 `conditions` 决定训练和采样时可使用的性质条件，例如：

```yaml
conditions:
  formation_energy_per_atom:
    type: scalar
    scale: 1.029374
    shift: -1.219803
  band_gap:
    type: scalar
    scale: 1.417749
    shift: 0.791848
  e_above_hull:
    type: scalar
    scale: 0.023265
    shift: 0.017478
```

`scale` 和 `shift` 必须与训练数据统计量一致。不能只在命令行中写一个模型没有训练过的条件名；生成脚本会警告，但该条件没有可靠的模型语义。

### Multimodal and Diff2Flow

- `diffusion_multimodal.py` 已有 MP-20 单条件和三条件配置，包含双通道 Adapter、对比学习和 Curriculum Learning；
- `diff2flow.py` 当前属于实验模块，没有对应的正式训练配置和标准评估入口；
- 新实验应先建立独立配置与可复现基线，不要直接替换当前 `Diffusion` 主线。


## Setup Environment

### Environment Files

当前保留的环境配置文件如下：

| 文件 | 用途 |
|---|---|
| `environment.yml` | 推荐的 Conda 环境入口，使用 Python 3.10、CUDA 12.4 PyTorch 和匹配的 PyG 轮子 |
| `requirements.txt` | 既有环境快照，仅用于版本核对，不建议在不同操作系统间直接安装 |
| `setup/` | 既有安装辅助文件，为兼容旧环境保留 |
| `.env.template` | 本机路径和可选凭据模板；实际 `.env` 不提交 |

### Conda 环境

当前服务器 GPU 节点使用 NVIDIA 550.54.15 驱动（CUDA 12.4），环境固定为
Python 3.10、PyTorch 2.4.1 cu124、PyTorch Geometric 2.7.0，以及匹配的
torch-scatter/torch-sparse `pt24cu124` 轮子。

首次创建环境：

```bash
conda env create -f environment.yml
conda activate cgdit
python -c "import torch, torch_geometric; print(torch.__version__, torch.version.cuda, torch_geometric.__version__)"
python -m pytest -q
```

稳定性评估需要 MatGL 时，在已激活的环境中安装：

```bash
python -m pip install "matgl>=2,<3"
```

已有同名环境需要同步时：

```bash
conda env update --name cgdit --file environment.yml
conda activate cgdit
```

该环境固定为 CUDA 12.4 构建，不能混装 `+cpu`、`pt24cpu` 或其他 CUDA
版本的扩展。登录节点看不到 GPU 属于正常现象；CUDA 可用性必须在 SLURM 分配的
GPU 节点内检查。本次验证使用 `rtx4090` 分区并实际分配到 RTX 3090；集群状态会
变化，提交前应先用 `sinfo` 检查可用分区。例如从项目根目录提交检查：

```bash
sbatch --partition=rtx4090 --gres=gpu:1 --time=00:05:00 --wrap='python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(), torch.version.cuda)"'
```

### Configure `.env`

复制模板并填写当前服务器的绝对路径：

```bash
cp .env.template .env
```

```dotenv
PROJECT_ROOT=/absolute/path/to/test_CGDiT
HYDRA_JOBS=/absolute/path/to/test_CGDiT/output
WANDB_DIR=/absolute/path/to/test_CGDiT/wandb
WANDB_API_KEY=
MP_API_KEY=
```

`WANDB_API_KEY` 和 `MP_API_KEY` 只保存在本机 `.env`，不得写入代码、notebook
或提交到 Git。W&B 默认使用离线模式。

本文后续命令均假定已经激活 Conda 环境。例如：

```shell
python -m cgdit.run data=mp_20 model=exp_mp20_base expname=mp20_base
```


## Dataset

### Dataset Layout

每个数据集通常包含：

```text
data/<dataset>/
├── train.csv
├── val.csv
├── test.csv
├── train_sym.pt     # 数据预处理缓存，不是所有数据集都已经生成
├── val_sym.pt
└── test_sym.pt
```

当前数据配置包括：

| 配置名 | 主要性质字段 | 当前状态 |
|---|---|---|
| `mp_20` | `formation_energy_per_atom`, `band_gap`, `e_above_hull` | CSV 与 symmetry cache 已存在 |
| `carbon_24` | `energy_per_atom` | CSV 已存在 |
| `perov_5` | `heat_all`, `heat_ref`, `dir_gap`, `ind_gap` | CSV 已存在 |
| `c2db_51` | `gap`, `ehull`, `hform` | CSV 已存在 |
| `mpts_52` | `formation_energy_per_atom`, `energy_above_hull` | CSV 已存在 |
| `test_data` | MP-20 风格测试字段 | CSV 与 symmetry cache 已存在 |
| `magndata` | `tc`、派生门控标签 `high_tc` | 固定无化学式泄漏划分，支持四组 Tc 基准和高/低 Tc 门控回归 |

CSV 至少需要与对应 `conf/data/*.yaml` 中引用的字段一致。结构列使用 `cif`，材料标识通常使用 `material_id`。

### Add a New Dataset

1. 在 `data/<new_dataset>/` 放置 `train.csv`、`val.csv` 和 `test.csv`；
2. 确保三份 CSV 的字段一致；
3. 复制一个相近的 `conf/data/*.yaml`；
4. 修改 `root_path`、`prop`、`prop_list`、`max_atoms` 和 batch size；
5. 用 `train.pl_trainer.fast_dev_run=true` 先做数据加载测试；
6. 检查生成的 `*_sym.pt` 是否与当前 CSV 对应，CSV 改变后不能盲目复用旧缓存。


## Training

### Basic Training Command

```shell
python cgdit/run.py data=mp_20 model=exp_mp20_base expname=mp20_base
```

- `data`：对应 `conf/data/<name>.yaml`；
- `model`：对应 `conf/model/<name>.yaml`；
- `expname`：写入 Hydra 输出目录，也用于 W&B run/group 名称；
- `train`：对应 `conf/train/<name>.yaml`，默认使用 `default`。

### Debug Before Full Training

```shell
python cgdit/run.py data=test_data model=exp_mp20_base expname=debug train.pl_trainer.fast_dev_run=true logging.wandb.mode=offline
```

该命令用于检查配置、数据、模型前向传播和训练循环，不用于获得正式模型。

### MP-20 Base and Conditional Models

```shell
# 无性质条件
python cgdit/run.py data=mp_20 model=exp_mp20_base expname=mp20_base

# 单条件
python cgdit/run.py data=mp_20 model=exp_mp20_fe expname=mp20_fe
python cgdit/run.py data=mp_20 model=exp_mp20_bg expname=mp20_bg
python cgdit/run.py data=mp_20 model=exp_mp20_eh expname=mp20_eh

# 双条件
python cgdit/run.py data=mp_20 model=exp_mp20_fe_bg expname=mp20_fe_bg
python cgdit/run.py data=mp_20 model=exp_mp20_fe_eh expname=mp20_fe_eh
python cgdit/run.py data=mp_20 model=exp_mp20_bg_eh expname=mp20_bg_eh

# 三条件
python cgdit/run.py data=mp_20 model=exp_mp20_fe_bg_eh expname=mp20_fe_bg_eh
```

服务器批量运行脚本位于：

```text
submit_python/run_all_mp20.sh
submit_python/run_remaining_mp20.sh
```

`run_remaining_mp20.sh` 当前包含服务器绝对路径，迁移服务器后必须先修改；`run_all_mp20.sh` 仍依赖外部提供 `HYDRA_JOBS` 和 `WABDB_DIR`。

### Fine-tuning

```shell
python cgdit/run.py data=mpts_52 model=diffusion_finetuned train=finetuned expname=mpts52_finetuned train.finetune_from_ckpt=<checkpoint_path>
```

必须在命令行覆盖 `train.finetune_from_ckpt`。当前 `conf/train/finetuned.yaml` 中保存的是旧机器上的具体检查点路径，不应直接复用。

### Training Outputs

Hydra 输出路径由 `conf/default.yaml` 决定。训练产物仍保存在本次运行根目录，生成和评估结果使用分层目录：

```text
output/singlerun/YYYY-MM-DD/HH-MM-SS-<expname>/
├── epoch=<epoch>-step=<step>.ckpt
├── hparams.yaml
├── run.log
├── generated_structures/
│   ├── pilot/
│   ├── formal/
│   └── test/
└── evaluations/
    ├── structural_metrics/
    ├── property_predictions/
    ├── property_metrics/
    └── source_manifest.json
```

- `hparams.yaml` 是模型重建与结果追踪的重要依据；
- `.ckpt` 是模型权重；
- `run.log` 是本次运行日志；
- `generated_structures/` 保存结构张量，并继续按模板/ab initio、条件/无条件分类；
- `evaluations/` 保存结构指标、逐结构性质预测、汇总性质指标和来源清单；
- 迁移模型时至少要同时保留 `.ckpt` 和 `hparams.yaml`。

旧版直接保存在模型目录下的 `eval_gen_*.pt` 和 `eval_metrics_gen_*.json`
可通过以下命令迁移。默认只预览，增加 `--apply` 才会执行；原文件会先备份到
`output/_legacy_layout_backup_20260822/`：

```shell
python -m scripts.cli.tools.migrate_output_layout
python -m scripts.cli.tools.migrate_output_layout --apply
python -m scripts.cli.tools.migrate_output_layout --verify
```

训练脚本只会在本次 Hydra 输出目录中寻找检查点。重新启动命令通常会创建新的时间戳目录，因此不能把它理解为会自动恢复任意旧目录中的训练。


## Generation

可复用代码位于 `cgdit/generation`，命令行入口位于 `scripts/cli/generation`。新代码应导入 `cgdit.generation`，而不是导入旧的 `scripts/*.py`。

### 1. Template Generation

根据训练集中的结构模板进行无条件生成：

```shell
python -m scripts.cli.generation.generate --model_path <model_path> --label template_uncond
```

输出：

```text
<model_path>/generated_structures/formal/template/unconditional/eval_gen_template_uncond.pt
```

### 2. Property-guided Generation

```shell
python -m scripts.cli.generation.generate --model_path <model_path> --property_name band_gap --target_value 2.0 --guidance_scale 3.0 --label bg_2p0_cfg3
```

条件名必须存在于该模型目录的 `hparams.yaml -> model -> conditions` 中。

### 3. Ab Initio Generation

完全从头随机产生晶体骨架：

```shell
python -m scripts.cli.generation.generate --model_path <model_path> --ab_initio --label ab_initio_random
```

使用训练集原子数经验分布：

```shell
python -m scripts.cli.generation.generate --model_path <model_path> --ab_initio --use_empirical_prior --label ab_initio_empirical
```

指定空间群并执行性质引导：

```shell
python -m scripts.cli.generation.generate --model_path <model_path> --ab_initio --use_empirical_prior --target_sg 221 --property_name formation_energy_per_atom --target_value -3.5 --guidance_scale 2.0 --label sg221_fe_n3p5
```

### 4. CSP / Reconstruction Generation

该步骤依据测试集的原子类型、对称性和结构条件进行重构采样，并生成 `eval_diff_*.pt`：

```shell
python -m scripts.cli.generation.generate_reconstruction --model_path <model_path> --dataset mp_20 --num_evals 1 --label csp_mp20
```

输出：

```text
<model_path>/eval_diff_csp_mp20.pt
```

历史脚本名是 `evaluate.py`，但它实际执行的是结构生成，不是指标计算。

### 5. Sample from Pre-defined Symmetries

```shell
python -m scripts.cli.generation.generate_symmetry --model_path <model_path> --save_path <save_path> --spacegroup 58 --wyckoff_letters 2a,2d,4g --atom_types Mn,Li,O
```

Wyckoff 字母也可以简写：

```shell
python -m scripts.cli.generation.generate_symmetry --model_path <model_path> --save_path <save_path> --spacegroup 58 --wyckoff_letters adg --atom_types Mn,Li,O
```

批量 JSON 示例：

```json
[
  {
    "spacegroup_number": 58,
    "wyckoff_letters": ["2a", "2d", "4g"],
    "atom_types": ["Mn", "Li", "O"]
  },
  {
    "spacegroup_number": 194,
    "wyckoff_letters": "abff",
    "atom_types": ["Tm", "Tm", "Ni", "As"]
  }
]
```

```shell
python -m scripts.cli.generation.generate_symmetry --model_path <model_path> --save_path <save_path> --json_file <json_file>
```

生成空间群查询模板：

```shell
python -m scripts.cli.tools.generate_symmetry_queries --output-dir <query_dir> --repeats 5
```

## Evaluation

### Evaluation Order

对一个训练完成的模型，推荐按以下顺序执行：

```text
Checkpoint + hparams.yaml
        ↓
Generate eval_gen_*.pt or eval_diff_*.pt
        ↓
General metrics: validity / distribution / coverage / reconstruction
        ↓
Stability: MLIP energy / relaxation / E_hull
        ↓
Novelty / uniqueness / property prediction
        ↓
Export CIF and select candidates for DFT
```

### 1. Generation Metrics

生成时显式指定 `--label`，评估时必须使用相同的 label：

```shell
python -m scripts.cli.generation.generate --model_path <model_path> --label mp20_uncond
python -m scripts.cli.evaluation.evaluate_metrics --root_path <model_path> --tasks gen --label mp20_uncond --gt_file data/mp_20/test.csv --calc_prop false
```

输出：

```text
<model_path>/evaluations/structural_metrics/eval_metrics_gen_mp20_uncond.json
```

生成指标包括：

| 指标 | 含义 | 趋势 |
|---|---|---|
| `comp_valid` | SMACT 组成有效率 | 越高越好 |
| `struct_valid` | 几何结构有效率 | 越高越好 |
| `valid` | 组成和结构同时有效 | 越高越好 |
| `wdist_density` | 密度分布 Wasserstein 距离 | 越低越好 |
| `wdist_num_elems` | 元素种类数分布距离 | 越低越好 |
| `wdist_prop` | 性质分布距离 | 越低越好 |
| `cov_recall` | 对真实结构空间的覆盖 | 越高越好 |
| `cov_precision` | 生成结果位于真实分布附近的比例 | 越高越好 |
| `amsd_recall/precision` | 结构指纹平均最小距离 | 越低越好 |
| `amcd_recall/precision` | 组成指纹平均最小距离 | 越低越好 |

当前 `GenEval` 默认要求至少 1000 个有效生成结构。生成数量不足时会报 `not enough valid crystals`，这时应增加采样数量，而不是直接把不同规模的指标混在一起比较。

如需性质分布指标，可以使用项目内置性质模型或 MatGL：

```shell
python -m scripts.cli.evaluation.evaluate_metrics --root_path <model_path> --tasks gen --label mp20_uncond --gt_file data/mp_20/test.csv --calc_prop true --prop_model_path matgl
```

### 2. CSP / Reconstruction Metrics

```shell
python -m scripts.cli.generation.generate_reconstruction --model_path <model_path> --dataset mp_20 --label csp_mp20
python -m scripts.cli.evaluation.evaluate_metrics --root_path <model_path> --tasks csp --label csp_mp20 --gt_file data/mp_20/test.csv
```

重构指标包括：

- `match_rate`：预测结构与真实结构的 StructureMatcher 匹配率；
- `rms_dist`：匹配结构的平均 RMS 位移。

### 3. Stability, Novelty and Uniqueness

```shell
python -m scripts.cli.evaluation.evaluate_stability --input <model_path>/generated_structures/formal/template/unconditional/eval_gen_mp20_uncond.pt --train-csv data/mp_20/train.csv --output-dir <model_path>/evaluations/stability --samples-per-family 3
```

可选弛豫：

```shell
python -m scripts.cli.evaluation.evaluate_stability --input <eval_gen.pt> --train-csv data/mp_20/train.csv --relax
```

该流程包括：

- 生成张量到 Pymatgen Structure 的转换；
- 宏观晶格、体积、密度等分布；
- RDF Wasserstein 距离；
- MatGL MLIP 能量与可选结构弛豫；
- Materials Project phase diagram 与 `E_hull`；
- 稳定性漏斗图；
- StructureMatcher 唯一性和相对训练集新颖性。

边界说明：

- 稳定性部分默认只抽取最多 3 个二元和 3 个三元结构；
- `--samples-per-family` 控制二元、三元结构各自的最大样本数；
- 其他元素数目的结构当前不会进入昂贵的相图稳定性阶段；
- Materials Project 数据需要 `.env` 中的 `MP_API_KEY`；
- MatGL 势模型需要已安装或可下载，服务器离线运行前应提前缓存。

### 4. Property Evaluation Model

当前项目性质预测器使用 `M3GNetSurrogate`：

```shell
python -m scripts.cli.evaluation.predict_property --ckpt <property_model.ckpt> --input <structures.csv> --output <predictions.csv> --prop formation_energy_per_atom
```

如果输入 CSV 没有真实性质标签：

```shell
python -m scripts.cli.evaluation.predict_property --ckpt <property_model.ckpt> --input <structures.csv> --output <predictions.csv> --no_gt
```

有真实标签时输出 MAE 和 RMSE。

绘制 parity plot：

```shell
python -m scripts.cli.visualization.plot_property_results --run-dir <property_model_run_dir>
```

该目录应包含：

```text
test_preds.npy
test_targets.npy
```

### 5. Post-processing

`post_processing` 中仍保留一批旧分析脚本：

```text
pt_convert_cif.py
struct_analysis.py
calc_sun_rate.py
eval_gen_struct_band_gap.py
predict_band_gap.py
sym_struct_gen_analysis.py
plot_sym_struct_gen_results.py
```

这些脚本中的部分路径仍是历史硬编码路径。运行前必须检查输入文件、训练集路径和输出目录，不能直接假定它们会自动读取最新模型。


## Complete Project Workflow

如果要基于现有项目重复并继续开发，完整顺序是：

1. 激活 `cgdit` 环境并配置 `.env`；
2. 准备 `train.csv`、`val.csv`、`test.csv`；
3. 核对 `conf/data/*.yaml` 的字段和数据路径；
4. 选择或新建 `conf/model/*.yaml`；
5. 先执行 `fast_dev_run`；
6. 正式训练并保存 checkpoint、`hparams.yaml` 和日志；
7. 对训练模型执行模板生成、Ab Initio 生成或 CSP 重构；
8. 使用同一个 label 计算生成或重构指标；
9. 对候选结构进行稳定性、新颖性和性质预测；
10. 将筛选后的结构导出 CIF，进入 DFT 或进一步实验验证；
11. 新开发只修改 `cgdit` 中的公共实现，`scripts/cli` 保持为轻量入口；
12. 新指标必须记录输入结果文件、训练集基准、代码版本和完整参数。


## Python API

新代码推荐直接导入公共模块：

```python
from cgdit.generation.symmetry import generate_structures_from_dataset
from cgdit.evaluation.metrics import Crystal, GenEval, RecEval
from cgdit.evaluation.visualization import plot_property_parity
```

旧的根目录兼容脚本和 `scripts/legacy/` 已清理；Python API 统一从 `cgdit`
导入，命令行统一使用 `python -m scripts.cli.<group>.<command>`。


## Known Limitations

- uv 与 Conda 当前固定为 CUDA 12.4 构建，须通过 SLURM GPU 作业运行 CUDA 任务；
- `main.py` 不是项目入口；
- `conf/train/finetuned.yaml` 含有旧检查点绝对路径；
- `submit_python/run_remaining_mp20.sh` 含有旧服务器绝对路径；
- `post_processing` 中仍有历史硬编码路径；
- 通用生成 CLI 当前一次只显式覆盖一个 `property_name/target_value`；多条件模型训练已经存在，但真正的多目标联合采样接口仍需继续工程化；
- 稳定性评估当前只对抽样的二元和三元结构执行昂贵计算；
- 当前没有“一条命令批量评估所有模型目录”的总调度器，现阶段需要逐模型调用评估入口；
- Diff2Flow 仍是实验实现，不应作为已验证主线结论。


## Citation

本项目建立在 DiffCSP、Pymatgen、PyXtal、PyTorch Geometric、PyTorch Lightning、Hydra、Matminer、SMACT、MatGL 和 Materials Project 生态之上。正式发表或发布代码时，应补充本项目论文信息以及所使用上游工作的标准引用。


## Acknowledgments

感谢开源晶体生成、图神经网络、材料数据库和机器学习势社区提供的基础代码与数据工具。


# Improvement

## 如何改进

### 1. 多目标条件生成

当前通用生成接口一次只接受一个 `property_name/target_value`。下一阶段应设计统一的多条件输入结构，并明确处理：

- 条件缺失；
- 条件量纲和归一化；
- 多条件冲突；
- CFG 中无条件、单条件和多条件分支；
- 不同条件组合之间的公平评估。

### 2. 修改 Backbone

可以将 `CSPNet` 的部分消息传递模块升级为 Graph Transformer，并研究性质条件与晶体节点之间的 Cross-Attention：

```math
\operatorname{Attention}(Q=\text{Node}, K=\text{Condition}, V=\text{Condition})
```

### 3. 升级条件注入机制

1. 当前主线将 Property Embedding 与 Time Embedding 结合，交互方式相对直接；
2. 可以升级为 AdaLN / AdaGN，由条件预测每一层归一化的 scale 和 shift；
3. 可以使用 Cross-Attention，使每个原子节点主动查询条件；
4. 可以加入 Auxiliary Regressor，要求中间表示保留目标性质信息；
5. 多模态实验已经引入 Adapter、对比损失和 Curriculum Learning，应与普通条件模型使用同一数据划分和同一评估协议比较。

### 4. 工程化评估

下一阶段应增加一个模型清单驱动的批量评估器：

```text
models.yaml
    ↓
逐模型生成
    ↓
统一指标计算
    ↓
稳定性与性质预测
    ↓
汇总 CSV / JSON / 图表
```

批量评估器必须保存模型目录、checkpoint、Git commit、生成参数、随机种子、数据集版本和每个指标的失败原因。

### 5. 生成—DFT 闭环

最终目标是耦合高精度性质预测器与多目标生成模型，通过“生成模型初筛—MLIP 弛豫—性质预测—DFT 精算”的闭环锁定优异材料，并进一步分析多种目标性质耦合的微观机制。

### 6. 结合 RAE / Representation Learning

可以研究更强的晶体潜表示学习，再将生成过程建立在稳定、可解释的晶体表示空间中。但在引入新表示之前，应先建立当前直接扩散模型的严格生成速度、有效率、覆盖率和稳定性基线。
