# BasinGuide：3 eV 条件生成的 500 结构诊断实验

实验编号：`bg3_500_seed42_20260914`。本文件冻结阶段 00 的执行范围，计算结果尚待任务完成。

## 已确定的任务

- 使用已有带隙专用条件生成模型 `output/singlerun/2026-06-30/11-28-44-mp20_bg/epoch=784-step=665680.ckpt`。
- 条件输入明确设为 `band_gap=3.0`，条件强度 `guidance_scale=1.0`，1000 步原采样器，`step_lr=1e-5`。
- 生成随机种子 42，20 个一批，共 25 批，500 个未经性质筛选的候选，最多 20 原子。
- 使用训练集原子数经验先验，全新空间群/Wyckoff 骨架；元素由生成器生成。本轮按用户指定的现有模型进行从头条件生成，不采用方案初稿中建议的固定成分任务。
- 使用 seed 42 带隙预测器 `output/singlerun/2026-08-16/15-18-32-mp20_predictor_bg/epoch=274-step=233200.ckpt`；其 hparams 中 `random_seed=42` 已核对。
- 达标区间冻结为 `[2.5, 3.5] eV`。不替换预测不达标的候选，失败保留在 500 个分母中。
- 所有 Python 工作在 `/share/home/xlzou/Anaconda3/envs/cgdit` 中完成，VASP 通过服务器模块加载。

## MLIP 的选择

采用 SevenNet-Omni-i12、`modal=mpa`，联合原子/晶格 FIRE 弛豫，`fmax=0.03 eV/Å`，最多 1000 步，保存 ASE 轨迹及终态。

官方公开基准列出的 Omni-i12 晶体终态 RMSD 为 0.0617 Å，优于同系列较小模型；`mpa` 是官方推荐的通用 PBE(+U) 通道。权重可公开下载，且模型可与 cgdit 的 PyTorch 2.6 一起运行。这里的选择是当前可获得、可运行的高精度候选，不声称在所有化学体系上绝对最准确。UMA-M 权重的直接访问返回 401，未绕过其访问限制。

- [SevenNet 官方模型说明](https://sevennet.readthedocs.io/en/latest/user_guide/pretrained.html)
- [SevenNet-Omni 论文](https://www.nature.com/articles/s41467-026-70195-8)
- [官方 Omni-i12 权重](https://github.com/MDIL-SNU/SevenNet/releases/download/v0.12.1.cp/checkpoint_sevennet_omni_i12.pth)
- 权重 SHA256：`771543388f360d2762e09f62de685bfb85e9f637a06575f2545f03974d6f0522`

## 第一性原理协议

500 个数组元素对应 500 个候选，每个候选在 fat 分区申请 20 MPI 进程、64 GB 内存。每个元素顺序执行以下步骤：

1. 生成态 SCF 单点：得到未弛豫结构的 DFT 带隙，检查预测器是否误判。
2. 从原始生成态直接 DFT 弛豫：原子与晶胞共同优化；最多两个 200 步段。独立于 MLIP，不用 MLIP 终态替换起点。
3. 收敛终态 SCF：取得最终均匀网格带隙及电荷密度。
4. 用终态电荷密度进行高对称路径非自洽能带计算，保存完整能带输出。路径采用 Seekpath 原晶胞坐标接口。
5. 使用 seed 42 预测器评价 DFT 终态，便于与初态预测比较。

采用 VASP 6.3.1、PAW PBE 势库 `/share/home/xlzou/psp/POT_GGA_PAW_PBE`，MPRelaxSet 的元素势版本及成分依赖 Hubbard-U/MAGMOM 规则；记录每种 POTCAR 的 SHA256。`ISPIN=2`、`ISYM=0`、`LASPH=True`、`LREAL=False`、`ISMEAR=0`、`SIGMA=0.05 eV`、`EDIFF=1e-6 eV`、`EDIFFG=-0.02 eV/Å`。截断能为 `max(520 eV, 1.3×最高 ENMAX)`。弛豫 k 点密度为 1000 k 点·原子，初态与终态 SCF 均为 3000 k 点·原子。

这是单一磁性初始化、无 SOC 的 PBE(+U) 诊断，不代表搜索了全部磁性基态，也不将计算带隙当作实验带隙。能带路径与均匀网格带隙分别保存；达标主指标使用终态 SCF 网格带隙。初态 SCF 失败不阻止直接弛豫分支。未收敛结果不会标记成可靠终态。

## 提交顺序和资源约束

生成/初态预测 GPU 作业完成后，MLIP GPU 作业与 fat DFT 数组分别启动。两个 GPU 作业使用依赖关系串行，每次仅申请 `rtx4090` 分区的一张 `gpu:1`，不申请其他计算卡。提交前发现该分区实际 RTX 4090 节点关机，当前可用节点型号为 RTX 3090；遵循用户指定分区和单卡限制，使用分区中的可用 GPU。fat 数组不设人为并发限制，由调度器分配。

DFT 数组与 MLIP 作业均结束后运行轻量汇总，输出逐结构配对表与命中率。自动汇总不等于持续监控；提交后等待用户通知，再开展科学结果分析。

## 文件与状态

运行根目录：`output/basinguide/bg3_500_seed42_20260914/`。

| 文件或目录 | 内容 |
| --- | --- |
| `submission.json` | 实际提交的任务编号及依赖 |
| `environment_freeze.txt` | cgdit 安装完成后的依赖快照 |
| `generation.pt` | 原始 500 个生成输出 |
| `cohort.json` | 全部候选身份、结构导出状态与初态预测 |
| `samples/0000/generated.*` | 第 0 个候选的原始 JSON/CIF/POSCAR |
| `samples/0000/mlip/` | MLIP 轨迹、终态与终态预测 |
| `samples/0000/dft/` | 初态 SCF、直接弛豫、终态 SCF 与路径能带 |
| `paired_results.csv` | 500 个样本的配对结果，保留失败 |
| `summary.json` | 原始预测到 DFT 终态的四类计数、总体命中率 |
| `logs/` | 调度输出与错误日志 |

对应代码为 `scripts/basinguide_diagnostic.py`；提交脚本在 `submit_python/basinguide_*.slurm`。

## 实际提交记录

2026-09-14 00:24（Asia/Shanghai）已提交并核验：

| 作业 | 编号 | 依赖 |
| --- | --- | --- |
| 500 结构条件生成与 seed 42 初态预测 | `669414` | 无；交接时已启动 |
| SevenNet-Omni-i12 弛豫与终态预测 | `669415` | 生成成功后启动，单卡串行 |
| 500 个直接 DFT/SCF/能带工作流 | `669416_[0-499]` | 生成成功后启动 |
| 逐结构配对结果汇总 | `669417` | MLIP 和 DFT 数组结束后启动 |

生成与 MLIP 均申请 `rtx4090` 分区一张 GPU；交接时生成作业实际分配到 RTX 3090。两个 GPU 作业不会同时运行。fat 正式数组已完整提交，不设并发节流。

预检通过：3 项测试、生成权重加载、seed 42 预测器推理、SevenNet 能量/力/应力、分区 GPU 上 CUDA scatter。VASP 预检 `669402` 已正常退出且电子收敛；其 LiF 小样本仅用于验证链路，不计入 500 个正式候选。`preflight_passed.json` 保存模型/代码校验信息及预检结果。

阶段 00 已落实；阶段 01 的正式计算已提交，500 个候选及其 DFT 结果仍需等待计算完成。本轮按要求停止持续监督，收到用户完成通知后再分析。
