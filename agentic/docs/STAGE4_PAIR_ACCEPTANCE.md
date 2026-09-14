# 第四阶段：文献参考结构配对验收

日期：2026-09-14。完成离线结构核验、全划分查重、配对导出和既有 CGDiT 数据加载；未训练模型、未调用 API、未进行 DFT。

## 已验证的结果

| 项目 | Mn4N | Mn3AlC |
|---|---|---|
| 原文报告温度 / K | 745 | 272 |
| 核结构原子数 | 5 | 5 |
| 晶格常数 / Å | 3.865 | 3.869 |
| 最短周期原子距离 / Å | 1.9325 | 1.9345 |
| 空间群 | Pm-3m，221 | Pm-3m，221 |
| 原数据结构匹配 | train: 0.274_Mn4N | 无 |
| CrystalNN 有向边数 | 36 | 36 |

对称性在 symprec=0.001、0.01、0.1 Å，angle_tolerance=5° 下相同。参考结构为三维周期、分数坐标、全占位有序核结构，无氧化态和磁矩赋值。两个参考 CIF 均无解析警告。距离与空间群检查不等于热力学稳定性或合成验证。

温度来自上一阶段已视觉核查的 DOI 10.1103/PhysRev.125.1893，Table II，印刷页1894；原文表头为历史用语 Néel temperature。结合 Tables I、V 和第三节的 FiM/FM 磁序归一为 Curie 事件，保留原表头。这里只记录文献比较 745 > 272 K，温差473 K；50 K 是筛选间隔，不是误差界或统计置信度。原文 PDF 的 SHA256 随配对保存。

## 查重与划分

实际解析 train/val/test 共1284条 CIF，失败0条。比较采用 CIF 的实际组成及元素敏感 StructureMatcher：ltol=0.01、stol=0.05（归一化距离容差）、angle_tol=1°、scale=False、primitive_cell=True、attempt_supercell=False。结果是在这些容差及表示条件下的结构匹配，不是全部磁相或超胞等价性的穷尽证明。

Mn4N 匹配训练记录0.274_Mn4N；未发现任一端点与 val/test 的结构重复。两端点共同指定为 `train_extension_pipeline_only`，不能将此配对再当作独立测试集。后续扩展仍需按来源/化学家族分组，当前检查不保证无更宽泛的家族信息泄漏。

159条原数据 CIF 有有限精度坐标取整警告，均逐条记录于 parser_audit.json；本对参考结构及匹配的 Mn4N 记录无此警告。未发现占位修正警告。参考 Mn3AlC 不与 Mn3AlN 合并；存在原文 C/N 身份冲突的0.275_Mn3AlN仅从本轮新配对中排除，原始文件保留。

## 可读取的产物

- `data/curation/stage4_pair/endpoints.csv`：两个参考端点，字段 material_id、pretty_formula、cif、tc；tc 单位 K，仅用于接口验收。
- `data/curation/stage4_pair/pairs.jsonl`：一个有向偏好配对，含端点 ID/行号、来源哈希、事件解释、划分和用途限制。
- `results/stage4/endpoint_graphs.pt`：由项目现有 CrystDataset 新建的图缓存，仅供本地受信管线使用。
- `results/stage4/structure_validation.json`、`split_audit.json`、`parser_audit.json`、`graph_validation.json`、`source_integrity.json`：结构、划分、警告、图与原数据校验。

使用项目原配置的 primitive=False、niggli=False、graph_method=crystalnn、use_space_group=False，实际通过 CrystDataset.__getitem__ 和 PyG Batch.from_data_list，得到2图、10节点。核验元素原子序数、边索引范围、有限坐标、温度标签及配对方向；未对这两个样本重估任何训练标准化统计量。

运行环境 pymatgen 2025.10.7、spglib 2.6.0、torch 2.6.0+cu118、torch-geometric 2.7.0。已有环境的 pyg-lib/torch-sparse 因 GLIBC_2.27 缺失被禁用，本次图预处理和普通批处理仍成功；未验证完整 GPU 训练路径。环境警告保存在 graph_validation.json。

## 使用边界与下一步

`pipeline_ready=true` 表示文献—参考结构—图—配对接口验收完成。`production_training_eligible=false` 保留：目前只有一个由助手核查原文的比较，两个核结构是重建参考晶胞，缺少独立泛化评估。没有训练损失、Tc 精度提升或新材料发现结果。

特别是这一对改变了化学组成；项目已有的固定原子类型采样能力并不能直接视为组成偏好学习接口。下一步应扩展可追溯的独立实验比较，检查现有扩散模型是否对原子类型建模，再据此选择组成条件下的结构学习或独立的组成选择层。不能仅把这个 JSONL 接入固定组成采样器就声称完成 DPO。

原项目8个数据/说明文件在执行前后 SHA256 均不变。全部新增产物在 agentic 下。本轮新增4项针对跨划分阻断、不完整审计、C/N区分及配对语义的测试；连同既有测试共28项通过。

## 复现

在项目根目录运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /share/home/xlzou/Anaconda3/envs/cgdit/bin/python agentic/scripts/stage4_pair_acceptance.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /share/home/xlzou/Anaconda3/envs/cgdit/bin/python -m unittest discover -s agentic/tests -v
```

遵循 karpathy-guidelines 的最小修改与可验证验收要求，以及 pymatgen 技能的周期结构、单位、警告、容差和溯源检查要求。
