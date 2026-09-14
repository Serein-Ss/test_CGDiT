# 第五阶段：跨组成扩散接口验收

2026-09-14。应用 karpathy-guidelines，只在 agentic 内增加验收脚本和记录。未修改项目模型或训练数据。

## 主要结论

标准 `cgdit.pl_modules.diffusion.Diffusion` 已有元素类型 D3PM、元素预测头和损失；`fixed_atom_types` 是可选采样约束。因此不必因为上一阶段的固定组成采样方式，立即另建组成选择模型。跨组成偏好在表示上可行，偏好损失接口尚未实现。

上一阶段普通图加载成功不能代替扩散前向验收：现有 forward 需要 spacegroup、anchor_index、ops、ops_inv。本轮单独用 use_space_group=True、tolerance=0.01 新建两个端点的对称图缓存，保持原配置文件不变。

## 实际运行结果

- 两个结构均为SG221，各3个轨道，共6个独立元素类型变量；批处理 anchor 正确偏移且不跨图。
- 使用真实 Diffusion 和 CSPNet 类，CPU随机初始化小模型：hidden_dim=32、1层、20时间步、seed=20260914。配置见 smoke_config.json。
- forward、总损失 backward 均成功且有限；元素头梯度范数0.146505，晶格头1.060090。
- 坐标损失与梯度均0；直接检查两个结构的 Wyckoff 仿射操作线性部分，全部秩0，当前这些固定特殊位置没有可学习的连续坐标自由度。本例不能验证一般位置坐标训练。
- 无 optimizer step，逐参数检查权重未变。未使用 Tc 条件；温度只留在来源端点数据中。本轮不报告模型学习、偏好优化或材料性能提升。
- 现有28项回归测试通过；新增运行脚本自身核验前向、梯度、轨道、位点对应和参数不变。

已有 pyg-lib/torch-sparse 的 GLIBC 警告仍保存在结果中；本次 CPU 路径成功不代表完整 GPU 环境已验证。

## 消除位点错配

pyxtal 对两个端点采用不同的原子排序。Mn4N 的第0行角点对应 Mn3AlC 的第3行，而不是第0行。已按周期分数位置计算并核验一一映射：`[3,0,1,2,4]`；对齐后元素分别为 `[Mn,Mn,Mn,Mn,N]` 和 `[Al,Mn,Mn,Mn,C]`。

此处无视元素的映射仅用于同一框架的位点/噪声对齐，绝不能用于化学去重。上一阶段元素敏感去重保持不变。该映射只在本次相同原点、相同框架的两个晶胞上验证，不是通用结构对齐器。

## 已明确的后续训练接口

`training_interface_contract.json` 固化以下约束，状态为 specified_not_implemented：

1. 为每个结构返回去噪能量，不能直接使用已把整批平均的现有 loss。
2. 对同一对使用相同任务/家族条件；不能分别把745 K和272 K喂给条件编码器，形成不同条件下的伪比较。
3. 显式保存时间步及加噪输入，策略模型和冻结参考模型复用同一份输入；两端点通过已验证映射耦合噪声。
4. 可评估的候选损失为 `softplus(beta*((E_policy_w-E_reference_w)-(E_policy_l-E_reference_l)))`。其中E是明确权重与归一化的逐结构去噪代理量。它尚未实现、尚未证明是精确对数似然；不能直接冠以已经验证的DPO。
5. 先限于相同框架/轨道拓扑，使用经核验的预训练生成模型作为参考。随机验收模型不能充当科学实验基线。独立来源/家族留出测试必须与训练对分离。

## 独立证据扩展情况

沿已有2016论文参考文献[8]定位到 Fe–Ni 氮化物的1999实验工作，保存在 literature_followup.json。本轮没有新增可信数值配对：JAP DOI访问未成功；另一篇[出版商页面](https://www.scientific.net/MSF.302-303.484)只核实题名、作者和出处，未给出可用Tc数值。两篇可能共享实验样品，不能按两个DOI计作两组独立证据。已有表中Ni4N的125 K来自外推，继续排除为直接实测标签。

## 复现与下一步

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /share/home/xlzou/Anaconda3/envs/cgdit/bin/python agentic/scripts/stage5_diffusion_smoke.py
```

下一步按上述接口实现独立的逐结构去噪评分验收，并核验可用预训练生成检查点；文献侧继续取得Fe–Ni原文及具体样本点，避免在一对样本上训练后宣称泛化。API调用0次，DFT计算0次。
