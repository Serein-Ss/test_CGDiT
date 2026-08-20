# OrbitPO 文章执行计划

> 工作题目：**OrbitPO: Measure-Correct Policy Optimization on Crystal Symmetry Quotient Spaces**
> 中文题目：**晶体对称商空间上的测度一致策略优化**
> 项目分支：`newton`
> 文档性质：预注册式执行计划；所有结果位置均为待实验占位，不得提前填写结论
> 当前状态：阶段一由外部服务器执行；阶段二至四的核心代码已落地；正式 GPU 强化学习实验等待基线评估与性质预测器上传

---

## 1. 文章要回答的唯一核心问题

空间群约束晶体扩散的反向过程同时包含：

1. 独立 Wyckoff 轨道上的离散元素类别；
2. 独立轨道代表点上的周期分数坐标；
3. 空间群允许的低维晶格子空间。

如果 PPO/GRPO 仍然在完整晶胞和六维环境空间中计算概率，就会重复计算对称复制原子，并在晶格退化维度上使用不成立的普通高斯密度。本文检验：

> 在真实独立自由度上定义元素、坐标和晶格的联合策略测度，是否能够降低由晶胞大小、Wyckoff 多重度和空间群有效维数造成的策略比偏置，并在相同生成与性质查询预算下更稳定地提升通用材料性质？

文章不以“首次将 PPO/GRPO 用于材料扩散”为主张，也不以“使用多目标奖励”为主张。

---

## 2. 主方法与内部研究模块的边界

### 2.1 正文公开主线

正文围绕以下三项内容组织：

1. **Orbit-consistent D3PM**：元素扩散、损失与采样均只在独立轨道代表点上进行，再广播到完整轨道。
2. **Symmetry-quotient transition measure**：晶格使用空间群有效子空间高斯，坐标使用独立轨道代表点上的 wrapped Gaussian，元素使用轨道级 categorical probability。
3. **Algorithm-agnostic PIRL evaluation**：同一商空间策略分别采用 PPO 和 GRPO，并检验共同随机数配对验收层对两种算法是否都有增益。

### 2.2 只做内部效果检查、不写入文章结果总结的内容

以下模块仍然完整实现、运行和保存结果，但不进入摘要、正文结论、主结果表和文章最终结果总结：

1. **H2：通道—时间信用分配**；
2. **predictor→MLFF→DFT 多保真闭环**。

它们单独形成内部技术报告，用于判断下一篇文章是否值得展开。除非后续明确改变投稿策略，否则不能根据正结果临时加入正文，也不能因负结果删除实验记录。

---

## 3. 可检验的文章贡献

### C1：轨道一致的离散扩散

对每个轨道代表点 $o\in\mathcal O$：

\[
\log p_\theta^A
=\sum_{o\in\mathcal O}
\log p_\theta(A_{o,t-1}\mid s_t).
\]

完整晶胞内的等价原子只接受广播结果，不重复采样、不重复计入损失或策略概率。

验证证据：

- 任意采样步，同一轨道的元素完全一致；
- 改变同一轨道的复制多重度不改变轨道级离散损失与 log-prob；
- PyG 多晶体批处理后不存在跨晶体 anchor 映射；
- 轨道级训练与轨道级推理使用相同测度。

### C2：测度一致的混合策略概率

联合概率定义为：

\[
\log\pi_\theta(a_t\mid s_t)
=\log p_\theta^K+\log p_\theta^X+\log p_\theta^A.
\]

其中：

- $p^K$：只在 `CrystalFamily.masks[spacegroup]` 激活的晶格坐标上计算；
- $p^X$：在独立轨道代表点的三维环面上使用有限镜像 wrapped Gaussian；
- $p^A$：在独立轨道代表点上使用 D3PM 后验 categorical probability。

必须保留原始 joint log-prob 作为 importance ratio。每自由度归一化只允许作为诊断量，不能冒充精确联合概率。

当前 CGDiT 数据只提供轨道展开操作，没有提供完整 site-stabilizer 元数据。因此正文应准确表述为“orbit-representative torus measure”，不能声称已经解析所有特殊 Wyckoff 位置的低维切空间。若以后补充 site-symmetry stabilizer，必须新增切空间秩测试后才能升级该主张。

### C3：PIRL 是否具有算法普适性

PIRL 在本文中定义为算法外层：

1. 固定 probe 条件、空间群、原子数和噪声种子；
2. 新旧策略使用完全相同的连续与离散随机流；
3. 计算成对性质差与安全指标差；
4. 用 bootstrap 下置信界决定接受、衰减或拒绝更新；
5. 同一决策尺度作用于 PPO 外部 advantage 和 GRPO 组内标准化后的 advantage。

它不能改变 PPO/GRPO 的 importance ratio，也不能把共同随机数描述成无偏梯度估计器。

---

## 4. 当前代码状态与文件映射

| 功能 | 文件 | 状态 | 下一验证 |
|---|---|---|---|
| 轨道代表点选择/广播 | `cgdit/rl/symmetry_quotient.py` | 已实现 | 用真实 MP20 batch 检查 anchor 偏移 |
| 轨道级前向 D3PM | `cgdit/pl_modules/diffusion.py` | 已实现 | 重新训练/微调后的验证损失 |
| 轨道级 D3PM 损失 | `cgdit/pl_modules/training_utils/diffusion_loss.py` | 已实现 | 与完整晶胞损失做多重度消融 |
| `sample_rl()` 轨迹 | `cgdit/pl_modules/diffusion.py` | 已实现 | GPU 长轨迹显存和吞吐 |
| 轨迹数据契约 | `cgdit/rl/trajectory.py` | 已实现 | checkpoint/resume 序列化 |
| 商空间 transition log-prob | `cgdit/rl/transition_logprob.py` | 已实现 | 大/小噪声 wrapped sum 收敛 |
| 可微概率重放 | `cgdit/rl/rollout.py` | 已实现 | 真实 decoder 梯度有限性 |
| PPO/GRPO | `cgdit/rl/objectives.py` | 已实现 | 真实 rollout smoke training |
| 统一训练目标入口 | `cgdit/rl/trainer.py` | 已实现 | 接入上传后的 reward evaluator |
| PIRL 配对验收 | `cgdit/rl/policy_improvement.py` | 已实现 | 固定 probe 的真实更新验收 |
| 共同随机数采样 | `cgdit/rl/paired_probe.py` | 已实现 | 新旧真实 checkpoint 配对 |
| H2 信用分配 | `cgdit/rl/channel_time_credit.py` | 已实现 | 中间状态与反事实 reward 生成 |
| 通用性质奖励 | `cgdit/rl/rewards.py` | 已实现 | 用 evaluator 标定 target/tolerance |
| 多保真预算编排 | `cgdit/rl/multifidelity.py` | 已实现 | 接入真实 predictor/MLFF/DFT adapter |

当前单元测试覆盖的是概率与接口正确性，不等价于已完成 GPU 训练、MLFF 松弛或 DFT 计算。

---

## 5. 五阶段执行路线

### 阶段一：冻结外部基线和独立评价器

状态：**等待用户从其他服务器上传。**

### 需要上传的生成基线

至少包含：

- `mp20_base`；
- `mp20_fe`；
- `mp20_bg`；
- `mp20_eh`；
- `mp20_fe_bg_eh`；
- 固定 seed 列表；
- 每个模型相同样本数、NFE、guidance scale 和 GPU 时间；
- 生成结构原始 `.pt` 或 `.csv/.cif`，不能只上传汇总均值。

### 需要上传的性质预测器

每个性质至少需要训练 reward predictor 和独立 evaluator：

- formation energy per atom；
- band gap；
- energy above hull；
- checkpoint；
- train/validation/test split 标识；
- MAE、RMSE、校准曲线数据；
- 训练 seed 与模型配置；
- 若为 ensemble，保存每个成员的独立预测。

### 阶段一验收条件

- 训练 reward 和文章 evaluator 不是同一个 checkpoint；
- 所有模型的样本数和查询预算可追踪；
- 生成结构能够被当前 uv 环境重新加载；
- 不因只保留成功结构而产生筛选偏差；
- 基线结果冻结后再确定 RL 性质阈值。

---

### 阶段二：轨道级离散扩散

代码状态：**核心实现完成。**

### 已完成内容

- 前向扩散只对独立轨道代表元素加噪；
- 加噪结果广播到完整轨道；
- D3PM 训练损失只接收代表点；
- 反向 D3PM 只在代表点计算后验与采样；
- 固定元素与部分去噪路径也执行轨道广播；
- 普通 `sample()` API 保持原二元返回值。

### 上传基线后必须执行

1. 用旧 checkpoint 做兼容性采样，记录修正前后的空间群化学一致率；
2. 对 base 与三性质联合模型进行短程微调；
3. 若旧 checkpoint 在轨道级损失下明显失配，重新训练正式模型；
4. 后续所有 RL 方法只能从同一轨道一致 checkpoint 开始。

### 阶段二通过标准

- 轨道元素一致率为 100%；
- 没有跨 batch 轨道污染；
- 验证集 D3PM 损失稳定；
- validity 不低于冻结基线的预设容忍范围；
- 轨道多重度与每结构离散损失之间无系统相关。

---

### 阶段三：可重算 RL 轨迹

代码状态：**核心实现完成。**

每个 transition 已记录：

- `state_t`；
- corrector 后的 `state_half`；
- `state_next`；
- lattice、coordinate corrector、coordinate predictor、atom action；
- 实际连续噪声；
- old-policy 三通道 log-prob；
- timestep 与 stochastic 标记；
- space group、anchor 和 batch 元数据。

确定性的 $t=1$ 转移记录为零 log-prob，不参与策略梯度。

### 阶段三 GPU 验收

- 相同 `noise_seed` 的完整轨迹逐元素一致；
- old=current 时每个通道重放 log-prob 与记录值一致；
- 1000 步轨迹可以按时间步子采样，避免全部反向传播；
- 记录轨迹不会保留 rollout 计算图；
- 真实模型重放时 decoder 梯度非零、有限、可复现；
- 支持中断后从 old-policy checkpoint 和已保存 rollout 恢复。

---

### 阶段四：商空间概率与 PPO/GRPO

代码状态：**数学模块和统一目标入口完成；等待真实奖励闭环。**

### PPO 路径

- 使用外部 advantage；
- 初期采用终止 reward 减去固定或 EMA baseline；
- 不在第一轮加入价值网络；
- 使用 joint quotient ratio 做 clipping；
- 记录 channel log-prob 但不进行自由度归一化替换。

### GRPO 路径

- 相同目标条件形成 group；
- group size 首轮取 4 或 8；
- advantage 在每组内标准化；
- 与 PPO 使用相同轨迹、相同商空间概率和相同参数更新范围。

### 第一轮 smoke test

- 只使用 formation energy 单目标；
- 128–512 条 rollout；
- 只更新 decoder；
- 每次只抽取部分 timestep 重放；
- 先禁用 H2 与多保真反馈；
- 记录 reward、ratio、KL、clip fraction、梯度范数和轨道一致率。

### 阶段四通过标准

- old=current ratio 在数值误差内等于 1；
- current=reference 时采样 KL 接近 0；
- 无 NaN/Inf；
- importance ratio 不随原子数、空间群或轨道多重度系统漂移；
- PPO 和 GRPO 在至少两个 seed 上出现非偶然 reward 趋势；
- validity 与 uniqueness 没有立即坍缩。

---

### 阶段五：PIRL 对比、通用性质扩展和内部模块

### 5.1 PIRL 的 2×2 核心比较

固定商空间策略和相同预算，只改变 RL 算法与 PIRL：

| 编号 | RL 算法 | PIRL | 目的 |
|---|---|---:|---|
| P1 | PPO | 否 | PPO 基线 |
| P2 | PPO | 是 | 检验 PIRL 对 PPO 的作用 |
| G1 | GRPO | 否 | GRPO 基线 |
| G2 | GRPO | 是 | 检验 GRPO 是否已足够稳定 |

每次候选更新都在固定 probe 上用相同 noise seed 比较新旧策略。PIRL 输出：

- `accept`：LCB 明确为正，更新尺度 1；
- `attenuate`：均值为正但 LCB 不充分，更新尺度默认 0.25；
- `reject`：主指标不改善或安全指标退化，尺度 0，并保留最近 verified checkpoint。

### 5.2 “PIRL 是否算法普适”的判据

不能只看四条曲线谁最高。需要预先定义：

- 坏更新率：更新后 probe 主指标显著下降的比例；
- accepted update efficiency：每次被接受更新带来的 reward 增量；
- 样本效率：每 1000 次 predictor query 的 target hit 增量；
- 稳定性：不同 seed 的最终 reward 方差；
- 成本：PIRL probe 额外 NFE 和 GPU-hour；
- 等效界值：GRPO 与 GRPO+PIRL 的可接受差异范围。

解释规则：

1. P2 和 G2 都显著降低坏更新率且收益超过额外成本：支持 PIRL 算法普适；
2. P2 明显改善而 G2 与 G1 在等效界内：说明 GRPO 已足够稳定，PIRL 主要帮助 PPO；
3. G2 仍明显改善：说明组相对 advantage 不能替代策略更新验收；
4. 平均 reward 提升但 novelty/validity 下降：不得判定 PIRL 有效；
5. 仅一个 seed 改善：只能记录为探索性结果。

### 5.3 通用性质任务

按以下顺序逐步扩展：

1. formation energy 最小化；
2. band gap 目标值；
3. Ehull 最小化；
4. band gap 区间；
5. 低 Ehull + 指定 band gap 联合约束。

奖励全部使用有界函数，invalid 结构硬门控，ensemble uncertainty 作为惩罚项。正式阈值由阶段一独立 evaluator 的测试分布确定，不在看到 RL 结果后修改。

### 5.4 H2 内部实验，不进入文章结果总结

实现路线：

1. 将反向时间划分为 4–10 个固定桶；
2. 对每个桶的中间去噪状态估计 $\hat C_0(t)$ 性质；
3. 计算相邻桶 reward increment；
4. 在共同噪声下替换 lattice/coordinate/atom 单通道，得到反事实贡献；
5. 将终止 advantage 按 `[time bucket, channel]` 的贡献绝对值归一分配；
6. 各通道独立 clipping，但不得称为精确 joint ratio。

内部报告至少包含：

- 终止统一 advantage vs 时间分桶；
- 时间分桶 vs 通道—时间联合；
- 各桶/通道梯度方差；
- FE、BG、Ehull 对三个通道的贡献热图；
- 额外 predictor 查询成本；
- 是否出现 predictor exploitation 集中在某个通道。

### 5.5 多保真内部实验，不进入文章结果总结

执行四级验证：

| 层级 | 内容 | 预算原则 |
|---|---|---|
| L0 | 几何/结构/空间群有效性 | 全部结构 |
| L1 | predictor ensemble | 全部有效结构 |
| L2 | MLFF 单点与松弛 | 高 reward 与高 uncertainty 混合预算 |
| L3 | DFT 松弛/能带/真实性质 | 固定小预算 |

内部效果必须记录：

- 每级进入/退出数量；
- MLFF/DFT 失败与不收敛样本，不能静默删除；
- proxy 与 MLFF/DFT 的相关性和校准误差；
- 松弛前后性质、RMSD、晶格变化和最大残余力；
- 选择概率；若采用随机选择则计算 inverse propensity weight；
- 每次 DFT 调用获得的真实命中数；
- 高 reward exploitation quota 与高 uncertainty exploration quota。

当前代码已经实现预算编排、ensemble 均值/不确定度、校准奖励和外部选择概率权重。真实 MLFF/DFT evaluator 仍需根据服务器任务系统、势函数、赝势、泛函和收敛参数接入；在这些配置确定前不得伪造“已完成 DFT 闭环”。

---

## 6. 正式实验矩阵

### 6.1 主表方法

| 编号 | 方法 | 轨道级元素 | 商空间概率 | PIRL |
|---|---|---:|---:|---:|
| B0 | 原始条件 CGDiT | 否/旧实现 | 否 | 否 |
| B1 | 轨道一致 CGDiT | 是 | 不适用 | 否 |
| B2 | Best-of-N | 是 | 不适用 | 否 |
| B3 | ambient/full-cell PPO | 是 | 否 | 否 |
| B4 | ambient/full-cell GRPO | 是 | 否 | 否 |
| M1 | OrbitPO-PPO | 是 | 是 | 否 |
| M2 | OrbitPO-PPO-PIRL | 是 | 是 | 是 |
| M3 | OrbitPO-GRPO | 是 | 是 | 否 |
| M4 | OrbitPO-GRPO-PIRL | 是 | 是 | 是 |

### 6.2 概率消融

- 完整晶胞元素计数 vs 轨道代表点计数；
- 六维晶格高斯 vs active-subspace Gaussian；
- minimum-image Gaussian vs finite-image wrapped Gaussian；
- 元素独立逐原子采样 vs 轨道广播；
- 原始 joint ratio vs 仅用于诊断的 per-DOF log-prob。

### 6.3 公平性约束

所有比较必须对齐：

- 初始 checkpoint；
- 目标条件；
- 训练与采样 seed；
- 生成结构数量；
- diffusion NFE；
- predictor 查询次数；
- 可训练参数集合；
- optimizer、学习率和梯度裁剪；
- 最大 GPU-hour；
- probe 数量与 PIRL 额外成本单独报告。

正式结果至少三个随机种子，报告均值、标准差和单 seed 散点。不能只保留最优 seed。

---

## 7. 核心评价指标

### 7.1 生成质量

- composition validity；
- structure validity；
- overall validity；
- uniqueness；
- novelty；
- coverage precision/recall；
- density 与元素数分布距离；
- 空间群一致率；
- 轨道元素一致率。

### 7.2 性质提升

- 独立 evaluator 的性质均值与分布；
- target MAE；
- tolerance 内 hit rate；
- joint target hit rate；
- 最差目标偏差；
- OOD 目标条件上的泛化。

### 7.3 策略优化诊断

- joint/channel log-prob；
- importance-ratio 分布；
- approximate KL；
- clip fraction；
- effective sample size；
- reward/advantage 方差；
- 按原子数、空间群和 Wyckoff 多重度分层的 ratio/KL；
- 三通道梯度范数；
- accepted、attenuated、rejected update 数量；
- 每次验证更新后的 paired delta 与 LCB。

### 7.4 成本

- rollout、reward、optimization、probe 分项耗时；
- GPU-hour；
- 每个有效候选的 NFE；
- 每次性质查询和每 GPU-hour 的命中数。

---

## 8. 文章逐部分写作蓝图

### 8.1 Title

建议正文题目：

> OrbitPO: Measure-Correct Policy Optimization on Crystal Symmetry Quotient Spaces

题目必须同时出现“policy optimization”和“crystal symmetry quotient”，避免只写泛化的 reinforcement learning。

### 8.2 Abstract

摘要按五句话组织：

1. **背景**：强化学习能直接优化生成材料性质，但晶体反向扩散具有离散、周期和对称约束混合动作。
2. **缺口**：完整晶胞/环境空间概率会重复对称轨道并在退化晶格维度上定义不一致的策略比。
3. **方法**：提出 OrbitPO，联合轨道级 D3PM、代表点 wrapped likelihood 和空间群子空间 Gaussian。
4. **实验**：在相同预算下用 PPO/GRPO、多种通用性质、空间群/原子数分层和 PIRL 2×2 比较进行验证。
5. **结论**：只填写被三个 seed、独立 evaluator 和统计检验支持的结论。

摘要不写 H2、多保真闭环、高居里温度或迁移率。

### 8.3 Introduction

建议 5 段：

#### 第 1 段：材料逆向设计问题

需要说明结构空间巨大、性质 oracle 昂贵、条件扩散与后训练优化的互补关系。不要提前讨论具体算法细节。

#### 第 2 段：现有扩散 RL 的进展

说明 PPO/GRPO 已用于图像、分子或材料生成，因此简单拼接不是创新。引出晶体实空间动作的特殊性。

#### 第 3 段：关键技术缺口

用一个同一独立晶位被复制 2 次、4 次、8 次的例子解释：完整晶胞概率会人为改变 joint log-prob；再用立方晶格只有一个激活自由度说明六维高斯问题。

#### 第 4 段：本文方法

概述三个策略通道和统一商空间测度，说明 PIRL 是算法外层而不是替代 PPO/GRPO。

#### 第 5 段：贡献列表

只列 C1–C3，每项必须对应一个方法小节、一个关键实验和一个消融。

### 8.4 Related Work

建议分三小节：

1. symmetry-aware crystal diffusion；
2. reinforcement learning for diffusion/material generation；
3. policy-improvement verification and paired evaluation。

每节最后一句明确“已有工作解决了什么、尚未解决什么”。不能用“首个”措辞，除非投稿前完成一次更新后的系统检索。

### 8.5 Problem Formulation

需要定义：

- 晶体状态 $s_t=(K_t,X_t,A_t,c,SG,\mathcal O)$；
- predictor/corrector 子转移；
- 终止材料 reward；
- 独立轨道集合；
- 空间群晶格 active mask；
- PPO/GRPO old、current 和 reference policy；
- 真实联合概率与诊断归一化量的区别。

本节必须给出完整的符号表。

### 8.6 Methods

#### 8.6.1 Orbit-consistent discrete diffusion

写清前向加噪、轨道级损失、反向后验和广播。配一个两轨道晶胞示意图。

#### 8.6.2 Rank-aware lattice transition

说明 active mask、affine constraint 和有效维数。给出子空间 Gaussian 公式，解释为什么不使用奇异六维协方差的普通密度。

#### 8.6.3 Periodic coordinate transition

分别描述 corrector 与 predictor 概率；给出 wrapped Gaussian 有限镜像和数值稳定实现。强调只统计代表点。

#### 8.6.4 Mixed joint policy measure

给出三通道求和、importance ratio、KL 诊断与确定性末步处理。

#### 8.6.5 PPO and GRPO optimization

两种算法使用相同策略测度。PPO 说明 advantage 来源；GRPO 说明 group 构造、组内标准化和 group size。

#### 8.6.6 Paired improvement reliability layer

给出固定 probe、共同随机数、paired delta、bootstrap LCB 和 accept/attenuate/reject 规则。明确 PIRL 对两种算法都在 advantage 估计之后作用。

#### 8.6.7 Computational complexity

报告轨迹存储、概率重放、wrapped image sum 和 probe 的额外时间/显存复杂度。

### 8.7 Experimental Setup

需要完整写明：

- MP20 划分和预处理；
- 五个冻结 checkpoint 的角色；
- reward predictor 与独立 evaluator 的隔离方式；
- FE/BG/Ehull 的单位、目标、阈值和 tolerance；
- optimizer、学习率、batch/group size、更新 epoch、clip、KL、梯度裁剪；
- diffusion 步数、被抽样重放的 timestep；
- PPO/GRPO/PIRL 四组合；
- 所有基线；
- 三个或更多 seed；
- GPU 型号、GPU-hour、NFE 和查询预算；
- 统计检验与多重比较处理。

### 8.8 Results

正文建议按问题而不是按模块组织：

#### RQ1：轨道级离散扩散是否修复化学对称一致性？

需要轨道一致率、validity、损失—多重度关系和采样示例。

#### RQ2：商空间概率是否消除结构规模与空间群偏置？

需要 ratio/KL 对原子数、空间群有效维数、轨道多重度的分层图；需要 ambient 与 quotient 消融。

#### RQ3：OrbitPO 是否在相同预算下提升通用性质？

需要 FE、BG、Ehull 和联合约束的 hit rate、质量指标、成本指标，以及 PPO/GRPO 的公平比较。

#### RQ4：PIRL 对 PPO 和 GRPO 是否都有价值？

需要 2×2 交互图、坏更新率、LCB 验收统计、样本效率和额外成本。必须同时报告 GRPO+PIRL 与 GRPO 的等效性或差异区间。

Results 不展示 H2 和多保真闭环结果。

### 8.9 Discussion

讨论四点：

1. 为什么晶体对称商空间是概率问题而不只是数据增强；
2. OrbitPO 在何种空间群/多重度下收益最大；
3. PIRL 与 GRPO 的关系是互补、冗余还是算法相关；
4. 方法能否迁移到分子、表面、缺陷或其他带约束的生成空间。

不得把 predictor 上的提升直接称为真实材料发现。

### 8.10 Limitations

必须主动说明：

- 当前坐标策略使用每个轨道代表点三维环面，没有完整特殊 Wyckoff stabilizer 切空间；
- predictor 会有 OOD 与 reward hacking 风险；
- 长扩散轨迹重放成本高；
- DFT 未进入正文主证据时，结论限于独立代理评价；
- MP20 不能代表全部材料体系；
- 本文没有直接解决高居里温度和高迁移率设计。

### 8.11 Conclusion

只回答三件事：

1. 是否验证了测度偏置；
2. OrbitPO 是否在公平预算下改善通用性质与稳定性；
3. PIRL 对 PPO/GRPO 的实验结论是什么。

不引入正文未展示的新模块或未来应用结果。

### 8.12 Methods/Reproducibility Appendix

包含：

- 完整伪代码；
- 所有超参数；
- checkpoint 选择规则；
- seed 列表；
- wrapped Gaussian 镜像截断误差；
- D3PM 后验实现；
- 确定性末步处理；
- 单元测试清单；
- 失败 run 和排除标准；
- 硬件与软件环境；
- uv 锁文件哈希。

H2 与多保真效果保留为项目内部报告，不放入投稿附件，除非后续明确授权。

### 8.13 Data and Code Availability

投稿前需要准备：

- 数据集来源、版本和许可；
- 数据 split 文件；
- 生成结构与筛选日志；
- reward/evaluator checkpoint；
- `uv.lock`；
- 训练、恢复、评估命令；
- 随机 seed 与配置；
- 公开代码范围和暂不能公开内容的原因。

---

## 9. 主图和主表规划

### Figure 1：问题与方法总览

- 完整晶胞重复计数示意；
- 轨道代表点 D3PM；
- 晶格 active subspace；
- wrapped coordinate transition；
- PPO/GRPO 后接 PIRL 验收。

### Figure 2：概率正确性

- old=current ratio；
- density 数值/解析对照；
- ratio 对原子数；
- ratio 对 Wyckoff 多重度；
- ratio 对空间群有效维数。

### Figure 3：通用性质优化

- FE、BG、Ehull 的 hit rate；
- validity/uniqueness/novelty；
- reward—查询预算曲线；
- PPO 与 GRPO 对照。

### Figure 4：PIRL 算法普适性

- PPO/PPO+PIRL/GRPO/GRPO+PIRL 四曲线；
- 坏更新率；
- paired delta 与 LCB；
- 收益—额外成本图。

### Table 1：主结果

报告三 seed 的通用性质、质量与成本。禁止只报告最优值。

### Table 2：概率消融

报告轨道计数、晶格子空间、wrapped likelihood 的独立贡献。

### Table 3：PIRL 2×2 比较

报告坏更新率、最终 reward、hit rate、方差、probe 成本和等效性区间。

---

## 10. 结果文件和命名约定

正式运行时统一采用：

```text
output/rl/<date>/<method>/<property>/<seed>/
├── config.yaml
├── environment.txt
├── checkpoints/
├── rollouts/
├── rewards.parquet
├── policy_metrics.parquet
├── probe_metrics.parquet
├── generated_structures.pt
└── run_summary.json
```

内部实验另存：

```text
output/rl_internal/
├── h2_channel_time/
└── multifidelity/
```

文章绘图脚本只能读取冻结后的表格，不得在绘图时重新筛选样本或改变阈值。

---

## 11. 停止条件与决策门

### Gate A：不允许进入 RL

满足任一条件即停止：

- 正式 predictor/evaluator 未上传；
- reward 与 evaluator 使用同一 checkpoint；
- 轨道一致率不是 100%；
- old=current 概率重放失败；
- baseline 样本数或预算无法核对。

### Gate B：不允许进入长程训练

- smoke test 出现 NaN/Inf；
- ratio 与原子数/多重度仍明显相关；
- validity 快速坍缩；
- 真实 decoder 无有效梯度；
- 轨迹显存无法通过时间步抽样控制。

### Gate C：不允许形成论文主张

- 少于三个 seed；
- 只在 reward predictor 上有效、独立 evaluator 无效；
- 改善来自更多查询或更多 GPU 预算；
- 没有 ambient/full-cell 概率消融；
- PIRL 结论没有 2×2 公平比较；
- 负结果或失败 run 被选择性删除。

---

## 12. 上传基线后的立即执行顺序

1. 校验上传文件、checkpoint、数据 split 和结果 schema；
2. 复现阶段一基线的一个固定 seed；
3. 用旧 checkpoint 执行轨道级采样兼容评估；
4. 决定短程微调还是正式重训轨道一致模型；
5. 用真实 decoder 跑 transition replay 梯度 smoke test；
6. 单性质 PPO 128–512 rollout；
7. 单性质 GRPO 128–512 rollout；
8. 建立固定 probe 并运行 PPO/GRPO × PIRL 2×2 小实验；
9. 通过 Gate B 后扩展 FE、BG、Ehull 和联合约束；
10. 独立运行 H2 与多保真内部实验；
11. 冻结三个 seed 结果后生成文章图表；
12. 按第 8 节逐段撰写，所有数字从冻结表格自动引用。

---

## 13. 当前不能提前填写的内容

以下问题必须等待真实实验：

- OrbitPO 是否优于 ambient PPO/GRPO；
- PPO 还是 GRPO 更适合当前 CGDiT；
- PIRL 是否对两种算法都有效；
- GRPO 是否已经足够稳定而无需 PIRL；
- H2 是否降低梯度方差或只是增加 oracle 成本；
- predictor→MLFF→DFT 是否提高真实命中率；
- 轨道级 D3PM 是否需要从头重训；
- 最终可支持的期刊级创新强度。

本计划的作用是提前固定这些问题的检验方式，避免根据结果事后改变指标、阈值或文章主线。
