# Diff2Flow 对 CGDiT 的意义、可采用范围与实施计划

> 文档状态：历史归档；非当前研究主线
> 创建日期：2026-08-04
> 论文：Schusterbauer et al., *Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment*, CVPR 2025
> 本地来源：`D:\app\Zotero\storage\Y7JZMA8T\Schusterbauer 等 - 2025 - Diff2Flow Training Flow Matching Models via Diffusion Model Alignment.pdf`
> 项目：`E:\WORKSPACE\CodePlace\test_CGDiT`

> **项目状态更新（2026-08-20）**：本文档已归档。Diff2Flow 不再作为当前研究主线或开展 GRPO 的前置条件；只有直接 GRPO 闭环成立、实测 rollout 占总训练时间超过约 70%，且 D2F-L 试点满足总效率和质量门槛时，才重新启用。当前执行顺序和验收条件以 [`CrystalPIRL文章执行计划.md`](../../CrystalPIRL文章执行计划.md) 为准。

## 1. 结论

这篇工作对 CGDiT 有意义，而且可以采用，但不能把论文公式不加修改地同时用于晶格、周期分数坐标和离散原子类型。

最准确的判断是：

- **对已有扩散 checkpoint 的迁移和条件微调：价值高。** Diff2Flow 的核心是保留扩散模型已经学到的 prior，将其输入时间、插值轨迹和输出参数化对齐到 flow matching（FM），避免模型重新学习一遍“扩散输出怎样变成速度”。
- **对从零训练 CGDiT base 模型：价值有限。** 它不是通用的从零训练加速器，前提是已经存在性能可靠的 diffusion checkpoint。
- **对生成/采样加速：价值高。** FM 允许 ODE 积分和 Reflow；但论文的 2–4 步结果包含昂贵的配对数据生成与再次训练，并非免费获得。
- **对后续强化学习：潜在价值很高。** 如果每个 RL rollout 从数百/上千次模型调用降到几十次甚至更少，奖励优化的主要成本会显著下降。但必须先解决离散原子类型和周期坐标，不能只加速晶格通道。
- **对论文创新：有空间。** “把 Diff2Flow 用到晶体”本身不够；真正可能形成贡献的是，在空间群约束下，对“对称约化晶格 × 周期环面坐标 × 离散原子类型”这一乘积空间构造 diffusion-prior-to-flow alignment。

当前仓库的 `cgdit/pl_modules/diff2flow.py` 只是一个未接入配置的原生 FM 原型，不是论文中的 Diff2Flow 实现。不能据此声称已经采用了该论文方法。

## 2. 术语表

| 规范术语 | 定义 | 本文用法 |
|---|---|---|
| diffusion model（DM） | 通过预定义加噪过程和反向去噪学习数据分布 | 指当前 CGDiT `Diffusion` checkpoint |
| flow matching（FM） | 回归从 base distribution 到 data distribution 的时间依赖速度场 | 指原生直线路径或流形上的条件 FM |
| Diff2Flow | 对齐 DM 与 FM 的时间、interpolant 和输出目标后，用扩散 prior 初始化 FM | 专指 Schusterbauer et al. 的方法，不等同于普通 FM |
| Reflow | 用已训练 ODE 生成配对的 noise–data 样本，再训练更直的路径 | 主要用于少步推理，不是免费训练加速 |
| NFE | number of function evaluations | 一次生成中调用速度/去噪网络的次数 |
| D3PM | 离散状态扩散模型 | 当前 CGDiT 的原子类型通道 |
| torus | 周期分数坐标所在的平坦环面 | 坐标的 `mod 1` 几何，不是普通欧氏空间 |

## 3. 论文到底做了什么

### 3.1 问题定义

论文不是比较“从零训练 diffusion 和 FM 谁快”，而是回答：

> 已有一个预训练 diffusion foundation model 时，怎样用最少额外训练把其知识转移给 FM，而不让网络先花大量更新重新学习新的时间尺度、输入插值和输出参数化？

原文依据：PDF p.1–2，摘要与引言；官方版本为 [CVPR 2025 论文](https://www.openaccess.thecvf.com/content/CVPR2025/papers/Schusterbauer_Diff2Flow_Training_Flow_Matching_Models_via_Diffusion_Model_Alignment_CVPR_2025_paper.pdf)。

### 3.2 三项核心对齐

<a id="S-D2F-01"></a>

**S-D2F-01，p.4，Eq. (10)–(13)：轨迹和时间对齐**

对于 diffusion interpolant

\[
x_{DM}=\alpha_t x_{data}+\sigma_t x_{noise},
\]

论文定义

\[
x_{FM}=\frac{x_{DM}}{\alpha_t+\sigma_t},
\qquad
t_{FM}=\frac{\alpha_t}{\alpha_t+\sigma_t}.
\]

这样 FM 的 `t=0` 对应 noise，`t=1` 对应 data，并通过 `f_t^{-1}` 把连续 FM 时间映射回 diffusion 时间。

<a id="S-D2F-02"></a>

**S-D2F-02，p.5，Eq. (14)–(16)：输出目标对齐**

论文以 `v`-parameterized diffusion 为例，从 diffusion 输出恢复 data/noise 估计，再组合成 FM velocity。这样网络不必靠有限的 LoRA 参数自己学会一次大的参数化切换。

<a id="S-D2F-03"></a>

**S-D2F-03，p.5，Algorithms 1–2：训练与采样**

训练时先在 FM 轨迹上采样，再反变换到 DM 输入空间，调用预训练 diffusion 网络，并把输出解析成 FM velocity，最后优化标准 FM loss。采样时用同一映射和 Euler ODE 更新。

### 3.3 论文证据的正确解读

<a id="S-D2F-04"></a>

**S-D2F-04，p.6–8，Figs. 5/8 与 Tables 1–4**

- full fine-tuning 容量足够时，naive FM 最终可能追上，但 Diff2Flow 收敛更快；
- LoRA 容量受限时差距更大，因为 naive FM 难以同时完成任务迁移和参数化迁移；
- 文本到图像实验使用 20k iterations；部分曲线在约 2.5k iterations 已出现明显优势；
- 这些实验是图像 latent、卷积/注意力网络和连续高斯过程，不能直接证明晶体三通道也有相同收益。

<a id="S-D2F-05"></a>

**S-D2F-05，p.12，Reflow 实现成本**

论文的少步 Reflow 先用 40 sampling steps 生成约 180 万个 image–noise pairs，然后训练 60k gradient updates。粗略相当于仅配对生成就包含约 7200 万次网络调用，尚未计入 Reflow 训练。

因此：“可以 2–4 步采样”不等于“降低原始模型训练成本”。它更适合一个将被大量采样、强化学习 rollout 或高通量筛选的成熟模型。

## 4. CGDiT 三个通道的适配性

| 通道 | 当前训练形式 | 对论文假设的符合度 | 是否可直接采用 | 主要问题 |
|---|---|---:|---:|---|
| 对称约化晶格 | VP-style Gaussian，网络预测噪声 | 高 | 可做第一阶段 | 论文示例是 `v` prediction，CGDiT 是 `epsilon` prediction |
| 周期分数坐标 | wrapped-normal/VE score matching，`mod 1` | 中低 | 不能直接照搬 | 欧氏缩放 `x/(α+σ)` 在 torus 上不是全局合法的可逆映射 |
| 原子类型 | D3PM 离散扩散 | 低 | 不能直接照搬 | 论文只处理连续高斯变量，没有离散 flow |
| 条件编码 | CFG 条件 embedding | 高 | 可以保留 | 转换训练中要保持 condition dropout 语义 |
| 空间群约束 | `CrystalFamily` 投影和原子操作约束 | 中高 | 可以保留 | 需证明中间路径、速度场和投影在约束空间中闭合 |

### 4.1 晶格通道：最适合先做

当前 `cgdit/pl_modules/diffusion.py:137-156` 使用

\[
x_{DM}=\alpha_t x_{data}+\sigma_t\epsilon
\]

并让 decoder 预测 `epsilon`。虽然论文 Eq. (16) 展示的是 `v` parameterization，但可以为 epsilon prediction 推导：

\[
\hat x_{data}=\frac{x_{DM}-\sigma_t\hat\epsilon}{\alpha_t},
\]

\[
\hat v_{FM}=\hat x_{data}-\hat\epsilon
=\frac{\alpha_t+\sigma_t}{\alpha_t}
(x_{FM}-\hat\epsilon).
\]

这给出了无需改变 decoder 输出头的 FM velocity。风险是 `α→0` 端点可能数值不稳定，需要端点截断、稳定重写或改为 v-prediction transition head。

晶格已经投影到空间群允许的 `CrystalFamily` 子空间。若投影算子是线性的，数据、噪声、interpolant 和 velocity 都可继续投影，适合做最小可行性实验。

### 4.2 坐标通道：需要 torus-specific alignment

当前坐标是

\[
x_t=(x_{data}+\sigma_t\epsilon)\bmod 1
\]

并学习 wrapped-normal score。Diff2Flow 的欧氏缩放

\[
x_{FM}=x_{DM}/(\alpha+\sigma)
\]

与 `mod 1` 不可交换，而且在环面上通常不是全局可逆坐标变换。因此直接把 Eq. (10) 用到分数坐标会破坏周期等价关系。

合理路线有两条：

1. **流形概率流路线**：从 wrapped score 构造 torus 上的 probability-flow ODE velocity，再以 geodesic/最短周期位移训练 alignment；
2. **局部提升路线**：在 anchor 原子的切空间中选择一致的周期 lift，做局部 diffusion-to-flow 转换，然后通过群操作恢复等价原子。

[FlowMM](https://arxiv.org/abs/2406.04713)已经表明晶体 fractional coordinates 应视为 flat tori，并在该流形上构造 geodesic FM。它是这里必须超过或吸收的强基线，而不是可以忽略的相邻工作。

### 4.3 原子类型：必须是 hybrid 或 discrete flow

论文没有给 D3PM 到 discrete flow 的转换。可选路线：

- **低风险 hybrid**：晶格/坐标改为连续 FM，原子类型继续 D3PM，并用同一个连续时间映射同步；
- **完整 discrete flow**：把原子类型改为连续时间 Markov jump process/discrete flow matching；
- **连续松弛**：对类别 simplex 做 flow，再离散化；物理有效性和类别坍缩风险更高。

只保留 D3PM 的 hybrid 版本最容易落地，但少步推理的上限可能由原子类型通道决定。需要实测 D3PM 大步跳转对组成有效率和元素分布的损伤。

## 5. 当前 `diff2flow.py` 的代码审计

### 5.1 它实现了什么

`cgdit/pl_modules/diff2flow.py`：

- 在 `[0,1]` 均匀采 FM 时间；
- 对晶格使用 Gaussian-to-data 直线插值；
- 对坐标使用 torus 最短位移插值；
- 对原子类型保留 D3PM 式加噪和反向更新；
- 用 Euler 积分采样。

这些更接近“从零训练 hybrid crystal FM”的原型。

### 5.2 它没有实现论文 Diff2Flow 的部分

- 没有从已有 `Diffusion` checkpoint 初始化/加载；
- 没有 `t_FM ↔ t_DM` 的 `f_t`/`f_t^{-1}`；
- 没有 `x_FM ↔ x_DM` 的 interpolant rescaling；
- 没有将旧 diffusion output 解析成 FM velocity 的 objective change；
- 没有 LoRA/adapter 冻结策略，optimizer 仍会接收全部参数；
- 原子 loss 被简化成 clean-type cross entropy，不是当前正式 `DiffusionLoss` 的 D3PM VB + auxiliary loss；
- 没有被任何 Hydra model config 的 `_target_` 引用；
- 类本身没有当前 `Diffusion` 中的完整 `training_step`、`validation_step`、`test_step` 与统计链路。

结论：该文件目前应标记为“未接线研究原型”，不适合直接投入正式实验。

### 5.3 当前 checkpoint 限制

本地仓库递归检查未发现 `.ckpt`。Diff2Flow 必须从已经训练好的 diffusion prior 开始，因此实施前需要从远程训练节点取回一个明确版本的 base checkpoint，并同时保存：

- 对应 Git commit；
- Hydra resolved config；
- 数据 split/hash；
- scaler 与 property normalization；
- 基线生成指标和采样超参数。

## 6. 它对当前项目四个目标的意义

| 项目目标 | 意义 | 判断 |
|---|---|---|
| 缩短 base 模型从零训练 | Diff2Flow 依赖已有 prior | 不是首选 |
| 缩短性质条件微调 | 可复用 base，配合 LoRA/adapter 减少收敛负担 | 很适合 |
| 缩短晶体生成 | FM ODE 和后续 Reflow 可降低 NFE | 很适合，但要解决三通道 |
| 加速强化学习 | rollout 成本可能随 NFE 大幅下降 | 战略价值高 |

因此，它应放在训练加速计划中的位置是：

1. 先用 BF16、数据管线、早停、共享条件模型/adapter 解决 base 与条件训练成本；
2. 得到可靠 diffusion checkpoint 后，用 Diff2Flow 做 checkpoint conversion；
3. conversion 通过后，再考虑 Reflow；
4. 最终将少步模型作为 RL rollout policy 或 proposal model。

### 6.1 为什么不建议立即整体改成 FlowMM

这里需要区分两个决策：

- **采用 FlowMM**：按 Riemannian Flow Matching 重新定义晶格、周期坐标和原子类型，并从头训练一个晶体 flow 模型；
- **采用 Diff2Flow 思路**：保留已有 CGDiT decoder、条件模块、空间群约束和 diffusion checkpoint，逐通道转换为 flow/hybrid flow。

对当前项目，整体改成 FlowMM 的主要收益是推理 NFE 更少、周期坐标几何处理更自然，以及可以设计更合理的晶格 base distribution。FlowMM 在 MP-20 CSP 中使用与 DiffCSP 相同的底层网络，报告约 50 个积分步即可达到较高 match rate，而 DiffCSP 通常使用数百到 1000 步。

但整体重写的代价是：已有 checkpoint 难以直接复用；当前空间群投影、三性质条件控制、D3PM 和多模态模块都要重新接入；从头训练时间未必减少；已有 CGDiT 实验与 FlowMM 新模型之间难以做连续、公平的模型演化比较。

因此推荐的项目决策是：

| 当前目的 | 推荐决策 |
|---|---|
| 只想缩短一次 base 训练 | 不转 FlowMM，优先 BF16、早停、时间步加权和共享条件训练 |
| 需要生成数十万候选 | 值得做 flow/hybrid flow，推理节约可以摊平转换成本 |
| 准备进行 RL | 值得做，少步 rollout 可能决定 RL 是否可承受 |
| 想最大限度复用已有 checkpoint | 采用 Diff2Flow 式逐通道转换，不从头训练 FlowMM |
| 想做新的晶体生成架构论文 | 可将 FlowMM 作为强基线，但主方法应是空间群约束的 diffusion-to-hybrid-flow alignment |

简化后的决策阈值是：如果后续只生成几千个结构，转换可能不划算；如果要生成几十万结构、做多轮性质筛选或大量 RL rollout，减少 NFE 的累计收益通常比转换训练成本更重要。具体阈值必须用实测单结构耗时和转换训练 GPU-hours 计算。

## 7. 推荐实施路线

### 阶段 0：建立可转换的 diffusion 基线

- [ ] 取回一个 MP-20 base checkpoint；
- [ ] 固定 1000/200/100/50/20 步下的生成基线；
- [ ] 记录结构/组成有效率、match rate、coverage、SUN、性质分布和每结构 NFE；
- [ ] 用固定 3 个随机种子保存样本清单。

### 阶段 1：只转换晶格通道

- [ ] 保持原 decoder 和 time embedding；
- [ ] 实现单调可逆的 `t_fm_to_t_dm()`；
- [ ] 实现 `x_fm_to_x_dm()`；
- [ ] 将 epsilon prediction 稳定转换为 lattice FM velocity；
- [ ] 坐标和原子类型仍用原 diffusion/D3PM；
- [ ] 先只训练小型 transition head 或 LoRA；
- [ ] 比较 naive FM、Diff2Flow alignment 和冻结 diffusion 三者。

成功标准：晶格分布、体积/密度和有效率不下降，达到相同晶格指标所需 fine-tuning steps 至少降低 1.5×。

### 阶段 2：torus 坐标转换

- [ ] 选定 probability-flow ODE 或切空间 lift；
- [ ] 证明路径对 `x ≡ x+n` 周期等价；
- [ ] 证明 anchor/group operations 下的速度等变；
- [ ] 与原生 Riemannian FM、当前 wrapped diffusion 和 FlowMM 对比；
- [ ] 测试边界附近原子、长短晶格轴和高对称空间群。

成功标准：周期平移不改变输出；不同等价原胞的生成分布一致；坐标有效率和结构 match 不劣于 diffusion。

### 阶段 3：离散原子类型

- [ ] 先实现 continuous-FM + D3PM hybrid；
- [ ] 扫描 atom jump steps：1000/200/100/50/20/10；
- [ ] 评价组成有效率、元素频率、价态合理性和多元素 SUN；
- [ ] 再决定是否实现 discrete flow matching。

成功标准：少步时组成有效率和元素分布没有系统性坍缩，且三通道联合生成质量通过非劣效标准。

### 阶段 4：Reflow 与 RL

- [ ] 只在 conversion 模型已经能用 20–50 NFE 稳定生成后进行；
- [ ] 用模型生成的真实 crystal-noise pairs，而不是重新随机配对；
- [ ] 比较 1-rectification 在 2/4/8/16 NFE 下的性能；
- [ ] 把配对数据生成成本和训练成本计入总账；
- [ ] 在 RL 中报告 reward gain / total NFE / GPU-hours，而不只报告每次采样步数。

Reflow 的盈亏平衡条件可写为：

\[
N_{future}\,(C_{old}-C_{new})
>
C_{pair-generation}+C_{reflow-training}.
\]

只有预计未来生成/RL rollout 数量足够大时，Reflow 才真正节省总算力。

## 8. 最小实验矩阵

| 方法 | 初始化 | 晶格 | 坐标 | 原子 | 目的 |
|---|---|---|---|---|---|
| 当前 diffusion | diffusion ckpt | diffusion | wrapped diffusion | D3PM | 正式基线 |
| 原生 FM 原型 | random | FM | torus FM | D3PM/CE | 判断从零 FM，但不是 Diff2Flow |
| naive FM FT | diffusion ckpt | 直接 FM | 直接 FM | D3PM | 证明参数化错位成本 |
| D2F-L | diffusion ckpt | aligned FM | diffusion | D3PM | 最小可行性 |
| D2F-LC | diffusion ckpt | aligned FM | torus-aligned FM | D3PM | 连续两通道方案 |
| D2F-Hybrid | diffusion ckpt | aligned FM | torus-aligned FM | synchronized D3PM | 推荐完整工程方案 |
| D2F-Discrete | diffusion ckpt | aligned FM | torus-aligned FM | discrete flow | 研究型完整方案 |
| D2F-Reflow | D2F-Hybrid/Discrete | rectified | rectified | matched | 少步与 RL rollout |

公平性要求：

- 相同 decoder 参数量和 checkpoint；
- full FT 与 LoRA/adapter 分开报告；
- 相同 optimizer updates、样本数和生成预算；
- 同时报 wall-clock、GPU-hours、峰值显存和 NFE；
- 至少 3 seeds；
- 生成评价样本数完全相同。

## 9. 创新性判断

### 9.1 已经不新颖的内容

- 从零用 flow matching 生成晶体：已有 [FlowMM](https://arxiv.org/abs/2406.04713)、[CrystalFlow](https://arxiv.org/abs/2412.11693) 等；
- 在 torus 上做晶体坐标 geodesic flow：FlowMM 已系统处理；
- 连续坐标 + 离散类别的混合 flow：相关晶体/分子工作已出现；
- 仅仅把采样从 1000 步改成 Euler 100 步；
- 仅在晶体模型中使用 LoRA。

### 9.2 可能有价值的创新假设

> 在空间群约束的晶体乘积空间中，将已有三通道扩散 prior 对齐为 hybrid manifold flow；对晶格采用解析 diffusion-to-flow velocity，对周期坐标采用 torus probability-flow alignment，对原子类型采用时间同步的 discrete transport，从而以少量 adapter 更新保留生成质量并显著降低 NFE 和 RL rollout 成本。

需要形成的独立贡献：

1. **乘积空间的分通道 alignment**，而不是统一使用欧氏 Eq. (10)；
2. **空间群约束下的路径闭合/等变性证明或严格验证**；
3. **连续 FM 与离散 D3PM/discrete flow 的时间同步机制**；
4. **training conversion、inference 和 RL 三层总成本核算**；
5. **相对 FlowMM、CrystalFlow、原 diffusion、naive FM 的公平强基线**。

基于本轮检索，尚未发现与“预训练晶体 diffusion → 三通道 manifold/discrete flow alignment”完全相同的工作；这是检索推断而非穷尽性查新，正式立项前仍需围绕该准确表述做 Scopus/Web of Science/Crossref 专项查重。

## 10. 最终建议

Diff2Flow 当前不进入研究主线。只有直接 CGDiT-GRPO 已经形成最小闭环，并且实测 rollout 成本触发主计划中的 M5 门槛后，才按以下顺序开展：

1. **先完成直接 diffusion-GRPO 和成本 profiling。** 不以转换工作阻塞 RL 方法验证；
2. **不要直接运行当前 `diff2flow.py`。** 先把它标记为原型或另建真正的 alignment module；
3. **只做 D2F-L 晶格通道小实验。** 它用于验证 checkpoint prior 是否带来实际的端到端效率收益；
4. **如果 D2F-L 未达到至少 1.5 倍总效率，或 20–50 NFE 质量明显下降，就停止扩展。** 不进入 torus 和离散通道；
5. **若门槛通过，再处理 torus 坐标。** 原子类型先保留同步 D3PM hybrid；
6. **Reflow 最后做，并先计算未来 rollout 节省是否能覆盖配对生成和训练成本。**

一句话概括：

> 当前先直接在 CGDiT diffusion policy 上完成 GRPO；只有 rollout 成本经实测成为主要瓶颈时，Diff2Flow 才作为“少步生成器 → 低成本 RL rollout”的条件效率模块。

## 11. 来源与证据边界

1. Schusterbauer et al., [Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment](https://www.openaccess.thecvf.com/content/CVPR2025/papers/Schusterbauer_Diff2Flow_Training_Flow_Matching_Models_via_Diffusion_Model_Alignment_CVPR_2025_paper.pdf), CVPR 2025；[官方代码](https://github.com/CompVis/diff2flow)。
2. Miller et al., [FlowMM: Generating Materials with Riemannian Flow Matching](https://arxiv.org/abs/2406.04713), ICML 2024。
3. [CrystalFlow: A Flow-Based Generative Model for Crystalline Materials](https://arxiv.org/abs/2412.11693)。

论文页码依据用户提供的 17 页 PDF。项目结论同时基于当前工作区静态代码；本地没有 checkpoint，因而尚未进行数值或运行时验证。

## 12. 修订记录

| 版本 | 日期 | 修改内容 | 修改人 |
|---|---|---|---|
| v0.1 | 2026-08-04 | 完成论文公式、实验成本、三通道适配、代码现状、创新边界和实施计划分析 | Codex |
