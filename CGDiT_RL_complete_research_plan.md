# CGDiT 强化学习研究、创新设计与实施总览

> 项目：CGDiT 多目标晶体扩散生成
> 整理日期：2026-08-23
> 文档定位：对前期强化学习调研、CGDiT 代码分析、PIRL/PIPO 讨论、创新方案、实验设计和实施计划的统一整理
> 当前主线：从冻结无条件基础策略出发，以固定物理标度的目标性质奖励驱动 PPO/GRPO；`CrystalPIRL` 要求目标性质同时优于当前 verified policy 与永久冻结的原始基础策略，而稳定性、有效性、唯一性、新颖性和多样性只作为相对冻结基础策略的预注册抗坍塌约束，允许在容忍界内小幅下降；`OrbitPO` 提供对称约化概率基础
> 使用边界：本文档保留宽口径研究背景；当前文章范围、实验顺序和停止条件以 `CrystalPIRL文章执行计划.md` 为唯一执行依据

---

## 1. 结论摘要

CGDiT 可以引入强化学习，但以下内容不足以单独构成强创新：

- CGDiT + PPO/DDPO/GRPO；
- FE、BG、Ehull 或普通多目标奖励；
- 生成→性质打分→策略更新的一般闭环；
- KL、entropy、LoRA、replay 或减少去噪步数；
- 只在独立代表晶位上定义概率；
- 原样迁移 PIRL/PIPO 的滑动历史反馈。

推荐文章主线为：

> **固定物理标度的目标性质奖励同时产生 PPO/GRPO 优势与 CrystalPIRL 双锚点验收信号；每次候选更新的目标性质既与当前 verified policy 比较局部改进，也与永久冻结的原始基础策略比较绝对收益。稳定性、有效性、唯一性、新颖性和多样性不要求逐步提高，只要求相对冻结基础策略不越过预注册抗坍塌容忍界；通过后才接受，否则缩放复验、拒绝或回滚。OrbitPO 在晶格—周期坐标—元素的真实独立自由度上提供可重算的联合策略概率。**

核心贡献按重要性排序：

1. **Closed-loop general material generation reward**：以固定目标/容差、显式有效性和可选探索分量构造可跨 batch 比较的原始奖励，严格分离 raw reward、PPO/GRPO advantage 与 probe safety metrics。
2. **Dual-anchor risk-controlled policy-improvement verification**：把“按当前 reward 直接更新”改为候选更新—当前策略局部比较—冻结基础策略绝对比较—多指标置信门控—verified checkpoint 的闭环，避免策略在已经退化的当前状态附近产生“相对改善但仍劣于原始模型”的假改进。
3. **Algorithm-agnostic evidence**：在 PPO 和 GRPO 上均比较无验证、原始 PIPO 与 Paired-PIRL，区分“策略改进反馈有用”与“本文成对机制有额外价值”。
4. **OrbitPO probability foundation**：以轨道级 D3PM、代表点 wrapped likelihood 和空间群有效子空间 Gaussian 形成对称约化混合策略概率。

H2 通道—时间信用分配和 predictor→MLFF→DFT 多保真闭环继续实现并保存内部结果，但不进入当前文章摘要、主结果与结论。

创造性—稳定性—多样性加权奖励和 GRPO 已由 Chemeleon2 实现，因此奖励组件本身不能单独支撑新颖性。本文创新候选是：固定标度通用奖励既作为优化信号，又成为对 **候选策略更新本身** 的低方差、风险控制验收信号。原始 PIPO 用跨轮滑动历史进行回顾性调制；CrystalPIRL 使用同一固定 probe、共同随机数、成对差分、置信门控和回滚。是否构成强创新必须由 2×3 公平实验及奖励闭环消融而不是概念描述证明。

当前已接通训练 CLI、M3GNet reward adapter、PPO/GRPO 更新、checkpoint、固定 probe、共同随机数、bootstrap LCB、双锚点门控和 attenuate 重新验收。Gate 3c 作业 667914 验证了 PPO/GRPO 的 candidate-vs-current、candidate-vs-base 和拒绝路径；其中 GRPO 第二步只通过绝对门、未通过局部门，因而被正确拒绝。当前仍未把稳定性、唯一性、新颖性和多样性接入正式接受条件，也没有 holdout rollback、多 seed 和最终独立 evaluator；因此不能把工程诊断写成完整闭环或材料物性提升。

---

## 2. 研究范围与概念边界

“扩散模型中引入强化学习”至少包含三类不同问题，不能混写。

### 2.1 强化学习微调扩散生成器

这是当前项目的主要方向。反向扩散被视为多步马尔可夫决策过程：

\[
s_t=(x_t,t,c),\qquad
a_t=x_{t-1},\qquad
\pi_\theta(a_t\mid s_t)=p_\theta(x_{t-1}\mid x_t,c).
\]

最终生成结果 \(x_0\) 得到黑盒或不可微奖励 \(R(x_0)\)，再通过 REINFORCE、DDPO、PPO 或 GRPO 更新策略。

### 2.2 可微奖励直接反传

如果性质预测器完全可微，可以通过最终样本或部分扩散轨迹反向传播奖励，例如 DRaFT 和 AlignProp。这类方法应作为强基线，但严格来说不完全等同于 policy-gradient RL。

### 2.3 离线偏好优化

Diffusion-DPO、D3PO、Diffusion-KTO 等方法使用成对偏好或好/坏标签，不一定需要在线 rollout。对于已有 DFT 排序、稳定/不稳定或收敛/不收敛数据，这类方法也值得作为基线。

### 2.4 扩散模型作为机器人或控制策略

Diffuser、Decision Diffuser、Diffusion-QL、DPPO、SRPO、DTQL 等是“用扩散模型表示 RL policy/planner”，不是“用 RL 微调材料扩散生成器”。它们可以提供 trust region、Q-guidance 和快速策略方面的技术启发，但不是当前晶体生成工作的直接前序。

---

## 3. 通用扩散生成强化学习文献脉络

### 3.1 基础在线优化与奖励对齐

| 工作 | 时间 | 核心内容 | 对 CGDiT 的意义 |
|---|---:|---|---|
| [DDPO](https://arxiv.org/abs/2305.13301) | 2023 | 将反向扩散视为多步 MDP，直接使用 policy gradient/PPO 优化最终奖励 | 当前 direct diffusion RL 的基础基线 |
| [DPOK](https://arxiv.org/abs/2305.16381) | 2023 | 在线策略优化并加入预训练模型 KL 约束 | 防止性质优化后偏离晶体先验 |
| [ImageReward / ReFL](https://arxiv.org/abs/2304.05977) | 2023 | 训练奖励模型并通过反馈优化扩散模型 | 对应使用性质预测网络提供奖励 |
| [DRaFT](https://arxiv.org/abs/2309.17400) | 2023 | 通过可微奖励直接反向传播穿过扩散轨迹 | 可微性质预测器的强对照 |
| [AlignProp](https://arxiv.org/abs/2310.03739) | 2023 | 奖励反传、LoRA 和梯度检查点 | 低显存可微优化基线 |
| [Diffusion-DPO](https://arxiv.org/abs/2311.12908) | 2024 | 使用扩散 ELBO 进行离线偏好优化 | 可利用成对晶体偏好数据 |
| [D3PO](https://arxiv.org/abs/2311.13231) | 2024 | 不显式训练奖励模型的扩散轨迹偏好优化 | 适合人工或 DFT 排序反馈 |
| [Diffusion-KTO](https://arxiv.org/abs/2404.04465) | 2024 | 使用单样本好/坏反馈 | 适合稳定/不稳定、收敛/失败标签 |
| [Feedback-Efficient Online Fine-Tuning](https://arxiv.org/abs/2402.16359) | 2024 | 在奖励查询昂贵时提高在线优化效率 | 对 DFT/MLFF 奖励成本有直接意义 |
| [CTRL](https://arxiv.org/abs/2406.12120) | 2024–2025 | 通过离线分类器奖励和 KL 为预训练扩散增加新控制 | 适用于没有重新训练条件模型的目标 |
| [DRAKES](https://arxiv.org/abs/2410.13643) | 2024–2025 | 用 Gumbel-Softmax 对离散扩散轨迹执行奖励优化 | 对 D3PM 元素通道重要 |
| [Reward-guided iterative refinement](https://arxiv.org/abs/2502.14944) | 2025 | 推理时反复加噪—去噪并保留高奖励结果 | 不训练/低训练成本基线 |

系统综述可参考 [Understanding Reinforcement Learning-Based Fine-Tuning of Diffusion Models](https://arxiv.org/abs/2407.13734)。

### 3.2 GRPO、时序信用和效率方向

| 工作 | 时间 | 核心贡献 |
|---|---:|---|
| [Flow-GRPO](https://arxiv.org/abs/2505.05470) | 2025 | 将 flow/ODE 转为可探索 SDE，使用 GRPO 并减少去噪步数 |
| [DanceGRPO](https://arxiv.org/abs/2505.07818) | 2025 | 统一扩散和 rectified flow 的 GRPO 优化 |
| [Advantage Weighted Matching](https://arxiv.org/abs/2509.25050) | 2025 | 将 advantage 加权到 score/flow matching 目标，降低训练成本 |
| [TreeGRPO](https://arxiv.org/abs/2512.08153) | 2025 | 多轨迹共享前缀并分叉，减少重复 rollout 计算 |
| [Rethinking the RL Design Space for Diffusion Models](https://arxiv.org/abs/2602.04663) | 2026 | 强调最终样本似然估计可能比损失名称更关键 |
| [Stepwise-Flow-GRPO](https://arxiv.org/abs/2603.28718) | 2026 | 利用中间步骤的奖励增量解决统一终止信用问题 |

这些工作说明：将同一个最终 advantage 直接广播到全部扩散步只是起点，时间步选择、概率估计和 rollout 成本已经成为主要研究问题。

---

## 4. 晶体、材料和三维分子方向的直接先例

### 4.1 晶体材料方向

| 工作 | 时间 | 方法与结论 | 对创新边界的影响 |
|---|---:|---|---|
| [Band-gap-directed Crystal Generation](https://openreview.net/forum?id=AmNVqwrrNS) | 2024 | 使用带隙奖励引导晶体生成 | 带隙 predictor reward 本身不新颖 |
| [MatInvent](https://openreview.net/forum?id=Ovxfri7l5L) | 2025 | 等变去噪 MDP、policy optimization、reward-weighted KL、replay、diversity filter、单/多目标优化 | 普通多目标晶体 RL 已有直接先例 |
| [RLFEF](https://www.sciencedirect.com/science/article/pii/S0893608025010263) | 2025 | 形成能反馈、材料扩散 MDP、policy gradient、对称性分析 | 形成能奖励和“保持对称性”不能单独作为创新 |
| [Chemeleon2](https://www.nature.com/articles/s42256-026-01262-4) | 2026 | 潜空间扩散 + GRPO；稳定性、新颖性、有效性、多样性多目标奖励 | “GRPO + 多目标奖励”已经正式发表 |
| [Topological Materials ReFT](https://www.nature.com/articles/s41467-026-73321-8) | 2026 | DiffCSP++ 实空间 1000 步 PPO；晶格、坐标、元素联合概率；拓扑分类器奖励；DFT 验证 | 实空间晶体 PPO 已有先例 |
| [OMatG-IRL](https://arxiv.org/abs/2602.00424) | 2026 | 在 velocity-field/flow 模型上进行推理时 policy gradient，学习时间调度并显著降低采样成本 | “flow + RL”也不能单独作为创新 |
| [Synthesizability-Aware Materials Generation](https://openreview.net/pdf?id=o2TIUpiX0t) | 2026 | 将前驱体、毒性、成本、元素复杂度和合成规划加入多目标 RL | 单纯增加合成可行性奖励已有先例 |
| [CRYSTAL](https://openreview.net/pdf/94d95333b625bc19463eca098ff60038d639d590.pdf) | 2026 | 晶体语言模型/RLVR，协调稳定性、新颖性、有效性和多样性 | 是相邻工作，不是直接扩散先例 |

Chemeleon2 对当前项目尤其重要。它认为原子实空间 \((A,X,L)\) 同时混合离散变量和连续变量，会造成策略概率困难、rollout 成本高和梯度不稳定，因此转向统一连续潜空间。CGDiT 的机会不是重复潜空间 GRPO，而是证明：

> 在不牺牲原始晶体表示、空间群约束和可解释性的情况下，可以在对称商空间中正确处理异构动作概率和信用分配。

### 4.2 分子和生物方向

- [RLPF](https://arxiv.org/abs/2508.16521)：使用力场物理反馈优化等变三维分子扩散。
- [DRAKES](https://arxiv.org/abs/2410.13643)：离散 DNA/蛋白质扩散的奖励优化。
- [MolEditRL](https://arxiv.org/abs/2505.20131)：离散图扩散的性质优化与结构保持。
- [FTDiff](https://arxiv.org/abs/2606.01220)：蛋白口袋条件三维分子生成，结合 GRPO、快速采样和多目标奖励。

这些工作表明：物理奖励、离散扩散、阈值奖励和快速采样在相邻科学领域也已经出现。

---

## 5. PIRL/PIPO：从开环奖励优化到闭环策略改进

### 5.1 论文核心

[Policy Improvement Reinforcement Learning](https://arxiv.org/abs/2604.00860) 提出 PIRL，并给出插件式算法 PIPO。普通 PPO/GRPO 使用当前批次奖励或 advantage 更新模型，却不验证更新后的策略是否真的更好；PIPO 在下一轮生成后验证上一轮更新。

策略改进定义为：

\[
\Delta J_t=J_{\mathrm{RL}}(\theta_{t+1})-J_{\mathrm{RL}}(\theta_t).
\]

实际算法用当前批次平均奖励估计即时性能：

\[
\mu_t=\frac{1}{|\mathcal B_t|}\sum_{(q,y)\in\mathcal B_t}R(q,y).
\]

使用最近 \(K\) 轮的滑动历史统计：

\[
\mu_{\mathrm{his}}=\frac{1}{K}\sum_{k=1}^{K}\mu_{t-k},
\]

\[
\sigma_{\mathrm{his}}=
\sqrt{\frac{1}{K-1}\sum_{k=1}^{K}(\mu_{t-k}-\mu_{\mathrm{his}})^2}.
\]

标准化策略改进信号为：

\[
\xi_t=\frac{\mu_t-\mu_{\mathrm{his}}}{\sigma_{\mathrm{his}}+\epsilon}.
\]

上一轮局部 attribution \(a_{t-1,i}\) 被调制为：

\[
\hat r^{\mathrm{PI}}_{t,i}=a_{t-1,i}\phi_\lambda(\xi_t),
\]

其中：

\[
\phi_\lambda(x)=
\begin{cases}
x,&x\ge 0,\\
\lambda x,&x<0.
\end{cases}
\]

PIPO 保留原始 PPO/GRPO 的局部信用，只增加一次对上一轮更新效果的回顾性验证。

### 5.2 为什么不能原样照搬

PIPO 原文实验集中在语言模型数学、代码、工具调用和自蒸馏。它没有验证：

- 扩散长轨迹；
- 连续—离散混合动作；
- 多目标 Pareto 性能；
- 代理性质预测器误差；
- MLFF/DFT 多保真反馈；
- 高成本材料验证。

原文理论还依赖经验改进反馈与真实性能改进方向基本一致。在材料生成中，代理预测器的 OOD 误差、模型偏差和结构松弛效应可能直接破坏这一条件。

因此：

- `CGDiT + scalar PIPO`：是必须保留的原论文迁移基线，创新性中等；
- `CrystalPIRL`：以固定 probe、共同随机数、成对差分、bootstrap LCB、accept/attenuate/reject 和 rollback 形成文章核心方法；
- `OrbitPO`：为 PPO/GRPO 和策略验证提供对称约化的联合概率基础；
- Pareto、多保真和通道—时间归因作为扩展或内部研究，不能稀释主因果问题。

---

## 6. 当前 CGDiT 与强化学习的代码映射

### 6.1 已有生成结构

主要采样入口：

- `cgdit/pl_modules/diffusion.py:218`：`sample()`；
- `conf/model/exp_mp20_*.yaml`：模型配置，默认 1000 个扩散步；
- `cgdit/generation/conditioning.py`：多目标条件解析；
- `scripts/cli/generation/generate.py`：结构生成 CLI。

一次最终晶体可以表示为：

\[
C=(K,X,A),
\]

其中：

- \(K\)：crystal-family 晶格表示；
- \(X\)：周期分数坐标；
- \(A\)：离散原子类型；
- 条件 \(c\)：FE、BG、Ehull、空间群、原子数等。

### 6.2 三个反向扩散通道

#### 晶格

代码通过 `proj_k_to_spacegroup()` 将晶格变量投影到空间群允许的子空间。

#### 周期坐标

坐标使用 predictor-corrector 更新，经过 `anchor_index`、`ops` 和 `ops_inv` 映射，并通过 `% 1` 保持在周期晶胞中。

#### 原子类型

元素使用 D3PM categorical posterior：

- `cgdit/pl_modules/diffusion.py:447-465`；
- `cgdit/pl_modules/diff_utils/discrete_diff_utils.py:259-384`。

D3PM 已形成轨道级 posterior categorical log-prob 与 RL 概率重放接口；正式 checkpoint 上的 GPU 数值和梯度验证仍待完成。

### 6.3 CFG 与联合条件

`cgdit/pl_modules/diffusion.py:484` 的 `_get_model_output()` 已支持 conditional/unconditional 输出组合，因此现有 CFG、联合条件模型和强化学习必须在相同目标与计算预算下公平比较。

### 6.4 当前实现与真实缺口

当前分支已经实现：

- `sample_rl()` 的可重算轨迹接口；
- action、连续/离散随机量、old-policy 分通道 log-prob 与元数据契约；
- 晶格有效子空间、代表点 torus 坐标和轨道级 D3PM transition log-prob；
- 可微概率重放、PPO/GRPO 目标与统一 trainer；
- 固定 probe 的共同随机数采样；
- paired bootstrap LCB 和 accept/attenuate/reject 决策；
- 有界性质奖励、H2 信用分配以及多保真预算编排接口。

仍然缺少或尚未完成：

- 原始 PIPO 的滑动历史 anchor 与 retrospective modulation 对照；
- 已上传 reward predictor 的正式 RL 接入，以及仍缺失的最终独立 evaluator；
- 真实 checkpoint 上的 GPU rollout/replay/梯度 smoke test；
- 正式 PPO/GRPO 训练入口、配置与 checkpoint/resume 端到端验证；
- MLFF/DFT 真实计算 adapter 与服务器任务配置；
- 多 seed 正式实验、消融和文章证据。

因此，现有单元接口不能表述为“RL 训练已经完成”，Paired-PIRL 代码存在也不能替代原始 PIPO 对照。

---

## 7. 强化学习 MDP 定义

### 7.1 状态

\[
s_t=(K_t,X_t,A_t,c,\mathrm{SG},N,t).
\]

### 7.2 动作

策略动作定义为下一步采样状态，而不是简单把网络预测噪声当成动作：

\[
a_t=(K_{t-1},X_{t-\frac12},X_{t-1},A_{t-1}).
\]

坐标 corrector 和 predictor 应分别记录，避免把两次随机转移错误合并。

### 7.3 终止奖励

\[
R(C_0,c)=R(K_0,X_0,A_0;c).
\]

奖励可来自结构有效性、目标性质、稳定性、新颖性、多样性、合成可行性和不确定性。

### 7.4 策略

\[
\pi_\theta(a_t\mid s_t)
=p_\theta(C_{t-1}\mid C_t,c,\mathrm{SG},N).
\]

可按模型更新结构分解为：

\[
\log \pi_\theta
=\log p_\theta^K
+\log p_\theta^{X,corr}
+\log p_\theta^{X,pred}
+\log p_\theta^A.
\]

---

## 8. OrbitPO 概率基础：对称约化的异构联合策略概率

### 8.1 晶格子空间概率

晶格经过空间群投影后，在原始六维 crystal-family 空间中通常是退化分布。不能直接把投影后的六维向量当作满秩高斯计算概率。

建议为每个晶系或空间群构造允许子空间基：

\[
K=B_{\mathrm{SG}}z,
\]

并在独立坐标 \(z\) 中计算高斯密度：

\[
\log p_\theta^K
=\log \mathcal N(z_{t-1};\mu_\theta^z,\Sigma_t^z).
\]

需要验证：

- 基矩阵维数与各晶系自由度一致；
- 投影与重建数值稳定；
- 不同空间群的 KL 和梯度尺度可比较；
- 采样频率与理论密度一致。

### 8.2 周期坐标的 wrapped Gaussian

由于坐标执行 `% 1`，概率分布位于三维环面：

\[
p(x)=\sum_{n\in\mathbb Z^d}\mathcal N(x+n;\mu,\Sigma).
\]

可比较三种实现：

1. minimum-image 普通高斯近似；
2. 有限周期镜像截断；
3. Fourier/Jacobi-theta 形式或稳定近似。

需要对大噪声和小噪声区间分别验证误差。

### 8.3 对称轨道与独立晶位

当前坐标通过 `anchor_index` 和空间群操作展开完整结构。策略概率必须在独立 anchor/Wyckoff 自由度上计算，不能把对称复制原子重复求和。

否则会产生：

- Wyckoff 高多重度位置被过度加权；
- 原子数越大，joint log-prob 和 KL 越大；
- 不同空间群结构的 PPO ratio 不可比；
- 梯度偏向高多重度或大结构。

### 8.4 D3PM 元素概率

\[
\log p_\theta^A
=\sum_{i\in\text{independent sites}}
\log p_\theta(A_{t-1}^{i}\mid A_t,K_t,X_t,c).
\]

元素概率应在独立晶位上计算；若同一轨道元素被空间群复制，不应再次计入 joint log-prob。

### 8.5 联合概率与自由度归一化的边界

精确 joint log-prob 必须保留为独立变量上的求和：

\[
\log p_{\mathrm{joint}}=
\log p_K+\log p_X+\log p_A.
\]

为了平衡不同通道，可以在 loss 或诊断指标中使用每自由度归一化：

\[
\bar \ell_c=\ell_c/d_c.
\]

但归一化后的量不能被称为原始精确联合概率，也不应未经论证直接代替真实 importance ratio。

---

## 9. 最小 GRPO 与内部 H2：通道/时间步信用分配

### 9.1 Vanilla GRPO

对于相同目标条件 \(c\)，生成 \(G\) 个结构：

\[
C_0^{(i)}\sim\pi_{\theta_{old}},\quad i=1,\ldots,G.
\]

组内 advantage：

\[
A_i=
\frac{R_i-\operatorname{mean}(R_{1:G})}
{\operatorname{std}(R_{1:G})+\epsilon}.
\]

每步 importance ratio：

\[
\rho_{i,t}=
\exp\left[
\log\pi_\theta(a_{i,t}\mid s_{i,t})
-\log\pi_{\theta_{old}}(a_{i,t}\mid s_{i,t})
\right].
\]

GRPO clipped objective：

\[
L_{\mathrm{GRPO}}=-\mathbb E\left[
\min\left(
\rho_{i,t}A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i
\right)
\right].
\]

再加入参考模型 KL 和 entropy：

\[
L=L_{\mathrm{GRPO}}+\beta D_{KL}(\pi_\theta\|\pi_{ref})-\gamma H(\pi_\theta).
\]

### 9.2 问题：统一终止 advantage

将同一个最终 \(A_i\) 广播给全部 1000 个步骤无法回答：

- 早期元素选择是否真正产生了性能提升；
- 晶格变化是否只在特定时间段有效；
- 后期坐标精修是否决定稳定性；
- 某一通道是否在利用 predictor 漏洞。

### 9.3 通道和时间分桶

将 1000 步划分为若干时间区间 \(b\)，并分别估计：

\[
A_{i,b,c},\qquad c\in\{K,X,A\}.
\]

可采用：

- 中间状态 \(\hat C_0(t)\) 的性质预测增量；
- 通道反事实替换；
- learned value/reward-to-go；
- 相同噪声下只替换某一通道策略；
- 近似 Shapley contribution。

### 9.4 通道级 objective

\[
\rho_{i,b,c}=
\exp(\log p^c_\theta-\log p^c_{\theta_{old}}),
\]

\[
L_c=-\mathbb E\left[
\min(\rho_{i,b,c}A_{i,b,c},
\operatorname{clip}(\rho_{i,b,c})A_{i,b,c})
\right].
\]

各通道可以使用不同的：

- clip range；
- KL 权重；
- 学习率；
- timestep sampling 分布；
- 更新频率。

---

## 10. CrystalPIRL：PIPO 启发的成对策略改进验证

本节是当前文章的首要方法创新。它不把普通 reward feedback 重新命名为闭环，而是把每次参数更新视为待验证候选。必须同时维护三种策略角色：

- \(\pi_0\)：永久冻结的原始无条件基础策略，提供绝对性能锚点；
- \(\pi_k\)：第 \(k\) 轮最近一次通过验收的 verified policy，提供局部改进锚点；
- \(\pi'\)：由 PPO/GRPO 产生、尚未被接受的 candidate policy。

候选只有在同条件、同随机流的 probe 上相对 \(\pi_k\) 出现可信局部改善、相对 \(\pi_0\) 保持绝对改善，并满足生成质量非退化约束时，才成为新的 verified checkpoint。原始 PIPO 的滑动历史反馈保留为直接对照。

最终独立 evaluator 不参与 probe、门控或 checkpoint 选择；若某个模型参与这些决策，它应被称为 verifier，并另设独立最终评价器。

### 10.1 固定 probe set

建立与训练 batch 分离的固定验证集合：

\[
\mathcal P=\{(c_j,z_j,\mathrm{SG}_j,N_j)\}_{j=1}^{M}.
\]

其中 \(z_j\) 是固定初始噪声。旧策略和新策略使用完全相同的条件与噪声：

\[
C_j^{old}=G_{\theta_t}(c_j,z_j),
\]

\[
C_j^{new}=G_{\theta_{t+1}}(c_j,z_j).
\]

成对性能改进为：

\[
\Delta m_j=m(C_j^{new})-m(C_j^{old}).
\]

common random numbers 可以显著降低扩散采样随机性对策略比较的影响。

固定 probe 用于候选门控，另设不参与门控的 holdout probe 和最终独立 evaluator 检查多种子泛化。门控阈值、probe seed、样本量和 bootstrap 置信水平必须在观察正式结果前冻结。

### 10.2 多目标性能向量

不使用单一平均 reward，定义：

\[
\mathbf m(\theta)=
\begin{bmatrix}
\mathrm{JointHit}\\
\mathrm{Metastability}\\
\mathrm{Validity}\\
\mathrm{Novelty}\\
\mathrm{Uniqueness}\\
\mathrm{Diversity}\\
-\mathrm{Cost}
\end{bmatrix}.
\]

### 10.3 双重基线与 Pareto-safe 策略接受条件

对任一待验收指标 \(m\)，分别计算局部差和绝对差：

\[
\Delta m_j^{local}=m(C_j^{\pi'})-m(C_j^{\pi_k}),
\]

\[
\Delta m_j^{absolute}=m(C_j^{\pi'})-m(C_j^{\pi_0}).
\]

主性质目标必须同时满足：

\[
\operatorname{LCB}(\Delta R^{local})>0,
\]

\[
\operatorname{LCB}(\Delta R^{absolute})>0.
\]

如果任务采用多目标 Pareto 表述，则将 \(R\) 替换为预注册的 hypervolume 或有效目标产率。与此同时，候选相对冻结基础策略必须满足：

\[
\operatorname{LCB}(\Delta\mathrm{Validity}^{absolute})\ge -\delta_v,
\]

\[
\operatorname{LCB}(\Delta\mathrm{Stability}^{absolute})\ge -\delta_s,
\]

\[
\operatorname{LCB}(\Delta\mathrm{Uniqueness}^{absolute})\ge -\delta_u,
\]

\[
\operatorname{LCB}(\Delta\mathrm{Diversity}^{absolute})\ge -\delta_d.
\]

局部比较防止接受当前轮次的坏更新；绝对比较防止策略虽然相对已退化的 \(\pi_k\) 有所恢复，却仍低于 \(\pi_0\)。安全指标允许预注册的统计容忍量，而不允许随训练轮次累计退化。若候选只满足部分条件，则按预注册尺度进行线搜索并重新完整评估，不能直接根据均值选择最佳尺度。

最终决策为：

- `accept`：局部与绝对主目标 LCB 均通过，全部安全约束满足；
- `attenuate`：候选方向有信号但完整步长未通过，缩小后重新评估，只有重新通过双重门控才接受；
- `reject`：主目标不改善或任一安全指标越界，保留 \(\pi_k\)；
- `rollback`：holdout 或周期性独立审计发现已接受策略不再满足绝对基线约束，恢复最近通过审计的 checkpoint。

### 10.4 材料策略改进反馈

可以定义：

\[
\xi_t^{phys}=
\frac{\operatorname{LCB}(\Delta HV_t)}
{s_{HV}+\epsilon},
\]

再调制上一轮局部 advantage：

\[
\hat r^{PI}_{i,b,c}
=A_{i,b,c}\phi_\lambda(\xi_t^{phys}).
\]

如果某一安全指标明显退化，可采用：

- 负向 retrospective update；
- 减小上一更新方向；
- rollback 到 verified checkpoint；
- 增大对应 constraint multiplier；
- 暂停该通道更新。

### 10.5 通道级策略改进反馈

更强但更昂贵的版本计算：

\[
\xi_{t,c}^{phys},\qquad c\in\{K,X,A\}.
\]

通过在相同噪声下分别替换晶格、坐标或元素策略，估计各通道对性能提升的边际贡献。

---

## 11. 奖励体系

### 11.1 有效性硬门控

无效结构不进入昂贵性质评价：

\[
g_{valid}\in\{0,1\}.
\]

检查包括：

- 晶格正定；
- 无 NaN/Inf；
- 合理晶格尺度；
- 最小原子距离；
- SMACT 成分有效性；
- 空间群一致性；
- 合理原子数和密度；
- 结构构造成功。

### 11.2 目标性质奖励

对目标值 \(y_p^*\)：

\[
R_p=\exp\left[-\frac{(\hat y_p-y_p^*)^2}{2\tau_p^2}\right].
\]

对于只要求越低越好的性质，可以使用带阈值的饱和函数，避免模型无限追求 predictor 外推区间。

当前只启用两项性质：

- `formation_energy_per_atom`；
- `band_gap`。

`e_above_hull` 相关资产保留归档，但不进入当前 reward、probe 或文章主结果。

### 11.3 稳定性奖励

推荐使用 relaxed 结构而不是只依赖 as-generated 预测：

- MLFF relaxation 是否收敛；
- 残余力；
- 松弛结构变化；
- 形成能；
- Ehull；
- 必要时动力学稳定性。

### 11.4 新颖性和唯一性

应同时计算：

- 与训练集和 Materials Project 的 StructureMatcher novelty；
- 生成批内部 uniqueness；
- 连续结构描述符距离；
- 组成新颖性；
- 原型/局域基元新颖性。

真正的晶体原型新颖性仍是开放问题。可以研究对空间群、晶胞选择、原子置换和轻微畸变鲁棒的连续 novelty reward。

### 11.5 多样性

可使用：

- composition embedding 距离；
- structure fingerprint；
- AMD；
- MMD；
- batch marginal utility；
- coverage contribution。

需要警惕通过无意义的多元素复杂化实现 diversity reward hacking。

### 11.6 不确定性感知

预测器 ensemble 给出均值与不确定度：

\[
R_p^{robust}=R_p(\mu_p)-\lambda_u\sigma_p.
\]

可使用：

- 多个独立 M3GNet；
- deep ensemble；
- MC dropout；
- M3GNet 与其他图网络/MLFF 交叉验证；
- conformal calibration。

### 11.7 多目标聚合

不推荐只使用固定线性加权和。可比较：

- linear scalarization；
- geometric mean；
- worst-objective/Chebyshev；
- 乘法门控；
- constrained optimization；
- Pareto rank；
- hypervolume contribution。

推荐将有效性和稳定性作为约束，将功能性质作为目标，将新颖性和多样性作为探索控制。

---

## 12. 内部研究 H3：代理模型—MLFF—DFT 多保真闭环

### 12.1 四级 verifier

| 层级 | 内容 | 频率 | 作用 |
|---|---|---|---|
| Level 0 | 几何、SMACT、结构/空间群有效性 | 每个样本 | 快速硬过滤 |
| Level 1 | 冻结的 seed=42 FE/BG predictor | 每个有效样本 | 当前训练主奖励 |
| Level 2 | MLFF 松弛、能量和力 | 周期性/高价值样本 | 中保真物理筛选 |
| Level 3 | DFT 松弛、能带、真实性质 | 少量高价值样本 | 最终科学证据与校准 |

### 12.2 校准后的奖励

简单形式：

\[
R_{corrected}=R_{proxy}+\alpha(R_{physics}-R_{proxy}).
\]

更完整的方案应避免“只选择高预测奖励样本做 DFT”导致的选择偏差，可以考虑：

- 明确记录选择概率；
- inverse propensity weighting；
- doubly robust estimation；
- 主动学习中的 exploration quota；
- 高奖励与高不确定度混合采样。

### 12.3 闭环更新

DFT/MLFF 新数据可用于：

- 校准 reward model；
- 更新 predictor ensemble；
- 修正 PIPO 策略改进判断；
- 构建好/坏偏好数据；
- 触发离线 DPO/KTO 辅助训练；
- 重新评估历史候选和 Pareto 前沿。

---

## 13. 当前八个生成模型在 RL 研究中的角色

当前服务器保留的 MP20 生成模型为：

1. `mp20_base`；
2. `mp20_fe`；
3. `mp20_bg`；
4. `mp20_eh`；
5. `mp20_fe_bg`；
6. `mp20_fe_eh`；
7. `mp20_bg_eh`；
8. `mp20_fe_bg_eh`。

八组生成模型及 23 组 seed=42 正式结果完整保留，但当前 FE/BG 研究只启用 base、FE、BG、FE+BG 四个生成基线和对应 11 组评估。训练奖励与 probe 只使用 seed=42 FE/BG predictor；seed=123 FE/BG 虽已评估，但多 seed 体系尚未完整冻结，当前不组成 ensemble。Ehull 全部归档。训练/探针预测器不能兼作最终独立 evaluator。

推荐设计：

- `mp20_base` 作为无条件生成先验基线；
- `mp20_fe`、`mp20_bg` 作为单目标条件基线；
- `mp20_fe_bg` 作为双目标条件基线；
- `mp20_eh`、`mp20_fe_eh`、`mp20_bg_eh`、`mp20_fe_bg_eh` 仅作历史归档；
- 当前按 FE→BG→FE+BG 顺序推进，不把归档模型加入当前消融；
- 每个实验冻结对应预训练模型副本作为 `reference policy`；
- 每轮更新前策略作为 `old policy`；
- 同一个 RL policy 在不同目标向量上训练；
- 检验它对 FE、BG、FE+BG 目标和未见目标值的泛化；
- Ehull 当前不作为奖励、约束或主结果指标；如未来恢复，必须重新预注册研究范围。

这可以形成多目标晶体策略的实验贡献，但不能替代对称策略概率和闭环验证这两个方法贡献。

---

## 14. 面向优秀性能材料的任务定义

FE、BG 和 Ehull 是通用筛选指标，还不能单独定义“优秀性能”。正式研究应选择一个明确应用方向。

### 14.1 光伏半导体示例

- 指定直接带隙范围；
- 低 Ehull；
- 负形成能；
- 无毒、低成本、供应链可接受；
- 较低有效质量；
- 较高光吸收；
- DFT 松弛后仍保持目标性质。

### 14.2 宽禁带半导体示例

- 指定宽带隙区间；
- 稳定性；
- 击穿场强代理；
- 介电性质；
- 载流子有效质量；
- 合成可行性。

### 14.3 磁性材料示例

- 磁矩/磁化强度；
- 磁各向异性能；
- Curie 温度代理；
- 低供应链风险；
- 稳定性和可合成性。

### 14.4 热电材料示例

- 合适带隙；
- 低晶格热导率；
- 高功率因子；
- 热稳定；
- 可合成性。

最终最关键的成功率定义应是：

\[
\mathrm{DFTConfirmedJointHitRate}
=\frac{\text{DFT 验证后同时满足全部目标的稳定新颖结构数}}
{\text{进入 DFT 的候选数或总生成数}}.
\]

---

## 15. 当前代码模块映射

保留现有 `sample()` 用于普通生成，RL 逻辑集中在独立模块中：

```text
cgdit/
├── pl_modules/
│   └── diffusion.py                 # 轨道级扩散与 sample_rl()
└── rl/
    ├── trajectory.py                # 轨迹、动作、随机量和元数据契约
    ├── transition_logprob.py        # 晶格、torus、D3PM 转移概率
    ├── symmetry_quotient.py         # active mask、轨道代表点和广播
    ├── rollout.py                   # 轨迹概率重放
    ├── objectives.py                # PPO/GRPO 目标
    ├── trainer.py                   # 统一策略目标入口
    ├── policy_improvement.py        # Paired-PIRL LCB 与更新决策
    ├── paired_probe.py              # 固定条件和共同随机数配对采样
    ├── rewards.py                   # 通用有界性质奖励
    ├── channel_time_credit.py       # 内部 H2 信用分配
    └── multifidelity.py             # 内部多保真预算编排
scripts/cli/training/
└── legacy_rl_gpu_smoke.py           # 旧 smoke 脚本，不是正式训练入口
```

需要新增但尚不存在的最小文件是原始 PIPO 基线实现及正式训练配置/入口。不要把 `policy_improvement.py` 误写为已经实现原始 PIPO。

### 15.1 `sample_rl()` 建议返回

```text
final_crystals
trajectory[t]
├── state_t
├── state_t_minus_half
├── state_t_minus_1
├── sampled_noise
├── lattice_action
├── coord_corrector_action
├── coord_predictor_action
├── atom_action
├── old_log_prob_by_channel
├── ref_log_prob_by_channel
├── independent_dof_mask
└── condition_metadata
```

---

## 16. 实施顺序和成功标准

### 阶段 0：冻结研究问题

任务：选择一个具体功能材料方向，确定目标性质、稳定性和合成约束。

验证：形成固定任务表、阈值、DFT 方法和计算预算。

### 阶段 1：冻结非 RL 基线

完成：

- 原始八模型；
- CFG；
- best-of-\(N\)；
- rejection sampling；
- reward-weighted regression；
- 固定 seed 和相同计算预算。

验证：所有基线结构数、NFE、reward 查询数和 GPU 时间可追踪。

### 阶段 2：独立奖励与评价系统

冻结 seed=42 FE/BG reward predictor，并分别建立不参与训练和 checkpoint 选择的 FE/BG 独立 evaluator；多 seed predictor ensemble 延后到各成员完整训练和统一评估之后。

验证：测试集误差、校准曲线、OOD 检测和 predictor 间一致性明确。

### 阶段 3：transition log-prob 单元测试

分别实现并测试：

- 晶格子空间高斯；
- torus 坐标概率；
- D3PM categorical log-prob；
- symmetry/anchor mask；
- old=current 时 ratio≈1；
- current=reference 时 KL≈0；
- 梯度有限且可重复。

未通过此阶段，不进入长时间 RL 训练。

### 阶段 4：vanilla GRPO 最小闭环

初始建议：

- 一个目标或一个目标向量；
- group size 4–8 或 8–16；
- 128–512 条 rollout smoke test；
- LoRA 或 decoder 后部层微调；
- 参考模型冻结；
- 支持 checkpoint/resume；
- 记录 reward、KL、entropy、clip fraction 和各通道梯度。

验证：reward 存在可重复趋势，validity 和 uniqueness 不立即坍缩。

### 阶段 5：对称约化 GRPO

加入晶格子空间、周期概率和独立轨道计数。

验证：不同原子数、空间群和 Wyckoff 多重度不产生系统性 KL/梯度偏差，并优于 vanilla GRPO。

### 阶段 6：通道和时间信用分配

加入时间分桶、中间状态评价或反事实通道替换。

验证：梯度方差降低，联合命中率或样本效率提高；消融能解释不同通道的作用。

### 阶段 7：原始 PIPO 基线

分别为 PPO 和 GRPO 按原论文加入滑动历史 anchor 和 retrospective modulation；不得使用 Paired-PIRL 的固定 probe 逻辑污染该基线。

验证：实现与论文公式一致，历史窗口与调制信号可复算，并在相同预算下相对无验证基线报告坏更新率和训练方差。

### 阶段 8：CrystalPIRL paired policy improvement

加入固定 probe、共享连续与离散随机流、candidate-vs-current 局部比较、candidate-vs-frozen-base 绝对比较、多指标 bootstrap LCB、attenuate 后重新评估和 verified-checkpoint rollback。先用 predictor/verifier 完成通用性质实验；Pareto 与物理验证作为扩展。

验证：相对无验证和原始 PIPO，PPO/GRPO 的坏更新率下降，额外成本可接受，冻结策略在最终独立 evaluator 上保持性质与生成质量改善。

### 阶段 9：多保真闭环

加入 predictor ensemble、MLFF 和 DFT 主动验证。

验证：代理 reward 与 DFT 结果相关，选择偏差受控，DFT-confirmed hit rate 提升。

### 阶段 10：正式实验与论文证据

至少三个随机种子，完成主表、消融表、Pareto 图、训练稳定性图、成本图和 DFT 候选案例。

---

## 17. 实验矩阵

### 17.1 文章主矩阵：算法 × 更新验证

所有方法使用相同 OrbitPO 概率、初始 checkpoint、目标、seed、NFE、reward 查询和可训练参数：

| 编号 | RL 算法 | 更新验证 | 作用 |
|---|---|---|---|
| P0 | PPO | 无 | PPO 开放式更新基线 |
| P1 | PPO | 原始 PIPO | 原论文迁移基线 |
| P2 | PPO | Paired-PIRL | CrystalPIRL-PPO |
| G0 | GRPO | 无 | GRPO 开放式更新基线 |
| G1 | GRPO | 原始 PIPO | 原论文迁移基线 |
| G2 | GRPO | Paired-PIRL | CrystalPIRL-GRPO |

### 17.2 概率基础与外部基线

| 编号 | 方法 | 作用 |
|---|---|---|
| B0 | 原始 unconditional/base | 生成先验基线 |
| B1 | 条件模型/CFG | 条件生成基线 |
| B2 | best-of-N/rejection sampling | 不更新模型的预算基线 |
| B3 | ambient/full-cell PPO | OrbitPO 概率消融 |
| B4 | ambient/full-cell GRPO | OrbitPO 概率消融 |
| B5 | differentiable reward/DRaFT | 可选可微奖励强基线 |

### 17.3 内部扩展矩阵

H2 通道—时间信用分配、Pareto gate 与 predictor→MLFF→DFT 多保真实验单独存档，不进入当前文章主表和最终结果总结。

所有比较必须对齐生成结构数、条件目标、随机 seed、NFE、reward/probe 查询、参数更新规模和最大 GPU-hour。Paired-PIRL 的 probe 成本单独报告，并同时给出包含与不包含额外验证成本的预算对齐结果。

---

## 18. 评价指标

### 18.1 当前项目已有指标

`cgdit/evaluation/metrics.py`：

- `comp_valid`；
- `struct_valid`；
- `valid`；
- `wdist_density`；
- `wdist_num_elems`；
- `wdist_prop`；
- `cov_recall`；
- `cov_precision`。

`cgdit/evaluation/stability.py`：

- uniqueness；
- novelty；
- 稳定性/松弛相关分析。

### 18.2 性质控制指标

- 每个性质的 target MAE；
- 容差内 success rate；
- joint target hit rate；
- 最差目标误差；
- Pareto hypervolume；
- 不同目标组合的泛化；
- OOD 目标性能。

### 18.3 物理质量指标

- MLFF relaxation 收敛率；
- 松弛前后 RMSD/晶格变化；
- 最大残余力；
- 松弛后 FE 目标命中率；
- DFT relaxation 收敛率；
- DFT-confirmed target hit；
- 必要时 phonon/dynamic stability。

### 18.4 发现能力指标

- uniqueness；
- novelty；
- SUN/mSUN；
- prototype novelty；
- composition/structure coverage；
- 新化学体系数量；
- 每 1000 个样本的有效新候选数。

### 18.5 RL 训练指标

- total reward 和每个 reward component；
- policy-reference KL；
- joint/channel KL；
- entropy；
- clip fraction；
- importance-ratio 分布；
- advantage 方差；
- effective sample size；
- 各通道/时间段梯度范数；
- PIPO improvement signal；
- rejected/rectified/rolled-back update 数量；
- 模式坍缩程度。

### 18.6 成本指标

- 每个 update 的 rollout/优化/奖励耗时；
- GPU-hour；
- NFE；
- reward model 查询数；
- MLFF/DFT 调用数；
- 每 GPU-hour 的有效候选；
- 每次 DFT 验证获得的成功结构数。

---

## 19. 消融实验

### 19.1 概率与对称性

- 全部原子计数 vs 独立轨道计数；
- 原始 6D 晶格高斯 vs 子空间高斯；
- minimum-image vs truncated wrapped Gaussian；
- joint ratio vs channel ratio；
- 总和 vs 每自由度诊断归一化。

### 19.2 信用分配

- 统一终止 advantage；
- 时间分桶 advantage；
- channel-wise advantage；
- 中间 \(\hat C_0\) 奖励增量；
- 反事实通道替换。

### 19.3 奖励

- 仅性质奖励；
- + validity gate；
- + stability；
- + novelty/diversity；
- + uncertainty；
- linear vs geometric vs constrained/Pareto。

### 19.4 PIPO

- 无 PIPO；
- scalar historical PIPO；
- 固定 probe 但不同噪声；
- 固定 probe + 相同噪声；
- scalar improvement vs Pareto improvement；
- 无置信区间 vs LCB gate；
- global \(\xi\) vs channel-specific \(\xi_c\)。

### 19.5 多保真

- 单一 predictor；
- predictor ensemble；
- + MLFF；
- + DFT 校准；
- 无选择偏差修正 vs propensity/doubly robust 修正。

---

## 20. 创新判断与论文证据标准

### 20.1 不应作为核心创新的内容

- 把 GRPO/PPO 接到 CGDiT；
- 使用 FE/BG/Ehull 或普通多目标奖励；
- 一般性的生成—打分—更新循环；
- KL、entropy、LoRA、新颖性或多样性奖励；
- 只增加 PIPO 滑动平均；
- 只使用独立代表晶位、wrapped coordinate 或 Diff2Flow 加速。

### 20.2 中等创新或必要系统贡献

- 原始 PIPO 在晶体扩散中的系统迁移；
- OrbitPO 三通道概率的工程完整性；
- 三目标联合策略和实空间 direct RL；
- 合成可行性、供应链约束或内部多保真编排。

这些内容可以支撑完整系统，但不单独承担文章标题级创新。

### 20.3 当前最强创新候选

1. **固定 probe 的共同随机数成对策略比较**：把更新差异从跨批次采样噪声中分离出来；
2. **bootstrap LCB 与 verified-checkpoint 风险门控**：显式产生 accept/attenuate/reject，并量化错误接受、坏更新和回滚；
3. **跨 PPO/GRPO 的算法普适性检验**：用 2×3 设计证明成对验证是否超出原始 PIPO，而非把某一算法曲线变好视为充分证据；
4. **OrbitPO 对称约化联合概率基础**：确保策略比及验证建立在晶体真实独立自由度上，避免规模/多重度偏置成为混杂因素。

Pareto、安全约束、多保真反馈和通道—时间信用可增强方法，但在当前稿件中只作为内部扩展，不写入主结果总结。

### 20.4 论文主张必须满足

- 直接比较无验证、原始 PIPO 与 Paired-PIRL；
- PPO 与 GRPO 均至少三个随机 seed；
- probe、阈值、accept/attenuate/reject 与 rollback 规则预先冻结；
- 最终独立 evaluator 不参与训练、门控或 checkpoint 选择；
- 同时报告坏更新率、错误接受率、性质、生成质量和额外成本；
- 有 log-prob/ratio/KL 单元测试及 ambient/full-cell 消融；
- 保存所有失败 run，不根据结果更改指标或 probe；
- 只有证据支持时才使用“算法普适”或更强的新颖性表述。

---

## 21. 主要风险与应对

| 风险 | 表现 | 应对 |
|---|---|---|
| Reward hacking | predictor reward 提高但 DFT 变差 | ensemble、不确定性、MLFF/DFT 校准、独立 evaluator |
| Mode collapse | uniqueness/diversity 下降 | KL、entropy、batch diversity、Pareto gate、rollback |
| 空间群概率错误 | 不同 SG 的 KL/梯度系统偏差 | 子空间基、独立轨道、单元测试和解析分布验证 |
| 长轨迹方差 | 1000 步同一 advantage | 时间分桶、通道 credit、transition subsampling |
| PIPO 假改进 | 新批次随机性造成 \(\xi\) 错号 | 固定 probe、相同噪声、paired CI、LCB |
| 标量奖励掩盖退化 | BG 提升但 validity/unique 下降 | Pareto-safe constraints |
| DFT 选择偏差 | 只验证高代理奖励样本 | exploration quota、记录选择概率、偏差修正 |
| 算力不足 | rollout 主导训练时间 | 流式轨迹、重计算、时间步抽样、最后再评估 Diff2Flow |
| 目标定义不够科学 | FE/BG/Ehull 无法代表功能 | 选择具体材料任务并增加应用相关性质 |

---

## 22. Diff2Flow 的位置和启动门槛

当前不把 Diff2Flow 作为 RL 前置条件。

原因：

- 当前直接 diffusion sampler 已能形成轨迹；
- Diff2Flow 需要解决晶格、torus 坐标和离散元素三种通道的不同转换；
- 转换工作可能延迟真正的策略概率和奖励闭环验证；
- OMatG-IRL、Flow-GRPO 等已经使“flow + RL”成为拥挤方向。

只有满足以下条件时才启动：

1. direct GRPO 已经闭环；
2. 实测 rollout 占 RL wall time 超过约 70%，或当前速度无法满足计划样本量；
3. 从晶格通道 D2F-L 小试点开始；
4. 20–50 NFE 下质量非劣；
5. 端到端效率至少提高约 1.5 倍；
6. 预计长期节省大于转换开发成本。

否则继续直接 diffusion-GRPO。

---

## 23. 当前算力快照与成本提示

服务器历史测试中，RTX 4090 上单结构、单 batch pilot 的生成时间约为：

- 无条件生成：约 16–18 秒；
- 单目标条件生成：约 24–27 秒。

该数值只是当时 checkpoint、batch size=1 和采样配置的快照，不能直接线性外推正式 RL 训练。RL 成本还包括：

- group rollout；
- old/current/ref log-prob 重计算；
- reward predictor；
- MLFF relaxation；
- retrospective PIPO update；
- checkpoint 和评估。

应在最小 GRPO 阶段记录：

\[
T_{total}=T_{rollout}+T_{reward}+T_{update}+T_{verification}+T_{I/O}.
\]

只有 profiling 后才能决定是否值得进行采样器转换。

---

## 24. 最终推荐路线

建议按以下顺序执行：

```text
复现并冻结非 RL 基线、reward predictor、probe verifier 与最终独立 evaluator
→ sample_rl() 与 OrbitPO log-prob 单元测试
→ PPO/GRPO 无验证基线
→ 原始 PIPO 的滑动历史对照
→ CrystalPIRL 固定 probe、共同随机数、LCB 与 rollback
→ PPO/GRPO × 无验证/原始 PIPO/Paired-PIRL 2×3 正式实验
→ FE、BG 与 FE+BG 联合约束
→ 三随机种子、消融、成本和失败案例
→ H2 与 predictor/MLFF/DFT 内部扩展
→ 具体高性能材料任务
```

推荐论文方法主线：

> **CrystalPIRL: Paired Policy-Improvement Verification for Reinforcement Fine-Tuning of Symmetry-Constrained Crystal Diffusion**

推荐中文表述：

> **面向对称约束晶体扩散强化微调的成对策略改进验证**

当前论文的首要评价问题是：

> **在相同生成、性质查询和 GPU 预算下，Paired-PIRL 是否比开放式 PPO/GRPO 和原始 PIPO 更少接受退化更新，并在冻结后的独立评价中获得更稳定的通用性质提升？**

---

## 25. 主要参考入口

### 扩散 RL 与偏好优化

1. [DDPO](https://arxiv.org/abs/2305.13301)
2. [DPOK](https://arxiv.org/abs/2305.16381)
3. [ReFL / ImageReward](https://arxiv.org/abs/2304.05977)
4. [DRaFT](https://arxiv.org/abs/2309.17400)
5. [AlignProp](https://arxiv.org/abs/2310.03739)
6. [Diffusion-DPO](https://arxiv.org/abs/2311.12908)
7. [D3PO](https://arxiv.org/abs/2311.13231)
8. [Diffusion-KTO](https://arxiv.org/abs/2404.04465)
9. [DRAKES](https://arxiv.org/abs/2410.13643)
10. [Flow-GRPO](https://arxiv.org/abs/2505.05470)
11. [DanceGRPO](https://arxiv.org/abs/2505.07818)
12. [Policy Improvement Reinforcement Learning](https://arxiv.org/abs/2604.00860)

### 晶体与材料

13. [MatInvent](https://openreview.net/forum?id=Ovxfri7l5L)
14. [RLFEF](https://www.sciencedirect.com/science/article/pii/S0893608025010263)
15. [Chemeleon2](https://www.nature.com/articles/s42256-026-01262-4)
16. [Design Topological Materials by Reinforcement Fine-tuned Generative Model](https://www.nature.com/articles/s41467-026-73321-8)
17. [OMatG-IRL](https://arxiv.org/abs/2602.00424)
18. [Synthesizability-Aware Materials Generation](https://openreview.net/pdf?id=o2TIUpiX0t)
19. [CRYSTAL](https://openreview.net/pdf/94d95333b625bc19463eca098ff60038d639d590.pdf)

### 项目内补充文档

- `CrystalPIRL文章执行计划.md`：当前文章范围、五阶段实验路线和停止条件；
- `docs/archive/diff2flow_project_relevance_analysis.md`：已归档的 Diff2Flow 技术储备、适配边界与 go/no-go 条件；
- `CGDiT项目.md`：项目结构、模型、生成与评估说明。

---

## 26. 文档状态

本文档是当前统一研究蓝图，不代表所有创新假设已经被实验验证。

截至 2026-08-23 的新增工程证据：

- Gate 2 已通过完整轨迹复现、非零有限梯度、PPO/GRPO 更新和 checkpoint 输出；
- Gate 3b 作业 667909 在固定 seed=42、probe seed=4242、batch=16 下完成尺度扫描，两次跨进程复现哈希完全一致；
- PPO 在 scale=0.5 时得到正 LCB，GRPO 在 scale=0.25 和 1.0 时得到正 LCB，说明两种算法都存在候选改进方向且安全尺度不同；
- 这些结果只支持“更新方向与门控机制可诊断”，不支持多种子稳健性、生成质量非退化、真实材料稳定性或最终物性提升。
- Gate 3c 作业 667914 完成双锚点 GPU 诊断：PPO 两步均被拒绝，GRPO 第一步奖励门控接受、第二步因 local LCB 为负而拒绝；
- PPO 缩放候选重新 rollout 后 LCB 仍略低于零并被拒绝，证明 attenuate 不会绕过完整复验；
- 当前安全审计只有 validity，所有 checkpoint 均保持 `diagnostic_only`，未更新 verified checkpoint。

特别需要区分：

- **已验证事实**：当前采样器结构、三通道类型、D3PM 后验接口、GPU 完整轨迹可复现性、单步 PPO/GRPO 候选更新、双锚点 LCB、attenuate 复验与局部退化拒绝路径；
- **首要研究假设**：双重基线、共同随机数与多指标置信门控能使 CrystalPIRL 相对无验证和原始 PIPO 降低 PPO/GRPO 的坏更新率，并避免相对已退化策略产生假改进；OrbitPO 是排除策略概率偏置所需的基础；
- **成功结论**：只能在实现、单元测试、公平基线、多个随机种子和冻结后的最终独立评价后给出；只有涉及物理验证的主张才额外要求 MLFF/DFT 证据。
