# CGDiT 扩散模型强化学习调研、技术分析与工作计划

> 文档状态：v0.2，研究主目标已确认
> 创建日期：2026-08-04  
> 主目标确认日期：2026-08-09
> 适用仓库：`E:\WORKSPACE\CodePlace\test_CGDiT`  
> 调研截止日期：2026-08-04  
> 维护方式：在“待确认事项”和各阶段复选框中记录决策与进度

## 1. 文档目的

本文档集中记录以下内容：

1. 扩散模型中引入强化学习、奖励优化和偏好对齐的代表性工作；
2. 晶体、材料和三维分子生成方向的直接相关研究；
3. 当前 CGDiT 仓库与强化学习适配有关的代码现状；
4. 可落地的最小强化学习方案；
5. 值得进一步研究的方法创新方向；
6. 基线、实验矩阵、物理验证和消融计划；
7. 后续需要确认的研究决策。

本文档不是严格意义上的系统综述。检索优先使用 arXiv、OpenReview、CVF 和出版社原始页面，并按照 DOI 或“题名 + 第一作者”去重。由于本轮学术数据库 MCP 未挂载，未执行 Scopus/Web of Science 级别的穷尽检索；当前结果适合作为研究立项和方案设计依据，正式投稿前仍需进行一次针对拟定创新点的专项查重。

---

## 2. 结论摘要

### 2.1 当前判断

CGDiT 可以引入强化学习，而且其模型结构具备较好的研究价值：它直接处理晶格、分数坐标和离散原子类型，并通过空间群操作显式保持晶体对称性。

但是，截至 2026-08-04，以下做法已经不能单独构成有说服力的创新：

- 将扩散反向去噪过程写成 MDP；
- 将 PPO、DDPO 或 GRPO 用于晶体扩散模型；
- 使用形成能、带隙或能量高于凸包作为终端奖励；
- 添加 KL 正则、经验回放或普通多样性奖励；
- 使用 M3GNet、机器学习力场或性质预测器给晶体打分；
- 使用“稳定性 + 新颖性 + 多样性”线性或乘法多目标奖励；
- 单独利用强化学习减少扩散采样步数。

### 2.2 已确认的研究主目标

当前研究主目标正式锁定为：

> **直接在 CGDiT 扩散采样链上实现面向晶体异构动作空间的 GRPO；Diff2Flow 只作为后续效率增强模块。**

具体方法目标为：

> 在原始晶体空间而不是统一连续潜空间中，把 CGDiT 的逐步去噪过程定义为策略轨迹；分别为晶格、周期分数坐标和离散原子类型构造可计算且对称性约化的策略概率，通过分通道、分时间步信用分配执行 GRPO 更新，并使用有效性、目标性质、稳定性、新颖性、多样性和不确定性约束防止策略退化。

该方向利用了 CGDiT 的特有结构，正面处理现有潜空间晶体 RL 工作试图规避的异构动作概率问题。

研究边界如下：

- **当前主线**：原始 CGDiT diffusion policy 上的直接 GRPO；
- **核心方法贡献**：对称性约化的三通道策略概率，以及分通道、分时间步信用分配；
- **支撑贡献**：不确定性感知的多保真物理奖励闭环；
- **非当前主线**：Diff2Flow、FlowMM 重写、Reflow 和完整离散 flow；
- **Diff2Flow 启动条件**：直接 GRPO 闭环已经成立，且实测确认 rollout 成本是主要瓶颈，并且转换方案通过质量和总成本门槛。

### 2.3 推荐实施顺序

1. 先建立 CFG、best-of-N、rejection sampling 和 reward-weighted fine-tuning 基线；
2. 实现最小 vanilla GRPO，验证训练链路和奖励有效性；
3. 实现晶格、坐标、元素三路 factorized log-prob；
4. 引入对称性独立自由度归一化与分通道信用分配；
5. 完成时间步信用分配、稳定性和样本效率优化；
6. 接入不确定性奖励和 MLIP/DFT 多保真闭环；
7. 开展完整基线、消融、泛化和物理验证；
8. 只有触发效率门槛时，才启动 Diff2Flow 小规模转换实验。

---

## 3. 概念边界

“在扩散模型中引入强化学习”相关工作至少包含三类，不能混为一谈。

### 3.1 严格在线强化学习

把反向去噪过程看作轨迹：

\[
\tau=(x_T,x_{T-1},\ldots,x_0)
\]

使用不可微或黑盒终端奖励 \(R(x_0)\)，通过 REINFORCE、PPO、DDPO 或 GRPO 更新策略。

优点：

- 可使用结构有效性、MLIP、DFT、实验反馈等不可微奖励；
- 不需要通过完整采样链反向传播；
- 适合晶体生成中的离散元素和结构解析过程。

缺点：

- 梯度方差高；
- 奖励调用昂贵；
- 容易发生模式坍缩和 reward hacking；
- 必须正确计算或近似每一步策略概率。

### 3.2 可微奖励直接反传

ReFL、AlignProp 和 DRaFT 等方法通过部分或完整去噪链反传奖励梯度。

优点：梯度方差通常比策略梯度低，样本效率较高。

限制：奖励函数必须端到端可微，显存占用较高。当前 CGDiT 的 M3GNet 评价流程涉及晶体构图、离散原子类型、`pymatgen` 结构预处理和 `torch.no_grad()` 推理，因此不是最自然的首选。

### 3.3 离线偏好优化

Diffusion-DPO 等方法使用 winner–loser 样本对进行离线优化，不执行在线策略 rollout。

该方向适合后续积累了可靠的候选晶体偏好数据以后使用，例如：

- 同一组成下，DFT 松弛后能量更低的结构为 winner；
- 同一目标条件下，更接近目标性质且仍满足稳定性的结构为 winner；
- 专家或实验结果确认的结构优于被淘汰结构。

---

## 4. 通用扩散模型中的代表工作

| 工作 | 年份/状态 | 方法类型 | 核心内容 | 与 CGDiT 的关系 |
|---|---:|---|---|---|
| [Training Diffusion Models with Reinforcement Learning（DDPO）](https://openreview.net/forum?id=w5QmbPyzfg) | 2023 | 在线 RL | 把反向去噪视为多步 MDP，使用策略梯度直接优化黑盒终端奖励 | 最基础的 RL 适配范式 |
| [DPOK](https://openreview.net/forum?id=8OTPepXzeh) | NeurIPS 2023 | 在线 RL | 在策略优化中加入相对预训练模型的 KL 正则 | 晶体生成需要参考模型约束以避免失效和坍缩 |
| [ImageReward / ReFL](https://arxiv.org/abs/2304.05977) | NeurIPS 2023 | 奖励反传 | 训练人类偏好奖励模型，并在较晚去噪阶段直接优化奖励 | 可借鉴奖励建模，但不是当前黑盒物理奖励的首选 |
| [Aligning Text-to-Image Diffusion Models with Reward Backpropagation（AlignProp）](https://arxiv.org/abs/2310.03739) | 2024 | 奖励反传 | 通过去噪过程反传奖励梯度，结合 LoRA 和梯度检查点 | 可用于对比，但要求奖励可微 |
| [Directly Fine-Tuning Diffusion Models on Differentiable Rewards（DRaFT）](https://openreview.net/forum?id=1vmSEVL19f) | ICLR 2024 | 奖励反传 | 提出完整反传、截断反传和低方差变体 | 可作为可微代理奖励的对照方法 |
| [Diffusion Model Alignment Using Direct Preference Optimization](https://openaccess.thecvf.com/content/CVPR2024/html/Wallace_Diffusion_Model_Alignment_Using_Direct_Preference_Optimization_CVPR_2024_paper.html) | CVPR 2024 | 离线偏好优化 | 用扩散 ELBO 构造 DPO 目标，直接利用成对偏好数据 | 适合未来基于 DFT 排序构造偏好对 |
| [Confronting Reward Overoptimization for Diffusion Models（TDPO-R）](https://openreview.net/forum?id=v2o9rRJcEv) | ICML 2024 | 在线 RL | 利用扩散时间结构进行更细粒度信用分配，缓解奖励过优化 | 说明不能简单把同一终端奖励粗暴分配给全部时间步 |
| [Score as Action](https://openreview.net/forum?id=V5HEX6stS2) | ICML 2025 | 连续时间 RL | 将 score 视为连续时间控制动作，提高对求解器和时间离散的鲁棒性 | 更适合连续 SDE/flow 路线，可作为理论参考 |

### 4.1 通用工作给出的主要经验

1. 黑盒奖励适合策略梯度，端到端可微奖励适合直接反传；
2. KL 正则或参考策略约束几乎不可缺少；
3. 终端奖励会带来时间步信用分配困难；
4. 奖励持续升高不代表真实质量提高；
5. 过强优化常导致模式坍缩、分布漂移或代理奖励漏洞；
6. 必须使用独立评价器和分布外检查验证真实提升。

---

## 5. 晶体、材料和三维分子方向的相关工作

### 5.1 直接晶体扩散强化学习

| 工作 | 年份/状态 | 已完成内容 | 对本项目创新边界的影响 |
|---|---:|---|---|
| [MatInvent: Reinforcement Learning for 3D Crystal Diffusion Generation](https://openreview.net/forum?id=Ovxfri7l5L)；[扩展预印本](https://arxiv.org/abs/2511.03112) | AI4Mat-ICLR 2025 / 2025 预印本 | 等变去噪 MDP、奖励加权 KL、经验回放、多样性过滤，支持单目标和多目标优化 | “RL + 晶体扩散 + KL + 回放”已经不新 |
| [Reinforcement Learning with Formation Energy Feedback for Material Diffusion Models（RLFEF）](https://doi.org/10.1016/j.neunet.2025.108146) | Neural Networks 2026 | 以形成能为奖励，讨论材料扩散的策略梯度与晶体对称性保持 | “形成能反馈”和“RL 后保持对称性”不能单独作为创新 |
| [Guiding Generative Models to Uncover Diverse and Novel Crystals via Reinforcement Learning（Chemeleon2）](https://www.nature.com/articles/s42256-026-01262-4) | Nature Machine Intelligence 2026 | 潜空间扩散 + GRPO；奖励包含创造性、凸包稳定性、组成与结构多样性 | “GRPO + 稳定性/新颖性/多样性”已经形成正式发表工作 |
| [Open Materials Generation with Inference-Time Reinforcement Learning（OMatG-IRL）](https://arxiv.org/abs/2602.00424) | 2026 预印本/Workshop | 在推理阶段对 velocity field 进行策略梯度；可学习时间相关退火策略并大幅减少采样步数 | 单独做“RL 学习采样调度或减少步数”创新不足 |
| [CRYSTAL: Coordinated Multi-Objective Reinforcement Learning for Crystal Generation](https://openreview.net/pdf/94d95333b625bc19463eca098ff60038d639d590.pdf) | ICML 2026 AI for Science Workshop | GRPO、多目标物理奖励、显式新颖性和多样性、乘法奖励聚合 | 普通多目标组合和乘法奖励也已进入拥挤区域 |

### 5.2 相邻三维分子工作

| 工作 | 年份 | 主要贡献 | 可借鉴内容 |
|---|---:|---|---|
| [Guiding Diffusion Models with Reinforcement Learning for Stable Molecule Generation（RLPF）](https://arxiv.org/abs/2508.16521) | 2025 | 将 DDPO/PPO 用于等变三维分子扩散，以力场反馈提高物理稳定性 | 力/能量物理反馈、等变策略和稳定性评价 |
| [Uncertainty-Aware Multi-Objective RL-Guided Diffusion Models for 3D De Novo Molecular Design](https://arxiv.org/abs/2510.21153) | 2025 | 用代理模型预测不确定性动态调整多目标奖励 | 不确定性感知不能单独声称首创，但可作为晶体方法的重要组成部分 |
| [Fine-Tuning Diffusion Models for Molecular Generation via Reinforcement Learning and Fast Sampling（FTDiff）](https://arxiv.org/abs/2606.01220) | 2026 | GRPO 风格优化、阈值感知多目标奖励和快速采样 | 说明“GRPO + 多目标 + 快速采样”在相邻领域也已出现 |

### 5.3 重要非 RL 对照

- [WyckoffDiff](https://arxiv.org/abs/2502.06485)：直接在包含 Wyckoff 对称信息的离散表示上生成晶体；
- [Symmetry-aware Conditional Generation of Crystal Structures Using Diffusion Models](https://arxiv.org/abs/2601.08115)：将性质条件和 Wyckoff 对称生成结合；
- 当前 CGDiT 的普通条件生成和 classifier-free guidance；
- 传统高通量采样、rejection sampling 和 Bayesian/active learning 流程。

如果论文主张“对称性带来创新”，至少需要与这些对称生成工作区分，并证明创新位于策略概率与信用分配，而不仅是模型原本已经保持空间群对称性。

---

## 6. 当前 CGDiT 的代码现状

### 6.1 已确认的模型结构

当前 `Diffusion` 模型同时处理三类变量：

1. **晶格变量**：先转换到 6 维 `crys_fam` 表示，并投影到给定空间群允许的子空间；
2. **坐标变量**：对分数坐标使用周期噪声和 score matching，通过 `anchor_index`、`ops` 和 `ops_inv` 保持对称性；
3. **原子类型变量**：使用 D3PM/MaskDiffusion 进行离散扩散和 categorical posterior 采样。

关键代码位置：

- 模型初始化与三类扩散组件：`cgdit/pl_modules/diffusion.py:78`
- 条件编码器：`cgdit/pl_modules/diffusion.py:107`
- 普通扩散训练前向：`cgdit/pl_modules/diffusion.py:129`
- 当前采样入口及 `@torch.no_grad()`：`cgdit/pl_modules/diffusion.py:217`
- 反向轨迹循环：`cgdit/pl_modules/diffusion.py:296`
- D3PM posterior 和 categorical 动作：`cgdit/pl_modules/diffusion.py:451`
- 完整轨迹堆叠：`cgdit/pl_modules/diffusion.py:469`
- CFG 条件/无条件输出混合：`cgdit/pl_modules/diffusion.py:478`
- 晶格、坐标和元素损失：`cgdit/pl_modules/training_utils/diffusion_loss.py:41`

### 6.2 已有条件控制

仓库已经包含：

- 形成能条件；
- 能带条件；
- 能量高于凸包条件；
- 多属性条件版本；
- classifier-free guidance；
- 多模态条件 Adapter、对比学习和 curriculum learning。

相关配置：

- `conf/model/diffusion.yaml`
- `conf/model/exp_mp20_fe.yaml`
- `conf/model/exp_mp20_bg.yaml`
- `conf/model/exp_mp20_eh.yaml`
- `conf/model/exp_mp20_fe_bg_eh.yaml`
- `conf/model/exp_mp20_fe_bg_eh_multimodal.yaml`

因此强化学习实验必须与当前条件生成和 CFG 进行公平比较，不能只与无条件模型比较。

### 6.3 已有性质评价能力

仓库包含 M3GNet surrogate 和直接结构预测接口：

- `cgdit/prop_models/gnn_models/m3gnet.py`
- `conf/model/m3gnet.yaml`
- `scripts/predict_property.py`

当前脚本可以作为奖励模型接口原型，但在正式 RL 中不能直接把单个 surrogate 同时作为训练奖励和最终评价器。

### 6.4 当前缺口

当前实现尚缺：

- 每个反向步骤的 `log_prob`；
- rollout policy 与 reference policy 的概率比；
- 分结构的 KL；
- group-relative advantage；
- 奖励模型 ensemble 和不确定性；
- RL rollout buffer；
- 代理奖励与独立评价器分离；
- MLIP/DFT 多保真反馈接口；
- RL 专用配置、日志和测试。

### 6.5 当前验证限制

截至文档创建时，本地工作区未发现 `.ckpt` 文件。日志中存在远端训练 checkpoint 记录，但当前机器上无法直接加载，因此尚未实测：

- 当前基线的生成有效率；
- CFG 对目标性质的真实提升；
- M3GNet 对生成分布的 OOD 误差；
- RL rollout 的显存和耗时；
- 奖励分布与有效梯度比例。

---

## 7. 推荐的强化学习建模方式

### 7.1 MDP 定义

设条件为 \(c\)，包括目标性质、空间群、原子数或固定组成。定义：

\[
s_t=(t,A_t,X_t,L_t,c)
\]

\[
a_t=(A_{t-1},X_{t-1},L_{t-1})
\]

其中：

- \(A_t\)：原子类型；
- \(X_t\)：周期分数坐标；
- \(L_t\)：晶格或 `crys_fam` 表示；
- \(t\)：扩散时间步。

大多数物理奖励在完全去噪后计算：

\[
R(\tau)=R(A_0,X_0,L_0,c)
\]

### 7.2 最小 GRPO

在相同条件下生成一组 \(G\) 个候选。组内优势：

\[
\hat A_i=
\frac{R_i-\operatorname{mean}(R_1,\ldots,R_G)}
{\operatorname{std}(R_1,\ldots,R_G)+\epsilon}
\]

使用旧策略与当前策略的概率比：

\[
r_{i,t}(\theta)=
\exp\left(
\log\pi_\theta(a_{i,t}|s_{i,t})-
\log\pi_{\mathrm{old}}(a_{i,t}|s_{i,t})
\right)
\]

优化 clipped surrogate，并添加相对预训练 reference policy 的 KL：

\[
\mathcal L_{mathrm{RL}}=
-\mathbb E\left[
\min\left(
r_{i,t}\hat A_i,
\operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)\hat A_i
\right)
\right]
+\beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})
\]

### 7.3 CGDiT 的联合动作概率

联合策略暂定分解为：

\[
\log\pi_\theta
=
\log\pi_\theta^L
+\log\pi_\theta^X
+\log\pi_\theta^A
\]

#### 晶格通道

- 在 `crys_fam` 表示中计算高斯转移概率；
- 只在空间群投影后的有效独立子空间计算概率；
- 避免把被投影消除的自由度纳入 KL 和 log-prob。

#### 坐标通道

- 使用与周期边界一致的 wrapped-normal 或相应近似；
- 只在对称性独立 anchor 上计算动作概率；
- 不应对由空间群操作复制出的全部原子重复计数；
- 需要验证 `% 1` 周期映射前后 log-prob 的一致性。

#### 原子类型通道

- 使用 D3PM `sample_and_compute_posterior_q` 返回的 categorical logits；
- 对实际采样的 \(A_{t-1}\) 调用 `Categorical(logits).log_prob(action)`；
- 需要明确最终 `argmax` 步是否纳入策略梯度，或保持为确定性解码。

### 7.4 可变尺寸归一化

如果直接对所有自由度求和，大原子数结构会自然产生更大绝对 log-prob 和 KL。建议分别按以下数量归一化：

- 晶格有效独立自由度；
- Wyckoff/anchor 数；
- 独立元素位点数或原子数；
- 实际参与更新的时间步数。

需通过实验比较：

- 不归一化；
- 按全部原子归一化；
- 按 anchor/Wyckoff 独立自由度归一化；
- 三通道分别归一化后再加权。

### 7.5 时间步处理

当前模型包含 1000 个采样步，不能一开始对所有步骤带梯度更新。

建议依次测试：

1. 只更新最后 \(K\) 步；
2. 从早、中、晚阶段分层抽取时间步；
3. 每个 iteration 随机选择连续窗口；
4. 根据梯度方差或奖励敏感性自适应选择时间步。

第 4 项可作为后续研究扩展，但 OMatG-IRL 已研究学习时间相关采样策略，因此不能单独作为唯一创新。

---

## 8. 奖励设计

### 8.1 不推荐只用形成能

简单奖励：

\[
R=-E_{\mathrm{form}}
\]

存在明显不足：

- 负形成能不等于位于组成凸包上；
- 未松弛结构上的预测能量可能没有清晰物理意义；
- 不能证明动力学稳定；
- 不能证明有限温稳定、动力学可达或可合成；
- 代理模型在 RL 推动的 OOD 结构上可能被利用。

### 8.2 推荐分层奖励

暂定总奖励：

\[
R=
R_{\mathrm{valid}}
\left(
w_sR_{\mathrm{stability}}
+w_pR_{\mathrm{property}}
+w_nR_{\mathrm{novelty}}
+w_dR_{\mathrm{diversity}}
\right)
-\lambda_uR_{\mathrm{uncertainty}}
\]

#### 有效性门控

建议包含：

- 晶格可逆且体积在合理范围；
- 无严重原子重叠；
- 元素编号和组成合法；
- `pymatgen`/相关解析器可构造结构；
- 目标空间群约束满足；
- 必要时加入电荷中性或氧化态可行性。

无效结构应给予固定惩罚，但要对惩罚裁剪，避免少量异常值控制整个组优势。

#### 稳定性奖励

推荐使用经过结构松弛后的：

- 形成能；
- 能量高于凸包 \(E_{\mathrm{hull}}\)；
- 力和应力残差；
- 必要时的声子或有限温稳定性。

RL 在线阶段可使用 MLIP/代理模型，最终评价必须使用独立方法。

#### 目标性质奖励

对于目标值 \(p^*\)，可使用平滑命中奖励：

\[
R_{\mathrm{property}}
=\exp\left(-\frac{|\mu_p-p^*|}{\tau_p}\right)
\]

也可以对阈值任务使用平滑 sigmoid，避免纯二值奖励导致有效信号过少。

#### 新颖性奖励

同时区分：

- 组成新颖性；
- 结构新颖性；
- 相对于训练集的最近邻距离；
- 相对于当前 replay buffer 的重复程度。

新颖性不能只用 material ID 去重。

#### 多样性奖励

可比较：

- 组成指纹的组内距离；
- 结构表示或预训练嵌入的组内距离；
- MMD；
- DPP；
- marginal utility，即新样本对已探索集合覆盖的增益。

#### 不确定性奖励

对代理模型 ensemble 的均值和标准差，可构造保守奖励：

\[
R_{\mathrm{stab}}
=
\operatorname{sigmoid}\left(
\frac{\tau_{\mathrm{hull}}-\mu_{\mathrm{hull}}}{T}
\right)
-\lambda\sigma_{\mathrm{hull}}
\]

对于希望最大化的性质，可使用 lower confidence bound；对于希望最小化的能量，可使用相应的保守上界或风险惩罚。

### 8.3 防止 reward hacking

- [ ] 训练奖励模型与最终评价模型使用不同架构或不同数据划分；
- [ ] 使用至少 3 个代理模型构成 ensemble；
- [ ] 记录 ensemble 方差和跨模型排序一致性；
- [ ] 对 OOD 结构降低奖励或进入高保真复核队列；
- [ ] 定期使用 MLIP 松弛刷新 replay buffer 标签；
- [ ] 对少量高奖励样本进行 DFT 验证；
- [ ] 检查代理奖励提高但独立评价下降的情况；
- [ ] 保存高奖励失败案例，分析模型利用的奖励漏洞。

---

## 9. 推荐创新假设

### 9.1 主假设 H1：对称性约化的异构动作概率

> 在晶格、坐标和元素的原始晶体表示中，按空间群独立自由度构造联合策略概率，可以在保持晶体对称性的同时，比普通 joint-log-prob GRPO 获得更低梯度方差和更好的奖励—有效性 Pareto 权衡。

需要验证：

- 与潜空间 GRPO 或普通 direct-space GRPO 的区别；
- 是否避免对称复制原子造成的重复计数；
- 是否减少大结构的 KL 偏置；
- 是否在相同 reward-call budget 下提高目标命中率；
- 是否保持或提升有效率、唯一性和多样性。

### 9.2 主假设 H2：分通道、分时间步信用分配

> 晶格、坐标和元素三类动作对不同物理奖励的贡献不同；使用通道特异 advantage/KL 和时间分层更新，比把同一个终端优势分配给所有动作更稳定。

可能的对应关系：

- 晶格通道：体积、密度、应力、晶系相关奖励；
- 坐标通道：局域几何、短程排斥、力残差和结构稳定性；
- 元素通道：组成可行性、形成能、带隙和目标化学空间；
- 全局奖励：凸包稳定性、新颖性和最终目标性质。

需要通过消融证明这种拆分确实降低方差或提高样本效率，而不是只增加超参数。

### 9.3 第二贡献 H3：不确定性感知的多保真奖励闭环

> 使用代理模型 ensemble 不确定性进行保守奖励，并主动选择高价值/OOD 候选进入 MLIP 或 DFT 复核，可减少代理模型利用，提高有限高保真预算下的真实稳定候选发现率。

注意：不确定性感知 RL 在分子方向已有相关工作，必须与 H1/H2 组合，不能把 H3 单独包装为首创。

### 9.4 可选扩展 H4：条件可调的 Pareto 策略

> 使用目标权重或偏好向量作为策略条件，使一个 RL 后训练模型在推理时生成不同稳定性—性质—新颖性权衡，并泛化到训练未见的权重组合。

该方向近期竞争较强，正式采用前需要进一步专项检索。

---

## 10. 哪些结果才足以支持“有创新”

### 10.1 算法层面

- 有新的策略概率、归一化、信用分配或优化目标；
- 不是只换奖励函数或把现有 PPO/GRPO 接到 CGDiT；
- 能说明为什么该设计只在晶体异构表示和对称约束下必要；
- 最好具有可验证的理论性质或至少清晰的概率推导。

### 10.2 实验层面

- 在相同奖励调用预算下超过强基线；
- 目标性质提高时，有效率、唯一性和多样性不发生明显坍缩；
- 独立评价器仍确认改进；
- MLIP 松弛和 DFT 子集仍保留提升趋势；
- 至少 3 个随机种子并报告置信区间；
- 关键模块消融能够支持提出的机制，而不是只报告总模型最好。

### 10.3 科学层面

- 稳定性不能只用负形成能证明；
- 至少报告凸包稳定性和结构松弛结果；
- 重要候选需要进一步做声子/动力学稳定性；
- 若声称可合成，需要提供合成路径、前驱体、动力学或实验闭环证据；
- 明确区分 0 K 热力学稳定、有限温稳定、动力学稳定和可合成性。

---

## 11. 分阶段工作计划

### 11.1 总体执行表

以下日历时间按“1 名研究人员 + 1 张可用 GPU、已有 checkpoint、数据和基础环境可运行”估计。阶段可以部分重叠；DFT 排队时间不计入开发工时。主线预计需要 **8–12 周**形成完整算法与代理/MLIP 证据闭环，包含 DFT/声子验证时通常需要 **10–16 周**。

| 阶段 | 建议周次 | 预计人工时间 | 核心任务 | 交付物 | 验收/继续条件 |
|---|---:|---:|---|---|---|
| P0 资产冻结与环境复现 | 第 1 周前半 | 2–3 天 | 解压并登记 checkpoint、resolved config、数据 split/hash、scaler、Git commit；跑通推理 | `checkpoint_manifest`、可复现推理命令、环境记录 | 相同种子可重复生成，权重和配置严格匹配 |
| P1 非 RL 基线与成本剖析 | 第 1 周后半–第 2 周 | 3–5 天 | 预训练/条件 CGDiT、CFG、best-of-N、rejection sampling；测 1000 步 rollout 时间 | 基线表、每样本 NFE/wall time/显存、固定样本集 | 获得直接 RL 的质量下限和成本基准 |
| P2 奖励与独立评价器 | 第 2–3 周 | 4–7 天 | 有效性门控、性质/稳定性、新颖性、多样性、不确定性；独立 evaluator | 统一 reward API、奖励审计报告、异常样本集 | 奖励排序符合物理直觉，独立 evaluator 不复用训练奖励逻辑 |
| P3 RL 轨迹与概率单元测试 | 第 3–4 周 | 5–8 天 | `sample_rl()`、轨迹缓存、reference policy、旧/新 log-prob、checkpoint 恢复 | RL sampler、三通道测试、最小配置 | 无更新时 ratio≈1、KL≈0，固定轨迹可复算概率 |
| P4 Vanilla GRPO 最小闭环 | 第 4–5 周 | 3–5 天 | 组内 advantage、clip、KL、日志、LoRA/后部微调、短程 smoke test | 可恢复的 GRPO checkpoint、训练曲线 | 奖励有稳定趋势且 validity/diversity 未立即坍缩 |
| P5 异构与对称性约化 GRPO | 第 5–7 周 | 7–12 天 | 晶格高斯、torus 坐标、D3PM 元素概率；独立自由度归一化 | 主方法实现、H1 消融 | 不同原子数/空间群不产生系统性 KL 偏差，优于 vanilla GRPO |
| P6 通道/时间信用分配 | 第 7–8 周 | 5–8 天 | channel-wise advantage/KL；最后 K 步、随机/分层时间步 | H2 消融、效率—质量曲线 | 找到稳定且更省显存/梯度方差更低的更新方案 |
| P7 正式基线、消融与泛化 | 第 8–10 周 | 10–15 天 | 至少 3 seeds；CFG、best-of-N、RWR、DDPO、GRPO、H1/H2/H3 | 主结果表、消融表、Pareto/样本效率图 | 核心结论在 3 seeds 下成立，公平预算和失败案例齐全 |
| P8 多保真物理闭环 | 第 9–12 周 | 10–20 天 + 排队 | surrogate ensemble、MLIP 松弛、少量 DFT/声子验证 | 候选清单、MLIP/DFT 记录、校准结果 | 独立高保真成功率优于固定 surrogate 策略 |
| P9 论文证据整理 | 第 11–12 周 | 5–10 天 | 方法图、实验表、失败边界、专项查重、复现实验清单 | 论文级图表和 claim–evidence map | 算法、物理可信性、成本三条证据链闭合 |
| PX Diff2Flow 条件分支 | 不进入默认关键路径 | 7–15 天试点 | 只从 D2F-L 开始，测少步质量和总成本 | go/no-go 报告 | 仅在下述效率门槛触发后开展 |

### 11.2 里程碑与决策门

| 里程碑 | 最晚建议时间 | 必须回答的问题 | Go 条件 | No-go 后动作 |
|---|---:|---|---|---|
| M0：基础模型可复现 | 第 1 周 | checkpoint 是否与配置、scaler、数据一致？ | 固定种子推理通过 | 暂停 RL，先修复资产版本问题 |
| M1：奖励可信 | 第 3 周 | 高 reward 是否对应独立评价改善？ | 分项奖励可解释，异常结构不会系统性高分 | 重做奖励门控或 surrogate 校准 |
| M2：GRPO 数学正确 | 第 4 周 | ratio、KL、mask、归一化是否正确？ | ratio≈1、KL≈0、梯度有限且可重复 | 不进入长训练，只修单元测试 |
| M3：直接 RL 有信号 | 第 5 周 | 奖励提升是否超过 CFG/best-of-N？ | 固定预算下改善且有效性/多样性在容忍范围 | 调整奖励、KL、更新层和组大小；仍无信号则停主实验 |
| M4：核心创新成立 | 第 8 周 | 对称性约化和信用分配是否真正有贡献？ | 至少在梯度方差、样本效率或 Pareto 上有一致增益 | 降级为工程结果或重新定义方法贡献 |
| M5：是否需要 Diff2Flow | 第 8 周后 | rollout 是否已成为不可承受的主要瓶颈？ | rollout 占 RL wall time >70%，或计划样本量在现有预算内无法完成 | 继续直接 diffusion GRPO，不做转换 |
| M6：Diff2Flow 是否扩展 | PX 试点结束 | 转换成本能否被未来 rollout 节省抵消？ | D2F-L 总效率≥1.5×，20–50 NFE 质量非劣，且预计节省大于转换成本 | 停止 Diff2Flow，不扩展 torus/离散通道 |

### 11.3 单次训练时长与算力预算估计

当前 `output.zip` 提供了可用的本地历史证据：MP-20 base 在单 GPU、FP32、batch size 32、1000 diffusion steps 配置下，从开始训练到进入测试约 **39.1 小时**，得到 `epoch=869-step=737760.ckpt`；MP-20 `e_above_hull` 条件模型从 `step=339200` 恢复到 `step=788640` 约 **17.3 小时**。这些记录说明基础模型完整训练是“约 1–2 天/次”的量级，但日志没有 GPU 型号，不能把它直接当成当前服务器的速度承诺。

GRPO 的总耗时主要由 rollout 而不是参数更新决定，建议用阶段 P1 的实测值计算：

\[
T_{\mathrm{run}}
\approx
N_{\mathrm{update}}\left(T_{\mathrm{rollout}}(B_{\mathrm{cond}}\times G, NFE)
+T_{\mathrm{reward}}
+N_{\mathrm{epoch}}T_{\mathrm{backward}}\right).
\]

其中 (G) 是每个条件的组大小。时间步截断可以减少反向和 log-prob 计算，但如果仍用 1000 步生成完整晶体，它不会自动消除 rollout 的 1000 次网络调用。

在尚未获得当前 GPU 的 P1 profiling 前，只给出用于排期的宽区间：

| 运行级别 | 建议设置 | 单次预计时长 | 用途 |
|---|---|---:|---|
| 数学 smoke test | 10–50 updates，极小 batch/group，少量保存时间步 | 0.5–3 小时 | 检查 NaN、ratio、KL、恢复训练 |
| 最小 GRPO pilot | 100–500 updates，group 4–8，LoRA/后部更新 | 6–36 小时 | 判断是否存在可学习奖励信号 |
| 单个正式超参设置 | 1000–3000 updates，固定 reward budget | 1–4 GPU 天 | 形成一条可比较主结果 |
| 单方法 3 seeds | 3 个正式运行 | 3–12 GPU 天 | 统计稳定性与误差条 |
| 主实验矩阵 | 6–10 个关键方法/消融 × 3 seeds | 18–120 GPU 天 | 完整论文证据；多 GPU 可近线性并行缩短日历时间 |
| MLIP/DFT 验证 | 取决于候选数与队列 | 另计 | 不应混入 GRPO GPU 时间 |

上述范围必须在 P1 完成后，用“生成 100/1000 个晶体的实测 wall time、奖励耗时和单次 backward 耗时”重新计算。正式长训练前先提交一页算力预算，不用基础模型训练时长猜测 RL 时长。

### 阶段 0：确认研究任务和评价目标

目标：确定第一篇工作的最小科学问题，避免一开始同时优化所有性质。

- [ ] 确认首个数据集：建议从 MP-20 开始；
- [ ] 确认首个任务类型：固定组成 CSP、de novo 生成或性质条件生成；
- [ ] 确认首个目标：建议优先选择稳定性或“稳定性 + 一个功能性质”；
- [ ] 确认生成时空间群和原子数是否固定；
- [ ] 确认可用 GPU、MLIP 和 DFT 预算；
- [ ] 获取至少一个可加载的本地 CGDiT checkpoint；
- [ ] 获取或训练独立的形成能、凸包能和目标性质 surrogate。

验收条件：形成一页任务定义，明确输入条件、动作空间、奖励、数据集、预算和最终物理评价。

### 阶段 1：复现并冻结非 RL 基线

- [ ] 加载当前最佳 unconditional checkpoint；
- [ ] 加载当前最佳单条件/多条件 checkpoint；
- [ ] 固定生成种子和测试条件；
- [ ] 运行 CFG scale 扫描，例如 `0, 0.5, 1, 2, 4`；
- [ ] 实现 best-of-N 和 rejection sampling；
- [ ] 运行 reward-weighted fine-tuning 或高奖励样本 SFT；
- [ ] 统计 reward calls、wall time 和 GPU memory；
- [ ] 生成统一基线报告。

验收条件：相同条件下得到可重复的 validity、uniqueness、novelty、目标命中率和代理稳定性结果。

### 阶段 2：奖励系统和独立评价系统

- [ ] 统一 `Structure -> RewardComponents` 接口；
- [ ] 实现结构有效性门控；
- [ ] 实现性质目标奖励；
- [ ] 实现组成和结构新颖性；
- [ ] 实现组内多样性；
- [ ] 训练或加载 surrogate ensemble；
- [ ] 输出均值、标准差和 OOD 指标；
- [ ] 建立独立评价器；
- [ ] 对生成样本执行小规模 MLIP 松弛测试；
- [ ] 检查奖励分布、稀疏性和异常值。

验收条件：同一批结构能稳定输出分项奖励、总奖励、不确定性和独立评价结果。

### 阶段 3：最小 vanilla GRPO

- [ ] 新增独立的 `sample_rl()`，不修改现有推理语义；
- [ ] rollout 时保存状态、动作、时间步和旧 log-prob；
- [ ] 冻结 pretrained reference policy；
- [ ] 实现组内 advantage；
- [ ] 实现 clipped ratio；
- [ ] 实现 reference KL；
- [ ] 先只更新 conditioner、LoRA 或 decoder 后部；
- [ ] 实现 RL 专用 Hydra 配置；
- [ ] 实现 reward、KL、entropy、clip fraction、gradient norm 日志；
- [ ] 在小数据、小 batch 和少时间步设置下做 smoke test。

验收条件：奖励能在不立即破坏有效率的情况下稳定上升，且可以从 checkpoint 恢复。

### 阶段 4：对称性约化的三路策略概率

- [ ] 实现晶格有效子空间的高斯 log-prob；
- [ ] 实现 anchor 坐标的周期 log-prob；
- [ ] 实现 D3PM categorical log-prob；
- [ ] 实现按独立自由度的 size normalization；
- [ ] 对三通道分别记录 KL、entropy 和梯度；
- [ ] 检查空间群投影前后概率定义；
- [ ] 检查对称复制原子是否被重复计数；
- [ ] 比较 joint advantage 和 channel-wise advantage；
- [ ] 比较统一 KL 和 channel-wise KL budget。

验收条件：概率比在无参数更新时接近 1；reference KL 在相同策略时接近 0；不同原子数样本不再出现明显系统性 KL 偏差。

### 阶段 5：时间信用分配与效率优化

- [ ] 比较最后 \(K\) 步、随机时间步和分层时间步；
- [ ] 比较 `K = 1, 5, 10, 20, 50`；
- [ ] 记录不同时间段的梯度方差；
- [ ] 记录每种设置的 reward-call efficiency；
- [ ] 评估是否需要 critic；
- [ ] 评估是否需要 learned timestep scheduler；
- [ ] 尝试将 1000 步推理缩减到更少步骤，但保持独立于主创新的消融。

验收条件：选定一个稳定、显存可接受且相对全时间步具有明显效率优势的信用分配方案。

### 阶段 6：多保真主动闭环

- [ ] 建立候选优先级：高奖励、高不确定性、Pareto 边界或高新颖性；
- [ ] 使用 MLIP 松弛候选；
- [ ] 计算松弛后能量、力、应力和结构变化；
- [ ] 将 MLIP 结果回填 replay buffer；
- [ ] 选择小批量候选进入 DFT；
- [ ] 使用高保真结果校准 surrogate 或奖励；
- [ ] 比较有无不确定性门控的真实成功率；
- [ ] 记录发现一个 DFT 验证稳定候选的成本。

验收条件：在相同 MLIP/DFT 预算下，多保真策略比只使用固定 surrogate 找到更多独立评价通过的候选。

### 阶段 7：完整实验、消融和论文证据闭环

- [ ] 至少 3 个随机种子；
- [ ] 全部方法使用相同 reward-call budget；
- [ ] 完成 CFG、best-of-N、RWR、DDPO、vanilla GRPO 等基线；
- [ ] 完成 H1/H2/H3 消融；
- [ ] 完成不同数据集或性质的泛化实验；
- [ ] 完成独立 evaluator、MLIP 和 DFT 验证；
- [ ] 对代表候选进行声子或必要的动力学分析；
- [ ] 整理失败案例和 reward hacking 案例；
- [ ] 绘制 Pareto 前沿和样本效率曲线；
- [ ] 在确定题目和贡献表述后进行专项文献查重。

验收条件：算法提升、物理可信性和样本效率三条证据链均闭合。

### 条件阶段 X：Diff2Flow 效率增强

该阶段默认不执行，也不阻塞 P0–P9。只有 M5 被触发后才进入。

- [ ] 先计算剩余计划 rollout 数量和现有 NFE 总成本；
- [ ] 只实现或修正 D2F-L，不同时处理 torus 坐标和离散元素；
- [ ] 固定 checkpoint、样本数和评价器，比较 1000-step diffusion 与 20/50-NFE D2F-L；
- [ ] 报告转换训练、样本配对和推理的总 GPU-hours；
- [ ] 检查 validity、结构/组成多样性、空间群匹配和性质分布非劣性；
- [ ] 仅在收益门槛通过时扩展坐标通道；原子类型优先保留同步 D3PM hybrid。

验收条件：端到端总效率至少提高 1.5 倍，20–50 NFE 的核心质量指标不显著劣于直接 diffusion，且未来 rollout 节省大于转换成本；否则停止该分支。

---

## 12. 建议实验矩阵

| 编号 | 方法 | 是否更新模型 | 奖励类型 | 主要目的 |
|---|---|---:|---|---|
| B0 | 预训练无条件 CGDiT | 否 | 无 | 基础生成质量 |
| B1 | 条件 CGDiT | 否 | 条件标签 | 当前条件控制基线 |
| B2 | CFG scale sweep | 否 | 条件标签 | 判断简单 guidance 能达到的上限 |
| B3 | Best-of-N | 否 | 黑盒奖励 | 排除多采样筛选带来的提升 |
| B4 | Rejection sampling | 否 | 黑盒奖励 | 评价采样预算换性能的效率 |
| B5 | Reward-weighted fine-tuning | 是 | 黑盒奖励转样本权重 | 非 RL 训练基线 |
| B6 | DDPO/PPO | 是 | 终端奖励 | 标准在线 RL 基线 |
| B7 | Vanilla GRPO | 是 | 组相对终端奖励 | 当前最重要 RL 基线 |
| M1 | Factorized mixed-action GRPO | 是 | 三路联合概率 | 验证异构动作建模 |
| M2 | Symmetry-reduced GRPO | 是 | 独立自由度概率 | 验证对称性约化贡献 |
| M3 | Channel-wise credit/KL | 是 | 分通道奖励或优势 | 验证信用分配贡献 |
| M4 | M3 + uncertainty reward | 是 | 保守 ensemble 奖励 | 验证代理可靠性 |
| M5 | M4 + multi-fidelity loop | 是 | MLIP/DFT 反馈 | 验证真实发现效率 |

### 12.1 关键公平性约束

- 相同初始 checkpoint；
- 相同条件集合；
- 相同随机种子；
- 相同生成样本数；
- 相同 reward-call budget；
- 相同 MLIP/DFT 预算；
- 相同后处理和有效性筛选；
- 相同独立评价器；
- 报告 wall time 和 GPU memory，不能只报告最终奖励。

---

## 13. 评价指标

### 13.1 生成质量

- Validity；
- Uniqueness；
- Novelty；
- 组成多样性；
- 结构多样性；
- mSUN 或对应稳定—唯一—新颖联合指标；
- 空间群匹配率；
- 原子重叠率和晶格异常率。

### 13.2 目标性质

- 目标值 MAE；
- 目标区间命中率；
- 阈值通过率；
- 多目标 Pareto hypervolume；
- 不同目标权重的覆盖度；
- 未见目标或未见权重的泛化。

### 13.3 热力学和动力学

- 松弛前后形成能；
- 松弛后 \(E_{\mathrm{hull}}\)；
- 力和应力残差；
- 结构松弛 RMSD；
- 声子虚频；
- 必要时有限温 MD 稳定性。

### 13.4 强化学习训练

- 平均奖励及各分项奖励；
- reference KL；
- 三通道 KL；
- entropy；
- PPO/GRPO clip fraction；
- advantage 分布；
- 梯度范数和梯度方差；
- 无效 rollout 比例；
- 每次有效策略更新的 reward calls；
- reward model 与独立 evaluator 的相关性。

### 13.5 成本指标

- 每 1000 个样本的 GPU 小时；
- 每个稳定候选的代理模型调用次数；
- 每个 MLIP 验证通过候选的成本；
- 每个 DFT 验证通过候选的成本；
- 采样步数和生成 wall time。

---

## 14. 消融计划

### 14.1 策略概率消融

- [ ] 三通道联合概率 vs 只优化坐标；
- [ ] 只优化晶格；
- [ ] 只优化原子类型；
- [ ] 全原子坐标计数 vs anchor 计数；
- [ ] 无 size normalization vs 按原子数 vs 按独立自由度；
- [ ] 统一 KL vs 三通道 KL。

### 14.2 信用分配消融

- [ ] 同一终端 advantage 分配给全部通道；
- [ ] channel-wise advantage；
- [ ] 只更新后期时间步；
- [ ] 分层随机时间步；
- [ ] learned critic 或 temporal advantage。

### 14.3 奖励消融

- [ ] 只有目标性质；
- [ ] 加有效性；
- [ ] 加稳定性；
- [ ] 加新颖性；
- [ ] 加多样性；
- [ ] 加不确定性；
- [ ] 固定线性权重 vs 自适应或约束式多目标优化。

### 14.4 多保真消融

- [ ] 单一 surrogate；
- [ ] surrogate ensemble；
- [ ] ensemble + OOD 惩罚；
- [ ] ensemble + MLIP；
- [ ] ensemble + MLIP + DFT 主动更新。

---

## 15. 主要风险与应对

| 风险 | 可能表现 | 应对措施 |
|---|---|---|
| 策略概率定义错误 | ratio 爆炸、KL 非零基线异常 | 单元测试每个通道；相同策略 ratio≈1、KL≈0 |
| 对称复制重复计数 | 大空间群或大原子数样本主导梯度 | 只按 anchor/Wyckoff 独立自由度计数 |
| 奖励稀疏 | 大多 rollout 奖励相同 | 使用平滑阈值、组内归一化和分项奖励 |
| 模式坍缩 | 组成和结构多样性快速下降 | KL、entropy、多样性奖励、replay 和 early stopping |
| reward hacking | surrogate 奖励升高但独立评价下降 | ensemble、不确定性、独立 evaluator、MLIP/DFT 复核 |
| 训练成本过高 | 1000 步 rollout 无法承受 | 时间步子采样、截断更新、LoRA、缓存无梯度轨迹 |
| 形成能被误当稳定性 | 高奖励结构仍高于凸包 | 使用同化学空间凸包和松弛后能量 |
| 创新与现有工作重合 | 审稿时被认为是 MatInvent/Chemeleon2 复现 | 把贡献限定为原始异构空间策略概率和信用分配，并专项检索 |
| 本地缺少 checkpoint | 无法建立真实基线 | 先同步或导出远端最佳 checkpoint |

---

## 16. 待确认事项

以下选择会显著改变技术路线，需要在实现前确认。

### 16.1 首个任务

- [ ] A. 固定组成的晶体结构预测；
- [ ] B. 固定空间群/原子数的 de novo 生成；
- [ ] C. 目标性质条件生成；
- [ ] D. 多目标性质生成。

当前建议：先选 C，但在固定或受控空间群/原子数条件下进行，降低首轮变量数量。

### 16.2 首个目标性质

- [ ] 形成能；
- [ ] 能量高于凸包；
- [ ] 带隙；
- [ ] 形成能 + 带隙；
- [ ] 其他：________________。

当前建议：稳定性主奖励使用松弛后的 \(E_{\mathrm{hull}}\) 或其可靠代理，形成能只作为组成和能量信息的一部分；功能目标可先选带隙。

### 16.3 首个 RL 参数范围

- [ ] 只更新 conditioner；
- [ ] 更新 conditioner + decoder 最后若干层；
- [ ] LoRA；
- [ ] 全参数更新。

当前建议：优先 LoRA；如果暂不引入 LoRA，则更新 conditioner 和 decoder 后部，并保留完整 frozen reference。

### 16.4 高保真预算

- 可用 MLIP：________________
- 可用 GPU 数量：________________
- 可用 DFT 计算数量：________________
- 每轮最大候选数量：________________
- 是否具备声子计算预算：________________

### 16.5 论文主贡献

- [ ] H1：对称性约化的异构动作概率；
- [ ] H2：分通道、分时间步信用分配；
- [ ] H3：不确定性感知多保真奖励；
- [ ] H4：可调 Pareto 策略。

当前建议：H1 为核心方法贡献，H2 为必要机制贡献，H3 为物理可信和样本效率贡献；H4 作为可选扩展。

---

## 17. 决策记录

| 日期 | 决策 | 理由 | 影响文件/实验 | 状态 |
|---|---|---|---|---|
| 2026-08-04 | 建议以 symmetry-reduced heterogeneous GRPO 为主线 | 普通晶体 GRPO、多目标奖励和形成能反馈已有直接工作 | 待新增 RL sampler、log-prob、reward 和配置 | 已确认 |
| 2026-08-04 | 不把形成能单独视为稳定性证明 | 形成能不等价于凸包、动力学或可合成性 | 奖励和最终评价需加入 hull/relaxation | 待确认 |
| 2026-08-04 | 先完成非 RL 基线再做 RL | 排除 CFG 和 best-of-N 已能达到相同提升 | 基线实验矩阵 | 待确认 |
| 2026-08-09 | 直接 CGDiT diffusion-GRPO 为唯一当前主线 | 已有直接扩散 RL 先例，现有 `sample()` 可作为轨迹入口，无需先转 flow | P0–P9 主计划 | 已确认 |
| 2026-08-09 | Diff2Flow 降级为条件效率模块 | 其价值取决于 rollout 瓶颈和转换投资回收，不能阻塞方法验证 | 仅在 M5/M6 通过后启动 PX | 已确认 |

后续新增决策时，在此表追加记录，不删除历史项。

---

## 18. 建议的代码模块划分

以下仅是实施建议，尚未创建对应文件。

```text
cgdit/
  pl_modules/
    rl/
      rollout.py               # rollout 与轨迹缓存
      transition_logprob.py    # 晶格/坐标/元素三路 log-prob
      grpo_loss.py             # advantage、ratio、clip、KL
      reward.py                # 奖励组合接口
      replay_buffer.py         # 高奖励和高保真反馈缓存
      uncertainty.py           # ensemble 与 OOD 处理
    diffusion_rl.py            # 继承或包装现有 Diffusion

conf/
  model/
    diffusion_rl.yaml
  rl/
    grpo.yaml
    reward_stability.yaml
    reward_multiobjective.yaml

scripts/
  train_rl.py
  evaluate_rl.py
  build_reward_dataset.py
  validate_candidates_mlff.py
```

原则：

- 保留当前 `Diffusion.sample()` 的推理行为；
- RL 功能使用独立模块和配置；
- 每一处修改都应直接对应当前研究问题；
- 不在首版实现不需要的通用 RL 框架抽象。

---

## 19. 近期最小行动清单

在开始写 RL 代码前，建议先完成以下项目：

- [ ] 将远端最佳 CGDiT checkpoint 放到当前工作区可访问位置；
- [ ] 用固定条件生成一批基线结构；
- [ ] 跑通当前 M3GNet/性质评价脚本；
- [ ] 对生成结构检查 surrogate 的训练分布外程度；
- [ ] 完成 CFG 和 best-of-N 基线；
- [ ] 确认首个目标性质和 MLIP/DFT 预算；
- [ ] 根据确认后的 H1/H2 范围做一次专项文献查重；
- [ ] 再编写最小 `sample_rl()` 和三路 log-prob 单元测试。

---

## 20. 参考文献入口

1. Black, K. et al. [Training Diffusion Models with Reinforcement Learning](https://openreview.net/forum?id=w5QmbPyzfg), 2023.
2. Fan, Y. et al. [DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models](https://openreview.net/forum?id=8OTPepXzeh), NeurIPS 2023.
3. Xu, J. et al. [ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977), NeurIPS 2023.
4. Prabhudesai, M. et al. [Aligning Text-to-Image Diffusion Models with Reward Backpropagation](https://arxiv.org/abs/2310.03739), 2024.
5. Clark, K. et al. [Directly Fine-Tuning Diffusion Models on Differentiable Rewards](https://openreview.net/forum?id=1vmSEVL19f), ICLR 2024.
6. Wallace, B. et al. [Diffusion Model Alignment Using Direct Preference Optimization](https://openaccess.thecvf.com/content/CVPR2024/html/Wallace_Diffusion_Model_Alignment_Using_Direct_Preference_Optimization_CVPR_2024_paper.html), CVPR 2024.
7. Zhang, Z. et al. [Confronting Reward Overoptimization for Diffusion Models](https://openreview.net/forum?id=v2o9rRJcEv), ICML 2024.
8. Zhao, H. et al. [Score as Action](https://openreview.net/forum?id=V5HEX6stS2), ICML 2025.
9. Chen, J. et al. [MatInvent](https://openreview.net/forum?id=Ovxfri7l5L), AI4Mat-ICLR 2025；[扩展预印本](https://arxiv.org/abs/2511.03112).
10. Huang, J. et al. [Reinforcement Learning with Formation Energy Feedback for Material Diffusion Models](https://doi.org/10.1016/j.neunet.2025.108146), Neural Networks 194, 108146, 2026.
11. Park, H. & Walsh, A. [Guiding Generative Models to Uncover Diverse and Novel Crystals via Reinforcement Learning](https://www.nature.com/articles/s42256-026-01262-4), Nature Machine Intelligence 8, 1087–1099, 2026.
12. Hoellmer, P. & Martiniani, S. [Open Materials Generation with Inference-Time Reinforcement Learning](https://arxiv.org/abs/2602.00424), 2026.
13. Zhou, Z. et al. [Guiding Diffusion Models with Reinforcement Learning for Stable Molecule Generation](https://arxiv.org/abs/2508.16521), 2025.
14. Chen, L. et al. [Uncertainty-Aware Multi-Objective RL-Guided Diffusion Models for 3D De Novo Molecular Design](https://arxiv.org/abs/2510.21153), 2025.
15. Lin, G. et al. [Fine-Tuning Diffusion Models for Molecular Generation via Reinforcement Learning and Fast Sampling](https://arxiv.org/abs/2606.01220), 2026.
16. Kelvinius, F. E. et al. [WyckoffDiff](https://arxiv.org/abs/2502.06485), 2025.
17. Ishii, T. et al. [Symmetry-aware Conditional Generation of Crystal Structures Using Diffusion Models](https://arxiv.org/abs/2601.08115), 2026.

---

## 21. 修订记录

| 版本 | 日期 | 修改内容 | 修改人 |
|---|---|---|---|
| v0.1 | 2026-08-04 | 整理初步文献调研、CGDiT 适配分析、创新假设、实验矩阵和分阶段计划 | Codex |
| v0.2 | 2026-08-09 | 锁定直接 CGDiT-GRPO 主目标；将 Diff2Flow 改为条件效率分支；加入详细时间表、里程碑、训练时长和 go/no-go 门槛 | Codex |
