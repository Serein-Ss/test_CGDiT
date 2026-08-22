# 扩散模型引入强化学习的工作谱系

## 1. 先区分三个不同问题

1. **本调研主问题：用 RL 后训练生成式 diffusion/flow。** 去噪状态是 MDP 状态，反向转移是策略，最终样本由奖励评价。CGDiT 属于这一类。
2. **相邻问题：奖励反向传播或偏好优化。** AlignProp/DRaFT、Diffusion-DPO 等经常与 RL 比较，但可能不使用 policy gradient；仍应作为强基线。
3. **反方向问题：用 diffusion 表示机器人/控制策略。** Diffusion Policy、Diffuser、DPPO 等是在 RL 环境里使用扩散策略，不等于“用 RL 优化生成式扩散模型”，本报告不把它们计入主清单。

## 2. 通用生成式扩散后训练

### 2.1 黑盒奖励与策略梯度

| 工作 | 状态 | 方法要点 | 对 CGDiT 的启示 |
|---|---|---|---|
| [DDPO](https://openreview.net/forum?id=w5QmbPyzfg) | ICLR 2024 | 将每个去噪步视为动作，用可计算的逐步反向转移似然做 policy gradient / importance sampling | CGDiT 最基本的 MDP 与 ratio 推导起点 |
| [DPOK](https://openreview.net/forum?id=8OTPepXzeh) | NeurIPS 2023 | 在线策略优化并相对预训练模型做 KL 正则 | 说明 reference KL 是防止分布漂移的标准组件 |
| [TDPO-R](https://openreview.net/forum?id=v2o9rRJcEv) | ICML 2024 | 时间信用分配并抑制 reward overoptimization | “按时间步分配奖励”已有先例，不能单独声称首创 |
| [Diverse Image Generation with RL](https://openaccess.thecvf.com/content/CVPR2024/html/Miao_Training_Diffusion_Models_Towards_Diverse_Image_Generation_with_Reinforcement_Learning_CVPR_2024_paper.html) | CVPR 2024 | 把多样性直接纳入 RL 目标 | 多样性奖励不是材料领域特有创新 |
| [B2-DiffuRL](https://openaccess.thecvf.com/content/CVPR2025/html/Hu_Towards_Better_Alignment_Training_Diffusion_Models_with_Reinforcement_Learning_Against_CVPR_2025_paper.html) | CVPR 2025 | 反向渐进训练与 branch-based sampling，缓解稀疏终端奖励 | 可用于减少晶体长轨迹的方差/成本 |
| [SPO](https://openaccess.thecvf.com/content/CVPR2025/html/Liang_Aesthetic_Post-Training_Diffusion_Models_from_Generic_Preferences_with_Step-by-step_Preference_CVPR_2025_paper.html) | CVPR 2025 | step-by-step preference optimization | 对中间步信用分配构成先例 |
| [Score as Action](https://proceedings.mlr.press/v267/zhao25f.html) | ICML 2025 | 连续时间随机控制，将 score 视为 action | 若 CGDiT 转向连续时间/flow，需要作为理论基线 |
| [Flow-GRPO](https://arxiv.org/abs/2505.05470) | 2025 预印本 | 对 flow matching 进行在线 GRPO | CGDiT 的 Diff2Flow 路线不能只靠“flow + GRPO”主张创新 |
| [DanceGRPO](https://arxiv.org/abs/2505.07818) | 2025 预印本 | 将 GRPO 扩展到视觉生成扩散后训练 | group-relative advantage 已成为通用方案 |
| [SEPO](https://openreview.net/forum?id=rXFzVRZsbt) | NeurIPS 2025 | 面向离散 diffusion 和不可微奖励的 policy gradient | 对 CGDiT 的 D3PM 元素通道最直接的通用算法先例 |
| [GLID²E](https://openreview.net/forum?id=29YgYt69Kl) | 2025 | 离散序列 diffusion 的梯度自由 RL、clipped likelihood 和 reward shaping | 说明“离散 diffusion + 黑盒奖励”并非空白 |

#### 2025–2026 快速演进的优化与稳定化分支

| 工作 | 状态 | 解决的问题 | 与本项目的关系 |
|---|---|---|---|
| [Adjoint Matching](https://openreview.net/forum?id=xQBRrtQM8u) | ICLR 2025 | 将 diffusion/flow 的奖励微调写成随机最优控制，并要求 memoryless noise schedule | flow/diffusion 统一的理论强基线 |
| [LOOP](https://openreview.net/forum?id=i8WJhKn455) | TMLR 2026 | 结合 leave-one-out baseline、PPO clipping 和 importance sampling，权衡 REINFORCE/PPO 成本 | 与 GRPO 的无 critic 组内估计直接可比 |
| [Advantage Weighted Matching](https://arxiv.org/abs/2509.25050) | 2025 预印本 | 用 advantage 加权原 score/flow-matching 预训练目标，降低 DDPO 式噪声目标方差 | 必须防止把“advantage 加权扩散 loss”误报为新方法 |
| [G²RPO](https://openreview.net/forum?id=Rs8qiI2GDb) | 2025–2026 OpenReview | 多粒度轨迹比较以获得更精细 reward/advantage | 与拟议分时间步信用分配相邻 |
| [DiPOD](https://openreview.net/forum?id=KopdtMhn0U) | ICLR 2026 Workshop | 诊断 ELBO—likelihood 与 proxy—true policy gradient 双重漂移，加入 on-policy ELBO/self-distillation | 提醒逐步转移 surrogate 可能与最终策略分布漂移 |
| [Efficient Adjoint Matching](https://arxiv.org/abs/2605.11480) | 2026 预印本 | 线性 base drift 和闭式 adjoint，降低 Adjoint Matching 成本 | 若直接 PG rollout 太贵，可作为优化替代路线 |
| [FIND](https://openreview.net/forum?id=bw3xRlHRtC) | 2024–2025 OpenReview | 不更新去噪器，而用 policy optimization 学初始噪声分布 | 一步 MDP/初始分布优化的低成本对照 |
| [Direct Noise Optimization](https://proceedings.mlr.press/v267/tang25h.html) | ICML 2025 | 推理时直接优化注入噪声，并讨论 OOD reward hacking | 非 RL 训练但属于 tuning-free 强基线 |

这些工作使得“PPO 对扩散模型不稳定”“长轨迹信用稀疏”本身也不能作为新的问题陈述；CGDiT 必须把一般困难落实到空间群商空间和混合动作上。

### 2.2 可微奖励直接反传：不是严格 RL，但必须比较

| 工作 | 方法要点 | 适用边界 |
|---|---|---|
| [ImageReward/ReFL](https://arxiv.org/abs/2304.05977) | 训练偏好奖励模型并从部分去噪步反传奖励 | 奖励必须可微或有可微代理 |
| [DRaFT](https://openreview.net/forum?id=1vmSEVL19f) | 通过整个或截断的采样计算图反传奖励，提出低方差变体 | 对短采样或可微代理常比黑盒 PG 更省样本 |
| [AlignProp](https://arxiv.org/abs/2310.03739) | 直接将奖励梯度反传到扩散模型 | 晶体松弛、凸包和 DFT 通常不可端到端微分，但 surrogate 可以 |

因此，如果性质代理模型在当前代码中是可微的，只比较 PPO/GRPO 而不比较 reward backprop，会夸大 RL 的必要性。

### 2.3 偏好学习：无显式在线 RL 的强替代方案

| 工作 | 状态 | 方法要点 |
|---|---|---|
| [Diffusion-DPO](https://openaccess.thecvf.com/content/CVPR2024/html/Wallace_Diffusion_Model_Alignment_Using_Direct_Preference_Optimization_CVPR_2024_paper.html) | CVPR 2024 | 用扩散 ELBO 改写 DPO，直接学习 winner/loser 偏好 |
| [D3PO](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Using_Human_Feedback_to_Fine-tune_Diffusion_Models_without_Any_Reward_CVPR_2024_paper.html) | CVPR 2024 | 无显式 reward model 的偏好后训练 |
| [D2-DPO](https://openreview.net/forum?id=qs9CTsC32h) | ICLR 2025 Workshop | 为 CTMC 离散扩散推导 DPO；masking diffusion 有简化闭式目标 |
| [RDPO](https://openaccess.thecvf.com/content/ICCV2025/html/Wu_Rethinking_DPO-style_Diffusion_Aligning_Frameworks_ICCV_2025_paper.html) | ICCV 2025 | 修正 DPO 式扩散对齐中的中间奖励排序偏差 |

在材料任务中，可以从同一批结构的代理/DFT 排序构造偏好对；这会成为在线 RL 的低风险基线。

## 3. 晶体与材料领域的直接工作

| 工作 | 状态 | 生成骨干/优化 | 奖励与主要贡献 | 对拟议 CGDiT-RL 的约束 |
|---|---|---|---|---|
| [RLFEF](https://doi.org/10.1016/j.neunet.2025.108146) | Neural Networks 2026 | 材料 diffusion + policy gradient | 形成能反馈；给出 MDP/策略梯度等价和对称性保持论证 | “形成能 + policy gradient + 保持晶体对称”已被覆盖 |
| [MatInvent](https://openreview.net/forum?id=Ovxfri7l5L) / [扩展预印本](https://arxiv.org/abs/2511.03112) | ICLR 2025 AI4Mat；2025 扩展版 | 等变晶体 diffusion；reward-weighted KL、experience replay、diversity filter | 单/多性质优化，强调较少性质调用 | “多目标、回放、多样性、样本效率”均有直接先例 |
| [DiffCSP++-ReFT for topological materials](https://www.nature.com/articles/s41467-026-73321-8) | Nature Communications 2026 | DiffCSP++ 原子类型 A、坐标 F、晶格 k 的因子化转移；importance ratio + PPO clipping | XBERT 拓扑奖励；DFT、能带、表面态和声子验证 | 与“原始晶体空间 factorized PPO”高度重叠；其元素使用连续 one-hot DDPM，而 CGDiT 是 D3PM，这一差异值得利用 |
| [Chemeleon2](https://www.nature.com/articles/s42256-026-01262-4) | Nature Machine Intelligence 2026 | VAE 连续潜空间 diffusion + GRPO | creativity、stability、diversity 多目标；性质引导 | 已覆盖“晶体 diffusion + GRPO + 多目标”；论文明确以潜空间规避 A/X/L 混合动作不稳定，正好构成 CGDiT 直接空间方法的对手 |
| [OMatG-IRL](https://openreview.net/forum?id=82lQk0jw0c) | ICLR 2026 AI4Mat Spotlight | flow/stochastic interpolant 的 velocity field；推理时 policy gradient | 能量目标、随机扰动、学习时间退火和加速 CSP | “无需 score 的 flow RL、时间调度和推理时 RL”已有先例 |
| [Synthesizability-Aware Materials Generation](https://openreview.net/pdf?id=o2TIUpiX0t) | ICML 2026 AI for Science Workshop | 预训练晶体 diffusion + 多目标 RL | 前驱体可得性、毒性/成本/材料类别、可合成性与目标性质 | “把可合成性纳入奖励”已有明确先例 |
| [MatFlow codebase](https://github.com/schwallergroup/MatFlow) | 2026 开源研究代码，尚无明确正式引文 | FlowMM + RL/Flow-GRPO 风格优化 | 面向目标晶体生成 | 若采用 Diff2Flow，需持续跟踪该项目的论文状态 |

### 容易误收但应单列的材料 RL

- [CRYSTAL](https://openreview.net/pdf/94d95333b625bc19463eca098ff60038d639d590.pdf)（ICML 2026 AI for Science Workshop）是晶体文本/LLM 的多目标 GRPO，**不是扩散模型**；但其乘法多目标聚合和 reward hacking 讨论是奖励设计基线。
- “Towards Generating Stable Materials via LLM RL”、MatMind、CrystalReasoner 等是 LLM/自回归 RL，不能证明晶体 diffusion 方法的新颖性，但会竞争最终材料发现指标。
- 传统 CSP 中的 RL、离线 RL 晶体设计和 CrystalGym 不以反向扩散为策略，属于任务基线而非同类方法。

## 4. 分子与生物结构的相邻证据

分子/序列领域已有离散 diffusion、3D 等变 diffusion 与多目标 RL 后训练，包括 SEPO、GLID²E、TR2-D2、面向稳定分子的 reward fine-tuning，以及不确定性感知多目标 3D de novo 设计。这些工作说明：

- 非可微 oracle 奖励、离散动作、经验回放、多目标和不确定性都不是材料领域天然空白；
- CGDiT 的贡献必须来自**晶体空间群商结构与三类异构转移的结合**，不能只来自通用 RL 技巧；
- 使用代理模型 ensemble 和多保真复核很重要，但更适合作为防 reward hacking 的科学可信性贡献。

## 5. 方法选择建议

| 条件 | 首选方法 | 理由 |
|---|---|---|
| 奖励完全黑盒、只有最终结构可评分 | DDPO/PPO 或 GRPO | 不需要奖励梯度；可用逐步转移概率 |
| 同一条件能并行生成多个候选，且不想训练 critic | GRPO | 组内标准化 advantage，工程上更简单 |
| 奖励代理可微且采样轨迹可保留计算图 | DRaFT/AlignProp 基线 | 通常比高方差 REINFORCE 更省性质调用 |
| 已有候选排序但在线 oracle 昂贵 | Diffusion-DPO / D2-DPO | 可离线利用 winner/loser 对 |
| 元素通道是真正 categorical D3PM | SEPO/D2-DPO 思路 + 精确 categorical log-prob | 不应把元素当作伪连续高斯 |
| 改成 flow matching | Flow-GRPO / OMatG-IRL | 需要针对 velocity field 的正确概率/扰动定义 |

对当前 CGDiT，推荐先实现可验证的直接空间 GRPO 数学闭环，同时保留 reward backprop、DPO 和 best-of-N 作为强基线；不要一开始把所有算法堆在一起。
