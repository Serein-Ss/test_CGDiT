# CrystalPIRL 文章执行计划

> 方法型工作题目：**CrystalPIRL: Paired Policy-Improvement Verification for Reinforcement Fine-Tuning of Symmetry-Constrained Crystal Diffusion**
> Nature 风格候选题目：**Verified reinforcement learning for crystal structure generation**
> 中文题目：**面向晶体结构生成的可验证强化学习**
> 方法命名：`CrystalPIRL` 为文章主方法；`OrbitPO` 为其对称约化概率基础模块
> 项目分支：`newton`
> 更新日期：2026-08-23
> 文档性质：预注册式执行计划；所有结果位置均为待实验占位，不得提前填写结论
> 当前状态：Gate 2、四组 FE pilot、Gate 3b 单步尺度诊断和 Gate 3c 双锚点 GPU 诊断均已完成。作业 667914 验证了 candidate-vs-current、candidate-vs-base、LCB、attenuate 重新验收和拒绝路径；GRPO 第 2 步证明仅通过绝对基线但未通过局部基线的候选会被拒绝。当前在线安全门仍只有 validity，尚缺 stability、uniqueness、novelty、diversity、holdout rollback、多 seed、原始 PIPO 和最终独立 evaluator，因此所有 Gate 3c checkpoint 仍只能标记为 diagnostic-only。

---

## 1. 文章要回答的唯一核心问题

现有材料扩散强化学习通常根据当前 rollout 的终止奖励直接更新策略。即使平均奖励上升，长扩散轨迹、随机采样、代理预测器误差和晶格—坐标—元素混合动作仍可能使某次候选更新在真实策略层面退化。原始 PIPO 用滑动历史回顾上一轮更新，但不同轮次的样本噪声会混入策略差异。

本文检验：

> 将固定物理标度的通用材料奖励同时作为 PPO/GRPO 的优化信号和候选策略验收信号，让候选策略既与当前 verified policy 做局部成对比较，也与永久冻结的原始基础策略做绝对比较；只有性质改善和稳定性、有效性、唯一性、多样性非退化约束同时通过 LCB 门控时才更新 checkpoint，能否在相同预算下降低坏更新率，并避免“相对当前状态改善但仍劣于原始模型”的假改进？

`OrbitPO` 解决这项检验所依赖的概率基础：在空间群约束下，元素、周期坐标和晶格必须按真实独立自由度计算联合 transition log-prob，避免错误 importance ratio 干扰策略更新及其验证。因此，文章的因果链为：

```text
对称约化且可重算的策略概率
→ 固定物理标度的目标/有效性/探索奖励
→ PPO/GRPO 候选更新
→ 候选 vs 当前 verified policy：局部改进
→ 候选 vs 永久冻结 base policy：绝对改进与生成质量非退化
→ 固定/holdout probe + 共同随机数 + LCB 风险门控
→ 更少坏更新、更高预算效率和更可靠的性质提升
```

文章不以“首次将 PPO/GRPO 用于材料扩散”、创造性—稳定性—多样性加权奖励、一般性的生成—打分—更新闭环或单纯使用独立轨道表示为主张。Chemeleon2 已实现潜空间 GRPO 与多目标材料奖励；原始 PIPO 已经提出跨轮策略改进反馈。本文所称“闭环通用材料生成奖励”专指固定标度原始奖励参与候选策略的成对验收、置信门控和回滚，而不是普通加权和。投稿前不使用“首个”表述。

---

## 2. 主方法与内部研究模块的边界

### 2.1 正文公开主线

正文按以下优先级组织：

1. **Closed-loop general material generation reward**：固定目标、容差、有效性和可选探索分量，区分原始奖励、策略优势与 probe 指标。
2. **Dual-anchor policy-improvement verification**：固定 probe 上比较 candidate vs current verified policy 与 candidate vs frozen base policy，使用 paired delta、bootstrap LCB 和生成质量非退化约束维护 verified checkpoint。
3. **Algorithm-agnostic closed-loop evaluation**：分别在 PPO 和 GRPO 上比较无验证器、原始 PIPO 与 Paired-PIRL，判断改进是否跨算法成立，而不是只对 GRPO 有效。
4. **OrbitPO probability foundation**：元素采用轨道一致 D3PM，坐标采用代表点 wrapped likelihood，晶格采用空间群有效子空间 Gaussian；三者形成可重算的联合策略概率。

### 2.2 只做内部效果检查、不写入文章结果总结的内容

以下模块仍然完整实现、运行和保存结果，但不进入摘要、正文结论、主结果表和文章最终结果总结：

1. **H2：通道—时间信用分配**；
2. **predictor→MLFF→DFT 多保真闭环**。

它们单独形成内部技术报告，用于判断下一篇文章是否值得展开。除非后续明确改变投稿策略，否则不能根据正结果临时加入正文，也不能因负结果删除实验记录。

Nature 级文章仍需要对冻结后的最终候选做**独立终点验证**。该验证可以使用独立 MLFF/DFT 弛豫和性质复核，但不把多保真预算编排写成 CrystalPIRL 的在线模块，也不允许其结果回流到 checkpoint 选择。换言之，内部“多保真闭环”继续排除，预注册的盲法物理复核则作为证据质量控制保留。

---

## 3. 可检验的文章贡献

### C1：闭环通用材料生成奖励与 Paired-PIRL 风险控制

本文首先定义可跨 batch、跨更新比较的通用材料原始奖励。性质分量由固定目标和容差映射到 [0,1]，联合目标默认使用瓶颈聚合；无效结构得到显式负惩罚。创造性、稳定性、组成多样性和结构多样性采用 Chemeleon2 启发的模块化接口，但第一轮只监控，触发预注册坍缩阈值后才作为增强消融启用。

GRPO 可以对同一 group 的原始奖励进行标准化以构造 advantage，但不得覆盖原始奖励。Paired-PIRL 始终比较未经过批次 min–max 的固定标度奖励。因此，本贡献的闭环是：

$$
R_{raw}\rightarrow A_{PPO/GRPO}\rightarrow\pi_{candidate}
\rightarrow(\Delta R_{local}^{paired},\Delta R_{absolute}^{paired})\rightarrow
\{\mathrm{accept,attenuate,reject}\}.
$$

对固定 probe 中每个条件与随机流 $j$，同时计算：

$$
\Delta m_j^{local}=m(C_j^{candidate})-m(C_j^{verified}),
$$

$$
\Delta m_j^{absolute}=m(C_j^{candidate})-m(C_j^{base}).
$$

接受条件要求局部与绝对目标性质 LCB 均为正，并且 candidate 相对 frozen base 的 validity、stability、uniqueness、novelty 和 diversity 不越过预注册抗坍塌界。通用质量指标不要求相对 current 逐步提高，允许在预注册容忍范围内下降；它们的作用是阻止目标性质优化以模式坍塌为代价：

- `accept`：双重主目标 LCB 与全部安全约束满足，候选成为新的 verified checkpoint；
- `attenuate`：候选方向有信号但完整步长未通过；按预注册尺度缩小后重新完整评估，不能直接接受；
- `reject`：主指标不改善或任一安全指标退化，保留最近 verified checkpoint；
- `rollback`：holdout probe 或周期性独立审计发现绝对约束失效，恢复最近通过审计的 checkpoint。

训练 reward/probe evaluator 可以参与门控；最终独立 evaluator 不能参与训练、probe 决策或 checkpoint 选择，否则它不再是独立评价器。

训练 rollout 的拓扑 prompt 只从 MP-20 train split 抽取；固定 probe、holdout 和训练末配对验证使用 validation split 内互不重叠的索引；MP-20 test split 在强化学习期间完全保留，只用于冻结模型后的独立结果评估。probe 与 holdout 轨迹只用于验收，禁止参与梯度更新。

验证证据包括坏更新率、错误接受/错误拒绝率、accepted-update efficiency、rollback 次数、probe 额外成本以及最终独立评价结果。

#### C1.1 当前 Gate 3b 证据边界

2026-08-22 的作业 667909 已完成单种子单步尺度诊断：

| 算法 | 诊断中正 LCB 的尺度 | 最佳诊断奖励增量 | 解释 |
|---|---|---:|---|
| PPO | 0.5 | +0.0760 | 完整步长置信不足，半步存在正向信号 |
| GRPO | 0.25、1.0 | +0.0698（scale=1.0） | GRPO 的可接受尺度与 PPO 不同 |

两次跨进程完整轨迹哈希完全一致，证明当前随机流和诊断入口可复现。该实验只有 seed=42、probe seed=4242、batch=16，尚未接入双重基线和完整安全指标，因此只能作为方法工程证据，不能填写为文章的稳定性、有效性、多样性非退化结论。

#### C1.2 当前 Gate 3c 双锚点证据边界

2026-08-23 的作业 667914 在 RTX 3090 上完成，主作业及四个步骤均以退出码 0 结束，16 项预检测试全部通过。两种算法各执行两次候选更新：

| 算法与步骤 | local mean / LCB | absolute mean / LCB | 最终决策 | 解释 |
|---|---:|---:|---|---|
| PPO step 0 | 初始 +0.0250 / -0.0766；缩放复验 +0.0671 / -0.00124 | 同左 | reject | 初始要求 scale=0.25，但重新 rollout 后 LCB 仍未大于 0，未错误接受 |
| PPO step 1 | -0.0604 / -0.1802 | -0.0604 / -0.1802 | reject | 候选同时劣于 current 与 base |
| GRPO step 0 | +0.0698 / +0.000257 | +0.0698 / +0.000257 | accept | 初始时 current 等于 base，候选通过两种奖励门控 |
| GRPO step 1 | -0.0631 / -0.1818 | +0.00671 / +0.0000168 | reject | 候选仍略优于 base，但显著劣于已接受的 current；局部锚点阻止回退 |

这次实验验证了双锚点判定、attenuate 必须复验以及“任一主门失败即拒绝”的代码路径。它尚未证明 absolute anchor 能阻止“优于已退化 current 但低于 base”的反向案例，也未证明最终材料质量提升。所有记录的 `safety_audit.formal_complete` 均为 `false`，因为只评估了 validity；因此即使 GRPO step 0 的奖励决策为 `accept`，`verified_checkpoint_update` 仍为 `false`，生成的 checkpoint 只是诊断候选。

### C2：跨 PPO/GRPO 的算法普适性与原始 PIPO 对照

在完全相同的 OrbitPO 概率、初始 checkpoint、目标、seed 和预算下，对每种 RL 算法比较：

1. 无策略改进验证；
2. 原始 PIPO：滑动历史 anchor 与 retrospective modulation；
3. Paired-PIRL：固定 probe、共同随机数、成对差分、LCB 门控与回滚。

该设计回答两件不同的问题：Paired-PIRL 是否优于直接开放式更新；其收益是否超出原始 PIPO 已有的跨轮反馈。只有两种算法均显示方向一致且成本可接受的改善，才能支持“算法普适”；若只在 PPO 上有效，则结论必须收缩。

### C3：OrbitPO 对称约化的混合策略概率基础

#### C3.1 轨道一致的离散扩散

对每个已占据晶体学轨道的非对称单元代表点 $o\in\mathcal O$：

$$
\log p_\theta^A
=\sum_{o\in\mathcal O}
\log p_\theta(A_{o,t-1}\mid s_t).
$$

完整晶胞内的等价原子只接受广播结果，不重复采样、不重复计入损失或策略概率。

#### C3.2 测度一致的联合概率

$$
\log\pi_\theta(a_t\mid s_t)
=\log p_\theta^K+\log p_\theta^X+\log p_\theta^A,
$$

其中晶格只在空间群允许的有效子空间计算 Gaussian，坐标在轨道代表点的三维环面上计算有限镜像 wrapped Gaussian，元素在代表点上计算 D3PM 后验 categorical probability。原始 joint log-prob 用于 importance ratio；每自由度归一化只作为诊断量。

#### 术语与主张边界

“Wyckoff position”是空间群中的位置类型；一个具体晶体中实际被元素占据并由对称操作展开的实例才是本文处理的 crystallographic orbit。正文优先使用“已占据的独立晶体学轨道集合”与“非对称单元代表点”，不把二者笼统写成“独立 Wyckoff 轨道集合”。

当前 CGDiT 数据提供轨道展开操作，但没有完整 site-stabilizer 元数据。因此坐标模块只能主张 **orbit-representative symmetry-reduced torus measure**，不能声称已经实现所有特殊 Wyckoff 位置的完整低维商空间测度。若以后补充 site-symmetry stabilizer，必须增加切空间秩和规范化测试后才能升级该主张。

---

## 4. 当前代码状态与文件映射

| 功能 | 文件 | 状态 | 下一验证 |
|---|---|---|---|
| 轨道代表点选择/广播 | `cgdit/rl/symmetry_quotient.py` | 已实现 | 用真实 MP20 batch 检查 anchor 偏移 |
| 轨道级前向 D3PM | `cgdit/pl_modules/diffusion.py` | 已实现 | 重新训练/微调后的验证损失 |
| 轨道级 D3PM 损失 | `cgdit/pl_modules/training_utils/diffusion_loss.py` | 已实现 | 与完整晶胞损失做多重度消融 |
| `sample_rl()` 轨迹 | `cgdit/pl_modules/diffusion.py` | 已实现 | GPU 长轨迹显存和吞吐 |
| 轨迹数据契约 | `cgdit/rl/trajectory.py` | 已实现 | checkpoint/resume 序列化 |
| OrbitPO 对称约化 transition log-prob | `cgdit/rl/transition_logprob.py` | 已实现 | 大/小噪声 wrapped sum 收敛 |
| 可微概率重放 | `cgdit/rl/rollout.py` | 已实现 | 真实 decoder 梯度有限性 |
| PPO/GRPO | `cgdit/rl/objectives.py` | 已实现 | 真实 rollout smoke training |
| 统一训练目标入口 | `cgdit/rl/trainer.py` | 已实现 | 接入上传后的 reward evaluator |
| Paired-PIRL 配对验收 | `cgdit/rl/policy_improvement.py` | 已实现 | 固定 probe 的真实更新验收 |
| 原始 PIPO 对照 | 尚无对应实现 | 待实现 | 滑动历史 anchor 与 retrospective modulation |
| 共同随机数采样 | `cgdit/rl/paired_probe.py` | 已实现 | 新旧真实 checkpoint 配对 |
| H2 信用分配 | `cgdit/rl/channel_time_credit.py` | 已实现 | 中间状态与反事实 reward 生成 |
| 通用性质奖励 | `cgdit/rl/rewards.py` | 已实现 | 用 evaluator 标定 target/tolerance |
| 多保真预算编排 | `cgdit/rl/multifidelity.py` | 已实现 | 接入真实 predictor/MLFF/DFT adapter |

当前单元测试覆盖的是概率与接口正确性，不等价于已完成 GPU 训练、MLFF 松弛或 DFT 计算。

新增奖励代码状态：

| 功能 | 文件 | 状态 | 边界 |
|---|---|---|---|
| 固定标度性质与联合奖励 | cgdit/rl/rewards.py | 已实现 | 真实 predictor adapter 待接入 |
| 显式无效结构惩罚 | cgdit/rl/rewards.py | 已实现 | validity 判据需在实验契约冻结 |
| 创造性、稳定性、leave-one-out MMD | cgdit/rl/rewards.py | 已实现张量接口 | AMD/特征计算需接真实结构 |
| raw reward 配对验收 | cgdit/rl/policy_improvement.py | 已实现 | 真实固定 probe 待 GPU 验证 |
| MP20 奖励契约 | conf/rl/components/rewards/contracts/mp20.yaml | 已建立 | 正式 target/scale 在 WP0 冻结 |

---

## 5. 五阶段执行路线

### 阶段一：冻结外部基线、训练奖励模型和独立评价器

状态：**FE/BG 的 seed=42 训练奖励模型和四个当前生成基线已冻结；多 seed predictor 暂不启用，FE/BG 最终独立评价器仍缺失。**

当前训练奖励模型统一登记在 `conf/rl/components/rewards/registries/mp20.yaml`。这些 checkpoint 可用于 RL reward 与固定 probe 验证，但不得同时作为文章最终独立评价器。

### 需要上传的生成基线

当前 FE/BG 主线启用：

- `mp20_base`；
- `mp20_fe`；
- `mp20_bg`；
- `mp20_fe_bg`；

以下结果只保留归档，不进入当前实验矩阵：`mp20_eh`、`mp20_fe_eh`、`mp20_bg_eh`、`mp20_fe_bg_eh`。

当前实验还必须固定：

- 固定 RL seed 列表；
- 每个模型相同样本数、NFE、guidance scale 和 GPU 时间；
- 生成结构原始 `.pt` 或 `.csv/.cif`，不能只上传汇总均值。

当前性质执行顺序固定为 FE 单目标→BG 单目标→FE+BG 联合目标。Ehull 及含 Ehull 条件结果不进入当前 reward、probe、消融、主表或结论；历史文件保留但不调用。

### 需要上传的性质预测器

当前两个性质分别需要训练 reward predictor 和独立 evaluator：

- formation energy per atom；
- band gap；
- checkpoint；
- train/validation/test split 标识；
- MAE、RMSE、校准曲线数据；
- 训练 seed 与模型配置；
- 当前冻结 seed=42 单模型；未来启用 ensemble 时再保存每个成员的独立预测。

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

### 阶段四：OrbitPO 对称约化概率与 PPO/GRPO

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

### 阶段五：Paired-PIRL、原始 PIPO 对照、通用性质扩展和内部模块

### 5.1 PPO/GRPO × 三种验证机制的 2×3 核心比较

固定 OrbitPO 概率、初始 checkpoint 和总预算，只改变 RL 算法与更新验证机制：

| 编号 | RL 算法 | 更新验证 | 目的 |
|---|---|---|---|
| P0 | PPO | 无 | PPO 开放式更新基线 |
| P1 | PPO | 原始 PIPO | 检验滑动历史回顾反馈 |
| P2 | PPO | Paired-PIRL | 检验成对置信门控的额外价值 |
| G0 | GRPO | 无 | GRPO 开放式更新基线 |
| G1 | GRPO | 原始 PIPO | 检验 PIPO 对组相对优势的作用 |
| G2 | GRPO | Paired-PIRL | 检验 GRPO 是否仍需要更新验证 |

原始 PIPO 按论文使用滑动历史 anchor 和 retrospective modulation，不使用固定成对 probe。CrystalPIRL 对每次候选更新在固定 probe 上以相同 noise seed 同时比较 candidate-vs-current 和 candidate-vs-base，输出：

- `accept`：局部与绝对主目标 LCB 明确为正，且全部安全约束满足；
- `attenuate`：候选方向有信号但证据不足，产生预注册缩放候选并重新 rollout/验收；
- `reject`：主指标不改善或安全指标退化，尺度 0，并回滚/保留最近 verified checkpoint。
- `rollback`：holdout 审计发现累计策略低于冻结基础策略或安全指标越界，恢复最近安全 checkpoint。

probe 门控只能使用训练 reward 或单独的 verifier；最终独立 evaluator 只在冻结策略后使用。

### 5.2 Paired-PIRL 是否具有额外价值和算法普适性的判据

不能只看六条曲线谁最高。需要预先定义：

- 坏更新率：候选更新后 probe 主指标显著下降的比例；
- 错误接受率：被门控接受但在独立复测中退化的比例；
- accepted-update efficiency：每次接受更新带来的 reward 增量；
- 样本效率：每 1000 次 predictor query 的 target-hit 增量；
- 稳定性：不同 seed 的最终 reward 方差；
- 成本：历史反馈或 paired probe 的额外 NFE、查询和 GPU-hour；
- 等效界值：带验证方法与无验证方法之间的预注册等效区间。

解释规则：

1. P2/G2 相对 P0/G0 均降低坏更新率：支持成对验证跨算法有效；
2. P2/G2 还相对 P1/G1 改善：支持本文机制超出原始 PIPO 的额外贡献；
3. 只相对无验证基线有效、但不优于原始 PIPO：只能说明策略改进反馈有用，不能支持新验证机制；
4. PPO 改善而 GRPO 在等效界内：说明 Paired-PIRL 主要帮助 PPO，不能声称普适；
5. GRPO 仍明显改善：说明组相对 advantage 不能替代策略更新验收；
6. 平均 reward 提升但 validity/novelty 下降，或只在一个 seed 上改善：不得形成主结论。

### 5.3 通用性质任务

当前只按以下顺序执行：

1. formation energy 最小化；
2. band gap 目标值或预注册区间；
3. formation energy + band gap 联合约束。

奖励使用有界函数并对 invalid 结构硬门控。当前只使用冻结的 seed=42 单预测器，不使用未完整冻结的 ensemble uncertainty；正式阈值由阶段一冻结 evaluator 的测试分布确定，不在看到 RL 结果后修改。Ehull 延后到下一研究阶段。

闭环奖励固定为三层：

1. raw_reward：固定目标/容差与显式 invalid penalty，不做批次 min–max；
2. policy_advantage：PPO baseline 或 GRPO 组内标准化；
3. probe_metrics：原始奖励为主指标，validity、uniqueness、novelty 和 diversity 为安全指标。

Chemeleon2 对照消融只在 FE/BG 基础奖励通过后运行：

- property + validity；
- property + validity + composition diversity；
- property + validity + composition diversity + structure diversity；
- Chemeleon2 风格批次归一化奖励复现基线。

最后一项只作为复现对照，不用于 CrystalPIRL 验收。

### 5.4 H2 内部实验，不进入文章结果总结

实现路线：

1. 将反向时间划分为 4–10 个固定桶；
2. 对每个桶的中间去噪状态估计性质；
3. 计算相邻桶 reward increment；
4. 在共同噪声下替换 lattice/coordinate/atom 单通道，得到反事实贡献；
5. 将终止 advantage 按 `[time bucket, channel]` 的贡献绝对值归一分配；
6. 各通道独立 clipping，但不得称为精确 joint ratio。

内部报告至少保存统一 advantage、时间分桶、通道—时间联合三种方案的梯度方差、命中率、贡献热图、额外查询成本和 reward exploitation 风险。

### 5.5 多保真内部实验，不进入文章结果总结

执行 predictor→MLFF→DFT 四级验证，记录每级进入/退出数量、失败样本、代理与高保真结果的一致性、结构松弛变化、选择概率和每次 DFT 调用的真实命中数。高 reward exploitation quota 与高 uncertainty exploration quota 必须分开。

当前代码只实现预算编排、ensemble 统计、校准奖励和选择概率权重。真实 MLFF/DFT evaluator 仍需在势函数、赝势、泛函、收敛参数和任务系统明确后接入；未接入前不得表述为已完成物理闭环。

---

## 6. 正式实验矩阵

### 6.1 主表方法

| 编号 | 方法 | 轨道级元素 | OrbitPO 概率 | 更新验证 |
|---|---|---:|---:|---|
| B0 | 原始条件 CGDiT | 否/旧实现 | 否 | 无 |
| B1 | 轨道一致 CGDiT | 是 | 不适用 | 无 |
| B2 | Best-of-N | 是 | 不适用 | 无 |
| B3 | ambient/full-cell PPO | 是 | 否 | 无 |
| B4 | ambient/full-cell GRPO | 是 | 否 | 无 |
| M1 | OrbitPO-PPO | 是 | 是 | 无 |
| M2 | OrbitPO-PPO-PIPO | 是 | 是 | 原始 PIPO |
| M3 | CrystalPIRL-PPO | 是 | 是 | Paired-PIRL |
| M4 | OrbitPO-GRPO | 是 | 是 | 无 |
| M5 | OrbitPO-GRPO-PIPO | 是 | 是 | 原始 PIPO |
| M6 | CrystalPIRL-GRPO | 是 | 是 | Paired-PIRL |

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

按投稿层级准备两个版本：

1. **方法型期刊工作题目**：*CrystalPIRL: Paired Policy-Improvement Verification for Reinforcement Fine-Tuning of Symmetry-Constrained Crystal Diffusion*；
2. **Nature 风格候选题目**：*Verified reinforcement learning for crystal structure generation*。

Nature 主刊标题通常避免冒号、缩写和过长技术串，因此正文标题优先使用第二个版本；`CrystalPIRL` 在 summary paragraph 和 Fig. 1 首次定义。`OrbitPO` 作为概率基础只在摘要、方法和消融中定义，不再占据标题中心。最终标题必须等待 Fig. 2 和 Fig. 6 的证据强度确定，若独立物理验证不足，则改用更窄的“policy-improvement verification”表述。

### 8.2 Abstract

摘要按五句话组织：

1. **背景**：强化学习可直接优化生成材料性质，但晶体扩散更新受长轨迹、随机 rollout、代理误差和混合动作影响。
2. **缺口**：现有开放式更新和滑动历史 PIPO 都不能在共享随机条件下直接识别候选策略是否优于已验证策略。
3. **方法**：提出固定标度闭环通用材料生成奖励与 CrystalPIRL，以同一原始奖励连接 PPO/GRPO 优化和固定 probe 的成对验收，并以 OrbitPO 提供对称约化联合概率。
4. **实验**：在相同预算下进行 PPO/GRPO × 无验证器/原始 PIPO/Paired-PIRL 的 2×3 比较，并覆盖多种通用性质、空间群和结构规模。
5. **结论**：只填写被三个或更多 seed、预注册判据和最终独立 evaluator 支持的结果。

摘要不写 H2、多保真闭环、高居里温度或迁移率。

### 8.3 Introduction

建议 5 段：

1. 材料逆向设计的巨大结构空间、昂贵 oracle，以及条件生成与后训练优化的互补关系；
2. PPO/GRPO 已用于图像、分子和材料扩散，生成—打分—更新闭环本身不是创新；
3. 核心缺口是候选策略更新未经同条件、同随机流验证，滑动历史统计也会混入跨批次采样噪声；
4. 提出 CrystalPIRL，并说明可信比较需要 OrbitPO 对三种晶体动作给出可重算且对称一致的联合概率；
5. 按 C1–C3 列贡献，每项对应一个方法小节、关键实验和消融。

### 8.4 Related Work

建议分三小节：

1. symmetry-aware crystal diffusion and independent-site representations；
2. reinforcement fine-tuning for diffusion and material generation；
3. policy-improvement feedback, PIPO and paired evaluation。

必须明确：已有晶体 ReFT 已涉及代表晶位、周期坐标和联合概率，因此本文不把“使用独立轨道/商空间”单独声明为首创；方法差异集中在轨道级离散 D3PM、可验证概率实现以及 Paired-PIRL 更新门控。投稿前重新检索，不使用未经核实的“首个”。

### 8.5 Problem Formulation

定义晶体状态、predictor/corrector 子转移、终止 reward、已占据晶体学轨道集合、空间群晶格 active mask，以及 old/candidate/verified/reference 四类策略角色。区分：

- 原始 joint log-prob 与诊断性 per-DOF 归一化量；
- 训练 reward/probe verifier 与最终独立 evaluator；
- 原始 PIPO 的跨轮历史比较与 Paired-PIRL 的同 probe 成对比较。

### 8.6 Methods

#### 8.6.1 CrystalPIRL paired policy-improvement verification

给出 fixed probe、common random numbers、paired delta、bootstrap LCB、accept/attenuate/reject、verified checkpoint 和 rollback 伪代码。说明候选更新由 PPO/GRPO 产生，门控不改写 importance ratio。

#### 8.6.2 Closed-loop general material generation reward

定义 target/maximize/minimize/range 四类性质映射、显式 validity gate、FE+BG 瓶颈聚合、可选 creativity/stability/leave-one-out MMD 组件，以及 raw_reward、policy_advantage、probe_metrics 三层契约。说明为何组内标准化只用于梯度，而固定 probe 使用未归一化原始奖励；Chemeleon2 风格批次 min–max 只作为对照。

#### 8.6.3 Original PIPO comparison

严格实现滑动历史 anchor、标准化 improvement signal 与 retrospective modulation，列出与本文固定 probe 机制的差异，避免把原始 PIPO 当作“无验证”基线。

#### 8.6.4 Orbit-consistent discrete diffusion

写清代表点前向加噪、轨道级损失、反向后验与广播，并用“已占据晶体学轨道/非对称单元代表点”而不是含混的“独立 Wyckoff 轨道”。

#### 8.6.5 Rank-aware lattice transition

说明 active mask、affine constraint 和有效维数；给出有效子空间 Gaussian，解释为何不在退化六维环境空间使用普通密度。

#### 8.6.6 Periodic coordinate transition

描述 corrector/predictor 概率、有限镜像 wrapped Gaussian 和代表点计数；主动声明当前没有完整 site-stabilizer 切空间。

#### 8.6.7 Mixed joint policy measure and PPO/GRPO

给出三通道联合 log-prob、importance ratio、KL 诊断、确定性末步处理，以及 PPO 外部 advantage 和 GRPO 组内标准化的公平实现。

#### 8.6.8 Computational complexity

分别报告 rollout、概率重放、wrapped image sum、原始 PIPO 历史维护和 paired probe 的时间、显存及查询复杂度。

### 8.7 Experimental Setup

完整写明 MP20 划分、冻结 checkpoint、reward predictor/probe verifier/最终独立 evaluator 的隔离，FE/BG 目标及阈值、优化超参数、diffusion 步数、timestep 抽样、三个或更多 RL seed、GPU-hour/NFE/查询预算和统计方法。RL 实验 seed 与 predictor 训练 seed 分开记录。主实验是六组合，不是四组合。

固定 probe 不得在观察主实验结果后更换。最终独立 evaluator 只评估冻结策略，不参与 accept/reject 或 checkpoint 选择。

### 8.8 Results

正文按主图证据顺序组织：

- **RQ1 / Fig. 2：Paired-PIRL 是否比开放式更新和原始 PIPO 更可靠？** 报告局部/绝对 paired delta、LCB、坏更新率、错误接受率、rollback、accepted-update efficiency 和额外成本。
- **RQ2 / Fig. 2：这种作用是否跨 PPO/GRPO 成立？** 报告 $2\times3$ 交互、效应量与等效区间；允许得出算法相关或 GRPO 已足够稳定的负结论。
- **RQ3 / Fig. 3：相比 CFG、Best-of-N 和开放式 RL，是否更准确地控制 FE/BG 且不牺牲生成质量？** 报告完整性质分布、target MAE、valid-target yield、validity、stability、uniqueness、novelty 和 diversity。
- **RQ4 / Fig. 4：RL 改变了哪些材料搜索空间？** 报告全局生成质量、coverage、结构嵌入、元素/组成/空间群分布、Pareto 前沿和代表性失败结构。
- **RQ5 / Fig. 5：OrbitPO 和双锚点门控是否分别必要？** 报告 old=current、数值密度对照、ratio/KL 规模偏置和移除各 Gate 组件的消融。
- **RQ6 / Fig. 6：性质提升能否通过独立 evaluator 和结构弛豫复核？** 报告 predictor disagreement、弛豫前后配对变化、失败比例和盲法抽样的候选结构。

Results 不展示 H2，也不把 predictor→MLFF→DFT 多保真编排声明为方法贡献；独立 MLFF/DFT 只作为冻结结果的盲法终点验证。

### 8.9 Discussion

讨论：为什么更新验证不同于一般 reward feedback；共同随机数减少的是比较方差而不是提供无偏策略梯度；Paired-PIRL 与 PPO/GRPO 的互补或冗余关系；OrbitPO 的适用范围；以及向分子、表面、缺陷和其他约束生成空间迁移的条件。

### 8.10 Limitations

必须主动说明：

- 当前坐标策略没有完整特殊 Wyckoff stabilizer 切空间；
- probe verifier 与最终独立 evaluator 都可能有 OOD 风险；
- probe 会增加生成与性质查询成本；
- 长扩散轨迹重放成本高；
- 没有 DFT 正文主证据时，结论限于独立代理评价；
- MP20 不能代表全部材料体系；
- 本文不直接解决高居里温度和高迁移率设计。

### 8.11 Conclusion

只回答三件事：

1. Paired-PIRL 相对无验证和原始 PIPO 是否降低坏更新并提高预算效率；
2. 结论是否同时适用于 PPO 与 GRPO；
3. OrbitPO 是否为该比较提供了经验证、无明显规模偏置的策略概率基础。

### 8.12 Reproducibility Appendix

包含完整伪代码、原始 PIPO 实现、所有超参数、checkpoint/rollback 规则、probe 构造、seed、wrapped Gaussian 截断误差、D3PM 后验、确定性末步、单元测试、失败 run、排除标准、硬件环境和 `uv.lock` 哈希。

H2 与多保真效果保留为项目内部报告，不放入投稿附件，除非后续明确授权。

### 8.13 Data and Code Availability

准备数据来源与许可、split、生成结构与完整筛选日志、reward/probe/final evaluator checkpoint、`uv.lock`、训练/恢复/评估命令、seed、配置和公开范围。

---

## 9. 主图、扩展数据与证据顺序

### 9.1 可以参考 Chemeleon2，但不能照搬

Chemeleon2 的正文图按以下证据顺序展开：

| Chemeleon2 图号 | 核心内容 | 在本文中的可借鉴逻辑 |
|---|---|---|
| Fig. 1 | 潜空间扩散强化学习框架与 creativity、stability、diversity 奖励 | 先让读者理解“生成—打分—更新”的闭环和奖励语义 |
| Fig. 2 | GRPO 机制、GRPO 与 REINFORCE 对照、去除 diversity reward 的消融 | 紧接着说明策略更新怎样发生，以及为什么不会稳定地陷入局部最优 |
| Fig. 3 | 10,000 个样本的全局生成基准 | 在方法可信之后再展示最终生成质量 |
| Fig. 4 | 结构嵌入空间、novelty–stability 前沿和代表结构 | 说明 RL 改变了哪些材料空间，而不是只提高一个汇总数 |
| Fig. 5 | 元素与组成空间变化 | 检查探索偏置、化学空间迁移和潜在 reward hacking |
| Fig. 6 | RL 与 CFG/LoRA-CFG 的性质引导比较 | 最后证明方法能完成实际性质目标，同时保留材料可行性 |

本文应借鉴的是“框架 → 更新机制 → 主结果 → 搜索空间 → 性质应用”的证据递进，而不是复刻其版式或奖励。Chemeleon2 的主创新是潜空间 GRPO 和多目标材料奖励；本文必须把最独特的 **双锚点策略改进验证** 放在 Fig. 2 的中心，不能让 CrystalPIRL 看起来只是另一个 GRPO 奖励配置。

参考资料：

- [Chemeleon2 正式论文](https://www.nature.com/articles/s42256-026-01262-4)
- [Chemeleon2 奖励实现说明](https://github.com/hspark1212/chemeleon2/tree/main)
- [Nature 文章格式与展示项要求](https://www.nature.com/nature/for-authors/formatting-guide)
- [Nature 科研绘图规范](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/)

### 9.2 本文的整篇视觉论证

核心结论必须由以下顺序支撑：

```text
Fig. 1  方法是什么，奖励和更新如何连接
→ Fig. 2  候选更新是否真的、可靠地优于当前策略和冻结基础策略
→ Fig. 3  相比 CFG 与无门控 RL，性质控制是否更强且生成质量不退化
→ Fig. 4  最终生成集合在稳定性、有效性、唯一性、多样性和搜索空间上是否整体更好
→ Fig. 5  OrbitPO 与 CrystalPIRL 各模块是否确实产生预期作用
→ Fig. 6  独立评估和结构弛豫后，性质改善是否仍然成立
```

前三张图是文章的最小核心；Fig. 4–6 是将“工程上可运行”提升为“Nature 级可信证据”所需的补充主证据。若缺少 Fig. 6 所需的独立物理验证，文章主张必须限制为“代理模型评价下的可靠强化微调”，不能写成已发现真实性能优异的新材料。

### 9.3 Figure 1：CrystalPIRL 框架、OrbitPO 概率与闭环奖励

**一句话结论：** CrystalPIRL 将对称一致的晶体扩散策略更新与固定物理标度奖励、双锚点成对验证和回滚连接成可审计闭环。

**图型：** schematic-led composite；方法总览占主要面积，奖励和判定函数作为支撑面板。

**建议面板：**

| 面板 | 内容 | 必须回答的问题 | 所需数据 |
|---|---|---|---|
| a | 从冻结基础策略 $\pi_0$、当前 verified policy $\pi_k$ 到候选策略 $\pi'$ 的完整闭环 | 三种策略角色是什么，checkpoint 何时改变 | 无定量数据，使用代码真实流程 |
| b | OrbitPO 三通道策略：轨道级元素 D3PM、周期坐标 wrapped likelihood、晶格有效子空间 Gaussian | 为什么 importance ratio 与对称独立自由度一致 | 一条真实轨迹的通道、时间步和自由度示例 |
| c | 通用材料奖励：性质、validity、stability、uniqueness、diversity | 奖励如何从结构变成固定标度分数 | 冻结后的 reward 曲线、目标和容差 |
| d | candidate-vs-current 与 candidate-vs-base 的共同随机数 paired probe | 为什么既需要局部改进，也需要绝对改进 | 同一 probe 的三策略配对示意 |
| e | bootstrap LCB、accept/attenuate/reject/rollback | 更新怎样被接受、缩小、拒绝或回滚 | 一个 accept、一个 attenuate、一个 reject 的真实记录 |
| f | raw reward、PPO/GRPO advantage 与 probe metric 的分层 | 为什么 GRPO 组内标准化不会污染跨更新比较 | 奖励数据契约示意 |

**核心公式：**

$$
\Delta R_j^{\mathrm{local}}
=
R(C_j^{\pi'})-R(C_j^{\pi_k}),
$$

$$
\Delta R_j^{\mathrm{absolute}}
=
R(C_j^{\pi'})-R(C_j^{\pi_0}).
$$

正式接受条件为：

$$
\operatorname{LCB}\!\left(\Delta R^{\mathrm{local}}\right)>0,
\qquad
\operatorname{LCB}\!\left(\Delta R^{\mathrm{absolute}}\right)>0,
$$

并且所有预注册安全指标相对冻结基础策略不越过非退化容忍界。`attenuate` 只生成缩放候选，缩放后必须重新 rollout 和重新验收。

**图审风险：**

- 不要把奖励加权和画成文章的唯一创新；
- 不要把 OrbitPO 画成完整特殊 Wyckoff 商空间，当前只能主张轨道代表点的对称约化测度；
- 所有箭头必须对应真实代码路径，不能画尚未接通的 MLFF/DFT 在线反馈；
- 训练 evaluator、probe verifier 和最终独立 evaluator 必须用不同视觉符号区分。

### 9.4 Figure 2：CrystalPIRL 相对现有强化更新的可靠性提升

**一句话结论：** 在相同候选更新、预算和随机流下，CrystalPIRL 是否比开放式更新和原始 PIPO 更少接受坏更新，并阻止累计策略低于冻结基础模型。

这是全文最重要的结果图，应先证明“强化学习更新可信”，再展示最终性质指标。

**公平比较矩阵：**

| RL 算法 | 开放式更新 | 原始 PIPO | CrystalPIRL |
|---|---|---|---|
| PPO | P0 | P1 | P2 |
| GRPO | G0 | G1 | G2 |

**建议面板：**

| 面板 | 内容 | 主指标 |
|---|---|---|
| a | $2\times3$ 实验设计与相同预算说明 | seed、rollout、NFE、predictor 查询数、GPU-hour |
| b | 每次候选更新的 local delta 与 absolute delta，按 LCB 着色 | paired mean、95% LCB、accept/attenuate/reject |
| c | “只相对当前提高但低于 $\pi_0$”的反例轨迹 | 假改进率、绝对基线跌破次数 |
| d | 开放式更新、PIPO、CrystalPIRL 的坏更新率和错误接受率 | bad-update rate、false-accept rate、false-reject rate |
| e | 随训练预算变化的累计绝对 reward/性质收益 | 相对 $\pi_0$ 的累计增量及置信区间 |
| f | accepted-update efficiency 与额外成本 | 每次接受更新的真实收益、probe 查询和 GPU 时间 |
| g | PPO 与 GRPO 的交互效应 | 方法主效应、算法主效应、交互效应与单 seed 点 |

**可以直接与哪些工作比较：**

1. **最有因果解释力的对比**是同一 CGDiT/OrbitPO 主干上的 P0/P1/P2 与 G0/G1/G2，因为只有这组比较能隔离 CrystalPIRL 验证层的贡献。
2. **Chemeleon2**可以作为直接外部 RL 基线，但必须在同一 MP-20 划分、相同生成数量、相同 as-generated/relaxed 口径和同一独立评价流水线上重新运行。其论文表格中的数字只能作为文献背景，不能与本项目数字直接做显著性比较。
3. **CFG、Best-of-N 和无 RL 基础模型**不是策略验证方法，但可以回答“是否需要强化微调”以及“收益是否只是增加采样/筛选预算”。
4. 其他晶体 RL 方法只有在公开代码、checkpoint 和目标任务可对齐时才进入主表；否则放在 Related Work，不把不可复现实验拼入主图。

**必须冻结的统计定义：**
$$
\mathrm{BadUpdateRate}
=
\frac{N_{\mathrm{accepted},\,\Delta R_{\mathrm{holdout}}\le 0}}
{N_{\mathrm{accepted}}},
$$

$$
\mathrm{AcceptedUpdateEfficiency}
=
\frac{\sum_{u\in\mathcal A}
\max(0,\Delta R_{u}^{\mathrm{holdout}})}
{\mathrm{GPUHours}+\lambda N_{\mathrm{oracle\ queries}}}.
$$

$\lambda$ 必须在看正式结果之前固定，或者分别报告 GPU 成本和查询成本，不使用事后选择的合并权重。

### 9.5 Figure 3：相对 CFG 的形成能与带隙条件生成

**一句话结论：** CrystalPIRL 是否在提高形成能或带隙目标命中率的同时，保持基础模型的稳定性、有效性、唯一性和多样性。

**对比方法：**

- frozen unconditional CGDiT base；
- 已有 FE-CFG；
- 已有 BG-CFG；
- Best-of-N；
- PPO/GRPO 开放式强化微调；
- 原始 PIPO；
- CrystalPIRL-PPO；
- CrystalPIRL-GRPO。

每个目标任务只比较使用同一目标定义的方法；不能把 FE 模型和 BG 模型的命中率混在同一统计总体中。

**建议面板：**

| 面板 | 内容 | 推荐表达 |
|---|---|---|
| a | CFG、开放式 RL 与 CrystalPIRL 的条件生成路径 | 简化方法示意，颜色与 Fig. 1 一致 |
| b | 形成能的完整预测分布 | KDE/直方图加原始散点密度、目标区间阴影、基础数据分布 |
| c | 带隙的完整预测分布 | 与 b 使用相同统计和视觉语法 |
| d | target MAE、tolerance hit rate 与 valid-target yield | 带 95% CI 的点图，不只画命中率 |
| e | 性质控制与生成质量的 Pareto 图 | 横轴 target hit，纵轴质量复合指标或分别展示 |
| f | validity、stability、uniqueness、novelty、diversity 的非退化差值 | 相对 $\pi_0$ 的 paired difference/forest plot |
| g | 目标区间内的代表结构 | 结构图、组成、预测值、独立评价值和筛选来源 |

必须报告：

- 全体生成样本的性质分布，而不是只报告筛选后的命中结构；
- 无效结构计入分母后的 `valid-target yield`；
- 每个 seed 的独立点；
- 相同目标下的 Wasserstein distance 或 calibration error；
- 条件目标内结构的 uniqueness、novelty 和结构多样性；
- 生成前后元素数、原子数和空间群分布变化。

### 9.6 Figure 4：整体生成质量与材料搜索空间

**一句话结论：** 性质引导不是以模式坍缩或生成低质量结构为代价，且 RL 确实改变了材料搜索分布。

该图吸收 Chemeleon2 Fig. 3–5 的证据逻辑，但不需要机械拆成三张。

**建议面板：**

| 面板 | 内容 | 数据要求 |
|---|---|---|
| a | overall validity、stability、uniqueness、novelty、coverage/diversity | 每个冻结模型建议至少 10,000 个样本；资源不足时预注册更小样本并给 CI |
| b | novelty–stability 或 target-yield–diversity Pareto front | 所有方法使用同一独立 evaluator |
| c | 训练集、base、CFG、CrystalPIRL 的 UMAP/t-SNE 嵌入 | 降维只用于展示；结论由定量 coverage/MMD 支撑 |
| d | 元素周期表热图和 n-ary composition 分布 | 检查元素偏置与过度多元化 |
| e | 空间群、原子数、晶格参数和密度分布 | 检查是否偏离基础材料分布 |
| f | 四类代表结构及其最近邻 | 高 reward/高 novelty/高 stability/失败案例都要展示 |

不能只画“漂亮的成功结构”。必须同时保留：

- invalid/high-strain 结构；
- 重复结构与局部几何微扰；
- 极端多元化导致的 reward hacking；
- predictor 与独立 evaluator 分歧最大的结构。

### 9.7 Figure 5：OrbitPO 正确性与 CrystalPIRL 机制消融

**一句话结论：** 结果改善来自对称一致的概率和双锚点风险门控，而不是数值尺度、更多查询或偶然 seed。

**建议面板：**

| 面板 | 内容 |
|---|---|
| a | old=current 时 joint ratio、channel ratio 与 approximate KL |
| b | 数值积分/枚举与解析 transition probability 对照 |
| c | ratio/KL 对原子数、轨道多重度、空间群有效晶格维数的分层关系 |
| d | 去除 absolute anchor、共同随机数、LCB、安全约束、attenuation recheck 或 rollback 的消融 |
| e | ambient/full-cell probability 与 OrbitPO 的训练稳定性和规模偏置 |
| f | probe size、置信水平、安全容忍界和 bootstrap 次数的敏感性 |
| g | resume 前后相同 seed 的轨迹、optimizer 和 verified checkpoint 一致性 |

必须至少包含以下反例：

1. candidate 优于 current 但低于 base；
2. reward 提升但 validity/stability/uniqueness/diversity 越界；
3. 完整步长不通过而缩放候选复验通过；
4. fixed probe 通过但 holdout probe 失败并触发 rollback。

### 9.8 Figure 6：独立评价、结构弛豫与物理可信度

**一句话结论：** CrystalPIRL 的性质提升在独立模型和结构弛豫后仍然存在，而不是 reward predictor 的代理偏差。

这张图对 Nature 主刊级主张非常重要。它不需要把 `predictor→MLFF→DFT` 写成文章方法创新，但必须把独立验证作为结果可信度证据。

**建议面板：**

| 面板 | 内容 | 最低要求 |
|---|---|---|
| a | reward predictor、独立 evaluator、MLFF/DFT 的隔离协议 | 模型、训练数据、用途和信息隔离表 |
| b | reward predictor 与独立 evaluator 的性质相关性和误差 | 测试集与生成集分别报告，含 OOD 校准 |
| c | as-generated 与 relaxed 结构的 FE/BG 对比 | 配对散点、误差、结构变化和失败比例 |
| d | 弛豫前后 target hit 与质量指标 | 同一结构配对置信区间 |
| e | 独立选出的代表候选 | CIF、组成、空间群、性质、弛豫能量和最近邻 |
| f | predictor disagreement 与失败模式 | 展示最严重的代理偏差，而不是隐藏 |

若计算资源有限，先对预注册且与模型无关的结构子集做 DFT，例如每个方法从相同分层规则抽取相同数量的结构。不能只选择 reward 最高的 CrystalPIRL 候选。

### 9.9 Nature 主刊的展示项取舍

Nature Article 通常使用 4–6 个正文 figures/tables；复合图越大，可用正文篇幅越少。因此推荐两种版本：

**版本 A：Nature 主刊上限版本**

- Fig. 1：CrystalPIRL + OrbitPO + reward；
- Fig. 2：策略改进可靠性；
- Fig. 3：FE/BG vs CFG；
- Fig. 4：整体生成质量与搜索空间；
- Fig. 5：概率正确性与机制消融；
- Fig. 6：独立评价与物理验证；
- 主表全部移到 Extended Data/Source Data。

**版本 B：Nature Machine Intelligence / Nature Communications 紧凑版本**

- Fig. 1：框架与奖励；
- Fig. 2：策略可靠性与消融；
- Fig. 3：FE/BG vs CFG；
- Fig. 4：整体质量、搜索空间与代表结构；
- Fig. 5：独立评价与弛豫验证；
- OrbitPO 详细数值正确性放 Extended Data。

当前文章如果强调 CrystalPIRL，Fig. 2 必须比 Fig. 5 更靠前、更醒目；OrbitPO 是可信更新的概率基础，不应重新占据文章标题和第一结果位置。

### 9.10 Extended Data 建议顺序

Nature 主刊通常最多允许十个 Extended Data 展示项。建议预留：

1. **Extended Data Fig. 1**：基础模型、CFG、reward predictor 与独立 evaluator 的冻结检查；
2. **Extended Data Fig. 2**：完整 reward 映射、容差和权重敏感性；
3. **Extended Data Fig. 3**：六组合的所有 seed 训练曲线；
4. **Extended Data Fig. 4**：probe size、bootstrap、LCB 和安全容忍阈值敏感性；
5. **Extended Data Fig. 5**：OrbitPO 三通道数值概率测试；
6. **Extended Data Fig. 6**：原子数、空间群和轨道多重度分层结果；
7. **Extended Data Fig. 7**：FE/BG 全部目标、全部 seed 的分布与校准；
8. **Extended Data Fig. 8**：元素、组成、空间群、密度和结构嵌入的完整统计；
9. **Extended Data Fig. 9**：失败 run、reward hacking、模式坍缩和 rollback 案例；
10. **Extended Data Fig. 10**：MLFF/DFT 弛豫细节、额外候选结构和筛选审计。

大规模逐结构结果、完整超参数和原始表格放 Source Data 或数据仓库，不把几十张小图堆入 Supplementary Information。

### 9.11 完成这些图还缺哪些数据

| 证据块 | Nature 级最低证据 | 当前状态 | 完成动作 |
|---|---|---|---|
| CrystalPIRL 双锚点门控 | local/absolute LCB、attenuate 复验、rollback、holdout | Gate 3c 已验证双锚点、复验与拒绝路径；尚无 holdout/rollback | 接入完整安全门后完成多 seed、probe $\ge128$ 的正式 FE 试验 |
| 完整安全门 | validity、stability、uniqueness、novelty、composition/structure diversity | 当前在线门控只有 validity | 接入真实结构指标并冻结容忍界 |
| PPO/GRPO 普适性 | $2\times3$、至少 3 个 seed、相同预算 | 未完成 | 完成 P0/P1/P2/G0/G1/G2 |
| CFG 对比 | base、CFG、Best-of-N、RL 的同目标分布和质量 | 已有部分 CFG 生成结果 | 冻结结构集合并统一用独立 evaluator 重评 |
| 全局生成质量 | 每个冻结模型大样本、统一 mSUN/coverage/分布统计 | 待统一 | 预注册样本量，建议每模型 10,000 |
| 外部 RL 基线 | Chemeleon2 等方法在同一评估流水线重跑 | 未完成 | 只选择代码、权重和任务均可对齐的方法 |
| 独立性质评价 | 与 reward predictor 数据和用途隔离的 evaluator | 未冻结 | 上传/确认 checkpoint、测试误差和 OOD 校准 |
| 稳定性 | 独立 MLFF 的 $E_{\mathrm{hull}}$ 或一致代理 | 未接入正式门控 | 冻结模型、参考相图和阈值 |
| 物理验证 | 分层、等预算的 MLFF/DFT relaxation 与性质复核 | 尚无正文证据 | 预注册选择规则和结构数后计算 |
| 失败模式 | invalid、collapse、reward hacking、predictor disagreement | 尚未系统汇总 | 所有失败 run 和排除原因进入冻结表 |
| 成本 | rollout、reward、probe、优化、独立评价分项 | 日志已有部分字段 | 统一 GPU-hour、NFE、查询次数 |
| 可复现性 | config、seed、checkpoint、CIF、Source Data、代码版本 | 部分具备 | 冻结 release、环境锁和数据清单 |

### 9.12 每张主图的 Source Data 契约

每张图必须对应一个不可变的长表，而不是绘图脚本临时拼接：

| 图 | Source Data 主键 | 必要字段 |
|---|---|---|
| Fig. 1 | `run_id, update_id, probe_id, policy_role` | reward 分量、paired delta、LCB、决策、checkpoint |
| Fig. 2 | `algorithm, verifier, seed, update_id` | local/absolute delta、holdout delta、错误类型、成本 |
| Fig. 3 | `task, method, seed, structure_id` | 目标、预测值、独立值、valid、hit、结构来源 |
| Fig. 4 | `method, seed, structure_id` | validity、stability、unique、novel、embedding、组成统计 |
| Fig. 5 | `ablation, seed, transition_id` | channel log-prob、ratio、KL、DOF、空间群、轨道多重度 |
| Fig. 6 | `method, structure_id, fidelity` | 弛豫前后结构、能量、FE、BG、失败原因 |

绘图代码只允许读取冻结表；阈值、样本排除和分层抽样必须在生成 Source Data 时完成并记录。

### 9.13 Nature/Typora 图文规范

- 本文档使用 Typora 原生 Markdown：行内公式用 `$...$`，独立公式用 `$$...$$`；
- 不在表格单元格中放多行公式；
- 图标题不写入绘图区，完整标题放在 figure legend；
- 每张主图应有一个 hero panel，而不是平均分配的 dashboard；
- 正文图最终建议使用 183 mm 双栏宽，最大高度 170 mm；
- 最终尺寸下普通文字保持 5–7 pt，面板字母使用 8 pt 加粗小写；
- 图例必须报告精确 $n$、seed/replicate 定义、中心与离散度、置信区间或误差条定义；
- 图例控制在期刊要求内，初稿阶段也要与图放在同页；
- 最终主图保留 editable PDF/SVG 和 Source Data；PNG 只作为项目预览，不作为接收后的最终矢量主图；
- 颜色映射跨图固定：base 用灰色，CFG 用蓝色，开放式 RL 用橙色，PIPO 用紫色，CrystalPIRL 用绿色；PPO/GRPO 用点形或线型区分，避免再增加两套颜色；
- 所有主图必须显示单 seed 点或结构级原始点，不能只给均值柱状图。

### 9.14 实际绘图执行顺序

1. 冻结 Fig. 2 的更新级统计表，因为它决定 CrystalPIRL 核心主张是否成立；
2. 冻结 Fig. 3 的 base/CFG/RL 结构集合和独立性质预测；
3. 冻结 Fig. 4 的全局生成质量与结构嵌入；
4. 完成 Fig. 5 的概率消融和 Gate 反例；
5. 完成 Fig. 6 的独立评价与弛豫验证；
6. 最后绘制 Fig. 1，使框架图严格反映已经完成并验证的代码，而不是预想系统；
7. 所有主图完成后再决定哪些面板进入 Extended Data，不在结果出来前按“好看程度”删面板。

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
- Paired-PIRL 结论没有在 PPO/GRPO 下同时对照无验证与原始 PIPO；
- 负结果或失败 run 被选择性删除。

---

## 12. 上传基线后的立即执行顺序

1. [已完成工程验证] 将 frozen base $\pi_0$、current verified $\pi_k$ 和 candidate $\pi'$ 三种角色写入运行状态与日志 schema；
2. [已完成工程验证] 对每个候选同时计算 candidate-vs-current 与 candidate-vs-base 的 paired delta 和 LCB；
3. 把 validity、stability、uniqueness、diversity 接入硬约束，并冻结各自容忍阈值；
4. [已完成工程验证] 实现 attenuate 后重新 rollout/评估；Gate 3c 中 PPO 缩放复验未通过并被正确拒绝；
5. 建立不参与门控的 holdout probe，完成 Gate 3c 的多 seed、扩大 probe 正式 FE 扩展；
6. 在双重门控通过后重跑 P0/P2/G0/G2 多轮 pilot，报告坏更新率、错误接受率和累计绝对收益；
7. 实现原始 PIPO 独立代码路径，完成 PPO/GRPO × 无验证/原始 PIPO/CrystalPIRL 的 2×3 FE 比较；
8. 通过 FE Gate 后，用相同接口和预算依次扩展 BG、FE+BG 和 Ab initio empirical；
9. 新 orbit-corrected Base 上传后冻结正式 WP0/WP1 基线并复跑三 seed 主实验；
10. 最终独立 evaluator 只评估冻结策略；MLFF/DFT 作为内部物理复核，不参与回看式调参；
11. 独立运行 H2 与多保真内部实验；
12. 所有数字从冻结结果表自动进入文章，不根据结果事后更改门控指标或阈值。

---

## 13. 当前不能提前填写的内容

以下问题必须等待真实实验：

- Paired-PIRL 是否相对无验证和原始 PIPO 降低坏更新率；
- 双重基线门控是否能阻止累计退化，并同时保持稳定性、有效性、唯一性和多样性；
- 这种额外收益是否同时适用于 PPO 与 GRPO；
- OrbitPO 是否优于 ambient/full-cell 概率实现；
- GRPO 是否已经足够稳定而无需更新验证；
- H2 是否降低梯度方差或只是增加 oracle 成本；
- predictor→MLFF→DFT 是否提高真实命中率；
- 轨道级 D3PM 是否需要从头重训；
- 最终可支持的期刊级创新强度。

本计划的作用是提前固定这些问题的检验方式，避免根据结果事后改变指标、阈值或文章主线。
