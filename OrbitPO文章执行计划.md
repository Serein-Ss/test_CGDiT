# CrystalPIRL / OrbitPO 文章执行计划

> 工作题目：**CrystalPIRL: Paired Policy-Improvement Verification for Reinforcement Fine-Tuning of Symmetry-Constrained Crystal Diffusion**
> 中文题目：**面向对称约束晶体扩散强化微调的成对策略改进验证**
> 方法命名：`CrystalPIRL` 为文章主方法；`OrbitPO` 为其对称约化概率基础模块
> 项目分支：`newton`
> 文档性质：预注册式执行计划；所有结果位置均为待实验占位，不得提前填写结论
> 当前状态：完整 8 个 MP20 生成 checkpoint 和 23 组结果保留归档；当前研究只启用 FE/BG、seed=42 奖励预测器及 base/FE/BG/FE+BG 四个生成基线（11 组正式评估）。不同 seed 的预测器体系尚未完整冻结，暂不使用 ensemble；阶段二至四核心代码已落地；FE/BG 最终独立评估器、原始 PIPO 对照和正式 GPU 强化学习实验仍待完成

---

## 1. 文章要回答的唯一核心问题

现有材料扩散强化学习通常根据当前 rollout 的终止奖励直接更新策略。即使平均奖励上升，长扩散轨迹、随机采样、代理预测器误差和晶格—坐标—元素混合动作仍可能使某次候选更新在真实策略层面退化。原始 PIPO 用滑动历史回顾上一轮更新，但不同轮次的样本噪声会混入策略差异。

本文检验：

> 将固定物理标度的通用材料奖励同时作为 PPO/GRPO 的优化信号和候选策略验收信号，在固定 probe 条件下让候选策略与已验证策略共享目标、空间群、原子数和全部连续/离散随机流，用成对性能差及 bootstrap 下置信界决定接受、衰减或拒绝更新，能否在相同生成与性质查询预算下同时降低坏更新率，并稳定提升材料性质且不牺牲有效性和多样性？

`OrbitPO` 解决这项检验所依赖的概率基础：在空间群约束下，元素、周期坐标和晶格必须按真实独立自由度计算联合 transition log-prob，避免错误 importance ratio 干扰策略更新及其验证。因此，文章的因果链为：

```text
对称约化且可重算的策略概率
→ 固定物理标度的目标/有效性/探索奖励
→ PPO/GRPO 候选更新
→ 固定 probe + 共同随机数成对比较原始奖励与安全指标
→ LCB 风险门控与 verified checkpoint
→ 更少坏更新、更高预算效率和更可靠的性质提升
```

文章不以“首次将 PPO/GRPO 用于材料扩散”、创造性—稳定性—多样性加权奖励、一般性的生成—打分—更新闭环或单纯使用独立轨道表示为主张。Chemeleon2 已实现潜空间 GRPO 与多目标材料奖励；原始 PIPO 已经提出跨轮策略改进反馈。本文所称“闭环通用材料生成奖励”专指固定标度原始奖励参与候选策略的成对验收、置信门控和回滚，而不是普通加权和。投稿前不使用“首个”表述。

---

## 2. 主方法与内部研究模块的边界

### 2.1 正文公开主线

正文按以下优先级组织：

1. **Closed-loop general material generation reward**：固定目标、容差、有效性和可选探索分量，区分原始奖励、策略优势与 probe 指标。
2. **Paired policy-improvement verification**：固定 probe，新旧策略共享随机流，使用 paired delta、bootstrap LCB 与 accept/attenuate/reject 决策维护 verified checkpoint。
3. **Algorithm-agnostic closed-loop evaluation**：分别在 PPO 和 GRPO 上比较无验证器、原始 PIPO 与 Paired-PIRL，判断改进是否跨算法成立，而不是只对 GRPO 有效。
4. **OrbitPO probability foundation**：元素采用轨道一致 D3PM，坐标采用代表点 wrapped likelihood，晶格采用空间群有效子空间 Gaussian；三者形成可重算的联合策略概率。

### 2.2 只做内部效果检查、不写入文章结果总结的内容

以下模块仍然完整实现、运行和保存结果，但不进入摘要、正文结论、主结果表和文章最终结果总结：

1. **H2：通道—时间信用分配**；
2. **predictor→MLFF→DFT 多保真闭环**。

它们单独形成内部技术报告，用于判断下一篇文章是否值得展开。除非后续明确改变投稿策略，否则不能根据正结果临时加入正文，也不能因负结果删除实验记录。

---

## 3. 可检验的文章贡献

### C1：闭环通用材料生成奖励与 Paired-PIRL 风险控制

本文首先定义可跨 batch、跨更新比较的通用材料原始奖励。性质分量由固定目标和容差映射到 [0,1]，联合目标默认使用瓶颈聚合；无效结构得到显式负惩罚。创造性、稳定性、组成多样性和结构多样性采用 Chemeleon2 启发的模块化接口，但第一轮只监控，触发预注册坍缩阈值后才作为增强消融启用。

GRPO 可以对同一 group 的原始奖励进行标准化以构造 advantage，但不得覆盖原始奖励。Paired-PIRL 始终比较未经过批次 min–max 的固定标度奖励。因此，本贡献的闭环是：

\[
R_{raw}\rightarrow A_{PPO/GRPO}\rightarrow\pi_{candidate}
\rightarrow\Delta R_{raw}^{paired}\rightarrow
\{\mathrm{accept,attenuate,reject}\}.
\]

对固定 probe 中每个条件与随机流 $j$，比较 verified old policy 和 candidate new policy：

\[
\Delta m_j=m(C_j^{new})-m(C_j^{old}).
\]

对主指标的成对差计算 bootstrap 下置信界，并同时检查 validity、uniqueness、novelty 等安全指标：

- `accept`：主指标 LCB 明确为正且安全约束满足，候选成为新的 verified checkpoint；
- `attenuate`：均值为正但证据不足，只保留预注册的小更新尺度；
- `reject`：主指标不改善或安全指标退化，回滚到最近 verified checkpoint。

训练 reward/probe evaluator 可以参与门控；最终独立 evaluator 不能参与训练、probe 决策或 checkpoint 选择，否则它不再是独立评价器。

验证证据包括坏更新率、错误接受/错误拒绝率、accepted-update efficiency、rollback 次数、probe 额外成本以及最终独立评价结果。

### C2：跨 PPO/GRPO 的算法普适性与原始 PIPO 对照

在完全相同的 OrbitPO 概率、初始 checkpoint、目标、seed 和预算下，对每种 RL 算法比较：

1. 无策略改进验证；
2. 原始 PIPO：滑动历史 anchor 与 retrospective modulation；
3. Paired-PIRL：固定 probe、共同随机数、成对差分、LCB 门控与回滚。

该设计回答两件不同的问题：Paired-PIRL 是否优于直接开放式更新；其收益是否超出原始 PIPO 已有的跨轮反馈。只有两种算法均显示方向一致且成本可接受的改善，才能支持“算法普适”；若只在 PPO 上有效，则结论必须收缩。

### C3：OrbitPO 对称约化的混合策略概率基础

#### C3.1 轨道一致的离散扩散

对每个已占据晶体学轨道的非对称单元代表点 $o\in\mathcal O$：

\[
\log p_\theta^A
=\sum_{o\in\mathcal O}
\log p_\theta(A_{o,t-1}\mid s_t).
\]

完整晶胞内的等价原子只接受广播结果，不重复采样、不重复计入损失或策略概率。

#### C3.2 测度一致的联合概率

\[
\log\pi_\theta(a_t\mid s_t)
=\log p_\theta^K+\log p_\theta^X+\log p_\theta^A,
\]

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
| MP20 奖励契约 | conf/rl/reward_closed_loop_mp20.yaml | 已建立 | 正式 target/scale 在 WP0 冻结 |

---

## 5. 五阶段执行路线

### 阶段一：冻结外部基线、训练奖励模型和独立评价器

状态：**FE/BG 的 seed=42 训练奖励模型和四个当前生成基线已冻结；多 seed predictor 暂不启用，FE/BG 最终独立评价器仍缺失。**

当前训练奖励模型统一登记在 `conf/rl/reward_models_mp20.yaml`。这些 checkpoint 可用于 RL reward 与固定 probe 验证，但不得同时作为文章最终独立评价器。

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

原始 PIPO 按论文使用滑动历史 anchor 和 retrospective modulation，不使用固定成对 probe。Paired-PIRL 对每次候选更新在固定 probe 上以相同 noise seed 比较新旧策略，输出：

- `accept`：LCB 明确为正，更新尺度 1；
- `attenuate`：均值为正但 LCB 不充分，使用预注册尺度，默认 0.25；
- `reject`：主指标不改善或安全指标退化，尺度 0，并回滚/保留最近 verified checkpoint。

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

建议正文题目：

> CrystalPIRL: Paired Policy-Improvement Verification for Reinforcement Fine-Tuning of Symmetry-Constrained Crystal Diffusion

题目突出“策略更新是否真的改进”这一核心问题。`OrbitPO` 作为方法组件在摘要和 Methods 中定义，不再占据标题中心。

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

正文按问题组织：

- **RQ1：Paired-PIRL 是否比开放式更新和原始 PIPO 更可靠？** 报告坏更新率、错误接受率、paired delta/LCB、rollback 和额外成本。
- **RQ2：这种作用是否跨 PPO/GRPO 成立？** 报告 2×3 交互、效应量与等效区间；允许得出算法相关或 GRPO 已足够稳定的负结论。
- **RQ3：OrbitPO 是否提供正确且无系统规模偏置的策略概率？** 报告 old=current、数值密度对照、ratio/KL 对原子数、空间群有效维数和轨道多重度的分层结果。
- **RQ4：在相同预算下是否提升当前目标性质且不牺牲生成质量？** 报告 FE、BG、FE+BG 联合命中率、validity、uniqueness、novelty 和成本。
- **RQ5：闭环奖励是否优于只用于梯度的开放式奖励？** 比较固定标度闭环、Chemeleon2 风格批次归一化、是否加入多样性组件，以及它们的坏更新率、目标产率和模式坍缩指标。

Results 不展示 H2 与 predictor→MLFF→DFT 内部实验。

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

包含完整伪代码、原始 PIPO 实现、所有超参数、checkpoint/rollback 规则、probe 构造、seed、wrapped Gaussian 截断误差、D3PM 后验、确定性末步、单元测试、失败 run、排除标准、硬件环境和 Conda 环境清单哈希。

H2 与多保真效果保留为项目内部报告，不放入投稿附件，除非后续明确授权。

### 8.13 Data and Code Availability

准备数据来源与许可、split、生成结构与完整筛选日志、reward/probe/final evaluator checkpoint、`environment.yml` 与环境导出清单、训练/恢复/评估命令、seed、配置和公开范围。

---

## 9. 主图和主表规划

### Figure 1：CrystalPIRL 方法总览

以“候选更新→固定成对 probe→LCB 门控→accept/attenuate/reject→verified checkpoint”为视觉中心；OrbitPO 的轨道级 D3PM、晶格子空间和周期坐标概率作为候选更新的支撑模块。

### Figure 2：策略改进验证证据

展示共享随机流的新旧样本对、paired delta 分布、bootstrap LCB、坏更新/错误接受率以及原始 PIPO 与 Paired-PIRL 的差异。

### Figure 3：OrbitPO 概率正确性

展示 old=current ratio、数值/解析密度对照，以及 ratio/KL 对原子数、轨道多重度和空间群有效维数的关系。

### Figure 4：算法普适性与通用性质结果

展示 PPO/GRPO × 三种验证机制的 2×3 交互、FE/BG 及联合命中率、生成质量与收益—额外成本。

### Table 1：主结果

报告三个或更多 seed 的通用性质、质量、坏更新率和成本，禁止只报告最优值。

### Table 2：概率消融

报告轨道计数、晶格子空间和 wrapped likelihood 的独立贡献。

### Table 3：2×3 策略更新验证比较

报告无验证、原始 PIPO、Paired-PIRL 在 PPO/GRPO 下的最终 reward、独立 hit rate、坏更新率、错误接受率、方差、probe 成本和等效区间。

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
- Paired-PIRL 结论没有在 PPO/GRPO 下同时对照无验证与原始 PIPO；
- 负结果或失败 run 被选择性删除。

---

## 12. 上传基线后的立即执行顺序

1. 校验上传文件、checkpoint、数据 split 和结果 schema；
2. 复现阶段一基线的一个固定 seed；
3. 用旧 checkpoint 执行轨道级采样兼容评估；
4. 决定短程微调还是正式重训轨道一致模型；
5. 用真实 decoder 跑 transition replay 梯度 smoke test；
6. 先运行 FE 的 PPO、GRPO 128–512 rollout；
7. 实现并验证原始 PIPO 基线，完成 FE 的 PPO/GRPO × 无验证/原始 PIPO/Paired-PIRL 2×3 小实验；
8. 通过 Gate B 后，用相同接口和预算复跑 BG 的 2×3 实验；
9. FE 与 BG 单目标均通过后，再运行 FE+BG 联合约束；
10. 独立运行 H2 与多保真内部实验；
11. 冻结三个或更多 RL seed 结果后生成文章图表；predictor seed 仍固定为 42；
12. 按第 8 节逐段撰写，所有数字从冻结表格自动引用。

---

## 13. 当前不能提前填写的内容

以下问题必须等待真实实验：

- Paired-PIRL 是否相对无验证和原始 PIPO 降低坏更新率；
- 这种额外收益是否同时适用于 PPO 与 GRPO；
- OrbitPO 是否优于 ambient/full-cell 概率实现；
- GRPO 是否已经足够稳定而无需更新验证；
- H2 是否降低梯度方差或只是增加 oracle 成本；
- predictor→MLFF→DFT 是否提高真实命中率；
- 轨道级 D3PM 是否需要从头重训；
- 最终可支持的期刊级创新强度。

本计划的作用是提前固定这些问题的检验方式，避免根据结果事后改变指标、阈值或文章主线。
