# Paper Card：Policy Improvement Reinforcement Learning 及其对 CGDiT 高温磁性半导体生成的启示

> Source coverage: Full paper
> Extraction confidence: High
> Locator mode: page-grounded
> Primary analytical lens: methods
> Secondary analytical lens: None
> Context verification: Targeted external check
> Card completeness: Complete relative to supplied source

输入范围：arXiv v5 全文（25 页）、图 1–5、表 1–7、公式 1–22，以及针对晶体生成、磁性材料、居里温度和载流子迁移率的定向外部检索。论文是通用 LLM/Agent 强化学习方法论文，不是材料论文；项目映射与研究方案均单独标注为 `[Analysis]` 或 `[Hypothesis]`。

论文类型判定：方法论文；主要证据是算法定义、局部理论分析、跨任务实验和消融实验。

术语决策表：

| 原文或项目术语 | 本卡统一表述 | 决策 |
|---|---|---|
| Policy Improvement Reinforcement Learning | 策略改进强化学习（PIRL） | 保留作者术语 |
| Policy Improvement Policy Optimization | 策略改进策略优化（PIPO） | 保留作者术语 |
| policy-improvement feedback | 策略改进反馈 | 指跨迭代性能变化，不等同于单样本奖励 |
| Curie temperature | 居里温度 \(T_C\) | 材料目标术语 |
| carrier mobility | 载流子迁移率 \(\mu\) | 必须指定载流子、温度、浓度和方向 |
| magnetic space group | 磁空间群（MSG） | 与项目当前普通空间群严格区分 |
| energy above hull | 凸包上能量 \(E_\mathrm{hull}\) | 0 K 热力学稳定性代理 |
| Magnetic-Symmetry Pareto-Improvement CGDiT | 磁对称—Pareto 政策改进 CGDiT（MSPI-CGDiT） | `[Hypothesis]` 本卡提出的候选方向，不是论文作者术语 |

## 01 基本信息

| 字段 | 内容 |
|---|---|
| 标题 | *Policy Improvement Reinforcement Learning* |
| 作者 | Huaiyang Wang、Xiaojie Li、Xiaohan Wang、Zhixia Zhang、Xiaodong Lu、Zixuan Huang、Jiajun Chai、Guojun Yin、Deqing Wang、Haoyi Zhou、Yaodong Yang、Jianxin Li、Yikun Ban |
| 机构 | 北京航空航天大学、北京大学、美团 `[Paper: PDF p. 1]` |
| 载体与年份 | arXiv 预印本，2026；v1 于 2026-04-01，所读 v5 修订于 2026-07-07 `[Paper: Metadata]` |
| 标识符 | arXiv:2604.00860；DOI: 10.48550/arXiv.2604.00860 |
| 论文类型 | 强化学习方法论文 |
| 领域 | LLM/Agent 强化学习后训练、PPO/GRPO、自蒸馏 |
| 关键词 | PIRL、PIPO、policy improvement feedback、closed-loop RL、GRPO |
| 代码/项目页 | 作者首页列出 https://jacckma.github.io/pirl/ `[Paper: PDF p. 1]` |
| 数据 | MATH、TACO、ToolRL/RLLA、SciKnowEval 等任务数据；不是材料结构数据 `[Paper: PDF pp. 6–8, 16–18]` |
| 阅读日期 | 2026-08-16 |
| 对用户方向的定位 | `[Analysis]` 可作为 CGDiT 强化学习闭环的“跨迭代验收器”，但不能直接充当材料表示、物理 oracle 或主要创新 |

## 02 一句话总结

该论文针对强化学习更新后不验证策略是否真的变好的问题，用滑动历史性能锚点构造跨迭代改进信号，并在 PPO、组相对优化和自蒸馏的 LLM 任务中通过一次回顾式更新取得平均性能增益，但没有验证其对扩散模型、晶体结构或多目标物理设计的有效性。`[Paper: PDF pp. 1, 4–8]`

## 03 研究问题

- 具体问题：传统 RL 后训练由当前批次的奖励、优势或蒸馏目标驱动，更新完成后并不显式检查新策略是否优于旧策略。`[Paper: PDF pp. 1–3]`
- 重要性：有限采样、生成随机性和反馈噪声可能使局部目标提升与真实策略性能提升脱钩。`[Paper: PDF pp. 1–2]`
- 既有方法不足：PPO、GRPO、GSPO、DAPO 和 SDPO 的信用信号不同，但作者认为它们共享“开环更新”结构。`[Paper: PDF pp. 2–3]`
- 精确问题：能否用跨策略迭代的可观测性能增量调制已有局部学习信号，使训练形成可纠偏的闭环？

## 04 研究背景与发展路径

| 阶段 | 代表方法 | 优点 | 局限 | 本文位置 |
|---|---|---|---|---|
| 近端策略优化 | PPO | critic 和 GAE 提供细粒度优势 | 训练成本高；更新后不显式验收 | PIPO 的一个基算法 |
| 组相对策略优化 | GRPO、GSPO、DAPO | 组内标准化、可去 critic | 稀疏奖励边界下可能敏感；仍是局部开环更新 | PIPO 的主要实验族 |
| 自蒸馏优化 | SDPO | 用反馈条件教师形成稠密 token 信号 | 局部蒸馏信号不直接证明最终策略更好 | PIPO 的另一实验族 |
| 跨迭代闭环 | PIRL/PIPO | 显式比较当前表现与历史锚点 | 信号受批次组成、噪声、额外计算影响 | 作者声称的贡献 |

上述发展路径主要是论文作者对 RL 后训练领域的组织方式，而非本卡完成的系统综述。`[Paper: PDF pp. 2–3]`

## 05 论文识别的核心痛点

| 痛点 | 表现 | 原因或作者解释 | 论文证据 |
|---|---|---|---|
| 更新不验收 | 优化器只看当前批次信号，不检查新策略是否更优 | 开环策略优化缺少跨迭代反馈 | 图 1 `[Paper: PDF p. 1, Figure 1]` |
| 局部目标与最终性能不一致 | 局部 loss 改善仍可能导致策略退化 | 有限采样、随机生成和反馈噪声 | `[Paper: PDF pp. 1–2]` |
| 坏更新不能及时纠偏 | 性能下降后仍沿原方向训练 | 没有检测和抑制机制 | `[Paper: PDF pp. 2, 5–6]` |
| 组相对优化的边界敏感 | 成功率接近 0 或 1 时梯度尺度可能剧烈变化 | 二元奖励、非退化组和局部信赖域假设下的组标准化尺度 | 图 3 与推导 `[Paper: PDF pp. 9–10, 22–25]` |

## 06 核心思想

1. 表层方法：每次常规探索更新后，将新批次平均奖励与最近 \(K\) 次历史均值比较，得到标准化改进信号，再回到上一批轨迹执行一次强化或抑制更新。`[Paper: PDF pp. 4–6]`
2. 核心洞见：训练信号不仅要回答“哪个样本更好”，还要回答“上一次策略更新是否使整体策略更好”。`[Paper: PDF pp. 1, 4]`
3. `[Analysis]` 可迁移的一般原则：对昂贵、含噪的科学生成任务，不能仅监控训练 surrogate 的当前奖励；需要固定协议下的独立、跨迭代验收。但是材料多目标验收应使用向量/Pareto 改进，而不是直接复用论文的标量批均值。

## 07 方法总览

| 项目 | 内容 |
|---|---|
| 输入 | 查询 \(q\)、策略生成响应 \(y\)、可验证奖励 \(R(q,y)\)、已有局部 attribution/advantage |
| 输出 | 被 PIPO 调制后的策略参数 |
| 历史状态 | 最近 \(K\) 个批次平均奖励及其均值、标准差 |
| 改进反馈 | 当前批次均值相对历史锚点的标准化差异 \(\xi_t\) |
| 回顾对象 | 上一批次的轨迹及局部 attribution |
| 更新 | 先进行 PIPO 回顾更新，再进行当前批次的基算法探索更新 |
| 关键假设 | 批均值可比较；反馈符号能指示更新方向；更新处于局部信赖域；奖励噪声不过度支配均值 |

文本流程：

`当前策略采样 → 当前批平均奖励 → 与滑动历史锚点比较 → 构造改进反馈 → 调制上一批局部优势 → 回顾式 PIPO 更新 → 常规 PPO/GRPO/SDPO 更新 → 写入历史记忆`。`[Paper: PDF pp. 4–6, Algorithm 1]`

## 08 核心模块拆解

| 模块 | 功能 | 必要性 | 输入与输出 | 支持证据 | 移除后的已知或预期影响 |
|---|---|---|---|---|---|
| PIRL 目标 | 把累计跨迭代性能增量写成目标 | 将最终性能与迭代改进联系起来 | \(J(\theta_t)\rightarrow \Delta J_t\) | 定义 3.1、定理 3.3 `[Paper: PDF p. 4]` | `[Analysis]` 移除后只剩常规局部目标 |
| 历史锚点 | 计算最近 \(K\) 批均值和标准差 | 降低单一前一批比较的波动 | 历史 \(\mu\rightarrow\mu_\mathrm{his},\sigma_\mathrm{his}\) | 公式 8 `[Paper: PDF p. 4]` | \(K\) 消融显示过大窗口会退化；并非越平滑越好 `[Paper: PDF p. 18, Table 7]` |
| 改进奖励 | 用 \(\xi_t\) 调制局部 attribution | 将跨迭代方向反馈给样本/位置级更新 | \(a_{t-1,i},\xi_t\rightarrow\hat r^{PI}_{t,i}\) | 公式 9–10 `[Paper: PDF p. 5]` | \(\lambda\) 消融显示完全或过强抑制并非最优 `[Paper: PDF p. 8, Table 4]` |
| 回顾式目标 | 在上一批轨迹上再执行 clipped 更新 | 让验证结果作用于产生该结果的更新 | 旧轨迹、importance ratio → 参数更新 | 公式 11、算法 1 `[Paper: PDF p. 6]` | `[Analysis]` 移除后改进信号无法直接纠偏前一更新 |
| 组边界分析 | 解释 GRPO 在极端成功率处的尺度问题 | 提供 PIPO 稳定性的局部机制解释 | 二元奖励组 → 理想梯度尺度 | 命题/推论与图 3 `[Paper: PDF pp. 9–10, 22–25]` | 只解释受限理想情形，不能当作通用全局保证 |

## 09 必要公式与符号

### 9.1 策略改进目标

\[
\Delta J_t=J_{RL}(\theta_{t+1})-J_{RL}(\theta_t), \qquad
\max \sum_{t=1}^{T}\mathbb E[\Delta J_t].
\]

其中 \(J_{RL}\) 是期望任务奖励，\(\theta_t\) 是第 \(t\) 次策略。其和式望远镜展开后，在固定 \(\theta_0\) 时等价于最大化最终 \(J_{RL}(\theta_{T+1})\)。`[Paper: PDF p. 4, Equations 5–6, Theorem 3.3]`

直觉：把每一步“是否比之前更好”显式化。`[Analysis]` 但该等价关系本身是代数望远镜恒等式，不自动解决 \(\Delta J_t\) 的无偏估计和有效优化。

### 9.2 经验性能与历史锚点

\[
\mu_t=\frac{1}{|B_t|}\sum_{(q,y)\in B_t}R(q,y),
\quad
\xi_t=\frac{\mu_t-\mu_{\rm his}}{\sigma_{\rm his}}.
\]

\(B_t\) 为当前批次，\(\mu_{\rm his}\) 与 \(\sigma_{\rm his}\) 是最近 \(K\) 个批均值的统计量。`[Paper: PDF pp. 4–5, Equations 7–9]`

### 9.3 改进调制

\[
\hat r^{PI}_{t,i}=a_{t-1,i}\,\phi_\lambda(\xi_t),\qquad
\phi_\lambda(x)=
\begin{cases}
x,&x\ge 0,\\
\lambda x,&x<0.
\end{cases}
\]

\(a_{t-1,i}\) 是上一批的局部 attribution；\(\lambda\) 控制负改进时的抑制强度。`[Paper: PDF p. 5, Equation 10]`

### 9.4 `[Hypothesis]` 材料任务所需的替代改进量

对用户目标不能只用标量均值。候选定义是：

\[
\Delta HV_t=HV\!\left(\mathcal P_t; r\right)-HV\!\left(\mathcal P_{\rm his}; r\right),
\]

其中 \(\mathcal P\) 是由稳定性、\(T_C\) 下置信界、迁移率下置信界、有效性和多样性构成的非支配集合，\(HV\) 是相对固定参考点 \(r\) 的 Pareto 超体积。该公式不是原论文内容，需要独立实现和验证。

### 9.5 其余公式清单及保留理由

- Equation 2、Equation 3、Equation 4 分别定义 PPO/GAE、GRPO 组优势和 SDPO attribution；它们是基算法背景，不是 PIPO 的新公式。`[Paper: PDF p. 3]`
- Equation 6 与 Equation 7 已分别并入 9.1 的累计增量目标和 9.2 的经验批均值。`[Paper: PDF p. 4]`
- Equation 12、Equation 13、Equation 14、Equation 15、Equation 16、Equation 17、Equation 18、Equation 19 是二元奖励组的条件期望、组内成功数与理想梯度尺度推导。`[Paper: PDF pp. 22–24]`
- Equation 20、Equation 21、Equation 22 用于证明 PIPO 在局部信赖域下对标准更新方向的修正关系。`[Paper: PDF p. 25]`

这些附录公式已完成清点，但因依赖理想化二元奖励假设，未把全部推导复制到“必要公式”正文。

## 10 实验设计与证据链

### 10.1 设计清单

- 训练/评测域：数学推理、代码、工具调用、自蒸馏科学知识。`[Paper: PDF pp. 6–8, 16–18]`
- 主模型：Qwen3-4B-Base 与 Qwen3-8B-Base。`[Paper: PDF pp. 6–8]`
- 基线：PPO、GRPO、GSPO、DAPO、SDPO 及其 +PIPO 版本。`[Paper: PDF pp. 6–8]`
- 指标：主要为 Pass@1；补充 Pass@8、训练动态和 wall-clock。`[Paper: PDF pp. 7–8, 18–19]`
- 计算设置：附录报告单节点 8×NVIDIA H20 141 GB；PIPO 默认 \(K=8\)、\(\lambda=0.1\)。`[Paper: PDF p. 17, Table 5]`
- 随机性：主结果之外给出 3 个数据顺序随机种子的训练曲线。`[Paper: PDF p. 19, Figure 4]`

| 实验 | 检验主张 | 比较与条件 | 结果 | 可支持结论 | 不能支持的更强结论 | 来源 |
|---|---|---|---|---|---|---|
| 数学推理 | PIPO 是否跨算法提升平均 Pass@1 | 4B/8B，四类基算法 | 4B 平均：PPO 44.0→45.8，GRPO 43.9→46.1，GSPO 44.5→45.8，DAPO 46.4→49.2；8B 也均提升 | 在这些平均指标和设置下有一致增益 | 每个 benchmark 都单调提升；迁移到扩散模型也提升 | `[Paper: PDF p. 7, Table 1]` |
| 代码与工具 | 方法是否跨任务有效 | 4B，四类基算法 | 平均值均提高，例如 PPO 代码 45.4→47.0、工具 81.4→83.5 | 有一定跨任务稳健性 | 任意 reward/oracle 上都有效 | `[Paper: PDF p. 8, Table 2]` |
| 自蒸馏 | 能否调制非标量局部蒸馏信号 | SciKnowEval | GRPO 平均 57.9→60.4；SDPO 69.9→73.8 | 可接到自蒸馏族 | “Materials”子项提升等于材料生成能力 | `[Paper: PDF p. 8, Table 3]` |
| \(\lambda\) 消融 | 负反馈抑制强度 | GRPO 4B | 平均值在 \(\lambda=0.1\) 为 46.1，基线 43.9；\(\lambda=1\) 为 44.5 | 软抑制在该设置优于过强抑制 | \(0.1\) 可直接用于 CGDiT | `[Paper: PDF p. 8, Table 4]` |
| \(K\) 消融 | 历史窗口长度 | GRPO 4B | \(K=8\) 平均 46.1；\(K=32\) 降至 43.6，低于基线 43.9 | 窗口存在时效性权衡 | 更长历史总是更稳定 | `[Paper: PDF p. 18, Table 7]` |
| 训练动态 | PIPO 是否提高平台并抑制震荡 | Qwen3-4B-Base，四类优化器 | 固线 PIPO 版本总体达到更高、更稳定轨迹 | 支持主表之外的过程证据 | 曲线本身证明统计显著性 | `[Paper: PDF p. 7, Figure 2]` |
| Pass@8 | 增益是否只存在于单次采样 | Qwen3-4B-Base | 汇总结果总体提高，但仍有单项下降 | 增益不只限于 Pass@1 | 所有采样预算和任务均提升 | `[Paper: PDF p. 18, Table 6]` |
| 稳定性分析 | 是否缓解边界梯度尖峰 | 二元奖励理论 + GRPO 曲线 | 图 3 显示梯度尖峰与性能退化得到缓解 | 支持局部机制解释 | 全局收敛或所有奖励分布下的稳定性 | `[Paper: PDF pp. 9–10, Figure 3]` |
| 计算代价 | 插件是否廉价 | 4B wall-clock | PPO 约 +1.6%，critic-free 方法约 +39.7% 至 +49.0% | 成本取决于基算法 | PIPO 对晶体扩散几乎无额外代价 | `[Paper: PDF p. 19, Figure 5]` |

## 11 对结论的正确解释

- 任务边界：所有验证都是语言模型后训练；论文没有扩散、晶体或材料性质 oracle。`[Paper: PDF pp. 6–8]`
- 输入边界：奖励来自数学验证器、代码执行、工具反馈或知识任务，而非 DFT、Monte Carlo 或电子—声子 BTE。
- 端到端性：PIPO 是外接于已有优化器的插件，不替代局部 attribution，也不自动生成可复算的扩散动作 log-prob。`[Paper: PDF pp. 5–6]`
- 理论边界：局部对齐结论依赖符号一致反馈、有限梯度和信赖域；组边界推导又依赖固定 query、二元奖励和非退化采样组。`[Paper: PDF pp. 9–10, 20–25]`
- 统计边界：PIPO 比较的是经验批均值；若不同批次难度不同，\(\xi_t\) 同时包含策略变化和批次组成变化。`[Analysis]`
- 指标边界：表中平均值一致提升，不表示每个子任务都提升；例如部分单项出现下降。`[Paper: PDF pp. 7–8, Tables 1–3]`
- 成本边界：对 critic-free 方法增加约 40%–49% wall-clock；晶体扩散若还需松弛、DFT、磁交换和迁移率评估，代价可能更高。`[Paper: PDF p. 19, Figure 5] [Analysis]`

有界重述：PIPO 在作者测试的 LLM 后训练配置中，通过滑动历史锚点调制上一批局部信号，提高了多个算法的平均任务指标并缓解部分训练不稳定；这证明“跨迭代验收”值得研究，但不证明它能直接解决多目标材料生成。

## 12 作者明确承认的局限

No explicit author-acknowledged limitation was found in the supplied source.

### 作者提到但未作为正式 Limitations 列出的约束

- 作者说明绝对分数可能因 prompt、解码和归一化实现不同而变化。`[Paper: PDF pp. 6, 16–17]`
- 附录显示 PIPO 对 critic-free 方法存在明显 wall-clock 开销。`[Paper: PDF p. 19, Figure 5]`
- 理论命题均附有局部性、有限梯度、信赖域或理想化奖励假设。`[Paper: PDF pp. 9–10, 20–25]`

## 13 批判性分析

| `[Analysis]` 观察 | 潜在问题或替代解释 | 为什么重要 | 如何检验 | 依据 |
|---|---|---|---|---|
| PIRL 累计增量与终值等价主要来自望远镜求和 | 目标重写不等于实际估计器无偏或易优化 | 防止把代数对齐误读成算法保证 | 比较真实 held-out \(\Delta J\) 与 \(\xi_t\) 的符号一致率 | 公式 5–9 `[Paper: PDF pp. 4–5]` |
| 当前批与历史批可能不是同一难度 | 批次组成变化可被误判为策略进步/退步 | 材料候选的化学体系和原子数变化更大 | 固定验证面板、分层匹配批次或 common random numbers | 经验均值定义 `[Paper: PDF p. 4]` |
| 标量平均奖励会隐藏目标冲突 | \(T_C\) 上升时迁移率、带隙或稳定性可能下降 | 用户要的是同时高性能，不是平均分更高 | 报告 Pareto 前沿、超体积、各目标 hard constraint 命中率 | PIPO 使用标量 \(\mu_t\) `[Paper: PDF pp. 4–5]` |
| PIPO 复用上一批 attribution | 若高奖励源于错误 surrogate，回顾更新会放大 reward hacking | \(T_C\) 与迁移率代理都有系统误差 | 训练 surrogate 与验收 oracle 分离，并跟踪 disagreement/OOD | 算法 1 `[Paper: PDF p. 6]` |
| 单项结果并非总是提升 | 平均数可能掩盖任务间再分配 | 多性质材料设计最怕“牺牲一项换另一项” | 每项报告置信区间和最坏目标退化 | 表 1–3 `[Paper: PDF pp. 7–8]` |
| 论文的 Materials 子项是知识问答 | 不能作为材料发现迁移证据 | 避免名称相同造成证据错配 | 必须在结构生成和独立物理计算中重新验证 | SciKnowEval `[Paper: PDF p. 8, Table 3]` |
| 额外更新成本对 critic-free 方法较高 | 在 1000 步扩散与昂贵 oracle 下成本可能不可接受 | 决定方案是否可运行 | 同 reward-call、NFE、GPU-hour 的 BoN/CFG/GRPO/PIPO 对比 | 图 5 `[Paper: PDF p. 19]` |

## 14 学到的知识

### Agent-derived knowledge candidates

- 跨迭代验收与单样本信用分配是两个不同层次：PIPO 解决前者，PPO/GRPO/SDPO 仍负责后者。
- 历史窗口有偏差—方差—时效性权衡；论文的 \(K=8\) 只是特定任务经验值。
- 软抑制比“发现下降就完全反向/停止”更可能稳定，因为经验改进信号本身含噪。
- 对科学生成，真正可靠的闭环应把“训练用便宜代理”和“验收用独立高保真评估”分开。
- 对多目标材料，跨迭代改进应定义在 Pareto 集或约束满足率上，而不是把不同量纲性质随意相加。

## 15 与已有知识和当前项目的连接

### 15.1 与材料生成工作的连接

- `[External]` MatterGen 已经用适配器与 classifier-free guidance 进行组成、对称性和磁性密度等标量条件生成，因此“给晶体扩散增加磁性条件”本身不是足够强的创新。来源：[Nature 2025 MatterGen](https://www.nature.com/articles/s41586-025-08628-5)。
- `[External]` MatInvent 已把 RL 用于扩散晶体生成，并覆盖磁性密度以及多目标设计；因此“CGDiT + GRPO/PIPO + 多目标奖励”也不能单独宣称新颖。来源：[MatInvent, arXiv:2511.03112](https://arxiv.org/abs/2511.03112)。
- `[External]` SGEquiDiff 已处理普通空间群约束和空间群不变 likelihood，但其研究讨论把磁空间群列为未覆盖方向；定向检索未发现与本卡完全相同的“磁空间群联合生成 + Pareto-PIPO”实现，仍需正式 prior-art review。来源：[SGEquiDiff, arXiv:2505.10994](https://arxiv.org/abs/2505.10994)。
- `[External]` 磁结构搜索和磁空间群识别已有成熟算法与 spglib 实现，说明增加磁矩/时间反演对称性具有可操作基础，但不等于生成模型本身已有。来源：[Shinohara et al., IUCrJ 2023](https://doi.org/10.1107/S2053273323005016)。

### 15.2 与目标物理的连接

- `[External]` 高通量 \(T_C\) 路线通常先做自旋极化 DFT、提取交换参数，再用 Heisenberg Monte Carlo；一项 2D 筛选从 786 个结构中报告 26 个预测 \(T_C>400\) K 的候选。来源：[npj Computational Materials 2020](https://www.nature.com/articles/s41524-020-0300-2)。
- `[External]` 2D 磁体的 \(T_C\) 对交换各向异性和计算方法敏感，不能只用简单平均场近似。来源：[Physical Review Research 2021](https://doi.org/10.1103/PhysRevResearch.3.043024)。
- `[External]` 迁移率不是只看有效质量；高可信评估需要散射率和 Boltzmann 输运。AMSET 可作中等保真层，但作者指出 PBE 往往高估迁移率，且方法不适合金属。来源：[Nature Communications 2021](https://www.nature.com/articles/s41467-021-22440-5)；更高保真基准见 [Physical Review Research 2021](https://doi.org/10.1103/PhysRevResearch.3.043022)。
- `[External]` VSi\(_2\)N\(_4\) 单层被预测同时具有室温以上 \(T_C\)、2.01 eV 带隙和约 \(10^6\) cm² V⁻¹ s⁻¹ 的迁移率，说明目标组合在计算研究中存在先例；这个极大迁移率仍应以完整电子—声子计算复核。来源：[Nano Letters 2024](https://doi.org/10.1021/acs.nanolett.4c01416)。
- `[External]` In\(_2\)Mn\(_2\)O\(_7\) 筛选显示提高 \(T_C\) 的应变可能恶化有效质量，直接说明两个目标可能冲突。来源：[npj Computational Materials 2019](https://doi.org/10.1038/s41524-019-0208-x)。

### 15.3 与 CGDiT 当前代码的连接

- `[Analysis: local project audit]` CGDiT 已有三个异构生成通道：D3PM 元素、周期分数坐标、受空间群投影的晶格；现有轨迹保存结构状态，但没有动作 log-prob、old/reference policy 或 RL buffer。证据：`cgdit/pl_modules/diffusion.py:91-105,146-184,293-299,446-481`。
- `[Analysis: local project audit]` 当前 C2DB 配置只训练 `gap/ehull/hform`，本地 CSV 也只有这些性质与 `magmoms`，没有 \(T_C\) 或迁移率标签。证据：`conf/data/c2db_51.yaml:3-6`、`conf/model/experiments/exp_c2db_all.yaml:18-30`。
- `[Analysis: local project audit]` `conf/data/magndata.yaml`、`conf/model/property_predictors/diffusion_cspnet/tc_regressor.yaml` 和 `data/magndata/` 当前均为空占位，不能据此训练 \(T_C\) 模型。
- `[Analysis]` 因此论文最适合接在 CGDiT 的 RL 基线之后：先建立三通道可复算策略和独立物理评估，再加入 PIPO 式迭代验收。跳过这两步直接套公式 9–11，importance ratio 没有严格定义。

## 16 研究创意

### Agent-derived research candidates

### 候选 A（推荐主线）：MSPI-CGDiT——磁对称感知的多保真 Pareto 政策改进晶体扩散

- 起点问题：PIPO 只验收标量批均值，且不表示磁性自由度；CGDiT 目前只生成元素、坐标和晶格，普通空间群约束不会描述磁矩、时间反演或磁有序。`[Paper: PDF pp. 4–6] [Analysis: local project audit]`
- `[Hypothesis]` 核心假设：若在 CGDiT 中显式表示位点磁矩/磁有序及磁空间群，并用独立多保真 oracle 的 Pareto 超体积增量验收每轮策略，则在相同高保真调用预算下，能比 CFG、best-of-N、标量 GRPO 和标量 PIPO 找到更多“稳定、室温铁磁且高迁移率”的非支配候选，同时不牺牲新颖性和多样性。
- 相对论文的明确增量：
  1. **表示**：从 \((A,X,L)\) 扩展为 \((A,X,L,M,G_m)\)，其中 \(M\) 是位点磁矩/初始磁构型，\(G_m\) 是磁空间群或时间反演约束；
  2. **策略**：元素用精确 D3PM categorical log-prob，坐标只在独立 anchor/Wyckoff 自由度上计算周期密度，晶格只在有效投影子空间计概率，磁矩通道使用与共线/非共线设定一致的概率模型；
  3. **反馈**：以 held-out、分层匹配候选上的 \(\Delta HV\) 或 hard-constraint success-rate 改进替代标量 \(\xi_t\)；
  4. **物理**：训练 surrogate 与验收 oracle 分离，并使用下置信界而非点预测；
  5. **科学目标**：直接优化 \(T_C\) 与 \(\mu\)，而不是磁性密度或有效质量代理。
- 推荐的第一任务定义：`[Hypothesis]` 先限定为**二维本征铁磁半导体**，因为当前已有 C2DB 数据基础。建议锁定：\(E_\mathrm{hull}<50\) meV/atom、非零合理带隙、\(T_C^{LCB}>300\) K、300 K 指定载流子浓度下的最弱晶向迁移率 \(\mu_{e/h,\min}^{LCB}\) 超过数据分布的预注册阈值，并满足声子稳定性。若目标是三维体相或金属，应另立任务；金属更适合优化电导率/电阻率，而不是沿用半导体迁移率定义。
- 多保真闭环：
  1. F0：几何/价态/对称性/短程排斥过滤；
  2. F1：ensemble 预测 \(E_\mathrm{hull}\)、带隙、磁基态、\(T_C\)、迁移率及 OOD；
  3. F2：MLIP 松弛 + 自旋极化 DFT，枚举 FM/AFM/FiM 构型，计算 SOC/磁各向异性、带结构、有效质量、弹性和形变势；
  4. F3：交换参数 + Heisenberg Monte Carlo 得 \(T_C\)，AMSET 得 300 K 散射受限迁移率；
  5. F4：少量 EPW/Perturbo 电子—声子 BTE、声子和有限温复核。
- 验证（validation）：固定生成数、NFE、reward-call 和 GPU-hour；至少 3–5 个种子；比较未微调 CGDiT、CFG sweep、best-of-N、reward-weighted fine-tuning、普通三通道 GRPO、MatInvent 风格标量/最小值奖励、PIPO 标量版本和 MSPI-CGDiT。主指标是 F3 确认的 joint-success 数、Pareto hypervolume、每次高保真调用命中率；辅助指标是 SUN、磁空间群命中率、组成/结构多样性、surrogate→DFT 排名相关和校准误差。
- 证伪条件：在同预算下未超过 best-of-N/普通 GRPO；或 surrogate Pareto 前沿在 F3 复核后坍塌；或增加磁通道后只提高训练指标却降低有效结构率。
- 可能失败模式（possible failure modes）：高质量 \(T_C\)/\(\mu\) 联合数据太少；磁构型枚举和 DFT+U/SOC 对结果敏感；2D 迁移率被形变势近似严重高估；高 \(T_C\) 与高 \(\mu\) 本身冲突；RL 模式坍缩；磁空间群标签不完整。
- 创新状态：**partially checked**。定向检索已确认普通空间群扩散、磁性密度条件生成、材料扩散 RL、多目标 RL、磁空间群算法分别已有；尚未发现完全相同的组合，但在正式论文中只能称“候选主创新”，必须继续做系统 prior-art search，不能写“首次”。

### 候选 B（最小可行基线）：固定验证面板上的 Pareto-PIPO

- 起点问题：直接增加磁空间群通道工程量大，而 PIPO 的批次混杂会污染改进信号。
- `[Hypothesis]` 核心假设：不改变 CGDiT 的 \((A,X,L)\) 表示，仅用固定/分层匹配验证面板和不确定性感知 Pareto 改进信号，也能比随批次标量 PIPO 更稳定地提高高保真 joint-success。
- 相对论文的 delta：把随机新批次的标量均值验收改成固定协议下的 Pareto/约束验收；用 F1 训练奖励、F2/F3 周期验收；负反馈只降低或回滚对应性质通道的更新强度。
- 验证：与普通 GRPO、原始 PIPO、相同 oracle 预算的 best-of-N 比较；跟踪改进信号与下一轮 F3 指标的符号一致率。
- 证伪条件：固定面板导致过拟合，或 \(\Delta HV\) 与真实高保真命中不相关。
- 失败模式：验证面板太小；oracle 噪声大；Pareto 超体积对参考点敏感；没有磁自由度时结构几何不足以决定磁基态。
- 创新状态：**partially checked**，更适合作为主线 A 的验证基线或第一阶段论文，不足以单独支撑最强方法创新。

### 候选 C（第二贡献）：物理通道—扩散时间联合信用分配

- 起点问题：PIPO 只调制已有 attribution；把终端 \(T_C/\mu\) 奖励平均复制到 1000 个扩散步和全部元素/坐标/晶格动作会产生高方差和错误归因。
- `[Hypothesis]` 核心假设：按物理因果近似把奖励分配给元素/磁矩、坐标和晶格通道，并在后期去噪步强化局部几何信用，可降低梯度方差并减少高保真调用。
- delta：元素/磁矩通道接收磁交换、价态、带边与载流子类型信用；坐标通道接收局域配位、力残差、交换路径和形变势信用；晶格通道接收应变、各向异性、弹性和迁移率张量信用；全局稳定性/多样性作用于联合策略。
- 验证：同样本轨迹上比较 joint terminal advantage、仅时间分层、仅通道分层、通道×时间联合；指标包括梯度方差、KL、每 100 次 F3 评估发现的非支配候选数。
- 证伪条件：归因规则不降低方差或不提高高保真 Pareto 前沿。
- 失败模式：物理信用并非可加分解；代理分项彼此强耦合；额外模型/计算抵消样本效率。
- 创新状态：**unverified**；需要进一步核对扩散时间信用分配与材料多通道 RL 的直接先例。
