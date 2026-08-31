---
title: CGDiT 强化学习实际执行计划（更新审阅稿）
status: updated-review
date: 2026-08-23
branch: newton
scope: MP20 基础生成模型的形成能与带隙强化学习
primary_method: CrystalPIRL
probability_foundation: OrbitPO
reward_framework: closed-loop general material generation reward
---

# CGDiT 强化学习实际执行计划（更新审阅稿）

> 本文件合并原临时计划与 2026-08-21 条件生成结果分析。它是当前唯一的 RL 实际执行顺序，不是已经完成的实验结论。

## 0. 更新结论

原计划的工程主线仍然有效：先验证 OrbitPO 轨迹概率，再开展 PPO/GRPO，随后检验 CrystalPIRL，最后进行多 seed、消融和多保真验证。

本次根据现有结果作出以下修正。

| 原临时计划 | 本次更新 | 原因 |
|---|---|---|
| 所有 RL 从 mp20_base 开始 | 保持该设计：FE、BG、FE+BG 均从同一个 mp20_base 开始 | 基础模型没有性质条件输入，目标只通过奖励进入，最有利于识别 RL 的独立作用 |
| FE 首轮按最小化处理 | FE 固定为 -1.5 eV atom⁻¹ 的目标匹配任务 | 与现有条件 checkpoint 和基线结果保持一致，避免奖励无界追求极端低值 |
| Ab initio 无条件基线被视为缺失 | 旧 Base 的 Ab initio empirical 无条件 4096 样本已存在；新 Base 训练完成后再重建正式匹配基线 | 当前先开发 RL 闭环，WP0/WP1 延后到新扩散 Base checkpoint 就绪后 |
| 先完成六种 PIPO/PIRL 方法 | 先运行 PPO、GRPO、PPO+PIRL、GRPO+PIRL；原始 PIPO 放到第二层 | 先回答 PIRL 是否跨 PPO/GRPO 有效，减少早期变量 |
| 主要关注命中率 | 以完整分布、目标误差、联合密度和有效目标产率为主 | 命中率不能反映分布中心、长尾和双性质权衡 |
| H2 和多保真完全延后 | 保留为核心 RL 通过后的内部诊断阶段 | 用户要求实现并查看效果，但不作为最终文章摘要和主结论 |

新增一条核心主线：实现**闭环通用材料生成奖励**。它不把“多项奖励加权”包装成新方法，而是把固定物理标度的性质、有效性和可选探索分量同时用于策略优化与候选更新验收：

    fixed-scale raw reward
    → PPO/GRPO group advantage
    → candidate policy
    → paired raw-reward verification
    → candidate vs current verified policy
    → candidate vs permanently frozen base policy
    → accept / attenuate / reject
    → verified policy or rollback

Chemeleon2 的创造性、稳定性和 leave-one-out MMD 多样性用于组件与消融设计；其批次 min–max 不用于主方法的原始奖励，因为跨批次变化的标度不能支持 candidate 与 current/frozen-base 的跨策略比较。当前冻结契约见 conf/rl/components/rewards/contracts/mp20.yaml。

闭环永久维护三种策略：frozen base \(\pi_0\)、current verified policy \(\pi_k\) 和 candidate \(\pi'\)。任何候选都必须同时通过：

1. candidate vs current：证明本轮是局部改进；
2. candidate vs frozen base：证明累计结果仍是绝对改进；
3. candidate vs frozen base 的 validity、stability、uniqueness、diversity 非退化约束。

只相对当前策略变好但仍低于 \(\pi_0\) 的更新定义为假改进，不得接受。`attenuate` 只产生待重新评估的缩放候选，不等同于自动接受。

## 1. 当前证据与主张边界

### 1.1 已有证据

当前 seed-42 代理预测结果表明，Template 条件生成使完整性质分布向目标移动：

| 性质 | 无条件中位数 | 条件中位数 | 中位绝对目标误差 | 命中率 |
|---|---:|---:|---:|---:|
| FE | -0.796 | -1.325 | 0.956 → 0.326 | 3.00% → 10.06% |
| BG | 0.031 | 1.550 | 1.982 → 0.908 | 8.96% → 27.83% |

FE+BG 联合命中率由 0.610% 提高到：

- Template 条件生成：3.442%；
- Ab initio 条件生成：3.467%。

这说明目标区域在当前生成分布中可达，也为 RL 提供了外部性能参考；条件扩散模型不作为 RL 初始策略。正式 RL 从无条件 Base 模型出发，以隔离性质条件输入和强化学习更新的作用。

### 1.2 Gate 3b 工程诊断证据

作业 667909 在真实 GPU 上完成单种子单步尺度扫描，两次跨进程完整轨迹哈希一致：

| 算法 | 正 LCB 尺度 | 最佳诊断奖励增量 | 当前解释 |
|---|---|---:|---|
| PPO | 0.5 | +0.0760 | 更新方向有信号，完整步长证据不足 |
| GRPO | 0.25、1.0 | +0.0698（scale=1.0） | 与 PPO 的安全尺度不同 |

该结果仅使用 training seed=42、probe seed=4242、batch=16，且当前门控只覆盖奖励与基础有效性，尚未完成双重基线、稳定性、唯一性和多样性约束。因此它证明的是训练与诊断链路有效，不是正式材料生成质量结论。

### 1.3 Gate 3c 双锚点 GPU 诊断证据

作业 667914 在 RTX 3090 上完成，退出码为 0，16 项预检测试全部通过。PPO step 0 的 scale=0.25 缩放候选重新 rollout 后 LCB 仍为 -0.00124，最终被拒绝；PPO step 1 也被拒绝。GRPO step 0 的 local/absolute mean delta 为 +0.0698、LCB 为 +0.000257，奖励门控判定接受；GRPO step 1 相对 base 仍为正（mean +0.00671，LCB +0.0000168），但相对 current 为负（mean -0.0631，LCB -0.1818），因此被局部门正确拒绝。

该任务验证了双锚点、attenuate 复验和拒绝逻辑，但只评估 validity。由于 stability、uniqueness、novelty、composition diversity 和 structure diversity 缺失，所有记录均为 `diagnostic_only`，`verified_checkpoint_update=false`。这不是正式模型性能结论。


### 1.4 尚不能声称

在完成独立评价前，禁止声称：


- 代理预测性质等同于真实物性；
- Ab initio 条件控制已经严格优于无条件生成；
- 生成结构经弛豫后仍满足目标；
- RL、PIRL、H2 或多保真闭环已经产生实际提升；
- 单 seed 结果具有统计稳健性。

### 1.4 Ab initio 的实验口径

本项目中的 Ab initio 表示从头构造对称骨架，不表示 DFT 第一性原理计算。

当前 abinitio_empirical 实现只按训练集经验分布抽取原子数；空间群默认仍在 1–230 中随机抽取。所有结果按该实际实现解释。

## 2. 核心研究问题

### RQ1：OrbitPO 概率基础是否能支持真实策略更新？

必须验证轨道离散概率、周期坐标概率和晶格有效子空间概率能够一致重放，并产生有限、非零梯度。

### RQ2：闭环通用材料生成奖励能否使无条件基础模型形成目标性质偏好？

FE/BG 目标只通过 reward predictor 进入，不通过 classifier-free guidance、条件嵌入或条件 checkpoint 进入。评价对象不是单个最优样本，而是相对冻结 Base policy 的完整 FE/BG 分布、联合目标误差和有效目标产率变化。

这里的“通用”指同一奖励接口能够组合目标匹配、最小化、最大化、区间、有效性、稳定性、创造性和多样性，不表示当前实验已经覆盖全部材料性质。当前实证范围仍固定为 FE、BG 和 FE+BG。

### RQ3：CrystalPIRL 是否对 PPO 和 GRPO 都有效？

- 同时改善 PPO 和 GRPO：支持算法正交的策略改进机制；
- 只改善其中一种：结论收缩为算法特定增强；
- GRPO 与 GRPO+PIRL 等效：说明 GRPO 自身可能已经足够稳定；
- 两者均无改善：停止把 PIRL 作为文章主创新。

### RQ4：提升能否从 Template 泛化到 Ab initio？

Template 用于受控开发和主对照，Ab initio 用于检验新对称骨架上的泛化。两条路线都比较同一个 Base checkpoint 在 RL 更新前后的策略，不能用不同 checkpoint 之间的差异代替 RL 效果。

## 3. 固定模型角色

| 任务 | Initial / frozen base \(\pi_0\) | Current verified \(\pi_k\) | Candidate \(\pi'\) | 外部非 RL 基线 |
|---|---|---|---|---|
| FE | mp20_base 永久冻结副本 | 最近通过双重门控的 FE 策略 | 本轮 PPO/GRPO 更新 | 原始 FE CFG checkpoint |
| BG | mp20_base 永久冻结副本 | 最近通过双重门控的 BG 策略 | 本轮 PPO/GRPO 更新 | 原始 BG CFG checkpoint |
| FE+BG | mp20_base 永久冻结副本 | 最近通过双重门控的 FE+BG 策略 | 本轮 PPO/GRPO 更新 | 原始 FE+BG CFG checkpoint |

补充说明：

- mp20_base 是所有 RL 方法和性质任务的共同初始策略；
- FE/BG 目标不得写入 CFG、条件嵌入或采样 batch 的 condition 字段；
- RL 采样保持 condition_values 为空、guidance_scale 为 0；
- mp20_fe、mp20_bg、mp20_fe_bg 仅作为非 RL 外部性能参考，不能用于 RL 因果归因；
- frozen base 在整个实验期间不可被 optimizer、resume 或 checkpoint 覆盖；
- old/current verified policy 是每次更新的局部参考，但不能替代 frozen base 的绝对参考；
- reward predictor 固定为当前 seed-42 FE/BG 模型；
- seed-123 同架构模型只能做随机种子审计，不构成真正独立 evaluator；
- final evaluator 不参与训练、候选选择、PIRL accept/reject 或超参数选择。

## 4. 预注册目标与评价指标

### 4.1 性质目标

| 性质 | 目标 | 容差 |
|---|---:|---:|
| FE | -1.5 eV atom⁻¹ | ±0.06 eV atom⁻¹ |
| BG | 2.0 eV | ±0.45 eV |

执行顺序固定为：

1. FE 单目标；
2. BG 单目标；
3. FE+BG 联合目标。

Ehull 和其他性质不进入本轮研究。

### 4.2 主指标

主指标不是单纯命中率，而是：

1. 中位绝对目标误差；
2. FE/BG 完整一维分布；
3. FE–BG 二维联合密度；
4. 联合命中率；
5. 有效目标产率；
6. 结构有效率；
7. uniqueness 和 novelty；
8. 独立 evaluator 上的弛豫后性质保持率。

有效目标产率定义为：

    有效且所有目标均命中的结构数 / 全部生成结构数

所有失败结构保留在原始分母中，不静默删除。

### 4.3 随机种子与预算

- smoke seed：42；
- pilot seed：42；
- formal seeds：42、123、2026；
- pilot rollout：128–512；
- formal evaluation：每种方法、每个生成 seed 4096 个结构；
- 正式扩散步数、CFG scale、NFE、query 数和 GPU-hour 在 pilot 后冻结；
- GPU 任务名统一为 zrs-gen-rl。

## 5. 奖励设计

奖励分为原始物理奖励、策略优势和策略验收信号三层。三者不得混写：

1. raw_reward：固定目标与容差定义，不做批次 min–max，用于日志、跨更新比较和 Paired-PIRL；
2. policy_advantage：PPO 使用固定/EMA baseline，GRPO 只在同任务 group 内标准化；
3. probe_metrics：原始奖励为主指标，validity、uniqueness、novelty 和 diversity 为安全指标。

第一版只使用有界连续目标奖励和有效性门控，不加入尚不存在的模型不确定性。

单性质目标距离：

    d_p = abs((property - target) / tolerance)

冻结的有界分数与当前代码保持一致：

    r_p = exp(-0.5 d_p²)

联合性质以瓶颈式为主，加权均值作消融：

    r_joint = min(r_fe, r_bg)
    r_weighted = (w_fe r_fe + w_bg r_bg) / (w_fe + w_bg)

第一版总奖励：

    R_raw = valid_gate × property_reward
            - (1 - valid_gate) × invalid_penalty

借鉴 Chemeleon2 的可选组件：

- creativity：unique 且 novel 为 1，两者均不满足为 0，边界样本使用同组成 AMD 距离；
- stability：1 - clip(E_hull / E_max, 0, 1)；
- composition_diversity：组成特征的 leave-one-out MMD 边际贡献；
- structure_diversity：结构特征的 leave-one-out MMD 边际贡献。

MMD 边际贡献必须使用 pilot 前冻结的尺度映射到 [0,1]，不得在每个训练批次内重新 min–max。第一轮因果实验关闭这些辅助分量；只有达到预注册的坍缩触发条件后，才运行 property + validity + diversity 消融。

必须监控但暂不并入第一版奖励：

- uniqueness；
- novelty；
- 极端原子数和晶格；
- 重复组成或结构；
- 与初始策略的 KL。

如果出现模式坍缩，再按预注册顺序加入 KL 或重复惩罚，不在运行中临时改变定义。

代码映射：

| 功能 | 位置 | 当前状态 |
|---|---|---|
| target/maximize/minimize/range | cgdit/rl/rewards.py::property_reward | 已实现 |
| 显式无效惩罚 | robust_validity_gated_reward | 已实现 |
| 创造性、稳定性、leave-one-out MMD | cgdit/rl/rewards.py | 已实现张量级组件 |
| 固定标度多性质聚合 | general_material_reward | 已实现 |
| raw reward→paired gate | cgdit/rl/policy_improvement.py::decide_reward_improvement | 已实现 |
| 真实 checkpoint predictor adapter | 待接入 trainer | 未完成 |

## 6. 分阶段执行计划

### WP0：冻结实验契约

状态：暂缓。等待使用当前更新后的扩散代码重新训练 Base 模型并上传 checkpoint 后再执行。

操作：

- [ ] 记录 Git commit、uv.lock 哈希和 checkpoint 哈希；
- [ ] 冻结目标、容差、seed、采样步数和 CFG scale；
- [ ] 冻结 reward predictor；
- [ ] 登记 final evaluator，但允许 evaluator 未就绪时继续工程 smoke；
- [ ] 建立统一输出 schema；
- [ ] 冻结有效性和模式坍缩安全阈值，建议有效率相对基线下降不超过 3 个百分点。

产物：

- conf/rl/components/rewards/contracts/mp20.yaml；
- checkpoint/evaluator manifest；
- baseline manifest。

Gate 0：

- 新 Base checkpoint 未就绪时，不冻结正式实验契约；
- 未冻结 final evaluator：可做工程开发，不能启动正式论文主实验；
- 配置或 checkpoint 哈希不明确：停止。

### WP1：补齐匹配基线

状态：暂缓。与 WP0 一起在新 Base checkpoint 就绪后执行。

当前仅用于开发的旧模型结果：

- Base Template/Ab initio empirical 无条件；
- FE、BG、FE+BG 条件模型结果作为外部性能参考。

新 Base 到达后需要重新生成：

- 新 Base 的 Template 无条件基线；
- 新 Base 的 Ab initio empirical 无条件基线；
- 统一 seed 下的冻结 Base paired probe；
- 所有基线的性质表、结构指标和完整分布。

RL 因果比较固定为：

    frozen Base before RL → PPO/GRPO policy from the same Base
    frozen Base before RL → PPO/GRPO+PIRL policy from the same Base

条件 checkpoint 只用于回答“RL-from-Base 能否接近或超过 CFG 条件生成”，不参与“RL 是否有效”的主判断。

Gate 1：

- Base 与 RL policy 不能保持相同起点、route、seed、样本数和采样预算时，不解释 RL 效果；
- WP1 未完成时允许旧 Base 的 RL smoke 和 pilot，不产生正式论文主张。

### WP2：完成最小 RL 训练闭环

当前已实现：

- 轨道代表点与广播；
- sample_rl 轨迹；
- lattice/coord/atom 三通道 action、noise 和 log-prob；
- 晶格有效子空间 Gaussian；
- 周期坐标 wrapped Gaussian；
- 轨道 categorical log-prob；
- PPO/GRPO objective；
- Paired rollout、bootstrap LCB 和策略决策；
- H2 与多保真接口。
- 固定标度通用材料奖励及 Chemeleon2 风格的创造性、稳定性和边际多样性组件；
- 原始奖励与命名安全指标到 Paired-PIRL 门控的连接接口。
- 训练 CLI、PPO/GRPO 配置、M3GNet reward adapter、optimizer、checkpoint/resume 和结构化训练日志；
- 真实 GPU 上的 20、200、999 步轨迹与非零有限 decoder 梯度验收。

尚需完成：

- [ ] frozen base/current verified/candidate 三策略状态与双重比较；
- [ ] stability、uniqueness、diversity 的正式门控连接；
- [ ] attenuate 后重新 rollout 与再次验收；
- [ ] 独立 holdout probe 与周期性绝对基线审计；
- [ ] 多 seed、大 probe 的统计验收。

当前训练与诊断入口：

    scripts/cli/training/train_crystal_rl.py
    scripts/cli/training/diagnose_rl_update_scale.py
    conf/rl/experiments/ppo_fe.yaml
    conf/rl/experiments/grpo_fe.yaml

Gate 2：**已通过。**

- 20、200 和完整步数 sample_rl 可执行；
- 相同 noise_seed 轨迹可复现；
- old=current 时 ratio 约等于 1；
- current=reference 时 KL 约等于 0；
- 重放 log-prob 与记录一致；
- decoder 梯度有限且非零；
- 轨道元素一致率为 100%；
- resume 后下一步状态一致；
- 任一 NaN、Inf、无梯度或轨道破坏均停止扩大训练。

### WP3：FE Template 核心 pilot

先只运行四种方法：

| 编号 | 算法 | PIRL |
|---|---|---|
| P0 | PPO | 无 |
| P2 | PPO | 有 |
| G0 | GRPO | 无 |
| G2 | GRPO | 有 |

设置：

- initial/frozen base：旧 mp20_base 的永久冻结副本；current verified 从该副本开始独立演化；
- route：Template；
- target：FE=-1.5 eV atom⁻¹；
- condition_values：空；
- guidance_scale：0；
- seed：42；
- smoke rollout：16–64；
- pilot rollout：128–512；
- 所有方法使用相同 rollout、NFE、query 和参数更新范围。

当前证据状态：

- 远程 P0/G0/P2/G2 八更新 pilot 已完成；P2/G2 在当前小 probe 下全部拒绝候选，P0 接近中性，G0 出现退化，尚未通过科学 Gate 3；
- Gate 3b 表明 PPO scale=0.5、GRPO scale=0.25/1.0 存在正 LCB，证明候选方向并非全部无效；
- Gate 3c 已验证双锚点和 attenuation 复验：GRPO 第二步只通过 absolute gate、未通过 local gate，因而被正确拒绝；
- 这些结果仍是单 seed、小 probe、单一 validity 安全门，下一步是完整安全指标与 holdout，而不是直接扩展 BG。

Gate 3：

- reward 与目标误差方向一致；
- 相对冻结 Base 的完整 FE 分布向目标移动；
- seed-123 审计模型不出现相反方向；
- validity 不越过安全阈值；
- uniqueness 不发生明显坍缩；
- 训练成本支持扩大到正式规模。
- 原始奖励在不同 group 和更新轮次保持同一物理定义，GRPO 标准化不得覆盖日志中的 raw_reward；

FE Template 未通过时，不进入 BG、联合性质或 Ab initio RL。

### WP4：PIPO 与 PIRL 机制判定

核心四组出现稳定信号后，再实现原始 PIPO：

| 编号 | 算法 | 更新验证 |
|---|---|---|
| P1 | PPO | 原始 PIPO |
| G1 | GRPO | 原始 PIPO |

原始 PIPO 与 Paired-PIRL 必须使用独立代码路径。PIPO 不得使用固定 paired probe、paired delta 或 LCB。

Paired-PIRL 固定流程：

    permanently frozen base policy
    + current verified policy
    → PPO/GRPO candidate
    → frozen paired probe
    → candidate-vs-current local paired delta
    → candidate-vs-base absolute paired delta
    → reward + validity + stability + uniqueness + diversity LCB
    → accept / attenuate / reject
    → attenuate 后重新 rollout/验收，或 verified checkpoint 更新/回滚
    → holdout probe 周期性绝对基线审计

pilot 建议：

- diagnostic probe：当前 16，仅用于工程诊断；
- formal probe：至少 128；
- bootstrap：95%；
- attenuate scales：预注册候选集合，缩放后必须重新评估；
- final evaluator 不参与策略门控。

Gate 4：

- P2 相对 P0、P1 降低坏更新率；
- G2 相对 G0、G1 降低坏更新率；
- 额外 probe 查询和 GPU 成本单独报告；
- 若 PIRL 只帮助一种算法，主动收缩创新主张。
- 若仅组内标准化 advantage 改善而固定 probe 原始奖励不改善，不得声称闭环通用奖励有效。

### WP5：BG 与 FE+BG 扩展

在 FE Template 通过后按顺序复用同一代码：

1. BG Template：仍从 mp20_base 开始，只更换 BG 奖励；
2. FE+BG Template：仍从 mp20_base 开始，使用联合奖励；
3. 比较联合奖励的加权均值与瓶颈形式；
4. 若触发模式坍缩阈值，再比较 property + validity 与 property + validity + diversity。

核心方法先保持 P0、P2、G0、G2；只有在需要建立 PIPO 关系时再补 P1、G1。

Gate 5：

- BG 完整分布稳定向 2.0 eV 移动；
- 联合 RL 同时改善 FE 和 BG，不允许仅靠优化单一性质获得联合结论；
- 有效目标产率提高；
- 有效率、uniqueness 和 novelty 不发生预注册阈值外退化。

### WP6：Ab initio 泛化

在 Template 算法与奖励稳定后，使用同一方法进入 Ab initio empirical。

需要分别评价：

- 冻结 Base 的 Ab initio 无条件生成；
- 从同一 Base 更新得到的 PPO/GRPO policy；
- 从同一 Base 更新得到的 PPO+PIRL/GRPO+PIRL policy；
- 条件模型的 Ab initio 结果仅作为外部性能参考。

必须与 WP1 冻结的新 Base Ab initio 无条件基线比较。

重点报告：

- 性质分布；
- 空间群和原子数分层表现；
- PyXtal 骨架生成失败率；
- 结构有效率；
- 目标产率；
- Template 到 Ab initio 的性能下降。

Gate 6：

- 不能因为 Ab initio 的骨架分布不同而把路线差异解释为 RL 增益；
- 只有独立 evaluator 方向一致时，才声称 RL 泛化到新对称骨架。

### WP7：正式三 seed 与 OrbitPO 消融

formal seeds：

    42、123、2026

核心正式矩阵：

| Route | Baseline | PPO | PPO+PIRL | GRPO | GRPO+PIRL |
|---|---:|---:|---:|---:|---:|
| Template | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ab initio empirical | ✓ | ✓ | ✓ | ✓ | ✓ |

OrbitPO 消融：

- full-cell atom counting vs orbit counting；
- full lattice Gaussian vs active-subspace Gaussian；
- non-periodic coordinate likelihood vs wrapped likelihood；
- raw joint ratio vs 自由度错误归一化。

报告 ratio/KL 对原子数、轨道多重度和空间群有效维数的关系。

Gate 7：

- 至少三个 seed 方向一致；
- 不隐藏失败 run；
- final evaluator 与 reward predictor 方向不冲突；
- 同时报告等效和负结果；
- 预算不公平时不做算法优劣结论。

### WP8：H2 与多保真内部诊断

H2 消融：

    无信用分配
    仅时间信用
    仅通道信用
    通道 × 时间信用

多保真闭环：

    reward predictor
    → 独立 predictor 复核
    → MLFF 几何弛豫和稳定性筛选
    → 弛豫后 FE/BG 复算
    → 少量 DFT 验证

注意：

- MatterSim 等 MLFF 可用于能量、力、应力和弛豫；
- MLFF 通常不能直接替代电子带隙计算；
- BG 需要独立电子性质模型或 DFT；
- H2 和多保真结果保存完整内部报告，但不进入当前摘要和主结论，除非用户以后改变文章范围。

### WP9：新轨道 Base 模型完成后的正式切换

旧 mp20_base 只完成 RL 工程验证和探索性 pilot。新 orbit-corrected Base 模型训练完成并上传后：

- [ ] 检查新 Base checkpoint 与当前代码兼容；
- [ ] 执行暂缓的 WP0，冻结正式实验契约；
- [ ] 执行暂缓的 WP1，重建 Template 和 Ab initio 基线；
- [ ] 冻结新 Base 及其 reference 副本；
- [ ] 从新 Base 分别运行 FE、BG、FE+BG 的 P0/P2/G0/G2；
- [ ] 条件模型是否重训不构成 RL 正式实验的启动条件；
- 旧 checkpoint 标记为 legacy development baseline。

不得把旧 checkpoint 结果写成“由修正轨道损失训练得到”的结果。

## 7. 输出目录与日志

正式 RL 输出：

    output/rl/<date>/<method>/<property>/<route>/<seed>/
    ├── config.yaml
    ├── environment.txt
    ├── checkpoints/
    ├── rollouts/
    ├── rewards.parquet
    ├── policy_metrics.parquet
    ├── generated_structures.pt
    └── run_summary.json

内部扩展：

    output/rl_internal/
    ├── h2_channel_time/
    └── multifidelity/

每次运行必须记录：

- Git commit 与 uv.lock 哈希；
- checkpoint 和 predictor 哈希；
- seed、route、目标和容差；
- NFE、query、GPU-hour 和峰值显存；
- 所有失败样本与失败原因；
- 是否用于训练、选择、probe 或 final evaluation。

## 8. 论文主张契约

只有满足下列条件，才能支持主要论文主张：

1. PPO/GRPO 与其 PIRL 版本在相同预算下比较；
2. 至少三个生成 seed；
3. 完整分布和有效目标产率改善；
4. 结构有效性与多样性没有不可接受退化；
5. 独立 evaluator 方向一致；
6. Template 与 Ab initio 分开解释；
7. 轨道商空间概率消融支持 OrbitPO 的必要性；
8. 负结果、拒绝更新和失败 run 不被隐藏。
9. “闭环通用材料生成奖励”的主张必须同时由性质改善、坏更新率下降和安全指标不退化支持；不能只凭组件数量或训练 reward 上升成立。

如果仅 reward predictor 改善，结论只能是“优化了代理奖励”，不能写成“设计出目标性质材料”。

## 9. 立即执行顺序

1. [x] 完成训练 CLI、真实 GPU smoke、Gate 2、四组 FE pilot 和 Gate 3b 尺度诊断；
2. [x] 实现 frozen base/current verified/candidate 三策略运行状态与日志契约；
3. [x] 实现 candidate-vs-current 和 candidate-vs-base 双重 LCB；
4. [ ] 接入 validity、stability、uniqueness、diversity 非退化门控；
5. [ ] attenuate 后重新 rollout/验收已完成；周期性 rollback 尚未完成；
6. [ ] 建立 holdout probe，执行多 seed、probe≥128 的 Gate 3c 正式扩展；
7. [ ] 双重门控通过后重跑 P0/P2/G0/G2 多轮 FE pilot；
8. [ ] 实现 P1/G1 原始 PIPO，完成 FE 的 2×3 比较；
9. [ ] FE 通过后依次扩展 BG、FE+BG 和 Ab initio empirical；
10. [ ] 新 Base 上传后执行 WP0/WP1、兼容性和三 seed 正式实验；
11. [ ] 运行 OrbitPO 消融、独立 evaluator 冻结评估；
12. [ ] 最后运行 H2 与 predictor→MLFF→DFT 内部诊断。

## 10. 当前状态

状态：WP2/Gate 2、WP3 工程 pilot、Gate 3b 和 Gate 3c 双锚点 GPU 诊断已完成；科学 Gate 3 尚未通过，当前进入完整安全门、holdout 与多 seed 正式验证。

可以立即进行：

- 多指标非退化门控；
- holdout probe 与周期性 rollback；
- Gate 3c 的多 seed、probe≥128 正式 FE 扩展；
- P1/G1 原始 PIPO 实现。

当前阻塞正式主实验的事项：

- 新 orbit-corrected Base checkpoint 尚未就绪；
- final evaluator 尚未冻结；
- 完整安全门控、holdout 和 rollback 尚未完成；
- 当前只有单 seed、小 probe 的诊断证据；
- 只有一个正式生成 seed。
