# CGDiT 扩散模型训练加速调研、适配分析与验证计划

> 文档状态：初稿，待基准测试确认  
> 创建日期：2026-08-04  
> 适用仓库：`E:\WORKSPACE\CodePlace\test_CGDiT`  
> 目标：缩短 CGDiT 的训练墙钟时间，同时以非劣效实验确认生成质量和条件控制性能不下降

## 1. 先给结论

当前 CGDiT 最值得优先做的不是减少 `num_steps=1000`，而是同时减少“每个优化步骤的耗时”和“达到同等质量所需的优化步骤数”。

原因是：

- `cgdit/pl_modules/diffusion.py:131` 在每个训练 batch 只均匀抽取一个时间步；训练并不会把 1000 个反向扩散步骤全部展开。
- `validation_step()` 和 `test_step()` 也只调用一次 `forward()`；1000 步循环位于 `sample()` 中，主要影响生成而不是模型训练。
- 当前 `submit_python/run_all_mp20.sh` 顺序从头训练 8 套 base/单性质/双性质/三性质模型，这是总 GPU 时间最大的结构性重复。
- 当前主配置是单卡、FP32、每步记录日志；MP-20 的 DataLoader worker 全部为 0；最大 1000 epoch，而 early-stopping patience 为 100000，等价于基本不早停。
- CSPNet 每次前向都通过 `torch.ones`、`torch.block_diag` 和 `dense_to_sparse` 重新创建全连接图；该图只由每个样本的原子数决定，不依赖噪声坐标，存在保持图完全相同的等价加速空间。

按投入产出比，推荐顺序为：

1. 建立可复现的 20–50 epoch 计时与质量基线；
2. 先做 DataLoader、日志、全连接边构造、BF16 和 fused Adam 等低风险吞吐优化；
3. 启用真正有效的早停，并用 Min-SNR/非均匀时间步采样减少达到同等质量所需的 epoch；
4. 将 8 次从头训练改成“一次共享骨干 + 条件掩码训练”或“一次 base + 小型 property adapter”；
5. 最后才尝试 `torch.compile`、多卡 DDP 和材料表征对齐等中高风险改造。

不存在一种未经验证就能保证“更快且完全不降性能”的通用开关。本文把候选方法分为：

- **等价工程优化**：数学图或训练目标不变，是最接近无损的方案；
- **低精度/并行优化**：训练目标不变，但浮点轨迹会变化，需要非劣效验证；
- **收敛加速算法**：用更少 step 达到相同或更好结果，文献收益不能直接外推到晶体模型；
- **研究型改造**：可能成为论文贡献，但风险和验证成本较高。

## 2. 当前框架的训练成本诊断

### 2.1 已核实的训练链路

| 项目 | 当前状态 | 影响 |
|---|---|---|
| 训练时间步 | 每个 batch 均匀抽取一个 `t` | 1000 个扩散步不是训练单步耗时的 1000 倍 |
| 验证 | 单次带噪前向和 loss 统计 | 减少采样步数不会明显缩短验证 |
| GPU | 训练日志显示 RTX 4090；配置为 `devices: 1` | 有 Tensor Core，但当前 FP32 未充分利用 |
| 精度 | `precision: 32` | BF16 有潜在显著收益 |
| DataLoader | MP-20 train/val/test 均为 `num_workers: 0` | 主进程加载和拼 batch 可能让 GPU 等待 |
| 日志 | `log_every_n_steps: 1`、进度条每步刷新、W&B watch `all` | 产生同步、序列化和终端 I/O 开销 |
| 训练长度 | 1000 epoch | 总训练时间很长 |
| 早停 | patience 100000、每 5 epoch 验证 | 实际不会在 1000 epoch 前触发 |
| 实验组织 | 8 套条件组合依次从头训练 | 重复计算是总成本的最大来源 |
| 图构造 | 每步动态构建 dense block-diagonal 图再稀疏化 | 可做完全等价的边索引构造或缓存 |
| 优化器 | Adam，优化 `self.parameters()` | 条件实验仍更新全部参数，没有 PEFT 模式 |

需要纠正一个容易产生的误解：`cgdit/run.py` 只有在 `fast_dev_run` 时才强制将 worker 设为 0；日常 MP-20 训练的 worker=0 来自 `conf/data/mp_20.yaml` 本身，并非 `deterministic: true` 自动造成。

### 2.2 粗略墙钟下界

训练状态日志中可见约 `848 batch/epoch`、约 `4.85 iter/s`。只计训练 batch：

\[
T_{epoch}\approx 848/4.85\approx 175\text{ s}
\]

因此 1000 epoch 的纯训练下界约为 48.6 小时/模型；8 个模型约 389 GPU 小时，即约 16.2 天连续单卡时间。这个估算还没有加入验证、checkpoint、W&B、启动和 I/O，因此只能作为乐观下界，不能当作精确账单。

### 2.3 需要用 profiler 回答的问题

在修改模型前，先采集：

- GPU utilization、显存、power、SM/Tensor Core 利用率；
- 每步 `data_time`、forward、backward、optimizer、logging 的耗时；
- `cspnet.py` 中全连接边创建占比；
- `torch_scatter`、周期距离和 wrapped-normal 相关算子占比；
- W&B watch 和进度条开启/关闭的差值；
- BF16 后出现 FP32 fallback 或数值溢出的算子。

建议只用固定的 200 warm-up steps + 1000 timed steps 做微基准，不需要先跑完整 1000 epoch。

## 3. 方法调研与 CGDiT 适配判断

### 3.1 A 级：优先实施的近似无损工程优化

#### A1. 等价重写或缓存全连接边

当前位置：`cgdit/pl_modules/decoder/cspnet.py:173-175`。

当前方法先为每个晶体创建 `n×n` 全 1 dense 矩阵，再 `block_diag`，最后转成 sparse edge index。边集合只由 `num_atoms` 决定，因此可以：

- 直接通过 repeat/interleave 和 cumulative offsets 构建 source/destination；或
- 按原子数缓存局部完全图模板，再加 batch offset；或
- 在数据预处理阶段保存局部 edge index，collate 时只做偏移。

只要保留自环、边方向和排序语义，输出图可以逐元素相同，理论上不改变模型性能。这也是 `torch.compile` 前应先做的重构，否则 Python list、动态 `block_diag` 和 `dense_to_sparse` 很容易造成 graph break。

验证：固定一个 batch，对旧/新 `edge_index` 排序后逐元素相等；固定权重和随机噪声，三路输出与 loss 在 FP32 下相等。

#### A2. 降低日志与监控频率

建议基准候选：

- `log_every_n_steps: 20` 或 50；
- 进度条 `refresh_rate: 20`；
- `lr_monitor.logging_interval: epoch`；
- 正式长训关闭 `wandb.watch(log='all')`，仅调试梯度时开启；
- 继续保留 epoch 聚合指标、best checkpoint 和异常 loss 检测。

它不改变 forward/backward/optimizer，只减少同步和 I/O，属于最安全的优化。

#### A3. DataLoader 管线

Lightning 官方建议 GPU 训练使用 `num_workers>0`、`pin_memory=True`，多 epoch 时可用 `persistent_workers=True`。当前 DataLoader 没有后三者中的后两项，[官方速度指南](https://lightning.ai/docs/pytorch/stable/advanced/speed.html)明确指出 worker=0 可能成为瓶颈。

不应直接把 worker 设置为 CPU 核数。建议在目标 Linux 节点实测 `0/2/4/8`，比较 samples/s 和主机内存；PyG 的已缓存小图也可能在 worker 很多时因进程通信反而变慢。

首选候选：train `num_workers=4`、`pin_memory=True`、`persistent_workers=True`；val/test 可从 2 开始。Windows 本地和 Linux 集群要分别验证，因为 multiprocessing 行为不同。

#### A4. 有效早停与训练预算上限

目前 patience=100000 使早停失效。应根据已有训练日志中的验证曲线确定 plateau，而不是拍脑袋使用固定 epoch。

建议：

- 仍保留 `max_epochs=1000` 作为上限；
- 每 5 epoch 验证时，先测试 patience 10/20 个验证周期，即 50/100 epoch；
- 同时监控 `val_loss` 和少量固定种子生成集的有效率/性质命中率；
- 保存 best 与 last，避免 `val_loss` 与生成质量错位；
- 在同一基线曲线上回放早停规则，先估计它会在哪一 epoch 停止。

早停本身不能保证不损失性能，但若 stop 点之后的目标指标已进入噪声区间，它可以删除大量无收益 epoch。

### 3.2 B 级：硬件吞吐优化

#### B1. BF16 mixed precision

RTX 4090 支持 BF16 Tensor Core。Lightning 通过 mixed precision/autocast 支持 BF16，[官方文档](https://lightning.ai/docs/pytorch/stable/common/precision_basic.html)说明低精度可减少带宽并加速受支持 GPU 上的矩阵运算。

优先试 `bf16-mixed`，不先用 FP16，原因是 BF16 动态范围更接近 FP32，对扩散噪声尺度更稳。CGDiT 中以下部分建议保留 FP32 island：

- 方差、SNR、beta/alpha schedule；
- wrapped-normal/log-prob 或极小概率运算；
- `softmax`/离散后验中可能下溢的部分；
- 最终 loss reduction，若观察到不稳定。

预期：常见吞吐收益可能为 1.3–2×；Lightning 文档给出的是一般性上限而非 CGDiT 保证值。通过条件：无 NaN/Inf，三种 loss 的尺度和梯度范数稳定，最终非劣效测试通过。

#### B2. fused Adam 和 matmul 精度

PyTorch Adam 暴露 `fused`/`foreach` 参数，[官方 Adam API](https://docs.pytorch.org/docs/main/generated/torch.optim.Adam.html)和[优化器文档](https://docs.pytorch.org/docs/stable/optim)说明部分优化器存在更快的 fused 实现。可基准 `fused=True`，若当前 CUDA/PyTorch/torch-scatter 组合不兼容则回退。

还可测试 `torch.set_float32_matmul_precision('high')` 以启用更快的 TF32 路径。它会造成轻微浮点差异，因此与 BF16 一样需要质量验证，不能称为位级等价。

#### B3. 多卡 DDP

若集群能提供 2–4 张 GPU，DDP 是比 FSDP 更合适的选择。当前模型能放入一张 4090，没有必要承担参数分片复杂度。

为了尽量保持优化语义：

- 保持 global batch=32；2 卡时每卡 16，4 卡时每卡 8；
- 保持学习率、scheduler 和每个 epoch 的总样本数不变；
- 使用 DistributedSampler 并检查 epoch seed；
- 比较 1/2/4 卡 samples/s 与通信占比。

若扩大 global batch，则已经改变了优化问题，需要重新调学习率，不能再归入低风险无损加速。

#### B4. `torch.compile`

Lightning 官方支持编译模型并指出新 GPU 上可能获得显著加速，但实际收益取决于 graph break。[官方速度指南](https://lightning.ai/docs/pytorch/stable/advanced/speed.html)提供了入口。

CGDiT 的动态原子数、Python list、PyG scatter、dense-to-sparse、周期边界运算都可能导致重新编译或 graph break。因此顺序应是：先等价重写 edge 构造，再用 `TORCH_LOGS=graph_breaks,recompiles` 审计，最后测试 `torch.compile(..., dynamic=True)`。若编译时间摊销后收益小于 5%，不值得增加维护成本。

### 3.3 C 级：减少达到目标质量所需的优化步骤

#### C1. Min-SNR loss weighting

[Min-SNR（ICCV 2023）](https://openaccess.thecvf.com/content/ICCV2023/html/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.html)把不同时间步视作互相冲突的任务，通过截断 SNR 权重缓解梯度冲突；图像实验报告 3.4× 收敛加速和更好的最终 FID。

它适合 CGDiT，但不能把图像中的单一公式原封不动用于全部 loss。CGDiT 有：

- 连续晶格 loss；
- 周期分数坐标 loss；
- 离散原子类型 D3PM loss；
- 三者的噪声日程和尺度不同。

推荐先只对晶格与坐标通道按各自 SNR 加权，原子类型保持原 loss；记录每个时间桶的 loss 和梯度范数，再决定离散通道是否需要独立权重。论文的 3.4× 只能作为研究动机，不能作为项目工期承诺。

#### C2. 非均匀时间步采样

当前入口非常明确：`beta_scheduler.uniform_sample_t()`。

- [SpeeD（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_A_Closer_Look_at_Time_Steps_is_Worthy_of_Triple_CVPR_2025_paper.html)通过非对称时间步采样与加权，在其图像实验中报告约 3× 收敛加速。
- [Adaptive Non-Uniform Timestep Sampling（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/papers/Kim_Adaptive_Non-Uniform_Timestep_Sampling_for_Accelerating_Diffusion_Model_Training_CVPR_2025_paper.pdf)优先采样梯度方差更高的时间步。
- [Improved DDPM](https://arxiv.org/abs/2102.09672)的 loss-second-moment sampler 是更早且实现简单的参考基线。

CGDiT 的关键风险是三路 loss 在同一个 `t` 下难度不同。建议从 20 个时间桶收集：每通道均值、二阶矩、梯度范数和有效样本数；先实现带 importance correction 的 sampler，保证估计目标不被无意改变，然后比较共享 sampler 与三通道加权 sampler。

#### C3. 表征对齐加速

[REPA（ICLR 2025）](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d9e42b4d7163931f3689d6d6fbaa11d0-Abstract-Conference.html)将带噪中间表征对齐到冻结视觉编码器表征；其 SiT 图像实验用少于 400K steps 匹配了原先 7M steps 的效果。该数字高度依赖图像预训练表征，不能直接外推到晶体。

可形成 CGDiT 的研究型版本：

- 用冻结的 M3GNet/CHGNet/自监督晶体 GNN 提取干净晶体 teacher embedding；
- teacher embedding 预计算并缓存，避免每个 step 额外跑 teacher；
- 从 CSPNet 中间层读出 noisy structure embedding；
- 添加随时间步变化的 alignment loss；
- 检验它是否同时提升收敛速度、稳定性和条件性质控制。

当前 `DiffusionMultimodal` 已有结构–性质对比对齐，但它不是“冻结的干净结构教师指导带噪去噪表征”。把材料预训练教师、三通道噪声和空间群对称性结合，可能成为比单纯套用 Min-SNR 更有论文价值的贡献。

### 3.4 D 级：消除 8 次重复训练

#### D1. 一次共享的 masked multi-condition 训练

把 FE、band gap、energy above hull 三个条件视为一个可缺失条件向量，在训练时随机 mask 条件：

- 全 mask 对应 unconditional/base；
- 单条件、双条件、三条件通过随机 mask 同时覆盖；
- 推理时用 mask 指定可用条件组合；
- 保留 classifier-free condition dropout 以支持 CFG。

理想情况下，一次训练可替代当前 8 次从头训练，实验族总成本上限可接近 8× 降低。实际模型可能需要更多 step、不同 loss balance，不能预先承诺 8×，但这是当前最大的结构性节约方向。

它的创新性一般，除非进一步解决缺失条件、条件冲突和 Pareto 控制；其主要价值是工程与公平实验设计。

#### D2. 一次 base + property adapters

[MatterGen（Nature 2025）](https://www.nature.com/articles/s41586-025-08628-5)在直接相关的晶体扩散场景中，使用 adapter 对化学组成、对称性和标量性质条件做微调；[官方实现](https://github.com/microsoft/mattergen)也提供 property adapter fine-tuning 流程。这是 CGDiT 最有说服力的直接先例。

CGDiT 可采取：

1. 只训练一次 unconditional backbone；
2. 冻结 CSPNet 主干和扩散主输出层；
3. 训练 property encoder、插入各层的 scale/shift 或 bottleneck adapter，以及必要的输出校准层；
4. 多性质联合目标训练一个 adapter，而不是推理时简单拼接多个独立 adapter；
5. 与 full fine-tuning 比较性能和墙钟时间。

注意：参数量减少不等于 forward 计算按同样比例减少，因为冻结主干仍需完成前向。主要收益来自更小的反向图、优化器状态、较短的适配训练，以及重复利用 base。当前 `configure_optimizers()` 仍把 `self.parameters()` 全部交给 Adam，因此需要显式 `requires_grad=False` 与只收集 trainable parameters 的 PEFT 训练模式。

### 3.5 E 级：暂不优先

| 方法 | 不优先原因 |
|---|---|
| FlashAttention/SDPA | 当前 CSPNet 是全连接消息传递网络，不是标准 Transformer attention；不能直接获得收益 |
| FSDP/ZeRO-3 | 模型能放入单张 4090，通信和工程复杂度可能大于收益 |
| 减少采样步数/DPM-Solver | 主要加速生成，不缩短当前训练 forward/backward；可单独做推理项目 |
| 蒸馏/consistency model | 通常需要 teacher 生成轨迹或额外训练，目标是少步采样，不是最直接的训练加速 |
| Immiscible Diffusion | 数据–噪声匹配在图像上有潜力，但晶格、周期坐标和离散原子三路联合匹配复杂，先验不足 |
| 盲目加大 batch | 可能改变优化和泛化；应先通过 BF16/吞吐基准确认，再调学习率与 batch |

## 4. 推荐组合与预期收益

下面的数字是用于排序的经验区间，不是已经在 CGDiT 上测得的结果。

| 组合 | 主要目标 | 可能收益 | 性能风险 | 优先级 |
|---|---|---:|---|---:|
| 日志降频 + DataLoader 调优 | 每 step 更快 | 1.05–1.25× | 很低 | P0 |
| 等价 edge 构造 | 每 step 更快 | 取决于 profiler，可能 1.05–1.2× | 很低 | P0 |
| BF16 + FP32 islands | 每 step 更快 | 1.3–2× | 低–中 | P0 |
| fused Adam/TF32 | 每 step 更快 | 1.02–1.15× | 低 | P1 |
| 有效 early stopping | 更少 epoch | 取决于 plateau，可能 1.5–5× | 中 | P0 |
| Min-SNR/非均匀 `t` | 更少 optimizer steps | 文献约 3–3.4×；本项目应保守验证 1.5× 起 | 中 | P1 |
| shared masked condition | 8 次变 1 次 | 实验族约 5–8× | 中 | P0/P1 |
| base + adapters | 复用 base | 后续条件任务常见 1.2–2×/任务，并显著减小训练状态 | 中 | P1 |
| 2/4 卡 DDP | 墙钟并行 | 取决于通信与 batch，需实测 | 低–中 | 有资源时 P1 |
| REPA-style 晶体教师对齐 | 更少 steps + 论文创新 | 未知，可能显著 | 高 | P2 |

组合收益不能直接相乘。例如 DataLoader 优化在 BF16 后可能更重要，因为 GPU 计算变快后 CPU 更容易成为瓶颈；compile 和 edge 重写也有重叠收益。

## 5. “不降低性能”的操作性定义

不能只比较最后一个 `val_loss`。建议预先注册非劣效标准：

### 5.1 训练层面

- 固定相同 train/val/test split；
- 基线和候选至少 3 个随机种子；
- 报告达到既定质量阈值的 GPU-hours，而非只报告 epoch；
- 报告 steps/s、samples/s、峰值显存、总能耗（若可采集）；
- 记录三路 loss、梯度范数和时间桶统计。

### 5.2 生成质量层面

- 结构有效率、组成有效率；
- uniqueness、novelty、coverage/precision-recall；
- 密度、元素分布和结构描述符的分布距离；
- 固定随机种子和固定生成样本数，避免快方法用更多样本获益。

### 5.3 条件与材料性能层面

- FE、band gap、energy above hull 的 MAE/命中率；
- 多条件同时满足率；
- surrogate 预测之外，抽取同等数量样本做独立 MLIP 或 DFT 验证；
- 稳定、唯一、新颖（SUN）比例；
- 若论文声称可合成性，需另做 convex-hull、动力学/有限温稳定性和合成证据，不能用负形成能替代。

### 5.4 推荐非劣效门槛

先用基线 3 seeds 的标准差定义 margin。若必须预设，可暂用：

- 主生成成功率下降不超过 1 个百分点；
- 主性质 MAE 相对恶化不超过 1%；
- 其余指标的 95% bootstrap CI 与基线重叠；
- 同时 GPU-hours 至少降低 20%，才值得保留新增复杂度。

最终门槛应由你最看重的论文指标确认，以上不是领域统一标准。

## 6. 分阶段实施计划

### 阶段 0：冻结基线与 profiler

任务：

- [ ] 固定一个代表性配置，例如 base 和 FE+BG+EH；
- [ ] 采集 200 warm-up + 1000 timed steps；
- [ ] 保存 profiler trace、GPU 利用率、steps/s、显存；
- [ ] 记录 50/100/200/... epoch 的验证与固定生成集指标；
- [ ] 从现有长训曲线回放早停规则。

成功标准：能把总耗时拆成 data/forward/backward/optimizer/logging，并得到后续所有方法共用的非劣效基线。

### 阶段 1：P0 等价/低风险优化

实验按单因素加入：

1. 日志与进度条降频；
2. worker/pin/persistent benchmark；
3. 等价 edge index 构造；
4. BF16 + FP32 islands；
5. fused Adam/TF32。

成功标准：每一项均报告独立增益；组合后单卡 samples/s 提升至少 30%，且短训 loss 曲线和完整非劣效指标通过。

### 阶段 2：减少训练步数

实验：

- [ ] uniform `t` + 原 loss；
- [ ] uniform `t` + Min-SNR（连续两通道）；
- [ ] loss-second-moment sampler；
- [ ] adaptive non-uniform sampler；
- [ ] sampler + channel-aware weighting；
- [ ] 每个方法均报告达到同一质量阈值所需的 optimizer steps 和 GPU-hours。

成功标准：至少 1.5× 更少 GPU-hours 达到基线阈值，最终性能不劣于基线。

### 阶段 3：消除重复条件训练

实验：

- [ ] 8 个 full-training 基线中的代表性子集；
- [ ] shared masked multi-condition；
- [ ] base + full fine-tuning；
- [ ] base + scale/shift adapter；
- [ ] base + bottleneck adapter；
- [ ] 多性质联合 adapter。

成功标准：报告整个 8 条件实验族的总 GPU-hours，而非只比较一次 fine-tuning；单条件与多条件指标均达到非劣效。

### 阶段 4：研究型创新

在前 3 阶段稳定后再做：

- [ ] 缓存晶体 teacher embeddings；
- [ ] 提取 CSPNet 分层 noisy embeddings；
- [ ] 设计 symmetry-aware、channel-aware、time-aware alignment loss；
- [ ] 与 Min-SNR、普通 feature matching、无教师对照；
- [ ] 检验不同数据规模和不同性质上的泛化。

可能的论文主张：

> 对周期晶体的晶格、坐标与原子类型三路异构扩散，引入对称性与噪声通道感知的材料表征对齐和时间步重要性分配，在保持晶体生成质量与多性质条件命中率的同时，显著减少训练 GPU-hours。

只有当它同时超过强基线（BF16、Min-SNR、adaptive sampler、adapter）并给出机制证据时，才算有说服力的算法创新。

## 7. 最小建议配置草案（尚未实施）

第一轮只建议形成一个独立 `train/fast.yaml` 或命令行 override，不覆盖现有基线：

```yaml
pl_trainer:
  accelerator: gpu
  devices: 1
  precision: bf16-mixed
  log_every_n_steps: 20
  max_epochs: 1000

early_stopping:
  patience: 20  # 20 次验证，即当前每 5 epoch 验证时最多容忍约 100 epoch 无提升

progress_bar:
  refresh_rate: 20

wandb_watch:
  log: null

lr_monitor:
  logging_interval: epoch
```

DataLoader 的 `num_workers`、`pin_memory` 和 `persistent_workers` 应通过目标训练节点基准确定，不把某个数值硬编码成所有机器的默认值。

## 8. 推荐的最终决策

如果目标是尽快把当前 8 套实验跑完：

1. 先做日志/DataLoader/BF16/edge benchmark；
2. 恢复有效早停；
3. 改为一次 base + adapters，或直接训练 masked multi-condition 模型；
4. 不要先投入 FlashAttention、FSDP、蒸馏或少步采样。

如果目标是发表“扩散训练加速”方向论文：

1. 工程优化只作为公平基线；
2. 主方法选择“channel-aware timestep sampling/weighting + symmetry-aware material representation alignment”；
3. 将 wall-clock、GPU-hours、最终生成/条件/材料指标和多数据集泛化同时纳入主表；
4. 明确区分算法收敛加速、硬件吞吐加速和推理采样加速。

## 9. 参考文献与官方资料

针对 Diff2Flow 论文及其在 CGDiT 三通道上的可采用边界，另见 [`diff2flow_project_relevance_analysis.md`](diff2flow_project_relevance_analysis.md)。

1. Hang et al., [Efficient Diffusion Training via Min-SNR Weighting Strategy](https://openaccess.thecvf.com/content/ICCV2023/html/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.html), ICCV 2023.
2. Wang et al., [A Closer Look at Time Steps is Worthy of Triple Speed-Up for Diffusion Model Training](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_A_Closer_Look_at_Time_Steps_is_Worthy_of_Triple_CVPR_2025_paper.html), CVPR 2025.
3. Kim et al., [Adaptive Non-Uniform Timestep Sampling for Accelerating Diffusion Model Training](https://openaccess.thecvf.com/content/CVPR2025/papers/Kim_Adaptive_Non-Uniform_Timestep_Sampling_for_Accelerating_Diffusion_Model_Training_CVPR_2025_paper.pdf), CVPR 2025.
4. Nichol & Dhariwal, [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672), ICML 2021.
5. Yu et al., [Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d9e42b4d7163931f3689d6d6fbaa11d0-Abstract-Conference.html), ICLR 2025.
6. Zeni et al., [A generative model for inorganic materials design](https://www.nature.com/articles/s41586-025-08628-5), Nature 2025; [official MatterGen repository](https://github.com/microsoft/mattergen).
7. PyTorch Lightning, [Speed Up Model Training](https://lightning.ai/docs/pytorch/stable/advanced/speed.html).
8. PyTorch Lightning, [N-Bit Precision](https://lightning.ai/docs/pytorch/stable/common/precision_basic.html).
9. PyTorch, [Adam API](https://docs.pytorch.org/docs/main/generated/torch.optim.Adam.html) and [optimizer overview](https://docs.pytorch.org/docs/stable/optim).

## 10. 修订记录

| 版本 | 日期 | 修改内容 | 修改人 |
|---|---|---|---|
| v0.1 | 2026-08-04 | 完成训练瓶颈审计、加速方法调研、CGDiT 适配分级、非劣效标准和实施计划 | Codex |
