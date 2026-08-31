# Magndata-Tc 严格对照与 RL 结构生成

## 1. Magndata-Tc 数据

训练数据位于 `data/magndata/`，目标列为原始居里温度 `tc`，单位 K。
数据来自去除“结构等价且 Tc 相同”重复项后的 1297 条记录。所有结构统一转换为
primitive cell 和 Niggli reduced cell；12 条原胞超过 100 原子、1 条约化失败，最终
保留 1284 条。

固定划分使用随机种子42，并按 `pretty_formula` 分组，保证相同化学式不会跨越训练、
验证和测试集：

| split | structures | formula groups |
|---|---:|---:|
| train | 1023 | 830 |
| val | 122 | 100 |
| test | 139 | 110 |

训练集 Tc 均值为 `99.35190615835778 K`，总体标准差（ddof=0）为
`147.49836824580123 K`。哈希、过滤记录和泄漏检查在
`data/magndata/split_manifest.json` 中。

## 2. 四组严格对照

四组模型使用完全相同的数据划分、种子、优化器、训练轮数、早停规则和测试指标：

1. `m3gnet_scratch`：M3GNet 从头训练预测 Tc。
2. `diffusion_scratch`：与生成模型相同的 CSPNet 扩散骨干随机初始化，从头训练 Tc。
3. `diffusion_base_pretrained`：加载无条件 MP-20 基础扩散模型的全部 decoder 权重，
   重新初始化 Tc 标量头，再在 Magndata 上全量微调。
4. `diffusion_joint_pretrained`：加载 MP-20 三目标联合扩散模型的全部 decoder 权重，
   重新初始化 Tc 标量头，再在 Magndata 上全量微调。

默认对每组运行 `42、123、3407` 三个随机种子，共12次训练。主要指标为测试集 MAE
（K），同时统计 RMSE、median absolute error、bias、R2、Pearson r、Spearman rho、
MAE bootstrap 95% CI，以及三个种子的均值和标准差。

### 启动

```bash
cd /root/private_data/rszhong/workspace/test_CGDiT
conda activate cgdit

bash -n submit/run_magndata_tc_benchmarks.sh
bash -n submit_python/run_magndata_tc_benchmarks.sh

SEEDS="42 123 3407" \
BASE_PRETRAINED_DIFFUSION_MODEL="output/singlerun/2026-06-27/00-32-50-mp20_base" \
JOINT_PRETRAINED_DIFFUSION_MODEL="output/singlerun/2026-08-05/13-12-29-mp20_fe_bg_eh" \
GPU_ID=0 \
bash submit/run_magndata_tc_benchmarks.sh
```

已有三组实验 `20260824-045243` 时，不必重复训练。下面的命令会把旧结果复制到新的
四组对照目录，将旧 `diffusion_pretrained` 正名为 `diffusion_joint_pretrained`，并且
只训练缺少的3个基础模型迁移实验：

```bash
REUSE_BENCHMARK_ROOT="output/magndata_tc_benchmark/20260824-045243" \
BASE_PRETRAINED_DIFFUSION_MODEL="output/singlerun/2026-06-27/00-32-50-mp20_base" \
JOINT_PRETRAINED_DIFFUSION_MODEL="output/singlerun/2026-08-05/13-12-29-mp20_fe_bg_eh" \
SEEDS="42 123 3407" \
GPU_ID=0 \
bash submit/run_magndata_tc_benchmarks.sh
```

启动脚本会自行使用 `nohup` 提交后台 worker，不要在外面再套一层 `nohup`。

只做单种子试运行时：

```bash
SEEDS="42" GPU_ID=0 bash submit/run_magndata_tc_benchmarks.sh
```

### 查看进度和结果

```bash
bash submit/check_magndata_tc_benchmarks.sh
```

结果位于：

```text
output/magndata_tc_benchmark/<RUN_ID>/
├── m3gnet_scratch/seed_<SEED>/
├── diffusion_scratch/seed_<SEED>/
├── diffusion_base_pretrained/seed_<SEED>/
├── diffusion_joint_pretrained/seed_<SEED>/
├── per_run_metrics.csv
├── aggregate_metrics.csv
├── paired_differences.csv
├── test_predictions.csv
├── benchmark_summary.json
└── run_manifest.txt
```

## 3. 高/低 Tc 门控回归

该实验在固定的 Magndata 划分上训练一个二分类门控网络和两个回归专家。默认物理
阈值为 `300 K`：`tc >= 300 K` 记为高 Tc，否则为低 Tc。门控网络使用无条件
MP-20 基础扩散模型的 CSPNet decoder 作为预训练骨干，采用按训练集类别比例设置
`pos_weight` 的加权 BCE，并以验证集 PR-AUC 保存最佳 checkpoint。

当前固定数据在该阈值下的类别计数为：训练集低/高 Tc=`912/111`，验证集
`111/11`，测试集 `123/16`；因此训练时的默认正类权重由脚本从训练集动态计算为
`912 / 111 = 8.216...`，不是手写常数。

每个随机种子还训练两个专家：

1. `low_expert` 只用训练/验证集中的低 Tc 样本微调；
2. `high_expert` 只用训练/验证集中的高 Tc 样本微调；
3. 两个专家都完整加载相同种子的 `diffusion_base_pretrained` 全局 Tc 回归器；
4. 两个专家使用 Huber loss 和全局 Magndata 训练集的同一套 Tc 标准化，确保加载
   checkpoint 后初始预测函数不因更换标准化参数而改变；
5. 两个专家都对完整测试集预测，最后统一进行路由和对比。

主结果为软路由
`(1 - p_high) * low_prediction + p_high * high_prediction`。同时报告硬路由消融和
使用真实高/低标签的 oracle 路由上限。硬路由阈值只在验证集上选择：在高 Tc
召回率不低于 `0.80` 的候选中最大化 MCC，不读取测试标签选阈值。

### 启动

先确认四组严格对照结果 `20260824-113321` 仍在服务器上，然后执行：

```bash
cd /root/private_data/rszhong/workspace/test_CGDiT
conda activate cgdit

bash -n submit/run_magndata_tc_moe.sh
bash -n submit_python/run_magndata_tc_moe.sh

SEEDS="42 123 3407" \
TC_THRESHOLD_K=300 \
MIN_GATE_RECALL=0.80 \
BASE_DIFFUSION_MODEL="output/singlerun/2026-06-27/00-32-50-mp20_base" \
GLOBAL_TC_BENCHMARK_ROOT="output/magndata_tc_benchmark/20260824-113321" \
GPU_ID=0 \
bash submit/run_magndata_tc_moe.sh
```

启动入口已经使用 `nohup` 放到后台，不要再套一层 `nohup`。先用单种子做端到端
试运行时可执行：

```bash
SEEDS="42" GPU_ID=0 bash submit/run_magndata_tc_moe.sh
```

断点续跑需要沿用第一次输出的 `RUN_ID`：

```bash
RUN_ID=<原RUN_ID> GPU_ID=0 bash submit/run_magndata_tc_moe.sh
```

### 查看进度和结果

```bash
bash submit/check_magndata_tc_moe.sh
```

也可以指定某次任务：

```bash
bash submit/check_magndata_tc_moe.sh <RUN_ID>
```

每个种子应完成门控、低温专家、高温专家三次训练；三个种子共 `9/9`。结果位于：

```text
output/magndata_tc_moe/<RUN_ID>/
├── data/
│   ├── gate/{train,val,test}.csv
│   ├── low/{train,val,test}.csv
│   ├── high/{train,val,test}.csv
│   └── moe_data_manifest.json
├── seed_<SEED>/
│   ├── gate/
│   ├── low_expert/
│   └── high_expert/
├── gate_metrics_per_seed.csv
├── gate_metrics_aggregate.csv
├── regression_metrics_per_seed.csv
├── regression_metrics_aggregate.csv
├── routing_differences.csv
├── test_predictions.csv
├── moe_summary.json
└── run_manifest.txt
```

门控指标包括 PR-AUC、ROC-AUC、MCC、balanced accuracy、precision、recall、
specificity、F1 和混淆矩阵。回归部分对全测试集、真实低 Tc 子集、真实高 Tc
子集分别统计 MAE、RMSE、median AE、max AE、bias、R2、Pearson r、Spearman rho
和 MAE bootstrap 95% CI，并直接给出 soft/hard/oracle 相对全局回归器的差值。

## 4. RL 微调模型的目标结构生成

该任务在同一个后台进程中依次处理 FE-RL 和 BG-RL 两个策略模型。每个模型先生成
4096个 ab initio 结构，再生成4096个模板结构，合计4批、16384个结构。RL 已经把目标偏好写入策略，因此目标通过
`--evaluation_target` 记录并用于事后命中率计算，不会再次施加 CFG 条件。

每种生成模式都会得到：

- `eval_gen_*.pt`：原子种类、分数坐标、晶格、原子数和目标元数据；
- `eval_metrics_gen_*.json`：有效性、覆盖率、AMSD/AMCD、密度和元素数分布距离；
- `eval_properties_gen_*.csv`：Seed42 FE/BG/Ehull 预测器逐结构预测；
- `eval_property_metrics_gen_*.json`：性质均值、分布和目标命中率；
- `source_manifest.json` 与 `run_manifest.txt`：模型及产物来源。

### 同时运行两个 RL 模型

```bash
cd /root/private_data/rszhong/workspace/test_CGDiT
conda activate cgdit

bash -n submit/run_rl_target_generation.sh
bash -n submit_python/run_rl_target_generation.sh

FE_RL_MODEL_PATH="output/reinforcement_learning/2026-08-23/21-47-35-grpo-pipo-probe32/grpo_fe_seed42_pipo/model" \
BG_RL_MODEL_PATH="output/reinforcement_learning/2026-08-23/21-47-35-grpo-pipo-probe32/grpo_bg_seed42_pipo/model" \
FE_TARGET_VALUE="-1.5" \
BG_TARGET_VALUE="2.0" \
GPU_ID=0 \
bash submit/run_rl_target_generation.sh
```

启动入口只提交一次后台任务，worker 会先完成 FE-RL 的两种生成和评估，再执行
BG-RL。默认目标标签分别为 `fe_m1p5` 和 `bg_2`，一般无需修改。

任务支持按生成、结构指标和性质指标逐阶段断点续跑。若已有任务中断，上传修复后的
代码后使用原 `RUN_ID` 重新启动即可；已完成产物会显示为 `[SKIP]`。例如：

```bash
RUN_ID=20260824-051136 GPU_ID=0 bash submit/run_rl_target_generation.sh
```

PyTorch 2.2 等没有 `torch.serialization.add_safe_globals` 的环境会自动使用可信本地
生成文件的旧版兼容加载方式，无需升级 PyTorch。

### 查看进度和结果

```bash
bash submit/check_rl_target_generation.sh
```

结果位于：

```text
output/reinforcement_learning/<YYYY-MM-DD>/<HH-MM-SS-EXPERIMENT>/
├── fe/
│   ├── model/generated_structures/ 和 evaluations/
│   ├── property_summary_seed42/
│   └── policy_manifest.txt
├── bg/
│   ├── model/generated_structures/ 和 evaluations/
│   ├── property_summary_seed42/
│   └── policy_manifest.txt
└── run_manifest.txt
```

最终应有4个生成 `.pt`、4个结构指标 JSON、4个逐结构性质 CSV 和4个性质指标
JSON。每批用 Seed42 的 FE、BG、Ehull 预测器进行统一评估。

## 5. 单 GPU 执行建议

两个启动入口是彼此独立的后台任务，但每个都会使用 GPU。只有一张 RTX 4090 时，
建议先启动 RL 生成，完成后再启动 Magndata-Tc 训练；同时运行可能显著变慢，并在
大晶体 batch 上造成显存不足。两张 GPU 时可分别设置 `GPU_ID=0` 和 `GPU_ID=1`。
