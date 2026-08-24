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

## 3. RL 微调模型的目标结构生成

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

FE_RL_MODEL_PATH="output/rl_finetune/seed42_grpo_pipo_probe32/668001/grpo_fe_seed42_pipo/model" \
BG_RL_MODEL_PATH="output/rl_finetune/seed42_grpo_pipo_probe32/668001/grpo_bg_seed42_pipo/model" \
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
output/rl_generation/<RUN_ID>/
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

## 4. 单 GPU 执行建议

两个启动入口是彼此独立的后台任务，但每个都会使用 GPU。只有一张 RTX 4090 时，
建议先启动 RL 生成，完成后再启动 Magndata-Tc 训练；同时运行可能显著变慢，并在
大晶体 batch 上造成显存不足。两张 GPU 时可分别设置 `GPU_ID=0` 和 `GPU_ID=1`。
