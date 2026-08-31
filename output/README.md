# Output 目录说明

核对日期：2026-08-31

`output/` 只保存模型、生成结构、预测结果、评估结果、manifest 和可复现审计数据。运行时标准输出、错误和 `.log` 文件应写入 `logs/`。

## 统一目录

```text
output/
├── singlerun/
│   └── <YYYY-MM-DD>/<HH-MM-SS-experiment>/
├── reinforcement_learning/
│   └── <YYYY-MM-DD>/<HH-MM-SS-experiment>/
└── test/
    ├── compatibility/
    ├── reinforcement_learning/
    │   ├── <YYYY-MM-DD>/<HH-MM-SS-experiment>/
    │   └── archives/<YYYY-MM-DD>/
    └── structure_generation/
        └── <YYYY-MM-DD>/<HH-MM-SS-experiment>/
```

- `singlerun/`：Hydra 管理的生成模型和性质预测器训练结果。
- `reinforcement_learning/`：正式 RL 训练、模型生成、评估和审计结果。
- `test/`：兼容性验证、preflight、smoke、Gate 和其他非正式结果。

禁止重新创建 `rl_finetune/`、`rl_generation/`、`rl_diagnostics/`、`rl_analysis/` 或顶层 `compatibility/`。

## 单次生成模型或性质预测器运行

```text
singlerun/<YYYY-MM-DD>/<HH-MM-SS-experiment>/
├── .hydra/
├── epoch=*.ckpt
├── hparams.yaml
├── generated_structures/
└── evaluations/
```

性质预测器通常额外保存 `test_preds.npy` 和 `test_targets.npy`，不创建生成结构目录。

## 单次强化学习运行

```text
reinforcement_learning/<YYYY-MM-DD>/<HH-MM-SS-experiment>/
├── submission_manifest.txt
├── <algorithm_property_seed_method>/
│   ├── model/
│   ├── train_results/
│   ├── test_results/
│   └── model/generated_structures/ 与 model/evaluations/
├── analysis/
└── diagnostics/
```

同一个训练流水线中的训练、生成和评估结果应归入同一个运行目录。每个算法/性质模型的检查点和由该模型产生的结果保存在对应模型目录下；跨模型分析放在运行级 `analysis/`；manifest 和 JSON 审计数据放在 `diagnostics/`。

## 测试结果

测试使用与正式运行相同的日期/时间/实验名格式，但根目录固定为 `output/test/<category>/`。兼容性验证放在 `output/test/compatibility/`，强化学习测试放在 `output/test/reinforcement_learning/`。

结构生成前的验证性输入（例如 VASP band calculation 输入）放在 `test/structure_generation/`；用于留档但不应混入正式结果的测试或迁移压缩包放在相应测试类别的 `archives/`。

测试运行产生的日志分别进入 `logs/tests/<category>/`，正式 RL 日志进入 `logs/reinforcement_learning/`、`logs/structure_generation/reinforcement_learning/` 或 `logs/metric_evaluation/reinforcement_learning/`。

## 当前 RL 约束

当前 RL 正式任务主要关注 MP-20 的形成能（FE）和带隙（BG）。训练 reward/probe checkpoint 与最终独立 evaluator 必须分离；随机种子和输入模型来源必须写入 manifest。当前奖励模型登记见 `../conf/rl/components/rewards/registries/mp20.yaml`。
