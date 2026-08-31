# 日志目录说明

本目录只保存程序运行时产生的文本日志、标准输出/错误、锁文件和启动元数据。模型、检查点、预测结果、生成结构与评估结果仍分别保存在 `output/`、模型运行目录或对应结果目录中。

## 目录划分

- `generative_model_training/`：扩散生成模型训练、续训与恢复日志。
- `structure_generation/`：使用基线模型或强化学习模型生成晶体结构的日志。
- `metric_evaluation/`：结构指标、性质指标、基线评估与分析任务日志。
- `property_predictor_training/`：形成能、带隙、凸包能、Tc 等性质预测器训练日志。
- `reinforcement_learning/`：PPO、GRPO、PIPO、CrystalPIRL 等强化学习训练日志。
- `tests/`：pytest、preflight、smoke test 和可复现性检查日志。
- `other/`：不能归入上述任务的编排日志、状态报告、导入清单和 Notebook 检查点日志。

## 二级目录约定

- 本地实验优先使用 `<category>/<task>/<run_id>/`。
- Slurm 标准输出和错误使用 `<category>/<task>/slurm/`。
- 跨多个任务的主进程输出放入 `other/orchestration/`，各子任务日志仍进入各自类别。
- 测试日志与正式训练或生成日志分开；同一 `run_id` 可同时出现在任务目录和 `tests/` 中。
- `run.lock`、`master.pid`、`launch_manifest.txt` 等运行控制文件跟随所属任务；跨类别编排状态放入 `other/orchestration/`。

## 当前主要路径

```text
logs/
├── generative_model_training/
├── structure_generation/
├── metric_evaluation/
├── property_predictor_training/
├── reinforcement_learning/
├── tests/
└── other/
```

日志数据默认被 Git 忽略；只有本说明文件进入版本控制。
