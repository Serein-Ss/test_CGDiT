# 当前进度：第六阶段（2026-09-14）

已实现逐结构去噪评分，并实际加载MP20预训练生成基座；中断的文档归档现已补齐。

- 配对加噪、输入重放、逐结构能量、冻结参考偏好损失均已实现。
- 与项目原损失逐项一致；随机小模型3个时间步的偏好反向传播通过。
- MP20基座epoch869/step737760严格加载成功，通过3个时间步的端点评分。
- 33项回归测试通过；无优化器更新、API或DFT，原项目模型/数据未改。
- 仍只有1个管线配对，尚无正式训练或独立高Tc性能证据。

报告：[第六阶段评分验收](docs/STAGE6_SCORING_REPORT.md)。
代码：[pair_denoising.py](scripts/pair_denoising.py)。
检查点：[checkpoint_audit.json](results/stage6/checkpoint_audit.json)。

下一步：扩展独立实验比较并审查预训练重合，再开展有留出评价的偏好训练。上一阶段状态见 docs/STATUS_STAGE5.md。
