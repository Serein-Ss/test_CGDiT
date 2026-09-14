# 当前进度：第五阶段（2026-09-14）

已完成真实跨组成扩散前向与梯度接口验收。

- 标准CGDiT已有元素类型扩散，不需要仅因固定类型采样而另建组成模型。
- 两个端点已补齐对称轨道数据；CPU小型随机模型前向/反向通过，元素和晶格头梯度非零，权重未更新。
- 找到并解决端点对称化后的行序差异，保存位点映射；本例特殊位置坐标自由度为0。
- 保存后续逐结构偏好损失接口约束，尚未实现或训练偏好模型。
- 28项回归测试通过。本轮无API调用、无DFT；独立数值配对没有增加。

主报告：[第五阶段](docs/STAGE5_DIFFUSION_INTERFACE.md)。
运行结果：[diffusion_smoke.json](results/stage5/diffusion_smoke.json)。
接口约束：[training_interface_contract.json](results/stage5/training_interface_contract.json)。

下一步：逐结构去噪评分验收、预训练检查点核验和Fe–Ni原文样本提取。前阶段状态见 docs/STATUS_STAGE4.md。
