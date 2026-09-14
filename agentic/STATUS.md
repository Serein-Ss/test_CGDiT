# 当前进度：第八阶段（2026-09-14）

完成Fe–Ni结构可用性及现有管线配对的MP20重合审查。

- 对46513条元数据预筛，解析14条目标体系结构，原6个CSV未改。
- Mn4N/Mn3AlC均匹配本地MP20 train中的结构；不能作为预训练未见结构评价。
- Fe4N在MP20 train、Fe3NiN在val；理想结构不能替代实际氮缺位样品。
- 精确N0.92计量场景至少123原子，超出现有100原子配置且缺位排列未知。
- 保留Fe–Ni组成证据，暂缓精确结构监督；新增正式训练/独立评价对均0。
- 39项测试通过，无API、DFT或训练。

报告：[结构路线审查](docs/STAGE8_STRUCTURE_ROUTING.md)。
决定：[routing_decision.json](results/stage8/routing_decision.json)。

下一步优先寻找不同来源组、满占位且结构明确的实验比较，先核查结构与预训练重合再批量提取。前阶段记录见docs/STATUS_STAGE7.md。
