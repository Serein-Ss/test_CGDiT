# 当前进度：第三阶段（2026-09-14）

已执行 Mn4N 家族原始文献追溯、结构模板核查和比较线索整理。

- 取得并解析 4 篇全文；1962 年表格已视觉核查。
- 保存 5 条原始实验观测，5 条候选比较中 3 条通过 50 K 文献温差筛选。
- 确认被引原文 Mn3AlC 与本地 Mn3AlN 的 C/N 身份冲突，隔离用于新比较，原标签不变。
- 重建 Mn4N 和 Mn3AlC 两份 5 原子参考 CIF，明确不是实验 mcif 或新发现。
- 2 次 Intern-S1 提取完成，识别出 408 K 为补偿事件；24 项测试通过。
- 可直接进入既有扩散训练集的比较仍为 0：外部结构与样本划分尚待核验，Ga=0.24 的标称有序表示还超出原子数配置。

主报告：[STAGE3_NITRIDE_REPORT](docs/STAGE3_NITRIDE_REPORT.md)。

结果入口：

- [1962 年原始观测](data/curation/stage3_1962_observations.json)
- [1962 年候选比较](results/stage3/1962_candidate_pairs.json)
- [2022 年比较线索](results/stage3/literature_pair.json)
- [身份冲突记录](results/stage3/identity_conflict.json)
- [参考结构清单](data/curation/stage3_reference_structures/manifest.json)
- [本地结构模板清单](results/stage3/local_antiperovskite_candidates.csv)

下一步优先用 Mn4N/Mn3AlC 验收结构与证据管线，并定向扩展反钙钛矿碳氮化物的实验比较；不将补偿温度、理论预测或任意混占位排布当作已验证 Tc 监督。前两阶段记录保存在 docs/STATUS_STAGE1.md 和 docs/STATUS_STAGE2.md。
