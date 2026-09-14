# 当前进度：第四阶段（2026-09-14）

已完成首个文献配对的结构与真实 CGDiT 数据接口验收。

- Mn4N 745 K > Mn3AlC 272 K，温差473 K；保留原文磁序和历史表头解释。
- 两个5原子参考结构通过解析、占位及空间群容差检查。
- 扫描1284条原数据，解析失败0条；Mn4N只匹配train，Mn3AlC无重复，无val/test端点匹配。
- 实际 CrystDataset + CrystalNN + PyG 批处理成功：2图、10节点。
- 保存1个管线验收配对，正式训练配对仍为0；没有训练或新发现结论。
- 28项测试通过，原项目8个数据/说明文件哈希不变；本轮API调用0次。

报告：[第四阶段验收](docs/STAGE4_PAIR_ACCEPTANCE.md)。
配对：[pairs.jsonl](data/curation/stage4_pair/pairs.jsonl)。
端点：[endpoints.csv](data/curation/stage4_pair/endpoints.csv)。

下一步：扩展独立实验比较，并核查跨组成偏好与现有扩散模型的接口。现有固定原子类型采样不能直接承担这一跨组成配对的偏好学习。前一阶段记录见 docs/STATUS_STAGE3.md。
