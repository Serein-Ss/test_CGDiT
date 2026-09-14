# 当前进度：第二阶段（2026-09-14）

已完成本轮有界核查与扩展 API 提取；30 条试点的完整科学核验尚未完成。

## 可查看的成果

- [逐条核查表](docs/PILOT30_REVIEW.md)，另有 [CSV](data/curation/pilot30_review.csv) 和 [JSONL](data/curation/pilot30_review.jsonl)。
- [来源与引文](evidence/pilot30_sources_v2.json)，[检索范围和缺口](docs/STAGE2_RETRIEVAL.md)。
- [模型原始提取建议](data/curation/pilot30_intern_v2.jsonl)，响应缓存位于 results/pilot30_v2/。
- [家族可用性统计](results/pilot30_family_availability.csv)，[就绪性摘要](results/pilot30_readiness.json)。
- [研究方案](docs/tc_research/PROPOSAL.md)；上一阶段记录已保存为 [STATUS_STAGE1](docs/STATUS_STAGE1.md)。

## 本轮结果

30 条均有核查状态，27 条取得可追溯片段；LuMn2Ge2、Mn3As、TbMg 共 3 条仍缺少足以纳入的原始证据。取得片段不代表完成样品/结构匹配。

25 次实际 Intern-S1 请求，24 个案例通过 JSON 与引文匹配检查，1 个案例被拦截；另 3 个缺少证据的案例直接跳过。CeMnAsO 的 3 条记录共用一个文献案例，所以案例数与记录数不同。API 返回 token 总计 12,819，包含被拦截响应；后续复查全部使用缓存，新增请求 0。该数值不是费用说明。

被拦截的是 CeCo2P2：数值 440(5) K 对应正确，但模型把不连续文本重新拼接成所谓原句。缓存保留，未自动修补为合格结果。

在已知诊断案例上，第二版将 NdScO3 的转变类型保持 unknown，保留 Mn2Au >1000 K 和 KMnSb <295 K，将 CeMnAsO “below 7 K”标作上界，对 LaMnAsO 的 MnAs 杂相 317 K 返回无事件。仍发现 EuMnBi2 的 near 22 K 被标为 reported_value，已增加近似值丢失提示。这些是开发案例结果，不能当成独立测试集准确率。

助手事件审查统计：18 条 Néel 候选、1 条自旋重排与子晶格有序、2 条其他、8 条未知、1 条 Curie 候选（Mn4N）。部分为跨样品证据，仅说明来源事件，不能断言精确对应本地标签。CsMnF4 的铁磁结构有依据，但现有片段未核实其 9 K 转变值。

重要冲突包括：KMnSb 上界丢失；NaNdFeWO6 的约 25 K 在来源中属于 NaLaFeWO6，而 Nd 化合物约 21 K；Er2PtGe6 来源 4.9 K 与本地 9 K 不一致；LaMnAsO 的 317 K 存在 MnAs 杂相归属问题。数据标签仍保持原值，未静默纠正。

## 可训练性与执行决定

记录了 19 个候选家族线索，但已验证家族 0、可训练记录 0、可靠比较对 0。家族名称用于下一步检索，不能用相同空间群替代坐标/物相核对。测试集与验证集记录不进入训练比较对。

**当前不启动扩散偏好微调。** 这批诊断记录揭示了监督语义问题，但没有提供足够的同家族高 Curie 温度比较证据。下一轮应从训练集定向补充 FM/FiM 家族，而不是继续把高原始温度直接当高 Tc。

优先推进：追溯 Mn4N 745 K 的原始测量及相关氮化物家族；补齐原始 mcif、样品条件、磁转变上下两侧磁序和误差；低温 CsMnF4 可作为磁序区分对照，不作为高 Tc 成功案例。对另外 3 条未解决记录保留原始文献检索队列。只有出现足够的可比证据后，才开展小规模、固定预算的离线对齐实验。

16 项校验测试通过；8 份原数据快照与项目原文件逐一哈希一致。没有修改原训练标签，没有启动训练或 DFT，密钥不写入研究结果。
