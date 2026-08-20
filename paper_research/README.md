# CGDiT 文献调研与研究定位

> 检索截止：2026-08-16（Asia/Shanghai）
> 项目分支：`newton`
> 研究范围：与本项目直接可比的晶体/原子结构生成扩散模型，以及“用强化学习后训练生成式扩散模型”的工作。

> 当前执行依据：本目录只保存文献证据与检索边界；文章范围和实验决策以根目录的 [`OrbitPO文章执行计划.md`](../OrbitPO文章执行计划.md) 为准，宽口径研究背景见 [`CGDiT_RL_complete_research_plan.md`](../CGDiT_RL_complete_research_plan.md)。

## 结论先行

1. **当前 CGDiT 具备明确的工程组合价值，但尚不能仅凭现有代码主张方法学首创。** 空间群约束、周期坐标扩散、联合晶格/坐标/元素生成、D3PM 原子类型、性质条件和 Adapter 均已有相邻或直接先例。
2. **“首次把强化学习用于晶体扩散模型”已经不成立。** RLFEF、MatInvent、DiffCSP++-ReFT、Chemeleon2 和 OMatG-IRL 已覆盖形成能反馈、目标性质、多目标 GRPO、原子表示与潜空间、扩散与流模型等路线。
3. 当前最有希望形成可辩护贡献的方向是：**空间群商空间中的混合离散—连续策略优化**。具体是对 D3PM 元素转移、周期坐标转移和投影晶格转移给出一致且可复算的策略概率，并按独立 Wyckoff/anchor 与晶格有效自由度归一化；通道—时间信用分配当前仅作为内部效果检查。
4. 仅“接入 PPO/GRPO”“换成能量奖励”“增加多目标奖励”或“使用不确定性代理模型”都不足以单独构成创新；必须以严格概率定义、强基线、消融和独立物理验证支持主张。

## 文件说明

- [01_materials_diffusion_landscape.md](01_materials_diffusion_landscape.md)：材料/晶体生成扩散模型的发展脉络、任务分类、与 CGDiT 的重叠。
- [02_diffusion_rl_landscape.md](02_diffusion_rl_landscape.md)：扩散模型 RL 后训练的方法谱系及材料领域直接工作。
- [04_search_protocol.md](04_search_protocol.md)：检索式、纳入/排除标准、来源层级和“并非绝对穷尽”的边界。
- [literature_matrix.csv](literature_matrix.csv)：便于筛选和继续维护的结构化文献表。

## 使用建议

论文立项和实验执行应遵循根目录的 OrbitPO 计划；H2 与多保真闭环当前仅作为内部效果检查，不进入文章主结果。正式写“首次”之前，需按 `04_search_protocol.md` 再次做增量检索。
