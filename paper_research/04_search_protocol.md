# 可复现检索协议与边界

## 1. 检索问题

本次检索拆成三个问题：

1. 与 CGDiT 直接竞争的晶体/周期材料生成扩散模型有哪些？
2. 用强化学习、奖励反传或偏好优化后训练生成式 diffusion/flow 的方法有哪些？
3. 晶体/材料领域已经有哪些直接 RL + diffusion/flow 工作，因而哪些“首次”声明已经失效？

检索截止为 **2026-08-16**。工作状态按该日期记录，预印本、Workshop、会议和期刊不混写为同一证据等级。

## 2. 检索源与优先级

### 技术结论优先级

1. 正式期刊/会议出版页：Nature Portfolio、PMLR、CVF、ICLR/NeurIPS proceedings；
2. OpenReview 论文页；
3. 作者 arXiv 页面；
4. 作者/机构官方 GitHub，仅用于代码状态和实现可用性；
5. OpenAlex 和综述只用于发现候选与追踪引用，不作为关键算法结论的唯一依据。

本环境没有可调用的 PubMed/CrossRef/Scopus 学术 MCP，因此按 `nature-academic-search` 的回退规则使用了网页检索和技能自带 OpenAlex 脚本。OpenAlex 结果噪声较高，所有进入主表的关键方法均尽量回到论文原始页面核验。

## 3. 核心检索式

### 材料扩散

```text
(crystal OR crystalline OR materials OR periodic material)
AND (diffusion model OR score-based model OR denoising diffusion)
AND (generation OR inverse design OR crystal structure prediction)

(space group OR Wyckoff OR symmetry OR periodic boundary)
AND diffusion AND crystal generation

(conditional OR property-guided OR text-guided OR inpainting)
AND diffusion AND materials generation
```

以已知种子 CDVAE、DiffCSP、DiffCSP++、MatterGen、SymmCD、WyckoffDiff、SCIGEN 进行前向/后向名称追踪，并追加 2025、2026 年过滤。

### 通用 diffusion + RL

```text
(diffusion model OR flow matching OR discrete diffusion)
AND (reinforcement learning OR policy gradient OR PPO OR GRPO)
AND (fine-tuning OR post-training OR alignment)

diffusion AND (reward backpropagation OR direct preference optimization)

discrete diffusion AND (reinforcement learning OR policy gradient OR DPO)
```

以 DDPO、DPOK、DRaFT、AlignProp、Diffusion-DPO 为种子，追踪 continuous-time、step-wise credit、GRPO、flow 和 discrete diffusion 分支。

### 材料/晶体 RL

```text
(crystal OR materials) AND diffusion AND reinforcement learning
(crystal generation) AND (PPO OR GRPO OR policy gradient)
(crystal diffusion) AND (formation energy OR stability OR synthesizability) AND reward
```

并逐名核验：RLFEF、MatInvent、Chemeleon2、OMatG-IRL、topological-material ReFT、MatFlow、synthesizability-aware materials generation。

## 4. 纳入标准

- 论文直接生成或编辑材料/晶体原子结构，且 diffusion/score/flow 是生成核心；
- 或论文直接研究如何用 RL/奖励/偏好后训练生成式 diffusion；
- 或虽非 diffusion，但会成为当前研究主张的必要强基线；
- 至少能获得标题、方法摘要和稳定的原始来源链接；
- 同一工作的 Workshop、arXiv、期刊扩展版合并记录，并说明状态演化。

## 5. 排除或单列标准

- 仅用 diffusion 做显微图像、谱图、分割、去噪而不生成材料结构；
- 机器人/控制中的 Diffusion Policy、Diffuser、DPPO：它们是“diffusion 作为 RL policy”，不是“RL 后训练生成模型”；
- LLM/CIF、自回归、GFlowNet、传统 RL CSP：单列为任务竞争者；
- 只有二手博客、新闻或无稳定全文来源的条目不进入核心结论；
- 只在参考文献中出现、无法核验方法的条目放入待跟踪项，不据此声称覆盖。

## 6. 去重规则

按 DOI > OpenReview ID > arXiv ID > 规范化标题去重。以下视为同一研究线：

- MatInvent ICLR 2025 AI4Mat 与 2025-11 扩展预印本，但分别保留状态和新增作者/实验信息；
- Chemeleon2 预印本与 Nature Machine Intelligence 2026 正式版，以正式版为主；
- topological-material ReFT 的 2025 arXiv 与 Nature Communications 2026 正式版，以正式版为主；
- OMatG-IRL 的 arXiv 与 OpenReview，以 OpenReview 评审状态为主。

## 7. 完整性风险

- “所有相关工作”只能在明确数据库、检索式、语言和截止日期内成立；新预印本、未公开审稿稿件、不同术语命名的工作可能漏检；
- 2026 年文献变化快，投稿前必须再次执行增量检索；
- 预印本/Workshop 的主张不能与同行评议期刊等权；
- 网页索引可能把接受年份、上线年份和正式卷期混淆，本表以论文原始页面显示状态为准；
- 材料扩散范围若扩展到分子、聚合物、微结构、谱学和合成文本，需要另做专项检索，不能从本报告外推“全材料科学已穷尽”。

## 8. 投稿前增量检查清单

- [ ] 重跑上述三组检索式，限定最近 12 个月；
- [ ] 查询 H1/H2 的关键词组合：`symmetry quotient`, `mixed discrete continuous`, `D3PM`, `Wyckoff`, `policy gradient`, `credit assignment`；
- [ ] 检查 MatInvent、MatFlow、OMatG-IRL 的新版/正式发表状态；
- [ ] 检查 Chemeleon2 和 DiffCSP++-ReFT 的 citing papers；
- [ ] 对每个“first/novel”句子单独建立最近邻工作表；
- [ ] 更新 `literature_matrix.csv` 的状态、DOI 和备注；
- [ ] 把未复现的基线明确写成限制，不用弱替代基线暗示优越性。
