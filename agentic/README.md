# 高居里温度磁体：证据驱动生成研究

所有本方案的新产物集中在本目录。研究方案见 [PROPOSAL](docs/tc_research/PROPOSAL.md)，现有模型与标签审计见 [REPORT](docs/tc_audit_20260914/REPORT.md)，最新执行结果见 [STATUS](STATUS.md)。

- `data/raw/magndata/`：原训练数据的只读用途快照；原项目数据未改写。
- `data/derived/`：1284 条事件待核验清单、188 条优先队列、30 条试点。
- `evidence/pilot_sources.json`：文献短引文、链接、定位和证据层级。公式排版已转为纯文本，不能替代全文核验。
- `data/curation/intern_pilot.jsonl`：API 提取建议、引文检查及语义审查标记。
- `results/`：迁移校验清单、数据统计、API 探测、响应缓存及运行摘要。
- `scripts/`、`tests/`、`logs/`：可复现脚本、关键保护测试及运行日志。

使用官方远程接口 `https://chat.intern-ai.org.cn/api/v1/chat/completions`，模型固定 `intern-s1`。`apikey` 权限为 600，已加入本目录 `.gitignore`，不属于研究归档内容。分享或打包时必须排除该文件；不要直接打包整个目录。

在项目根目录运行：

```bash
python agentic/scripts/prepare_stage1.py
python agentic/scripts/run_evidence_pilot.py
python -m unittest discover -s agentic/tests -v
```

提取脚本依赖 `requests`，按输入和提示词哈希复用响应；当前输入限定为 5 个案例。首次运行会调用 API，重复运行使用缓存。它不会把提取结果写回训练标签，也不会启动训练或 DFT。提示词要求不等于模型已遵守要求；程序检查通过也不等于科学内容正确。


第二阶段复现（不覆盖第一阶段结果）：

```bash
python agentic/scripts/prepare_pilot30_review.py
python agentic/scripts/run_pilot30.py
python agentic/scripts/summarize_pilot30.py
python -m unittest discover -s agentic/tests -v
```

第二阶段输入为 28 个文献/未解决案例对应 30 条记录。25 个有证据案例调用 API，3 个跳过；响应按输入哈希缓存。来源注释由助手依据检索结果整理，当前不具备全自动检索闭环或独立金标准评估。


第三阶段入口：[原始证据报告](docs/STAGE3_NITRIDE_REPORT.md)。复现命令：

```bash
python agentic/scripts/stage3_nitride_feasibility.py
python agentic/scripts/stage3_source_pairs.py
python agentic/scripts/run_stage3_extraction.py
python -m unittest discover -s agentic/tests -v
```

需要 evidence/stage3/ 中已归档的原始论文文本。提取脚本复用缓存；参考 CIF 与实验 mcif 明确区分。

## 第四阶段

首个文献配对已完成结构查重与真实图加载，见 [验收报告](docs/STAGE4_PAIR_ACCEPTANCE.md)；配对位于 `data/curation/stage4_pair/`。它用于管线验收，尚未用于正式训练。

第五阶段已完成真实扩散前向/反向及位点对齐验收，见 [报告](docs/STAGE5_DIFFUSION_INTERFACE.md)。未进行正式训练。

第六阶段已实现逐结构去噪评分并严格加载MP20基座，见 [评分验收报告](docs/STAGE6_SCORING_REPORT.md)。未更新模型权重。

第七阶段已调用Intern-S1提取Fe–Ni氮化物原始实验表格，见 [证据报告](docs/STAGE7_FENI_EVIDENCE.md)。新增3条报告观测和2条通过间隔筛选的文献比较，尚未转为结构训练对。

第八阶段已完成结构映射及本地MP20重合审查，见 [路线报告](docs/STAGE8_STRUCTURE_ROUTING.md)。Fe–Ni暂保留为组成证据，避免向理想结构转移缺位样品的Tc。
