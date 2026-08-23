# 材料科学中的生成式扩散模型版图

## 1. 范围与口径

“材料科学中的所有扩散模型”横跨显微图像、谱学、微结构、动力学轨迹、分子、蛋白质和合成文本，无法用一个有限清单绝对穷尽。本调研采用与 CGDiT 研究问题一致的口径：

- 主纳入：无机晶体、周期材料、原子结构的生成、CSP、逆向设计、条件生成和结构修复；
- 次纳入：能改变竞争基线的 flow matching、Bayesian flow、LLM/自回归晶体生成；
- 排除：只把 diffusion 用于图像增强、分割、谱图去噪或一般机器人控制的工作；
- 技术判断优先依据论文主页、正式出版页、OpenReview/PMLR/CVF 或作者 arXiv，而非二手综述。

因此，下面是截至 2026-08-16 的**可复现高召回核心版图**，不是对全世界未索引工作的法律意义“穷尽证明”。

## 2. 主线发展

| 时间 | 工作与状态 | 表示/任务 | 核心贡献 | 与 CGDiT 的关系 |
|---|---|---|---|---|
| 2022 | [CDVAE, ICLR 2022](https://iclr.cc/virtual/2022/poster/7063) | VAE 潜变量 + 周期晶体图；扩散式 score decoder | 奠定周期材料生成与性质优化的公开基线 | CGDiT 的代码生态和评价协议可追溯到 CDVAE/DiffCSP 系列 |
| 2023 | [DiffCSP, NeurIPS 2023](https://openreview.net/forum?id=DNdN26m2Jk) | 晶格 + 周期分数坐标联合扩散；CSP 与 de novo | 将周期-E(3) 等变性直接纳入联合晶体扩散 | CGDiT 文档明确说明建立在 DiffCSP 思路上，重叠度高 |
| 2024 | [GemsDiff, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/30224) | 晶格与坐标的等变向量场 | 以几何向量场改进晶体生成 | 属于坐标/晶格 score 建模的直接基线 |
| 2024 | [Con-CDVAE, Computational Materials Today](https://doi.org/10.1016/j.commt.2024.100003) | 性质条件 CDVAE | 在晶体生成中显式加入性质约束 | 覆盖“性质条件生成”的一般主张 |
| 2024 | [Cond-CDVAE, npj Computational Materials](https://www.nature.com/articles/s41524-024-01443-y) | 组成、压力条件的 CSP | 在大规模局域极小结构数据上做压力/组成条件生成 | 覆盖条件 CSP；与 CGDiT 多条件路线相邻 |
| 2024 | [DiffCSP++, ICLR 2024](https://openreview.net/forum?id=jkvZ7v4OmP) | 空间群约束的晶格与 Wyckoff 坐标 | 将空间群约束直接置入生成过程 | 与 CGDiT 的空间群投影、anchor/对称操作最接近的上游基线之一 |
| 2024 | [UniMat / Scalable Diffusion, ICLR 2024](https://arxiv.org/abs/2311.09235) | 统一材料张量表示 | 探索可扩展、跨材料空间的扩散生成 | 对“通用材料扩散表示”构成先例 |
| 2024 | [FlowMM, ICML 2024](https://proceedings.mlr.press/v235/miller24a.html) | 晶格、周期坐标、二进制原子类型的 Riemannian flow matching | 用材料流形上的 flow matching 减少采样步数 | 严格说不是 diffusion，但必须作为效率强基线；CGDiT 已有 Diff2Flow 实验分支 |
| 2025 | [MatterGen, Nature](https://www.nature.com/articles/s41586-025-08628-5) | 原子类型、坐标、晶格联合扩散；Adapter 条件化 | 跨元素周期表生成并支持化学、对称、标量性质及多条件约束，含实验合成 | 覆盖联合三通道、Adapter、性质/对称/多条件的一般创新声明 |
| 2025 | [SymmCD, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/3a14ae9951e8153a8fc814b5f506b5b7-Paper-Conference.pdf) | 非对称单元 + 对称变换的混合扩散 | 显式生成晶体对称性并推广到不同空间群 | 与 CGDiT 的 anchor/对称操作表示直接相邻 |
| 2025 | [WyckoffDiff, ICML 2025](https://proceedings.mlr.press/v267/ekstrom-kelvinius25a.html) | 空间群/Wyckoff protostructure 的离散扩散 | 对称性由表示保证，并引入对称敏感评价 | 覆盖“离散扩散 + Wyckoff 对称”的主张，但不生成完整连续坐标 |
| 2025 | [SGEquiDiff, NeurIPS 2025](https://openreview.net/forum?id=NWP8KYKC0c) | 空间群等变坐标扩散 | 构造空间群等变扩散与不变似然 | 对空间群下概率/等变性的理论主张构成直接基线 |
| 2025 | [SCIGEN, Nature Materials](https://www.nature.com/articles/s41563-025-02355-y) | 在扩散生成中施加 honeycomb、kagome 等几何约束 | 结构约束驱动量子材料发现，并实验合成候选 | 表明“硬结构约束 + 扩散 + 实验验证”已经成立 |
| 2025 | [TGDMat, arXiv](https://arxiv.org/abs/2503.00522) | 文本引导的联合扩散 | 文本语义控制周期材料生成 | 对多模态条件生成构成相邻先例 |
| 2025 | [InvDesFlow-AL, npj Computational Materials](https://www.nature.com/articles/s41524-025-01830-z) | 条件扩散 + 主动学习工作流 | 在功能材料逆向设计中闭环更新 | 表明“生成—性质预测—主动学习闭环”不宜单独作为新意 |
| 2026 | [ChargeDIFF, Nature Communications](https://www.nature.com/articles/s41467-026-73985-2) | 原子类型 A、坐标 X、晶格 L、电子电荷密度 C | D3PM 处理元素、DDPM 处理晶格/电荷、wrapped score 处理周期坐标 | 与 CGDiT 的异构三通道最接近；新增电荷密度使其任务更广 |
| 2026 | [DiffCrysGen, npj Computational Materials](https://www.nature.com/articles/s41524-026-02147-1) | 完整晶体端到端 score diffusion | 强调统一、加速的无机晶体生成 | 对“统一端到端晶体扩散/效率”构成新基线 |
| 2026 | [XtalPaint, npj Computational Materials](https://www.nature.com/articles/s41524-026-02090-1) | score-based inpainting 与 H 原子重建 | 扩散用于部分结构补全和缺失轻元素恢复 | 若 CGDiT 扩展为修复/补全，需纳入基线 |
| 2026 | [Amorphous materials diffusion, npj Computational Materials](https://www.nature.com/articles/s41524-025-01901-1) | 非晶原子结构 | 将生成扩散拓展到非周期长程有序之外 | 不直接竞争当前晶体主线，但界定“材料结构”范围 |
| 2026 | [DAO, Nature Communications](https://www.nature.com/articles/s41467-026-72362-3) | DiffCSP 式生成模型 + 配套能量预测模型 | 生成/预测 Siamese foundation models | 对“生成器 + 性质/能量预测器协同”构成先例 |
| 2026 | [WyckoffDiff-Adapter, Computational Materials Science](https://doi.org/10.1016/j.commatsci.2026.114789) | 离散 WyckoffDiff + MatterGen 式 Adapter | 同时控制空间群、化学体系和能量高于凸包 | 对 CGDiT 的对称 + 多性质 Adapter 主张重叠很强 |

## 3. 按科学问题分类

### 3.1 表示与物理约束

- 周期坐标/晶格联合建模：DiffCSP、MatterGen、FlowMM、DiffCrysGen；
- 显式空间群/Wyckoff：DiffCSP++、SymmCD、WyckoffDiff、SGEquiDiff、WyckoffDiff-Adapter；
- 离散元素扩散：WyckoffDiff 及采用 D3PM 的 MatterGen/Chemeleon1/ChargeDIFF 类框架；
- 额外物理场：ChargeDIFF 将电子电荷密度纳入生成；
- 特定结构约束：SCIGEN；
- 非完整结构任务：XtalPaint 的 inpainting/reconstruction。

### 3.2 条件控制与逆向设计

- 标量性质、化学体系、多条件 Adapter：MatterGen；
- 组成/压力：Cond-CDVAE；
- 空间群 + 化学体系 + 稳定性：WyckoffDiff-Adapter；
- 文本条件：TGDMat；
- 主动学习与闭环筛选：InvDesFlow-AL；
- 约束稀有结构 motif：SCIGEN。

### 3.3 必须纳入但不属于严格 diffusion 的竞争路线

- [FlowMM](https://proceedings.mlr.press/v235/miller24a.html) 与 [CrystalFlow](https://www.nature.com/articles/s41467-025-64364-4)：flow matching，通常采样步数更少；
- CrystalFormer/WyckoffTransformer/LLM-CIF：自回归或文本生成；
- CrysBFN/SymmBFN：Bayesian flow；
- GFlowNet、遗传算法、随机结构搜索和离线 RL：目标优化基线，而非扩散模型。

如果论文只与 CDVAE/DiffCSP 比较而忽略 2024–2026 的 flow、显式对称生成和 RL 后训练模型，结论会明显过时。

## 4. 对 CGDiT 创新性的直接影响

### 已有强先例，不能直接作为主创新

- 联合生成晶格、坐标和原子类型；
- 周期边界、E(3)/空间群对称约束；
- D3PM 元素类型；
- 性质条件、CFG、Adapter、多条件生成；
- 对称非对称单元/Wyckoff anchor；
- 对比学习或 curriculum 本身；
- 用代理性质模型筛选或主动闭环。

### 仍可能形成贡献，但必须实证

- **同一个严格概率框架下处理投影晶格、torus 坐标和 D3PM 元素动作**；
- **只按空间群独立自由度计算策略概率、KL 与 advantage，避免对称复制和大晶胞偏置**；
- **从三通道异构动力学出发的信用分配，而不是给整条轨迹复制同一个优势**；
- 在相同奖励调用和高保真预算下，证明其优于 DiffCSP++-ReFT/MatInvent 类直接空间 RL 与 Chemeleon2 潜空间 GRPO。

当前文章范围和实验要求见 [CrystalPIRL文章执行计划.md](../CrystalPIRL文章执行计划.md)；宽口径创新判断见 [CGDiT_RL_complete_research_plan.md](../CGDiT_RL_complete_research_plan.md)。
