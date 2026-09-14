# 第八阶段：结构映射与预训练重合审查

2026-09-14。按 karpathy-guidelines 与 pymatgen 工作流完成本地检查，无API、DFT或模型训练。

## 主要结果

扫描本地MP20的45229条记录和MAGNDATA的1284条记录，共46513条。先用 elements、pretty_formula、CIF中的化学式元数据筛选 Fe–N、Fe–Ni–N、Mn–N、Mn–Al–C 四个体系，再解析命中的14条CIF。元数据完全不可用0条，候选CIF解析失败0条。

这是化学体系预筛后的候选结构审查，不是46513个CIF全部解析；若三个元数据来源同时错误，可能漏筛。筛选结果和源文件哈希可复现，不宣称绝对无漏检。

| 参考组成 | 本地MP20记录 | 划分 | 严格匹配 | 宽松匹配 |
|---|---|---|---|---|
| Mn4N | mp-505622 | train | 否 | 是 |
| Mn3AlC | mp-4593 | train | 否 | 是 |
| Fe4N | mp-535 | train | 否 | 是 |
| Fe3NiN | mp-567703 | val | 是 | 是 |

MAGNDATA中的Mn4N仍匹配train记录0.274_Mn4N。所筛体系中没有找到与标称Fe3.6Ni0.4N相同组成的候选。

匹配使用元素敏感ElementComparator和归一化组成，primitive_cell=True、attempt_supercell=False。严格参数：ltol=.01、stol=.05、angle_tol=1°、scale=False；宽松参数：ltol=.2、stol=.3、angle_tol=5°、scale=True。stol为归一化距离容差，不是Å。宽松匹配允许体积归一化和几何偏差，不等于实验样本完全相同。

Fe4N/Fe3NiN的参考为依据原文位点和晶格参数构造的理想满占位周期模板，坐标为分数坐标、晶格单位Å；仅用于名义结构重合检查，未赋Tc、磁矩或氧化态。没有把模板当作实际氮缺位样品。候选中3条出现有限精度坐标取整警告，均保存在structure_audit.json。

## 对评价的直接影响

Mn4N/Mn3AlC这一个管线配对的两端都与当前本地MP20训练结构宽松匹配，因此不应作为“预训练未见结构”的发现评价。它仍可用于接口验收；结构出现过不等于模型训练过相应Tc标签，也不自动禁止后续偏好微调。

这里核对的是检查点配置指向的现有本地数据，不是经版本锁定的历史训练快照；因此记录为预训练重合风险，不能无条件证明历史每一条输入。Fe3NiN位于val，不能称作训练已见，但也不能当成与整个预训练流程完全无关的新留出样本。

## Fe–Ni缺位结构能否进入训练

| 标称组成 | 理想计量的最小整数表示 | 若N占位精确取0.92 |
|---|---|---|
| Fe3.6Ni0.4N | 5个化学式单位，25原子 | 25个化学式单位，123原子 |
| Fe3NiN | 1个化学式单位，5原子 | 25个化学式单位，123原子 |

N0.92是把文献8%报告值视作精确值时的算术场景，不是经确定的精确样品组成。123原子是计量下界，不保证存在符合实验的有序晶胞；Ni/空位具体排列还未知。该场景超过现有100原子配置，也超过基座MP20的20原子训练数据范围。仅调高配置不能解决结构和监督证据问题。

因此本轮决定：**不把Fe–Ni的实测Tc强行附到满占位模板上；保留组成、转变事件和误差证据，暂不增加精确结构训练对。**Fe4N自身报告小于3%氮缺位，也没有唯一缺位构型。

## 主线调整及产物

新增正式结构训练对0，批准独立评价对0。下一轮优先选择来自不同来源组、满占位且有明确实验结构的材料比较，先检查结构可得性和MP20重合，再投入批量提取。Fe–Ni线索保留，暂不继续枚举任意缺位排布；若日后采用组成级或结构集合监督，需要另行验证方法，不能沿用当前单结构接口假装已解决。

- `results/stage8/structure_audit.json`：完整筛选、结构匹配、参数、警告和源文件哈希。
- `data/curation/stage8_structure_candidates.csv`：14条候选CIF副本、来源及划分，不附实验Tc标签。
- `results/stage8/vacancy_feasibility.json`：计量场景。
- `results/stage8/routing_decision.json`：后续用途和材料路线决定。
- `scripts/stage8_structure_audit.py`、`tests/test_stage8.py`：可复现审查与3项缺位计量回归测试。

全部39项测试通过。6个输入CSV在扫描前后哈希不变。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /share/home/xlzou/Anaconda3/envs/cgdit/bin/python agentic/scripts/stage8_structure_audit.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /share/home/xlzou/Anaconda3/envs/cgdit/bin/python -m unittest discover -s agentic/tests -v
```
