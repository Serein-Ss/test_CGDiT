# 第六阶段：逐结构去噪评分与预训练基座验收

实验日期：2026-09-14。本次续执行补齐上次因自动审批额度限制未写入的文档和完整性检查，保留已完成的运行结果。应用 karpathy-guidelines，新增文件均在 agentic 内。

## 已实现与验证

`scripts/pair_denoising.py` 实现显式配对加噪、逐结构去噪评分、冻结参考的偏好损失。`scripts/stage6_scoring_acceptance.py` 执行数值和权重验收。

每端点返回 lattice、coordinate、atom、total 四项，保留长度2的端点向量。归一化沿用原项目：晶格6分量均值、坐标节点/分量均值、原子类型独立轨道均值，以及模型 cost 权重。指定时间步的外部权重为1，三时间点检查不是完整时间积分估计。

加噪使用相同时间步、共享晶格噪声、按位点映射对应的坐标噪声和轨道均匀随机数；掩码概率读取现有 MaskDiffusion。策略和参考模型复用同一份加噪输入。映射必须为双射并保持周期位置、轨道关系。重复索引和错配被拒绝。

当前接口限定CPU、无条件模型、两个同大小框架。两端点没有分别使用745 K和272 K作为条件。本次只验证现有特殊位置结构；一般位置操作方向和不同框架尚未验证。

偏好损失为：

`softplus(beta * [(E_policy_w-E_reference_w) - (E_policy_l-E_reference_l)])`

E为去噪代理量。此式不是已证明的精确DPO对数似然，也不是Tc回归损失。参考分数停止梯度。

小型随机模型在 t=1、10、20 均通过：

- 相同种子逐张量重放一致，对应轨道掩码一致。
- 四项评分有限，端点均值与项目原 loss_fn 逐项一致，rtol=1e-5、atol=1e-6。本例两端点原子数相同。
- 同权重策略/参考评分完全一致，初始偏好损失为 log(2)。
- 实际偏好反向传播成功；元素头梯度范数约0.018782、0.018770、0.113134，冻结参考无梯度。
- 数值检查确认梯度方向正确，交换标签反转偏好，拒绝整批标量输入。
- 优化器更新0次，策略参数逐项未变。

新增5项回归测试，与已有测试合计33项通过。日志为 `logs/stage6_tests.log`；完整运行验收为 `results/stage6/scoring_acceptance.json`。t=10实际加噪输入另存 `explicit_corruption.json`。

## 预训练基座

由 `conf/rl/components/runtime/base.yaml` 定位到：

`output/singlerun/2026-06-27/00-32-50-mp20_base/epoch=869-step=737760.ckpt`

普通搜索会忽略输出目录；本轮检查实际运行路径后找到该基座。它是MP20通用晶体生成模型，非Tc训练模型。文件148184178字节，epoch869、global_step737760，SHA256为：

`faa7fd6c8bca092f13f98fa21350a672df3b146acbebd09b481d35cf9d214fb5`

先静态检查序列化类型，再显式允许所需OmegaConf类和内建类型，保持 `torch.load(weights_only=True)`。按配套 hparams 构建当前 Diffusion，`strict=True` 加载成功，无缺失或多余权重键。检查点原文件未改写，路径与配置/权重哈希保存在 `checkpoint_audit.json`。

冻结基座在 t=1、500、1000 的两端点评分均有限，并与原损失均值一致。评分次序会随探测时间步改变，不能以单步去噪能量给Tc排序。此结果证明接口兼容，未证明高Tc知识或生成质量。

MP20预训练与未来评价材料的重合尚未审查。MAGNDATA内部无端点重复，不等于预训练层面无泄漏。

## 复现与边界

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /share/home/xlzou/Anaconda3/envs/cgdit/bin/python agentic/scripts/stage6_scoring_acceptance.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /share/home/xlzou/Anaconda3/envs/cgdit/bin/python -m unittest discover -s agentic/tests -v
```

本阶段无API、DFT、生成新材料或权重更新。可选GLIBC扩展警告保存在 warnings.json；此次CPU路径成功不代表GPU训练已验证。原项目数据和模型代码的哈希校验见 integrity.json。

下一步优先扩展独立、可溯源的实验比较，并审查MP20预训练重合，再开展有留出评价的偏好训练。目前仅一个管线配对，单对拟合不能作为泛化证据。
