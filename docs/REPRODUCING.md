# 从干净克隆复现生成模型与性质预测器

_发布范围：2026-10-02；从原始数据重新训练，不包含历史权重或agentic子项目。_

---

## 📦 输入与范围

`data/manifest.json`固定25个CSV和1个划分JSON的SHA256、大小及CSV行数。包含MP-20、C2DB、Carbon-24、Perov-5、MPTS-52、MAGNDATA和测试数据；不重划分、不改标签。

`data/*/*.pt`是[CrystDataset](../cgdit/pl_data/dataset.py)根据CSV自动重建的图缓存，不是必需的下载输入；`data/*/analysis/`是可再生分析。首次训练会比命中缓存时慢，不应让多个新任务同时写同一份缓存。

`output/`、`logs/`、`wandb/`、`tmp/`、`transfer/`、`agentic/`不发布。原始数据遵守各数据集原有来源和使用限制；本次记录输入快照，不重新授予数据许可。

## 🔧 环境与检查

以下命令从仓库根目录运行。先克隆`newton`并进入目录：

```bash
git clone --branch newton git@github.com:Serein-Ss/test_CGDiT.git
cd test_CGDiT
uv sync --locked
export PROJECT_ROOT="$PWD"
export HYDRA_JOBS="$PROJECT_ROOT/output"
export WANDB_DIR="$PROJECT_ROOT/wandb"
export WANDB_MODE=offline
mkdir -p "$HYDRA_JOBS" "$WANDB_DIR" logs
uv run --locked python -m scripts.cli.data.verify_inputs
uv run --locked python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

也可复制`.env.template`为`.env`并填写新服务器路径。不要复制旧`.venv`，不要上传密钥。主要环境是Python 3.10、Torch 2.4.1/CUDA 12.4及匹配PyG扩展，详见锁文件；Conda备选为`environment.yml`，不是uv锁文件的逐包等价承诺。

在分配到的GPU节点训练；登录节点不暴露GPU不代表环境损坏。Slurm分区、账户、Conda位置均需按新服务器改。不要照抄本服务器的作业编号或绝对路径。

## 🧪 生成模型训练、验证与测试

MP-20基础模型：

```bash
uv run --locked python -m cgdit.run data=mp_20 \
  model=experiments/exp_mp20_base expname=mp20_base_seed42 \
  train.random_seed=42 data.preprocess_workers=4 logging.wandb.mode=offline
```

训练入口[run.py](../cgdit/run.py)先`fit`，按配置周期验证，再对最佳checkpoint运行`test`。输出包含`.hydra/`、`hparams.yaml`和checkpoint；保留完整运行目录供后续加载。

| 模型 | `model=`配置 |
| --- | --- |
| Base | `experiments/exp_mp20_base` |
| FE CFG | `experiments/exp_mp20_fe` |
| BG CFG | `experiments/exp_mp20_bg` |
| FE+BG CFG | `experiments/exp_mp20_fe_bg` |

用不同`expname`分别执行。C2DB、Carbon-24等须同时改`data=`和匹配的`conf/model/experiments/`配置，不能只改数据路径。预处理worker数按CPU配额设置；不要为赶时间擅自改扩散步数或模型结构。

## 📊 性质预测器训练与独立调用

先训练FE，再以同样方式训练BG；本例不需要历史生成模型权重：

```bash
uv run --locked python -m cgdit.run data=mp_20_surrogate \
  model=property_predictors/m3gnet/regression \
  data.prop=formation_energy_per_atom expname=mp20_predictor_fe_seed42 \
  train.random_seed=42 data.preprocess_workers=4 logging.wandb.mode=offline

uv run --locked python -m cgdit.run data=mp_20_surrogate \
  model=property_predictors/m3gnet/regression \
  data.prop=band_gap expname=mp20_predictor_bg_seed42 \
  train.random_seed=42 data.preprocess_workers=4 logging.wandb.mode=offline
```

M3GNet在训练后测试阶段输出预测值和真值`test_preds.npy`、`test_targets.npy`。输入统计在`conf/data/mp_20_surrogate.yaml`，预测返回物理单位；不要混用不同性质的checkpoint。

```bash
# 将占位路径换成上一步真实生成的checkpoint；仅加载可信权重。
uv run --locked python -m scripts.cli.evaluation.predict_property \
  --ckpt 'output/singlerun/日期/FE运行目录/epoch=实际值.ckpt' \
  --input data/mp_20/test.csv --prop formation_energy_per_atom \
  --output output/fe_test_predictions.csv --device cuda
```

BG使用BG权重并传`--prop band_gap`。输入CSV至少含`material_id`和`cif`；没有性质真值时传`--no_gt`。预测器对齐图可用`python -m scripts.cli.visualization.plot_property_results --run-dir 实际预测器目录`。

批量入口`uv run --locked bash submit_python/run_all_mp20_predictors.sh`会顺序训练FE、BG、EH三种预测器；本轮已移除它对预存`.pt`缓存的强制要求。

## 🔄 从头生成与结果评价

训练完后，将`BASE_RUN`设为包含`hparams.yaml`和checkpoint的真实模型目录。以下4096样本是正式评价示例，不是本次发布时执行的任务：

```bash
BASE_RUN='output/singlerun/日期/Base运行目录'
uv run --locked python -m scripts.cli.generation.generate \
  --model_path "$BASE_RUN" --ab_initio --use_empirical_prior \
  --batch_size 16 --num_batches_to_samples 256 --max_atoms 20 --seed 42 \
  --label abinitio_empirical_base_n4096_seed42

uv run --locked python -m scripts.cli.evaluation.evaluate_metrics \
  --root_path "$BASE_RUN" --label abinitio_empirical_base_n4096_seed42 \
  --tasks gen --gt_file data/mp_20/test.csv --seed 42 --num_workers 4 --calc_prop false
```

CFG生成应改用对应条件模型并传`--condition band_gap=2.0`等已训练的条件；联合模型可重复传入`--condition formation_energy_per_atom=-1.5 --condition band_gap=2.0`。Base不传CFG条件。Template仅用于工程测试，不代替从头生成对照。

首次仅检查链路时，可用`--num_batches_to_samples 1`并把标签样本数改为16；但不能据此做正式统计。通用结构评价默认需要1000个有效结构，少样本可能不满足该要求。

FE/BG生成性质评价：

```bash
uv run --locked python -m scripts.cli.evaluation.evaluate_generated_properties \
  --root_path "$BASE_RUN" \
  --fe_run 'output/singlerun/日期/FE预测器运行目录' \
  --bg_run 'output/singlerun/日期/BG预测器运行目录' \
  --output_dir output/generated_property_evaluation --num_workers 4
```

该入口只自动发现正式`*_n4096_seed42.pt`文件，必须保留上述命名和真实样本数。结构生成和各模型评价分别保存在`generated_structures/`与`evaluations/`；文件索引由项目输出路径工具维护。

MP相图、MatGL等额外稳定性评价需要`uv sync --locked --extra stability`、相应外部模型/数据库访问和参考快照。DFT另需合法VASP及赝势。它们不属于仅靠原始训练CSV就能重建的外部资产。

## ✅ 验收边界与故障排查

- 数据完整性：`uv run --locked python -m scripts.cli.data.verify_inputs`须通过；输入变化应重建并审核manifest，不绕过校验。
- 配置与单元测试：`uv run --locked pytest`；可选MatGL相关测试需安装`stability`额外依赖。
- DFT集成测试需要额外的seekpath及合法POTCAR；缺失时明确跳过，不视为DFT验证通过。绘图单元测试使用合成夹具，不要求旧训练结果或论文原图。
- 模型加载：必须保留新训练的`hparams.yaml`与checkpoint配套；原服务器旧模型不随Git上传。
- RL奖励/基座配置中的旧模型路径，需指向新训练的运行目录后再使用；本指南的核心训练/预测不依赖agentic。
- 这份发布提供从头复现所需代码、配置和原始输入，不声明已经在另一台服务器重跑完整训练，也不承诺跨硬件逐位相同或历史论文数值完全相同。

本文依照操作指南结构组织，发布时验证数据哈希、配置、相关测试及干净克隆；长时间GPU全训练和正式生成需在目标服务器执行。

2026-10-02发布验收：在不含output、logs、wandb、tmp、transfer或agentic的隔离发布副本中，使用本服务器现有uv解释器执行全套测试，结果为173通过、1跳过（可选DFT的seekpath未安装）；26份输入哈希全部匹配。该验证没有重新安装环境，也没有重跑长时间GPU训练。原工作目录的历史BasinGuide日志会触发输出布局检查，未为测试删除这些历史数据。
