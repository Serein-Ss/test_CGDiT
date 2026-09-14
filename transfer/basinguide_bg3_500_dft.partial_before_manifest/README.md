# BasinGuide 500 结构 DFT 迁移包

本包可从任意目录执行，不依赖原服务器绝对路径，不需要 GPU、PyTorch 或 CGDiT 模型权重。500 个候选已经生成，seed 42 初态带隙预测位于 `campaign/cohort.json`。

## 在新服务器运行

1. 解压，在新服务器的 `cgdit` 环境安装 `requirements.txt`。可先执行 `conda create -n cgdit python=3.11`，然后 `conda activate cgdit`、`python -m pip install -r requirements.txt`。
2. 编辑 `site_env.sh`，填写本机 conda 激活命令和 VASP/MPI 模块。VASP 需本机合法可用；原服务器 VASP 可执行文件不包含在包内。
3. 验证：`python run_dft.py --check`。
4. 查看一个样本的实际输入：`python run_dft.py --index 0 --prepare-only`。文件生成在 `prepared_inputs/0000/`，不会启动计算。
5. 提交：`PARTITION=你的CPU分区 ACCOUNT=你的账户 NTASKS=20 bash submit.sh`。无需账户时省略 ACCOUNT。默认一次提交全部 500 项、不限制并发；如希望同时运行 100 项，可设置 `MAX_CONCURRENT=100`。每结构只用一个节点，默认 20 MPI 进程、64 GB 内存；可用 `NTASKS`、`MEMORY`、`WALLTIME` 修改资源。INCAR 的 NCORE=4，建议进程数为 4 的倍数，并按新机器性能选择。
6. 非 Slurm 服务器也可逐结构执行：`NPROCS=20 python run_dft.py --index 0`。默认启动命令为 `mpirun -np 20 vasp_std`，也可设置 `VASP_COMMAND` 为本机所需命令。

提交脚本记录真实任务编号，防止无意重复提交。若需要测试一小批，可用 `ARRAY=0-1`，之后核对记录再提交剩余索引。部分任务重启前应把其 `campaign/samples/编号/dft` 移到备份位置，避免覆盖旧计算；完整成功的任务会自动跳过。

## 计算协议与结果

每个候选按原协议运行：生成态 SCF 单点 → 从生成态直接优化原子和晶胞（最多 2×200 步）→ 终态 SCF → 使用终态电荷密度的非自洽路径能带。详细设置见 `original_execution.md`，主指标为终态均匀网格 SCF 带隙落在 2.5—3.5 eV 内。

`potentials/` 包含这 500 个结构实际需要的原始 POTCAR 变体，去重保存并记录 SHA256；仅供用户在已有 VASP 授权的服务器内部使用，不应上传公开仓库。若目的服务器更换赝势/泛函/U 参数，则不再是本批同一计算协议。

新服务器的终态 DFT 结构不额外运行 GPU 带隙预测器，标记为 `deferred_to_original_seed42_predictor`；真实 DFT 带隙、初态 seed 42 预测及它们的比较不受影响。回传 DFT 终态后，可在原服务器用相同预测器补算。

完成后自动生成 `campaign/paired_results.csv` 和 `campaign/summary.json`；也可手动执行 `python run_dft.py --collect`。失败和缺失样本仍保留在 500 个分母中。初态/终态 XML、OUTCAR、CONTCAR、能带及逐阶段结果保存在各样本的 `dft/` 内。

## 快照与旧任务

原服务器现有 DFT 和 MLIP 结果保存在 `reference_results/`，包含用户要求停止时未完成的计算，只作参考快照。新服务器默认从冻结的 500 个生成态重新计算，避免把不完整弛豫误认为收敛。各阶段的 `result.json` 和收敛标记用于区分已完成结果与中止文件。

已按用户要求取消原服务器 MLIP `669415`、DFT 数组 `669416` 和汇总 `669417`，没有删除任何原始结果。新服务器结果不会自动包含原服务器尚未完成的 MLIP 分支。

## 计算完成后回传

将整个 `basinguide_bg3_500_dft` 目录打包回传，保留 campaign、logs 和 submitted_jobs.txt。在原服务器解压后放到：

`/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/output/basinguide/returned/basinguide_bg3_500_dft/`

在新服务器该目录的父目录执行 `tar -czf basinguide_bg3_500_dft_completed.tar.gz basinguide_bg3_500_dft`。将完成包传回原服务器 `output/basinguide/returned/` 并在此处解压。结果路径应形如 `returned/basinguide_bg3_500_dft/campaign/samples/0000/dft/result.json`。不要覆盖原始实验目录；回传后通知我检查收敛并分析。

`transfer_manifest.json` 记录全部包内原始文件 SHA256，解压后先执行校验。`original_submission.json` 仅用于追溯，不能作为新服务器任务编号使用。运行脚本之外的 `scripts/basinguide_diagnostic.py` 是冻结的原协议源码；通过 `run_dft.py` 使用时，原路径和 MPI 命令已由便携适配器替换。
