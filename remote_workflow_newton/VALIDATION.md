# 验证记录

源码基础：newton ab8a752；新增工作流测试采用实际提交版本。

## 已执行

- 核心采样、条件处理、性质预测、生成性质评价测试：34 passed。
- 两路调度、资源不足禁止启动、失败不算完成、非有限预测拒绝、非有限生成报告：5 passed。
- 四种生成配置与FE/BG预测器配置经过Hydra组合验证。

## 真实GPU训练

本地GPU为RTX 3060 12 GiB，PyTorch 2.7.1。以64个20原子结构、train batch=32、eval batch=16、32位精度实际运行：

- base、FE、BG、FE+BG四个生成器全部退出0，保存epoch=0-step=2.ckpt，并完成最佳checkpoint测试。
- FE/BG预测器各seed=42/123/3407，共六个全部退出0，保存checkpoint，测试预测和目标均为有限数。
- 每个任务一个epoch、两个训练/验证/测试batch。该验证覆盖执行链路，不代表正式模型收敛。

测试数据train/val/test重复，仅验证执行链路，不能作为模型准确率或泛化性能。
本地曾观察到W&B额外模型归档阶段停滞，执行器关闭log_model及梯度watch后，上述十个任务均正常退出；Lightning模型保存保留。
本地原始记录位于tmp/local_workflow_smoke/report.json及逐任务日志，模型和日志不提交到Git。

## 本地两路并发实测

同一64个20原子结构、batch=32/16、32位精度，各运行两个联合生成器及FE/BG预测器，每个任务两epochs，并完成最佳checkpoint测试：

| 调度 | 四个任务总耗时 | 显存最小余量 | RAM最小余量 | 结果 |
| --- | --- | --- | --- | --- |
| 顺序 | 141.08秒 | 8.05 GiB | 13.31 GiB | 4/4退出0 |
| 两路并发 | 85.72秒 | 6.08 GiB | 12.16 GiB | 4/4退出0 |

实测短程加速比1.646，无OOM。耗时包含Python导入及日志启动；不能据此预测4090完整训练的加速比。
GPU余量来自nvidia-smi每0.5秒采样，RAM余量来自psutil；本地Windows不涉及远程Linux容器cgroup。
原始资源报告位于tmp/local_workflow_smoke/concurrency/report.json，公开摘录见LOCAL_VALIDATION.json。

## 采样检查暴露的限制

联合模型仅训练一个epoch后，以FE=-1.5、BG=2.0、CFG=2.0实际完成一次完整采样，CLI退出0并保存PT。
该样本frac_coords有限，但lengths为Inf、angles为NaN，因此**没有通过有限几何检查**，不能声称这些冒烟权重生成了有效结构。
这是短程权重的观察结果，尚不能判断正式收敛模型是否存在同样问题，也不修改晶格数值来掩盖结果。
执行器在生成文件旁保存.diagnostics.json；发现非有限几何会明确警告，正式结构评价仍按原始结果统计有效性。
远程smoke的一个epoch采样同样仅作为诊断；preflight/训练通过与有效生成通过必须分别判断。

## 远程4090

尚无远程SSH连接资料，未在用户服务器执行。用户粘贴环境为PyTorch 2.2.1/CUDA 12.1，与本地不同。
不能将本地通过解释为远程通过，也不能以短冒烟保证完整1000/300 epochs训练成功。
请先运行validate_concurrency.py，报告在output/<run-id>/concurrency_validation.json。报告通过后才允许--parallel 2。
