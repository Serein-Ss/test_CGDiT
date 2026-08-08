# @Author : Serein
# @Time : 2026/3/14 10:57
import os
import glob
import pandas as pd
import torch
from pymatgen.core import Structure
import matgl
from tqdm import tqdm
import warnings

# 忽略一些无害的 pymatgen 警告
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    # 你的模型输出根目录 (与你的 bash 脚本一致)
    base_dir = "../output/singlerun/2026-03-13/01-29-11-mp_20_bg"

    # 定义需要评估的四个文件夹及其对应的目标带隙
    directories = {
        "unconstrained_0eV": (os.path.join(base_dir, "unconstrained_all_sg_0eV"), 0.0),
        "constrained_0eV": (os.path.join(base_dir, "constrained_all_sg_0eV"), 0.0),
        "unconstrained_4eV": (os.path.join(base_dir, "unconstrained_all_sg_4eV"), 4.0),
        "constrained_4eV": (os.path.join(base_dir, "constrained_all_sg_4eV"), 4.0),
    }

    # 设置设备 (GPU 或 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用设备: {device}")
    print("[*] 正在加载 matgl PyG 预训练带隙模型 (MEGNet-MP-2019.4.1-BandGap-mfi)...")

    # 加载带隙预测的 MEGNet 预训练模型 (基于 PyG)
    model = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")
    model = model.to(device)
    model.eval()

    results = []

    # 遍历每个生成条件文件夹
    for condition, (dir_path, target_gap) in directories.items():
        print(f"\n========================================")
        print(f"[*] 开始处理: {condition} (目标带隙: {target_gap} eV)")

        if not os.path.exists(dir_path):
            print(f"[!] 警告: 未找到文件夹，跳过 -> {dir_path}")
            continue

        cif_files = glob.glob(os.path.join(dir_path, "*.cif"))
        print(f"[*] 找到 {len(cif_files)} 个 CIF 文件。")

        if len(cif_files) == 0:
            continue

        # 批量预测
        for cif in tqdm(cif_files, desc=f"预测 {condition}"):
            try:
                # 1. 使用 pymatgen 读取 CIF 构建 Structure
                struct = Structure.from_file(cif)

                # 2. 使用 matgl 预测 (内部会自动进行 graph 转换)
                with torch.no_grad():
                    # predict_structure 返回一个 tensor
                    pred = model.predict_structure(struct)
                    predicted_bg = float(pred.item())

                # 3. 保存结果
                results.append({
                    "filename": os.path.basename(cif),
                    "condition": condition,
                    "target_band_gap": target_gap,
                    "predicted_band_gap": predicted_bg
                })

            except Exception as e:
                # 记录解析或预测失败的结构
                print(f"\n[!] 预测文件 {os.path.basename(cif)} 时出错: {e}")

    # 将结果保存并进行统计分析
    if results:
        df = pd.DataFrame(results)
        # 计算误差
        df['error'] = df['predicted_band_gap'] - df['target_band_gap']
        df['abs_error'] = df['error'].abs()

        # 预测结果落盘
        csv_path = os.path.join(base_dir, "band_gap_predictions_eval.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[*] 所有预测结果已详细保存至: {csv_path}")

        # 打印统计摘要
        print("\n=== 🎯 CFG 带隙控制效果统计摘要 ===")
        summary = df.groupby(['condition', 'target_band_gap']).agg(
            valid_samples=('filename', 'count'),
            mean_pred_bg=('predicted_band_gap', 'mean'),
            std_pred_bg=('predicted_band_gap', 'std'),
            mae=('abs_error', 'mean')
        ).reset_index()

        # 格式化输出
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(summary.to_string(index=False))
        print("========================================")
    else:
        print("\n[!] 没有生成任何有效的预测结果，请检查 CIF 文件路径是否正确。")


if __name__ == "__main__":
    main()