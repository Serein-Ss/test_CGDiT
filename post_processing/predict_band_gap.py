import os
import torch
import pandas as pd
import matgl
from pathlib import Path
from pymatgen.core import Structure

# 忽略一些 DGL 或 PyTorch 的运行警告，保持控制台整洁
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")

# ==========================================
# ⚙️ 全局配置区域 (直接在这里修改参数)
# ==========================================

# 运行模式选择:
# "TEST"        -> 读取 CSV 测试集并评估模型性能
# "PREDICT_DIR" -> 读取目录下的所有 CIF 文件并生成用于 VASP 验证的表格
RUN_MODE = "TEST"

# --- [1] 测试模式参数 (RUN_MODE = "TEST") ---
TEST_CSV_PATH = "../data/mp_20/test.csv"  # 你的测试集表格路径
TEST_OUTPUT_CSV = "../data/mp_20/test_evaluation_results.csv"  # 测试性能评估结果的保存路径

# --- [2] 文件夹预测模式参数 (RUN_MODE = "PREDICT_DIR") ---
PREDICT_CIF_DIR = "../output/singlerun/2026-03-13/01-29-11-mp_20_bg"  # 你生成的 CIF 所在的总目录
PREDICT_OUTPUT_CSV = "../output/singlerun/2026-03-13/01-29-11-mp_20_bg/matgl_bandgap_predictions.csv"  # 预测结果的保存路径


# ==========================================
# 🚀 核心功能定义
# ==========================================

def load_matgl_model():
    """加载 MatGL 预训练的 MEGNet 多保真度带隙模型"""
    model_name = "MEGNet-MP-2019.4.1-BandGap-mfi"
    print(f"正在加载模型: {model_name} ...")

    # 【关键修复】：强制设置后端为 DGL，如果失败则直接抛出明确的提示
    try:
        matgl.set_backend('DGL')
    except Exception as e:
        print(f"\n❌ 致命错误: 无法将 matgl 后端设置为 DGL。")
        print(f"具体报错信息: {e}")
        print(f"原因: MEGNet 预训练模型严格依赖 'dgl' 库。")
        print(f"解决办法: 请在终端执行 `pip install dgl` 后再次运行本脚本。\n")
        return None

    try:
        model = matgl.load_model(model_name)
        print("✅ 模型加载成功！\n")
        return model
    except Exception as e:
        print(f"❌ 首次加载模型失败: {e}")
        print("⚠️ 可能是由于网络中断导致本地模型缓存损坏。正在尝试自动清理缓存并重试...")
        try:
            matgl.clear_cache()
            print("✅ 本地缓存已清理，正在重新下载模型（请保持网络畅通）...")
            model = matgl.load_model(model_name)
            print("✅ 模型重新加载成功！\n")
            return model
        except Exception as e2:
            print(f"❌ 重新加载依然失败。如果是网络问题，请尝试开启代理，或者检查 dgl 库是否安装正确: {e2}")
            return None


def evaluate_csv_dataset(model, csv_path, output_csv):
    """【模式 1】读取 CSV 测试集，评估模型带隙预测性能"""
    if not os.path.exists(csv_path):
        print(f"错误：找不到测试集文件 {csv_path}，请检查路径。")
        return

    print(f"正在读取测试集: {csv_path}")
    df_test = pd.read_csv(csv_path)
    total_samples = len(df_test)
    print(f"共找到 {total_samples} 个测试样本，开始性能评估...")

    # 确立输出文件夹存在
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # 定义保真度：由于 MP 数据库里的真实带隙大部分是用 PBE 泛函算的，所以我们这里拿模型 PBE 的预测结果去算误差最合理
    fidelity_pbe = torch.tensor([0])  # 0 代表 PBE
    fidelity_hse = torch.tensor([2])  # 2 代表 HSE (仅作为参考输出)

    results = []

    for i, row in df_test.iterrows():
        mat_id = row.get('material_id', f'unknown_{i}')
        formula = row.get('pretty_formula', 'unknown')
        true_bg = float(row.get('band_gap', 0.0))
        cif_string = row.get('cif', '')

        if not isinstance(cif_string, str) or not cif_string.strip():
            print(f"[{i + 1}/{total_samples}] 跳过: {mat_id} CIF 数据为空")
            continue

        try:
            # 直接从内存里的多行字符串解析 CIF 结构，不需要保存为实体文件
            structure = Structure.from_str(cif_string, fmt="cif")

            # 模型预测 (多保真度模型支持 state_attr 输入)
            bg_pbe = model.predict_structure(structure=structure, state_attr=fidelity_pbe).item()
            bg_hse = model.predict_structure(structure=structure, state_attr=fidelity_hse).item()

            # 物理截断：将微小的负带隙视为 0 eV (金属)
            bg_pbe = max(0.0, bg_pbe)
            bg_hse = max(0.0, bg_hse)

            # 计算误差
            abs_error_pbe = abs(bg_pbe - true_bg)

            results.append({
                "material_id": mat_id,
                "化学式": formula,
                "真实带隙_eV": round(true_bg, 4),
                "PBE_预测带隙_eV": round(bg_pbe, 4),
                "HSE_预测带隙_eV": round(bg_hse, 4),
                "PBE_绝对误差_eV": round(abs_error_pbe, 4)
            })

            # 每 10 个打印一次进度
            if (i + 1) % 10 == 0 or (i + 1) == total_samples:
                print(
                    f"[{i + 1}/{total_samples}] {mat_id} ({formula}) | 真实: {true_bg:.3f} | 预测 PBE: {bg_pbe:.3f} | 误差: {abs_error_pbe:.3f}")

        except Exception as e:
            print(f"[{i + 1}/{total_samples}] 解析/预测失败: {mat_id} -> {e}")

    # 统计并保存
    if results:
        df_res = pd.DataFrame(results)
        mae = df_res["PBE_绝对误差_eV"].mean()

        print("\n========================================")
        print(f"✅ 测试评估完成！共成功处理 {len(results)} 个结构。")
        print(f"📊 模型在当前测试集上的平均绝对误差 (MAE, PBE): {mae:.4f} eV")

        df_res.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"💾 详细的对比与误差结果已保存至: {output_csv}")
    else:
        print("⚠️ 没有成功预测任何数据，请检查 CSV 格式和 CIF 字符串是否正确。")


def predict_cif_directory(model, base_cif_dir, output_csv):
    """【模式 2】遍历指定的文件夹，为生成的 CIF 批量预测带隙并对齐 VASP 任务名"""
    base_cif_dir = Path(base_cif_dir).resolve()
    if not base_cif_dir.exists():
        print(f"错误：找不到来源文件夹 {base_cif_dir}")
        return

    cif_files = list(base_cif_dir.rglob("*.cif"))
    if len(cif_files) == 0:
        print(f"未在 {base_cif_dir} 及其子目录中找到任何 CIF 文件。")
        return

    print(f"共找到 {len(cif_files)} 个生成的 CIF 文件。开始批量预测...")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    fidelity_pbe = torch.tensor([0])
    fidelity_hse = torch.tensor([2])
    results = []

    for i, cif_path in enumerate(cif_files):
        try:
            structure = Structure.from_file(cif_path)
            formula = structure.composition.reduced_formula

            relative_dir = cif_path.parent.relative_to(base_cif_dir)
            prefix = cif_path.stem
            task_name = f"{prefix}_{formula}"

            bg_pbe = model.predict_structure(structure=structure, state_attr=fidelity_pbe).item()
            bg_hse = model.predict_structure(structure=structure, state_attr=fidelity_hse).item()

            bg_pbe = max(0.0, bg_pbe)
            bg_hse = max(0.0, bg_hse)

            results.append({
                "相对路径": str(relative_dir) if str(relative_dir) != "." else "根目录",
                "序号": prefix,
                "化学式": formula,
                "VASP_任务名": task_name,
                "PBE_带隙_eV (预测)": round(bg_pbe, 4),
                "HSE_带隙_eV (预测)": round(bg_hse, 4),
                "原始CIF路径": str(cif_path)
            })

            print(f"[{i + 1}/{len(cif_files)}] 已预测: {task_name} | PBE: {bg_pbe:.2f} | HSE: {bg_hse:.2f}")

        except Exception as e:
            print(f"[{i + 1}/{len(cif_files)}] 预测失败: {cif_path.name} -> 错误: {e}")

    if results:
        df = pd.DataFrame(results)
        try:
            # 尝试按照数字序号对结果进行优雅的排序
            df['序号_int'] = pd.to_numeric(df['序号'], errors='coerce')
            df = df.sort_values(by=['相对路径', '序号_int']).drop(columns=['序号_int'])
        except:
            pass

        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n========================================")
        print(f"✅ 所有目录结构预测已完成！结果已汇总并保存至: {output_csv}")


if __name__ == "__main__":
    # 加载模型
    model = load_matgl_model()

    if model:
        if RUN_MODE == "TEST":
            print(f"=== 当前处于: [TEST] 测试模式 ===")
            evaluate_csv_dataset(model, TEST_CSV_PATH, TEST_OUTPUT_CSV)
        elif RUN_MODE == "PREDICT_DIR":
            print(f"=== 当前处于: [PREDICT_DIR] 文件夹批量预测模式 ===")
            predict_cif_directory(model, PREDICT_CIF_DIR, PREDICT_OUTPUT_CSV)
        else:
            print(f"未知的 RUN_MODE: {RUN_MODE}")