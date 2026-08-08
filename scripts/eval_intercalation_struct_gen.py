import argparse
import os
import ast
import traceback
import pandas as pd
from pymatgen.io.cif import CifWriter
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# 假设这些是你的原始 API 导入
from sample_api import construct_dataset_from_json, construct_dataset_from_syminfo, generate_structures_from_dataset


def generate_from_csv(args):
    """从 CSV 读取条件并批量生成结构"""
    tar_dir = args.save_path
    os.makedirs(tar_dir, exist_ok=True)

    # 1. 读取 CSV 数据
    print(f"Loading CSV data from {args.csv_file}...")
    df = pd.read_csv(args.csv_file)

    generated_structures_info = []

    for index, row in df.iterrows():
        try:
            spg = int(row['spacegroup.number'])

            # --- 核心修复：基于真实 CIF 提取不对称单元的位点和元素 ---

            # 从 ground-truth CIF 中加载真实结构并分析对称性
            true_structure = Structure.from_str(row['cif'], fmt="cif")
            sga = SpacegroupAnalyzer(true_structure)
            sym_dataset = sga.get_symmetry_dataset()

            if sym_dataset is None:
                raise ValueError("无法从 CIF 解析空间群和 Wyckoff 信息。")

            wyckoff_letters = []
            atom_types = []

            # sym_dataset['equivalent_atoms'] 记录了晶胞中每个原子等价于不对称单元中的哪个原子
            # 我们利用它来提取去重后的 不对称单元 (Asymmetric Unit) 的原子类型和 Wyckoff 字母
            seen_eq_indices = set()
            for i, eq_idx in enumerate(sym_dataset['equivalent_atoms']):
                if eq_idx not in seen_eq_indices:
                    seen_eq_indices.add(eq_idx)
                    wyckoff_letters.append(sym_dataset['wyckoffs'][i])
                    # 提取该位点的元素符号 (例如 'Mn', 'Co')
                    atom_types.append(true_structure[i].species.elements[0].symbol)

            print(f"Generating structure {index}: SPG={spg}")
            print(f"  --> Extracted Atom Types: {atom_types}")
            print(f"  --> Extracted Wyckoffs:   {wyckoff_letters}")

            # 构建 Dataset，此时 len(wyckoff_letters) == len(atom_types) 完美对应
            dataset = construct_dataset_from_syminfo(spg, wyckoff_letters, atom_types)

            # 调用 API 生成
            structure_list = generate_structures_from_dataset(
                args.model_path,
                dataset,
                args.batch_size,
                args.step_lr,
                args.guidance_scale
            )

            # 保存第一个生成的结构 (假设每个条件生成1个或者取第1个)
            gen_structure = structure_list[0] if structure_list else None

            if gen_structure is not None:
                tar_file = os.path.join(tar_dir, f"generated_{index}.cif")
                writer = CifWriter(gen_structure)
                writer.write_file(tar_file)

                generated_structures_info.append({
                    'index': index,
                    'formula': row['formula'],
                    'true_cif_str': row['cif'],
                    'gen_cif_path': tar_file,
                    'gen_structure': gen_structure
                })
            else:
                print(f"Index {index}: Error generating structure (Return None).")

        except Exception as e:
            # --- 核心调试逻辑：打印完整报错堆栈 ---
            print(f"\n[!] Error processing row {index}: {e}")
            traceback.print_exc()
            print("-" * 50)

    return generated_structures_info


def evaluate_structures(generated_structures_info):
    """对比生成的结构与 CSV 中的真实结构 (Ground Truth)"""
    # 评估部分的逻辑保持不变...
    if not generated_structures_info:
        print("\n[!] 警告: 成功生成的结构数量为 0，跳过评估。")
        return

    print("\n--- 开始评估生成的结构差异 ---")

    # 初始化 StructureMatcher，它会考虑晶格缩放、平移、旋转等，是对比晶体最严谨的工具
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)

    results = []

    for info in generated_structures_info:
        index = info['index']
        true_cif_str = info['true_cif_str']
        gen_structure = info['gen_structure']

        try:
            # 从 CSV 的 CIF 字符串中加载真实结构
            true_structure = Structure.from_str(true_cif_str, fmt="cif")

            # 1. 判断两个结构是否在拓扑/对称性上匹配
            is_match = matcher.fit(true_structure, gen_structure)

            # 2. 计算均方根位移 (RMSD)，值越小表示坐标匹配度越高
            rms_dist_info = matcher.get_rms_dist(true_structure, gen_structure)
            rmsd = rms_dist_info[0] if rms_dist_info else None

            # 3. 对比体积差异
            vol_diff = abs(true_structure.volume - gen_structure.volume) / true_structure.volume

            results.append({
                'Index': index,
                'Formula': info['formula'],
                'Is_Match': is_match,
                'RMSD': rmsd,
                'Volume_Error': f"{vol_diff:.2%}"
            })

        except Exception as e:
            print(f"Error evaluating structure {index}: {e}")
            results.append({'Index': index, 'Is_Match': False, 'RMSD': None, 'Volume_Error': 'Error'})

    # 输出结果报告
    df_results = pd.DataFrame(results)
    print("\n结构对比报告:")
    print(df_results.to_string())
    df_results.to_csv("evaluation_report.csv", index=False)
    print("\n报告已保存至 evaluation_report.csv")


def main(args):
    # --- 核心修复：强制转换为绝对路径解决 Hydra 报错 ---
    args.model_path = os.path.abspath(args.model_path)
    args.save_path = os.path.abspath(args.save_path)
    if args.csv_file:
        args.csv_file = os.path.abspath(args.csv_file)
    if args.json_file:
        args.json_file = os.path.abspath(args.json_file)
    # ----------------------------------------------------

    # 如果指定了 csv 文件，则走 CSV 批量生成和评估流程
    if args.csv_file:
        generated_info = generate_from_csv(args)
        if args.evaluate:
            evaluate_structures(generated_info)
    else:
        # 保留你原有的单次生成逻辑...
        print("未提供 CSV 文件，请使用原有的 json 或命令行参数生成模式。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--csv_file', default='', type=str, help="包含 Ground Truth 的 CSV 文件路径")
    parser.add_argument('--evaluate', action='store_true', help="是否在生成后自动进行结构对比")

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--step_lr', default=1e-5, type=float)
    parser.add_argument('--spacegroup', default=0, type=int)
    parser.add_argument('--wyckoff_letters', default='', type=str)
    parser.add_argument('--atom_types', default='', type=str)
    parser.add_argument('--json_file', default='', type=str)
    parser.add_argument('--guidance_scale', default=1.0, type=float)

    args = parser.parse_args()
    main(args)