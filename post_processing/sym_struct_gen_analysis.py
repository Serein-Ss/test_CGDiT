import os
import glob
import math
import json
from collections import Counter
from tqdm import tqdm

try:
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
except ImportError:
    print("错误: 找不到 pymatgen 库！")
    exit(1)


def evaluate_folder(folder_path, name, structures_per_sg=5, symprec=0.1):
    """
    评估指定文件夹内生成的晶体结构的空间群准确率和元素多样性。
    :param symprec: 寻找空间群时的对称性容忍度(容差)。生成模型通常带有微小噪声，推荐 0.1。
    """
    if not os.path.exists(folder_path):
        print(f"\n⚠️ 找不到文件夹: {folder_path}")
        return None, None

    cif_files = glob.glob(os.path.join(folder_path, "*.cif"))
    total_files = len(cif_files)

    if total_files == 0:
        print(f"\n⚠️ 文件夹 {folder_path} 中没有找到 .cif 文件。")
        return None, None

    print(f"\n" + "=" * 50)
    print(f"📊 开始评估 [{name}] 文件夹 ({total_files} 个结构)")
    print(f"容差 symprec = {symprec}")
    print("=" * 50)

    # 细分统计指标
    case_both_match = 0  # 真实和骨架都匹配 (完美生成)
    case_skeleton_only = 0  # 仅骨架匹配 (发生元素导致的对称性破缺)
    case_neither_match = 0  # 骨架都不匹配 (几何坐标生成失败)
    case_real_only = 0  # 仅真实结构匹配 (极罕见异常情况)

    correct_sg_count = 0  # 考虑元素的真实空间群匹配数
    correct_skeleton_sg_count = 0  # 抹去元素差异后的骨架空间群匹配数
    valid_structures_count = 0

    all_elements = []  # 用于计算元素覆盖率和香农熵
    unique_compositions = set()  # 用于计算化学式多样性

    # 解析所有 CIF 文件
    for cif_file in tqdm(cif_files, desc="Parsing & Analyzing"):
        filename = os.path.basename(cif_file)
        # 从文件名提取编号 (例如 "12.cif" -> 12)
        try:
            file_idx = int(filename.split('.')[0])
        except ValueError:
            continue

        # 根据生成逻辑反推其理论上（目标）的空间群编号
        target_sg = (file_idx - 1) // structures_per_sg + 1

        try:
            # 读取结构
            struct = Structure.from_file(cif_file)
            valid_structures_count += 1

            # 1. 评估元素多样性
            composition = struct.composition
            unique_compositions.add(composition.reduced_formula)  # 记录简化化学式
            for element in composition.elements:
                all_elements.append(element.symbol)

            # 2. 评估真实空间群准确性 (受元素种类影响)
            sga = SpacegroupAnalyzer(struct, symprec=symprec)
            detected_sg = sga.get_space_group_number()
            is_real_match = (detected_sg == target_sg)

            if is_real_match:
                correct_sg_count += 1

            # 3. 评估骨架空间群准确性 (抹除元素差异)
            # 将所有原子替换为同一种元素(例如全部设为 'C' 碳原子)，仅考察几何坐标是否满足对称性
            skeleton_struct = Structure(
                lattice=struct.lattice,
                species=["C"] * len(struct),  # 全部替换为同一种元素
                coords=struct.frac_coords,
                coords_are_cartesian=False
            )
            sga_skeleton = SpacegroupAnalyzer(skeleton_struct, symprec=symprec)
            detected_skeleton_sg = sga_skeleton.get_space_group_number()
            is_skeleton_match = (detected_skeleton_sg == target_sg)

            if is_skeleton_match:
                correct_skeleton_sg_count += 1

            # 4. 统计匹配情况细分
            if is_real_match and is_skeleton_match:
                case_both_match += 1
            elif is_skeleton_match and not is_real_match:
                case_skeleton_only += 1
            elif not is_skeleton_match and not is_real_match:
                case_neither_match += 1
            else:
                case_real_only += 1

        except Exception as e:
            # 解析失败或无法计算对称性
            pass

    # --- 数据汇总与计算 ---

    # 空间群准确率计算
    if valid_structures_count > 0:
        sg_accuracy = (correct_sg_count / valid_structures_count) * 100
        skeleton_sg_accuracy = (correct_skeleton_sg_count / valid_structures_count) * 100
    else:
        sg_accuracy = 0.0
        skeleton_sg_accuracy = 0.0

    # 元素多样性：香农熵
    element_counts = Counter(all_elements)
    total_elements_generated = sum(element_counts.values())
    entropy = 0.0
    if total_elements_generated > 0:
        for count in element_counts.values():
            p_i = count / total_elements_generated
            entropy -= p_i * math.log(p_i)  # 自然对数

    # 计算各细分情况的百分比
    pct_both = (case_both_match / valid_structures_count * 100) if valid_structures_count else 0
    pct_skel_only = (case_skeleton_only / valid_structures_count * 100) if valid_structures_count else 0
    pct_neither = (case_neither_match / valid_structures_count * 100) if valid_structures_count else 0

    # 记录和打印评估报告
    report_lines = [
        f"\n✅ [{name}] 评估报告:",
        f"  ▶ 成功读取结构数: {valid_structures_count} / {total_files}",
        f"  ▶ 真实空间群匹配率 (考虑元素): {sg_accuracy:.2f}% ({correct_sg_count}/{valid_structures_count})",
        f"  ▶ 骨架空间群匹配率 (抹除元素): {skeleton_sg_accuracy:.2f}% ({correct_skeleton_sg_count}/{valid_structures_count})",
        f"  ▶ --- 对称性匹配情况深度分析 ---",
        f"    ✔️ 完美匹配 (坐标与元素均正确): {pct_both:.2f}% ({case_both_match} 个)",
        f"    ⚠️ 对称性破缺 (仅坐标正确, 元素致使降级): {pct_skel_only:.2f}% ({case_skeleton_only} 个)",
        f"    ❌ 生成失败 (坐标本身不具备目标对称性): {pct_neither:.2f}% ({case_neither_match} 个)",
        f"  ▶ --- 多样性指标 ---",
        f"  ▶ 生成的不同元素种类总数: {len(element_counts)} 种",
        f"  ▶ 生成的不同化学式总数: {len(unique_compositions)} 种",
        f"  ▶ 元素分布香农熵 (Shannon Entropy): {entropy:.4f}"
    ]

    # 打印出现频率最高的前 5 种元素，方便直观感受
    top_5 = element_counts.most_common(5)
    report_lines.append(f"  ▶ 最常生成的 Top 5 元素: {', '.join([f'{el[0]}({el[1]}次)' for el in top_5])}")

    report_text = "\n".join(report_lines)
    print(report_text)

    # 封装为字典以便保存为 JSON
    metrics = {
        "name": name,
        "valid_structures_count": valid_structures_count,
        "total_files": total_files,
        "sg_accuracy_percent": sg_accuracy,
        "skeleton_sg_accuracy_percent": skeleton_sg_accuracy,
        "breakdown": {
            "perfect_match_count": case_both_match,
            "symmetry_breaking_count": case_skeleton_only,
            "failed_geometry_count": case_neither_match,
            "real_only_anomalies_count": case_real_only
        },
        "unique_elements_count": len(element_counts),
        "unique_compositions_count": len(unique_compositions),
        "shannon_entropy": entropy,
        "top_5_elements": {el[0]: el[1] for el in top_5}
    }

    return report_text, metrics


if __name__ == "__main__":
    # 配置你的两个输出文件夹路径 (与你的 run_sample_test.sh 保持一致)
    DIR_UNCONSTRAINED = "/fs1/home/xlzhou/zrs/workspace/test_CGDiT/output/singlerun/2026-03-10/15-50-41-mp_20/unconstrained_all_sg"
    DIR_CONSTRAINED = "/fs1/home/xlzhou/zrs/workspace/test_CGDiT/output/singlerun/2026-03-10/15-50-41-mp_20/constrained_all_sg"

    # 执行评估
    report1, metrics1 = evaluate_folder(DIR_UNCONSTRAINED, name="方法一：不指定元素 (Unconstrained)", symprec=0.1)
    report2, metrics2 = evaluate_folder(DIR_CONSTRAINED, name="方法二：严格指定元素 (Constrained)", symprec=0.1)

    # 将结果保存到文件
    output_dir = "../output/singlerun/2026-03-10/15-50-41-mp_20"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 1. 保存为人类可读的 TXT 报告
    report_path = os.path.join(output_dir, "sym_struct_gen_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        if report1: f.write(report1 + "\n\n")
        if report2: f.write(report2 + "\n\n")

    # 2. 保存为机器可读的 JSON 数据，方便后续画图或分析
    json_path = os.path.join(output_dir, "sym_struct_gen_report.json")
    all_metrics = {}
    if metrics1: all_metrics["unconstrained"] = metrics1
    if metrics2: all_metrics["constrained"] = metrics2

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4, ensure_ascii=False)

    print(f"\n💾 评估结果已成功保存至:")
    print(f"  - 文本报告: {report_path}")
    print(f"  - JSON数据: {json_path}")