# @Author : Serein
# @Time : 2025/12/15 14:00
# @Modified: 2025/12/15 15:30 (Fixed KeyError 'lattices' & shape mismatch)

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings

warnings.filterwarnings("ignore", message="No Pauling electronegativity")

import re
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 引入 Lattice 类用于从 lengths/angles 构建晶格
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ================= 配置区域 =================
# 1. 生成数据的 .pt 文件路径
pt_path = Path(r"../output/singlerun/2026-03-10/15-50-41-mp_20/eval_gen_mp_20.pt")

# 2. (可选) 参考数据集的 CSV 路径 (例如 val.csv 或 train.csv)
# 参考数据通常包含 columns: ["pretty_formula", "spacegroup.number"]
reference_csv_path = r"../data/mp_20/train.csv"


# ===========================================

def count_unique_elements_from_formula(formula):
    """
    辅助函数：从化学式字符串中统计元素数量
    例如: "BaTiO3" -> 3
    """
    if not isinstance(formula, str):
        return 0
    elements = re.findall(r'[A-Z][a-z]*', formula)
    return len(set(elements))


def safe_numpy(tensor):
    """
    安全地将 Tensor 转为 numpy，并处理 batch 维度
    如果 tensor 形状是 [1, N, ...]，去掉第一个维度
    如果 tensor 形状是 [N, ...]，保持不变
    """
    arr = tensor.cpu().numpy()
    if arr.ndim > 1 and arr.shape[0] == 1:
        return arr.squeeze(0)
    return arr


def analyze_structures(file_path):
    """
    从 .pt 文件加载 Tensor 数据并进行 Pymatgen 分析
    兼容 'lattices' 矩阵模式 和 'lengths'/'angles' 参数模式
    """
    print(f"正在加载生成数据: {file_path}")
    data = torch.load(file_path, map_location="cpu", weights_only=False)

    # 1. 加载基础数据 (使用 safe_numpy 处理维度)
    num_atoms_per_struc = safe_numpy(data['num_atoms']).flatten()
    all_frac_coords = safe_numpy(data['frac_coords'])
    all_atom_types = safe_numpy(data['atom_types']).flatten()

    total_structures = len(num_atoms_per_struc)
    print(f"总生成结构数: {total_structures}")

    # 2. 处理晶格信息 (Lattices vs Lengths/Angles)
    lattices_matrix = None
    lattices_params = None

    if 'lattices' in data:
        # 情况 A: 直接包含 3x3 矩阵
        lattices_matrix = safe_numpy(data['lattices'])
        print("检测到晶格矩阵数据 (lattices)")
    elif 'lengths' in data and 'angles' in data:
        # 情况 B: 包含晶胞参数
        lengths = safe_numpy(data['lengths'])  # [N, 3]
        angles = safe_numpy(data['angles'])  # [N, 3]
        lattices_params = (lengths, angles)
        print("检测到晶胞参数数据 (lengths/angles)")
    else:
        raise KeyError(f"数据文件中缺少晶格信息。可用键: {list(data.keys())}")

    results = []
    current_idx = 0

    print("开始分析生成结构的对称性与元素组成...")
    for i in tqdm(range(total_structures)):
        n_atoms = num_atoms_per_struc[i]

        # 切片获取当前结构的原子信息
        struct_coords = all_frac_coords[current_idx: current_idx + n_atoms]
        struct_species = all_atom_types[current_idx: current_idx + n_atoms]

        current_idx += n_atoms

        # 获取当前结构的晶格对象
        if lattices_matrix is not None:
            # 直接使用 3x3 矩阵
            struct_lattice = lattices_matrix[i]
        else:
            # 使用 lengths/angles 构建 Lattice 对象
            l = lattices_params[0][i]  # [a, b, c]
            a = lattices_params[1][i]  # [alpha, beta, gamma]
            struct_lattice = Lattice.from_parameters(
                a=l[0], b=l[1], c=l[2],
                alpha=a[0], beta=a[1], gamma=a[2]
            )

        # 3. 统计独特元素数量
        unique_elements = np.unique(struct_species)
        num_unique = len(unique_elements)

        # 4. 对称性分析
        try:
            pmg_struct = Structure(
                lattice=struct_lattice,
                species=struct_species,
                coords=struct_coords,
                coords_are_cartesian=False
            )

            formula = pmg_struct.formula
            sga = SpacegroupAnalyzer(pmg_struct, symprec=0.01)
            sg_symbol = sga.get_space_group_symbol()
            sg_number = sga.get_space_group_number()
            crystal_system = sga.get_crystal_system()
        except Exception as e:
            # 如果构建失败 (例如晶胞参数非法)，记录错误
            formula = "Error"
            sg_symbol = "Error"
            sg_number = -1
            crystal_system = str(e)

        results.append({
            "index": i,
            "num_atoms": n_atoms,
            "num_unique_elements": num_unique,
            "formula": formula,
            "sg_symbol": sg_symbol,
            "sg_number": sg_number,
            "crystal_system": crystal_system
        })

    df = pd.DataFrame(results)

    # 保存 CSV
    output_path = Path(file_path).parent
    output_csv = output_path / "structure_analysis_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"生成数据分析完成！CSV已保存至: {output_csv}")

    return df, output_path


def plot_element_distribution(dfs, output_dir):
    """
    绘制元素数量分布对比图 (Unary, Binary, Ternary...)
    """
    print("正在绘制元素分布图...")
    new_prop = "num_unique_elements"

    colors = {'Generated': '#1f77b4', 'Reference': '#ff7f0e'}
    alphas = {'Generated': 0.6, 'Reference': 0.4}

    values_list = []

    for split, df in dfs.items():
        if new_prop not in df.columns:
            if "pretty_formula" in df.columns:
                df[new_prop] = df["pretty_formula"].apply(count_unique_elements_from_formula)
            else:
                print(f"Warning: split '{split}' 缺少 {new_prop} 或 pretty_formula 列，跳过。")
                continue
        values_list.append(df[new_prop].dropna().values)

    if not values_list:
        print("没有有效数据用于绘制元素分布图。")
        return

    all_values = np.concatenate(values_list)
    vmin, vmax = int(np.min(all_values)), int(np.max(all_values))
    # 确保 bins 至少有一定宽度
    if vmin == vmax:
        bins_int = np.arange(vmin - 1, vmax + 2) - 0.5
    else:
        bins_int = np.arange(vmin, vmax + 2) - 0.5

    plt.figure(figsize=(8, 5))

    for split, df in dfs.items():
        if new_prop not in df.columns: continue

        data = df[new_prop].dropna().values

        # 计算 density
        plt.hist(data, bins=bins_int, alpha=alphas.get(split, 0.5),
                 label=f"{split} (N={len(data)})",
                 color=colors.get(split, 'gray'),
                 edgecolor='black', linewidth=0.5, density=True)

    plt.xlabel("Number of Unique Elements")
    plt.ylabel("Density (Frequency)")
    plt.xticks(np.arange(vmin, vmax + 1))
    plt.legend()
    plt.grid(alpha=0.25, axis='y')

    save_path = output_dir / "gen_elements_hist.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


def plot_spacegroup_distribution(dfs, output_dir):
    """
    绘制空间群分布对比图，带晶系背景
    """
    print("正在绘制空间群分布图...")

    crystal_systems = [
        ("Triclinic", (1, 2), "#e0e0e0"),
        ("Monoclinic", (3, 15), "#ffcccc"),
        ("Orthorhombic", (16, 74), "#ccffcc"),
        ("Tetragonal", (75, 142), "#ccccff"),
        ("Trigonal", (143, 167), "#ffffcc"),
        ("Hexagonal", (168, 194), "#ffccff"),
        ("Cubic", (195, 230), "#ccffff"),
    ]

    plt.figure(figsize=(15, 6))

    for name, (start, end), color in crystal_systems:
        plt.axvspan(start - 0.5, end + 0.5, facecolor=color, alpha=0.3, edgecolor='none')
        center = (start + end) / 2
        width = end - start + 1
        if width > 5:
            plt.text(center, 1.01, name, ha='center', va='bottom', fontsize=9,
                     fontweight='bold', transform=plt.gca().get_xaxis_transform())
        elif width <= 5 and name == "Triclinic":
            plt.text(center, 1.01, "Tri", ha='center', va='bottom', fontsize=8, rotation=90,
                     transform=plt.gca().get_xaxis_transform())

    colors = {'Generated': '#1f77b4', 'Reference': '#ff7f0e'}
    alphas = {'Generated': 0.7, 'Reference': 0.5}

    max_density = 0

    for split, df in dfs.items():
        col_name = None
        if "sg_number" in df.columns:
            col_name = "sg_number"
        elif "spacegroup.number" in df.columns:
            col_name = "spacegroup.number"

        if not col_name:
            continue

        data = df[col_name].dropna().values
        if len(data) == 0: continue

        counts, edges = np.histogram(data, bins=np.arange(0.5, 231.5, 1))

        # 处理可能的除零错误 (如果 data 为空)
        if len(data) > 0:
            density = counts / len(data)
            max_density = max(max_density, density.max()) if density.size > 0 else 0

            centers = (edges[:-1] + edges[1:]) / 2

            plt.bar(
                centers,
                density,
                width=1.0,
                align="center",
                alpha=alphas.get(split, 0.5),
                color=colors.get(split, 'gray'),
                label=f"{split} (N={len(data)})"
            )

    plt.xlim(0, 231)
    # 增加一点头部空间，防止 max_density 为 0
    top_limit = max_density * 1.2 if max_density > 0 else 1.0
    plt.ylim(0, top_limit)

    plt.xlabel("Space Group Number (1-230)")
    plt.ylabel("Density")
    plt.legend(loc="upper left", frameon=True, facecolor='white', framealpha=0.9)

    boundary_ticks = [1] + [sys[1][1] for sys in crystal_systems]
    plt.xticks(boundary_ticks, rotation=45)
    plt.grid(axis='y', alpha=0.3)

    save_path = output_dir / "gen_spacegroup_hist.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


def main():
    # 1. 分析生成的数据
    df_gen, output_dir = analyze_structures(pt_path)

    # 2. 准备数据字典
    dfs = {
        'Generated': df_gen
    }

    # 3. 尝试加载参考数据
    if reference_csv_path and os.path.exists(reference_csv_path):
        print(f"正在加载参考数据: {reference_csv_path}")
        try:
            df_ref = pd.read_csv(reference_csv_path)
            dfs['Reference'] = df_ref
        except Exception as e:
            print(f"加载参考数据失败: {e}")
    elif reference_csv_path:
        print(f"Warning: 参考文件路径不存在: {reference_csv_path}")

    # 4. 开始绘图
    print("\n--- 开始绘图 ---")
    plot_element_distribution(dfs, output_dir)
    plot_spacegroup_distribution(dfs, output_dir)

    print("\n所有任务完成。")


if __name__ == "__main__":
    main()
