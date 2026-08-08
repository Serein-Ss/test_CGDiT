# @Author : Serein (Modified for Dynamic Folders)
# @Time : 2026/3/14
import json
import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from tqdm import tqdm

from pymatgen.core import Composition, Element, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ================= 设置科研绘图样式 =================
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'legend.frameon': True,
    'legend.edgecolor': 'black'
})


# ================= 第一部分：基于 JSON 的评估结果图表 =================

def plot_accuracy_comparison(data, output_path):
    """
    绘制准确率对比图。
    如果 JSON 中的 key 发生了变化，这里会自动提取所有包含 sg_accuracy_percent 的项。
    """
    items = []
    for key, val in data.items():
        if isinstance(val, dict) and 'sg_accuracy_percent' in val:
            items.append({
                'label': key.replace('_all_sg', '').replace('_', '\n'),
                'real': val.get('sg_accuracy_percent', 0),
                'skeleton': val.get('skeleton_sg_accuracy_percent', 0)
            })

    if not items: return

    labels = [item['label'] for item in items]
    real_acc = [item['real'] for item in items]
    skeleton_acc = [item['skeleton'] for item in items]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2), 6))
    rects1 = ax.bar(x - width / 2, real_acc, width, label='Real Structure SG Accuracy', color='#2ca02c',
                    edgecolor='black')
    rects2 = ax.bar(x + width / 2, skeleton_acc, width, label='Skeleton (Geo-only) SG Accuracy', color='#1f77b4',
                    edgecolor='black')

    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Space Group Accuracy Comparison', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, max(max(real_acc), max(skeleton_acc), 10) + 25)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontweight='bold', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'accuracy_comparison.png'), dpi=300)
    plt.close()


# ================= 第二部分：基于 CIF 解析的深度分析 =================

def parse_cif_directory(folder_path, symprec=0.1):
    """解析指定目录下的所有CIF文件"""
    if not os.path.exists(folder_path):
        return pd.DataFrame()

    cif_files = glob.glob(os.path.join(folder_path, "*.cif"))
    if not cif_files:
        return pd.DataFrame()

    results = []
    folder_name = os.path.basename(folder_path)
    for cif_file in tqdm(cif_files, desc=f"Parsing {folder_name}", leave=False):
        try:
            struct = Structure.from_file(cif_file)
            formula = struct.composition.reduced_formula
            sga = SpacegroupAnalyzer(struct, symprec=symprec)
            sg_number = sga.get_space_group_number()
            results.append({"formula": formula, "sg_number": sg_number})
        except Exception:
            continue

    return pd.DataFrame(results)


def count_unique_elements_from_formula(formula):
    if not isinstance(formula, str) or formula == "Error":
        return 0
    elements = re.findall(r'[A-Z][a-z]*', formula)
    return len(set(elements))


def get_color_map(labels):
    """为不同的标签生成固定的颜色映射"""
    standard_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    color_map = {}
    for i, label in enumerate(labels):
        if 'Reference' in label:
            color_map[label] = '#333333'  # 参考集固定为深灰色
        else:
            color_map[label] = standard_colors[i % len(standard_colors)]
    return color_map


def plot_unique_elements_hist(dfs, output_dir):
    """绘制元素种类分布对比图"""
    print("  -> Plotting element type distribution...")
    plt.figure(figsize=(10, 6))

    color_map = get_color_map(dfs.keys())

    all_nums = []
    for label, df in dfs.items():
        col = "formula" if "formula" in df.columns else "pretty_formula"
        if col not in df.columns: continue
        nums = df[col].apply(count_unique_elements_from_formula).dropna().values
        if len(nums) > 0:
            all_nums.append(nums)
            plt.hist(nums, bins=np.arange(0.5, 10.5, 1), alpha=0.6,
                     color=color_map[label], label=f"{label} (N={len(nums)})",
                     edgecolor='black', density=True, histtype='stepfilled')

    plt.xlabel("Number of Unique Elements", fontweight='bold')
    plt.ylabel("Density", fontweight='bold')
    plt.title("Element Diversity Comparison", fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "gen_elements_hist.png"), dpi=300)
    plt.close()


def plot_spacegroup_hist(dfs, output_dir):
    """绘制空间群分布对比"""
    print("  -> Plotting space group distribution...")
    plt.figure(figsize=(15, 7))

    # 背景晶系划分
    crystal_systems = [
        ("Tri", (1, 2), "#e0e0e0"), ("Mono", (3, 15), "#ffcccc"),
        ("Ortho", (16, 74), "#ccffcc"), ("Tetra", (75, 142), "#ccccff"),
        ("Trig", (143, 167), "#ffffcc"), ("Hexa", (168, 194), "#ffccff"),
        ("Cubic", (195, 230), "#ccffff"),
    ]
    for name, (start, end), color in crystal_systems:
        plt.axvspan(start - 0.5, end + 0.5, facecolor=color, alpha=0.2, edgecolor='none')
        plt.text((start + end) / 2, 1.01, name, ha='center', transform=plt.gca().get_xaxis_transform(), fontsize=9)

    color_map = get_color_map(dfs.keys())
    max_d = 0

    for label, df in dfs.items():
        col = "sg_number" if "sg_number" in df.columns else "spacegroup.number"
        if col not in df.columns: continue
        data = df[col].dropna().values
        data = data[data > 0]
        if len(data) == 0: continue

        counts, edges = np.histogram(data, bins=np.arange(0.5, 231.5, 1))
        density = counts / len(data)
        max_d = max(max_d, density.max())
        plt.plot(np.arange(1, 231), density, color=color_map[label], label=f"{label}", linewidth=1.5)
        plt.fill_between(np.arange(1, 231), 0, density, color=color_map[label], alpha=0.1)

    plt.xlim(0, 231)
    plt.ylim(0, max_d * 1.2)
    plt.xlabel("Space Group Number", fontweight='bold')
    plt.ylabel("Density", fontweight='bold')
    plt.title("Space Group Distribution Comparison", fontweight='bold', pad=20)
    plt.legend(loc="upper right")
    plt.grid(alpha=0.2)
    plt.savefig(os.path.join(output_dir, "gen_spacegroup_hist.png"), dpi=300)
    plt.close()


def plot_periodic_table_heatmap(df, output_dir, suffix=""):
    """绘制周期表热力图"""
    # ... (保持原函数逻辑，增加 suffix 用于区分不同文件夹的结果)
    # 此处省略重复的绘图细节，主要确保存储文件名包含标签
    pass


if __name__ == "__main__":
    # 基础工作目录
    base_dir = "../output/singlerun/2026-03-13/01-29-11-mp_20_bg"
    reference_csv_path = "../data/mp_20/train.csv"
    output_dir = os.path.join(base_dir, "sym_struct_gen_multi_compare")
    os.makedirs(output_dir, exist_ok=True)

    # 1. 动态搜索文件夹
    # 获取目录下所有文件夹
    all_subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    # 筛选目标文件夹
    unconstrained_dirs = [d for d in all_subdirs if d.startswith("unconstrained_all_sg")]
    constrained_dirs = [d for d in all_subdirs if d.startswith("constrained_all_sg")]

    print(f"🔍 识别到 {len(unconstrained_dirs)} 个无约束文件夹和 {len(constrained_dirs)} 个有约束文件夹")

    # 2. 解析 CIF 数据
    dfs = {}

    # 加载参考集
    if os.path.exists(reference_csv_path):
        print(f"✅ 加载参考集: {reference_csv_path}")
        dfs['Reference'] = pd.read_csv(reference_csv_path)

    # 循环解析所有匹配的文件夹
    for dname in unconstrained_dirs + constrained_dirs:
        path = os.path.join(base_dir, dname)
        df_temp = parse_cif_directory(path)
        if not df_temp.empty:
            # 使用文件夹名作为 Key
            dfs[dname] = df_temp
            print(f"  -> 已解析: {dname} ({len(df_temp)} structures)")

    # 3. 绘图
    if dfs:
        # 对比图表
        plot_unique_elements_hist(dfs, output_dir)
        plot_spacegroup_hist(dfs, output_dir)

        # JSON 准确率报告（如果存在）
        json_path = os.path.join(base_dir, "sym_struct_gen_report.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                plot_accuracy_comparison(json.load(f), output_dir)
    else:
        print("❌ 未发现有效数据。")

    print(f"\n🎉 分析完成！对比图表已保存至: {output_dir}")