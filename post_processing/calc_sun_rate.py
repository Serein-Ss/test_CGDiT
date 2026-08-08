import torch
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
import warnings
from tqdm import tqdm
import pandas as pd
import os
import sys
import typing

# --- 修复低版本 Python (<3.11) 导入 mp_api 时找不到 NotRequired 的 bug ---
try:
    from typing import NotRequired
except ImportError:
    try:
        from typing_extensions import NotRequired

        # 动态将其注入到 typing 模块中，骗过 emmet-core 的导入
        typing.NotRequired = NotRequired
    except ImportError:
        pass
# --------------------------------------------------------------------------

# 修复 Windows 下的 OpenMP 冲突错误 (OMP: Error #15)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ----------------- 修改点 -----------------
# 改进的 mp-api 导入和错误捕获，保留真实报错信息
try:
    from mp_api.client import MPRester

    MP_API_AVAILABLE = True
    MP_IMPORT_ERROR = None
except Exception as e:
    MP_API_AVAILABLE = False
    MP_IMPORT_ERROR = e
# ----------------------------------------

# 忽略一些 Pymatgen 的常见警告
warnings.filterwarnings("ignore")


def pt_to_structures(pt_data):
    """
    将 .pt 文件中的数据转换为 pymatgen 的 Structure 对象列表。
    """
    structures = []

    try:
        # 如果保存的是已经是 pymatgen Structure 的列表：
        if isinstance(pt_data, list) and isinstance(pt_data[0], Structure):
            return pt_data

        # 如果保存的是张量字典 (例如: num_atoms, atom_types, frac_coords, lengths, angles)
        if isinstance(pt_data, dict):
            num_atoms = pt_data['num_atoms'].cpu().numpy()
            # 这里将模型输出的原子序数统一加 1，还原真实的物理元素序数
            atom_types = pt_data['atom_types'].cpu().numpy() + 1
            frac_coords = pt_data['frac_coords'].cpu().numpy()

            # 处理晶格信息
            if 'lattice' in pt_data:
                lattices = pt_data['lattice'].cpu().numpy()
            else:
                lengths = pt_data['lengths'].cpu().numpy()
                angles = pt_data['angles'].cpu().numpy()
                lattices = []
                for l, a in zip(lengths, angles):
                    lattices.append(Lattice.from_parameters(*l, *a).matrix)

            start_idx = 0
            for i, n in enumerate(num_atoms):
                types_i = atom_types[start_idx: start_idx + n]
                coords_i = frac_coords[start_idx: start_idx + n]
                lattice_i = lattices[i]

                try:
                    struct = Structure(lattice_i, types_i, coords_i, coords_are_cartesian=False)
                    structures.append(struct)
                except Exception as e:
                    # 如果模型生成了非法的原子序数，则忽略该结构
                    pass

                start_idx += n

    except Exception as e:
        print(f"整个字典数据结构解析发生致命错误: {e}")

    return structures


def check_validity(structure, distance_threshold=0.5):
    """
    检查结构的有效性 (Structural Validity - S)
    标准：任意两个原子之间的最短距离不能小于 distance_threshold (通常为 0.5 Å)
    """
    try:
        dist_matrix = structure.distance_matrix
        # 排除对角线（自己到自己的距离）
        np.fill_diagonal(dist_matrix, np.inf)
        if np.min(dist_matrix) < distance_threshold:
            return False
        return True
    except:
        return False


def group_by_formula(structures):
    """按化学式对结构进行分组，极大加速结构匹配"""
    grouped = {}
    for s in structures:
        formula = s.composition.reduced_formula
        if formula not in grouped:
            grouped[formula] = []
        grouped[formula].append(s)
    return grouped


def load_dataset_csvs(csv_paths):
    """从 CSV 文件加载数据集并按化学式分组"""
    dataset_grouped = {}
    total_loaded = 0
    for path in csv_paths:
        if not os.path.exists(path):
            print(f"警告: 文件不存在 {path}，将跳过。")
            continue

        print(f"正在加载数据集: {path}")
        df = pd.read_csv(path)
        for cif_str in tqdm(df['cif'], desc=f"解析 {os.path.basename(path)}"):
            try:
                struct = Structure.from_str(cif_str, fmt="cif")
                formula = struct.composition.reduced_formula
                if formula not in dataset_grouped:
                    dataset_grouped[formula] = []
                dataset_grouped[formula].append(struct)
                total_loaded += 1
            except Exception:
                pass
    print(f"共从 CSV 中成功加载了 {total_loaded} 个参考结构。")
    return dataset_grouped


def calculate_sun(gen_file_path, dataset_csv_paths=None, mp_api_key=None):
    """
    计算 S.U.N. (Structural validity, Uniqueness, Novelty) 比例
    """
    print(f"正在加载生成的结构文件: {gen_file_path}")
    gen_data = torch.load(gen_file_path, map_location='cpu', weights_only=False)
    gen_structures = pt_to_structures(gen_data)
    total_gen = len(gen_structures)
    print(f"共加载了 {total_gen} 个生成结构。")

    if total_gen == 0:
        return

    # 1. 计算 Validity (S)
    print("\n--- 1. 计算 Structural Validity (S) ---")
    valid_structures = [s for s in gen_structures if check_validity(s)]
    valid_count = len(valid_structures)
    validity_ratio = valid_count / total_gen
    print(f"有效的结构数量: {valid_count} / {total_gen}")
    print(f"Validity (S) 比例: {validity_ratio * 100:.2f}%")

    if valid_count == 0:
        print("没有有效的结构，结束计算。")
        return

    # 2. 计算 Uniqueness (U)
    print("\n--- 2. 计算 Uniqueness (U) ---")
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True)

    print("正在按化学式对生成的有效结构进行分组...")
    grouped_valid = group_by_formula(valid_structures)

    unique_structures = []
    print("正在进行组内匹配筛选唯一结构...")
    for formula, structs in tqdm(grouped_valid.items(), desc="Uniqueness Matching"):
        if len(structs) == 1:
            unique_structures.append(structs[0])
        else:
            groups = matcher.group_structures(structs)
            for group in groups:
                unique_structures.append(group[0])

    unique_count = len(unique_structures)
    uniqueness_ratio = unique_count / valid_count if valid_count > 0 else 0
    print(f"唯一的结构数量: {unique_count} / {valid_count} (在有效结构中)")
    print(f"Uniqueness (U) 比例: {uniqueness_ratio * 100:.2f}%")

    # 3. 计算 Novelty (N)
    print("\n--- 3. 计算 Novelty (N) ---")
    novel_structures_dataset = []

    # 3.1 针对本地 CSV 数据集计算 Novelty
    if dataset_csv_paths:
        print("\n>> 3.1 针对本地数据集计算 Novelty <<")
        ref_dataset_grouped = load_dataset_csvs(dataset_csv_paths)

        for u_struct in tqdm(unique_structures, desc="Checking vs Dataset"):
            formula = u_struct.composition.reduced_formula
            is_novel = True

            if formula in ref_dataset_grouped:
                for t_struct in ref_dataset_grouped[formula]:
                    if matcher.fit(u_struct, t_struct):
                        is_novel = False
                        break

            if is_novel:
                novel_structures_dataset.append(u_struct)

        novel_count_ds = len(novel_structures_dataset)
        novelty_ratio_ds = novel_count_ds / unique_count if unique_count > 0 else 0
        print(f"相对于数据集新颖的结构数量: {novel_count_ds} / {unique_count}")
        print(f"Dataset Novelty 比例: {novelty_ratio_ds * 100:.2f}%")
    else:
        print("未提供本地数据集路径，跳过 Dataset Novelty 计算。")
        novel_structures_dataset = unique_structures

    # 3.2 针对在线 Materials Project (MP) 计算 Novelty
    if mp_api_key:
        print("\n>> 3.2 针对在线 Materials Project (MP) 计算 Novelty <<")

        # ----------- 修复的判断逻辑 -----------
        if not MP_API_AVAILABLE:
            print(f"❌ 警告: 无法使用在线 MP 查询！")
            print(f"⚠️ mp_api 导入失败，真实的错误原因是: {MP_IMPORT_ERROR}")
            print(
                "💡 请在终端运行: python -c \"from mp_api.client import MPRester\" 查看缺失的具体依赖，并尝试 pip install --upgrade mp-api pydantic emmet-core")
            return
        # --------------------------------------

        novel_structures_mp = []
        try:
            print("正在提取需要查询的独立化学式并建立本地缓存...")
            # 提取独一无二的化学式，极大地减少网络请求次数
            unique_formulas = list(set([s.composition.reduced_formula for s in novel_structures_dataset]))
            print(f"共有 {len(unique_formulas)} 种不同的化学式需要向 MP 查询。")

            mp_cache = {formula: [] for formula in unique_formulas}

            with MPRester(mp_api_key) as mpr:
                batch_size = 20  # 分批次查询以防止 URL 过长或超时
                for i in tqdm(range(0, len(unique_formulas), batch_size), desc="Downloading Cache from MP"):
                    batch_formulas = unique_formulas[i:i + batch_size]
                    try:
                        # 使用列表批量搜索，显著加速
                        mp_docs = mpr.materials.summary.search(formula=batch_formulas, fields=["structure"])
                        for doc in mp_docs:
                            mp_struct = doc.structure
                            if mp_struct:
                                red_form = mp_struct.composition.reduced_formula
                                if red_form not in mp_cache:
                                    mp_cache[red_form] = []
                                mp_cache[red_form].append(mp_struct)
                    except Exception as e:
                        print(f"批量查询出错，自动回退至单条查询: {e}")
                        # 回退机制：如果某些 MP_API 版本不支持批量列表，则逐个进行（仍然因为去重而比原版快）
                        for f in batch_formulas:
                            try:
                                mp_docs_single = mpr.materials.summary.search(formula=f, fields=["structure"])
                                for doc in mp_docs_single:
                                    mp_struct = doc.structure
                                    if mp_struct:
                                        red_form = mp_struct.composition.reduced_formula
                                        if red_form not in mp_cache:
                                            mp_cache[red_form] = []
                                        mp_cache[red_form].append(mp_struct)
                            except Exception as e2:
                                print(f"查询 MP 化学式 {f} 时出错: {e2}")

            # 全部都在本地内存中快速执行 StructureMatcher 比对
            for u_struct in tqdm(novel_structures_dataset, desc="Checking vs MP Online Cache"):
                formula = u_struct.composition.reduced_formula
                is_novel = True

                # 获取该化学式对应的所有 MP 下载结构
                mp_structs_for_formula = mp_cache.get(formula, [])

                for mp_struct in mp_structs_for_formula:
                    if matcher.fit(u_struct, mp_struct):
                        is_novel = False
                        break

                if is_novel:
                    novel_structures_mp.append(u_struct)

            novel_count_mp = len(novel_structures_mp)
            novelty_ratio_mp = novel_count_mp / unique_count if unique_count > 0 else 0
            print(f"相对于 MP 在线数据库新颖的结构数量: {novel_count_mp} / {unique_count}")
            print(f"MP Novelty (N) 比例: {novelty_ratio_mp * 100:.2f}%")

            success_ratio = novel_count_mp / total_gen
            print(f"\n>>> 最终全局 S.U.N. 成功率 (同时有效、唯一、且对 MP 新颖): {success_ratio * 100:.2f}% <<<")

        except Exception as e:
            print(f"连接或查询 MP API 过程中发生错误: {e}")
    else:
        print("\n未提供 MP_API_KEY，已跳过在线 MP 数据库比对。")
        if dataset_csv_paths:
            success_ratio = novel_count_ds / total_gen
            print(f"\n>>> 局部 S.U.N. 成功率 (同时有效、唯一、且对本地数据集新颖): {success_ratio * 100:.2f}% <<<")


if __name__ == "__main__":
    # 请根据你的实际路径调整
    GEN_PT_FILE = "../output/singlerun/2026-03-10/15-50-41-mp_20/eval_gen_mp_20.pt"

    DATASET_CSV_PATHS = [
        "../data/mp_20/train.csv",
        "../data/mp_20/val.csv",
        "../data/mp_20/test.csv"
    ]

    MP_API_KEY = os.getenv("MP_API_KEY")

    calculate_sun(GEN_PT_FILE, dataset_csv_paths=DATASET_CSV_PATHS, mp_api_key=MP_API_KEY)
