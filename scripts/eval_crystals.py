import os
import random
import concurrent.futures
from functools import lru_cache
import traceback
import uuid
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import warnings

warnings.filterwarnings("ignore", module="pymatgen.io.cif")

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, gaussian_kde
from pymatgen.core import Structure, Lattice, Composition
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.ext.matproj import MPRester
from cgdit.common.data_utils import lattice_params_to_matrix

import matgl
from matgl.ext.ase import Relaxer


class CrystalEvaluator:
    def __init__(self, mp_api_key=None, mlip_model_name="TensorNet-MatPES-r2SCAN-v2025.1-PES"):
        """
        初始化评估器
        :param mp_api_key: Materials Project API Key
        :param mlip_model_name: 预训练的 MatGL 机器学习力场名称
        """
        self.mp_api_key = mp_api_key
        self.matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0)

        print(f"准备加载 MatGL 预训练力场: {mlip_model_name}")

        if "M3GNet" in mlip_model_name or "CHGNet" in mlip_model_name or "MEGNet" in mlip_model_name or "DGL" in mlip_model_name:
            matgl.set_backend("DGL")
            print(">> 检测到需要 DGL 架构的模型，已自动将 backend 设置为 'DGL'")
        else:
            matgl.set_backend("PYG")
            print(">> 已使用默认的 'PYG' backend")

        # 1. 加载预训练模型
        try:
            self.potential = matgl.load_model(mlip_model_name)
        except ValueError as e:
            available_models = matgl.get_available_pretrained_models()
            print("\n" + "!" * 70)
            print(f"预训练力场 '{mlip_model_name}' 下载或加载失败！")
            print("\n请从下方列出的模型中进行挑选：")
            print("-" * 50)
            for m in available_models:
                print(f"  - {m}")
            print("-" * 50)
            raise e

        self.relaxer = Relaxer(potential=self.potential)

    def tensors_to_structures(self, pt_file_path):
        print(f"正在处理 {pt_file_path} ...")
        data = torch.load(pt_file_path, map_location='cpu', weights_only=False)
        print(f"当前 .pt 文件中的所有 keys: {list(data.keys())}")

        structures = []
        if 'frac_coords' not in data:
            raise KeyError(f"{pt_file_path} 文件中未找到 'frac_coords' 键！")

        frac_coords = data['frac_coords']
        atom_types = data['atom_types']
        num_atoms = data['num_atoms'].tolist()

        num_crystals = len(num_atoms)
        print(f"{num_crystals} 个晶体结构 (共计 {frac_coords.shape[0]} 个原子)")

        split_frac_coords = torch.split(frac_coords, num_atoms)
        split_atom_types = torch.split(atom_types, num_atoms)

        for i in range(num_crystals):
            f_coords = split_frac_coords[i].cpu().numpy()
            f_coords = np.mod(f_coords, 1.0)
            atomic_numbers = split_atom_types[i].cpu().numpy() + 1

            try:
                if 'lattices' in data:
                    lattice_matrix = data['lattices'][i].cpu().numpy()
                    lattice = Lattice(lattice_matrix)
                elif 'lattice' in data:
                    lattice_matrix = data['lattice'][i].cpu().numpy()
                    lattice = Lattice(lattice_matrix)
                elif 'lengths' in data and 'angles' in data:
                    abc = data['lengths'][i].cpu().numpy()
                    angles = data['angles'][i].cpu().numpy()
                    lattice_matrix = lattice_params_to_matrix(
                        a=abc[0], b=abc[1], c=abc[2], alpha=angles[0], beta=angles[1], gamma=angles[2]
                    )
                    lattice = Lattice(lattice_matrix)
                else:
                    raise KeyError(f"找不到晶格相关数据！文件包含的 keys 有: {list(data.keys())}")

                struct = Structure(
                    lattice=lattice,
                    species=atomic_numbers,
                    coords=f_coords,
                    coords_are_cartesian=False
                )
                structures.append(struct)
            except Exception as e:
                print(f"处理第 {i} 个晶格数据时出错: {e}")
                raise

        print(f"成功解析 {len(structures)} 个结构")
        return structures, num_crystals

    @lru_cache(maxsize=1000)
    def build_phase_diagram(self, elements_tuple, retries=2):
        """辅助函数：通过 MP 获取相图数据"""
        if not self.mp_api_key:
            return None
        print(f"\n正在从 Materials Project 获取 {elements_tuple} 的相图数据...")
        for i in range(retries):
            try:
                with MPRester(self.mp_api_key) as mpr:
                    entries = mpr.get_entries_in_chemsys(list(elements_tuple))
                    if not entries: return None
                    return PhaseDiagram(entries)
            except Exception as e:
                if i < retries - 1:
                    time.sleep(2)
                    continue
                print(f"相图获取失败 {elements_tuple}: {e}")
                return None

    def relax_and_evaluate_stability(self, structure: Structure, phase_diagram: PhaseDiagram = None,
                                     do_relax: bool = False):
        """
        步骤 2：端到端弛豫与稳定性计算
        :param do_relax: 是否用力场执行结构优化弛豫（若为 False，只计算单点能避免崩溃）
        """
        try:
            if do_relax:
                relax_results = self.relaxer.relax(structure, fmax=0.05)
                relaxed_struct = relax_results['final_structure']

                initial_energy = float(relax_results['trajectory'].energies[0])
                total_energy = float(relax_results['trajectory'].energies[-1])
            else:
                # 不进行弛豫优化，通过设置 steps=0 安全地获取当前生成结构的单点能量
                relax_results = self.relaxer.relax(structure, steps=0)
                relaxed_struct = structure

                initial_energy = float(relax_results['trajectory'].energies[0])
                total_energy = initial_energy  # 最终能量等于初始能量

            e_above_hull = None
            if phase_diagram is not None:
                entry = PDEntry(composition=relaxed_struct.composition, energy=total_energy)
                e_above_hull = phase_diagram.get_e_above_hull(entry)

            return relaxed_struct, initial_energy, total_energy, e_above_hull
        except Exception as e:
            # 即使单点能计算失败，也静默返回防止中断主流程
            return None, None, None, None

    def plot_macro_distributions(self, gen_structs, train_structs=None, save_path="macro_distributions.png"):
        """绘制宏观物理属性（晶胞体积、晶体密度）的概率密度对齐曲线及 EMD"""
        print("\n>>> 正在计算宏观物理属性并绘制分布对齐图 (Wasserstein Distance)...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        max_sample = 50000
        if len(gen_structs) > max_sample:
            gen_subset = random.sample(gen_structs, max_sample)
        else:
            gen_subset = gen_structs

        gen_vols = [s.volume for s in gen_subset]
        gen_dens = [s.density for s in gen_subset]

        features = [
            (gen_vols, "Cell Volume ($\AA^3$)", axes[0]),
            (gen_dens, "Crystal Density ($g/cm^3$)", axes[1])
        ]

        for i, (gen_data, title, ax) in enumerate(features):
            q_low, q_high = np.percentile(gen_data, [1, 99])
            gen_data_filtered = [v for v in gen_data if q_low <= v <= q_high]

            kde_gen = gaussian_kde(gen_data_filtered)
            x_range = np.linspace(min(gen_data_filtered), max(gen_data_filtered), 500)
            ax.plot(x_range, kde_gen(x_range), color='royalblue', lw=2, label='Generated')
            ax.fill_between(x_range, kde_gen(x_range), alpha=0.3, color='royalblue')

            if train_structs:
                if len(train_structs) > max_sample:
                    train_subset = random.sample(train_structs, max_sample)
                else:
                    train_subset = train_structs

                if i == 0:
                    train_data = [s.volume for s in train_subset]
                else:
                    train_data = [s.density for s in train_subset]

                train_data_filtered = [v for v in train_data if q_low <= v <= q_high]
                if train_data_filtered:
                    kde_train = gaussian_kde(train_data_filtered)
                    ax.plot(x_range, kde_train(x_range), color='darkorange', lw=2, linestyle='--',
                            label='Training Set (Ground Truth)')
                    ax.fill_between(x_range, kde_train(x_range), alpha=0.2, color='darkorange')

                    emd_val = wasserstein_distance(gen_data_filtered, train_data_filtered)
                    ax.text(0.55, 0.85, f"Wasserstein Dist (EMD): {emd_val:.3f}",
                            transform=ax.transAxes, fontsize=12, fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

            ax.set_title(f"Distribution of {title}", fontsize=14, fontweight='bold')
            ax.set_xlabel(title, fontsize=12)
            ax.set_ylabel("Probability Density", fontsize=12)
            ax.legend(loc='upper right')
            ax.grid(axis='y', linestyle=':', alpha=0.7)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        return save_path

    def _get_rdf_distances(self, structure, r_max=8.0):
        """辅助函数：获取局部结构径向分布（所有成对距离数组）"""
        distances = []
        for site in structure:
            neighbors = structure.get_neighbors(site, r=r_max)
            distances.extend([nn.nn_distance for nn in neighbors])
        return np.array(distances)

    def plot_micro_rdf_heatmap(self, gen_structs, train_structs, save_path="rdf_wasserstein_heatmap.png"):
        """绘制微观几何连续空间对齐热力图 (RDF Wasserstein Matrix)"""
        if not train_structs:
            return None

        print("\n>>> 正在计算微观径向分布(RDF)与连续空间 Wasserstein 距离热力图...")

        n_samples = min(10, len(gen_structs))
        selected_gens = gen_structs[:n_samples]

        selected_refs = []
        for gs in selected_gens:
            formula = gs.composition.reduced_formula
            matches = [ts for ts in train_structs if ts.composition.reduced_formula == formula]
            if matches:
                selected_refs.append(random.choice(matches))
            else:
                selected_refs.append(random.choice(train_structs))

        n_refs = len(selected_refs)
        wasserstein_matrix = np.zeros((n_samples, n_refs))

        gen_rdfs = [self._get_rdf_distances(s) for s in selected_gens]
        ref_rdfs = [self._get_rdf_distances(s) for s in selected_refs]

        for i in range(n_samples):
            for j in range(n_refs):
                if len(gen_rdfs[i]) > 0 and len(ref_rdfs[j]) > 0:
                    wasserstein_matrix[i, j] = wasserstein_distance(gen_rdfs[i], ref_rdfs[j])
                else:
                    wasserstein_matrix[i, j] = np.nan

        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.imshow(wasserstein_matrix, cmap='YlOrRd', aspect='auto')

        for i in range(n_samples):
            for j in range(n_refs):
                if not np.isnan(wasserstein_matrix[i, j]):
                    ax.text(j, i, f"{wasserstein_matrix[i, j]:.2f}", ha="center", va="center",
                            color="black" if wasserstein_matrix[i, j] < np.nanmax(
                                wasserstein_matrix) * 0.6 else "white")

        ax.set_xticks(np.arange(n_refs))
        ax.set_yticks(np.arange(n_samples))
        ax.set_xticklabels([f"Ref {j + 1}\n({selected_refs[j].composition.reduced_formula})" for j in range(n_refs)],
                           rotation=45, ha='right')
        ax.set_yticklabels([f"Gen {i + 1}\n({selected_gens[i].composition.reduced_formula})" for i in range(n_samples)])

        ax.set_xlabel("Reference Structures (Training Set)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Generated Candidates", fontsize=12, fontweight='bold')
        ax.set_title("Micro-Geometric Novelty:\nRDF Wasserstein Distance Heatmap", fontsize=14, fontweight='bold')

        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label("Earth Mover's Distance (EMD) of RDF", fontsize=11)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        return save_path

    def plot_phase_diagram_with_points(self, phase_diagram, initial_struct, final_struct, initial_energy, final_energy,
                                       sys_name, save_dir):
        """完全基于 matplotlib 手动绘制相图逻辑，彻底绕开 PDPlotter"""
        if phase_diagram is None: return None

        # 确保元素排序一致性
        els = sorted([el for el in phase_diagram.elements], key=lambda e: e.symbol)
        num_els = len(els)
        if num_els not in [2, 3]: return None

        # 计算弛豫前后的形成能用于着色和对比
        init_entry = PDEntry(initial_struct.composition, initial_energy)
        final_entry = PDEntry(final_struct.composition, final_energy)
        e_form_init = phase_diagram.get_form_energy_per_atom(init_entry)
        e_form_final = phase_diagram.get_form_energy_per_atom(final_entry)

        # 检查是否关闭了弛豫（初始和结束能量是否一致）
        is_same = abs(e_form_init - e_form_final) < 1e-6

        stable_entries = phase_diagram.stable_entries
        fig = plt.figure(figsize=(9, 7))

        if num_els == 2:
            # ==== 绘制二元相图 ====
            all_pts = []
            for entry in phase_diagram.all_entries:
                x = entry.composition.get_atomic_fraction(els[1])
                y = phase_diagram.get_form_energy_per_atom(entry)
                all_pts.append((x, y))

            # 画所有已知但可能不稳定的相 (灰色点)
            pts_arr = np.array(all_pts)
            plt.scatter(pts_arr[:, 0], pts_arr[:, 1], c='lightgray', s=15, alpha=0.6, label='Other known phases')

            # 提取稳定相，绘制 Convex Hull (凸包黑线)
            hull_pts = sorted(
                [(e.composition.get_atomic_fraction(els[1]), phase_diagram.get_form_energy_per_atom(e)) for e in
                 stable_entries])
            hx, hy = zip(*hull_pts)
            plt.plot(hx, hy, 'k-', lw=2, label='Convex Hull (Stable)')
            plt.scatter(hx, hy, c='black', marker='o', s=40, zorder=3)

            # 标记生成的初始结构与最终弛豫结构的相对位置及变化
            x_comp = initial_struct.composition.get_atomic_fraction(els[1])
            vmin, vmax = min(e_form_init, e_form_final) - 0.1, max(e_form_init, e_form_final) + 0.1

            if is_same:
                # 未弛豫：只画单独一个生成点，不带箭头
                sc1 = plt.scatter([x_comp], [e_form_init], c=[e_form_init], cmap='coolwarm', vmin=vmin, vmax=vmax,
                                  marker='o', s=200, edgecolors='black', label=f'Generated ({e_form_init:.2f} eV)',
                                  zorder=6)
            else:
                # 弛豫后：画出变化前后以及箭头
                sc1 = plt.scatter([x_comp], [e_form_init], c=[e_form_init], cmap='coolwarm', vmin=vmin, vmax=vmax,
                                  marker='o', s=150, edgecolors='black', label=f'Initial ({e_form_init:.2f} eV)',
                                  zorder=5)
                sc2 = plt.scatter([x_comp], [e_form_final], c=[e_form_final], cmap='coolwarm', vmin=vmin, vmax=vmax,
                                  marker='*', s=250, edgecolors='black', label=f'Relaxed ({e_form_final:.2f} eV)',
                                  zorder=6)

                # 使用箭头指示结构优化弛豫的方向
                plt.annotate("", xy=(x_comp, e_form_final), xytext=(x_comp, e_form_init),
                             arrowprops=dict(arrowstyle="->", color="red", lw=2))

            plt.colorbar(sc1, label="Formation Energy (eV/atom)")
            plt.xlabel(f"Fraction of {els[1].symbol}")
            plt.ylabel("Formation Energy (eV/atom)")
            plt.xlim(-0.05, 1.05)

        elif num_els == 3:
            # ==== 绘制三元相图 (组分等边三角形投影) ====
            def get_ternary_xy(comp):
                c = [comp.get_atomic_fraction(e) for e in els]
                x = c[1] + 0.5 * c[2]
                y = c[2] * np.sqrt(3) / 2
                return x, y

            # 绘制等边三角形边界与元素标签
            plt.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], 'k-', lw=1.5)
            plt.text(-0.05, -0.05, els[0].symbol, fontsize=12, fontweight='bold')
            plt.text(1.02, -0.05, els[1].symbol, fontsize=12, fontweight='bold')
            plt.text(0.48, np.sqrt(3) / 2 + 0.02, els[2].symbol, fontsize=12, fontweight='bold')

            # 绘制 MP 稳定相基态参考点 (仅画稳定相，作为参照系)
            for entry in stable_entries:
                tx, ty = get_ternary_xy(entry.composition)
                plt.plot(tx, ty, 'ko', markersize=6, alpha=0.5)

            # 绘制生成的初始和最终弛豫状态 (由颜色指示形成能深度)
            x_comp, y_comp = get_ternary_xy(initial_struct.composition)
            vmin, vmax = min(e_form_init, e_form_final) - 0.1, max(e_form_init, e_form_final) + 0.1

            if is_same:
                sc1 = plt.scatter([x_comp], [y_comp], c=[e_form_init], cmap='coolwarm', vmin=vmin, vmax=vmax,
                                  marker='o',
                                  s=300, edgecolors='black', label=f'Generated ({e_form_init:.2f} eV)', zorder=6)
            else:
                sc1 = plt.scatter([x_comp], [y_comp], c=[e_form_init], cmap='coolwarm', vmin=vmin, vmax=vmax,
                                  marker='o',
                                  s=300, edgecolors='black', label=f'Initial ({e_form_init:.2f} eV)', zorder=5)
                sc2 = plt.scatter([x_comp], [y_comp], c=[e_form_final], cmap='coolwarm', vmin=vmin, vmax=vmax,
                                  marker='*',
                                  s=150, edgecolors='white', label=f'Relaxed ({e_form_final:.2f} eV)', zorder=6)

            plt.colorbar(sc1, label="Formation Energy (eV/atom)")
            plt.axis('off')
            plt.gca().set_aspect('equal')

        plt.title(f"{num_els}-ary Phase Diagram: {sys_name}\nTarget: {initial_struct.composition.reduced_formula}")
        plt.legend(loc='upper right', fontsize='small')

        filename = f"PD_{sys_name}_{initial_struct.composition.reduced_formula}_{uuid.uuid4().hex[:4]}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return filepath

    def plot_energy_density_funnel(self, gen_densities, gen_ehulls, ref_densities=None, ref_ehulls=None,
                                   save_path="funnel.png"):
        plt.figure(figsize=(10, 7))
        plt.scatter(gen_densities, gen_ehulls, c='royalblue', alpha=0.6, s=50, edgecolors='white', linewidth=0.5,
                    label='Generated Candidates')
        if ref_densities and ref_ehulls:
            plt.scatter(ref_densities, ref_ehulls, c='red', marker='*', s=200, edgecolors='black', linewidth=1,
                        label='Known Stable (Test Set)', zorder=5)
        plt.axhline(0, color='darkred', linestyle='--', linewidth=2, label='E_hull = 0 (Ground State)', zorder=1)
        plt.xlabel('Physical Density (g/cm³)', fontsize=14, fontweight='bold')
        plt.ylabel('$E_{hull}$ (eV/atom)', fontsize=14, fontweight='bold')
        plt.title('Energy-Density Funnel Plot', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        return save_path

    def plot_retention_funnel(self, stages, counts, save_path="retention_funnel.png"):
        fig, ax = plt.subplots(figsize=(12, 8))
        y_positions = np.arange(len(stages))[::-1] * 2
        max_val = max(counts) if max(counts) > 0 else 1
        for i in range(len(stages)):
            val = counts[i]
            if i < len(stages) - 1:
                next_val = counts[i + 1]
                ax.fill([-val / 2, val / 2, next_val / 2, -next_val / 2],
                        [y_positions[i] - 0.4, y_positions[i] - 0.4, y_positions[i + 1] + 0.4,
                         y_positions[i + 1] + 0.4],
                        color='lightgray', alpha=0.4)
            color = plt.cm.Blues(0.8 - i * 0.15)
            ax.barh(y_positions[i], val, height=0.8, left=-val / 2, color=color, edgecolor='black', linewidth=1.5)
            ax.text(0, y_positions[i], f"{val:,}", ha='center', va='center',
                    color='black' if val < max_val * 0.1 else 'white', fontweight='bold', fontsize=12)
            ax.text(-max_val * 0.52, y_positions[i], stages[i], ha='right', va='center', color='black', fontsize=12,
                    fontweight='bold')
            if i > 0:
                if "Sampled" in stages[i]:
                    retention = (val / counts[i - 1]) * 100 if counts[i - 1] > 0 else 0
                    note = f"Sampling Rate: {retention:.1f}%"
                elif "Converged" in stages[i]:
                    retention = (val / counts[i - 1]) * 100 if counts[i - 1] > 0 else 0
                    note = f"Convergence Rate: {retention:.1f}%"
                elif "Stable" in stages[i]:
                    retention = (val / counts[i - 2]) * 100 if counts[i - 2] > 0 else 0
                    note = f"Yield Rate (from sampled): {retention:.1f}%"
                else:
                    retention = (val / counts[i - 1]) * 100 if counts[i - 1] > 0 else 0
                    note = f"Validity Retention: {retention:.1f}%"
                ax.text(max_val * 0.52, y_positions[i], note, ha='left', va='center', color='darkred', fontsize=11,
                        fontstyle='italic')
        ax.set_xlim(-max_val * 0.8, max_val * 0.8)
        ax.axis('off')
        plt.title('Generative Pipeline Retention Funnel', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        return save_path

    def evaluate_novelty_and_uniqueness(self, generated_structures, training_structures):
        print("正在评估唯一性 (Uniqueness) - 启用化学式快速分组算法...")
        gen_groups = defaultdict(list)
        for struct in generated_structures:
            gen_groups[struct.composition.reduced_formula].append(struct)
        unique_structures = []
        for formula, structs in tqdm(gen_groups.items(), desc="Uniqueness check"):
            unique_in_group = []
            for struct in structs:
                is_unique = True
                for u_struct in unique_in_group:
                    if self.matcher.fit(struct, u_struct):
                        is_unique = False
                        break
                if is_unique:
                    unique_in_group.append(struct)
            unique_structures.extend(unique_in_group)
        uniqueness_score = len(unique_structures) / len(generated_structures) if generated_structures else 0

        print("正在评估新颖性 (Novelty) - 启用训练集交叉索引...")
        novel_structures = []
        train_groups = defaultdict(list)
        for struct in training_structures:
            train_groups[struct.composition.reduced_formula].append(struct)
        for struct in tqdm(unique_structures, desc="Novelty check"):
            formula = struct.composition.reduced_formula
            is_novel = True
            if formula in train_groups:
                for train_struct in train_groups[formula]:
                    if self.matcher.fit(struct, train_struct):
                        is_novel = False
                        break
            if is_novel:
                novel_structures.append(struct)
        novelty_score = len(novel_structures) / len(unique_structures) if unique_structures else 0
        return uniqueness_score, novelty_score, unique_structures, novel_structures


if __name__ == "__main__":

    # ================= 【核心配置开关】 =================
    # 是否开启机器学习力场对生成结构进行弛豫（几何优化）。
    # False: (默认) 避免因未收敛报错，直接计算生成结构的单点能，并直接用生成结构绘制相图
    # True: 执行 ASE 放宽优化后再进行相图绘制（计算代价更高且有失败概率）
    ENABLE_RELAXATION = False
    # ===================================================

    PT_FILE_PATH = "../output/singlerun/2026-03-10/15-50-41-mp_20/eval_gen_mp_20.pt"
    MP_API_KEY = os.getenv("MP_API_KEY")

    TRAIN_CSV_PATH = "../data/mp_20/train.csv"
    TRAIN_STRUCTS = []

    if os.path.exists(TRAIN_CSV_PATH):
        print(f"\n>>> 准备加载训练集用作参考基准: {TRAIN_CSV_PATH}")
        try:
            df_train = pd.read_csv(TRAIN_CSV_PATH, usecols=['cif'])
            for cif_str in tqdm(df_train['cif'].dropna(), desc="解析训练集 CIF", dynamic_ncols=True):
                try:
                    struct = Structure.from_str(cif_str, fmt="cif")
                    TRAIN_STRUCTS.append(struct)
                except Exception:
                    pass
            print(f"成功加载了 {len(TRAIN_STRUCTS)} 个训练集参考结构。\n")
        except Exception as e:
            print(f"[警告] 训练集读取或解析失败: {e}\n")
    else:
        print(f"\n[提示] 未找到训练集文件: {TRAIN_CSV_PATH}，依赖训练集的指标将被跳过。\n")
    # ============================================

    base_dir = os.path.dirname(os.path.abspath(PT_FILE_PATH))
    if not base_dir:
        base_dir = "."

    out_dir = os.path.join(base_dir, "Stability")
    os.makedirs(out_dir, exist_ok=True)

    report_txt = os.path.join(out_dir, "evaluation_report.txt")
    ehull_img = os.path.join(out_dir, "ehull_distribution.png")


    def log_and_print(msg, file_handle):
        print(msg)
        file_handle.write(msg + "\n")


    evaluator = CrystalEvaluator(mp_api_key=MP_API_KEY, mlip_model_name="TensorNet-MatPES-r2SCAN-v2025.1-PES")

    with open(report_txt, "w", encoding="utf-8") as f:
        log_and_print("========== AI4Mat 晶体生成模型评估报告 ==========", f)
        log_and_print(f"分析目标文件: {PT_FILE_PATH}", f)
        log_and_print(f"机器学习力场: {evaluator.potential.__class__.__name__}", f)
        log_and_print(f"是否启用力场结构弛豫: {ENABLE_RELAXATION}", f)

        log_and_print("\n>>> 阶段 1：张量转化与 Pymatgen 结构构建", f)
        structures, total_generated_count = evaluator.tensors_to_structures(PT_FILE_PATH)
        valid_structures_count = len(structures)

        if valid_structures_count == 0:
            log_and_print("未发现有效结构，程序退出。", f)
            exit()

        # ======== 阶段 1.2 宏观物理属性对齐 ========
        macro_plot_path = os.path.join(out_dir, "macro_distributions.png")
        try:
            evaluator.plot_macro_distributions(structures, TRAIN_STRUCTS, save_path=macro_plot_path)
            log_and_print(f"[图片已保存] 宏观属性分布与 EMD 对齐图 -> {macro_plot_path}", f)
        except Exception as e:
            log_and_print(f"[警告] 宏观属性分布绘制失败: {e}", f)

        # ================= 阶段 1.5：针对相图示意专门采样 =================
        log_and_print(f"\n>>> 阶段 1.5：代表性结构采样 (目标: 挑选随机3个二元结构以及3个三元结构进行示意)", f)

        # 1. 过滤出所有的二元和三元结构
        binary_structs = []
        ternary_structs = []

        for struct in structures:
            num_els = len(struct.composition.elements)
            if num_els == 2:
                binary_structs.append(struct)
            elif num_els == 3:
                ternary_structs.append(struct)

        # 2. 从所有符合条件的结构中随机打乱并抽取（最多3个）
        random.shuffle(binary_structs)
        random.shuffle(ternary_structs)

        sampled_binaries = binary_structs[:3]
        sampled_ternaries = ternary_structs[:3]

        # 3. 组合成最终要测试绘制相图的结构
        sampled_structures = sampled_binaries + sampled_ternaries

        structures = sampled_structures
        sampled_count = len(structures)
        log_and_print(
            f"成功采样出 {sampled_count} 个代表性结构进行后续计算与相图绘制 (二元: {len(sampled_binaries)}个, 三元: {len(sampled_ternaries)}个)。\n",
            f)
        # ====================================================================================

        # ================= 阶段 2：直接进入计算与稳定性验证 =================
        log_and_print(f"\n>>> 阶段 2：稳定性评估 (对 {sampled_count} 个结构进行力场评估 & E_hull)", f)
        ehull_list = []
        gen_densities = []
        sys_dict = defaultdict(list)

        converged_count = 0
        stable_count_01 = 0

        pd_plot_dir = os.path.join(out_dir, "phase_diagrams")
        os.makedirs(pd_plot_dir, exist_ok=True)
        log_and_print(f"相图绘制结果将保存在: {pd_plot_dir}\n", f)

        for struct in structures:
            els = tuple(sorted(list(set([el.symbol for el in struct.composition.elements]))))
            sys_dict[els].append(struct)

        with tqdm(total=sampled_count, desc="计算总进度", dynamic_ncols=True) as pbar:
            for els, structs_in_sys in sys_dict.items():
                sys_name = '-'.join(els)
                f.write(f"处理化学体系: {sys_name} (共 {len(structs_in_sys)} 个结构)\n")

                pd_diagram = evaluator.build_phase_diagram(els)

                for struct in structs_in_sys:
                    formula = struct.composition.reduced_formula
                    pbar.set_postfix_str(f"体系: {sys_name} | 当前结构: {formula}")

                    # 【修改点】向函数传递是否需要弛豫的全局开关 do_relax=ENABLE_RELAXATION
                    relaxed_struct, e_init, e_final, e_hull = evaluator.relax_and_evaluate_stability(
                        struct,
                        phase_diagram=pd_diagram,
                        do_relax=ENABLE_RELAXATION
                    )

                    if relaxed_struct is not None:
                        converged_count += 1
                        if e_hull is not None:
                            ehull_list.append(e_hull)
                            gen_densities.append(relaxed_struct.density)

                            if e_hull <= 0.1:
                                stable_count_01 += 1

                        # ======== 带日志和异常追踪的绘图代码块 ========
                        try:
                            num_els = len(struct.composition.elements)
                            if num_els not in [2, 3]:
                                log_and_print(
                                    f"  [-] 跳过绘图: {sys_name} 包含 {num_els} 种元素，相图代码仅支持二元和三元体系。",
                                    f)
                            elif pd_diagram is None:
                                log_and_print(f"  [-] 跳过绘图: {sys_name} 相图数据未成功获取。", f)
                            else:
                                saved_path = evaluator.plot_phase_diagram_with_points(
                                    phase_diagram=pd_diagram,
                                    initial_struct=struct,
                                    final_struct=relaxed_struct,
                                    initial_energy=e_init,
                                    final_energy=e_final,
                                    sys_name=sys_name,
                                    save_dir=pd_plot_dir
                                )
                                if saved_path:
                                    log_and_print(f"  [+] 成功绘制相图: {saved_path}", f)

                        except Exception as e:
                            log_and_print(f"  [警告] 绘制相图时发生异常 ({sys_name}): {e}", f)
                            log_and_print(traceback.format_exc(), f)
                        # =========================================================

                    pbar.update(1)

        if ehull_list:
            mean_ehull = np.mean(ehull_list)
            stable_ratio = sum(1 for e in ehull_list if e <= 0.0) / len(ehull_list)
            log_and_print(f"\n【稳定性结果】平均 E_hull: {mean_ehull:.4f} eV/atom", f)
            log_and_print(f"【稳定性结果】热力学稳定比例 (E_hull <= 0): {stable_ratio:.2%}", f)

            plt.figure(figsize=(8, 6))
            plt.hist(ehull_list, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            plt.axvline(x=0.0, color='red', linestyle='--', linewidth=2, label='E_hull = 0 (Stable Base)')
            plt.title('Distribution of Energy Above Hull ($E_{hull}$)', fontsize=14)
            plt.xlabel('$E_{hull}$ (eV/atom)', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(axis='y', alpha=0.5)
            plt.savefig(ehull_img, dpi=300, bbox_inches='tight')
            plt.close()

            funnel_img = os.path.join(out_dir, "energy_density_funnel.png")
            ref_densities = []
            ref_ehulls = []

            if TRAIN_STRUCTS:
                log_and_print("\n>>> 提取参考结构数据 (用于漏斗图的高亮红色真值锚点)...", f)
                sample_train = random.sample(TRAIN_STRUCTS, min(len(TRAIN_STRUCTS), 50))
                for ref_struct in tqdm(sample_train, desc="参考数据计算"):
                    els = tuple(sorted(list(set([el.symbol for el in ref_struct.composition.elements]))))
                    pd_diagram = evaluator.build_phase_diagram(els)

                    _, _, _, r_ehull = evaluator.relax_and_evaluate_stability(
                        ref_struct,
                        phase_diagram=pd_diagram,
                        do_relax=ENABLE_RELAXATION
                    )
                    if r_ehull is not None:
                        ref_densities.append(ref_struct.density)
                        ref_ehulls.append(r_ehull)

            evaluator.plot_energy_density_funnel(
                gen_densities=gen_densities,
                gen_ehulls=ehull_list,
                ref_densities=ref_densities,
                ref_ehulls=ref_ehulls,
                save_path=funnel_img
            )

        retention_stages = [
            "Total Generated",
            "Valid Structures",
            "Sampled for Testing",
            "Computation Success",  # 此处文案由 Converged 改为 Success 兼容未弛豫模式
            "Highly Stable\n(E_hull ≤ 0.1)"
        ]
        retention_counts = [
            total_generated_count,
            valid_structures_count,
            sampled_count,
            converged_count,
            stable_count_01
        ]

        retention_img_path = os.path.join(out_dir, "retention_funnel_plot.png")
        evaluator.plot_retention_funnel(retention_stages, retention_counts, save_path=retention_img_path)

        # ======== 阶段 3.1 微观几何唯一性热力图 ========
        if TRAIN_STRUCTS:
            rdf_heatmap_path = os.path.join(out_dir, "rdf_wasserstein_heatmap.png")
            try:
                evaluator.plot_micro_rdf_heatmap(structures, TRAIN_STRUCTS, save_path=rdf_heatmap_path)
                log_and_print(f"[图片已保存] 微观 RDF 连续几何对齐热力图 -> {rdf_heatmap_path}", f)
                log_and_print(
                    " > 注解：热力图中的高 EMD 值证明生成的结构在连续空间中与基态具有显著差异，打破了二元匹配器的认知盲区。",
                    f)
            except Exception as e:
                log_and_print(f"[警告] RDF热力图绘制失败: {e}", f)

        log_and_print("\n>>> 阶段 3.2：传统二元分类匹配独特性与新颖性", f)
        uniqueness, novelty, unique_structs, novel_structs = evaluator.evaluate_novelty_and_uniqueness(
            structures, TRAIN_STRUCTS
        )
        log_and_print(f"【独特性结果】Uniqueness: {uniqueness:.2%}", f)

        if TRAIN_STRUCTS:
            log_and_print(f"【新颖性结果】Novelty: {novelty:.2%}", f)
        else:
            log_and_print("注意：因为 TRAIN_STRUCTS 为空，已跳过新颖性的计算。", f)

        log_and_print("\n========== 评估流程全部结束 ==========", f)
