import torch
import numpy as np
from ase import Atoms
from ase.io import write
import os
from pathlib import Path


def lattice_params_to_matrix_numpy(lengths, angles):
    """
    将晶格常数转换为 3x3 笛卡尔矩阵
    lengths: (N, 3)
    angles: (N, 3) in degrees
    """
    angles_r = np.deg2rad(angles)
    coses = np.cos(angles_r)
    sins = np.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    val = np.clip(val, -1.0, 1.0)
    gamma_star = np.arccos(val)

    # Vector A: [l0*sin(beta), 0, l0*cos(beta)]
    vector_a = np.stack([
        lengths[:, 0] * sins[:, 1],
        np.zeros(lengths.shape[0]),
        lengths[:, 0] * coses[:, 1]
    ], axis=1)

    # Vector B: [-l1*sin(alpha)*cos(gamma_star), l1*sin(alpha)*sin(gamma_star), l1*cos(alpha)]
    vector_b = np.stack([
        -lengths[:, 1] * sins[:, 0] * np.cos(gamma_star),
        lengths[:, 1] * sins[:, 0] * np.sin(gamma_star),
        lengths[:, 1] * coses[:, 0]
    ], axis=1)

    # Vector C: [0, 0, l2]
    vector_c = np.stack([
        np.zeros(lengths.shape[0]),
        np.zeros(lengths.shape[0]),
        lengths[:, 2]
    ], axis=1)

    return np.transpose(np.stack([vector_a, vector_b, vector_c]), (1, 0, 2))


def process_batch(batch_data, batch_name, output_dir):
    """
    处理单个批次/单个eval切片的数据并保存为 CIF
    """
    try:
        num_atoms_batch = batch_data['num_atoms'].cpu().numpy()
        frac_coords = batch_data['frac_coords'].cpu().numpy()
        atom_types = batch_data['atom_types'].cpu().numpy()

        if 'lengths' in batch_data and 'angles' in batch_data:
            lengths = batch_data['lengths'].cpu().numpy()
            angles = batch_data['angles'].cpu().numpy()
            lattice_matrices = lattice_params_to_matrix_numpy(lengths, angles)
        elif 'lattices' in batch_data:
            lattice_matrices = batch_data['lattices'].cpu().numpy()
        else:
            print(f"[{batch_name}] 跳过: 找不到晶格(lattice)信息。")
            return

    except KeyError as e:
        print(f"[{batch_name}] 跳过: 缺少必要的数据键 {e}")
        return

    num_crystals = len(num_atoms_batch)
    print(f"[{batch_name}] 正在处理 {num_crystals} 个晶体结构...")

    atom_idx_start = 0
    success_count = 0

    for i in range(num_crystals):
        n_atoms = int(num_atoms_batch[i])
        atom_idx_end = atom_idx_start + n_atoms

        current_pos = frac_coords[atom_idx_start:atom_idx_end]
        current_types = atom_types[atom_idx_start:atom_idx_end]
        current_atomic_numbers = current_types + 1  # 修正: 0->H
        current_cell = lattice_matrices[i]

        # 异常检测：NaN 或 Inf
        if np.any(np.isnan(current_cell)) or np.any(np.isinf(current_cell)):
            atom_idx_start = atom_idx_end
            continue

        # 异常检测：体积塌缩
        vol = np.abs(np.linalg.det(current_cell))
        if vol < 1e-4:
            atom_idx_start = atom_idx_end
            continue

        try:
            atoms = Atoms(
                numbers=current_atomic_numbers,
                scaled_positions=current_pos,
                cell=current_cell,
                pbc=True
            )

            formula = atoms.get_chemical_formula(mode='hill')
            filename = os.path.join(output_dir, f"{batch_name}_id_{i}_{formula}.cif")
            write(filename, atoms, format='cif')
            success_count += 1

        except Exception as e:
            print(f"写入 CIF 失败 (Crystal {i}): {e}")

        atom_idx_start = atom_idx_end

    print(f"[{batch_name}] 完成！成功保存 {success_count}/{num_crystals} 个有效结构。")


def main():
    # ==========================================
    # 在这里直接修改你的输入和输出文件路径
    # ==========================================
    pt_file_str = '/root/private_data/rszhong/workspace/test_CGDiT/output/singlerun/2026-03-10/15-50-41-mp_20/eval_diff_mp_20_10.pt'
    output_dir_str = '/root/private_data/rszhong/workspace/test_CGDiT/output/singlerun/2026-03-10/15-50-41-mp_20/eval_diff_mp_20_10'

    pt_file = Path(pt_file_str).resolve()

    # 如果把输出路径留空 (output_dir_str = '')，默认在 pt 文件同级目录下建一个文件夹
    if output_dir_str == '':
        output_dir = pt_file.parent / f"{pt_file.stem}_cifs"
    else:
        output_dir = Path(output_dir_str).resolve()

    os.makedirs(output_dir, exist_ok=True)

    print(f"正在加载文件: {pt_file}")
    try:
        data = torch.load(pt_file, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        print("错误: 找不到指定的文件。")
        return

    # 兼容 evaluate.py 生成的多维度结构 (num_evals > 1 的情况)
    if isinstance(data, dict) and 'num_atoms' in data:
        if data['num_atoms'].ndim == 2:
            num_evals = data['num_atoms'].shape[0]
            print(f"检测到存在多次评估 (num_evals = {num_evals})，正在切片处理...")
            for e in range(num_evals):
                sub_data = {}
                # 将第 e 次生成的数据切片取出来
                for k in ['num_atoms', 'frac_coords', 'atom_types', 'lengths', 'angles', 'lattices']:
                    if k in data:
                        sub_data[k] = data[k][e]
                process_batch(sub_data, f"eval_{e}", output_dir)
        else:
            # 兼容正式生成入口输出的单维度结构
            process_batch(data, "batch_0", output_dir)

    # 兼容 List 格式存储的批次数据
    elif isinstance(data, list):
        for i, batch_item in enumerate(data):
            batch_dict = batch_item[0] if isinstance(batch_item, (tuple, list)) else batch_item
            if isinstance(batch_dict, dict):
                process_batch(batch_dict, f"batch_{i}", output_dir)

    print(f"\n全部处理完毕! 请前往 '{output_dir}' 文件夹查看 CIF 文件。")


if __name__ == "__main__":
    main()
