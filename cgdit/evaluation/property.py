"""
Property prediction script using a trained M3GNetSurrogate checkpoint.

Usage examples:
  # Predict formation_energy_per_atom from a CSV with 'cif' column:
  python scripts/predict_property.py \
      --ckpt output/singlerun/<date>/<time>/best.ckpt \
      --input data/mp_20/test.csv \
      --output predictions.csv

  # Predict without ground truth (CSV needs only 'material_id' and 'cif' columns):
  python scripts/predict_property.py \
      --ckpt output/.../best.ckpt \
      --input my_structures.csv \
      --output my_predictions.csv \
      --no_gt
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Batch
from tqdm import tqdm

from cgdit.prop_models.gnn_models.m3gnet import M3GNetSurrogate
from cgdit.common.data_utils import (
    preprocess, add_scaled_lattice_prop, build_crystal_graph
)
from pymatgen.core.structure import Structure


# ──────────────────────────────────────────────────────────────────────────────
# Helper: convert a single cached_data entry into a PyG Data object
# (mirrors CrystDataset.__getitem__ but without space_group / scaler)
# ──────────────────────────────────────────────────────────────────────────────
def cache_entry_to_data(data_dict, prop: str | None = None) -> Data:
    (frac_coords, atom_types, lengths, angles,
     edge_indices, to_jimages, num_atoms) = data_dict['graph_arrays']

    data = Data(
        frac_coords=torch.tensor(frac_coords, dtype=torch.float),
        atom_types=torch.LongTensor(atom_types),
        lengths=torch.tensor(lengths, dtype=torch.float).view(1, -1),
        angles=torch.tensor(angles, dtype=torch.float).view(1, -1),
        edge_index=torch.LongTensor(edge_indices.T).contiguous(),
        to_jimages=torch.LongTensor(to_jimages),
        num_atoms=num_atoms,
        num_bonds=edge_indices.shape[0],
        num_nodes=num_atoms,
    )

    if prop and prop in data_dict:
        val = data_dict[prop]
        y = torch.tensor([val], dtype=torch.float)
        data.y = y

    return data


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def predict(
        ckpt_path: str,
        input_csv: str,
        output_csv: str,
        prop: str = 'formation_energy_per_atom',
        graph_method: str = 'crystalnn',
        niggli: bool = True,
        primitive: bool = False,
        lattice_scale_method: str = 'scale_length',
        num_workers: int = 4,
        batch_size: int = 32,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        has_gt: bool = True,
):
    print(f"[predict] Loading checkpoint: {ckpt_path}")
    model = M3GNetSurrogate.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.to(device)

    # ── 1. 预处理 CSV 中的 CIF 字符串，构建晶体图 ──────────────────────────
    print(f"[predict] Preprocessing structures from: {input_csv}")
    df = pd.read_csv(input_csv)
    material_ids = df['material_id'].tolist()

    prop_list = [prop] if has_gt and prop in df.columns else []

    # preprocess() 读 CSV -> 并行解析 CIF -> 返回 graph_arrays 列表
    cached_data = preprocess(
        input_csv,
        num_workers=num_workers,
        niggli=niggli,
        primitive=primitive,
        graph_method=graph_method,
        prop_list=prop_list,
    )

    # 归一化晶格参数（与训练时保持一致）
    add_scaled_lattice_prop(cached_data, lattice_scale_method)

    # ── 2. 构建 PyG Data 列表 ─────────────────────────────────────────────
    print("[predict] Building PyG Data objects ...")
    data_list = []
    for entry in cached_data:
        gt_prop = prop if has_gt and prop in entry else None
        data_list.append(cache_entry_to_data(entry, prop=gt_prop))

    # ── 3. 分 batch 推理 ──────────────────────────────────────────────────
    print(f"[predict] Running inference on {len(data_list)} structures ...")
    all_preds = []
    all_targets = []

    for i in tqdm(range(0, len(data_list), batch_size)):
        batch_slice = data_list[i: i + batch_size]
        batch = Batch.from_data_list(batch_slice).to(device)

        with torch.no_grad():
            preds = model(batch)

        all_preds.append(preds.cpu().numpy())

        if has_gt and hasattr(batch, 'y') and batch.y is not None:
            all_targets.append(batch.y.cpu().numpy().flatten())

    all_preds = np.concatenate(all_preds)

    # ── 4. 保存结果 ────────────────────────────────────────────────────────
    result = pd.DataFrame({
        'material_id': material_ids,
        f'predicted_{prop}': all_preds,
    })

    if all_targets:
        all_targets = np.concatenate(all_targets)
        result[f'true_{prop}'] = all_targets
        mae = np.mean(np.abs(all_preds - all_targets))
        rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
        print(f"\n[Results] MAE  = {mae:.4f}")
        print(f"[Results] RMSE = {rmse:.4f}")

    result.to_csv(output_csv, index=False)
    print(f"\n[predict] Saved {len(result)} predictions → {output_csv}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Direct Structure prediction (no CSV needed)
# ──────────────────────────────────────────────────────────────────────────────
def predict_structures(
        ckpt_path: str,
        structures: list,
        graph_method: str = 'crystalnn',
        niggli: bool = True,
        primitive: bool = False,
        lattice_scale_method: str = 'scale_length',
        batch_size: int = 32,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> np.ndarray:
    """
    Predict properties for a list of pymatgen Structure objects directly.

    Args:
        ckpt_path:   Path to the trained M3GNetSurrogate checkpoint.
        structures:  List of pymatgen Structure objects.
        ...

    Returns:
        numpy array of shape (N,) with predicted property values.
    """
    model = M3GNetSurrogate.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.to(device)

    data_list = []
    for struct in tqdm(structures, desc="Building graphs"):
        if niggli:
            struct = struct.get_reduced_structure()
        graph_arrays = build_crystal_graph(struct, graph_method)
        entry = {'graph_arrays': graph_arrays}
        data_list.append(cache_entry_to_data(entry))

    add_scaled_lattice_prop(
        [{'graph_arrays': d.to('cpu')} for d in data_list],
        lattice_scale_method
    )

    all_preds = []
    for i in range(0, len(data_list), batch_size):
        batch = Batch.from_data_list(data_list[i: i + batch_size]).to(device)
        with torch.no_grad():
            preds = model(batch)
        all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def build_parser():
    parser = argparse.ArgumentParser(description='Predict crystal properties with M3GNet surrogate')
    parser.add_argument('--ckpt',        required=True,  help='Path to trained checkpoint (.ckpt)')
    parser.add_argument('--input',       required=True,  help='Input CSV with material_id + cif columns')
    parser.add_argument('--output',      default='predictions.csv', help='Output CSV path')
    parser.add_argument('--prop',        default='formation_energy_per_atom', help='Property name in CSV')
    parser.add_argument('--graph_method',default='crystalnn')
    parser.add_argument('--batch_size',  default=32,  type=int)
    parser.add_argument('--num_workers', default=4,   type=int)
    parser.add_argument('--no_gt',       action='store_true', help='Input CSV has no ground truth column')
    parser.add_argument('--device',      default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser


def cli(argv=None):
    args = build_parser().parse_args(argv)
    predict(
        ckpt_path=args.ckpt,
        input_csv=args.input,
        output_csv=args.output,
        prop=args.prop,
        graph_method=args.graph_method,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        has_gt=not args.no_gt,
        device=args.device,
    )


if __name__ == '__main__':
    cli()
