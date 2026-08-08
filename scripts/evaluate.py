# @Author : Serein
# @Time : 2025/11/8 21:56
import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch

from eval_utils import load_model, lattices_to_params_shape, recommand_step_lr

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal.symmetry import Group

import copy
import numpy as np


def diffusion(loader, model, num_evals, step_lr=1e-5, guidance_scale=1.0, property_name=None, target_value=None):

    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []

    if property_name is not None and target_value is not None:
        print(f"\n[INFO] Dataset condition overridden for CSP/Reconstruction!")
        print(f"       Target Property: '{property_name}' set to {target_value}")
        print(f"       Using Classifier-Free Guidance with scale = {guidance_scale}\n")
    else:
        print(f"\n[INFO] Standard Unconditional/Original CSP Evaluation (CFG scale = {guidance_scale}).\n")

    for idx, batch in enumerate(loader):

        if torch.cuda.is_available():
            batch.cuda()

        if property_name is not None and target_value is not None:
            batch_size = batch.num_graphs
            target_tensor = torch.full((batch_size, 1), target_value, dtype=torch.float, device=batch.device)
            # 强制将 batch 中对应的属性替换为你想要测试的目标值
            setattr(batch, property_name, target_tensor)

        batch_all_frac_coords = []
        batch_all_lattices = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lattices = []

        for eval_idx in range(num_evals):

            print(f'batch {idx + 1} / {len(loader)}, sample {eval_idx + 1} / {num_evals}')
            outputs, traj = model.sample(batch, step_lr=step_lr, guidance_scale=guidance_scale)
            batch_frac_coords.append(outputs['frac_coords'].detach().cpu())
            batch_num_atoms.append(outputs['num_atoms'].detach().cpu())
            batch_atom_types.append(outputs['atom_types'].detach().cpu())
            batch_lattices.append(outputs['lattices'].detach().cpu())

        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lattices.append(torch.stack(batch_lattices, dim=0))

        input_data_list = input_data_list + batch.to_data_list()

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lattices = torch.cat(lattices, dim=1)
    lengths, angles = lattices_to_params_shape(lattices)
    input_data_batch = Batch.from_data_list(input_data_list)

    return (
        frac_coords, atom_types, lattices, lengths, angles, num_atoms, input_data_batch
    )


def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path).resolve()
    model, test_loader, cfg = load_model(model_path, load_data=True)

    if torch.cuda.is_available():
        model.to('cuda')
        print('CUDA is available!')

    print('Evaluate the diffusion model for CSP/Reconstruction.')

    if args.property_name:
        model_conditions = cfg.model.get('conditions', {}).keys()
        if args.property_name not in model_conditions:
            print(f"\n[WARNING] You are trying to condition on '{args.property_name}', "
                  f"but the model config only lists these conditions: {list(model_conditions)}.\n"
                  f"Please double check if this is the correct model.\n")

        if args.target_value is None:
            raise ValueError("You provided '--property_name', but forgot to provide '--target_value'.")

    step_lr = args.step_lr if args.step_lr >= 0 else recommand_step_lr['csp'][args.dataset]

    start_time = time.time()
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms, input_data_batch) = diffusion(
        test_loader,
        model,
        args.num_evals,
        step_lr,
        args.guidance_scale,
        property_name=args.property_name,
        target_value=args.target_value
    )

    if args.label == '':
        if args.property_name:
            diff_out_name = f"eval_diff_{args.property_name}_{args.target_value}_scale_{args.guidance_scale}.pt"
        else:
            diff_out_name = f"eval_diff_uncond_scale_{args.guidance_scale}.pt"
    else:
        diff_out_name = f'eval_diff_{args.label}.pt'

    print(f"\nEvaluation completed in {time.time() - start_time:.2f} seconds.")
    print(f"Results saved to: {model_path / diff_out_name}")

    torch.save({
        'eval_setting': args,
        'input_data_batch': input_data_batch,
        'frac_coords': frac_coords,
        'num_atoms': num_atoms,
        'atom_types': atom_types,
        'lattices': lattices,
        'lengths': lengths,
        'angles': angles,
        'time': time.time() - start_time,
        # 保存条件信息供后续 metrics 脚本读取分析
        'property_name': args.property_name,
        'target_value': args.target_value,
        'guidance_scale': args.guidance_scale
    },
        model_path / diff_out_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--step_lr', default=-1, type=float)
    parser.add_argument('--num_evals', default=1, type=int)
    # 引导控制参数
    parser.add_argument('--guidance_scale', default=1.0, type=float,
                        help='Classifier-free guidance scale. > 1.0 increases condition strength.')
    # 属性覆盖参数
    parser.add_argument('--property_name', type=str, default=None,
                        help='The name of the property to force upon the test set (e.g., band_gap).')
    parser.add_argument('--target_value', type=float, default=None,
                        help='The target value to overwrite in the test set.')
    parser.add_argument('--label', default='')
    args = parser.parse_args()
    main(args)