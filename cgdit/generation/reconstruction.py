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

from cgdit.common.evaluation_utils import load_model, lattices_to_params_shape, recommand_step_lr
from cgdit.generation.conditioning import (
    add_condition_arguments,
    apply_condition_values,
    condition_label,
    condition_values_from_args,
    seed_generation,
    validate_condition_values,
)

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal.symmetry import Group

import copy
import numpy as np


def diffusion(loader, model, num_evals, step_lr=1e-5, guidance_scale=1.0,
              condition_values=None, condition_configs=None):

    condition_values = condition_values or {}
    condition_configs = condition_configs or {}

    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []

    if condition_values:
        print(f"\n[INFO] Dataset conditions overridden for CSP/Reconstruction!")
        for name, value in condition_values.items():
            print(f"       Target Property: '{name}' set to {value}")
        print(f"       Using Classifier-Free Guidance with scale = {guidance_scale}\n")
    else:
        print(f"\n[INFO] Standard Unconditional/Original CSP Evaluation (CFG scale = {guidance_scale}).\n")

    for idx, batch in enumerate(loader):

        if torch.cuda.is_available():
            batch = batch.cuda()

        apply_condition_values(batch, condition_values, condition_configs)

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
    model.eval()

    if torch.cuda.is_available():
        model.to('cuda')
        print('CUDA is available!')

    print('Evaluate the diffusion model for CSP/Reconstruction.')

    condition_values = condition_values_from_args(args)
    condition_configs = cfg.model.get('conditions', {})
    validate_condition_values(condition_values, condition_configs)
    seed_generation(args.seed)

    step_lr = args.step_lr if args.step_lr >= 0 else recommand_step_lr['csp'][args.dataset]

    start_time = time.time()
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms, input_data_batch) = diffusion(
        test_loader,
        model,
        args.num_evals,
        step_lr,
        args.guidance_scale,
        condition_values=condition_values,
        condition_configs=condition_configs,
    )

    if args.label == '':
        diff_out_name = (
            f"eval_diff_{condition_label(condition_values)}_scale_{args.guidance_scale:g}.pt"
        )
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
        'conditions': condition_values,
        'property_name': args.property_name,
        'target_value': args.target_value,
        'guidance_scale': args.guidance_scale,
        'seed': args.seed,
    },
        model_path / diff_out_name)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--step_lr', default=-1, type=float)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--seed', default=9999, type=int)
    # 引导控制参数
    parser.add_argument('--guidance_scale', default=1.0, type=float,
                        help='Classifier-free guidance scale. > 1.0 increases condition strength.')
    # 属性覆盖参数
    add_condition_arguments(parser)
    parser.add_argument('--property_name', type=str, default=None,
                        help='Legacy single-condition name. Prefer repeatable --condition NAME=VALUE.')
    parser.add_argument('--target_value', type=float, default=None,
                        help='Legacy single-condition target value.')
    parser.add_argument('--label', default='')
    return parser


def cli(argv=None):
    main(build_parser().parse_args(argv))


if __name__ == '__main__':
    cli()
