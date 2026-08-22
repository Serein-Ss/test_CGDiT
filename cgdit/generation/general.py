import time
import argparse
import torch
from tqdm import tqdm
from pathlib import Path
from torch_geometric.data import DataLoader, Data
from torch.utils.data import Dataset
from cgdit.common.evaluation_utils import load_model, lattices_to_params_shape
from cgdit.common.output_paths import (
    generation_output_path,
    record_generated_structure,
)
from cgdit.generation.conditioning import (
    add_condition_arguments,
    apply_condition_values,
    condition_label,
    condition_values_from_args,
    seed_generation,
    validate_condition_values,
)
import numpy as np

from pyxtal import pyxtal
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def diffusion(
        loader,
        model,
        step_lr,
        condition_values=None,
        condition_configs=None,
        guidance_scale=1.0
):
    condition_values = condition_values or {}
    condition_configs = condition_configs or {}
    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []

    if condition_values:
        print("Dataset conditions overwritten!")
        for name, value in condition_values.items():
            print(f"Target Property: '{name}' set to {value}")
        print(f"Using Classifier-Free Guidance with scale = {guidance_scale}")
    else:
        print("Standard unconditional generation (sampling from data distribution).")

    for idx, batch in enumerate(tqdm(loader, desc="Generating")):
        if torch.cuda.is_available():
            batch = batch.cuda()

        apply_condition_values(batch, condition_values, condition_configs)

        sample_guidance_scale = guidance_scale if condition_values else 0.0
        outputs, traj = model.sample(batch, step_lr=step_lr, guidance_scale=sample_guidance_scale)

        frac_coords.append(outputs['frac_coords'].detach().cpu())
        num_atoms.append(outputs['num_atoms'].detach().cpu())
        atom_types.append(outputs['atom_types'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())

    frac_coords = torch.cat(frac_coords, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    lattices = torch.cat(lattices, dim=0)
    lengths, angles = lattices_to_params_shape(lattices)

    return (
        frac_coords, atom_types, lattices, lengths, angles, num_atoms
    )


class SampleDataset(Dataset):
    def __init__(self, dataset, total_num, seed=9999):
        super().__init__()
        self.total_num = total_num
        self.dataset = dataset
        self.is_carbon = dataset == 'carbon'
        self.seed = seed
        np.random.seed(self.seed)
        self.indexes = np.random.choice(len(self.dataset), total_num)

    def __len__(self) -> int:
        return self.total_num

    def __getitem__(self, index):
        idx = self.indexes[index]
        data = self.dataset[idx]
        return data


class AbInitioDataset(Dataset):
    """
    如果不使用 empirical_prior，就在 1-230 空间群中完全随机盲抽
    如果使用 empirical_prior，就根据训练集的统计概率来抽取 (原子数)
    """

    def __init__(self, total_num, train_set=None, use_empirical_prior=False, spacegroups=None, max_atoms=52,
                 max_atomic_num=100, seed=9999, max_attempts=1000):
        super().__init__()
        self.total_num = total_num
        self.use_empirical_prior = use_empirical_prior

        self.spacegroups = spacegroups if spacegroups is not None else list(range(1, 231))
        self.max_atoms = max_atoms
        self.max_atomic_num = max_atomic_num
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.max_attempts = max_attempts
        np.random.seed(seed + index if 'index' in locals() else seed)

        # 预先计算原子数的先验概率分布
        self.atom_choices = None
        self.atom_probs = None

        if self.use_empirical_prior and train_set is not None:
            atom_counts = []
            for i in range(len(train_set)):
                num_atoms = train_set[i].num_atoms
                n = int(num_atoms.item()) if hasattr(num_atoms, "item") else int(num_atoms)
                if n <= self.max_atoms:
                    atom_counts.append(n)

            # 计算唯一值及其对应的出现频率
            unique_atoms, counts = np.unique(atom_counts, return_counts=True)
            self.atom_choices = unique_atoms
            self.atom_probs = counts / counts.sum()

    def __len__(self) -> int:
        return self.total_num

    def __getitem__(self, index):
        success = False
        last_error = None

        for _ in range(self.max_attempts):
            # 空间群：始终在指定范围内随机抽取
            sg = int(np.random.choice(self.spacegroups))

            # 原子数：是否使用数据集的先验概率分布？
            if self.use_empirical_prior and self.atom_choices is not None:
                num_ions = int(np.random.choice(self.atom_choices, p=self.atom_probs))
            else:
                num_ions = int(np.random.randint(2, self.max_atoms + 1))

            my_crystal = pyxtal()
            try:
                # 交给 pyxtal 进行全新分配。
                my_crystal.from_random(3, sg, ['C'], [num_ions])
                if my_crystal.valid:
                    pmg_struct = my_crystal.to_pymatgen()
                    actual_num_atoms = len(pmg_struct)

                    if actual_num_atoms <= self.max_atoms:
                        success = True
            except Exception as exc:
                last_error = exc

            if success:
                break

        if not success:
            message = (
                "Unable to generate a valid ab initio structure after "
                f"{self.max_attempts} attempts."
            )
            if last_error is not None:
                raise RuntimeError(message) from last_error
            raise RuntimeError(message)

        # 构建图节点初始状态 (全为高斯噪声与 MASK)
        frac_coords = torch.rand((actual_num_atoms, 3), dtype=torch.float32)
        atom_types = torch.full((actual_num_atoms,), self.max_atomic_num, dtype=torch.long)
        lattices = torch.tensor(pmg_struct.lattice.matrix, dtype=torch.float32).unsqueeze(0)
        num_atoms_tensor = torch.tensor([actual_num_atoms], dtype=torch.long)

        # 重新提取空间群与全新的锚点信息
        sga = SpacegroupAnalyzer(pmg_struct, symprec=0.1)
        sym_dataset = sga.get_symmetry_dataset()
        spacegroup = torch.tensor([sym_dataset['number']], dtype=torch.long)

        equivalent_atoms = sym_dataset['equivalent_atoms']
        anchor_index = torch.tensor(equivalent_atoms, dtype=torch.long)

        # 动态计算新的仿射变换算子 (ops)
        ops, ops_inv = [], []
        sym_ops = sga.get_symmetry_operations()

        for i, site in enumerate(pmg_struct):
            anchor_idx = equivalent_atoms[i]
            anchor_site = pmg_struct[anchor_idx]

            found_op = None
            for op in sym_ops:
                mapped_coord = op.operate(anchor_site.frac_coords)
                diff = mapped_coord - site.frac_coords
                if np.allclose(diff - np.round(diff), 0, atol=1e-3):
                    found_op = op
                    break

            if found_op is None:
                ops.append(np.eye(4))
                ops_inv.append(np.eye(4))
            else:
                ops.append(found_op.affine_matrix)
                ops_inv.append(found_op.inverse.affine_matrix)

        ops = torch.tensor(np.array(ops), dtype=torch.float32)
        ops_inv = torch.linalg.pinv(ops[:, :3, :3])

        return Data(
            frac_coords=frac_coords,
            atom_types=atom_types,
            lattices=lattices,
            num_atoms=num_atoms_tensor,
            spacegroup=spacegroup,
            ops=ops,
            ops_inv=ops_inv,
            anchor_index=anchor_index,
            num_nodes=actual_num_atoms,
        )


def main(args):
    model_path = Path(args.model_path).resolve()
    # load_data=True 必须开启
    model, loaders, cfg = load_model(
        model_path, load_data=True, testing=False)
    model.eval()

    if torch.cuda.is_available():
        model.to('cuda')

    condition_values = condition_values_from_args(args)
    condition_configs = cfg.model.get('conditions', {})
    validate_condition_values(condition_values, condition_configs)
    seed_generation(args.seed)

    total_samples = args.batch_size * args.num_batches_to_samples

    if args.ab_initio:
        print(f"\nAb Initio 模式：动态生成晶体拓扑骨架")

        train_loader, _ = loaders
        train_set = train_loader.dataset

        if args.use_empirical_prior:
            print(f"- 经验先验 (Empirical Prior)：根据训练集统计原子数量的概率分布并进行抽取")
            print(f"- 空间群抽取：随机盲抽")
        else:
            print(f"- 完全随机 (Uniform Random)：在 1-230 空间群和 2-{args.max_atoms} 原子间等概率随机抽取。")

        target_sgs = [args.target_sg] if args.target_sg else None

        test_set = AbInitioDataset(
            total_num=total_samples,
            train_set=train_set,
            use_empirical_prior=args.use_empirical_prior,
            spacegroups=target_sgs,
            max_atoms=args.max_atoms,
            seed=args.seed
        )
    else:
        print("\nTemplate 模式：拷贝已有训练集的底层数学骨架结构")
        train_loader, _ = loaders
        train_set = train_loader.dataset
        test_set = SampleDataset(train_set, total_samples, seed=args.seed)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    start_time = time.time()

    (frac_coords, atom_types, lattices, lengths, angles, num_atoms) = diffusion(
        test_loader,
        model,
        args.step_lr,
        condition_values=condition_values,
        condition_configs=condition_configs,
        guidance_scale=args.guidance_scale
    )

    if args.label == '':
        mode_str = "ab_initio_empirical" if (args.ab_initio and args.use_empirical_prior) else \
                   ("ab_initio_random" if args.ab_initio else "template")
        label = f'{mode_str}_{condition_label(condition_values)}_scale_{args.guidance_scale:g}'
    else:
        label = args.label

    output_path = generation_output_path(model_path, label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generation completed in {time.time() - start_time:.2f}s.")
    print(f"Saving results to {output_path}")

    torch.save({
        'eval_setting': args,
        'frac_coords': frac_coords,
        'num_atoms': num_atoms,
        'atom_types': atom_types,
        'lengths': lengths,
        'angles': angles,
        'conditions': condition_values,
        'property_name': args.property_name,
        'target_value': args.target_value,
        'guidance_scale': args.guidance_scale,
        'seed': args.seed,
    }, output_path)
    record_generated_structure(output_path)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--step_lr', default=1e-5, type=float)
    parser.add_argument('--num_batches_to_samples', default=20, type=int)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--seed', default=9999, type=int)
    parser.add_argument('--label', default='')

    # 性质条件生成
    add_condition_arguments(parser)
    parser.add_argument('--property_name', type=str, default=None,
                        help='Legacy single-condition name. Prefer repeatable --condition NAME=VALUE.')
    parser.add_argument('--target_value', type=float, default=None,
                        help='Legacy single-condition target value.')
    parser.add_argument('--guidance_scale', type=float, default=1.0,
                        help='Classifier-free guidance scale.')

    # Ab Initio 模式及其优化
    parser.add_argument('--ab_initio', action='store_true',
                        help='Enable true ab initio generation without relying on dataset structures.')
    parser.add_argument('--use_empirical_prior', action='store_true',
                        help='If using ab_initio, force the (spacegroup, num_atoms) distribution to match the training dataset.')
    parser.add_argument('--target_sg', type=int, default=None,
                        help='Specific spacegroup to generate in ab initio mode (e.g. 225).')
    parser.add_argument('--max_atoms', type=int, default=52,
                        help='The maximum number of atoms allowed per crystal when generating ab initio.')

    return parser


def cli(argv=None):
    main(build_parser().parse_args(argv))


if __name__ == '__main__':
    cli()
