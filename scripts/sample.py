import argparse
import torch
from pymatgen.io.cif import CifWriter
from sample_api import construct_dataset_from_json, construct_dataset_from_syminfo, generate_structures_from_dataset
from pathlib import Path
import os


def main(args):
    args.model_path = str(Path(args.model_path).resolve())
    args.save_path = str(Path(args.save_path).resolve())
    tar_dir = args.save_path
    os.makedirs(tar_dir, exist_ok=True)

    if args.json_file != '':
        dataset = construct_dataset_from_json(args.json_file)

    else:
        assert args.spacegroup > 0 and args.wyckoff_letters != ''
        if ',' in args.wyckoff_letters:
            wyckoff_letters = args.wyckoff_letters.split(',')
        else:
            wyckoff_letters = args.wyckoff_letters

        if args.atom_types != '':
            atom_types = args.atom_types.split(',')
        else:
            atom_types = None

        dataset = construct_dataset_from_syminfo(args.spacegroup, wyckoff_letters, atom_types)

    # 在这里将 guidance_scale 传给 API
    structure_list = generate_structures_from_dataset(
        args.model_path,
        dataset,
        args.batch_size,
        args.step_lr,
        args.guidance_scale
    )

    print("Saving structures.")
    for i, structure in enumerate(structure_list):
        tar_file = os.path.join(tar_dir, f"{i + 1}.cif")
        if structure is not None:
            writer = CifWriter(structure)
            writer.write_file(tar_file)
        else:
            print(f"{i + 1} Error Structure.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--step_lr', default=1e-5, type=float)
    parser.add_argument('--spacegroup', default=0, type=int)
    parser.add_argument('--wyckoff_letters', default='', type=str)
    parser.add_argument('--atom_types', default='', type=str)
    parser.add_argument('--json_file', default='', type=str)

    # 新增 guidance_scale 命令行参数，默认保持为 1.0 (即不放大引导)
    parser.add_argument('--guidance_scale', default=1.0, type=float,
                        help="Classifier-Free Guidance Scale. >1.0 increases condition fidelity.")

    args = parser.parse_args()

    main(args)