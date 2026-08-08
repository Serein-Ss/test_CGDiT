# @Author : Serein
# @Time : 2025/11/8 21:57
from collections import Counter
import argparse
import os
import json

import numpy as np
from pathlib import Path
from tqdm import tqdm
from p_tqdm import p_map
from scipy.stats import wasserstein_distance
import pandas as pd

from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty
from pyxtal import pyxtal
import pickle

import sys

sys.path.append('.')

from eval_utils import (
    smact_validity, structure_validity, CompScaler, get_fp_pdist,
    load_config, load_data, get_crystals_list, prop_model_eval, compute_cov)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

Percentiles = {
    'mp20': np.array([-3.17562208, -2.82196882, -2.52814761]),
    'carbon': np.array([-154.527093, -154.45865733, -154.44206825]),
    'perovskite': np.array([0.43924842, 0.61202443, 0.7364607]),
}

COV_Cutoffs = {
    'mp20': {'struc': 0.4, 'comp': 10.},
    'carbon': {'struc': 0.2, 'comp': 4.},
    'perovskite': {'struc': 0.2, 'comp': 4},
}


class Crystal(object):

    def __init__(self, crys_array_dict, model_output=False):
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.dict = crys_array_dict

        if model_output or np.any(self.atom_types == 0):
            self.atom_types = self.atom_types + 1
            self.dict['atom_types'] = self.atom_types

        self.get_structure()
        self.get_composition()
        self.get_validity()
        self.get_fingerprints()

    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        if np.isnan(self.lengths).any() or np.isnan(self.angles).any() or np.isnan(self.frac_coords).any():
            self.constructed = False
            self.invalid_reason = 'nan_value'
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = 'unrealistically_small_lattice'

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        try:
            self.comp_valid = smact_validity(self.elems, self.comps)
        except Exception:
            self.comp_valid = False

        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [CrystalNNFP.featurize(
                self.structure, i) for i in range(len(self.structure))]
        except Exception:
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)


class RecEval(object):

    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3):
        assert len(pred_crys) == len(gt_crys)
        self.matcher = StructureMatcher(
            stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys

    def get_match_rate_and_rms(self):
        def process_one(pred, gt, is_valid):
            if not is_valid:
                return None
            try:
                rms_dist = self.matcher.get_rms_dist(
                    pred.structure, gt.structure)
                rms_dist = None if rms_dist is None else rms_dist[0]
                return rms_dist
            except Exception:
                return None

        validity = [c1.valid and c2.valid for c1, c2 in zip(self.preds, self.gts)]

        rms_dists = []
        for i in tqdm(range(len(self.preds))):
            rms_dists.append(process_one(
                self.preds[i], self.gts[i], validity[i]))
        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(self.preds)
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        return {'match_rate': match_rate,
                'rms_dist': mean_rms_dist}

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms())
        return metrics


class GenEval(object):

    def __init__(self, pred_crys, gt_crys, n_samples=1000, eval_model_name=None, calc_prop=True, prop_model_path=None):
        self.crys = pred_crys
        self.gt_crys = gt_crys
        self.n_samples = n_samples
        self.eval_model_name = eval_model_name

        # 新增控制参数
        self.calc_prop = calc_prop
        self.prop_model_path = prop_model_path

        valid_crys = [c for c in pred_crys if c.valid]
        if len(valid_crys) >= n_samples:
            sampled_indices = np.random.choice(
                len(valid_crys), n_samples, replace=False)
            self.valid_samples = [valid_crys[i] for i in sampled_indices]
        else:
            raise Exception(
                f'not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}')

    def get_validity(self):
        comp_valid = np.array([c.comp_valid for c in self.crys]).mean()
        struct_valid = np.array([c.struct_valid for c in self.crys]).mean()
        valid = np.array([c.valid for c in self.crys]).mean()
        return {'comp_valid': comp_valid,
                'struct_valid': struct_valid,
                'valid': valid}

    def get_density_wdist(self):
        pred_densities = [c.structure.density for c in self.valid_samples]
        gt_densities = [c.structure.density for c in self.gt_crys]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {'wdist_density': wdist_density}

    def get_num_elem_wdist(self):
        pred_nelems = [len(set(c.structure.species))
                       for c in self.valid_samples]
        gt_nelems = [len(set(c.structure.species)) for c in self.gt_crys]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {'wdist_num_elems': wdist_num_elems}

    def get_prop_wdist(self):
        # 1. 检查是否跳过计算
        if not self.calc_prop:
            print("Property calculation is disabled. Skipping.")
            return {'wdist_prop': None}

        # 2. 如果提供了 prop_model_path，则使用该路径(或matgl预训练)来预测
        if self.prop_model_path is not None:
            import torch
            import matgl

            if self.prop_model_path.lower() == 'matgl':
                print("Loading pre-trained matgl BandGap model...")
                # 默认加载 matgl 内部预训练好的带隙模型
                try:
                    model = matgl.load_model("MEGNet-MP-2018.6.1-BandGap-mfi")
                except:
                    model = matgl.load_model("M3GNet-MP-2018.6.1-Eform")
            else:
                print(f"Loading local matgl checkpoint from {self.prop_model_path}...")
                if self.prop_model_path.endswith('.ckpt'):
                    # 尝试用 PyTorch Lightning 的格式加载
                    from matgl.ext.pytorch_lightning import MatglLightningModule
                    try:
                        lit_module = MatglLightningModule.load_from_checkpoint(self.prop_model_path)
                        model = lit_module.model
                    except Exception as e:
                        print(f"Lightning Load Failed: {e}. Trying direct PyTorch load...")
                        model = torch.load(self.prop_model_path)
                else:
                    # 正常保存的 matgl model 目录
                    model = matgl.load_model(self.prop_model_path)

            # 使用模型进行预测辅助函数
            def predict_properties(crys_list, desc="Predicting Properties"):
                props = []
                for c in tqdm(crys_list, desc=desc):
                    try:
                        val = model.predict_structure(c.structure)
                        if isinstance(val, torch.Tensor):
                            val = val.detach().cpu().item()
                        props.append(float(val))
                    except Exception as e:
                        # 对于生成出的偶尔完全不合法的结构提供 fallback
                        props.append(0.0)
                return props

            pred_props = predict_properties(self.valid_samples, desc="Predicting Generated Properties")
            gt_props = predict_properties(self.gt_crys, desc="Predicting Ground Truth Properties")

            wdist_prop = wasserstein_distance(pred_props, gt_props)
            return {'wdist_prop': wdist_prop}

        # 3. 如果没提供路径但启用了计算，回退到原代码原有的逻辑（通过 hparams 寻找）
        elif self.eval_model_name is not None:
            print("Using original property model fallback (via Hydra)...")
            pred_props = prop_model_eval(self.eval_model_name, [
                c.dict for c in self.valid_samples])
            gt_props = prop_model_eval(self.eval_model_name, [
                c.dict for c in self.gt_crys])
            wdist_prop = wasserstein_distance(pred_props, gt_props)
            return {'wdist_prop': wdist_prop}

        else:
            return {'wdist_prop': None}

    def get_coverage(self):
        cutoff_dict = COV_Cutoffs[self.eval_model_name]
        (cov_metrics_dict, combined_dist_dict) = compute_cov(
            self.crys, self.gt_crys,
            struc_cutoff=cutoff_dict['struc'],
            comp_cutoff=cutoff_dict['comp'])
        return cov_metrics_dict

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_validity())
        metrics.update(self.get_density_wdist())
        metrics.update(self.get_prop_wdist())
        metrics.update(self.get_num_elem_wdist())
        metrics.update(self.get_coverage())
        return metrics


def get_file_paths(root_path, task, label='', suffix='pt'):
    if args.label == '':
        out_name = f'eval_{task}.{suffix}'
    else:
        out_name = f'eval_{task}_{label}.{suffix}'
    out_name = os.path.join(root_path, out_name)
    return out_name


def get_crystal_array_list(file_path, batch_idx=0):
    data = load_data(file_path)
    if batch_idx == -1:
        batch_size = data['frac_coords'].shape[0]
        crys_array_list = []
        for i in range(batch_size):
            tmp_crys_array_list = get_crystals_list(
                data['frac_coords'][i],
                data['atom_types'][i],
                data['lengths'][i],
                data['angles'][i],
                data['num_atoms'][i])
            crys_array_list.append(tmp_crys_array_list)
    elif batch_idx == -2:
        crys_array_list = get_crystals_list(
            data['frac_coords'],
            data['atom_types'],
            data['lengths'],
            data['angles'],
            data['num_atoms'])
    else:
        crys_array_list = get_crystals_list(
            data['frac_coords'][batch_idx],
            data['atom_types'][batch_idx],
            data['lengths'][batch_idx],
            data['angles'][batch_idx],
            data['num_atoms'][batch_idx])

    if 'input_data_batch' in data:
        batch = data['input_data_batch']
        if isinstance(batch, dict):
            true_crystal_array_list = get_crystals_list(
                batch['frac_coords'], batch['atom_types'], batch['lengths'],
                batch['angles'], batch['num_atoms'])
        else:
            true_crystal_array_list = get_crystals_list(
                batch.frac_coords, batch.atom_types, batch.lengths,
                batch.angles, batch.num_atoms)
    else:
        true_crystal_array_list = None

    return crys_array_list, true_crystal_array_list


def get_gt_crys_ori(cif):
    structure = Structure.from_str(cif, fmt='cif')
    lattice = structure.lattice
    crys_array_dict = {
        'frac_coords': structure.frac_coords,
        'atom_types': np.array([_.Z for _ in structure.species]),
        'lengths': np.array(lattice.abc),
        'angles': np.array(lattice.angles)
    }
    return Crystal(crys_array_dict)


def process_data_parallel(data_list, func, num_workers=1, desc="Processing"):
    """
    一个通用的数据处理函数，支持单进程和多进程切换。
    解决内存爆炸的关键：当 num_workers <= 1 时，不使用 p_map，而是使用普通循环。
    """
    if num_workers > 1:
        return p_map(func, data_list, num_cpus=num_workers, desc=desc)
    else:
        results = []
        for item in tqdm(data_list, desc=desc):
            results.append(func(item))
        return results


def main(args):
    args.root_path = str(Path(args.root_path).resolve())
    all_metrics = {}

    cfg = load_config(args.root_path)
    eval_model_name = cfg.data.eval_model_name

    # 转化布尔参数
    calc_prop = (str(args.calc_prop).lower() in ['true', '1', 'yes'])

    if 'gen' in args.tasks:

        gen_file_path = get_file_paths(args.root_path, 'gen', args.label)
        recon_file_path = get_file_paths(args.root_path, 'recon', args.label)
        crys_array_list, _ = get_crystal_array_list(gen_file_path, batch_idx=-2)

        gen_crys = process_data_parallel(
            crys_array_list,
            lambda x: Crystal(x, model_output=True),
            num_workers=args.num_workers,
            desc="Gen Crystals"
        )

        if args.gt_file != '':
            csv = pd.read_csv(args.gt_file)
            gt_crys = process_data_parallel(
                csv['cif'],
                get_gt_crys_ori,
                num_workers=args.num_workers,
                desc="GT Crystals"
            )
            gt_crys = [c for c in gt_crys if c is not None]
        else:
            _, true_crystal_array_list = get_crystal_array_list(
                recon_file_path)
            gt_crys = process_data_parallel(
                true_crystal_array_list,
                lambda x: Crystal(x, model_output=False),
                num_workers=args.num_workers,
                desc="GT Crystals"
            )

        # 传递计算性质选项
        gen_evaluator = GenEval(
            gen_crys, gt_crys,
            eval_model_name=eval_model_name,
            calc_prop=calc_prop,
            prop_model_path=args.prop_model_path
        )
        gen_metrics = gen_evaluator.get_metrics()
        all_metrics.update(gen_metrics)

    else:

        recon_file_path = get_file_paths(args.root_path, 'diff', args.label)
        batch_idx = 0
        crys_array_list, true_crystal_array_list = get_crystal_array_list(
            recon_file_path, batch_idx=batch_idx)
        if args.gt_file != '':
            csv = pd.read_csv(args.gt_file)
            gt_crys = process_data_parallel(
                csv['cif'],
                get_gt_crys_ori,
                num_workers=args.num_workers,
                desc="GT Crystals"
            )
            gt_crys = [c for c in gt_crys if c is not None]

        else:
            gt_crys = process_data_parallel(
                true_crystal_array_list,
                lambda x: Crystal(x, model_output=False),
                num_workers=args.num_workers,
                desc="GT Crystals"
            )

        pred_crys = process_data_parallel(
            crys_array_list,
            lambda x: Crystal(x, model_output=True),
            num_workers=args.num_workers,
            desc="Pred Crystals"
        )
        rec_evaluator = RecEval(pred_crys, gt_crys)
        recon_metrics = rec_evaluator.get_metrics()
        all_metrics.update(recon_metrics)

    print(all_metrics)

    task_str = "_".join(args.tasks)
    if args.label == '':
        metrics_out_file = f'eval_metrics_{task_str}.json'
    else:
        metrics_out_file = f'eval_metrics_{task_str}_{args.label}.json'
    metrics_out_file = os.path.join(args.root_path, metrics_out_file)

    # only overwrite metrics computed in the new run.
    if Path(metrics_out_file).exists():
        with open(metrics_out_file, 'r') as f:
            written_metrics = json.load(f)
            if isinstance(written_metrics, dict):
                written_metrics.update(all_metrics)
            else:
                with open(metrics_out_file, 'w') as f:
                    json.dump(all_metrics, f)
        if isinstance(written_metrics, dict):
            with open(metrics_out_file, 'w') as f:
                json.dump(written_metrics, f)
    else:
        with open(metrics_out_file, 'w') as f:
            json.dump(all_metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--label', default='')
    parser.add_argument('--tasks', nargs='+', default=['csp'])
    parser.add_argument('--gt_file', default='')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='Number of worker processes. Default 1 (single process) to save memory.')

    # --- 新增参数 ---
    parser.add_argument('--calc_prop', type=str, default='true', choices=['true', 'false'],
                        help='Whether to calculate property distance metrics (default: true)')
    parser.add_argument('--prop_model_path', type=str, default=None,
                        help='Path to the property predictor (.ckpt or dir). Use "matgl" to load a default pre-trained matgl BandGap model.')

    args = parser.parse_args()
    main(args)