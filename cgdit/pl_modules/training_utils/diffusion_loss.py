# @Author : Serein
# @Time : 2025/11/30 11:09
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from typing import Optional, Callable, Dict

from cgdit.pl_modules.diff_utils.diff_utils import d_log_p_wrapped_normal
from cgdit.pl_modules.diff_utils.discrete_diff_utils import (
DiscreteDiffusionBase, compute_kl_reverse_process
)
from cgdit.rl.symmetry_quotient import representative_indices


class DiffusionLoss(nn.Module):
    def __init__(self,
                 d3pm_diffusion,
                 num_atom_types,
                 cost_lattice=1.0,
                 cost_coord=1.0,
                 cost_atom=1.0,
                 hybrid_lambda=0.01):
        """
        初始化损失计算模块
        Args:
            d3pm_diffusion: 离散扩散过程的实例 (MaskDiffusion)
            num_atom_types: 真实原子类型的数量 (不含 Mask)
            cost_lattice: 晶格损失权重
            cost_coord: 坐标损失权重
            cost_atom: 原子类型损失权重
            hybrid_lambda: D3PM Loss 中的混合权重
        """
        super().__init__()
        self.d3pm = d3pm_diffusion
        self.num_atom_types = num_atom_types
        self.cost_lattice = cost_lattice
        self.cost_coord = cost_coord
        self.cost_atom = cost_atom
        self.hybrid_lambda = hybrid_lambda

    def calc_lattice_loss(self, pred_crys_fam, target_crys_fam):
        """
        计算晶格参数的 MSE Loss
        """
        return F.mse_loss(pred_crys_fam, target_crys_fam)

    def calc_coord_loss(self, pred_x, batch, rand_x_anchor, sigmas_per_atom, sigmas_norm_per_atom):
        """
        计算分数坐标的 Score Matching Loss
        Args:
            pred_x: 模型预测的坐标输出 (全局坐标，需要投影回 anchor)
            batch: 图数据 batch，包含 ops_inv 等
            rand_x_anchor: 原始加噪时使用的 anchor 噪声
            sigmas_per_atom: 每个原子的 sigma
            sigmas_norm_per_atom: 每个原子的归一化 sigma
        """
        # pred_x shape: [N, 3] -> pred_x_proj shape: [N, 3]
        pred_x_proj = torch.einsum('bij, bj-> bi', batch.ops_inv, pred_x)

        # tar_x_anchor = \nabla_x log p(x)
        tar_x_anchor = d_log_p_wrapped_normal(
            sigmas_per_atom * rand_x_anchor,
            sigmas_per_atom
        ) / torch.sqrt(sigmas_norm_per_atom)

        loss_coord = F.mse_loss(pred_x_proj, tar_x_anchor)
        return loss_coord

    def calc_atom_loss(self, pred_atom_logits, x_start_atoms, input_atom_types,
                       t_discrete, batch_idx, num_graphs, anchor_index):
        """
        计算原子类型的 D3PM Loss (VB term or Cross Entropy)
        Args:
            pred_atom_logits: 模型输出的 logits
            x_start_atoms: 真实的原子类型 (0-99)
            input_atom_types: 当前时刻加噪后的原子类型 (包含 MASK)
            t_discrete: 离散时间步
            batch_idx: batch index vector
            num_graphs: batch size
        """
        orbit_representatives = representative_indices(anchor_index)
        loss_atom = compute_d3pm_loss(
            diffusion=self.d3pm,
            score_model_logits=pred_atom_logits[orbit_representatives],
            t=t_discrete[orbit_representatives],
            x_start=x_start_atoms[orbit_representatives],
            x_t=input_atom_types[orbit_representatives],
            batch_idx=batch_idx[orbit_representatives],
            batch_size=num_graphs,
            hybrid_lambda=self.hybrid_lambda,
            reduce='mean',
        )
        return loss_atom

    def forward(self,
                pred_crys_fam, rand_crys_fam,
                pred_x, batch, rand_x_anchor, sigmas_per_atom, sigmas_norm_per_atom,
                pred_atom_logits, x_start_atoms, input_atom_types, t_discrete):
        """
        前向传播计算总损失
        """
        # 1. Lattice Loss
        loss_lattice = self.calc_lattice_loss(pred_crys_fam, rand_crys_fam)

        # 2. Coordinate Loss
        loss_coord = self.calc_coord_loss(
            pred_x, batch, rand_x_anchor, sigmas_per_atom, sigmas_norm_per_atom
        )

        # 3. Atom Type Loss
        loss_atom = self.calc_atom_loss(
            pred_atom_logits,
            x_start_atoms,
            input_atom_types,
            t_discrete,
            batch.batch,
            batch.num_graphs,
            batch.anchor_index,
        )

        # 4. Weighted Sum
        total_loss = (
            self.cost_lattice * loss_lattice +
            self.cost_coord * loss_coord +
            self.cost_atom * loss_atom
        )

        return {
            'loss': total_loss,
            'loss_lattice': loss_lattice,
            'loss_coord': loss_coord,
            'loss_atom_types': loss_atom
        }


# 计算每个样本的聚合损失
def aggregate_per_sample(
        loss_per_row: torch.Tensor,
        batch_idx: Optional[torch.Tensor],
        reduce: str,
        batch_size: int,
):
    """
    将每个原子(node)的损失聚合为每个样本(sample/graph)的损失。

    Args:
        loss_per_row: [num_atoms, ...] 每个原子的损失
        batch_idx: [num_atoms] 原子的 batch 索引 (PyG 的 batch.batch)
        reduce: 'mean' 或 'sum'
        batch_size: batch 中的图数量
    """
    # 先把多余的维度平均掉 (例如如果有多个特征维度)
    if loss_per_row.ndim > 1:
        loss_per_row = torch.mean(loss_per_row.reshape(loss_per_row.shape[0], -1), dim=1)

    if batch_idx is None:
        return loss_per_row
    else:
        # 使用 scatter 进行聚合
        loss_per_sample = scatter(
            src=loss_per_row,
            index=batch_idx,
            dim=0,
            dim_size=batch_size,
            reduce=reduce,
        )
    return loss_per_sample


def compute_d3pm_loss(
        diffusion: DiscreteDiffusionBase,
        score_model_logits: torch.Tensor,
        t: torch.Tensor,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        batch_idx: torch.Tensor,
        batch_size: int,
        hybrid_lambda: float = 0.01,
        reduce: str = "mean",
) -> torch.Tensor:
    """
    计算 D3PM 混合损失 (KL Divergence + Auxiliary CrossEntropy)。

    Args:
        diffusion: D3PM 扩散对象 (MaskDiffusion)
        score_model_logits: 模型预测的 x0 的 logits [num_atoms, num_classes]
        t: 当前时间步 [batch_size]
        x_start: 真实的 x0 (原子类型) [num_atoms]
        x_t: 当前噪声输入 xt (原子类型) [num_atoms]
        batch_idx: 原子的 batch 索引 [num_atoms]
        batch_size: 样本数量
        hybrid_lambda: 辅助损失系数
        reduce: 聚合方式 'mean' 或 'sum'
    """
    current_dim = score_model_logits.shape[-1] # 100
    target_dim = diffusion.dim  # 101

    if current_dim == target_dim - 1:
        # F.pad 参数格式 (left, right, top, bottom, ...)
        # (0, 1) 表示在最后一个维度的左边补0列，右边补1列
        score_model_logits = F.pad(score_model_logits, (0, 1), value=-1e9)
    # 1. 时间步对齐
    # diffusion.py 里传入的 t 是 [Batch_Size]，但 compute_kl 需要 [Num_Atoms]
    if t.size(0) == batch_size:
        t_per_atom = t[batch_idx]
    elif t.size(0) == x_start.size(0):
        t_per_atom = t
    else:
        # Fallback (in case of single sample batch) or mismatch
        t_per_atom = t[batch_idx]

    # 2. 定义去噪函数接口
    # compute_kl_reverse_process 需要一个函数 fn(x, t) -> logits
    # 因为我们已经算好了 logits，直接返回即可
    def denoise_fn(targets, timestep):
        return score_model_logits

    # 3. 调用核心计算函数 (来自 discrete_diff_utils.py)
    # 注意：这里的 x_start 和 x_t 必须是 LongTensor
    metrics_dict = compute_kl_reverse_process(
        x_start=x_start.long(),
        t=t_per_atom,
        x_t_plus_1=x_t.long(),
        diffusion=diffusion,
        denoise_fn=denoise_fn,
        predict_x0=True,  # 我们预测的是 x0
        log_space=True,  # 我们提供的是 logits
        hybrid_lambda=hybrid_lambda,
        use_cached_transition=False  # 简单起见，不缓存
    )

    # 4. 获取逐原子的总损失
    loss_per_atom = metrics_dict["loss"]

    # 5. 聚合回逐样本 (Per-Structure) 损失
    loss_per_structure = aggregate_per_sample(
        loss_per_atom,
        batch_idx=batch_idx,
        reduce=reduce,
        batch_size=batch_size
    )

    # 6. 对所有样本求平均，得到最终标量 Loss
    return loss_per_structure.mean()
