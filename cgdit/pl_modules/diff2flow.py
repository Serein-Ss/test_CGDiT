# @Author : Serein
# @Time : 2026/5/17 22:03
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
import hydra
import pytorch_lightning as pl
from torch_scatter import scatter
from tqdm import tqdm

from cgdit.common.data_utils import lattice_params_to_matrix_torch
from cgdit.pl_modules.lattice.crystal_family import CrystalFamily
from cgdit.pl_modules.diff_utils.discrete_diff_utils import (
    MaskDiffusion, create_discrete_diffusion_schedule, q_sample
)
from cgdit.pl_modules.cfg_utils import ConditioningEncoder

MAX_ATOMIC_NUM = 100


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        if hasattr(self.hparams, "model"):
            self._hparams = self.hparams.model

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        val_check_interval = self.hparams.logging.get('val_check_interval', 1)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": val_check_interval,
            },
        }


class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # time 现在是 [0, 1] 的连续浮点数
        # 为了与原本的 scale 对齐，可以将其放大，例如乘以 1000
        time = time * 1000.0
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class FlowMatchingCrystal(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.decoder = hydra.utils.instantiate(
            self.hparams.decoder,
            latent_dim=self.hparams.time_dim,
            _recursive_=False
        )
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.crystal_family = CrystalFamily()

        # --- 离散变量：保持 D3PM 逻辑，将连续 t 映射到离散 timesteps ---
        self.num_atom_types = MAX_ATOMIC_NUM
        self.mask_token_id = MAX_ATOMIC_NUM
        d3pm_hparams = self.hparams.d3pm_scheduler
        self.discrete_timesteps = 1000  # 假设使用1000步作为离散的刻度
        self.d3pm_schedule = create_discrete_diffusion_schedule(
            kind=d3pm_hparams.kind,
            beta_min=d3pm_hparams.beta_min,
            beta_max=d3pm_hparams.beta_max,
            num_steps=self.discrete_timesteps
        )
        self.d3pm = MaskDiffusion(
            dim=self.num_atom_types + 1,
            schedule=self.d3pm_schedule
        )
        self.q_sample = q_sample

        # --- Classifier-Free Guidance ---
        condition_configs = self.hparams.get('conditions', {})
        cond_dropout_prob = self.hparams.get('cond_dropout_prob', 0.1)
        self.conditioner = ConditioningEncoder(
            time_dim=self.time_dim,
            condition_configs=condition_configs,
            cond_dropout_prob=cond_dropout_prob
        )

        # 连续变量（晶格、坐标）我们将在 forward 中直接使用 Flow Matching (MSE) 损失
        # 离散变量的损失如果你有定制的类，可以继续调用
        self.hybrid_lambda = self.hparams.get('hybrid_lambda', 0.01)

    def forward(self, batch, batch_idx=None):
        batch_size = batch.num_graphs
        dev = self.device

        # 1. Flow Matching 时间采样 t ~ U(0, 1)
        times = torch.rand(batch_size, device=dev)
        time_emb = self.time_embedding(times)

        cond_emb = self.conditioner(batch)
        time_emb = time_emb + cond_emb

        # 2. 晶格 Flow Matching (Lattice)
        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        lattices = self.crystal_family.de_so3(lattices)
        x1_crys_fam = self.crystal_family.m2v(lattices)
        x1_crys_fam = self.crystal_family.proj_k_to_spacegroup(x1_crys_fam, batch.spacegroup)

        x0_crys_fam = torch.randn_like(x1_crys_fam)
        x0_crys_fam = self.crystal_family.proj_k_to_spacegroup(x0_crys_fam, batch.spacegroup)

        # 核心直线插值：x_t = t * x_1 + (1-t) * x_0
        t_crys = times[:, None]
        input_crys_fam = t_crys * x1_crys_fam + (1. - t_crys) * x0_crys_fam
        input_crys_fam = self.crystal_family.proj_k_to_spacegroup(input_crys_fam, batch.spacegroup)
        # Flow 的预测目标是速度场：v_t = x_1 - x_0
        target_v_crys = x1_crys_fam - x0_crys_fam

        # 3. 坐标 Flow Matching (Coordinates - 注意环面拓扑 Torus)
        x1_frac_coords = batch.frac_coords
        x0_frac_coords_raw = torch.randn_like(x1_frac_coords)

        # 保持对称性约束
        rand_x_anchor = x0_frac_coords_raw[batch.anchor_index]
        rand_x_anchor = (batch.ops_inv[batch.anchor_index] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
        x0_frac_coords = (batch.ops[:, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)

        # Torus 上的最短路径流：确保差值在 [-0.5, 0.5] 之间
        diff_coord = (x1_frac_coords - x0_frac_coords + 0.5) % 1.0 - 0.5
        target_v_coord = diff_coord

        t_coord = times.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (x0_frac_coords + t_coord * diff_coord) % 1.0

        # 4. 离散原子类型 D3PM 加噪 (兼容 Flow 时间)
        x_start_atoms = batch.atom_types.long() - 1
        x_start_atoms = torch.clamp(x_start_atoms, min=0, max=self.num_atom_types - 1)

        times_per_atom = times.repeat_interleave(batch.num_atoms)
        t_discrete = (times_per_atom * (self.discrete_timesteps - 1)).long() + 1

        input_atom_types = self.q_sample(
            x_start=x_start_atoms,
            t=t_discrete,
            diffusion=self.d3pm,
            return_logits=False
        )

        # 5. 模型前向预测
        pred_v_crys, pred_v_coord, pred_atom_types_logits = self.decoder(
            time_emb,
            input_atom_types,
            input_frac_coords,
            input_crys_fam,
            batch.num_atoms,
            batch.batch
        )
        pred_v_crys = self.crystal_family.proj_k_to_spacegroup(pred_v_crys, batch.spacegroup)

        # 6. 计算损失
        # Flow Matching Loss 是非常干净的 MSE，直接监督向量场 (Vector Field)
        loss_lattice = F.mse_loss(pred_v_crys, target_v_crys)
        loss_coord = F.mse_loss(pred_v_coord, target_v_coord)

        # 原子的离散交叉熵损失 (此处保留一个简化的实现，你可以接回你的 self.loss_fn)
        loss_atom_types = F.cross_entropy(
            pred_atom_types_logits.view(-1, self.num_atom_types + 1),
            x_start_atoms.view(-1)
        )

        loss = loss_lattice * self.hparams.cost_lattice + \
               loss_coord * self.hparams.cost_coord + \
               loss_atom_types * self.hparams.cost_atom

        return {
            'loss': loss,
            'loss_lattice': loss_lattice,
            'loss_coord': loss_coord,
            'loss_atom_types': loss_atom_types
        }

    @torch.no_grad()
    def sample(self, batch, num_steps=100, guidance_scale=1.0, fixed_atom_types=None):
        """
        基于 Euler 方法的 Flow Matching 采样
        """
        batch_size = batch.num_graphs

        # --- 初始化 x_0 (对应 Flow 的 t=0，也就是纯噪声) ---
        x_t_coord = torch.rand([batch.num_nodes, 3]).to(self.device)
        crys_fam_t = torch.randn([batch_size, 6]).to(self.device)
        crys_fam_t = self.crystal_family.proj_k_to_spacegroup(crys_fam_t, batch.spacegroup)

        if fixed_atom_types is not None:
            fixed_atom_types = fixed_atom_types.long().to(self.device)
            fixed_atom_types = torch.clamp(fixed_atom_types, min=0, max=self.num_atom_types - 1)
            t_max_tensor = torch.full(fixed_atom_types.shape, self.discrete_timesteps - 1,
                                      device=self.device, dtype=torch.long)
            atom_types_t, _ = self.q_sample(x_start=fixed_atom_types, t=t_max_tensor, diffusion=self.d3pm)
        else:
            atom_types_t = self.d3pm.sample_stationary(batch.atom_types.shape).to(self.device)

        # 确保 x_0 的坐标也符合晶体对称性
        x_t_anchor = x_t_coord[batch.anchor_index]
        x_t_anchor = (batch.ops_inv[batch.anchor_index] @ x_t_anchor.unsqueeze(-1)).squeeze(-1)
        x_t_coord = (batch.ops[:, :3, :3] @ x_t_anchor.unsqueeze(-1)).squeeze(-1)

        # 积分步长 dt
        dt = 1.0 / num_steps

        # --- 欧拉积分 (Euler Integration) 从 t=0 到 t=1 ---
        for step in tqdm(range(num_steps)):
            t = step * dt
            times = torch.full((batch_size,), t, device=self.device)
            time_emb = self.time_embedding(times)

            # 获取模型的预测速度 (Velocity)
            pred_v_crys, pred_v_coord, pred_atom_types_logits = self._get_model_output(
                time_emb, atom_types_t, x_t_coord, crys_fam_t,
                batch.num_atoms, batch.batch, batch_obj=batch, guidance_scale=guidance_scale
            )

            # --- 更新连续变量 (x_{t+dt} = x_t + v_t * dt) ---
            # 1. 更新晶格
            crys_fam_t = crys_fam_t + dt * pred_v_crys
            crys_fam_t = self.crystal_family.proj_k_to_spacegroup(crys_fam_t, batch.spacegroup)

            # 2. 更新坐标
            # 保持坐标预测更新时的对称性
            pred_x_proj = torch.einsum('bij, bj-> bi', batch.ops_inv, pred_v_coord)
            pred_x_anchor = scatter(pred_x_proj, batch.anchor_index, dim=0, reduce='mean')[batch.anchor_index]
            pred_v_coord = (batch.ops[:, :3, :3] @ pred_x_anchor.unsqueeze(-1)).squeeze(-1)

            x_t_coord = (x_t_coord + dt * pred_v_coord) % 1.0

            # --- 3. 更新离散变量 (原子的 D3PM 离散反向步骤) ---
            # Flow 方向是从 0 -> 1，而扩散的去噪方向是从 T -> 0。
            # 需要将 Flow 的步数对应回 Discrete 的降噪步数
            diff_t_current = int((1.0 - t) * self.discrete_timesteps)
            diff_t_next = int((1.0 - (t + dt)) * self.discrete_timesteps)

            if fixed_atom_types is None:
                pred_x0_atom_logits = pred_atom_types_logits[:, :self.num_atom_types]
                t_discrete_tensor = torch.full(atom_types_t.shape, diff_t_current, device=self.device, dtype=torch.long)

                full_logits = torch.full((atom_types_t.shape[0], self.num_atom_types + 1), -1e9, device=self.device)
                full_logits[:, :self.num_atom_types] = pred_x0_atom_logits
                pred_x0_atom_probs = full_logits.softmax(dim=-1)

                p_t_minus_1_logits, _ = self.d3pm.sample_and_compute_posterior_q(
                    x_0=pred_x0_atom_probs, t=t_discrete_tensor - 1, samples=atom_types_t.long(),
                    make_one_hot=False, return_logits=True, step_size=max(1, diff_t_current - diff_t_next)
                )

                if diff_t_next > 0:
                    atom_types_t = torch.distributions.Categorical(logits=p_t_minus_1_logits).sample()
                else:
                    atom_types_t = torch.argmax(p_t_minus_1_logits, dim=-1)

        l_final = self.crystal_family.v2m(crys_fam_t)

        # 输出最终结构
        return {
            'atom_types': atom_types_t,
            'frac_coords': x_t_coord % 1.,
            'lattices': l_final,
            'crys_fam': crys_fam_t
        }

    def _get_model_output(self, time_emb_base, atom_types, frac_coords, crys_fam,
                          num_atoms, batch_idx, batch_obj, guidance_scale=1.0):
        if guidance_scale == 1.0:
            cond_emb = self.conditioner(batch_obj, force_mask=False)
            return self.decoder(time_emb_base + cond_emb, atom_types, frac_coords, crys_fam, num_atoms, batch_idx)

        cond_emb_uncond = self.conditioner(batch_obj, force_mask=True)
        pred_crys_uncond, pred_x_uncond, pred_logits_uncond = self.decoder(
            time_emb_base + cond_emb_uncond, atom_types, frac_coords, crys_fam, num_atoms, batch_idx
        )

        cond_emb_cond = self.conditioner(batch_obj, force_mask=False)
        pred_crys_cond, pred_x_cond, pred_logits_cond = self.decoder(
            time_emb_base + cond_emb_cond, atom_types, frac_coords, crys_fam, num_atoms, batch_idx
        )

        pred_crys_fam = pred_crys_uncond + guidance_scale * (pred_crys_cond - pred_crys_uncond)
        pred_x = pred_x_uncond + guidance_scale * (pred_x_cond - pred_x_uncond)
        pred_atom_types_logits = pred_logits_uncond + guidance_scale * (pred_logits_cond - pred_logits_uncond)

        return pred_crys_fam, pred_x, pred_atom_types_logits
