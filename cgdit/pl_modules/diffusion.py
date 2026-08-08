import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any

import hydra
import pytorch_lightning as pl
from torch_scatter import scatter
from tqdm import tqdm

from cgdit.common.data_utils import (
    lattice_params_to_matrix_torch)

from cgdit.pl_modules.lattice.crystal_family import CrystalFamily
from cgdit.pl_modules.diff_utils.discrete_diff_utils import (
    MaskDiffusion, create_discrete_diffusion_schedule, q_sample
)

from cgdit.pl_modules.cfg_utils import ConditioningEncoder
from cgdit.pl_modules.training_utils import DiffusionLoss


MAX_ATOMIC_NUM = 100


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
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


### Model definition

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Diffusion(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.decoder = hydra.utils.instantiate(self.hparams.decoder,
                                               latent_dim=self.hparams.time_dim,
                                               _recursive_=False)
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.crystal_family = CrystalFamily()

        # D3PM init
        self.num_atom_types = MAX_ATOMIC_NUM
        self.mask_token_id = MAX_ATOMIC_NUM
        d3pm_hparams = self.hparams.d3pm_scheduler
        self.d3pm_schedule = create_discrete_diffusion_schedule(
            kind=d3pm_hparams.kind,
            beta_min=d3pm_hparams.beta_min,
            beta_max=d3pm_hparams.beta_max,
            num_steps=self.beta_scheduler.timesteps  # 保持与连续扩散步数一致
        )
        self.d3pm = MaskDiffusion(
            dim=self.num_atom_types + 1,  # 0-99 是元素, 100 是 [MASK]
            schedule=self.d3pm_schedule
        )
        self.q_sample = q_sample

        # --- Classifier-Free Guidance Init ---
        condition_configs = self.hparams.get('conditions', {})
        cond_dropout_prob = self.hparams.get('cond_dropout_prob', 0.1)

        self.conditioner = ConditioningEncoder(
            time_dim=self.time_dim,
            condition_configs=condition_configs,
            cond_dropout_prob=cond_dropout_prob
        )

        # --- Loss Function Init ---
        self.loss_fn = DiffusionLoss(
            d3pm_diffusion=self.d3pm,
            num_atom_types=self.num_atom_types,
            cost_lattice=self.hparams.cost_lattice,
            cost_coord=self.hparams.cost_coord,
            cost_atom=self.hparams.cost_atom,
            hybrid_lambda=self.hparams.get('hybrid_lambda', 0.01)
        )

        # breakpoint()

    def forward(self, batch, batch_idx=None):
        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        cond_emb = self.conditioner(batch)
        time_emb = time_emb + cond_emb

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta = self.beta_scheduler.betas[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        # --- Lattice 加噪 ---
        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        lattices = self.crystal_family.de_so3(lattices)

        ori_crys_fam = self.crystal_family.m2v(lattices)
        ori_crys_fam = self.crystal_family.proj_k_to_spacegroup(ori_crys_fam, batch.spacegroup)

        rand_crys_fam = torch.randn_like(ori_crys_fam)
        rand_crys_fam = self.crystal_family.proj_k_to_spacegroup(rand_crys_fam, batch.spacegroup)

        input_crys_fam = c0[:, None] * ori_crys_fam + c1[:, None] * rand_crys_fam
        input_crys_fam = self.crystal_family.proj_k_to_spacegroup(input_crys_fam, batch.spacegroup)

        # --- Coordinate 加噪 ---
        frac_coords = batch.frac_coords
        rand_x = torch.randn_like(frac_coords)

        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]

        rand_x_anchor = rand_x[batch.anchor_index]
        rand_x_anchor = (batch.ops_inv[batch.anchor_index] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
        rand_x = (batch.ops[:, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.

        # --- Atom Type 加噪 (离散) ---
        x_start_atoms = batch.atom_types.long() - 1 # 0-99 是真实的原子的数据，100 是 MASK 标记
        # print(x_start_atoms)
        x_start_atoms = torch.clamp(x_start_atoms, min=0, max=self.num_atom_types - 1)

        times_per_atom = times.repeat_interleave(batch.num_atoms)
        t_discrete = times_per_atom.long() + 1
        t_discrete = torch.clamp(t_discrete, max=self.beta_scheduler.timesteps - 1)

        input_atom_types = self.q_sample(
            x_start=x_start_atoms,
            t=t_discrete,
            diffusion=self.d3pm,
            return_logits=False
        )

        # --- Model Prediction ---
        # pred_atom_types_logits 应该是 [N, num_atom_types + 1] 的形状
        pred_crys_fam, pred_x, pred_atom_types_logits = self.decoder(
            time_emb,
            input_atom_types, # batch.atom_types
            input_frac_coords,
            input_crys_fam,
            batch.num_atoms,
            batch.batch
        )
        pred_crys_fam = self.crystal_family.proj_k_to_spacegroup(pred_crys_fam, batch.spacegroup)

        # --- Loss Calculation ---
        output_dict = self.loss_fn(
            pred_crys_fam=pred_crys_fam,
            rand_crys_fam=rand_crys_fam, # Target for lattice
            pred_x=pred_x,
            batch=batch,
            rand_x_anchor=rand_x_anchor, # Raw noise for coord target calculation
            sigmas_per_atom=sigmas_per_atom,
            sigmas_norm_per_atom=sigmas_norm_per_atom,
            pred_atom_logits=pred_atom_types_logits,
            x_start_atoms=x_start_atoms,
            input_atom_types=input_atom_types,
            t_discrete=t_discrete
        )

        # breakpoint()
        return output_dict

    @torch.no_grad()
    def sample(self, batch, diff_ratio=1.0, step_lr=1e-5, guidance_scale=1.0, fixed_atom_types=None):
        batch_size = batch.num_graphs

        if fixed_atom_types is not None:
            fixed_atom_types = fixed_atom_types.long().to(self.device)
            fixed_atom_types = torch.clamp(fixed_atom_types, min=0, max=self.num_atom_types - 1)

        x_T = torch.rand([batch.num_nodes, 3]).to(self.device)
        crys_fam_T = torch.randn([batch_size, 6]).to(self.device)
        crys_fam_T = self.crystal_family.proj_k_to_spacegroup(crys_fam_T, batch.spacegroup)

        if fixed_atom_types is not None:
            # 如果 diff_ratio < 1，会在后面被覆盖，这里先按 T 初始化
            t_max_tensor = torch.full(fixed_atom_types.shape, self.beta_scheduler.timesteps - 1,
                                      device=self.device, dtype=torch.long)
            atom_types_T, _ = self.q_sample(x_start=fixed_atom_types, t=t_max_tensor, diffusion=self.d3pm)
        else:
            # 否则，从全 MASK 分布开始
            atom_types_T = self.d3pm.sample_stationary(batch.atom_types.shape).to(self.device)

        if diff_ratio < 1:
            time_start = int(self.beta_scheduler.timesteps * diff_ratio)
            lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
            lattices = self.crystal_family.de_so3(lattices)
            ori_crys_fam = self.crystal_family.m2v(lattices)
            ori_crys_fam = self.crystal_family.proj_k_to_spacegroup(ori_crys_fam, batch.spacegroup)

            frac_coords = batch.frac_coords

            rand_crys_fam, rand_x = torch.randn_like(ori_crys_fam), torch.randn_like(frac_coords)

            alphas_cumprod = self.beta_scheduler.alphas_cumprod[time_start]
            beta = self.beta_scheduler.betas[time_start]

            c0 = torch.sqrt(alphas_cumprod)
            c1 = torch.sqrt(1. - alphas_cumprod)

            sigmas = self.sigma_scheduler.sigmas[time_start]

            rand_x_anchor = rand_x[batch.anchor_index]
            rand_x_anchor = (batch.ops[batch.anchor_index, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
            rand_x = (batch.ops[:, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)

            crys_fam_T = c0 * ori_crys_fam + c1 * rand_crys_fam
            x_T = (frac_coords + sigmas * rand_x) % 1.

            # 增加对离散数据的加噪
            t_discrete_start = torch.full(batch.atom_types.shape, time_start,
                                          device=self.device, dtype=torch.long)

            if fixed_atom_types is not None:
                # 使用传入的 fixed_atom_types 加噪到 time_start
                atom_types_T, _ = self.q_sample(x_start=fixed_atom_types, t=t_discrete_start,
                                                diffusion=self.d3pm)
            else:
                # 使用 batch 中原有的 atom_types 加噪 (常规流程)
                x_start_atoms = batch.atom_types.long()
                atom_types_T, _ = self.q_sample(x_start=x_start_atoms, t=t_discrete_start,
                                                diffusion=self.d3pm)

        else:
            time_start = self.beta_scheduler.timesteps - 1

        l_T = self.crystal_family.v2m(crys_fam_T)

        x_T_all = torch.cat([x_T[batch.anchor_index], torch.ones(batch.ops.size(0), 1).to(x_T.device)],
                            dim=-1).unsqueeze(-1)  # N * 4 * 1

        x_T = (batch.ops @ x_T_all).squeeze(-1)[:, :3] % 1.  # N * 3

        traj = {time_start: {
            'num_atoms': batch.num_atoms,
            'atom_types': atom_types_T, # batch.atom_types,
            'frac_coords': x_T % 1.,
            'lattices': l_T,
            'crys_fam': crys_fam_T
        }}

        for t in tqdm(range(time_start, 0, -1)):
            times = torch.full((batch_size,), t, device=self.device)

            time_emb = self.time_embedding(times)

            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            alphas_cumprod_next = self.beta_scheduler.alphas_cumprod[t - 1]

            alphas_cumprod_next = torch.sqrt(alphas_cumprod_next)

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']
            crys_fam_t = traj[t]['crys_fam']
            atom_types_t = traj[t]['atom_types']

            # Corrector

            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr / (sigma_norm * (self.sigma_scheduler.sigma_begin) ** 2)
            std_x = torch.sqrt(2 * step_size)

            rand_x_anchor = rand_x[batch.anchor_index]
            rand_x_anchor = (batch.ops_inv[batch.anchor_index] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
            rand_x = (batch.ops[:, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)

            # pred_crys_fam, pred_x, pred_atom_types_logits = self.decoder(
            #     time_emb,
            #     atom_types_t,
            #     x_t,
            #     crys_fam_t,
            #     batch.num_atoms,
            #     batch.batch
            # )

            # Corrector 使用 _get_model_output
            pred_crys_fam, pred_x, pred_atom_types_logits = self._get_model_output(
                time_emb,
                atom_types_t,
                x_t,
                crys_fam_t,
                batch.num_atoms,
                batch.batch,
                batch_obj=batch, # 传入 batch 对象以提取条件
                guidance_scale=guidance_scale
            )

            pred_x = pred_x * torch.sqrt(sigma_norm)

            pred_x_proj = torch.einsum('bij, bj-> bi', batch.ops_inv, pred_x)
            pred_x_anchor = scatter(pred_x_proj, batch.anchor_index, dim=0, reduce='mean')[batch.anchor_index]

            pred_x = (batch.ops[:, :3, :3] @ pred_x_anchor.unsqueeze(-1)).squeeze(-1)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x

            crys_fam_t_minus_05 = crys_fam_t

            atom_types_t_minus_05 = atom_types_t

            frac_coords_all = torch.cat(
                [x_t_minus_05[batch.anchor_index], torch.ones(batch.ops.size(0), 1).to(x_t_minus_05.device)],
                dim=-1).unsqueeze(-1)  # N * 4 * 1

            x_t_minus_05 = (batch.ops @ frac_coords_all).squeeze(-1)[:, :3] % 1.  # N * 3

            # Predictor

            rand_crys_fam = torch.randn_like(crys_fam_T)
            rand_crys_fam = self.crystal_family.proj_k_to_spacegroup(rand_crys_fam, batch.spacegroup)
            ori_crys_fam = crys_fam_t
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))

            rand_x_anchor = rand_x[batch.anchor_index]
            rand_x_anchor = (batch.ops_inv[batch.anchor_index] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
            rand_x = (batch.ops[:, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)

            # pred_crys_fam, pred_x, pred_atom_types_logits = self.decoder(
            #     time_emb,
            #     atom_types_t_minus_05, # atom_types_t
            #     x_t_minus_05,
            #     crys_fam_t,
            #     batch.num_atoms,
            #     batch.batch)

            # Predictor 使用 _get_model_output
            pred_crys_fam, pred_x, pred_atom_types_logits = self._get_model_output(
                time_emb,
                atom_types_t_minus_05, # atom_types_t
                x_t_minus_05,
                crys_fam_t,
                batch.num_atoms,
                batch.batch,
                batch_obj=batch,
                guidance_scale=guidance_scale
            )

            pred_x = pred_x * torch.sqrt(sigma_norm)

            crys_fam_t_minus_1 = c0 * (ori_crys_fam - c1 * pred_crys_fam) + sigmas * rand_crys_fam
            crys_fam_t_minus_1 = self.crystal_family.proj_k_to_spacegroup(crys_fam_t_minus_1, batch.spacegroup)

            pred_x_proj = torch.einsum('bij, bj-> bi', batch.ops_inv, pred_x)
            pred_x_anchor = scatter(pred_x_proj, batch.anchor_index, dim=0, reduce='mean')[batch.anchor_index]
            pred_x = (batch.ops[:, :3, :3] @ pred_x_anchor.unsqueeze(-1)).squeeze(-1)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x

            l_t_minus_1 = self.crystal_family.v2m(crys_fam_t_minus_1)

            frac_coords_all = torch.cat(
                [x_t_minus_1[batch.anchor_index], torch.ones(batch.ops.size(0), 1).to(x_t_minus_1.device)],
                dim=-1).unsqueeze(-1)  # N * 4 * 1

            x_t_minus_1 = (batch.ops @ frac_coords_all).squeeze(-1)[:, :3] % 1.  # N * 3

            # 更新离散变量，只关心真实的元素类型 (0-99)，忽略 [MASK] (100)
            if fixed_atom_types is not None:
                # === 模式 A: 固定原子类型 ===
                target_t = t - 1
                if target_t == 0:
                    atom_types_t_minus_1 = fixed_atom_types
                else:
                    # 中间步骤：从 Clean 数据前向加噪到 t-1
                    t_next_tensor = torch.full(fixed_atom_types.shape, target_t,
                                               device=self.device, dtype=torch.long)
                    atom_types_t_minus_1, _ = self.q_sample(
                        x_start=fixed_atom_types,
                        t=t_next_tensor,
                        diffusion=self.d3pm
                    )
            else:
                # === 模式 B: 全生成 ===
                # 使用模型预测 logits 进行后验采样
                pred_x0_atom_logits = pred_atom_types_logits[:, :self.num_atom_types]
                t_discrete_tensor = torch.full(atom_types_t.shape, t, device=self.device, dtype=torch.long)
                atom_types_t_long = atom_types_t_minus_05.long()

                full_logits = torch.full((atom_types_t.shape[0], self.num_atom_types + 1), -1e9, device=self.device)
                full_logits[:, :self.num_atom_types] = pred_x0_atom_logits
                pred_x0_atom_probs = full_logits.softmax(dim=-1)

                p_t_minus_1_logits, _ = self.d3pm.sample_and_compute_posterior_q(
                    x_0=pred_x0_atom_probs, t=t_discrete_tensor - 1, samples=atom_types_t_long,
                    make_one_hot=False, return_logits=True, step_size=1
                )

                if t > 1:
                    atom_types_t_minus_1 = torch.distributions.Categorical(logits=p_t_minus_1_logits).sample()
                else:
                    atom_types_t_minus_1 = torch.argmax(p_t_minus_1_logits, dim=-1)

            traj[t - 1] = {
                'num_atoms': batch.num_atoms,
                'atom_types': atom_types_t_minus_1, # batch.atom_types,
                'frac_coords': x_t_minus_1 % 1.,
                'lattices': l_t_minus_1,
                'crys_fam': crys_fam_t_minus_1
            }

        traj_stack = {
            'num_atoms': batch.num_atoms,
            'all_atom_types': torch.stack([traj[i]['atom_types'] for i in range(time_start, -1, -1)]),
            'all_frac_coords': torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices': torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        return traj[0], traj_stack

    def _get_model_output(self,
                          time_emb_base,
                          atom_types, frac_coords, crys_fam,
                          num_atoms, batch_idx, batch_obj, guidance_scale=1.0
                          ):
        """
        辅助函数：处理 Classifier-Free Guidance 的一次模型前向传播
        """
        # 1. 正常/有条件推理 (guidance_scale=1.0)
        if guidance_scale == 1.0:
            # force_mask=False 表示使用真实条件（如果有），或者在 dropout 模式下由 Conditioner 内部决定
            cond_emb = self.conditioner(batch_obj, force_mask=False)
            final_time_emb = time_emb_base + cond_emb
            return self.decoder(final_time_emb, atom_types, frac_coords, crys_fam, num_atoms, batch_idx)

        # 2. CFG 推理 (guidance_scale != 1.0)
        # A. 无条件分支 (Unconditional) -> force_mask=True，强制将所有条件替换为 Null Embedding
        cond_emb_uncond = self.conditioner(batch_obj, force_mask=True)
        time_emb_uncond = time_emb_base + cond_emb_uncond

        out_uncond = self.decoder(
            time_emb_uncond, atom_types, frac_coords, crys_fam, num_atoms, batch_idx
        )
        pred_crys_uncond, pred_x_uncond, pred_logits_uncond = out_uncond

        # B. 有条件分支 (Conditional) -> force_mask=False
        cond_emb_cond = self.conditioner(batch_obj, force_mask=False)
        time_emb_cond = time_emb_base + cond_emb_cond

        out_cond = self.decoder(
            time_emb_cond, atom_types, frac_coords, crys_fam, num_atoms, batch_idx
        )
        pred_crys_cond, pred_x_cond, pred_logits_cond = out_cond

        # C. 混合 (pred = uncond + s * (cond - uncond))
        # 这里的 output 包含三个部分：晶格参数、坐标、原子类型 Logits
        pred_crys_fam = pred_crys_uncond + guidance_scale * (pred_crys_cond - pred_crys_uncond)
        pred_x = pred_x_uncond + guidance_scale * (pred_x_cond - pred_x_uncond)
        pred_atom_types_logits = pred_logits_uncond + guidance_scale * (pred_logits_cond - pred_logits_uncond)

        return pred_crys_fam, pred_x, pred_atom_types_logits

    def on_after_backward(self):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here

        total_norm = 0.
        for nm, p in self.decoder.named_parameters():
            try:
                param_norm = p.grad.data.norm(2)
                total_norm = total_norm + param_norm.item() ** 2
            except:
                pass
        total_norm = total_norm ** (1. / 2)

        self.log_dict({
            'grad_norm': total_norm
        },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch, batch_idx)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_atom_types = output_dict['loss_atom_types']
        loss = output_dict['loss']

        self.log_dict(
            {'train_loss': loss,
             'lattice_loss': loss_lattice,
             'coord_loss': loss_coord,
             'atom_loss': loss_atom_types},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if loss.isnan() or loss.isinf():
            print(batch_idx)
            return None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, output_dict, prefix):

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_atom_types = output_dict['loss_atom_types']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': loss_lattice,
            f'{prefix}_coord_loss': loss_coord,
            f'{prefix}_atom_loss': loss_atom_types,
        }

        return log_dict, loss

