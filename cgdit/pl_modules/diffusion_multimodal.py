# @Author : Serein
# @Time : 2026/06/27
from typing import Any

import torch

from cgdit.pl_modules.diffusion import Diffusion
from cgdit.pl_modules.cfg_utils.multimodal_adapter import (
    MultimodalConditioningEncoder,
    CurriculumScheduler,
)


class DiffusionMultimodal(Diffusion):
    """
    多模态双通道扩散模型（对比 Diffusion 的改进版本）。

    改动点：
      1. 将 ConditioningEncoder 替换为 MultimodalConditioningEncoder
         （双通道：性质分支 + 结构分支，门控融合）
      2. training_step 中加入 InfoNCE 对比损失，权重由 CurriculumScheduler 动态调整
      3. 条件 dropout 概率随训练进程动态变化（Curriculum Learning）
      4. 新增日志指标：contrastive_loss / contrastive_weight / curriculum_dropout

    继承自 Diffusion，推理接口（sample / _get_model_output）完全不变。

    新增 hparams（在 model yaml 的对应字段下配置）：
      proj_dim:                 int   = 128    # 投影空间维度
      contrastive_temperature:  float = 0.07   # InfoNCE 温度
      curriculum:
        warmup_epochs:          int   = 100    # 对比损失 warmup 轮数
        ramp_epochs:            int   = 200    # 线性增加阶段长度
        max_contrastive_weight: float = 0.1    # 对比损失最大权重
        warmup_dropout:         float = 0.3    # warmup 阶段的 cond_dropout
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # 读取 multimodal 相关配置
        condition_configs = self.hparams.get('conditions', {})
        cond_dropout_prob = self.hparams.get('cond_dropout_prob', 0.1)
        proj_dim          = self.hparams.get('proj_dim', 128)
        contrast_temp     = self.hparams.get('contrastive_temperature', 0.07)

        # 替换父类的 ConditioningEncoder
        self.conditioner = MultimodalConditioningEncoder(
            time_dim=self.time_dim,
            condition_configs=condition_configs,
            cond_dropout_prob=cond_dropout_prob,
            proj_dim=proj_dim,
            contrastive_temperature=contrast_temp,
        )

        # Curriculum 调度器
        cur = self.hparams.get('curriculum', {})
        self.curriculum = CurriculumScheduler(
            warmup_epochs=cur.get('warmup_epochs', 100),
            ramp_epochs=cur.get('ramp_epochs', 200),
            max_contrastive_weight=cur.get('max_contrastive_weight', 0.1),
            base_dropout=cond_dropout_prob,
            warmup_dropout=cur.get('warmup_dropout', 0.3),
        )

    # -------------------------------------------------------------------------
    # training_step：加入对比损失 + Curriculum 调度
    # -------------------------------------------------------------------------

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # 1. 根据当前 epoch 更新 dropout 概率（Curriculum Phase 1/2/3）
        epoch = self.current_epoch
        current_dropout    = self.curriculum.get_cond_dropout(epoch)
        contrastive_weight = self.curriculum.get_contrastive_weight(epoch)
        self.conditioner.cond_dropout_prob = current_dropout

        # 2. 前向传播（与父类相同，conditioner 内部顺带计算对比损失）
        output_dict = self(batch, batch_idx)

        loss_lattice    = output_dict['loss_lattice']
        loss_coord      = output_dict['loss_coord']
        loss_atom_types = output_dict['loss_atom_types']
        diffusion_loss  = output_dict['loss']

        # 3. 取对比损失
        contrastive_loss = self.conditioner._contrastive_loss

        # 4. 总损失 = 扩散损失 + curriculum权重 * 对比损失
        total_loss = diffusion_loss + contrastive_weight * contrastive_loss

        # 5. 日志
        self.log_dict(
            {
                'train_loss':          total_loss,
                'diffusion_loss':      diffusion_loss,
                'contrastive_loss':    contrastive_loss,
                'contrastive_weight':  contrastive_weight,
                'curriculum_dropout':  current_dropout,
                'lattice_loss':        loss_lattice,
                'coord_loss':          loss_coord,
                'atom_loss':           loss_atom_types,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if total_loss.isnan() or total_loss.isinf():
            print(f"[Warning] NaN/Inf loss at batch_idx={batch_idx}, skipping.")
            return None

        return total_loss

    # -------------------------------------------------------------------------
    # on_after_backward：仅监控 decoder 梯度（与父类相同）
    # -------------------------------------------------------------------------

    def on_after_backward(self):
        total_norm = 0.
        for _, p in self.decoder.named_parameters():
            try:
                total_norm += p.grad.data.norm(2).item() ** 2
            except Exception:
                pass
        total_norm = total_norm ** 0.5

        self.log_dict(
            {'grad_norm': total_norm},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
