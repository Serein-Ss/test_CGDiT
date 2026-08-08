# @Author : Serein
# @Time : 2026/06/27
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from typing import Dict, Any

from .conditional_embedding_utils import ScalarEmbedding, CategoricalEmbedding


# =============================================================================
# Channel B: Structure Branch
# =============================================================================

class StructureEncoder(nn.Module):
    """
    轻量级晶体结构编码器。
    输入：batch 中的原子类型 + 晶格参数
    输出：图级别结构表示 [B, time_dim]

    不引入额外的消息传递，仅做原子嵌入的 mean-pool + 晶格嵌入融合，
    保持与现有框架的低耦合性。
    """

    def __init__(self, hidden_dim: int, max_atom_types: int = 100):
        super().__init__()
        self.atom_emb = nn.Embedding(max_atom_types + 1, hidden_dim, padding_idx=0)

        self.lattice_mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, batch) -> torch.Tensor:
        # 原子嵌入 [N_atoms, hidden_dim]
        atom_feat = self.atom_emb(batch.atom_types.long().clamp(0, 99))

        # Mean-pool 到图级别 [B, hidden_dim]
        atom_graph = scatter(
            atom_feat, batch.batch, dim=0,
            dim_size=batch.num_graphs, reduce='mean'
        )

        # 晶格嵌入 [B, 6] -> [B, hidden_dim]
        # 优先使用经过归一化的 scaled_lengths，与 DiffusionLoss 保持一致
        if hasattr(batch, 'scaled_lengths'):
            lengths = batch.scaled_lengths
        else:
            lengths = batch.lengths
        lattice_in = torch.cat([lengths, batch.angles], dim=-1)
        if lattice_in.dim() == 3:
            lattice_in = lattice_in.squeeze(1)  # [B, 1, 6] -> [B, 6]

        lattice_feat = self.lattice_mlp(lattice_in)  # [B, hidden_dim]

        # 融合 [B, hidden_dim]
        fused = self.fusion_mlp(torch.cat([atom_graph, lattice_feat], dim=-1))
        return fused


# =============================================================================
# Projection Head & Contrastive Loss
# =============================================================================

class ProjectionHead(nn.Module):
    """投影头：将各通道嵌入映射到对比学习的公共空间。"""

    def __init__(self, in_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.SiLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InfoNCELoss(nn.Module):
    """
    双向 InfoNCE 对比损失。

    Positive pair: (z_struct_i, z_prop_i) —— 同一晶体的结构与性质表示
    Negative pair: (z_struct_i, z_prop_j), i ≠ j —— Batch 内跨晶体对

    Loss = (L(struct->prop) + L(prop->struct)) / 2
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_struct: torch.Tensor, z_prop: torch.Tensor) -> torch.Tensor:
        B = z_struct.size(0)
        if B < 2:
            # batch 过小时跳过（避免对比学习退化）
            return torch.tensor(0., device=z_struct.device)

        z_struct = F.normalize(z_struct, dim=-1)  # [B, proj_dim]
        z_prop   = F.normalize(z_prop,   dim=-1)  # [B, proj_dim]

        sim = torch.matmul(z_struct, z_prop.T) / self.temperature  # [B, B]
        labels = torch.arange(B, device=z_struct.device)

        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss


# =============================================================================
# Curriculum Scheduler
# =============================================================================

class CurriculumScheduler:
    """
    三阶段 Curriculum Learning 调度器。

    Phase 1 - Warmup  [0, warmup_epochs):
        只训练扩散主干，对比权重=0，dropout 保持较高值（模型先学好无条件生成）

    Phase 2 - Ramp    [warmup_epochs, warmup_epochs + ramp_epochs):
        对比权重线性增加 0 → max_contrastive_weight
        条件 dropout 线性降低 warmup_dropout → base_dropout

    Phase 3 - Stable  [warmup_epochs + ramp_epochs, ∞):
        对比权重 = max_contrastive_weight
        条件 dropout = base_dropout
    """

    def __init__(
        self,
        warmup_epochs: int = 100,
        ramp_epochs: int = 200,
        max_contrastive_weight: float = 0.1,
        base_dropout: float = 0.1,
        warmup_dropout: float = 0.3,
    ):
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.max_contrastive_weight = max_contrastive_weight
        self.base_dropout = base_dropout
        self.warmup_dropout = warmup_dropout

    def _progress(self, epoch: int) -> float:
        """返回 ramp 阶段的进度 [0, 1]，warmup 时为 0，stable 时为 1。"""
        if epoch < self.warmup_epochs:
            return 0.0
        elif epoch < self.warmup_epochs + self.ramp_epochs:
            return (epoch - self.warmup_epochs) / self.ramp_epochs
        return 1.0

    def get_contrastive_weight(self, epoch: int) -> float:
        return self.max_contrastive_weight * self._progress(epoch)

    def get_cond_dropout(self, epoch: int) -> float:
        p = self._progress(epoch)
        return self.warmup_dropout - (self.warmup_dropout - self.base_dropout) * p


# =============================================================================
# Dual-Channel Multimodal Conditioning Encoder
# =============================================================================

class MultimodalConditioningEncoder(nn.Module):
    """
    双通道多模态 Adapter（对比 ConditioningEncoder 的改进版本）。

    Channel A (Property Branch):
        数值/类别性质 → Scalar/CategoricalEmbedding → 性质嵌入
        保持与原版 ConditioningEncoder 完全相同的逻辑（CFG dropout、null embedding）

    Channel B (Structure Branch):
        晶体图 → StructureEncoder → 结构嵌入
        不参与 CFG dropout（结构信息始终可见）

    Fusion:
        可学习门控：gate = σ(W[prop_emb; struct_emb])
        fused = gate * prop_emb + (1 - gate) * struct_emb

    Contrastive Alignment:
        训练时在投影空间计算 InfoNCE，对齐结构与性质表示
        对比损失通过 self._contrastive_loss 暴露给外部的 training_step
    """

    def __init__(
        self,
        time_dim: int,
        condition_configs: Dict[str, Dict[str, Any]],
        cond_dropout_prob: float = 0.1,
        proj_dim: int = 128,
        contrastive_temperature: float = 0.07,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.cond_dropout_prob = cond_dropout_prob
        self.proj_dim = proj_dim

        # ── Channel A: Property Branch ────────────────────────────────────
        self.embedders = nn.ModuleDict()
        self.null_embeddings = nn.ParameterDict()

        for prop_name, cfg in condition_configs.items():
            if cfg['type'] == 'scalar':
                self.embedders[prop_name] = ScalarEmbedding(
                    time_dim,
                    scale=cfg.get('scale', 1.0),
                    shift=cfg.get('shift', 0.0)
                )
            elif cfg['type'] == 'categorical':
                self.embedders[prop_name] = CategoricalEmbedding(
                    time_dim, num_classes=cfg.get('num_classes', 10)
                )
            else:
                raise ValueError(f"Unknown condition type: {cfg['type']}")
            self.null_embeddings[prop_name] = nn.Parameter(torch.randn(1, time_dim))

        # ── Channel B: Structure Branch ───────────────────────────────────
        self.structure_encoder = StructureEncoder(hidden_dim=time_dim)

        # ── Projection Heads（用于对比学习）──────────────────────────────
        self.prop_proj   = ProjectionHead(time_dim, proj_dim)
        self.struct_proj = ProjectionHead(time_dim, proj_dim)

        # ── InfoNCE Loss ──────────────────────────────────────────────────
        self.contrastive_loss_fn = InfoNCELoss(temperature=contrastive_temperature)

        # ── Gated Fusion ──────────────────────────────────────────────────
        self.gate = nn.Sequential(
            nn.Linear(time_dim * 2, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, 1),
            nn.Sigmoid()
        )

        # 训练时对比损失暂存，供 training_step 取用
        self._contrastive_loss: torch.Tensor = torch.tensor(0.)

    def forward(
        self,
        batch,
        cond_drop_prob: float = None,
        force_mask: bool = False,
    ) -> torch.Tensor:
        """
        与原版 ConditioningEncoder.forward 接口完全兼容。

        Returns:
            fused_emb: [B, time_dim]，直接加到 time_emb 上。
            同时将对比损失写入 self._contrastive_loss（训练模式下）。
        """
        batch_size = batch.num_graphs
        device = batch.batch.device
        current_prob = self.cond_dropout_prob if cond_drop_prob is None else cond_drop_prob

        # ── Channel A: Property Branch ────────────────────────────────────
        prop_total_emb = torch.zeros(batch_size, self.time_dim, device=device)
        raw_prop_embs = []  # 原始性质嵌入（不经 dropout），用于对比学习

        for prop_name, embedder in self.embedders.items():
            if not hasattr(batch, prop_name):
                null_emb = self.null_embeddings[prop_name].expand(batch_size, -1)
                prop_total_emb = prop_total_emb + null_emb
                continue

            data = getattr(batch, prop_name)
            cond_emb = embedder(data)   # [B, time_dim]
            null_emb = self.null_embeddings[prop_name].expand(batch_size, -1)

            if force_mask:
                mask = torch.zeros(batch_size, 1, device=device)
            elif self.training and current_prob > 0:
                mask = torch.bernoulli(
                    torch.full((batch_size, 1), 1 - current_prob, device=device)
                )
            else:
                mask = torch.ones(batch_size, 1, device=device)

            prop_total_emb = prop_total_emb + mask * cond_emb + (1 - mask) * null_emb
            raw_prop_embs.append(cond_emb)  # 保留原始嵌入用于对比

        # ── Channel B: Structure Branch ───────────────────────────────────
        struct_emb = self.structure_encoder(batch)  # [B, time_dim]

        # ── Gated Fusion ──────────────────────────────────────────────────
        gate_val  = self.gate(torch.cat([prop_total_emb, struct_emb], dim=-1))  # [B, 1]
        fused_emb = gate_val * prop_total_emb + (1 - gate_val) * struct_emb    # [B, time_dim]

        # ── Contrastive Loss（仅训练时计算，不经 dropout 的原始嵌入）────────
        if self.training and len(raw_prop_embs) > 0:
            prop_repr = torch.stack(raw_prop_embs, dim=0).mean(dim=0)  # [B, time_dim]
            z_prop   = self.prop_proj(prop_repr)    # [B, proj_dim]
            z_struct = self.struct_proj(struct_emb) # [B, proj_dim]
            self._contrastive_loss = self.contrastive_loss_fn(z_struct, z_prop)
        else:
            self._contrastive_loss = torch.tensor(0., device=device)

        return fused_emb
