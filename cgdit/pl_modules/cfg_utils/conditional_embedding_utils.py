# @Author : Serein
# @Time : 2025/11/9 01:22
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any


class ScalarEmbedding(nn.Module):
    """
    用于连续数值性质（如形成能、带隙）的嵌入。
    对应 MatterGen 的 scalar 处理部分，但更简洁。
    """

    def __init__(self, time_dim: int, scale: float = 1.0, shift: float = 0.0):
        super().__init__()
        self.scale = scale
        self.shift = shift
        # MLP: [1] -> [time_dim] -> [time_dim]
        self.net = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch] or [Batch, 1]
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # 简单的归一化 (x - mean) / std，参数由外部传入
        x = (x - self.shift) / self.scale
        return self.net(x)


class CategoricalEmbedding(nn.Module):
    """
    用于离散类别性质（如空间群 1-230）的嵌入。
    """

    def __init__(self, time_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(num_classes + 1, time_dim)  # +1 用于处理潜在的 0-indexing 问题或 padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch]
        return self.embedding(x.long())


class ConditioningEncoder(nn.Module):
    """
    条件编码器主类。
    管理多个条件（Scalar 或 Categorical），并处理 Classifier-Free Guidance 的 Dropout。
    """

    def __init__(
            self,
            time_dim: int,
            condition_configs: Dict[str, Dict[str, Any]],
            cond_dropout_prob: float = 0.1
    ):
        """
        Args:
            time_dim: 嵌入维度，需与 Diffusion 的 time_embedding 一致。
            condition_configs: 字典配置，例如：
                {
                    'formation_energy_per_atom': {'type': 'scalar', 'scale': 1.0, 'shift': -1.5},
                    'spacegroup': {'type': 'categorical', 'num_classes': 230}
                }
            cond_dropout_prob: 训练时丢弃条件的概率 (CFG)。
        """
        super().__init__()
        self.time_dim = time_dim
        self.cond_dropout_prob = cond_dropout_prob

        self.embedders = nn.ModuleDict()
        self.null_embeddings = nn.ParameterDict()  # 存储每个属性的“空”嵌入 (Unconditional Token)

        for prop_name, cfg in condition_configs.items():
            if cfg['type'] == 'scalar':
                self.embedders[prop_name] = ScalarEmbedding(
                    time_dim, scale=cfg.get('scale', 1.0), shift=cfg.get('shift', 0.0)
                )
            elif cfg['type'] == 'categorical':
                self.embedders[prop_name] = CategoricalEmbedding(
                    time_dim, num_classes=cfg.get('num_classes', 10)
                )
            else:
                raise ValueError(f"Unknown condition type: {cfg['type']}")

            self.null_embeddings[prop_name] = nn.Parameter(torch.randn(1, time_dim))

    def forward(self, batch, cond_drop_prob: float = None, force_mask: bool = False) -> torch.Tensor:
        """
        Args:
            batch: PyG Batch 对象，包含 batch.formation_energy_per_atom 等属性。
            cond_drop_prob: 覆盖默认的 dropout 概率。
            force_mask: 强制丢弃所有条件（用于推理时的 Unconditional 生成）。
        Returns:
            total_cond_embedding: [Batch, time_dim]
        """
        batch_size = batch.num_graphs
        device = batch.batch.device

        total_emb = torch.zeros(batch_size, self.time_dim, device=device)

        current_prob = self.cond_dropout_prob if cond_drop_prob is None else cond_drop_prob

        for prop_name, embedder in self.embedders.items():
            if not hasattr(batch, prop_name):
                null_emb = self.null_embeddings[prop_name].expand(batch_size, -1)
                total_emb = total_emb + null_emb
                continue

            data = getattr(batch, prop_name)
            cond_emb = embedder(data)  # [B, dim]
            null_emb = self.null_embeddings[prop_name].expand(batch_size, -1)

            if force_mask:
                # 推理时：强制无条件
                mask = torch.zeros(batch_size, 1, device=device)
            elif self.training and current_prob > 0:
                # 训练时：随机 Dropout
                # Bernoulli: 1 = Keep Condition, 0 = Drop Condition
                mask = torch.bernoulli(torch.full((batch_size, 1), 1 - current_prob, device=device))
            else:
                # 推理时：默认保留条件 (Guidance Scale > 0 的部分)
                mask = torch.ones(batch_size, 1, device=device)

            term_emb = mask * cond_emb + (1 - mask) * null_emb

            # 5. 累加到总嵌入中 (MatterGen 是 concat，但通常 sum 效果也很好且不改变维度)
            # 如果希望 concat，需要修改 decoder 的输入维度
            total_emb = total_emb + term_emb

        return total_emb