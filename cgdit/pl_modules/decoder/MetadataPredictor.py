# @Author : Serein
# @Time : 2025/10/30 14:41
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import hydra
from torch.distributions import Normal, Categorical


class MetadataPredictor(nn.Module):
    """
    解码器 (Decoder): 从噪声 z 预测元数据
    完全复用并优化了你提供的逻辑
    """

    def __init__(self, latent_dim=256, max_atoms=100, num_spacegroups=230, max_atomic_num=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_atoms = max_atoms
        self.num_spacegroups = num_spacegroups
        self.max_atomic_num = max_atomic_num

        # 共享特征提取层
        self.shared_mlp = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU()
        )

        # 1. 预测原子数量 (分类问题)
        self.num_atoms_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, max_atoms + 1)
        )

        # 2. 预测空间群 (分类问题)
        self.sg_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, num_spacegroups + 1)
        )

        # 3. 预测化学成分 (每个元素出现的概率)
        self.composition_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, max_atomic_num)
        )

    def forward(self, z):
        features = self.shared_mlp(z)
        num_atoms_logits = self.num_atoms_head(features)
        sg_logits = self.sg_head(features)
        comp_logits = self.composition_head(features)
        return num_atoms_logits, sg_logits, comp_logits


class MetadataEncoder(nn.Module):
    """
    编码器 (Encoder): 将真实元数据映射到潜空间 z
    """

    def __init__(self, latent_dim=256, max_atoms=100, num_spacegroups=230, max_atomic_num=100):
        super().__init__()

        # Embedding layers
        self.num_atoms_emb = nn.Embedding(max_atoms + 1, 64)
        self.sg_emb = nn.Embedding(num_spacegroups + 1, 128)
        # Composition is a vector of floats (proportions), we map it linearly
        self.comp_emb = nn.Linear(max_atomic_num, 128)

        self.encoder_mlp = nn.Sequential(
            nn.Linear(64 + 128 + 128, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU()
        )

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)

    def forward(self, num_atoms, spacegroup, composition):
        # inputs:
        # num_atoms: [B] (long)
        # spacegroup: [B] (long)
        # composition: [B, 100] (float, normalized sum=1)

        h_n = self.num_atoms_emb(num_atoms)
        h_s = self.sg_emb(spacegroup)
        h_c = self.comp_emb(composition)

        h = torch.cat([h_n, h_s, h_c], dim=-1)
        h = self.encoder_mlp(h)

        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var


class MetadataVAE(pl.LightningModule):
    """
    VAE Lightning Module 用于训练
    """

    def __init__(self,
                 latent_dim=256,
                 max_atoms=100,
                 num_spacegroups=230,
                 max_atomic_num=100,
                 kl_weight=0.001,
                 lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = MetadataEncoder(latent_dim, max_atoms, num_spacegroups, max_atomic_num)
        self.decoder = MetadataPredictor(latent_dim, max_atoms, num_spacegroups, max_atomic_num)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, num_atoms, spacegroup, composition):
        mu, log_var = self.encoder(num_atoms, spacegroup, composition)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def training_step(self, batch, batch_idx):
        # 假设 batch 是 PyTorch Geometric 的 Batch 对象或包含以下字段的字典
        # 需要在 DataLoader 里预处理好 composition 向量 (B, 100)

        num_atoms = batch.num_atoms  # [B]
        spacegroup = batch.spacegroup  # [B]
        # composition 需要根据 atom_types 预计算，或者直接存在 batch 里
        if hasattr(batch, 'composition_vector'):
            composition = batch.composition_vector
        else:
            # 简单的 on-the-fly 计算 (如果 batch.atom_types 存在)
            # 这里仅作示例，建议在 Dataset 中预处理
            composition = torch.zeros(batch.num_graphs, self.hparams.max_atomic_num, device=self.device)
            # ... 逻辑省略，假设输入已有 composition ...

        (pred_n_logits, pred_sg_logits, pred_c_logits), mu, log_var = self(num_atoms, spacegroup, composition)

        # --- Losses ---
        # 1. Reconstruction Loss
        loss_n = F.cross_entropy(pred_n_logits, num_atoms)
        loss_sg = F.cross_entropy(pred_sg_logits, spacegroup)

        # Composition Loss (KL divergence between predicted dist and true dist)
        # pred_c_logits -> Softmax -> Dist
        loss_c = F.kl_div(F.log_softmax(pred_c_logits, dim=-1), composition, reduction='batchmean')

        recon_loss = loss_n + loss_sg + loss_c * 10  # 加大成分权值

        # 2. KL Divergence for VAE (Regularization)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kld_loss = kld_loss / batch.num_graphs  # Normalize by batch size

        total_loss = recon_loss + self.hparams.kl_weight * kld_loss

        self.log_dict({
            'train_loss': total_loss,
            'recon_loss': recon_loss,
            'kld_loss': kld_loss,
            'acc_sg': (pred_sg_logits.argmax(1) == spacegroup).float().mean()
        })

        return total_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    @torch.no_grad()
    def sample(self, num_samples, device):
        """
        推理用的采样函数
        """
        # 从标准正态分布采样 z
        z = torch.randn(num_samples, self.hparams.latent_dim, device=device)

        # 解码
        num_atoms_logits, sg_logits, comp_logits = self.decoder(z)

        # 后处理 logits 得到具体数值

        # 1. Num Atoms
        num_atoms_probs = F.softmax(num_atoms_logits, dim=-1)
        # 屏蔽 0 原子 (index 0)
        num_atoms_probs[:, 0] = 0
        pred_num_atoms = torch.multinomial(num_atoms_probs, 1).squeeze(-1)  # [B]

        # 2. Space Group
        sg_probs = F.softmax(sg_logits, dim=-1)
        sg_probs[:, 0] = 0
        pred_sg = torch.multinomial(sg_probs, 1).squeeze(-1)  # [B]

        # 3. Atom Types (Composition)
        # 基于预测的成分概率分布，为每个晶体采样 n 个原子
        comp_probs = F.softmax(comp_logits, dim=-1)  # [B, 100]

        pred_atom_types_batch = []
        pred_batch_idx = []

        for i in range(num_samples):
            n = pred_num_atoms[i].item()
            # 从分布中采样 n 个原子
            # replacement=True 允许重复元素
            if n > 0:
                atoms = torch.multinomial(comp_probs[i], n, replacement=True)
                # atoms 是 0-99，对应 H-Fm，+1 变成 1-100 用于后续处理
                atoms = atoms + 1
                pred_atom_types_batch.append(atoms)
                pred_batch_idx.append(torch.full((n,), i, device=device, dtype=torch.long))
            else:
                # Fallback purely just in case
                pred_atom_types_batch.append(torch.tensor([1], device=device))
                pred_batch_idx.append(torch.tensor([0], device=device))

        return {
            'num_atoms': pred_num_atoms,  # [B]
            'spacegroup': pred_sg,  # [B]
            'atom_types_list': pred_atom_types_batch  # List of tensors
        }