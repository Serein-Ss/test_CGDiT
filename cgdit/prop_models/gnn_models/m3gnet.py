import torch
import torch.nn as nn
from torch_scatter import scatter
from typing import Any, List, Optional, Tuple, Literal

from cgdit.common.data_utils import get_pbc_distances, create_line_graph_index, compute_theta_and_phi
from cgdit.pl_modules.diffusion import BaseModule
from cgdit.prop_models.gnn_models.utils import polynomial_cutoff
from cgdit.prop_models.gnn_models._layers import (
    ActivationFunction, BondExpansion, EmbeddingBlock, SphericalBesselWithHarmonics,
    MLP, GatedMLP, ThreeBodyInteractions, WeightedReadOut, M3GNetBlock, Set2SetReadOut,
    WeightedAtomReadOut, ReduceReadOut,
)


class M3GNet(nn.Module):
    """The main M3GNet model."""

    def __init__(
            self,
            num_elements: int = 100,
            dim_node_embedding: int = 64,
            dim_edge_embedding: int = 64,
            dim_state_embedding: int = 0,
            ntypes_state: int | None = None,
            dim_state_feats: int | None = None,
            max_n: int = 3,
            max_l: int = 3,
            nblocks: int = 3,
            rbf_type: Literal["Gaussian", "SphericalBessel"] = "SphericalBessel",
            is_intensive: bool = True,
            readout_type: Literal["set2set", "weighted_atom", "reduce_atom"] = "weighted_atom",
            task_type: Literal["classification", "regression"] = "regression",
            cutoff: float = 5.0,
            threebody_cutoff: float = 4.0,
            units: int = 64,
            ntargets: int = 1,
            use_smooth: bool = False,
            use_phi: bool = False,
            niters_set2set: int = 3,
            nlayers_set2set: int = 3,
            field: Literal["node_feat", "edge_feat"] = "node_feat",
            include_state: bool = False,
            activation_type: Literal["swish", "tanh", "sigmoid", "softplus2", "softexp"] = "swish",
            dropout: float | None = None,
            **kwargs,
    ):
        """
        Initializes the M3GNet model.
        :param num_elements:
        :param dim_node_embedding:
        :param dim_edge_embedding:
        :param dim_state_embedding:
        :param max_n:
        :param max_l:
        :param nblocks:
        :param rbf_type:
        :param is_intensive:
        :param readout_type:
        :param task_type:
        :param use_smooth:
        :param use_phi:
        :param niters_set2set:
        :param nlayers_set2set:
        :param cutoff:
        :param ntargets:
        :param threebody_cutoff:
        :param units:
        :param activation_type:
        :param ntypes_state:
        :param dim_state_feats:
        :param include_state:
        :param field:
        :param dropout:
        :param kwargs:
        """
        super().__init__()

        try:
            activation: nn.Module = ActivationFunction[activation_type].value()
        except KeyError:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from None

        self.num_elements = num_elements or 100

        self.bond_expansion = BondExpansion(max_l, max_n, cutoff, rbf_type=rbf_type, smooth=use_smooth)

        degree = max_n * max_l * max_l if use_phi else max_n * max_l
        degree_rbf = max_n if use_smooth else max_n * max_l

        self.embedding = EmbeddingBlock(
            degree_rbf=degree_rbf,
            dim_node_embedding=dim_node_embedding,
            dim_edge_embedding=dim_edge_embedding,
            ntypes_node=num_elements,
            ntypes_state=ntypes_state,
            dim_state_feats=dim_state_feats,
            include_state=include_state,
            dim_state_embedding=dim_state_embedding,
            activation=activation,
        )

        self.basis_expansion = SphericalBesselWithHarmonics(
            max_n=max_n,
            max_l=max_l,
            cutoff=cutoff,
            use_phi=use_phi,
            use_smooth=use_smooth,
        )

        self.three_body_interactions = nn.ModuleList(
            {
                ThreeBodyInteractions(
                    update_network_atom=MLP(
                        dims=[dim_node_embedding, degree],
                        activation=nn.Sigmoid(),
                        activate_last=True,
                    ),
                    update_network_bond=GatedMLP(in_feats=degree, dims=[dim_edge_embedding], use_bias=False),
                )
                for _ in range(nblocks)
            }
        )

        dim_state_feats = dim_state_embedding

        self.graph_layers = nn.ModuleList(
            {
                M3GNetBlock(
                    degree=degree_rbf,
                    activation=activation,
                    conv_hiddens=[units, units],
                    dim_node_feats=dim_node_embedding,
                    dim_edge_feats=dim_edge_embedding,
                    dim_state_feats=dim_state_feats,
                    include_state=include_state,
                    dropout=dropout,
                )
                for _ in range(nblocks)
            }
        )

        if is_intensive:
            input_feats = dim_node_embedding if field == "node_feat" else dim_edge_embedding
            if readout_type == "set2set":
                self.readout = Set2SetReadOut(
                    in_feats=input_feats,
                    n_iters=niters_set2set,
                    n_layers=nlayers_set2set,
                    field=field,
                )
                readout_feats = 2 * input_feats + dim_state_feats if include_state else 2 * input_feats  # type: ignore
            elif readout_type == "weighted_atom":
                self.readout = WeightedAtomReadOut(in_feats=input_feats, dims=[units, units], activation=activation)  # type: ignore[assignment]
                readout_feats = units + dim_state_feats if include_state else units  # type: ignore
            else:
                self.readout = ReduceReadOut("mean", field=field)  # type: ignore
                readout_feats = input_feats + dim_state_feats if include_state else input_feats  # type: ignore

            dims_final_layer = [readout_feats, units, units, ntargets]
            self.final_layer = MLP(dims_final_layer, activation, activate_last=False)
            if task_type == "classification":
                self.sigmoid = nn.Sigmoid()

        else:
            if task_type == "classification":
                raise ValueError("Classification task cannot be extensive.")
            self.final_layer = WeightedReadOut(
                in_feats=dim_node_embedding,
                dims=[units, units],
                num_targets=ntargets,  # type: ignore
            )

        self.max_n = max_n
        self.max_l = max_l
        self.n_blocks = nblocks
        self.units = units
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff
        self.include_state = include_state
        self.task_type = task_type
        self.is_intensive = is_intensive

    def forward(
            self,
            batch,
            return_all_layer_output: bool = False,
    ):
        """
        Forward pass of the M3GNet model.
        :param batch:
        :param return_all_layer_output:
        :return:
        """

        pbc_distances_out = get_pbc_distances(
            coords=batch.frac_coords,
            edge_index=batch.edge_index,
            lengths=batch.lengths,
            angles=batch.angles,
            to_jimages=batch.to_jimages,
            num_atoms=batch.num_atoms,
            num_bonds=batch.num_bonds,
            coord_is_cart=False,
            return_distance_vec=True
        )
        bond_dist = pbc_distances_out["distances"]
        bond_vec = pbc_distances_out["distance_vec"]

        rbf = self.bond_expansion(bond_dist)

        if hasattr(batch, "line_edge_index") and batch.line_edge_index is not None:
             line_edge_index = batch.line_edge_index
        else:
             line_edge_index = create_line_graph_index(batch, self.threebody_cutoff)

        if line_edge_index.size(1) > 0:
            angle_results = compute_theta_and_phi(
                bond_vec=bond_vec,
                bond_dist=bond_dist,
                line_edge_index=line_edge_index,
                directed=False, # M3GNet 默认 False
                eps=1e-7
            )
            three_body_basis = self.basis_expansion(
                triple_bond_lengths=angle_results["triple_bond_lengths"],
                cos_theta=angle_results["cos_theta"],
                phi=angle_results["phi"]
            )
        else:
            out_dim = self.basis_expansion.out_dim if hasattr(self.basis_expansion, 'out_dim') else 0
            three_body_basis = torch.zeros(
                (0, out_dim),
                device=bond_dist.device,
                dtype=bond_dist.dtype,
            )

        three_body_cutoff = polynomial_cutoff(bond_dist, self.threebody_cutoff)
        state_attr = getattr(batch, "state_attr", None)

        node_feat, edge_feat, state_feat = self.embedding(batch.atom_types, rbf, state_attr)

        layer_outputs = {}
        if return_all_layer_output:
            layer_outputs["embedding"] = {
                "node_feat": node_feat, "edge_feat": edge_feat, "state_feat": state_feat
            }

        for i in range(self.n_blocks):
            edge_feat = self.three_body_interactions[i](
                edge_index=batch.edge_index,
                line_edge_index=line_edge_index,
                three_basis=three_body_basis,
                three_cutoff=three_body_cutoff,
                node_feat=node_feat,
                edge_feat=edge_feat
            )
            node_feat, edge_feat, state_feat = self.graph_layers[i](
                node_feat=node_feat,
                edge_index=batch.edge_index,
                edge_feat=edge_feat,
                rbf=rbf,
                state_feat=state_feat,
                batch_index=batch.batch,
            )

            if return_all_layer_output:
                layer_outputs[f"graph_conv_{i + 1}"] = {
                    "node_feat": node_feat,
                    "edge_feat": edge_feat,
                    "state_feat": state_feat,
                }

        if self.is_intensive:
            readout_vec = self.readout(node_feat, batch.batch)
            if self.include_state:
                 readout_vec = torch.cat([readout_vec, state_feat], dim=1)
            output = self.final_layer(readout_vec)
            if self.task_type == "classification":
                output = self.sigmoid(output)
        else:
            atomic_properties = self.final_layer(node_feat)
            output = scatter(atomic_properties, batch.batch, dim=0, reduce="sum")
            if return_all_layer_output:
                layer_outputs["readout"] = atomic_properties

        if return_all_layer_output:
            layer_outputs["final"] = output
            return layer_outputs

        if output.shape[-1] == 1:
            return output.squeeze(-1)
        return output


class M3GNetSurrogate(BaseModule):
    """
    M3GNet Surrogate Model
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.target_prop = self.hparams.get("target_prop", "y")
        target_mean = float(self.hparams.get("target_mean", 0.0))
        target_std = float(self.hparams.get("target_std", 1.0))
        if target_std <= 0:
            raise ValueError(f"target_std must be positive, got {target_std}.")
        self.register_buffer("target_mean", torch.tensor(target_mean), persistent=False)
        self.register_buffer("target_std", torch.tensor(target_std), persistent=False)
        hp = self.hparams.copy()

        if "dim_node_embedding" in hp:
            hp.setdefault("dim_node_embedding", hp["dim_node_embedding"])
        if "dim_edge_embedding" in hp:
            hp.setdefault("dim_edge_embedding", hp["dim_edge_embedding"])
        if "dim_state_embedding" in hp:
            hp.setdefault("dim_state_embedding", hp["dim_state_embedding"])
        if "units" in hp:
            hp.setdefault("units", hp["units"])
        if "n_atom_types" in hp:
            hp.setdefault("num_elements", hp.pop("n_atom_types"))
        if "readout" in hp:
            readout_map = {"weighted": "weighted_atom", "mean": "reduce_atom", "set2set": "set2set"}
            val = hp["readout"]
            hp.setdefault("readout_type", readout_map.get(val, val))
        if "n_blocks" in hp:
            hp.setdefault("nblocks", hp["n_blocks"])

        self.model = M3GNet(**hp)

        self.loss_fn = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, batch):
        normalized_preds = self.model(batch)
        return normalized_preds * self.target_std + self.target_mean

    def _get_targets(self, batch):
        """
        根据配置的属性名，动态从 batch 中提取数据。
        """
        if hasattr(batch, 'y') and batch.y is not None:
            return batch.y

        target = getattr(batch, self.target_prop, None)
        if target is not None:
            return target

        for prop in ['formation_energy_per_atom', 'e_above_hull', 'band_gap']:
            val = getattr(batch, prop, None)
            if val is not None:
                return val

        available_keys = batch.keys if hasattr(batch, "keys") else "unknown"
        raise KeyError(f"Targets not found. Config: {self.target_prop}. Keys: {available_keys}")

    def compute_stats(self, preds: torch.Tensor, targets: torch.Tensor, prefix: str):
        """
        统一计算 Loss 和 MAE，供 validation 和 test 共用

        Args:
            preds: 预测值
            targets: 真实值
            prefix: 日志前缀 (如 'val', 'test')

        Returns:
            log_dict: 包含各项指标的字典
            loss: 主要 loss 用于反向传播或 scheduler 监控
        """
        normalized_preds = (preds - self.target_mean) / self.target_std
        normalized_targets = (targets - self.target_mean) / self.target_std
        loss = self.loss_fn(normalized_preds, normalized_targets)
        mae = self.mae(preds, targets)

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_mae': mae,
        }
        return log_dict, loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        preds = self(batch)
        targets = self._get_targets(batch)

        if targets.dim() > 1: targets = targets.squeeze()

        normalized_preds = (preds - self.target_mean) / self.target_std
        normalized_targets = (targets - self.target_mean) / self.target_std
        loss = self.loss_fn(normalized_preds, normalized_targets)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss detected at batch_idx {batch_idx}")
            return None

        self.log_dict(
            {'train_loss': loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs if hasattr(batch, 'num_graphs') else 1
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        preds = self(batch)
        targets = self._get_targets(batch)

        if targets.dim() > 1: targets = targets.squeeze()

        log_dict, loss = self.compute_stats(preds, targets, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs if hasattr(batch, 'num_graphs') else 1
        )
        return loss

    def on_test_epoch_start(self):
        """测试开始前，初始化空列表用于存储预测值和真实值"""
        self.test_preds = []
        self.test_targets = []

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        preds = self(batch)
        targets = self._get_targets(batch)

        if targets.dim() > 1: targets = targets.squeeze()

        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(targets.detach().cpu())

        # 使用 compute_stats 统一逻辑
        log_dict, loss = self.compute_stats(preds, targets, prefix='test')

        self.log_dict(
            log_dict,
            batch_size=batch.num_graphs if hasattr(batch, 'num_graphs') else 1
        )
        return loss

    def on_test_epoch_end(self):
        """测试结束后，将所有 batch 的结果拼接并保存"""
        import numpy as np
        import os

        # 拼接所有 batch 的数据
        all_preds = torch.cat(self.test_preds).numpy()
        all_targets = torch.cat(self.test_targets).numpy()

        # 获取 Hydra 运行目录 (Trainer 的 root_dir)
        save_dir = self.trainer.default_root_dir

        # 保存为 .npy 文件
        preds_path = os.path.join(save_dir, "test_preds.npy")
        targets_path = os.path.join(save_dir, "test_targets.npy")

        np.save(preds_path, all_preds)
        np.save(targets_path, all_targets)

        print(f"\n==========================================================")
        print(f"Test targets saved to: {targets_path}")
        print(f"Test predictions saved to: {preds_path}")
        print(f"==========================================================")

        # 清理内存
        self.test_preds.clear()
        self.test_targets.clear()

    def on_after_backward(self) -> None:
        """
        监控梯度范数：有助于发现梯度爆炸（值激增）或梯度消失（值趋近0）的问题。
        """
        total_norm = 0.0
        # 遍历 self.model (M3GNet) 的所有参数
        for p in self.model.parameters():
            if p.grad is not None:
                try:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                except Exception:
                    # 忽略混合精度可能导致的临时未缩放梯度问题
                    pass
        total_norm = total_norm ** 0.5

        self.log(
            'grad_norm',
            total_norm,
            on_step=True,
            on_epoch=True,
            prog_bar=False # 通常不需要在进度条显示，记录到 logger (如 TensorBoard) 即可
        )
