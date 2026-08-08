import itertools
from enum import Enum
from typing import Sequence, Callable, Literal

import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear, ModuleList
from torch_scatter import scatter, scatter_sum
from torch_geometric.nn import global_mean_pool, global_add_pool, Set2Set
from torch_geometric.data import Data

from cgdit.prop_models.gnn_models.utils import (
    GaussianExpansion, SphericalBesselFunction, ExpNormalFunction,
    SphericalHarmonicsFunction, combine_sbf_shf,
)


def get_spherical_bessel_roots(n: int) -> torch.Tensor:
    """预计算球贝塞尔函数的根 (用于基函数扩展)"""
    rng = np.arange(1, n + 1)
    roots = rng * np.pi
    return torch.from_numpy(roots).float()


class SoftPlus2(nn.Module):
    """SoftPlus2 activation function:
    out = log(exp(x)+1) - log(2)
    softplus function that is 0 at x=0, the implementation aims at avoiding overflow.
    """

    def __init__(self) -> None:
        """Initializes the SoftPlus2 class."""
        super().__init__()
        self.ssp = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate activation function given the input tensor x.

        Args:
            x (torch.tensor): Input tensor

        Returns:
            out (torch.tensor): Output tensor
        """
        return self.ssp(x) - math.log(2.0)


class SoftExponential(nn.Module):
    """Soft exponential activation.
    When x < 0, SoftExponential(x,alpha) = -log(1-alpha(x+alpha))/alpha
    When x = 0, SoftExponential(x,alpha) = 0
    When x > 0, SoftExponential(x,alpha) = (exp(alpha*x)-1)/alpha + alpha.

    References: https://arxiv.org/pdf/1602.01321.pdf
    """

    def __init__(self, alpha: float | None = None):
        """
        Init SoftExponential with alpha value.

        Args:
            alpha (float): adjustable Torch parameter during the training.
        """
        super().__init__()

        # initialize alpha
        if alpha is None:
            self.alpha = nn.Parameter(torch.tensor(0.0))
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha))

        self.alpha.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate activation function given the input tensor x.

        Args:
            x (torch.tensor): Input tensor

        Returns:
            out (torch.tensor): Output tensor
        """
        if self.alpha == 0.0:
            return x
        if self.alpha < 0.0:
            return -torch.log(1.0 - self.alpha * (x + self.alpha)) / self.alpha
        return (torch.exp(self.alpha * x) - 1.0) / self.alpha + self.alpha


class ActivationFunction(Enum):
    """Enumeration of optional activation functions."""

    swish = nn.SiLU
    sigmoid = nn.Sigmoid
    tanh = nn.Tanh
    softplus = nn.Softplus
    softplus2 = SoftPlus2
    softexp = SoftExponential


class MLP(nn.Module):
    """An implementation of a multi-layer perceptron."""

    def __init__(
        self,
        dims: Sequence[int],
        activation: Callable[[Tensor], Tensor] | None = None,
        activate_last: bool = False,
        bias_last: bool = True,
    ) -> None:
        """:param dims: Dimensions of each layer of MLP.
        :param activation: Activation function.
        :param activate_last: Whether to apply activation to last layer.
        :param bias_last: Whether to apply bias to last layer.
        """
        super().__init__()
        self._depth = len(dims) - 1
        self.layers = ModuleList()

        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            if i < self._depth - 1:
                self.layers.append(Linear(in_dim, out_dim, bias=True))

                if activation is not None:
                    self.layers.append(activation)  # type: ignore
            else:
                self.layers.append(Linear(in_dim, out_dim, bias=bias_last))

                if activation is not None and activate_last:
                    self.layers.append(activation)  # type: ignore

    def __repr__(self) -> str:
        dims = []

        for layer in self.layers:
            if isinstance(layer, Linear):
                dims.append(f"{layer.in_features} \u2192 {layer.out_features}")
            else:
                dims.append(layer.__class__.__name__)

        return f"MLP({', '.join(dims)})"

    @property
    def last_linear(self) -> Linear:
        """Return the last linear layer in the network."""
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                return layer
        msg = "MLP must contain at least one Linear layer."
        raise RuntimeError(msg)

    @property
    def depth(self) -> int:
        """Returns depth of MLP."""
        return self._depth

    @property
    def in_features(self) -> int:
        """Return input features of MLP."""
        first_layer = self.layers[0]
        assert isinstance(first_layer, Linear), "First layer must be Linear"
        return first_layer.in_features

    @property
    def out_features(self) -> int:
        """Returns output features of MLP."""
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                return layer.out_features
        raise RuntimeError

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply each layer in turn."""
        x = inputs
        for layer in self.layers:
            x = layer(x)

        return x


class GatedMLP(nn.Module):
    """An implementation of a Gated multi-layer perceptron."""

    def __init__(self, in_feats: int, dims: Sequence[int], activate_last: bool = True, use_bias: bool = True):
        """:param in_feats: Dimension of input features.
        :param dims: Architecture of neural networks.
        :param activate_last: Whether applying activation to last layer or not.
        :param use_bias: Whether applying bias in MLP.
        """
        super().__init__()
        self.in_feats = in_feats
        self.dims = [in_feats, *dims]
        self._depth = len(dims)
        self.layers = nn.Sequential()
        self.gates = nn.Sequential()
        self.use_bias = use_bias
        self.activate_last = activate_last
        for i, (in_dim, out_dim) in enumerate(zip(self.dims[:-1], self.dims[1:], strict=False)):
            if i < self._depth - 1:
                self.layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
                self.gates.append(nn.Linear(in_dim, out_dim, bias=use_bias))
                self.layers.append(nn.SiLU())
                self.gates.append(nn.SiLU())
            else:
                self.layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
                if self.activate_last:
                    self.layers.append(nn.SiLU())
                self.gates.append(nn.Linear(in_dim, out_dim, bias=use_bias))
                self.gates.append(nn.Sigmoid())

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs) * self.gates(inputs)


class BondExpansion(nn.Module):
    """Expand pair distances into a set of spherical bessel or gaussian functions."""

    def __init__(
        self,
        max_l: int = 3,
        max_n: int = 3,
        cutoff: float = 5.0,
        rbf_type: Literal["SphericalBessel", "Gaussian", "ExpNorm"] = "SphericalBessel",
        smooth: bool = False,
        initial: float = 0.0,
        final: float = 5.0,
        num_centers: int = 100,
        width: float = 0.5,
    ) -> None:
        """
        Args:
            max_l (int): order of angular part
            max_n (int): order of radial part
            cutoff (float): cutoff radius
            rbf_type (str): type of radial basis function .i.e. either "SphericalBessel", "ExpNorm" or 'Gaussian'
            smooth (bool): whether apply the smooth version of spherical bessel functions or not
            initial (float): initial point for gaussian expansion
            final (float): final point for gaussian expansion
            num_centers (int): Number of centers for gaussian expansion.
            width (float): width of gaussian function.
        """
        super().__init__()

        self.max_n = max_n
        self.cutoff = cutoff
        self.max_l = max_l
        self.smooth = smooth
        self.num_centers = num_centers
        self.width = width
        self.initial = initial
        self.final = final
        self.rbf_type = rbf_type

        if rbf_type.lower() == "sphericalbessel":
            self.rbf = SphericalBesselFunction(max_l, max_n, cutoff, smooth)  # type:ignore[assignment]
        elif rbf_type.lower() == "gaussian":
            self.rbf = GaussianExpansion(initial, final, num_centers, width)  # type:ignore[assignment]
        elif rbf_type.lower() == "expnorm":
            self.rbf = ExpNormalFunction(cutoff, num_centers, True)  # type:ignore[assignment]
        else:
            raise ValueError("Undefined rbf_type, please use SphericalBessel or Gaussian instead.")

    def forward(self, bond_dist: torch.Tensor):
        """Forward.

        Args:
        bond_dist: Bond distance

        Return:
        bond_basis: Radial basis functions
        """
        bond_basis = self.rbf(bond_dist)
        return bond_basis


class EmbeddingBlock(nn.Module):
    """Embedding block for generating node, bond and state features."""

    def __init__(
        self,
        degree_rbf: int,
        activation: nn.Module,
        dim_node_embedding: int,
        dim_edge_embedding: int | None = None,
        dim_state_feats: int | None = None,
        ntypes_node: int | None = None,
        include_state: bool = False,
        ntypes_state: int | None = None,
        dim_state_embedding: int | None = None,
    ):
        """
        Args:
            degree_rbf (int): number of rbf
            activation (nn.Module): activation type
            dim_node_embedding (int): dimensionality of node features
            dim_edge_embedding (int): dimensionality of edge features
            dim_state_feats: dimensionality of state features
            ntypes_node: number of node labels
            include_state: Whether to include state embedding
            ntypes_state: number of state labels
            dim_state_embedding: dimensionality of state embedding.
        """
        super().__init__()
        self.include_state = include_state
        self.ntypes_state = ntypes_state
        self.dim_node_embedding = dim_node_embedding
        self.dim_edge_embedding = dim_edge_embedding
        self.dim_state_feats = dim_state_feats
        self.ntypes_node = ntypes_node
        self.dim_state_embedding = dim_state_embedding
        self.activation = activation
        if ntypes_state and dim_state_embedding is not None:
            self.layer_state_embedding = nn.Embedding(ntypes_state, dim_state_embedding)  # type: ignore
        elif dim_state_feats is not None:
            self.layer_state_embedding = nn.Sequential(  # type:ignore[assignment]
                nn.LazyLinear(dim_state_feats, bias=False, dtype=torch.float32),
                activation,
            )
        if ntypes_node is not None:
            self.layer_node_embedding = nn.Embedding(ntypes_node, dim_node_embedding)
        else:
            self.layer_node_embedding = nn.Sequential(  # type:ignore[assignment]
                nn.LazyLinear(dim_node_embedding, bias=False, dtype=torch.float32),
                activation,
            )
        if dim_edge_embedding is not None:
            dim_edges = [degree_rbf, dim_edge_embedding]
            self.layer_edge_embedding = MLP(dim_edges, activation=activation, activate_last=True)

    def forward(self, node_attr, edge_attr, state_attr):
        """Output embedded features.

        Args:
            node_attr: node attribute
            edge_attr: edge attribute
            state_attr: state attribute

        Returns:
            node_feat: embedded node features
            edge_feat: embedded edge features
            state_feat: embedded state features
        """
        if self.ntypes_node is not None:
            node_feat = self.layer_node_embedding(node_attr)
        else:
            node_feat = self.layer_node_embedding(node_attr.to(torch.float32))
        if self.dim_edge_embedding is not None:
            edge_feat = self.layer_edge_embedding(edge_attr.to(torch.float32))
        else:
            edge_feat = edge_attr
        if self.include_state is True:
            if self.ntypes_state and self.dim_state_embedding is not None:
                state_feat = self.layer_state_embedding(state_attr)
            elif self.dim_state_feats is not None:
                state_attr = torch.unsqueeze(state_attr, 0)
                state_feat = self.layer_state_embedding(state_attr.to(torch.float32))
            else:
                state_feat = state_attr
        else:
            state_feat = None
        return node_feat, edge_feat, state_feat


class SphericalBesselWithHarmonics(nn.Module):
    """Expansion of basis using Spherical Bessel and Harmonics."""

    def __init__(self, max_n: int, max_l: int, cutoff: float, use_smooth: bool, use_phi: bool):
        """
        Init SphericalBesselWithHarmonics.

        Args:
            max_n: Degree of radial basis functions.
            max_l: Degree of angular basis functions.
            cutoff: Cutoff sphere.
            use_smooth: Whether using smooth version of SBFs or not.
            use_phi: Using phi as angular basis functions.
        """
        super().__init__()

        assert max_n <= 64
        self.max_n = max_n
        self.max_l = max_l
        self.cutoff = cutoff
        self.use_phi = use_phi
        self.use_smooth = use_smooth

        # retrieve formulas
        self.shf = SphericalHarmonicsFunction(self.max_l, self.use_phi)
        if self.use_smooth:
            self.sbf = SphericalBesselFunction(self.max_l, self.max_n * self.max_l, self.cutoff, self.use_smooth)
        else:
            self.sbf = SphericalBesselFunction(self.max_l, self.max_n, self.cutoff, self.use_smooth)

    def forward(self, triple_bond_lengths, cos_theta, phi):
        """

        :param triple_bond_lengths:
        :param cos_theta:
        :param phi:
        :return:
        """
        sbf = self.sbf(triple_bond_lengths)
        shf = self.shf(cos_theta, phi)
        return combine_sbf_shf(sbf, shf, max_n=self.max_n, max_l=self.max_l, use_phi=self.use_phi)


class ThreeBodyInteractions(nn.Module):

    def __init__(self, update_network_atom, update_network_bond):
        super().__init__()
        self.update_network_atom = update_network_atom
        self.update_network_bond = update_network_bond

    def forward(
        self,
        edge_index,
        line_edge_index,
        three_basis,
        three_cutoff,
        node_feat,
        edge_feat,
    ):
        # 1. atom update
        updated_atoms = self.update_network_atom(node_feat)

        # 2. 取三体中第二条 bond 的终止原子
        end_atom_indices = edge_index[1, line_edge_index[1]]
        end_atom_features = updated_atoms[end_atom_indices]

        # 3. basis × atom
        basis = three_basis * end_atom_features

        # 4. cutoff 权重
        weights = three_cutoff[line_edge_index].prod(dim=0)
        basis = basis * weights[:, None]

        # 5. 聚合回原始 bond
        target_edge = line_edge_index[0]
        new_bonds = scatter(
            basis,
            target_edge,
            dim=0,
            dim_size=edge_feat.size(0),
            reduce="sum",
        )

        if new_bonds.numel() == 0:
            return edge_feat

        # 6. bond update
        return edge_feat + self.update_network_bond(new_bonds)


class M3GNetBlock(nn.Module):
    """
    PyG version of M3GNetBlock.
    This block operates directly on PyG Data.
    """

    def __init__(
        self,
        degree: int,
        activation: nn.Module,
        conv_hiddens: list[int],
        dim_node_feats: int,
        dim_edge_feats: int,
        dim_state_feats: int = 0,
        include_state: bool = False,
        dropout: float | None = None,
    ):
        super().__init__()

        self.include_state = include_state
        self.activation = activation

        # -----------------------------
        # compute input sizes (same as DGL)
        # -----------------------------
        if include_state:
            edge_in = 2 * dim_node_feats + dim_edge_feats + dim_state_feats
            node_in = 2 * dim_node_feats + dim_edge_feats + dim_state_feats
            state_in = dim_node_feats + dim_state_feats

            self.conv = M3GNetGraphConv.from_dims(
                degree=degree,
                include_state=True,
                edge_dims=[edge_in, *conv_hiddens, dim_edge_feats],
                node_dims=[node_in, *conv_hiddens, dim_node_feats],
                state_dims=[state_in, *conv_hiddens, dim_state_feats],
                activation=self.activation,
            )
        else:
            edge_in = 2 * dim_node_feats + dim_edge_feats
            node_in = 2 * dim_node_feats + dim_edge_feats

            self.conv = M3GNetGraphConv.from_dims(
                degree=degree,
                include_state=False,
                edge_dims=[edge_in, *conv_hiddens, dim_edge_feats],
                node_dims=[node_in, *conv_hiddens, dim_node_feats],
                state_dims=None,
                activation=self.activation,
            )

        self.dropout = nn.Dropout(dropout) if dropout else None

    # --------------------------------------------------
    # forward: PyG Data -> PyG Data
    # --------------------------------------------------
    def forward(
            self,
            node_feat,
            edge_index,
            edge_feat,
            rbf,
            state_feat=None,
            batch_index=None,
    ):
        """
        Args:
            data.x         : [N, dim_node_feats]
            data.edge_index: [2, E]
            data.edge_attr : [E, dim_edge_feats]
            data.rbf       : [E, degree]
            data.state     : [B, dim_state_feats] (optional)
            data.batch     : [N] (optional)

        Returns:
            Updated PyG Data
        """

        edge_attr, node_attr, state = self.conv(
            x=node_feat,
            edge_index=edge_index,
            edge_attr=edge_feat,
            rbf=rbf,
            state=state_feat,
            batch=batch_index,
        )

        if self.dropout:
            node_attr = self.dropout(node_attr)
            edge_attr = self.dropout(edge_attr)
            if state is not None:
                state = self.dropout(state)

        return node_attr, edge_attr, state


class EdgeSet2Set(nn.Module):
    """
    Set2Set readout over edges for PyG.
    """

    def __init__(self, input_dim: int, n_iters: int, n_layers: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim  # Set2Set doubles feature dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=self.output_dim, hidden_size=self.input_dim, num_layers=n_layers)

    def forward(self, data: Data):
        """
        Args:
            data.edge_attr: [E, F]
            data.batch (optional): [E], edge-batch assignment

        Returns:
            Tensor: [batch_size, 2*F]
        """
        feat = data.edge_attr
        batch = getattr(data, "edge_batch", None)
        if batch is None:
            batch = torch.zeros(feat.size(0), dtype=torch.long, device=feat.device)

        batch_size = int(batch.max().item()) + 1

        h = (
            feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
            feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
        )

        q_star = feat.new_zeros(batch_size, 2 * self.input_dim)

        for _ in range(self.n_iters):
            q, h = self.lstm(q_star.unsqueeze(0), h)  # [1, B, F] -> [1, B, F]
            q = q.view(batch_size, self.input_dim)  # [B, F]

            # attention over edges
            q_expand = q[batch]  # [E, F]
            e = (feat * q_expand).sum(dim=-1, keepdim=True)  # [E,1]
            alpha = torch.softmax(e, dim=0)  # softmax over all edges (PyG version)
            r = feat * alpha  # weighted edge features
            readout = scatter(r, batch, dim=0, reduce="sum")  # [B, F]

            q_star = torch.cat([q, readout], dim=-1)  # [B, 2*F]

        return q_star


class Set2SetReadOut(nn.Module):
    """
    General Set2Set readout for node or edge features.
    """

    def __init__(self, in_feats: int, n_iters: int, n_layers: int, field: str = "node_feat"):
        super().__init__()
        self.field = field
        self.n_iters = n_iters
        self.n_layers = n_layers

        if field == "node_feat":
            self.set2set = Set2Set(in_feats, processing_steps=n_iters, num_layers=n_layers)
        elif field == "edge_feat":
            self.set2set = EdgeSet2Set(in_feats, n_iters, n_layers)
        else:
            raise ValueError("Field must be 'node_feat' or 'edge_feat'")

    def forward(self, input_feat, batch_index):
        if self.field == "node_feat":
            return self.set2set(input_feat, batch_index)
        else:
            return self.set2set(input_feat)


class WeightedReadOut(nn.Module):
    """Feed node features into Gated MLP as readout for atomic properties (PyG)."""

    def __init__(self, in_feats: int, dims: Sequence[int], num_targets: int):
        """
        Args:
            in_feats: input features (nodes)
            dims: NN architecture for Gated MLP
            num_targets: number of target properties.
        """
        super().__init__()
        self.in_feats = in_feats
        self.dims = [in_feats, *dims, num_targets]
        self.gated = GatedMLP(
            in_feats=in_feats,
            dims=self.dims,
            activate_last=False
        )

    def forward(self, input_feat, batch_index):
        """
        Args:
            data: PyG Data or Batch object, must contain data.x

        Returns:
            atomic_properties: Tensor [num_nodes, num_targets]
        """
        atomic_properties = self.gated(input_feat)
        return atomic_properties


class WeightedAtomReadOut(nn.Module):
    """Weighted atom readout for graph properties (PyG)."""

    def __init__(self, in_feats: int, dims: Sequence[int], activation: nn.Module):
        """
        Args:
            in_feats: input features (nodes)
            dims: NN architecture for MLP
            activation: activation function
        """
        super().__init__()
        self.dims = [in_feats, *dims]
        self.activation = activation

        self.mlp = MLP(
            dims=self.dims,
            activation=self.activation,
            activate_last=True
        )

        self.weight = nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
        )

    def forward(self, input_feat, batch_index):
        """
        Args:
            data: PyG Data or Batch object
                  must contain data.x and data.batch

        Returns:
            graph_properties: Tensor [num_graphs, dims[-1]]
        """
        h = self.mlp(input_feat)         # [num_nodes, hidden_dim]
        w = self.weight(input_feat)      # [num_nodes, 1]

        # 节点级加权
        h_weighted = h * w      # broadcast 自动完成

        # 图级加权求和
        h_g_sum = global_add_pool(h_weighted, batch_index)

        return h_g_sum


class ReduceReadOut(nn.Module):
    """
    Reduce node or edge attributes into lower dimensional tensors.
    """

    def __init__(self, op: str = "mean", field: str = "node_feat"):
        super().__init__()
        self.op = op
        self.field = field

    def forward(self, input_feat, batch_index):
        if batch_index is None:
            batch_index = torch.zeros(input_feat.size(0), dtype=torch.long, device=input_feat.device)

        if self.op == "mean":
            return scatter(input_feat, batch_index, dim=0, reduce="mean")
        elif self.op == "sum":
            return scatter(input_feat, batch_index, dim=0, reduce="sum")
        elif self.op == "max":
            return scatter(input_feat, batch_index, dim=0, reduce="max")
        else:
            raise ValueError(f"Unsupported op: {self.op}")


class M3GNetGraphConv(nn.Module):
    """
    PyG implementation of M3GNetGraphConv
    """

    def __init__(
        self,
        include_state: bool,
        edge_update_func: nn.Module,
        edge_weight_func: nn.Module,
        node_update_func: nn.Module,
        node_weight_func: nn.Module,
        state_update_func: nn.Module | None,
    ):
        super().__init__()
        self.include_state = include_state
        self.edge_update_func = edge_update_func
        self.edge_weight_func = edge_weight_func
        self.node_update_func = node_update_func
        self.node_weight_func = node_weight_func
        self.state_update_func = state_update_func

    # ----------------------------------------------------
    # Edge update: e_ij ← e_ij + φ_e(...) ⊙ W_e(rbf)
    # ----------------------------------------------------
    def edge_update(self, x, edge_index, edge_attr, rbf, state):
        src, dst = edge_index  # i -> j

        vi = x[src]
        vj = x[dst]

        if self.include_state:
            u = state.expand(edge_attr.size(0), -1)
            inputs = torch.cat([vi, vj, edge_attr, u], dim=-1)
        else:
            inputs = torch.cat([vi, vj, edge_attr], dim=-1)

        edge_msg = self.edge_update_func(inputs)
        edge_weight = self.edge_weight_func(rbf)
        return edge_attr + edge_msg * edge_weight

    # ----------------------------------------------------
    # Node update: v_i ← v_i + Σ_j φ_v(...) ⊙ W_v(rbf)
    # ----------------------------------------------------
    def node_update(self, x, edge_index, edge_attr, rbf, state):
        src, dst = edge_index

        vi = x[src]
        vj = x[dst]

        if self.include_state:
            u = state.expand(edge_attr.size(0), -1)
            inputs = torch.cat([vi, vj, edge_attr, u], dim=-1)
        else:
            inputs = torch.cat([vi, vj, edge_attr], dim=-1)

        node_msg = self.node_update_func(inputs)
        node_weight = self.node_weight_func(rbf)
        msg = node_msg * node_weight

        # 聚合到 dst 节点（等价于 DGL update_all sum）
        agg = scatter(msg, dst, dim=0, dim_size=x.size(0), reduce="sum")
        return x + agg

    # ----------------------------------------------------
    # State update: u ← φ_u([u, mean(v)])
    # ----------------------------------------------------
    def state_update(self, x, state, batch):
        if self.state_update_func is None:
            return state

        pooled = global_mean_pool(x, batch)
        inputs = torch.cat([state, pooled], dim=-1)
        return self.state_update_func(inputs)

    # ----------------------------------------------------
    # Forward
    # ----------------------------------------------------
    def forward(
        self,
        x,              # node_feat [N, Fv]
        edge_index,     # [2, E]
        edge_attr,      # [E, Fe]
        rbf,            # [E, degree]
        state=None,     # [B, Fu] or [1, Fu]
        batch=None,     # [N]
    ):
        if self.include_state and state is None:
            raise ValueError("state_feat is required when include_state=True")

        # Edge update
        edge_attr = self.edge_update(x, edge_index, edge_attr, rbf, state)

        # Node update
        x = self.node_update(x, edge_index, edge_attr, rbf, state)

        # State update
        if self.include_state:
            if batch is None:
                batch = x.new_zeros(x.size(0), dtype=torch.long)
            state = self.state_update(x, state, batch)

        return edge_attr, x, state

    @staticmethod
    def from_dims(
            degree,
            include_state,
            edge_dims,
            node_dims,
            state_dims,
            activation,
    ):
        edge_update_func = GatedMLP(edge_dims[0], edge_dims[1:])
        edge_weight_func = nn.Linear(degree, edge_dims[-1], bias=False)

        node_update_func = GatedMLP(node_dims[0], node_dims[1:])
        node_weight_func = nn.Linear(degree, node_dims[-1], bias=False)

        state_update_func = (
            MLP(state_dims, activation, activate_last=True)
            if include_state else None
        )

        return M3GNetGraphConv(
            include_state,
            edge_update_func,
            edge_weight_func,
            node_update_func,
            node_weight_func,
            state_update_func,
        )

