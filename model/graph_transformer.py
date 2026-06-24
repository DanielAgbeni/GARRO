"""
Graph Transformer Encoder with Virtual Star Node.

Converts a variable-size NetworkX network state graph into a fixed-size
latent vector suitable for the PPO Actor-Critic head.

Architecture
------------
1. Linear node embedding: [cpu, buf_occ, ingress, egress, star_flag] → hidden_dim
2. N × TransformerConv layers with:
   - 4-dimensional edge attributes [bw_norm, util, delay_norm, pkt_loss]
   - Multi-head self-attention (num_heads heads)
   - Residual connections + LayerNorm for training stability
3. Virtual "star node" appended to every graph:
   - Connected bidirectionally to ALL real nodes
   - Enables O(1)-hop global message passing (solves over-squashing in large WAN graphs)
4. global_mean_pool → fixed-size latent regardless of graph size
5. Two-layer output MLP

This design follows the GTSR architecture (Wang et al., 2025):
    A_{ij}^{(h)} = softmax(q_i·k_j^T / √d_h + ψ(e_{ij}))
where ψ(e_{ij}) is the learned spatial bias from TransformerConv's edge_attr.
"""
from __future__ import annotations

from typing import List

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, global_mean_pool


# ── Constants ─────────────────────────────────────────────────────────────────

# Real node input features (before star-flag is appended):
#   [cpu, buffer_occ, ingress_rate, egress_rate]
NODE_FEAT_DIM = 4

# Edge input features:
#   [bw_normalised, utilisation, delay_normalised, packet_loss]
EDGE_FEAT_DIM = 4

# Neutral edge attributes for star-node ↔ real-node connections
_STAR_EDGE_ATTR: List[float] = [1.0, 0.0, 0.0, 0.0]


# ── Encoder ───────────────────────────────────────────────────────────────────

class GraphTransformerEncoder(nn.Module):
    """
    Multi-layer Graph Transformer with virtual star node.

    Parameters
    ----------
    hidden_dim  : int    Embedding / latent dimension (default 128)
    num_heads   : int    Attention heads per TransformerConv layer (default 4)
    num_layers  : int    Number of stacked TransformerConv layers (default 3)
    dropout     : float  Dropout rate applied inside TransformerConv (default 0.1)

    Input
    -----
    data : torch_geometric.data.Data
        - data.x          Node features  [N+1, NODE_FEAT_DIM+1]  (+1 for star flag)
        - data.edge_index Directed edge index  [2, E_total]
        - data.edge_attr  Edge features  [E_total, EDGE_FEAT_DIM]
        - data.batch      Batch vector   [N+1]

    Output
    ------
    latent : torch.Tensor  Shape [batch_size, hidden_dim]
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads:  int = 4,
        num_layers: int = 3,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input dim = NODE_FEAT_DIM + 1 (star-node indicator flag)
        self.node_embed = nn.Linear(NODE_FEAT_DIM + 1, hidden_dim)

        # Stack of TransformerConv layers
        # Each head produces hidden_dim // num_heads channels; concat=True
        # restores output to hidden_dim.
        self.conv_layers = nn.ModuleList([
            TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                edge_dim=EDGE_FEAT_DIM,
                dropout=dropout,
                concat=True,
            )
            for _ in range(num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Final projection MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = self.node_embed(data.x)                   # [N+1, hidden_dim]

        for conv, norm in zip(self.conv_layers, self.layer_norms):
            residual = x
            x = conv(x, data.edge_index, data.edge_attr)
            x = norm(x + residual)                     # Residual + LayerNorm
            x = F.relu(x)

        # Pool over all nodes (including star node — it summarises global state)
        latent = global_mean_pool(x, data.batch)       # [batch, hidden_dim]
        latent = self.output_mlp(latent)
        return latent


# ── NetworkX → PyG conversion ─────────────────────────────────────────────────

def nx_to_pyg(
    G: nx.Graph,
    device: torch.device = torch.device("cpu"),
) -> Data:
    """
    Convert a NetworkX graph with current telemetry attributes into a
    PyTorch Geometric Data object, with a virtual star node appended.

    Node features per real node  : [cpu, buffer_occ, ingress_rate, egress_rate, 0.0]
    Star node features            : [0.0, 0.0, 0.0, 0.0, 1.0]

    Edge features per real edge  : [bw_norm, util, delay_norm, packet_loss]
    Star-node ↔ real-node edges  : [1.0, 0.0, 0.0, 0.0]  (neutral / high-capacity)

    All edges are added bidirectionally.

    Parameters
    ----------
    G      : nx.Graph       NetworkX graph with telemetry attributes
    device : torch.device   Target device for tensors (default CPU)

    Returns
    -------
    data : torch_geometric.data.Data
    """
    nodes   = sorted(G.nodes())
    n_real  = len(nodes)
    idx_map = {n: i for i, n in enumerate(nodes)}

    # ── Node features ────────────────────────────────────────────────────
    node_feats: List[List[float]] = []
    for n in nodes:
        attrs = G.nodes[n]
        feat  = [
            float(np.clip(attrs.get("cpu",          0.5), 0.0, 1.0)),
            float(np.clip(attrs.get("buffer_occ",   0.3), 0.0, 1.0)),
            float(np.clip(attrs.get("ingress_rate", 0.5), 0.0, 1.0)),
            float(np.clip(attrs.get("egress_rate",  0.5), 0.0, 1.0)),
            0.0,   # star-node flag = 0 for real nodes
        ]
        node_feats.append(feat)

    # Virtual star node (last index = n_real)
    node_feats.append([0.0, 0.0, 0.0, 0.0, 1.0])

    # ── Edge index + attributes ─────────────────────────────────────────
    src_list:  List[int]        = []
    dst_list:  List[int]        = []
    attr_list: List[List[float]] = []

    for u, v, data_e in G.edges(data=True):
        ui = idx_map[u]
        vi = idx_map[v]
        bw_norm    = float(np.clip(data_e.get("bandwidth",   1000) / 10_000.0, 0.0, 1.0))
        util       = float(np.clip(data_e.get("utilization", 0.0),             0.0, 1.0))
        delay_norm = float(np.clip(data_e.get("delay",       1.0)  / 100.0,    0.0, 1.0))
        pkt_loss   = float(np.clip(data_e.get("packet_loss", 0.0),             0.0, 1.0))
        attr       = [bw_norm, util, delay_norm, pkt_loss]

        for s, d in [(ui, vi), (vi, ui)]:   # Bidirectional
            src_list.append(s)
            dst_list.append(d)
            attr_list.append(attr)

    # Star-node ↔ all real nodes (bidirectional)
    star_idx = n_real
    for i in range(n_real):
        for s, d in [(star_idx, i), (i, star_idx)]:
            src_list.append(s)
            dst_list.append(d)
            attr_list.append(_STAR_EDGE_ATTR)

    x          = torch.tensor(node_feats, dtype=torch.float32, device=device)
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long,    device=device)
    edge_attr  = torch.tensor(attr_list,            dtype=torch.float32, device=device)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
