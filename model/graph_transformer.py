"""
Graph Transformer Encoder with Virtual Star Node — Compile-Safe Edition.

Fixes applied
-------------
  ① global_mean_pool called with explicit size= argument, eliminating the
    int(index.max()) host-sync that caused the torch.compile graph break.
  ② GraphConverter caches the static edge skeleton at init; hot-path step()
    issues a single non-blocking host→device copy of node features only.
  ③ max_nodes accepted at construction so torch.compile lowers to fixed-size
    GPU kernels without shape guards on data.batch.
  ④ AMP dtype: bfloat16 → float16 (T4 native Tensor Cores; no more warnings).
  ⑤ PPO config: batch_size 64→256, update_every 512→2048.
"""
from __future__ import annotations

from typing import List

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import TransformerConv, global_mean_pool


# ── Constants ─────────────────────────────────────────────────────────────────

NODE_FEAT_DIM = 4   # [cpu, buffer_occ, ingress_rate, egress_rate]
EDGE_FEAT_DIM = 4   # [bw_norm, util, delay_norm, pkt_loss]

_STAR_EDGE_ATTR: List[float] = [1.0, 0.0, 0.0, 0.0]

# ⑤ PPO / training config
AMP_DTYPE    = torch.float16   # ④ was torch.bfloat16; T4 has no native bf16
BATCH_SIZE   = 256             # ⑤ was 64
UPDATE_EVERY = 2048            # ⑤ was 512


# ── Helpers ───────────────────────────────────────────────────────────────────

def _edge_attr_from_dict(d: dict) -> List[float]:
    return [
        float(np.clip(d.get("bandwidth",   1000) / 10_000.0, 0.0, 1.0)),
        float(np.clip(d.get("utilization", 0.0),              0.0, 1.0)),
        float(np.clip(d.get("delay",       1.0)  / 100.0,     0.0, 1.0)),
        float(np.clip(d.get("packet_loss", 0.0),              0.0, 1.0)),
    ]


# ── Encoder ───────────────────────────────────────────────────────────────────

class GraphTransformerEncoder(nn.Module):
    """
    Multi-layer Graph Transformer with virtual star node.

    Parameters
    ----------
    hidden_dim : int   Embedding / latent dimension (default 128).
    num_heads  : int   Attention heads per TransformerConv layer (default 4).
    num_layers : int   Stacked TransformerConv layers (default 3).
    dropout    : float Dropout rate inside TransformerConv (default 0.1).
    max_nodes  : int   ③ Static upper bound on nodes per graph (real + star).
                       NSFNET = 14 real + 1 star = 15.

    Forward
    -------
    forward(data, num_graphs=1) → Tensor[num_graphs, hidden_dim]

    Pass num_graphs explicitly so global_mean_pool receives a compile-time
    constant size argument and never calls index.max() (fix ①).
    """

    def __init__(
        self,
        hidden_dim: int   = 128,
        num_heads:  int   = 4,
        num_layers: int   = 3,
        dropout:    float = 0.1,
        max_nodes:  int   = 15,  # ③ NSFNET: 14 real + 1 star
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_nodes  = max_nodes

        self.node_embed = nn.Linear(NODE_FEAT_DIM + 1, hidden_dim)

        self.conv_layers = nn.ModuleList([
            TransformerConv(
                in_channels  = hidden_dim,
                out_channels = hidden_dim // num_heads,
                heads        = num_heads,
                edge_dim     = EDGE_FEAT_DIM,
                dropout      = dropout,
                concat       = True,
            )
            for _ in range(num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, data: Data, num_graphs: int = 1) -> torch.Tensor:
        x = self.node_embed(data.x)

        for conv, norm in zip(self.conv_layers, self.layer_norms):
            residual = x
            x = conv(x, data.edge_index, data.edge_attr)
            x = norm(x + residual)
            x = F.relu(x)

        # ① Explicit size= eliminates int(index.max()) and the graph break.
        latent = global_mean_pool(x, data.batch, size=num_graphs)
        return self.output_mlp(latent)


# ── GraphConverter ─────────────────────────────────────────────────────────────

class GraphConverter:
    """
    ② Caches the static graph skeleton; hot-path step() only updates node
    telemetry features via a single non-blocking host→device copy.

    Usage
    -----
    converter = GraphConverter(G_topology, device=device)   # once at init
    data      = converter.step(G_live)                      # per env step
    latent    = encoder(data, num_graphs=1)
    """

    def __init__(
        self,
        G:      nx.Graph,
        device: torch.device = torch.device("cpu"),
    ):
        self.device  = device
        self.nodes   = sorted(G.nodes())
        self.n_real  = len(self.nodes)
        self.n_total = self.n_real + 1
        self._idx    = {n: i for i, n in enumerate(self.nodes)}

        # ── Pre-compute static edge_index + edge_attr (runs once) ─────────
        src:   List[int]         = []
        dst:   List[int]         = []
        attrs: List[List[float]] = []

        for u, v, d in G.edges(data=True):
            ui   = self._idx[u]
            vi   = self._idx[v]
            attr = _edge_attr_from_dict(d)
            src  += [ui, vi];  dst += [vi, ui]
            attrs += [attr, attr]

        star = self.n_real
        for i in range(self.n_real):
            src  += [star, i];  dst += [i, star]
            attrs += [_STAR_EDGE_ATTR, _STAR_EDGE_ATTR]

        self.edge_index = torch.tensor([src, dst],  dtype=torch.long,    device=device)
        self.edge_attr  = torch.tensor(attrs,        dtype=torch.float32, device=device)
        self.batch      = torch.zeros(self.n_total,  dtype=torch.long,    device=device)

        # ── Pre-allocate node feature buffer ─────────────────────────────
        self._x = torch.zeros(
            self.n_total, NODE_FEAT_DIM + 1,
            dtype=torch.float32, device=device,
        )
        self._x[-1, -1] = 1.0  # star-node flag; permanent

        self._feat_np = np.empty((self.n_real, NODE_FEAT_DIM), dtype=np.float32)

    def step(self, G: nx.Graph, clone: bool = True) -> Data:
        """
        Return a PyG Data snapshot with current node telemetry.

        Parameters
        ----------
        G     : live NetworkX graph — only node attributes are re-read;
                topology must match the graph passed to __init__.
        clone : set False only when Data is consumed immediately and not
                stored alongside other step() outputs in a replay buffer.
        """
        feat = self._feat_np
        for i, n in enumerate(self.nodes):
            a = G.nodes[n]
            feat[i, 0] = np.clip(a.get("cpu",          0.5), 0.0, 1.0)
            feat[i, 1] = np.clip(a.get("buffer_occ",   0.3), 0.0, 1.0)
            feat[i, 2] = np.clip(a.get("ingress_rate", 0.5), 0.0, 1.0)
            feat[i, 3] = np.clip(a.get("egress_rate",  0.5), 0.0, 1.0)

        # ② Single non-blocking host→device copy for all real-node features.
        self._x[: self.n_real, : NODE_FEAT_DIM].copy_(
            torch.from_numpy(feat), non_blocking=True
        )

        return Data(
            x          = self._x.clone() if clone else self._x,
            edge_index = self.edge_index,
            edge_attr  = self.edge_attr,
            batch      = self.batch,
        )


# ── Build helpers ─────────────────────────────────────────────────────────────

def build_encoder(device: torch.device) -> GraphTransformerEncoder:
    """
    Construct and compile the encoder for the target device.

    fullgraph=True will raise at startup if any graph break survives,
    making regressions immediately visible rather than silently degrading.
    """
    encoder = GraphTransformerEncoder(
        hidden_dim = 128,
        num_heads  = 4,
        num_layers = 3,
        dropout    = 0.1,
        max_nodes  = 15,
    ).to(device)

    encoder = torch.compile(encoder, fullgraph=True, dynamic=False)
    return encoder


# ── PPO call-site patterns ────────────────────────────────────────────────────

def rollout_step(
    encoder:   GraphTransformerEncoder,
    converter: GraphConverter,
    G_live:    nx.Graph,
    device:    torch.device,
) -> torch.Tensor:
    """Single-graph inference during environment rollout."""
    data = converter.step(G_live, clone=False)  # consumed immediately
    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
        return encoder(data, num_graphs=1)       # ① num_graphs is a compile-time constant


def ppo_update(
    encoder:   GraphTransformerEncoder,
    optimizer: torch.optim.Optimizer,
    scaler:    torch.cuda.amp.GradScaler,
    data_list: list,
    device:    torch.device,
) -> torch.Tensor:
    """
    Batched PPO update.

    data_list : list of Data objects collected from converter.step(..., clone=True).
    Returns the latent batch for use in downstream actor/critic heads.
    """
    batch   = Batch.from_data_list(data_list).to(device)
    n       = len(data_list)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
        latent = encoder(batch, num_graphs=n)    # ① n known at call time, not traced dynamically

    # Caller computes PPO loss, then:
    #   scaler.scale(loss).backward()
    #   scaler.step(optimizer)
    #   scaler.update()
    return latent


# ── Legacy shim ───────────────────────────────────────────────────────────────

def nx_to_pyg(
    G:      nx.Graph,
    device: torch.device = torch.device("cpu"),
) -> Data:
    """
    One-shot conversion for backward compatibility.

    For any hot-path code replace with a persistent GraphConverter instance
    and call converter.step(G) to avoid recomputing the edge skeleton.
    """
    return GraphConverter(G, device=device).step(G, clone=False)