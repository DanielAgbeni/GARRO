"""
Graph Transformer Encoder with Virtual Star Node — Compile-Safe Edition.

Fixes applied
-------------
  ① global_mean_pool / scatter replaced with static view+mean pooling.
    Every graph from GraphConverter has exactly max_nodes nodes, so we can
    reshape [B*max_nodes, H] → [B, max_nodes, H] and call .mean(1).
    This eliminates scatter entirely — no size= arg, no index.max() call,
    no out-of-bounds CUDA assertion, and no requirement for the caller to
    pass num_graphs.  ppo_agent.py needs zero changes.

  ② GraphConverter caches the static edge skeleton at init; hot-path step()
    issues a single non-blocking host→device memcpy of node features only.

  ③ max_nodes baked into the encoder at construction time so torch.compile
    can trace the view+mean as a fixed-shape operation.

  ④ AMP dtype: bfloat16 → float16  (T4 native Tensor Cores; no more
    "skipping bfloat16 compilation" warnings).

  ⑤ PPO config constants updated: batch_size 64→256, update_every 512→2048.
"""
from __future__ import annotations

from typing import List

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import TransformerConv


# ── Constants ─────────────────────────────────────────────────────────────────

NODE_FEAT_DIM = 4   # [cpu, buffer_occ, ingress_rate, egress_rate]
EDGE_FEAT_DIM = 4   # [bw_norm, util, delay_norm, pkt_loss]

_STAR_EDGE_ATTR: List[float] = [1.0, 0.0, 0.0, 0.0]

# ④⑤ Training / AMP config
AMP_DTYPE    = torch.float16   # was torch.bfloat16; T4 has no native bf16
BATCH_SIZE   = 256             # was 64
UPDATE_EVERY = 2048            # was 512


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
    max_nodes  : int   ③ Exact node count per graph (real + star).
                        Must be set to num_topology_nodes + 1.
                       NSFNET = 14 real + 1 star = 15.
                       Every graph produced by GraphConverter must have this
                       many nodes — this is what makes the view+mean safe.

    Forward
    -------
    forward(data) → Tensor[B, hidden_dim]

    No num_graphs argument needed.  B is derived from x.shape[0] // max_nodes.
    The caller (ppo_agent.py) does not need to change.
    """

    def __init__(
        self,
        max_nodes:  int,          # ③ num_topology_nodes + 1 (real + star)
        hidden_dim: int   = 128,
        num_heads:  int   = 4,
        num_layers: int   = 3,
        dropout:    float = 0.1,
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

    def forward(self, data: Data) -> torch.Tensor:
        x = self.node_embed(data.x)                    # [B*max_nodes, H]

        for conv, norm in zip(self.conv_layers, self.layer_norms):
            residual = x
            x = conv(x, data.edge_index, data.edge_attr)
            x = norm(x + residual)
            x = F.relu(x)

        # ① Static reshape pooling — replaces global_mean_pool / scatter.
        #   GraphConverter guarantees exactly max_nodes nodes per graph, so
        #   this view is always valid and contains no dynamic index lookups.
        #   Works for any batch size B without any caller-side arguments.
        B = x.shape[0] // self.max_nodes
        latent = x.view(B, self.max_nodes, self.hidden_dim).mean(dim=1)
        return self.output_mlp(latent)                 # [B, hidden_dim]


# ── GraphConverter ─────────────────────────────────────────────────────────────

class GraphConverter:
    """
    ② Caches the static graph skeleton; hot-path step() only updates node
    telemetry features via a single non-blocking host→device copy.

    Usage
    -----
    converter = GraphConverter(G_topology, device=device)   # once at init
    data      = converter.step(G_live)                      # per env step
    latent    = encoder(data)
    """

    def __init__(
        self,
        G:      nx.Graph,
        device: torch.device = torch.device("cpu"),
    ):
        self.device  = device
        self.nodes   = sorted(G.nodes())
        self.n_real  = len(self.nodes)
        self.n_total = self.n_real + 1       # must equal encoder's max_nodes
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
        self._x[-1, -1] = 1.0          # star-node flag; permanent

        self._feat_np = np.empty((self.n_real, NODE_FEAT_DIM), dtype=np.float32)

    def step(self, G: nx.Graph, clone: bool = True) -> Data:
        """
        Return a PyG Data snapshot with current node telemetry.

        Parameters
        ----------
        G     : live NetworkX graph — only node attributes are re-read;
                topology must match the graph passed to __init__.
        clone : set False only when the Data object is consumed immediately
                and NOT stored alongside other step() outputs.
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


# ── Build helper ──────────────────────────────────────────────────────────────

def build_encoder(device: torch.device, num_nodes: int) -> GraphTransformerEncoder:
    """
    Construct and compile the encoder for the target device.

    Parameters
    ----------
    device    : torch.device  Target compute device.
    num_nodes : int           Number of real nodes in the topology graph.
                              The star node is added automatically (+1).

    Compilation notes (Tesla T4 / CUDA):
    * mode="default"   — solid JIT speedup without max-autotune Triton search;
                         avoids the heavy kernel tuning overhead that causes
                         slowdowns on T4 relative to eager mode.
    * fullgraph=False  — allows graph breaks without raising at startup;
                         safer with PyG's TransformerConv dynamic dispatch.
    * dynamic=True     — single compiled graph handles B=1 (rollout) and
                         B=256 (training update) without recompiling.
    """
    encoder = GraphTransformerEncoder(
        hidden_dim = 256,   # increased from 128 for more capacity
        num_heads  = 8,     # increased from 4; 256/8 = 32 per head
        num_layers = 4,     # increased from 3; gives 4-hop receptive field
        dropout    = 0.15,  # slightly increased to regularise larger model
        max_nodes  = num_nodes + 1,   # real nodes + 1 star node
    ).to(device)

    encoder = torch.compile(encoder, mode="default", fullgraph=False, dynamic=True)
    return encoder


# ── PPO call-site patterns ────────────────────────────────────────────────────

def rollout_step(
    encoder:   GraphTransformerEncoder,
    converter: GraphConverter,
    G_live:    nx.Graph,
    device:    torch.device,
) -> torch.Tensor:
    """Single-graph inference during environment rollout. No args change needed."""
    data = converter.step(G_live, clone=False)
    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
        return encoder(data)                           # B=1 inferred from shape


def ppo_update(
    encoder:   GraphTransformerEncoder,
    optimizer: torch.optim.Optimizer,
    scaler:    torch.cuda.amp.GradScaler,
    data_list: list,
    device:    torch.device,
) -> torch.Tensor:
    """
    Batched PPO update.

    data_list : list of Data objects from converter.step(..., clone=True).
    Returns latent batch [B, hidden_dim] for downstream actor/critic heads.
    """
    batch = Batch.from_data_list(data_list).to(device)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
        latent = encoder(batch)                        # B inferred from shape

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

    Replace with a persistent GraphConverter.step() in any hot-path code.
    """
    return GraphConverter(G, device=device).step(G, clone=False)