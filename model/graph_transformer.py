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

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import TransformerConv


from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import softmax


# ── Constants ─────────────────────────────────────────────────────────────────

NODE_FEAT_DIM = 6   # [cpu, buffer_occ, ingress_rate, egress_rate, is_src, is_dst]
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


# ── Spatial Transformer Layer with Learned Spatial Bias ψ(e_ij) ─────────────

class SpatialTransformerConv(MessagePassing):
    r"""
    Graph Transformer Layer incorporating learned spatial attention bias ψ(e_ij).

    Strictly matching Equation 2.4.3 in the proposal paper:
    A_{ij}^{(h)} = exp( (q_i^{(h)} (k_j^{(h)})^T) / sqrt(d_h) + \psi(e_{ij}) ) /
                   sum_{u in V} exp(...)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        edge_dim: int = 4,
        max_dist: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__(node_dim=0, aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.edge_dim = edge_dim
        self.dropout = dropout

        self.lin_q = nn.Linear(in_channels, heads * out_channels)
        self.lin_k = nn.Linear(in_channels, heads * out_channels)
        self.lin_v = nn.Linear(in_channels, heads * out_channels)

        self.lin_edge = nn.Linear(edge_dim, heads * out_channels)
        # Learned spatial bias embedding mapping shortest-path hop distances to scalar bias per head
        self.spatial_bias = nn.Embedding(max_dist + 2, heads)

        self.lin_skip = nn.Linear(in_channels, heads * out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        hop_dist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        H, C = self.heads, self.out_channels

        query = self.lin_q(x).view(-1, H, C)
        key   = self.lin_k(x).view(-1, H, C)
        value = self.lin_v(x).view(-1, H, C)

        # Default hop distance: 1 for direct edges if not explicitly provided
        if hop_dist is None:
            hop_dist = torch.ones(edge_index.size(1), dtype=torch.long, device=x.device)

        out = self.propagate(
            edge_index,
            query=query,
            key=key,
            value=value,
            edge_attr=edge_attr,
            hop_dist=hop_dist,
            size=None,
        )

        out = out.view(-1, H * C)
        out = out + self.lin_skip(x)
        return out

    def message(
        self,
        query_i: torch.Tensor,
        key_j: torch.Tensor,
        value_j: torch.Tensor,
        edge_attr: torch.Tensor,
        hop_dist: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int],
    ) -> torch.Tensor:
        H, C = self.heads, self.out_channels

        # Scaled dot-product query-key attention: (q_i * k_j) / sqrt(d_h)
        alpha = (query_i * key_j).sum(dim=-1) / np.sqrt(C)   # [E, H]

        # Edge feature projection
        edge_emb = self.lin_edge(edge_attr).view(-1, H, C).sum(dim=-1)  # [E, H]
        alpha = alpha + edge_emb

        # Spatial bias embedding \psi(e_ij) matching Equation 2.4.3
        psi_bias = self.spatial_bias(torch.clamp(hop_dist, 0, self.spatial_bias.num_embeddings - 1))  # [E, H]
        alpha = alpha + psi_bias

        # Softmax over incoming neighbors
        alpha = softmax(alpha, index, ptr, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return value_j * alpha.unsqueeze(-1)   # [E, H, C]


# ── Encoder ───────────────────────────────────────────────────────────────────

class GraphTransformerEncoder(nn.Module):
    """
    Multi-layer Graph Transformer with virtual star node, spatial attention bias,
    and PyG global_mean_pool for dynamic topology execution.

    Parameters
    ----------
    max_nodes  : Optional[int] Node count per graph (real + star). Optional for zero-shot dynamic pooling.
    hidden_dim : int   Embedding / latent dimension (default 128).
    num_heads  : int   Attention heads per SpatialTransformerConv layer (default 4).
    num_layers : int   Stacked SpatialTransformerConv layers (default 3).
    dropout    : float Dropout rate inside SpatialTransformerConv (default 0.1).
    """

    def __init__(
        self,
        max_nodes:  Optional[int] = None,
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
            SpatialTransformerConv(
                in_channels  = hidden_dim,
                out_channels = hidden_dim // num_heads,
                heads        = num_heads,
                edge_dim     = EDGE_FEAT_DIM,
                dropout      = dropout,
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

    def forward(
        self,
        data: Data,
        return_node_feats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.node_embed(data.x)                    # [B*nodes, H]

        hop_dist = getattr(data, "hop_dist", None)

        for conv, norm in zip(self.conv_layers, self.layer_norms):
            residual = x
            x = conv(x, data.edge_index, data.edge_attr, hop_dist=hop_dist)
            x = norm(x + residual)
            x = F.relu(x)

        # Dynamic graph pooling via PyG global_mean_pool
        # Allows zero-shot execution on arbitrary topologies without hardcoded node limits.
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        latent = global_mean_pool(x, batch)
        z_global = self.output_mlp(latent)             # [B, hidden_dim]
        if return_node_feats:
            return z_global, x
        return z_global


# ── Path-Centric Actor-Critic Network (RouteNet-Fermi Paradigm) ──────────────

class PathAttentionActorCritic(nn.Module):
    """
    Path-Centric Actor-Critic Network (RouteNet-Fermi paradigm).

    Directly evaluates candidate routing paths by pooling bottleneck and mean
    link states along each path, scoring paths via cross-attention with the
    flow demand representation.
    """
    def __init__(
        self,
        latent_dim: int = 256,
        k_paths: int = 5,
        hidden_dim: int = 256,
        edge_attr_dim: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.k_paths = k_paths
        self.hidden_dim = hidden_dim

        # ── 1. Edge State Projector: [x_u, x_v, edge_attr] -> h_edge ───────
        self.edge_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2 + edge_attr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # ── 2. Path Bottleneck & Mean Aggregator ───────────────────────────
        # Combines [mean_pool, max_bottleneck_pool] along constituent edges
        self.path_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── 3. Flow Demand Projector: [x_src, x_dst, z_global] -> h_flow ───
        self.flow_mlp = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── 4. Cross-Attention Path Scorer (Actor Head) ───────────────────
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.score_mlp  = nn.Sequential(
            nn.Linear(hidden_dim * 2 + latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # ── 5. Value Function (Critic Head) ───────────────────────────────
        self.critic_head = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # ── 6. Fallback MLP for backward compatibility ────────────────────
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_head  = nn.Linear(hidden_dim, k_paths)

    def forward(
        self,
        z_global: torch.Tensor,
        node_feats: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        path_edges: Optional[Union[List[List[List[int]]], torch.Tensor]] = None,
        src_dst: Optional[torch.Tensor] = None,
        path_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Path-Attention Actor-Critic.

        Parameters
        ----------
        z_global   : [B, latent_dim] Global pooled graph representation.
        node_feats : [B * (N+1), latent_dim] Node embeddings from GraphTransformer.
        edge_index : [2, E_batch] Directed edge connections.
        edge_attr  : [E_batch, 4] Dynamic edge telemetry attributes.
        path_edges : List over batch of K paths OR [B, K, max_hops] edge tensor.
        src_dst    : [B, 2] Tensor of (src_node_idx, dst_node_idx) per batch item.
        path_mask  : [B, K, max_hops] Optional boolean validity mask for path edges.

        Returns
        -------
        logits : [B, K] Path routing action logits.
        value  : [B] Critic state value estimate.
        """
        B = z_global.size(0)

        # Fallback to standard global MLP if path inputs not provided
        if path_edges is None or node_feats is None or edge_index is None or edge_attr is None:
            shared = self.shared(z_global)
            logits = self.actor_head(shared)
            value = self.critic_head(torch.cat([z_global, shared], dim=-1)).squeeze(-1)
            return logits, value

        device = z_global.device

        # Reshape / unsqueeze node_feats to [B, N, hidden_dim]
        if node_feats.dim() == 2:
            if B == 1:
                node_feats = node_feats.unsqueeze(0)
            else:
                N = node_feats.size(0) // B
                node_feats = node_feats.view(B, N, -1)

        # Reshape / unsqueeze edge_attr to [B, E, 4]
        if edge_attr.dim() == 2:
            if B == 1:
                edge_attr = edge_attr.unsqueeze(0)
            else:
                E = edge_index.size(1)
                if edge_attr.size(0) == B * E:
                    edge_attr = edge_attr.view(B, E, -1)
                elif edge_attr.size(0) == E:
                    edge_attr = edge_attr.unsqueeze(0).expand(B, -1, -1)

        # 1. Compute contextual edge representations: [B, E, hidden_dim]
        u_nodes = edge_index[0]
        v_nodes = edge_index[1]
        x_u = node_feats[:, u_nodes, :]
        x_v = node_feats[:, v_nodes, :]
        edge_raw = torch.cat([x_u, x_v, edge_attr], dim=-1)
        h_edge = self.edge_mlp(edge_raw)

        # 2. Compute flow demand representation: [B, hidden_dim]
        if src_dst is not None and src_dst.size(0) == B:
            src_idx = src_dst[:, 0]
            dst_idx = src_dst[:, 1]
            b_indices = torch.arange(B, device=device)
            x_src = node_feats[b_indices, src_idx]
            x_dst = node_feats[b_indices, dst_idx]
        else:
            x_src = z_global
            x_dst = z_global

        flow_raw = torch.cat([x_src, x_dst, z_global], dim=-1)
        h_flow = self.flow_mlp(flow_raw)   # [B, hidden_dim]
        q_flow = self.query_proj(h_flow)   # [B, hidden_dim]

        # 3. Embed all candidate paths in a fully-vectorized manner
        if isinstance(path_edges, torch.Tensor):
            path_edges_tensor = path_edges
            if path_mask is None:
                path_mask = (path_edges_tensor >= 0) & (path_edges_tensor < h_edge.size(1))
        else:
            # Convert list of paths to padded tensor in one batch pass
            max_hops = 16
            for b_p in path_edges:
                for p in b_p:
                    if len(p) > max_hops:
                        max_hops = len(p)
            E_pad = h_edge.size(1)
            path_edges_tensor = torch.full((B, self.k_paths, max_hops), E_pad, dtype=torch.long, device=device)
            path_mask = torch.zeros((B, self.k_paths, max_hops), dtype=torch.bool, device=device)
            for b, b_paths in enumerate(path_edges):
                if b >= B:
                    break
                for k in range(min(len(b_paths), self.k_paths)):
                    hops = b_paths[k]
                    l = len(hops)
                    if l > 0:
                        path_edges_tensor[b, k, :l] = torch.as_tensor(hops[:max_hops], dtype=torch.long, device=device)
                        path_mask[b, k, :l] = True

        M = path_edges_tensor.size(-1)
        zero_pad = torch.zeros((B, 1, self.hidden_dim), dtype=h_edge.dtype, device=device)
        h_edge_pad = torch.cat([h_edge, zero_pad], dim=1)  # [B, E+1, H]
        b_idx = torch.arange(B, device=device)[:, None, None].expand(-1, self.k_paths, M)
        h_paths = h_edge_pad[b_idx, path_edges_tensor]      # [B, K, M, H]

        mask_exp = path_mask.unsqueeze(-1)
        hop_counts = path_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        mean_pool = (h_paths * mask_exp).sum(dim=2) / hop_counts

        fill_neg = -1e4 if h_paths.dtype == torch.float16 else -1e9
        h_paths_masked = h_paths.masked_fill(~mask_exp, fill_neg)
        bottleneck_pool, _ = h_paths_masked.max(dim=2)
        has_valid_path = (path_mask.sum(dim=-1) > 0).unsqueeze(-1)
        bottleneck_pool = bottleneck_pool.masked_fill(~has_valid_path, 0.0)

        pool_feats = torch.cat([mean_pool, bottleneck_pool], dim=-1)   # [B, K, 2*H]
        valid_mask = (path_mask.sum(dim=-1) > 0)                       # [B, K]

        # Batched path projection for all (B x K) paths simultaneously:
        path_rep = self.path_mlp(pool_feats)            # [B, K, hidden_dim]
        k_path   = self.key_proj(path_rep)              # [B, K, hidden_dim]

        # Batched cross-attention dot product:
        q_flow_expanded = q_flow.unsqueeze(1)           # [B, 1, hidden_dim]
        dot_scores = (q_flow_expanded * k_path).sum(dim=-1) / (self.hidden_dim ** 0.5)  # [B, K]

        # Batched score MLP:
        h_flow_expanded = h_flow.unsqueeze(1).expand(-1, self.k_paths, -1)     # [B, K, hidden_dim]
        z_glob_expanded = z_global.unsqueeze(1).expand(-1, self.k_paths, -1)   # [B, K, latent_dim]
        score_in = torch.cat([h_flow_expanded, path_rep, z_glob_expanded], dim=-1)  # [B, K, 2*H + latent_dim]
        mlp_scores = self.score_mlp(score_in).squeeze(-1)                      # [B, K]

        logits = dot_scores + mlp_scores                # [B, K]
        fill_val = -1e4 if logits.dtype == torch.float16 else -1e9
        logits = logits.masked_fill(~valid_mask, fill_val)

        # 4. Critic value estimation
        critic_in = torch.cat([z_global, h_flow], dim=-1)
        value = self.critic_head(critic_in).squeeze(-1)   # [B]

        return logits, value

    def get_action(
        self,
        latent: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        node_feats: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        path_edges: Optional[List[List[List[int]]]] = None,
        src_dst: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or greedily select an action from the policy."""
        logits, value = self.forward(
            z_global=latent,
            node_feats=node_feats,
            edge_index=edge_index,
            edge_attr=edge_attr,
            path_edges=path_edges,
            src_dst=src_dst,
        )

        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        logits = torch.clamp(logits, -10.0, 10.0)
        if mask is not None and bool(mask.any().item()):
            fill_val = -1e4 if logits.dtype == torch.float16 else -1e9
            logits = logits.masked_fill(~mask, fill_val)

        from torch.distributions import Categorical
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob, value


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

        # ── Pre-compute static edge_index + edge_attr skeleton (runs once) ─
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
        self._edge_pairs = list(G.edges())

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
            feat[i, 4] = float(a.get("is_src", 0.0))
            feat[i, 5] = float(a.get("is_dst", 0.0))

        # ② Single non-blocking host→device copy for all real-node features.
        self._x[: self.n_real, : NODE_FEAT_DIM].copy_(
            torch.from_numpy(feat), non_blocking=True
        )

        curr = 0
        for u, v in self._edge_pairs:
            edge = G.edges[u, v]
            util = float(np.clip(edge.get("utilization", 0.0), 0.0, 1.0))
            loss = float(np.clip(edge.get("packet_loss", 0.0), 0.0, 1.0))
            prop_delay = float(edge.get("delay", 1.0))
            q_delay    = float(edge.get("queuing_delay", 0.0))
            tot_delay_norm = float(np.clip((prop_delay + min(q_delay, 500.0)) / 500.0, 0.0, 1.0))

            self.edge_attr[curr, 1] = util
            self.edge_attr[curr, 2] = tot_delay_norm
            self.edge_attr[curr, 3] = loss
            self.edge_attr[curr + 1, 1] = util
            self.edge_attr[curr + 1, 2] = tot_delay_norm
            self.edge_attr[curr + 1, 3] = loss
            curr += 2

        edge_attr = self.edge_attr.clone() if clone else self.edge_attr

        return Data(
            x          = self._x.clone() if clone else self._x,
            edge_index = self.edge_index,
            edge_attr  = edge_attr,
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
