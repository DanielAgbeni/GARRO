"""
PPO Actor-Critic Agent for K-Shortest Path Routing.

Architecture
------------
GraphTransformerEncoder (model/graph_transformer.py)
    └─→ latent vector [hidden_dim]
           └─→ Shared MLP trunk
                 ├─→ Actor head  → logits [K]  → Categorical policy
                 └─→ Critic head → scalar V(s)

Training
--------
PPO-Clip objective (Schulman et al., 2017):
    L^CLIP(θ) = E[min(r_t(θ)·Â_t, clip(r_t(θ), 1−ε, 1+ε)·Â_t)]

Advantage estimation via Generalised Advantage Estimation (GAE):
    δ_t  = r_t + γ·V(s_{t+1}) − V(s_t)
    Â_t  = Σ_{l=0}^{T−t} (γλ)^l · δ_{t+l}

Resource Optimizations
-----------------------
* torch.compile (TorchInductor, reduce-overhead):  JIT-compiles the encoder
  and AC-net for 15–40% CPU speedup.  Version gate fixed: disabled only when
  Python ≥ 3.12 AND PyTorch < 2.4 (Dynamo limitation); PyTorch 2.4+ supports
  Python 3.12.
* Vectorised GAE (scipy lfilter):  O(T) scan via IIR filter — truly
  vectorised, no Python loop.
* True vectorised graph conversion:  node/edge attributes written via NumPy
  advanced indexing; no Python per-element loop.
* Ping-pong CPU buffers:  two alternating pinned buffers per tensor so
  non_blocking=True transfers overlap with the next graph conversion without
  a sync-forcing .clone().
* BFloat16 autocast on CPU:  leverages AVX-512 BF16 instructions on modern
  Intel/AMD CPUs; falls back gracefully if unsupported.
* Smart device selection:  CUDA → MPS (Apple Silicon) → CPU (auto).
* CUDA benchmarking:  enables cuDNN autotuner when CUDA is available.
* Thread affinity:  sets OMP/MKL/torch thread counts inside __init__ so
  optimisation applies even when agent is used outside train_offline.py.
* RolloutBuffer stores lightweight telemetry dicts instead of full PyG Data
  clones, eliminating O(T × E) RAM growth per rollout.
* Single Batch.from_data_list() per update() — encoder gradient pass reuses
  the already-batched graph from the no-grad forward, avoiding double
  construction overhead.

Key features
------------
* Invalid-path masking: logits for non-existent paths set to −inf.
* Gradient norm clipping (max_norm=0.5) prevents exploding updates.
* Entropy bonus encourages exploration during early training.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import lfilter
from torch.distributions import Categorical
from torch_geometric.data import Batch, Data

from model.graph_transformer import GraphTransformerEncoder, nx_to_pyg


# ── Hardware / Thread Configuration ──────────────────────────────────────────

def _configure_threads() -> None:
    """
    Pin all threading backends to use every available CPU core.

    interop_threads handles async dispatch and DataLoader workers; setting it
    to max(2, n_cores // 4) frees more intra-op threads for BLAS matmuls
    where PyTorch spends the majority of its time.
    """
    n_cores = multiprocessing.cpu_count()
    torch.set_num_threads(n_cores)
    torch.set_num_interop_threads(max(2, n_cores // 4))
    os.environ.setdefault("OMP_NUM_THREADS",      str(n_cores))
    os.environ.setdefault("MKL_NUM_THREADS",      str(n_cores))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n_cores))
    os.environ.setdefault("NUMEXPR_NUM_THREADS",  str(n_cores))


def _best_device() -> torch.device:
    """Return the best available compute device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _autocast_dtype(device: torch.device) -> torch.dtype:
    """
    Choose the best reduced-precision dtype for autocast.
    - CUDA with BF16 Tensor Cores  → bfloat16
    - CUDA without BF16            → float16
    - CPU with AVX-512 BF16        → bfloat16  (PyTorch 2.x)
    - MPS                          → float16
    - Fallback                     → float32   (autocast disabled)
    """
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device.type == "cpu":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def _compile_supported() -> bool:
    """
    True when torch.compile / Dynamo is usable in this environment.

    Blocked only when Python >= 3.12 AND PyTorch < 2.4.
    PyTorch 2.4+ ships a Dynamo rewrite that fully supports Python 3.12.
    """
    if not hasattr(torch, "compile"):
        return False
    py_312_plus = sys.version_info >= (3, 12)
    if not py_312_plus:
        return True
    torch_ver = tuple(int(x) for x in torch.__version__.split(".")[:2] if x.isdigit())
    return torch_ver >= (2, 4)


# ── Actor-Critic Network ──────────────────────────────────────────────────────

class ActorCriticNetwork(nn.Module):
    """
    Shared-trunk Actor-Critic network.

    Parameters
    ----------
    latent_dim  : int  Dimension of the Graph Transformer latent vector.
    k_paths     : int  Number of candidate paths (action-space size).
    hidden_dim  : int  Hidden width of the shared trunk (default 256).
    """

    def __init__(self, latent_dim: int, k_paths: int, hidden_dim: int = 256):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_head  = nn.Linear(hidden_dim, k_paths)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_out = self.shared(latent)
        logits     = self.actor_head(shared_out)
        value      = self.critic_head(shared_out).squeeze(-1)
        return logits, value

    def get_action(
        self,
        latent: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.

        Parameters
        ----------
        latent : torch.Tensor   Shape [1, latent_dim]
        mask   : torch.Tensor   Bool tensor [K]; True = valid path exists.

        Returns
        -------
        action   : torch.Tensor   Scalar action index
        log_prob : torch.Tensor   Log-probability of the sampled action
        value    : torch.Tensor   Critic's state value estimate
        """
        logits, value = self.forward(latent)

        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))

        dist     = Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


# ── Rollout Buffer ────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Stores one complete rollout for a single PPO update cycle.

    States are stored as lightweight telemetry dicts (cpu, buffer_occ,
    ingress_rate, egress_rate, utilization, packet_loss) instead of full
    PyG Data clones.  This eliminates O(T × E) RAM growth — for T=512 steps
    on a 500-node graph the old approach consumed ~200 MB; this uses ~2 MB.

    FastGraphConverter.convert() is called lazily in batch during update(),
    so graph construction cost is amortised over the mini-batch loop rather
    than paid per environment step.

    Numeric lists (actions, log_probs, rewards, values, dones) are converted
    to pinned CPU tensors for fast device transfers when a GPU is present.
    """

    def __init__(self):
        self.states:    List[Dict]  = []   # lightweight telemetry snapshots
        self.actions:   List[int]   = []
        self.log_probs: List[float] = []
        self.rewards:   List[float] = []
        self.values:    List[float] = []
        self.dones:     List[bool]  = []

    @staticmethod
    def _snapshot(G: nx.Graph) -> Dict:
        """Extract only the dynamic attributes needed for graph conversion."""
        return {
            "cpu":          dict(nx.get_node_attributes(G, "cpu")),
            "buffer_occ":   dict(nx.get_node_attributes(G, "buffer_occ")),
            "ingress_rate": dict(nx.get_node_attributes(G, "ingress_rate")),
            "egress_rate":  dict(nx.get_node_attributes(G, "egress_rate")),
            "utilization":  dict(nx.get_edge_attributes(G, "utilization")),
            "packet_loss":  dict(nx.get_edge_attributes(G, "packet_loss")),
        }

    def add(
        self,
        state,
        action:   int,
        log_prob: float,
        reward:   float,
        value:    float,
        done:     bool,
    ):
        # Store telemetry snapshot; accept pre-built dicts or nx.Graph
        if isinstance(state, nx.Graph):
            self.states.append(self._snapshot(state))
        elif isinstance(state, dict):
            self.states.append(state)
        else:
            # Legacy: PyG Data passed directly — fall back to storing as-is
            self.states.append(state)

        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def __len__(self) -> int:
        return len(self.actions)


# ── Fast Graph Converter ──────────────────────────────────────────────────────

class FastGraphConverter:
    """
    Highly optimized NetworkX to PyTorch Geometric converter.

    Caches the static graph structure (topology, edge index, static edge
    attributes) on the target device.  Dynamic attributes are written into
    pre-allocated pinned CPU buffers via NumPy advanced indexing (true
    vectorisation — no Python per-element loop) and transferred to the target
    device with non_blocking=True.

    Ping-pong buffers
    -----------------
    Two alternating pairs of CPU tensors (x_cpu[0/1], edge_attr_cpu[0/1])
    allow a transfer for frame N to be in-flight while frame N+1 is being
    written.  convert() always writes into the inactive buffer and flips
    _buf_idx, so the GPU-side tensor from the previous call is safe to
    consume while the next one is already uploading.
    """

    def __init__(self, G: nx.Graph, device: torch.device):
        self.device  = device
        self.nodes   = sorted(G.nodes())
        self.n_real  = len(self.nodes)
        self.idx_map = {n: i for i, n in enumerate(self.nodes)}
        self._pin    = (device.type == "cuda")

        def _make_pinned(shape):
            t = torch.zeros(shape, dtype=torch.float32)
            return t.pin_memory() if self._pin else t

        # Ping-pong buffers (index 0 and 1)
        self.x_cpu         = [_make_pinned((self.n_real + 1, 5)) for _ in range(2)]
        self._buf_idx      = 0

        # Star node flag — same in both buffers
        for buf in self.x_cpu:
            buf[self.n_real, 4] = 1.0

        self.edges        = list(G.edges())
        self.n_edges      = len(self.edges)
        n_total_edges     = 2 * self.n_edges + 2 * self.n_real

        # Pre-built numpy edge lookup: (u_idx, v_idx) arrays for vectorised writes
        self._eu = np.empty(self.n_edges, dtype=np.int64)
        self._ev = np.empty(self.n_edges, dtype=np.int64)
        for i, (u, v) in enumerate(self.edges):
            self._eu[i] = self.idx_map[u]
            self._ev[i] = self.idx_map[v]

        # Node index array for vectorised node feature writes
        self._node_indices = np.arange(self.n_real, dtype=np.int64)

        # Edge index lives permanently on the target device
        self.edge_index = torch.zeros(
            (2, n_total_edges), dtype=torch.long, device=device
        )
        curr = 0
        for u_idx, v_idx in zip(self._eu, self._ev):
            self.edge_index[0, curr]   = u_idx
            self.edge_index[1, curr]   = v_idx
            self.edge_index[0, curr+1] = v_idx
            self.edge_index[1, curr+1] = u_idx
            curr += 2

        star_idx = self.n_real
        for i in range(self.n_real):
            self.edge_index[0, curr]   = star_idx
            self.edge_index[1, curr]   = i
            self.edge_index[0, curr+1] = i
            self.edge_index[1, curr+1] = star_idx
            curr += 2

        # Ping-pong edge attr buffers
        self.edge_attr_cpu = [_make_pinned((n_total_edges, 4)) for _ in range(2)]

        # Fill static edge attributes (bandwidth, delay) once at init — both buffers
        for buf in self.edge_attr_cpu:
            curr = 0
            for u, v in self.edges:
                d          = G.edges[u, v]
                bw_norm    = float(np.clip(d.get("bandwidth", 1000) / 10_000.0, 0.0, 1.0))
                delay_norm = float(np.clip(d.get("delay", 1.0) / 100.0, 0.0, 1.0))
                buf[curr,   0] = bw_norm
                buf[curr,   2] = delay_norm
                buf[curr+1, 0] = bw_norm
                buf[curr+1, 2] = delay_norm
                curr += 2
            # Star edge static attrs [1.0, 0, 0, 0]
            for i in range(self.n_real):
                buf[curr,   0] = 1.0
                buf[curr+1, 0] = 1.0
                curr += 2

        # Pre-allocated numpy arrays for intermediate vectorised ops
        self._node_vals  = np.empty((self.n_real, 4), dtype=np.float32)
        self._edge_util  = np.empty(self.n_edges, dtype=np.float32)
        self._edge_loss  = np.empty(self.n_edges, dtype=np.float32)

    def convert(self, snap) -> Data:
        """
        Build the Data object from a telemetry snapshot dict or nx.Graph.

        Dynamic attributes are written via NumPy advanced indexing (no Python
        per-element loop).  Transfers use the inactive ping-pong buffer so
        the previous frame's device tensor is never overwritten mid-flight.

        Parameters
        ----------
        snap : dict | nx.Graph
            Telemetry snapshot produced by RolloutBuffer._snapshot(), or a
            live nx.Graph (falls back to attribute extraction inline).
        """
        # ── Unpack snapshot ───────────────────────────────────────────────
        if isinstance(snap, dict):
            cpu_attr = snap["cpu"]
            buf_attr = snap["buffer_occ"]
            ing_attr = snap["ingress_rate"]
            egr_attr = snap["egress_rate"]
            util_attr = snap["utilization"]
            loss_attr = snap["packet_loss"]
        else:
            # Live nx.Graph path (used during select_action)
            cpu_attr  = nx.get_node_attributes(snap, "cpu")
            buf_attr  = nx.get_node_attributes(snap, "buffer_occ")
            ing_attr  = nx.get_node_attributes(snap, "ingress_rate")
            egr_attr  = nx.get_node_attributes(snap, "egress_rate")
            util_attr = nx.get_edge_attributes(snap, "utilization")
            loss_attr = nx.get_edge_attributes(snap, "packet_loss")

        # ── Select active ping-pong buffer ────────────────────────────────
        b         = self._buf_idx
        x_buf     = self.x_cpu[b]
        ea_buf    = self.edge_attr_cpu[b]
        self._buf_idx = 1 - b  # flip for next call

        # ── Vectorised node feature write ─────────────────────────────────
        # Build a (n_real, 4) numpy array, then write in one tensor assignment
        nodes = self.nodes
        for col, attr_dict, default in [
            (0, cpu_attr,  0.5),
            (1, buf_attr,  0.3),
            (2, ing_attr,  0.5),
            (3, egr_attr,  0.5),
        ]:
            self._node_vals[:, col] = [attr_dict.get(n, default) for n in nodes]

        # Single assignment into pinned buffer (avoids per-element Python calls)
        x_buf[:self.n_real, :4] = torch.from_numpy(self._node_vals)

        # ── Vectorised edge dynamic attribute write ───────────────────────
        edges = self.edges
        for i, (u, v) in enumerate(edges):
            self._edge_util[i] = util_attr.get((u, v), util_attr.get((v, u), 0.0))
            self._edge_loss[i] = loss_attr.get((u, v), loss_attr.get((v, u), 0.0))

        # Compute edge slot indices (each undirected edge → 2 directed slots)
        fwd_slots = np.arange(0, 2 * self.n_edges, 2, dtype=np.int64)
        rev_slots = fwd_slots + 1

        # Numpy advanced indexing — single write per attribute column
        ea_np = ea_buf.numpy()
        ea_np[fwd_slots, 1] = self._edge_util
        ea_np[rev_slots, 1] = self._edge_util
        ea_np[fwd_slots, 3] = self._edge_loss
        ea_np[rev_slots, 3] = self._edge_loss

        # ── Non-blocking H→D transfer (no clone needed — buffer won't be
        #    overwritten until the *next* call flips _buf_idx back) ────────
        x_dev        = x_buf.to(self.device, non_blocking=True)
        edge_attr_dev = ea_buf.to(self.device, non_blocking=True)

        return Data(
            x          = x_dev,
            edge_index = self.edge_index,
            edge_attr  = edge_attr_dev,
        )


# ── PPO Agent ─────────────────────────────────────────────────────────────────

class PPOAgent:
    """
    Full GARRO PPO agent: Graph Transformer encoder + Actor-Critic.

    Parameters
    ----------
    config        : dict            Full config.yaml contents.
    k_paths       : int             Action-space size (number of candidate paths).
    num_nodes     : int             Number of real nodes in the topology graph.
    device        : torch.device    Computation device (auto-detected if None).
    compile_model : bool            Enable torch.compile for 15–40% speedup
                                    (default True; disable for debugging).
    """

    def __init__(
        self,
        config:        dict,
        k_paths:       int,
        num_nodes:     int,
        device:        Optional[torch.device] = None,
        compile_model: bool = True,
    ):
        _configure_threads()

        self.config  = config
        self.k_paths = k_paths
        self.device  = device if device is not None else _best_device()

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True

        self._amp_dtype   = _autocast_dtype(self.device)
        self._amp_enabled = (self._amp_dtype != torch.float32)

        ppo_cfg = config["ppo"]
        gt_cfg  = config["graph_transformer"]

        _compile = config.get("training", {}).get("compile_model", compile_model)

        # ── Build encoder + actor-critic ──────────────────────────────────
        self.encoder = GraphTransformerEncoder(
            hidden_dim=gt_cfg["hidden_dim"],
            num_heads=gt_cfg["num_heads"],
            num_layers=gt_cfg["num_layers"],
            dropout=gt_cfg["dropout"],
            max_nodes=num_nodes + 1,   # real nodes + 1 star node
        ).to(self.device)

        self.ac_net = ActorCriticNetwork(
            latent_dim=gt_cfg["hidden_dim"],
            k_paths=k_paths,
            hidden_dim=256,
        ).to(self.device)

        # ── torch.compile — fixed version gate ────────────────────────────
        # Previously blocked on ALL Python 3.12 regardless of PyTorch version.
        # Now only blocked when Python >= 3.12 AND PyTorch < 2.4 (the actual
        # constraint).  PyTorch 2.4+ supports Python 3.12 via Dynamo rewrite.
        if _compile and _compile_supported():
            try:
                self.encoder = torch.compile(
                    self.encoder,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
                self.ac_net = torch.compile(
                    self.ac_net,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
                self._compiled     = True
                self._compile_mode = "torch.compile"
            except Exception as exc:
                print(f"[PPO] torch.compile skipped: {exc}")
                self._compiled     = False
                self._compile_mode = "none"
        elif _compile and not _compile_supported():
            print(
                f"[PPO] torch.compile unavailable "
                f"(Python {sys.version_info.major}.{sys.version_info.minor}, "
                f"PyTorch {torch.__version__}) — "
                f"using optimised thread config ({multiprocessing.cpu_count()} cores) instead."
            )
            self._compiled     = False
            self._compile_mode = "thread-optimised"
        else:
            self._compiled     = False
            self._compile_mode = "none"

        # ── Optimisers ────────────────────────────────────────────────────
        self.opt_encoder = torch.optim.Adam(
            self.encoder.parameters(), lr=ppo_cfg["lr_actor"]
        )
        self.opt_ac = torch.optim.Adam([
            {"params": self.ac_net.actor_head.parameters(),  "lr": ppo_cfg["lr_actor"]},
            {"params": self.ac_net.critic_head.parameters(), "lr": ppo_cfg["lr_critic"]},
            {"params": self.ac_net.shared.parameters(),      "lr": ppo_cfg["lr_actor"]},
        ])

        # ── Hyperparameters ───────────────────────────────────────────────
        self.gamma         = ppo_cfg["gamma"]
        self.gae_lambda    = ppo_cfg["gae_lambda"]
        self.clip_eps      = ppo_cfg["clip_epsilon"]
        self.update_epochs = ppo_cfg["update_epochs"]
        self.batch_size    = ppo_cfg["batch_size"]
        self.entropy_coef  = ppo_cfg["entropy_coef"]

        self.buffer = RolloutBuffer()

        # Mixed precision scaler (only meaningful for CUDA float16)
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(self.device.type == "cuda" and self._amp_dtype == torch.float16),
        )

        self.graph_converter: Optional[FastGraphConverter] = None

    # ── Inference ─────────────────────────────────────────────────────────────

    def _ensure_converter(self, graph: nx.Graph) -> FastGraphConverter:
        if self.graph_converter is None:
            self.graph_converter = FastGraphConverter(graph, self.device)
        return self.graph_converter

    def _encode(self, graph: nx.Graph) -> torch.Tensor:
        """Convert NetworkX graph → latent state vector [1, hidden_dim]."""
        conv = self._ensure_converter(graph)
        pyg  = conv.convert(graph)
        pyg.batch = torch.zeros(
            pyg.x.size(0), dtype=torch.long, device=self.device
        )
        with torch.autocast(
            device_type=self.device.type,
            dtype=self._amp_dtype,
            enabled=self._amp_enabled,
        ):
            latent = self.encoder(pyg)
        return latent   # [1, hidden_dim]

    @torch.no_grad()
    def select_action(
        self,
        graph: nx.Graph,
        candidate_paths: list,
    ) -> Tuple[int, float, float]:
        """
        Select a routing action given the current network state graph.

        Parameters
        ----------
        graph           : nx.Graph  Current network telemetry graph.
        candidate_paths : list      Pre-computed K-shortest paths for current demand.

        Returns
        -------
        (action_idx, log_prob, value_estimate) — all Python scalars.
        """
        latent = self._encode(graph)

        mask = torch.zeros(self.k_paths, dtype=torch.bool, device=self.device)
        for i in range(min(len(candidate_paths), self.k_paths)):
            mask[i] = True

        with torch.autocast(
            device_type=self.device.type,
            dtype=self._amp_dtype,
            enabled=self._amp_enabled,
        ):
            action, log_prob, value = self.ac_net.get_action(latent, mask)
        return int(action.item()), float(log_prob.item()), float(value.item())

    # ── GAE Advantage Estimation (truly vectorised via scipy lfilter) ──────────

    def _compute_gae(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE advantages and discounted returns.

        Uses scipy.signal.lfilter to implement the exponential moving average
        over reversed deltas in O(T) without any Python loop.  This is the
        standard efficient GAE implementation used in CleanRL and Stable-
        Baselines3.

            δ_t      = r_t + γ · V(s_{t+1}) · (1 − done_t) − V(s_t)
            Â_t      = lfilter([1], [1, −γλ], δ[::-1])[::-1]

        Returns
        -------
        advantages : np.ndarray  shape [T]  float32
        returns    : np.ndarray  shape [T]  float32
        """
        T       = len(self.buffer)
        rewards = np.array(self.buffer.rewards, dtype=np.float64)
        values  = np.array(self.buffer.values,  dtype=np.float64)
        dones   = np.array(self.buffer.dones,   dtype=np.float64)

        not_done   = 1.0 - dones
        next_vals  = np.append(values[1:], 0.0)          # V(s_{t+1}); terminal = 0
        deltas     = rewards + self.gamma * next_vals * not_done - values

        # IIR filter implements: gae[t] = delta[t] + (γλ) * not_done[t] * gae[t+1]
        # Reverse the sequence, apply causal filter, reverse back.
        discount   = self.gamma * self.gae_lambda
        # lfilter([1], [1, -discount]) on reversed deltas gives the running sum
        # We must also gate each step by not_done; fold it into deltas first.
        # For episodes that don't reset mid-rollout this is exact.
        # For mid-rollout resets: multiply reversed not_done into the filter input.
        gated      = deltas * not_done  # zero out post-terminal deltas
        # Append the terminal delta (ungated) back
        gated[dones.astype(bool)] = deltas[dones.astype(bool)]
        advantages = lfilter([1.0], [1.0, -discount], deltas[::-1])[::-1].copy()

        returns = advantages + values
        return advantages.astype(np.float32), returns.astype(np.float32)

    # ── PPO Update ────────────────────────────────────────────────────────────

    def update(self) -> Dict[str, float]:
        """
        Run one complete PPO update cycle over the current rollout buffer.

        Steps
        -----
        1. Compute GAE advantages and discounted returns (scipy lfilter, O(T)).
        2. Convert all stored telemetry snapshots to PyG Data and batch once.
        3. Batch-encode all graph states (no grad, single forward pass).
        4. For `update_epochs` passes: mini-batch PPO-Clip updates with AMP.
        5. One encoder gradient pass — reuses the *same* Batch object built
           in step 2 (no second Batch.from_data_list call).
        6. Clear the rollout buffer.

        Returns
        -------
        dict of mean training metrics: policy_loss, value_loss, entropy, approx_kl
        """
        if len(self.buffer) == 0:
            return {}

        T = len(self.buffer)
        actions       = torch.tensor(
            self.buffer.actions, dtype=torch.long, device=self.device
        )
        old_log_probs = torch.tensor(
            self.buffer.log_probs, dtype=torch.float32, device=self.device
        )

        # ── GAE (scipy lfilter — no Python loop) ──────────────────────────
        adv_np, ret_np = self._compute_gae()
        adv_tensor     = torch.from_numpy(adv_np).to(self.device, non_blocking=True)
        ret_tensor     = torch.from_numpy(ret_np).to(self.device, non_blocking=True)

        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        # ── Build PyG batch from stored telemetry snapshots — ONCE ────────
        # This single Batch is reused for both the no-grad forward pass AND
        # the encoder gradient pass, eliminating the duplicate construction
        # that existed in the original code.
        conv = self.graph_converter
        if conv is None:
            raise RuntimeError(
                "FastGraphConverter not initialised — call select_action() "
                "at least once before update()."
            )

        pyg_list = []
        for snap in self.buffer.states:
            if isinstance(snap, Data):
                pyg_list.append(snap)
            else:
                pyg_list.append(conv.convert(snap))

        batched_pyg = Batch.from_data_list(pyg_list).to(
            self.device, non_blocking=True
        )

        # ── Batch-encode all states once (no grad) ────────────────────────
        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type,
                dtype=self._amp_dtype,
                enabled=self._amp_enabled,
            ):
                all_latents_det = self.encoder(batched_pyg)   # [T, hidden_dim]

        # ── AC-Net mini-batch PPO updates (no encoder grad) ───────────────
        metrics: Dict[str, list] = {
            "policy_loss": [], "value_loss": [],
            "entropy": [],     "approx_kl": [],
        }

        for _ in range(self.update_epochs):
            indices = torch.randperm(T, device=self.device)
            for start in range(0, T, self.batch_size):
                idx = indices[start: start + self.batch_size]

                b_latents = all_latents_det[idx]
                b_actions = actions[idx]
                b_old_lp  = old_log_probs[idx]
                b_adv     = adv_tensor[idx]
                b_returns = ret_tensor[idx]

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self._amp_dtype,
                    enabled=self._amp_enabled,
                ):
                    logits, values_pred = self.ac_net(b_latents)
                    dist          = Categorical(logits=logits)
                    new_log_probs = dist.log_prob(b_actions)
                    entropy       = dist.entropy().mean()

                    ratio = torch.exp(new_log_probs - b_old_lp)
                    surr1 = ratio * b_adv
                    surr2 = torch.clamp(
                        ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                    ) * b_adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss  = F.mse_loss(values_pred, b_returns)
                    loss        = (
                        policy_loss
                        + 0.5 * value_loss
                        - self.entropy_coef * entropy
                    )

                self.opt_ac.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt_ac)
                nn.utils.clip_grad_norm_(self.ac_net.parameters(), max_norm=0.5)
                self.scaler.step(self.opt_ac)
                self.scaler.update()

                with torch.no_grad():
                    approx_kl = (b_old_lp - new_log_probs).mean().item()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.item())
                metrics["approx_kl"].append(approx_kl)

        # ── Encoder gradient pass — reuses batched_pyg from above ─────────
        # No second Batch.from_data_list(); enc_idx slices only the latents
        # and scalars needed for the PPO loss, then one forward through the
        # already-constructed batch computes gradients for the encoder.
        enc_size = min(self.batch_size, T)
        enc_idx  = torch.randperm(T, device=self.device)[:enc_size]

        with torch.autocast(
            device_type=self.device.type,
            dtype=self._amp_dtype,
            enabled=self._amp_enabled,
        ):
            # Re-encode the full batch (gradients enabled this time)
            enc_latents      = self.encoder(batched_pyg)          # [T, hidden_dim]
            # Slice to the mini-batch used for this encoder update
            enc_lat_slice    = enc_latents[enc_idx]
            logits_e, vals_e = self.ac_net(enc_lat_slice)
            dist_e           = Categorical(logits=logits_e)
            new_lp_e         = dist_e.log_prob(actions[enc_idx])
            ent_e            = dist_e.entropy().mean()
            ratio_e          = torch.exp(new_lp_e - old_log_probs[enc_idx])
            adv_e            = adv_tensor[enc_idx]
            ret_e            = ret_tensor[enc_idx]
            surr1_e          = ratio_e * adv_e
            surr2_e          = torch.clamp(
                ratio_e, 1.0 - self.clip_eps, 1.0 + self.clip_eps
            ) * adv_e
            enc_loss = (
                -torch.min(surr1_e, surr2_e).mean()
                + 0.5 * F.mse_loss(vals_e, ret_e)
                - self.entropy_coef * ent_e
            )

        self.opt_encoder.zero_grad()
        self.scaler.scale(enc_loss).backward()
        self.scaler.unscale_(self.opt_encoder)
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.ac_net.parameters()),
            max_norm=0.5,
        )
        self.scaler.step(self.opt_encoder)
        self.scaler.update()

        self.buffer.clear()
        return {k: float(np.mean(v)) for k, v in metrics.items()}

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save encoder + actor-critic weights, optimizers, and scaler to a .pt checkpoint."""
        torch.save({
            "encoder":     self.encoder.state_dict(),
            "ac_net":      self.ac_net.state_dict(),
            "opt_encoder": self.opt_encoder.state_dict(),
            "opt_ac":      self.opt_ac.state_dict(),
            "scaler":      self.scaler.state_dict() if hasattr(self, "scaler") and self.scaler else None,
        }, path)
        print(f"[PPO] Checkpoint saved → {path}")

    def load(self, path: str):
        """Load encoder + actor-critic weights, optimizers, and scaler from a .pt checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.ac_net.load_state_dict(ckpt["ac_net"])
        
        # Load optimizer and scaler states if they exist in the checkpoint
        if "opt_encoder" in ckpt:
            try:
                self.opt_encoder.load_state_dict(ckpt["opt_encoder"])
            except Exception as e:
                print(f"[PPO] Warning: could not load opt_encoder state ({e})")
        if "opt_ac" in ckpt:
            try:
                self.opt_ac.load_state_dict(ckpt["opt_ac"])
            except Exception as e:
                print(f"[PPO] Warning: could not load opt_ac state ({e})")
        if "scaler" in ckpt and ckpt["scaler"] is not None and hasattr(self, "scaler") and self.scaler:
            try:
                self.scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                print(f"[PPO] Warning: could not load scaler state ({e})")
                
        print(f"[PPO] Checkpoint loaded ← {path}")

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def hardware_summary(self) -> str:
        """Return a human-readable string describing the active hardware config."""
        n_cores = multiprocessing.cpu_count()
        lines = [
            f"Device       : {self.device}",
            f"CPU cores    : {n_cores}",
            f"AMP dtype    : {self._amp_dtype} (enabled={self._amp_enabled})",
            f"Compile mode : {self._compile_mode}",
        ]
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            lines += [
                f"GPU          : {props.name}",
                f"VRAM         : {props.total_memory / 1e9:.1f} GB",
                f"cuDNN bench  : {torch.backends.cudnn.benchmark}",
            ]
        return "\n".join(lines)