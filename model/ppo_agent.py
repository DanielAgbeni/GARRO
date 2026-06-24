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

Key features
------------
* Invalid-path masking: logits for non-existent paths set to −inf.
* Gradient norm clipping (max_norm=0.5) prevents exploding updates.
* Entropy bonus encourages exploration during early training.
* Rollout buffer stores NetworkX graph snapshots for correct gradient flow
  through the encoder during the PPO update phase.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Batch, Data

from model.graph_transformer import GraphTransformerEncoder, nx_to_pyg


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
        self.actor_head  = nn.Linear(hidden_dim, k_paths)  # logits over K paths
        self.critic_head = nn.Linear(hidden_dim, 1)         # state value V(s)

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
            # Set logits for invalid / non-existent paths to -inf
            logits = logits.masked_fill(~mask, float("-inf"))

        dist     = Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


# ── Rollout Buffer ────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Stores one complete rollout for a single PPO update cycle.

    States are stored as deep-copied NetworkX graphs (not tensors) because
    the Graph Transformer needs to re-encode them with gradients enabled
    during the PPO update pass.
    """

    def __init__(self):
        self.states:    List[nx.Graph] = []
        self.actions:   List[int]      = []
        self.log_probs: List[float]    = []
        self.rewards:   List[float]    = []
        self.values:    List[float]    = []
        self.dones:     List[bool]     = []

    def add(
        self,
        state,
        action:   int,
        log_prob: float,
        reward:   float,
        value:    float,
        done:     bool,
    ):
        if isinstance(state, Data):
            self.states.append(state.clone())
        else:
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
    Caches structural properties and writes dynamic attributes on CPU
    before moving to GPU in a single copy operation to prevent CUDA stalls.
    """
    def __init__(self, G: nx.Graph, device: torch.device):
        self.device = device
        self.nodes = sorted(G.nodes())
        self.n_real = len(self.nodes)
        self.idx_map = {n: i for i, n in enumerate(self.nodes)}
        
        # Pre-allocate node features on CPU
        self.x_cpu = torch.zeros((self.n_real + 1, 5), dtype=torch.float32, device="cpu")
        self.x_cpu[self.n_real, 4] = 1.0  # Star node flag = 1.0, others = 0.0
        
        self.edges = list(G.edges())
        self.n_edges = len(self.edges)
        
        self.edge_indices = []
        for u, v in self.edges:
            self.edge_indices.append((self.idx_map[u], self.idx_map[v]))
            
        n_total_edges = 2 * self.n_edges + 2 * self.n_real
        
        # Keep edge index on target device (never changes)
        self.edge_index = torch.zeros((2, n_total_edges), dtype=torch.long, device=device)
        curr = 0
        for u_idx, v_idx in self.edge_indices:
            self.edge_index[0, curr] = u_idx
            self.edge_index[1, curr] = v_idx
            self.edge_index[0, curr+1] = v_idx
            self.edge_index[1, curr+1] = u_idx
            curr += 2
            
        star_idx = self.n_real
        for i in range(self.n_real):
            self.edge_index[0, curr] = star_idx
            self.edge_index[1, curr] = i
            self.edge_index[0, curr+1] = i
            self.edge_index[1, curr+1] = star_idx
            curr += 2
            
        # Pre-allocate edge attributes on CPU
        self.edge_attr_cpu = torch.zeros((n_total_edges, 4), dtype=torch.float32, device="cpu")
        
        # Fill static edge attributes (bandwidth and delays) on CPU
        curr = 0
        for u, v in self.edges:
            d = G.edges[u, v]
            bw_norm = float(np.clip(d.get("bandwidth", 1000) / 10_000.0, 0.0, 1.0))
            delay_norm = float(np.clip(d.get("delay", 1.0) / 100.0, 0.0, 1.0))
            self.edge_attr_cpu[curr, 0] = bw_norm
            self.edge_attr_cpu[curr, 2] = delay_norm
            self.edge_attr_cpu[curr+1, 0] = bw_norm
            self.edge_attr_cpu[curr+1, 2] = delay_norm
            curr += 2
            
        # Star edge attributes: [1.0, 0.0, 0.0, 0.0]
        for i in range(self.n_real):
            self.edge_attr_cpu[curr, 0] = 1.0
            self.edge_attr_cpu[curr+1, 0] = 1.0
            curr += 2

    def convert(self, G: nx.Graph) -> Data:
        """Build the Data object on CPU and move to GPU in one operation."""
        cpu = nx.get_node_attributes(G, 'cpu')
        buf = nx.get_node_attributes(G, 'buffer_occ')
        ing = nx.get_node_attributes(G, 'ingress_rate')
        egr = nx.get_node_attributes(G, 'egress_rate')
        
        # Fast write to CPU tensor
        for idx, n in enumerate(self.nodes):
            self.x_cpu[idx, 0] = cpu.get(n, 0.5)
            self.x_cpu[idx, 1] = buf.get(n, 0.3)
            self.x_cpu[idx, 2] = ing.get(n, 0.5)
            self.x_cpu[idx, 3] = egr.get(n, 0.5)
            
        util = nx.get_edge_attributes(G, 'utilization')
        loss = nx.get_edge_attributes(G, 'packet_loss')
        
        curr = 0
        for u, v in self.edges:
            ut = util.get((u, v), util.get((v, u), 0.0))
            ls = loss.get((u, v), loss.get((v, u), 0.0))
            
            self.edge_attr_cpu[curr, 1] = ut
            self.edge_attr_cpu[curr, 3] = ls
            self.edge_attr_cpu[curr+1, 1] = ut
            self.edge_attr_cpu[curr+1, 3] = ls
            curr += 2
            
        # Push completed tensors to device in single transfers
        x_gpu = self.x_cpu.to(self.device, non_blocking=True)
        edge_attr_gpu = self.edge_attr_cpu.to(self.device, non_blocking=True)
            
        return Data(x=x_gpu.clone(), edge_index=self.edge_index, edge_attr=edge_attr_gpu.clone())


# ── PPO Agent ─────────────────────────────────────────────────────────────────

class PPOAgent:
    """
    Full GARRO PPO agent: Graph Transformer encoder + Actor-Critic.

    Parameters
    ----------
    config   : dict           Full config.yaml contents.
    k_paths  : int            Action-space size (number of candidate paths).
    device   : torch.device   Computation device.
    """

    def __init__(
        self,
        config:  dict,
        k_paths: int,
        device:  torch.device = torch.device("cpu"),
    ):
        self.config   = config
        self.k_paths  = k_paths
        self.device   = device

        ppo_cfg = config["ppo"]
        gt_cfg  = config["graph_transformer"]

        # ── Build encoder + actor-critic ──────────────────────────────────
        self.encoder = GraphTransformerEncoder(
            hidden_dim=gt_cfg["hidden_dim"],
            num_heads=gt_cfg["num_heads"],
            num_layers=gt_cfg["num_layers"],
            dropout=gt_cfg["dropout"],
        ).to(device)

        self.ac_net = ActorCriticNetwork(
            latent_dim=gt_cfg["hidden_dim"],
            k_paths=k_paths,
            hidden_dim=256,
        ).to(device)

        # ── Optimisers ────────────────────────────────────────────────────
        # Encoder + actor share lr_actor; critic head uses lr_critic.
        self.opt_encoder = torch.optim.Adam(
            self.encoder.parameters(), lr=ppo_cfg["lr_actor"]
        )
        self.opt_ac = torch.optim.Adam([
            {"params": self.ac_net.actor_head.parameters(),  "lr": ppo_cfg["lr_actor"]},
            {"params": self.ac_net.critic_head.parameters(), "lr": ppo_cfg["lr_critic"]},
            {"params": self.ac_net.shared.parameters(),      "lr": ppo_cfg["lr_actor"]},
        ])

        # ── Hyperparameters ───────────────────────────────────────────────
        self.gamma        = ppo_cfg["gamma"]
        self.gae_lambda   = ppo_cfg["gae_lambda"]
        self.clip_eps     = ppo_cfg["clip_epsilon"]
        self.update_epochs = ppo_cfg["update_epochs"]
        self.batch_size   = ppo_cfg["batch_size"]
        self.entropy_coef = ppo_cfg["entropy_coef"]

        self.buffer = RolloutBuffer()

        # Automatic Mixed Precision for hardware acceleration (Tensor Cores on GPU)
        self.enable_autocast = (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.enable_autocast)
        self.graph_converter = None
        self.last_pyg = None

    # ── Inference ─────────────────────────────────────────────────────────────

    def _encode(self, graph: nx.Graph) -> torch.Tensor:
        """Convert NetworkX graph → latent state vector [1, hidden_dim]."""
        if self.graph_converter is None:
            self.graph_converter = FastGraphConverter(graph, self.device)
            
        pyg = self.graph_converter.convert(graph)
        self.last_pyg = pyg
        
        pyg.batch = torch.zeros(
            pyg.x.size(0), dtype=torch.long, device=self.device
        )
        with torch.autocast(device_type=self.device.type, enabled=self.enable_autocast):
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

        # Build validity mask: True for paths that actually exist
        mask = torch.zeros(self.k_paths, dtype=torch.bool, device=self.device)
        for i in range(min(len(candidate_paths), self.k_paths)):
            mask[i] = True

        with torch.autocast(device_type=self.device.type, enabled=self.enable_autocast):
            action, log_prob, value = self.ac_net.get_action(latent, mask)
        return int(action.item()), float(log_prob.item()), float(value.item())

    # ── PPO Update ────────────────────────────────────────────────────────────

    def update(self) -> Dict[str, float]:
        """
        Run one complete PPO update cycle over the current rollout buffer.

        Steps:
        1. Compute GAE advantages and discounted returns.
        2. Re-encode all stored graph states (with gradients).
        3. For `update_epochs` passes: mini-batch PPO-Clip updates.
        4. Clear the rollout buffer.

        Returns
        -------
        dict of mean training metrics: policy_loss, value_loss, entropy, approx_kl
        """
        if len(self.buffer) == 0:
            return {}

        T = len(self.buffer)
        actions      = torch.tensor(self.buffer.actions,   dtype=torch.long,    device=self.device)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)

        # ── GAE Advantage Estimation ─────────────────────────────────────
        advantages: List[float] = []
        gae        = 0.0
        next_value = 0.0

        for t in reversed(range(T)):
            delta = (
                self.buffer.rewards[t]
                + self.gamma * next_value * (1.0 - float(self.buffer.dones[t]))
                - self.buffer.values[t]
            )
            gae = (
                delta
                + self.gamma * self.gae_lambda
                * (1.0 - float(self.buffer.dones[t]))
                * gae
            )
            advantages.insert(0, gae)
            next_value = self.buffer.values[t]

        adv_tensor     = torch.tensor(advantages,          dtype=torch.float32, device=self.device)
        values_tensor  = torch.tensor(self.buffer.values,  dtype=torch.float32, device=self.device)
        returns_tensor = adv_tensor + values_tensor

        # Normalise advantages for stable gradients
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        # ── Pre-encode all states (detached) — FAST, no grad, done once ─────
        # States are stored directly as Data objects in the buffer to avoid redundant conversions
        graph_states = self.buffer.states
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, enabled=self.enable_autocast):
                pyg_list = []
                for pyg in graph_states:
                    if not isinstance(pyg, Data):
                        if self.graph_converter is None:
                            self.graph_converter = FastGraphConverter(pyg, self.device)
                        pyg = self.graph_converter.convert(pyg)
                    pyg_list.append(pyg)
                batched_pyg = Batch.from_data_list(pyg_list).to(self.device)
                all_latents_det = self.encoder(batched_pyg)   # [T, hidden_dim] detached

        # ── AC-Net PPO Mini-Batch Updates (no encoder grad — fast) ───────
        metrics: Dict[str, List[float]] = {
            "policy_loss": [], "value_loss": [],
            "entropy": [], "approx_kl": [],
        }

        for _ in range(self.update_epochs):
            indices = torch.randperm(T, device=self.device)
            for start in range(0, T, self.batch_size):
                idx = indices[start: start + self.batch_size]

                b_latents = all_latents_det[idx]   # no grad through encoder
                b_actions = actions[idx]
                b_old_lp  = old_log_probs[idx]
                b_adv     = adv_tensor[idx]
                b_returns = returns_tensor[idx]

                with torch.autocast(device_type=self.device.type, enabled=self.enable_autocast):
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

        # ── Encoder Update: single gradient pass on a random subset ───────
        enc_size = min(self.batch_size, T)
        enc_idx  = torch.randperm(T, device=self.device)[:enc_size]

        with torch.autocast(device_type=self.device.type, enabled=self.enable_autocast):
            enc_pyg_list = []
            for i in enc_idx.tolist():
                pyg = graph_states[i]
                if not isinstance(pyg, Data):
                    if self.graph_converter is None:
                        self.graph_converter = FastGraphConverter(pyg, self.device)
                    pyg = self.graph_converter.convert(pyg)
                enc_pyg_list.append(pyg)
            batched_enc_pyg = Batch.from_data_list(enc_pyg_list).to(self.device)
            enc_latents = self.encoder(batched_enc_pyg)
            logits_e, vals_e = self.ac_net(enc_latents)
            dist_e    = Categorical(logits=logits_e)
            new_lp_e  = dist_e.log_prob(actions[enc_idx])
            ent_e     = dist_e.entropy().mean()
            ratio_e   = torch.exp(new_lp_e - old_log_probs[enc_idx])
            adv_e     = adv_tensor[enc_idx]
            ret_e     = returns_tensor[enc_idx]
            surr1_e   = ratio_e * adv_e
            surr2_e   = torch.clamp(
                ratio_e, 1.0 - self.clip_eps, 1.0 + self.clip_eps
            ) * adv_e
            enc_loss  = (
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
        """Save encoder + actor-critic weights to a .pt checkpoint."""
        torch.save({
            "encoder": self.encoder.state_dict(),
            "ac_net":  self.ac_net.state_dict(),
        }, path)
        print(f"[PPO] Checkpoint saved → {path}")

    def load(self, path: str):
        """Load encoder + actor-critic weights from a .pt checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.ac_net.load_state_dict(ckpt["ac_net"])
        print(f"[PPO] Checkpoint loaded ← {path}")
