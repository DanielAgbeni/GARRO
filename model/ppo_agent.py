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
from torch_geometric.data import Data

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
        state:    nx.Graph,
        action:   int,
        log_prob: float,
        reward:   float,
        value:    float,
        done:     bool,
    ):
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

    # ── Inference ─────────────────────────────────────────────────────────────

    def _encode(self, graph: nx.Graph) -> torch.Tensor:
        """Convert NetworkX graph → latent state vector [1, hidden_dim]."""
        pyg  = nx_to_pyg(graph, self.device)
        pyg.batch = torch.zeros(
            pyg.x.size(0), dtype=torch.long, device=self.device
        )
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
        # The encoder is updated separately via a single gradient pass below.
        # This avoids double-backward and keeps mini-batch updates O(T) not O(T×epochs).
        graph_states = self.buffer.states
        with torch.no_grad():
            det_list: List[torch.Tensor] = []
            for graph_snap in graph_states:
                pyg = nx_to_pyg(graph_snap, self.device)
                pyg.batch = torch.zeros(
                    pyg.x.size(0), dtype=torch.long, device=self.device
                )
                det_list.append(self.encoder(pyg))
        all_latents_det = torch.cat(det_list, dim=0)   # [T, hidden_dim] detached

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
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac_net.parameters(), max_norm=0.5)
                self.opt_ac.step()

                with torch.no_grad():
                    approx_kl = (b_old_lp - new_log_probs).mean().item()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.item())
                metrics["approx_kl"].append(approx_kl)

        # ── Encoder Update: single gradient pass on a random subset ───────
        enc_size = min(self.batch_size, T)
        enc_idx  = torch.randperm(T, device=self.device)[:enc_size]

        enc_lat_list: List[torch.Tensor] = []
        for i in enc_idx.tolist():
            pyg = nx_to_pyg(graph_states[i], self.device)
            pyg.batch = torch.zeros(
                pyg.x.size(0), dtype=torch.long, device=self.device
            )
            enc_lat_list.append(self.encoder(pyg))   # grad tracked here

        enc_latents = torch.cat(enc_lat_list, dim=0)
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
        enc_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.ac_net.parameters()),
            max_norm=0.5,
        )
        self.opt_encoder.step()

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
