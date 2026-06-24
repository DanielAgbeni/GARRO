"""
Digital Twin — M/M/1/K Queuing Gymnasium Environment.

Models the SDN network as a collection of M/M/1/K queues, one per
switch interface. Used exclusively for Phase 1 offline PPO training.
No live network, Mininet, or OS-Ken controller is required.

MDP formulation
---------------
State  : Flat vector of normalised node + edge features.
         (The Graph Transformer handles variable-size graph input;
          this env exposes a fixed-size Box space for the baseline MLP
          and provides the live NetworkX graph for the GT encoder.)
Action : Integer ∈ [0, K-1] selecting one of the pre-computed
         K-shortest candidate paths for the current (src, dst) demand.
Reward : Multi-objective function as defined in the GARRO MDP:
         R = α1·(T_actual/T_req) − α2·D_path − α3·L_packet − α4·σ²_util
"""
import copy
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
try:
    # Standard import path
    from gymnasium.core import Env as GymEnv
    from gymnasium import spaces
except (ImportError, AttributeError):
    # Fallback for non-standard installs
    import gym as gymnasium_fallback
    GymEnv = gymnasium_fallback.Env
    from gymnasium_fallback import spaces


# ── Helper: M/M/1/K steady-state metrics ─────────────────────────────────────

def mm1k_metrics(lam: float, mu: float, K: int
                 ) -> Tuple[float, float, float]:
    """
    Compute M/M/1/K queue steady-state metrics in closed form.

    Parameters
    ----------
    lam : float  Packet arrival rate λ (packets/s)
    mu  : float  Packet service rate μ (packets/s)
    K   : int    Maximum buffer capacity (packets)

    Returns
    -------
    E_Q         : Expected queue length (packets)
    P_overflow  : Probability of buffer overflow (≈ packet loss probability)
    mean_delay  : Average per-packet queuing delay (ms) via Little's Law
    """
    if mu <= 0:
        return float(K), 1.0, float("inf")

    rho = lam / mu
    eps = 1e-9

    if abs(rho - 1.0) < eps:
        # Special case: ρ = 1  →  uniform distribution over [0, K]
        P0         = 1.0 / (K + 1)
        P_overflow = P0
        E_Q        = K / 2.0
    else:
        denom = 1.0 - rho ** (K + 1)
        if abs(denom) < eps:
            return float(K), 1.0, float("inf")
        P0         = (1.0 - rho) / denom
        P_overflow = P0 * (rho ** K)
        # E[Q] = ρ·[1 − (K+1)ρ^K + K·ρ^(K+1)] / [(1−ρ)·(1−ρ^(K+1))]
        numerator  = rho * (1.0 - (K + 1) * rho ** K + K * rho ** (K + 1))
        E_Q        = numerator / ((1.0 - rho) * denom)

    # Little's Law: W = E[Q] / λ_eff     (convert to ms)
    lam_eff    = lam * (1.0 - P_overflow)
    mean_delay = (E_Q / lam_eff * 1_000) if lam_eff > eps else float("inf")

    return (
        float(np.clip(E_Q, 0.0, K)),
        float(np.clip(P_overflow, 0.0, 1.0)),
        float(mean_delay),
    )


# ── Environment ───────────────────────────────────────────────────────────────

class MM1KNetworkEnv(GymEnv):
    """
    Gymnasium environment modelling an SDN network using M/M/1/K queuing.

    Key design decisions
    --------------------
    * The observation returned by reset() / step() is a **flat numpy vector**
      for compatibility with tabular/MLP baselines and Ray RLlib.
    * The PPO+GraphTransformer agent accesses `env.G` (the live NetworkX graph)
      directly via `agent.select_action(env.G, env.candidate_paths)`.
    * K-shortest paths are pre-computed at init for all (src, dst) pairs so
      that no expensive graph search occurs during the training hot-loop.
    * Traffic is re-injected at every step via a Modulated Gravity model that
      produces bursty, non-stationary arrival rates (microburst factors 1/2/5).
    """

    metadata = {"render_modes": []}

    def __init__(self, graph: nx.Graph, config: dict):
        super().__init__()
        self.G       = graph.copy()
        self.cfg     = config
        self.K       = config["network"]["k_paths"]
        self.K_buf   = config["mm1k"]["buffer_capacity"]
        self.base_lam = config["mm1k"]["base_arrival_rate"]
        self.base_mu  = config["mm1k"]["base_service_rate"]
        self.alpha    = [
            config["reward_weights"]["alpha1"],
            config["reward_weights"]["alpha2"],
            config["reward_weights"]["alpha3"],
            config["reward_weights"]["alpha4"],
        ]

        self.num_nodes  = self.G.number_of_nodes()
        self.num_edges  = self.G.number_of_edges()
        self.edges_list: List[Tuple[int, int]] = list(self.G.edges())

        # ── Pre-compute K-shortest paths ──────────────────────────────────
        # Uses delay as the path weight so the "shortest" path is the
        # lowest-latency path (OSPF-equivalent default).
        self._all_paths: Dict[Tuple[int, int], List[List[int]]] = {}
        print(f"[MM1KEnv] Pre-computing K={self.K} shortest paths "
              f"for {self.num_nodes} nodes …", flush=True)
        self._precompute_paths()
        print("[MM1KEnv] Path pre-computation done.", flush=True)

        # ── Current episode state ─────────────────────────────────────────
        self.current_src: int        = 0
        self.current_dst: int        = 1
        self.candidate_paths: List   = []
        self.step_count: int         = 0
        self.max_steps: int          = config["training"]["max_steps_per_episode"]

        # Per-edge traffic rates (updated each step)
        self.lambda_map: Dict[Tuple[int, int], float] = {}
        self.mu_map:     Dict[Tuple[int, int], float] = {}

        # ── Gymnasium spaces ──────────────────────────────────────────────
        # Node: 4 features × num_nodes  |  Edge: 4 features × num_edges
        node_feat_dim = self.num_nodes * 4
        edge_feat_dim = self.num_edges * 4
        obs_dim       = node_feat_dim + edge_feat_dim

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.K)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _precompute_paths(self):
        """Pre-compute up to K shortest (by delay) simple paths for every pair."""
        nodes = list(self.G.nodes())
        for src in nodes:
            for dst in nodes:
                if src == dst:
                    continue
                try:
                    paths = list(nx.shortest_simple_paths(
                        self.G, src, dst, weight="delay"
                    ))
                    self._all_paths[(src, dst)] = paths[: self.K]
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    self._all_paths[(src, dst)] = []

    def _simulate_traffic(self):
        """
        Inject Modulated Gravity traffic onto all edges.

        Burst model:
            burst_factor ~ Categorical([1.0, 2.0, 5.0], p=[0.70, 0.20, 0.10])
            λ_edge = base_λ × burst_factor × Uniform(0.5, 1.5)
            μ_edge = base_μ × Uniform(0.8, 1.2)

        Updates edge attributes:
            utilization  ← ρ = λ / μ   (clipped to [0, 1])
            packet_loss  ← P_overflow from M/M/1/K
        """
        for u, v in self.edges_list:
            burst = float(np.random.choice(
                [1.0, 2.0, 5.0], p=[0.70, 0.20, 0.10]
            ))
            lam = self.base_lam * burst * np.random.uniform(0.5, 1.5)
            mu  = self.base_mu  * np.random.uniform(0.8, 1.2)

            self.lambda_map[(u, v)] = lam
            self.mu_map[(u, v)]     = mu

            _, P_overflow, _ = mm1k_metrics(lam, mu, self.K_buf)

            self.G.edges[u, v]["utilization"] = float(
                np.clip(lam / (mu + 1e-9), 0.0, 1.0)
            )
            self.G.edges[u, v]["packet_loss"] = P_overflow

    def _get_obs(self) -> np.ndarray:
        """
        Build the flat observation vector from current graph state.

        Node features (per node): [cpu, buffer_occ, ingress_rate, egress_rate]
        Edge features (per edge): [bw_norm, utilization, delay_norm, packet_loss]
        All values normalised to [0.0, 1.0].
        """
        node_feats: List[float] = []
        for n in sorted(self.G.nodes()):
            attrs = self.G.nodes[n]
            # In the digital twin we simulate node stats with mild noise
            cpu        = float(np.clip(attrs.get("cpu", 0.5)
                                        + np.random.normal(0, 0.05), 0.0, 1.0))
            buf_occ    = float(np.clip(attrs.get("buffer_occ", 0.3)
                                        + np.random.normal(0, 0.05), 0.0, 1.0))
            ingress    = float(np.clip(attrs.get("ingress_rate", 0.5)
                                        + np.random.normal(0, 0.05), 0.0, 1.0))
            egress     = float(np.clip(attrs.get("egress_rate", 0.5)
                                        + np.random.normal(0, 0.05), 0.0, 1.0))
            node_feats.extend([cpu, buf_occ, ingress, egress])

        edge_feats: List[float] = []
        for u, v in self.edges_list:
            d = self.G.edges[u, v]
            bw_norm   = float(np.clip(d.get("bandwidth", 1000) / 10_000.0, 0.0, 1.0))
            util      = float(np.clip(d.get("utilization", 0.0), 0.0, 1.0))
            delay_norm = float(np.clip(d.get("delay", 1.0) / 100.0, 0.0, 1.0))
            pkt_loss  = float(np.clip(d.get("packet_loss", 0.0), 0.0, 1.0))
            edge_feats.extend([bw_norm, util, delay_norm, pkt_loss])

        return np.array(node_feats + edge_feats, dtype=np.float32)

    def _compute_reward(self, path: List[int]) -> float:
        """
        Compute the multi-objective GARRO reward for a selected path.

        R = α1·(T_actual / T_req) − α2·D_norm − α3·L_pkt − α4·σ²_util

        Parameters
        ----------
        path : List[int]  Sequence of node IDs representing the chosen route.

        Returns
        -------
        float  Scalar reward in approx. range [−1.0, +0.5].
        """
        if not path or len(path) < 2:
            return -10.0  # Severe penalty for invalid/empty path

        path_edges = list(zip(path[:-1], path[1:]))

        total_delay  = 0.0
        total_loss   = 0.0   # Combined loss along path (series model)
        min_bw       = float("inf")

        for u, v in path_edges:
            # Edges are undirected — try both directions
            edge = self.G.edges.get((u, v)) or self.G.edges.get((v, u))
            if edge is None:
                return -10.0  # Edge not found — path is invalid

            lam = self.lambda_map.get((u, v),
                  self.lambda_map.get((v, u), self.base_lam))
            mu  = self.mu_map.get((u, v),
                  self.mu_map.get((v, u), self.base_mu))

            _, P_overflow, delay_ms = mm1k_metrics(lam, mu, self.K_buf)

            prop_delay   = edge.get("delay", 1.0)
            total_delay += prop_delay + min(delay_ms, 500.0)  # cap queuing at 500ms

            # Series packet loss: P_end = 1 − Π(1 − P_loss_i)
            total_loss = 1.0 - (1.0 - total_loss) * (1.0 - P_overflow)
            min_bw     = min(min_bw, edge.get("bandwidth", 1_000))

        # ── Global utilisation variance (penalises unbalanced loads) ──────
        all_utils = [
            self.G.edges[u, v].get("utilization", 0.0)
            for u, v in self.edges_list
        ]
        util_variance = float(np.var(all_utils))

        # ── Throughput ratio ───────────────────────────────────────────────
        T_actual = min_bw * (1.0 - total_loss)
        T_req    = min_bw                          # Demand = full link capacity
        tput_ratio = T_actual / (T_req + 1e-9)

        # ── Normalise delay (target < 50 ms → normalise over 500 ms range) ─
        delay_norm = min(total_delay / 500.0, 1.0)

        reward = (
              self.alpha[0] * tput_ratio
            - self.alpha[1] * delay_norm
            - self.alpha[2] * total_loss
            - self.alpha[3] * util_variance
        )
        return float(reward)

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.step_count = 0

        # Pick a random (src, dst) flow demand for this episode
        nodes = list(self.G.nodes())
        pair  = self.np_random.choice(len(nodes), size=2, replace=False)
        self.current_src = int(nodes[pair[0]])
        self.current_dst = int(nodes[pair[1]])
        self.candidate_paths = self._all_paths.get(
            (self.current_src, self.current_dst), []
        )

        # If no paths exist between the sampled pair, fall back to first pair
        # with available paths (should rarely happen in connected topologies)
        if not self.candidate_paths:
            for (s, d), paths in self._all_paths.items():
                if paths:
                    self.current_src, self.current_dst = s, d
                    self.candidate_paths = paths
                    break

        self._simulate_traffic()
        obs  = self._get_obs()
        info = {"src": self.current_src, "dst": self.current_dst}
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.step_count += 1

        # Map discrete action index → actual node path
        if action < len(self.candidate_paths):
            selected_path = self.candidate_paths[action]
        else:
            # Out-of-range action → fall back to shortest path
            selected_path = (
                self.candidate_paths[0] if self.candidate_paths else []
            )

        reward = self._compute_reward(selected_path)

        # Apply utilisation increase along the selected path edges
        for u, v in zip(selected_path[:-1], selected_path[1:]):
            for key in [(u, v), (v, u)]:
                if key in self.G.edges:
                    self.G.edges[key]["utilization"] = float(
                        np.clip(self.G.edges[key]["utilization"] + 0.05, 0.0, 1.0)
                    )

        # Advance traffic simulation
        self._simulate_traffic()

        terminated = self.step_count >= self.max_steps
        truncated  = False
        obs        = self._get_obs()
        info       = {
            "path":   selected_path,
            "src":    self.current_src,
            "dst":    self.current_dst,
            "reward": reward,
        }
        return obs, reward, terminated, truncated, info

    def get_graph_copy(self) -> nx.Graph:
        """Return a deep copy of the current NetworkX graph (for buffer storage)."""
        return copy.deepcopy(self.G)
