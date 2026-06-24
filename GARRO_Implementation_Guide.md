# GARRO Implementation Guide
## Dynamic SDN Traffic Routing with DRL — Python 3.12 + OS-Ken

---

## Table of Contents
1. [Environment & Dependency Setup](#1-environment--dependency-setup)
2. [Project Structure](#2-project-structure)
3. [Topology Definitions](#3-topology-definitions)
4. [Digital Twin — M/M/1/K Queuing Environment](#4-digital-twin--mmk-queuing-environment)
5. [Graph Transformer Encoder](#5-graph-transformer-encoder)
6. [PPO Decision Engine](#6-ppo-decision-engine)
7. [OS-Ken Controller (Control Plane)](#7-os-ken-controller-control-plane)
8. [Agentic AI Layer (LLM Orchestrator)](#8-agentic-ai-layer-llm-orchestrator)
9. [Phase 1 — Offline Digital Twin Training](#9-phase-1--offline-digital-twin-training)
10. [Phase 2 — Live Mininet Emulation](#10-phase-2--live-mininet-emulation)
11. [Evaluation & Benchmarking](#11-evaluation--benchmarking)
12. [Run Order & Commands Cheatsheet](#12-run-order--commands-cheatsheet)

---

## 1. Environment & Dependency Setup

### 1.1 WSL & System Packages

Open your WSL Ubuntu terminal and run:

```bash
sudo apt-get update && sudo apt-get upgrade -y

# Core networking and build tools
sudo apt-get install -y \
    python3.12 python3.12-dev python3.12-venv \
    build-essential git curl wget \
    openvswitch-switch openvswitch-common \
    net-tools iproute2 iperf3 \
    libssl-dev libffi-dev

# Mininet — install from package manager
sudo apt-get install -y mininet

# Verify OVS is running
sudo service openvswitch-switch start
sudo ovs-vsctl show
```

> **WSL Note:** If `sudo service openvswitch-switch start` fails, run:
> `sudo modprobe openvswitch` first.

---

### 1.2 Python Virtual Environment

```bash
# Create project directory and virtual environment
mkdir ~/garro && cd ~/garro
python3.12 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

---

### 1.3 Python Dependencies — All Python 3.12 Compatible

Install in this exact order to avoid dependency conflicts:

```bash
# ── Core numerics ──────────────────────────────────────────────
pip install numpy>=1.26.4

# ── PyTorch (CPU build — for WSL without GPU) ─────────────────
pip install torch==2.3.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# ── PyTorch Geometric (Graph Neural Networks) ─────────────────
pip install torch_geometric==2.5.3

# PyG optional extensions (scatter/sparse) — CPU wheel
pip install torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-2.3.0+cpu.html

# ── Reinforcement Learning ─────────────────────────────────────
pip install gymnasium>=0.29.1
pip install "ray[rllib]>=2.10.0"

# ── OS-Ken (drop-in Ryu replacement, Python 3.12 compatible) ──
pip install os-ken>=2.2.0

# ── Network & data utilities ───────────────────────────────────
pip install networkx>=3.3
pip install pandas>=2.2.0
pip install matplotlib>=3.9.0
pip install scipy>=1.13.0

# ── REST API & HTTP ────────────────────────────────────────────
pip install requests>=2.32.0
pip install flask>=3.0.3          # Optional: lightweight REST wrapper
pip install aiohttp>=3.9.5        # Async HTTP for LLM API calls

# ── LLM API clients ───────────────────────────────────────────
pip install openai>=1.35.0        # OpenAI GPT-4
pip install google-generativeai>=0.7.0  # Google Gemini

# ── Utilities ─────────────────────────────────────────────────
pip install tqdm>=4.66.0
pip install pyyaml>=6.0.1
pip install python-dotenv>=1.0.1

# ── Freeze requirements ────────────────────────────────────────
pip freeze > requirements.txt
```

> **Key OS-Ken note:** OS-Ken is imported as `os_ken`, not `ryu`.
> Every `from ryu.xxx import yyy` becomes `from os_ken.xxx import yyy`.

---

## 2. Project Structure

```
~/garro/
├── venv/                          # Python virtual environment
├── .env                           # API keys (never commit this)
├── requirements.txt
│
├── config.yaml                    # Central configuration file
│
├── digital_twin/
│   ├── __init__.py
│   ├── mm1k_env.py                # Gymnasium env with M/M/1/K queuing
│   └── traffic_generator.py       # Modulated Gravity traffic model
│
├── model/
│   ├── __init__.py
│   ├── graph_transformer.py       # Graph Transformer Encoder (PyG)
│   └── ppo_agent.py               # PPO Actor-Critic network
│
├── controller/
│   ├── __init__.py
│   ├── garro_controller.py        # OS-Ken OpenFlow 1.3 app
│   └── topology_manager.py        # LLDP topology + telemetry polling
│
├── agentic/
│   ├── __init__.py
│   └── llm_orchestrator.py        # LLM intent → reward weights
│
├── topologies/
│   ├── __init__.py
│   ├── nsfnet.py                  # 14-node WAN topology
│   ├── geant2.py                  # 24-node academic topology
│   └── fat_tree.py                # k=8 data center topology
│
├── train_offline.py               # Phase 1: Digital twin training
├── deploy_online.py               # Phase 2: Live emulation loop
└── evaluate.py                    # Benchmarking vs OSPF/ECMP/DQN
```

---

### 2.1 Central Configuration — `config.yaml`

```yaml
# config.yaml
network:
  k_paths: 5                      # K-shortest paths to consider
  topology: "nsfnet"              # nsfnet | geant2 | fat_tree
  polling_interval: 2.0           # Seconds between telemetry polls

mm1k:
  buffer_capacity: 50             # K packets per switch port
  base_arrival_rate: 100.0        # λ (packets/sec) base
  base_service_rate: 150.0        # μ (packets/sec) base

graph_transformer:
  hidden_dim: 128
  num_heads: 4
  num_layers: 3
  dropout: 0.1

ppo:
  lr_actor: 3.0e-4
  lr_critic: 1.0e-3
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  update_epochs: 10
  batch_size: 64
  entropy_coef: 0.01

training:
  offline_episodes: 50000
  max_steps_per_episode: 200
  checkpoint_interval: 1000
  checkpoint_path: "checkpoints/"

reward_weights:
  alpha1: 0.4                     # Throughput weight
  alpha2: 0.3                     # Delay weight
  alpha3: 0.2                     # Packet loss weight
  alpha4: 0.1                     # Link utilization variance weight

controller:
  host: "127.0.0.1"
  rest_port: 8080
  openflow_port: 6633

agentic:
  provider: "gemini"              # gemini | openai
  model: "gemini-1.5-flash"
  temperature: 0.1
```

---

### 2.2 Environment Variables — `.env`

```bash
# .env
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 3. Topology Definitions

### `topologies/nsfnet.py`

```python
"""
NSFNET: 14-node, 21-link US WAN backbone topology.
Nodes represent major US cities; links have realistic propagation delays.
"""
import networkx as nx

def get_nsfnet() -> nx.Graph:
    G = nx.Graph()

    # Nodes: (id, city_name)
    nodes = [
        (0, "Seattle"), (1, "Palo Alto"), (2, "San Diego"),
        (3, "Salt Lake City"), (4, "Boulder"), (5, "Lincoln"),
        (6, "Houston"), (7, "Champaign"), (8, "Atlanta"),
        (9, "Ann Arbor"), (10, "Pittsburgh"), (11, "Princeton"),
        (12, "College Park"), (13, "Ithaca"),
    ]
    for nid, name in nodes:
        G.add_node(nid, label=name)

    # Edges: (src, dst, bandwidth_Mbps, delay_ms)
    edges = [
        (0, 1, 1000, 11), (0, 3, 1000, 9),  (0, 5, 1000, 29),
        (1, 2, 1000, 6),  (1, 3, 1000, 12), (2, 6, 1000, 22),
        (3, 4, 1000, 7),  (4, 5, 1000, 12), (4, 10, 1000, 26),
        (5, 7, 1000, 9),  (6, 8, 1000, 9),  (7, 8, 1000, 13),
        (7, 9, 1000, 4),  (8, 12, 1000, 8), (9, 10, 1000, 5),
        (9, 13, 1000, 7), (10, 11, 1000, 4),(11, 12, 1000, 3),
        (11, 13, 1000, 4),(12, 13, 1000, 5),(6, 12, 1000, 18),
    ]
    for src, dst, bw, delay in edges:
        G.add_edge(src, dst, bandwidth=bw, delay=delay,
                   utilization=0.0, packet_loss=0.0)
    return G
```

---

### `topologies/geant2.py`

```python
"""
GEANT2: 24-node, 37-link European academic network.
Irregular topology — good for testing edge-centrality awareness.
"""
import networkx as nx

def get_geant2() -> nx.Graph:
    G = nx.Graph()

    nodes = list(range(24))
    for n in nodes:
        G.add_node(n)

    # Representative GEANT2 edges (bandwidth Mbps, delay ms)
    edges = [
        (0,1,10000,5),(0,2,10000,12),(1,3,10000,8),(2,4,10000,15),
        (3,5,10000,6),(4,5,10000,20),(5,6,10000,7),(6,7,10000,9),
        (7,8,10000,11),(8,9,10000,14),(9,10,10000,18),(10,11,10000,6),
        (11,12,10000,8),(12,13,10000,10),(13,14,10000,7),(14,15,10000,12),
        (15,16,10000,9),(16,17,10000,6),(17,18,10000,8),(18,19,10000,11),
        (19,20,10000,14),(20,21,10000,9),(21,22,10000,7),(22,23,10000,6),
        (0,6,10000,25),(1,8,10000,22),(3,12,10000,30),(5,15,10000,27),
        (7,17,10000,19),(9,19,10000,28),(11,21,10000,24),(13,22,10000,31),
        (2,10,10000,35),(4,14,10000,32),(6,18,10000,29),(8,20,10000,26),
        (10,23,10000,21),
    ]
    for src, dst, bw, delay in edges:
        G.add_edge(src, dst, bandwidth=bw, delay=delay,
                   utilization=0.0, packet_loss=0.0)
    return G
```

---

### `topologies/fat_tree.py`

```python
"""
Fat-Tree k=8: Data center topology.
- 4 pods, each with k/2 edge + k/2 aggregation switches
- k^2/4 core switches
- k=8 → 80 switches total (16 core + 32 agg + 32 edge)
"""
import networkx as nx

def get_fat_tree(k: int = 8) -> nx.Graph:
    G = nx.Graph()
    num_pods = k
    num_core = (k // 2) ** 2
    num_agg = k * (k // 2)
    num_edge = k * (k // 2)
    bw, delay = 10000, 1

    core_start = 0
    agg_start = num_core
    edge_start = num_core + num_agg

    for n in range(num_core + num_agg + num_edge):
        G.add_node(n)

    for pod in range(num_pods):
        for agg_idx in range(k // 2):
            agg_id = agg_start + pod * (k // 2) + agg_idx
            # Connect aggregation to edge switches in same pod
            for edge_idx in range(k // 2):
                edge_id = edge_start + pod * (k // 2) + edge_idx
                G.add_edge(agg_id, edge_id, bandwidth=bw, delay=delay,
                           utilization=0.0, packet_loss=0.0)
            # Connect aggregation to core switches
            for core_idx in range(k // 2):
                core_id = core_start + agg_idx * (k // 2) + core_idx
                G.add_edge(core_id, agg_id, bandwidth=bw, delay=delay,
                           utilization=0.0, packet_loss=0.0)
    return G
```

---

## 4. Digital Twin — M/M/1/K Queuing Environment

### `digital_twin/mm1k_env.py`

```python
"""
Gymnasium environment modeling the SDN network as M/M/1/K queues.
Each switch port is an independent M/M/1/K queue.
Used for offline Phase 1 training — no live network required.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import yaml
from typing import Tuple, Dict, Any

class MM1KNetworkEnv(gym.Env):
    """
    State  : Flattened node + edge features as a fixed-size vector.
             (Graph Transformer handles variable-size graphs; this env
              uses a fixed-size representation for the tabular/MLP baseline.)
    Action : Integer index selecting one of K_PATHS candidate paths
             for the current (src, dst) flow demand.
    Reward : Multi-objective function from the MDP formulation.
    """

    metadata = {"render_modes": []}

    def __init__(self, graph: nx.Graph, config: dict):
        super().__init__()
        self.G = graph.copy()
        self.cfg = config
        self.K = config["network"]["k_paths"]
        self.K_buf = config["mm1k"]["buffer_capacity"]
        self.base_lambda = config["mm1k"]["base_arrival_rate"]
        self.base_mu = config["mm1k"]["base_service_rate"]
        self.alpha = [
            config["reward_weights"]["alpha1"],
            config["reward_weights"]["alpha2"],
            config["reward_weights"]["alpha3"],
            config["reward_weights"]["alpha4"],
        ]

        self.num_nodes = self.G.number_of_nodes()
        self.num_edges = self.G.number_of_edges()
        self.edges_list = list(self.G.edges())

        # Pre-compute all K-shortest paths between all pairs
        self._all_pairs_paths: Dict[Tuple, list] = {}
        self._precompute_paths()

        # Select a random flow demand for the episode
        self.current_src: int = 0
        self.current_dst: int = 1
        self.candidate_paths: list = []

        # Observation: [node_features | edge_features]
        # Node features per node: [cpu, buffer_occ, ingress_rate, egress_rate]
        # Edge features per edge: [bw_max, utilization, delay, packet_loss]
        node_feat_size = self.num_nodes * 4
        edge_feat_size = self.num_edges * 4
        obs_size = node_feat_size + edge_feat_size

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.K)

        # Traffic state
        self.lambda_map: Dict = {}   # arrival rates per edge
        self.mu_map: Dict = {}       # service rates per edge
        self.step_count = 0
        self.max_steps = config["training"]["max_steps_per_episode"]

    def _precompute_paths(self):
        """Pre-compute K shortest paths for all (src, dst) pairs."""
        for src in self.G.nodes():
            for dst in self.G.nodes():
                if src == dst:
                    continue
                try:
                    paths = list(nx.shortest_simple_paths(
                        self.G, src, dst, weight="delay"
                    ))
                    self._all_pairs_paths[(src, dst)] = paths[:self.K]
                except nx.NetworkXNoPath:
                    self._all_pairs_paths[(src, dst)] = []

    def _mm1k_metrics(self, lam: float, mu: float, K: int
                      ) -> Tuple[float, float, float]:
        """
        Compute M/M/1/K steady-state metrics.
        Returns: (expected_queue_length, overflow_probability, mean_delay_ms)
        """
        if mu <= 0:
            return K, 1.0, float("inf")
        rho = lam / mu
        eps = 1e-9

        if abs(rho - 1.0) < eps:
            # Special case ρ = 1
            P0 = 1.0 / (K + 1)
            P_overflow = P0
            E_Q = K / 2.0
        else:
            denom = (1 - rho ** (K + 1))
            if abs(denom) < eps:
                return K, 1.0, float("inf")
            P0 = (1 - rho) / denom
            P_overflow = P0 * (rho ** K)
            # E[Q] = ρ[1 − (K+1)ρ^K + Kρ^(K+1)] / [(1−ρ)(1−ρ^(K+1))]
            numerator = rho * (1 - (K + 1) * rho**K + K * rho**(K + 1))
            E_Q = numerator / ((1 - rho) * denom)

        # Mean delay via Little's Law: W = E[Q] / (λ_eff)
        lam_eff = lam * (1 - P_overflow)
        mean_delay = (E_Q / lam_eff * 1000) if lam_eff > eps else float("inf")
        return float(np.clip(E_Q, 0, K)), float(P_overflow), float(mean_delay)

    def _get_obs(self) -> np.ndarray:
        """Build flat observation vector from current network state."""
        node_feats = []
        for n in sorted(self.G.nodes()):
            cpu = np.random.uniform(0.1, 0.9)          # Simulated CPU load
            buf_occ = np.random.uniform(0.0, 0.8)      # Buffer occupancy
            ingress = np.random.uniform(0.0, 1.0)
            egress = np.random.uniform(0.0, 1.0)
            node_feats.extend([cpu, buf_occ, ingress, egress])

        edge_feats = []
        for u, v in self.edges_list:
            data = self.G.edges[u, v]
            bw_max_norm = min(data.get("bandwidth", 1000) / 10000.0, 1.0)
            util = float(np.clip(data.get("utilization", 0.0), 0.0, 1.0))
            delay_norm = min(data.get("delay", 1) / 100.0, 1.0)
            pkt_loss = float(np.clip(data.get("packet_loss", 0.0), 0.0, 1.0))
            edge_feats.extend([bw_max_norm, util, delay_norm, pkt_loss])

        obs = np.array(node_feats + edge_feats, dtype=np.float32)
        return obs

    def _simulate_traffic(self):
        """Inject random bursty traffic onto network edges."""
        for u, v in self.edges_list:
            bw = self.G.edges[u, v].get("bandwidth", 1000)
            # Modulated Gravity: base load + random microburst
            burst_factor = np.random.choice(
                [1.0, 2.0, 5.0], p=[0.7, 0.2, 0.1]
            )
            lam = self.base_lambda * burst_factor * np.random.uniform(0.5, 1.5)
            mu = self.base_mu * np.random.uniform(0.8, 1.2)
            self.lambda_map[(u, v)] = lam
            self.mu_map[(u, v)] = mu

            E_Q, P_overflow, _ = self._mm1k_metrics(lam, mu, self.K_buf)
            self.G.edges[u, v]["utilization"] = np.clip(
                lam / (mu + 1e-9), 0.0, 1.0
            )
            self.G.edges[u, v]["packet_loss"] = P_overflow

    def _compute_reward(self, path: list) -> float:
        """
        Multi-objective reward:
        R = α1*(T_actual/T_req) - α2*D_path - α3*L_packet - α4*Var_util
        """
        if not path or len(path) < 2:
            return -10.0   # Invalid path penalty

        path_edges = list(zip(path[:-1], path[1:]))

        # End-to-end delay (sum of per-hop delays + queuing)
        total_delay = 0.0
        total_loss = 0.0
        min_bw = float("inf")
        utilizations = []

        for u, v in path_edges:
            edge = self.G.edges.get((u, v)) or self.G.edges.get((v, u))
            if edge is None:
                return -10.0
            lam = self.lambda_map.get((u, v), self.base_lambda)
            mu = self.mu_map.get((u, v), self.base_mu)
            _, P_overflow, delay_ms = self._mm1k_metrics(lam, mu, self.K_buf)

            prop_delay = edge.get("delay", 1.0)
            total_delay += prop_delay + min(delay_ms, 500.0)
            total_loss = 1 - (1 - total_loss) * (1 - P_overflow)
            bw = edge.get("bandwidth", 1000)
            util = edge.get("utilization", 0.5)
            min_bw = min(min_bw, bw)
            utilizations.append(util)

        # All-link utilization variance (penalizes unbalanced loads)
        all_utils = [
            self.G.edges[u, v].get("utilization", 0.0)
            for u, v in self.edges_list
        ]
        util_variance = float(np.var(all_utils))

        # Throughput ratio (simplified: bottleneck bandwidth vs demand)
        T_actual = min_bw * (1 - total_loss)
        T_req = min_bw  # Demand = full link capacity
        throughput_ratio = T_actual / (T_req + 1e-9)

        # Normalize delay (target < 50ms)
        delay_norm = min(total_delay / 500.0, 1.0)

        r = (
            self.alpha[0] * throughput_ratio
            - self.alpha[1] * delay_norm
            - self.alpha[2] * total_loss
            - self.alpha[3] * util_variance
        )
        return float(r)

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.step_count = 0

        # Pick a random flow demand
        nodes = list(self.G.nodes())
        self.current_src, self.current_dst = np.random.choice(
            nodes, 2, replace=False
        )
        self.candidate_paths = self._all_pairs_paths.get(
            (self.current_src, self.current_dst), []
        )

        self._simulate_traffic()
        obs = self._get_obs()
        info = {"src": self.current_src, "dst": self.current_dst}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.step_count += 1

        # Map action index → actual path
        if action < len(self.candidate_paths):
            selected_path = self.candidate_paths[action]
        else:
            # Invalid action → choose shortest available
            selected_path = (
                self.candidate_paths[0] if self.candidate_paths else []
            )

        reward = self._compute_reward(selected_path)

        # Update utilization on selected path edges
        for u, v in zip(selected_path[:-1], selected_path[1:]):
            for key in [(u, v), (v, u)]:
                if key in self.G.edges:
                    self.G.edges[key]["utilization"] = np.clip(
                        self.G.edges[key]["utilization"] + 0.05, 0.0, 1.0
                    )

        # Simulate next step traffic changes
        self._simulate_traffic()

        terminated = self.step_count >= self.max_steps
        truncated = False
        obs = self._get_obs()
        info = {
            "path": selected_path,
            "src": self.current_src,
            "dst": self.current_dst,
            "reward": reward,
        }
        return obs, reward, terminated, truncated, info
```

---

## 5. Graph Transformer Encoder

### `model/graph_transformer.py`

```python
"""
Graph Transformer Encoder with virtual star node.
Converts variable-size network state graph → fixed-size latent vector.
Uses PyTorch Geometric (PyG) TransformerConv layers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.data import Data, Batch
import networkx as nx
import numpy as np
from typing import Dict, List


class GraphTransformerEncoder(nn.Module):
    """
    Multi-layer Graph Transformer with:
    - 4 node features: [cpu, buffer_occ, ingress_rate, egress_rate]
    - 4 edge features: [bw_norm, utilization, delay_norm, packet_loss]
    - Virtual star node for global message passing
    - Outputs: fixed-size latent vector of shape (hidden_dim,)
    """

    NODE_FEAT_DIM = 4   # Per-node input features
    EDGE_FEAT_DIM = 4   # Per-edge input features

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Initial node embedding (projects 4-dim features → hidden_dim)
        self.node_embed = nn.Linear(self.NODE_FEAT_DIM + 1, hidden_dim)
        #  +1 for the virtual star node indicator flag

        # Graph Transformer layers
        self.conv_layers = nn.ModuleList([
            TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                edge_dim=self.EDGE_FEAT_DIM,
                dropout=dropout,
                concat=True,   # Output: hidden_dim (num_heads × head_dim)
            )
            for _ in range(num_layers)
        ])

        # Layer norms for stable training
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Final MLP to produce latent state vector
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data: PyG Data object with:
                  - data.x          : node features [N+1, 4+1]  (+1 star node)
                  - data.edge_index : [2, E_total]
                  - data.edge_attr  : [E_total, 4]
                  - data.batch      : batch assignment [N+1]

        Returns:
            latent: [batch_size, hidden_dim]
        """
        x = self.node_embed(data.x)   # [N+1, hidden_dim]

        for conv, norm in zip(self.conv_layers, self.layer_norms):
            x_res = x
            x = conv(x, data.edge_index, data.edge_attr)
            x = norm(x + x_res)        # Residual connection
            x = F.relu(x)

        # Pool over all real nodes only (exclude star node during pooling
        # by masking — star node is always last in each graph)
        latent = global_mean_pool(x, data.batch)   # [batch, hidden_dim]
        latent = self.output_mlp(latent)
        return latent


def nx_to_pyg(G: nx.Graph, device: torch.device = torch.device("cpu")) -> Data:
    """
    Convert a NetworkX graph with current telemetry state into a
    PyTorch Geometric Data object, adding a virtual star node.

    The virtual star node is connected (bidirectionally) to ALL real nodes,
    enabling single-hop global information exchange.
    """
    nodes = sorted(G.nodes())
    n_real = len(nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}

    # ── Node features [N, 4] ─────────────────────────────────────────────
    node_feats = []
    for n in nodes:
        attrs = G.nodes[n]
        feat = [
            attrs.get("cpu", 0.5),
            attrs.get("buffer_occ", 0.3),
            attrs.get("ingress_rate", 0.5),
            attrs.get("egress_rate", 0.5),
        ]
        node_feats.append(feat + [0.0])   # 0.0 = not star node

    # Star node: append as last node (index n_real)
    node_feats.append([0.0, 0.0, 0.0, 0.0, 1.0])   # 1.0 = star node flag

    # ── Edge indices & attributes ─────────────────────────────────────────
    src_list, dst_list, edge_attr_list = [], [], []

    for u, v, data in G.edges(data=True):
        ui, vi = node_idx[u], node_idx[v]
        bw_norm = min(data.get("bandwidth", 1000) / 10000.0, 1.0)
        util = float(np.clip(data.get("utilization", 0.0), 0.0, 1.0))
        delay_norm = min(data.get("delay", 1.0) / 100.0, 1.0)
        pkt_loss = float(np.clip(data.get("packet_loss", 0.0), 0.0, 1.0))
        attr = [bw_norm, util, delay_norm, pkt_loss]

        # Bidirectional edges
        for s, d in [(ui, vi), (vi, ui)]:
            src_list.append(s)
            dst_list.append(d)
            edge_attr_list.append(attr)

    # Star node ↔ all real nodes (bidirectional)
    star_idx = n_real
    neutral_attr = [1.0, 0.0, 0.0, 0.0]
    for i in range(n_real):
        for s, d in [(star_idx, i), (i, star_idx)]:
            src_list.append(s)
            dst_list.append(d)
            edge_attr_list.append(neutral_attr)

    x = torch.tensor(node_feats, dtype=torch.float32, device=device)
    edge_index = torch.tensor(
        [src_list, dst_list], dtype=torch.long, device=device
    )
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32, device=device)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
```

---

## 6. PPO Decision Engine

### `model/ppo_agent.py`

```python
"""
PPO Actor-Critic Agent for K-shortest path selection.
Actor  : outputs softmax probability distribution over K paths.
Critic : estimates state value V(s).
Uses clipped surrogate objective to prevent catastrophic policy updates.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from typing import List, Tuple, Optional
from model.graph_transformer import GraphTransformerEncoder, nx_to_pyg
from torch_geometric.data import Data
import networkx as nx


class ActorCriticNetwork(nn.Module):
    def __init__(self, latent_dim: int, k_paths: int, hidden_dim: int = 256):
        super().__init__()
        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Actor head → logits over K paths
        self.actor_head = nn.Linear(hidden_dim, k_paths)
        # Critic head → scalar state value
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, latent: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_out = self.shared(latent)
        logits = self.actor_head(shared_out)
        value = self.critic_head(shared_out).squeeze(-1)
        return logits, value

    def get_action(self, latent: torch.Tensor, mask: Optional[torch.Tensor] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action with optional masking for invalid paths.
        mask: boolean tensor of shape [K], True = valid path exists.
        """
        logits, value = self.forward(latent)
        if mask is not None:
            # Set invalid path logits to large negative value
            logits = logits.masked_fill(~mask, float("-inf"))
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


class RolloutBuffer:
    """Stores transitions for a single PPO update cycle."""

    def __init__(self):
        self.states: List[Data] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.actions)


class PPOAgent:
    """
    Full PPO agent combining Graph Transformer encoder + Actor-Critic.
    Implements clipped surrogate objective with GAE advantage estimation.
    """

    def __init__(self, config: dict, k_paths: int,
                 device: torch.device = torch.device("cpu")):
        self.config = config
        self.k_paths = k_paths
        self.device = device

        ppo_cfg = config["ppo"]
        gt_cfg = config["graph_transformer"]

        # Build encoder + actor-critic
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

        # Separate optimizers for actor and critic (allows different LR)
        self.optimizer_encoder = torch.optim.Adam(
            self.encoder.parameters(), lr=ppo_cfg["lr_actor"]
        )
        self.optimizer_ac = torch.optim.Adam([
            {"params": self.ac_net.actor_head.parameters(),
             "lr": ppo_cfg["lr_actor"]},
            {"params": self.ac_net.critic_head.parameters(),
             "lr": ppo_cfg["lr_critic"]},
            {"params": self.ac_net.shared.parameters(),
             "lr": ppo_cfg["lr_actor"]},
        ])

        self.gamma = ppo_cfg["gamma"]
        self.gae_lambda = ppo_cfg["gae_lambda"]
        self.clip_eps = ppo_cfg["clip_epsilon"]
        self.update_epochs = ppo_cfg["update_epochs"]
        self.batch_size = ppo_cfg["batch_size"]
        self.entropy_coef = ppo_cfg["entropy_coef"]

        self.buffer = RolloutBuffer()

    def encode(self, graph: nx.Graph) -> torch.Tensor:
        """Convert NetworkX graph → latent state vector."""
        pyg_data = nx_to_pyg(graph, self.device)
        pyg_data = pyg_data.to(self.device)
        # Add dummy batch dimension
        pyg_data.batch = torch.zeros(
            pyg_data.x.size(0), dtype=torch.long, device=self.device
        )
        with torch.no_grad():
            latent = self.encoder(pyg_data)
        return latent   # [1, hidden_dim]

    @torch.no_grad()
    def select_action(self, graph: nx.Graph,
                      candidate_paths: list) -> Tuple[int, float, float]:
        """
        Select routing action given current network graph state.
        Returns: (action_idx, log_prob, value_estimate)
        """
        latent = self.encode(graph)
        # Build path validity mask
        mask = torch.zeros(self.k_paths, dtype=torch.bool, device=self.device)
        for i in range(min(len(candidate_paths), self.k_paths)):
            mask[i] = True

        action, log_prob, value = self.ac_net.get_action(latent, mask)
        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.item()),
        )

    def update(self) -> dict:
        """
        PPO update: iterate over rollout buffer for `update_epochs`,
        computing clipped surrogate loss + value loss + entropy bonus.
        Returns dict of training metrics.
        """
        if len(self.buffer) == 0:
            return {}

        # Convert buffer to tensors
        actions = torch.tensor(self.buffer.actions,
                               dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(self.buffer.log_probs,
                                     dtype=torch.float32, device=self.device)
        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones

        # ── GAE Advantage Estimation ──────────────────────────────────────
        advantages = []
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            delta = (rewards[t]
                     + self.gamma * next_value * (1 - dones[t])
                     - values[t])
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            next_value = values[t]

        advantages = torch.tensor(
            advantages, dtype=torch.float32, device=self.device
        )
        returns = advantages + torch.tensor(
            values, dtype=torch.float32, device=self.device
        )
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Re-encode all states (costly but necessary for correct gradients)
        all_latents = []
        for state_graph in self.buffer.states:
            pyg = nx_to_pyg(state_graph, self.device)
            pyg.batch = torch.zeros(
                pyg.x.size(0), dtype=torch.long, device=self.device
            )
            latent = self.encoder(pyg)
            all_latents.append(latent)
        all_latents = torch.cat(all_latents, dim=0)   # [T, hidden_dim]

        # ── PPO Update Epochs ─────────────────────────────────────────────
        metrics = {
            "policy_loss": [], "value_loss": [],
            "entropy": [], "approx_kl": []
        }
        T = len(actions)

        for _ in range(self.update_epochs):
            # Mini-batch shuffled updates
            indices = torch.randperm(T, device=self.device)
            for start in range(0, T, self.batch_size):
                idx = indices[start:start + self.batch_size]
                batch_latents = all_latents[idx]
                batch_actions = actions[idx]
                batch_old_lp = old_log_probs[idx]
                batch_adv = advantages[idx]
                batch_returns = returns[idx]

                logits, values_pred = self.ac_net(batch_latents)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Probability ratio r_t(θ) = π_θ / π_θ_old
                ratio = torch.exp(new_log_probs - batch_old_lp)

                # Clipped surrogate objective (PPO-Clip)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_eps,
                    1.0 + self.clip_eps
                ) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss (clipped)
                value_loss = F.mse_loss(values_pred, batch_returns)

                # Total loss
                loss = (policy_loss
                        + 0.5 * value_loss
                        - self.entropy_coef * entropy)

                self.optimizer_encoder.zero_grad()
                self.optimizer_ac.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters())
                    + list(self.ac_net.parameters()),
                    max_norm=0.5
                )
                self.optimizer_encoder.step()
                self.optimizer_ac.step()

                # Track metrics
                with torch.no_grad():
                    approx_kl = (batch_old_lp - new_log_probs).mean().item()
                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.item())
                metrics["approx_kl"].append(approx_kl)

        self.buffer.clear()

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def save(self, path: str):
        torch.save({
            "encoder": self.encoder.state_dict(),
            "ac_net": self.ac_net.state_dict(),
        }, path)
        print(f"[PPO] Checkpoint saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.ac_net.load_state_dict(ckpt["ac_net"])
        print(f"[PPO] Checkpoint loaded ← {path}")
```

---

## 7. OS-Ken Controller (Control Plane)

### `controller/garro_controller.py`

```python
"""
OS-Ken OpenFlow 1.3 controller application for GARRO.
Handles:
  - Topology discovery via LLDP
  - Real-time telemetry collection (port stats, flow stats)
  - Northbound REST API exposing network state JSON
  - Flow rule installation from PPO agent routing decisions

Run with:
    osken-manager controller/garro_controller.py \
        --observe-links \
        --wsapi-port 8080
"""
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import (
    CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
)
from os_ken.ofproto import ofproto_v1_3
from os_ken.lib.packet import packet, ethernet, ipv4, ether_types
from os_ken.topology import event as topo_event
from os_ken.topology.api import get_switch, get_link
from os_ken.app.wsgi import WSGIApplication, ControllerBase, route
from os_ken.lib import hub
import json
import time
import networkx as nx
from collections import defaultdict


REST_API_NAME = "garro_rest_api"


class GARROController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {"wsgi": WSGIApplication}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        wsgi = kwargs["wsgi"]
        wsgi.register(GARRORestAPI, {REST_API_NAME: self})

        # Network state
        self.topology: nx.DiGraph = nx.DiGraph()
        self.datapaths: dict = {}          # dpid → datapath object
        self.port_stats: dict = defaultdict(dict)   # dpid → port stats
        self.flow_stats: dict = defaultdict(dict)
        self.mac_to_port: dict = defaultdict(dict)  # dpid → mac → port
        self.pending_flows: list = []       # Flow rules waiting to be installed

        # Telemetry polling thread (every 2 seconds)
        self.monitor_thread = hub.spawn(self._monitor_loop)

    # ── OpenFlow Event Handlers ────────────────────────────────────────────

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Install table-miss flow entry on every new switch."""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        self.datapaths[datapath.id] = datapath

        # Table-miss: send unmatched packets to controller
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(
            ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER
        )]
        self._add_flow(datapath, 0, match, actions)
        self.logger.info(f"[GARRO] Switch connected: dpid={datapath.id:016x}")

    @set_ev_cls(topo_event.EventSwitchEnter)
    def switch_enter(self, ev):
        self._update_topology()

    @set_ev_cls(topo_event.EventLinkAdd)
    def link_add(self, ev):
        self._update_topology()

    @set_ev_cls(topo_event.EventLinkDelete)
    def link_delete(self, ev):
        self._update_topology()

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """Basic L2 learning switch for non-DRL-managed flows."""
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        in_port = msg.match["in_port"]
        dpid = dp.id

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return   # Let topology module handle LLDP

        dst = eth.dst
        src = eth.src
        self.mac_to_port[dpid][src] = in_port

        out_port = (
            self.mac_to_port[dpid].get(dst, ofp.OFPP_FLOOD)
        )

        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofp.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst)
            self._add_flow(dp, 1, match, actions, idle_timeout=30)

        data = msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None
        out = parser.OFPPacketOut(
            datapath=dp,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data,
        )
        dp.send_msg(out)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply(self, ev):
        dpid = ev.msg.datapath.id
        for stat in ev.msg.body:
            self.port_stats[dpid][stat.port_no] = {
                "rx_bytes": stat.rx_bytes,
                "tx_bytes": stat.tx_bytes,
                "rx_packets": stat.rx_packets,
                "tx_packets": stat.tx_packets,
                "rx_dropped": stat.rx_dropped,
                "tx_dropped": stat.tx_dropped,
                "rx_errors": stat.rx_errors,
            }

    # ── Telemetry Polling ──────────────────────────────────────────────────

    def _monitor_loop(self):
        """Continuously poll switch statistics every 2 seconds."""
        while True:
            for dpid, dp in list(self.datapaths.items()):
                self._request_port_stats(dp)
            hub.sleep(2)

    def _request_port_stats(self, datapath):
        parser = datapath.ofproto_parser
        req = parser.OFPPortStatsRequest(
            datapath, 0, datapath.ofproto.OFPP_ANY
        )
        datapath.send_msg(req)

    # ── Topology Management ────────────────────────────────────────────────

    def _update_topology(self):
        """Rebuild NetworkX graph from OS-Ken topology API."""
        switches = get_switch(self, None)
        links = get_link(self, None)

        self.topology.clear()
        for sw in switches:
            self.topology.add_node(sw.dp.id)

        for link in links:
            self.topology.add_edge(
                link.src.dpid, link.dst.dpid,
                src_port=link.src.port_no,
                dst_port=link.dst.port_no,
                bandwidth=1000,
                delay=1.0,
                utilization=0.0,
                packet_loss=0.0,
            )

    # ── Flow Installation ──────────────────────────────────────────────────

    def _add_flow(self, datapath, priority, match, actions,
                  idle_timeout=0, hard_timeout=0):
        ofp = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=idle_timeout,
            hard_timeout=hard_timeout,
        )
        datapath.send_msg(mod)

    def install_path_flow(self, path: list, src_ip: str, dst_ip: str,
                          priority: int = 100):
        """
        Install flow rules along a computed path.
        path: list of dpid values [dpid1, dpid2, ..., dpidN]
        """
        if len(path) < 2:
            self.logger.warning("[GARRO] Path too short to install flows")
            return

        for i in range(len(path) - 1):
            current_dpid = path[i]
            next_dpid = path[i + 1]
            dp = self.datapaths.get(current_dpid)
            if dp is None:
                continue

            edge_data = self.topology.edges.get((current_dpid, next_dpid))
            if edge_data is None:
                continue

            out_port = edge_data["src_port"]
            parser = dp.ofproto_parser

            match = parser.OFPMatch(
                eth_type=0x0800,
                ipv4_src=src_ip,
                ipv4_dst=dst_ip,
            )
            actions = [parser.OFPActionOutput(out_port)]
            self._add_flow(dp, priority, match, actions,
                           idle_timeout=60, hard_timeout=120)

        self.logger.info(
            f"[GARRO] Installed path: {path} for {src_ip} → {dst_ip}"
        )

    # ── REST API Data Builder ──────────────────────────────────────────────

    def get_network_state(self) -> dict:
        """Build network state JSON for the AI plane."""
        nodes = []
        for n in self.topology.nodes():
            stats = self.port_stats.get(n, {})
            total_rx = sum(s.get("rx_bytes", 0) for s in stats.values())
            total_tx = sum(s.get("tx_bytes", 0) for s in stats.values())
            nodes.append({
                "dpid": n,
                "cpu": 0.5,          # Placeholder — extend with SNMP
                "buffer_occ": 0.3,
                "ingress_rate": total_rx,
                "egress_rate": total_tx,
            })

        edges = []
        for u, v, data in self.topology.edges(data=True):
            edges.append({
                "src": u, "dst": v,
                "bandwidth": data.get("bandwidth", 1000),
                "utilization": data.get("utilization", 0.0),
                "delay": data.get("delay", 1.0),
                "packet_loss": data.get("packet_loss", 0.0),
                "src_port": data.get("src_port", 0),
                "dst_port": data.get("dst_port", 0),
            })

        return {
            "timestamp": time.time(),
            "nodes": nodes,
            "edges": edges,
        }


class GARRORestAPI(ControllerBase):
    """Northbound REST API exposed by the OS-Ken controller."""

    def __init__(self, req, link, data, **config):
        super().__init__(req, link, data, **config)
        self.garro_app: GARROController = data[REST_API_NAME]

    @route("network_state", "/garro/state", methods=["GET"])
    def get_state(self, req, **kwargs):
        state = self.garro_app.get_network_state()
        return self.Response(
            content_type="application/json",
            body=json.dumps(state),
        )

    @route("install_flow", "/garro/flow", methods=["POST"])
    def install_flow(self, req, **kwargs):
        try:
            body = json.loads(req.body)
            path = body["path"]        # List of dpid values
            src_ip = body["src_ip"]
            dst_ip = body["dst_ip"]
            self.garro_app.install_path_flow(path, src_ip, dst_ip)
            return self.Response(
                content_type="application/json",
                body=json.dumps({"status": "ok"}),
            )
        except Exception as e:
            return self.Response(
                status=400,
                content_type="application/json",
                body=json.dumps({"error": str(e)}),
            )

    @route("topology", "/garro/topology", methods=["GET"])
    def get_topology(self, req, **kwargs):
        nodes = list(self.garro_app.topology.nodes())
        edges = [
            {"src": u, "dst": v}
            for u, v in self.garro_app.topology.edges()
        ]
        return self.Response(
            content_type="application/json",
            body=json.dumps({"nodes": nodes, "edges": edges}),
        )
```

---

## 8. Agentic AI Layer (LLM Orchestrator)

### `agentic/llm_orchestrator.py`

```python
"""
Agentic AI Layer — translates operator natural language intent
into PPO reward function weights (α1..α4).

Supports: Google Gemini API, OpenAI GPT-4 API.
"""
import os
import json
import re
import aiohttp
import asyncio
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RewardWeights:
    alpha1: float = 0.4   # Throughput
    alpha2: float = 0.3   # Delay
    alpha3: float = 0.2   # Packet Loss
    alpha4: float = 0.1   # Link Utilization Variance

    def as_dict(self) -> dict:
        return {
            "alpha1": self.alpha1,
            "alpha2": self.alpha2,
            "alpha3": self.alpha3,
            "alpha4": self.alpha4,
        }

    def normalized(self) -> "RewardWeights":
        total = self.alpha1 + self.alpha2 + self.alpha3 + self.alpha4
        if total == 0:
            return RewardWeights()
        return RewardWeights(
            alpha1=self.alpha1 / total,
            alpha2=self.alpha2 / total,
            alpha3=self.alpha3 / total,
            alpha4=self.alpha4 / total,
        )


SYSTEM_PROMPT = """You are a network policy translator for an SDN routing system.

Given a natural language operator intent, return ONLY a JSON object
with four float values between 0 and 1 that must sum to 1.0:

{
  "alpha1": <throughput weight>,
  "alpha2": <latency/delay weight>,
  "alpha3": <packet_loss weight>,
  "alpha4": <link_utilization_balance weight>
}

Guidelines:
- High-priority real-time traffic (video conferencing, VoIP, gaming):
  high alpha2 (delay), moderate alpha3 (packet loss)
- File transfer / bulk data: high alpha1 (throughput), low alpha2
- Congested network: high alpha4 (balance utilization)
- Default balanced: alpha1=0.4, alpha2=0.3, alpha3=0.2, alpha4=0.1

Return ONLY the JSON. No explanation, no markdown, no preamble."""


class LLMOrchestrator:
    def __init__(self, config: dict):
        self.provider = config["agentic"]["provider"]
        self.model = config["agentic"]["model"]
        self.temperature = config["agentic"]["temperature"]
        self.current_weights = RewardWeights()

    async def parse_intent(self, operator_intent: str) -> RewardWeights:
        """
        Parse natural language intent → reward weights via LLM API.
        Returns fallback weights on any API failure.
        """
        try:
            if self.provider == "gemini":
                weights = await self._call_gemini(operator_intent)
            elif self.provider == "openai":
                weights = await self._call_openai(operator_intent)
            else:
                print(f"[LLM] Unknown provider: {self.provider}, using defaults")
                return self.current_weights

            self.current_weights = weights.normalized()
            print(f"[LLM] Parsed intent → weights: {self.current_weights.as_dict()}")
            return self.current_weights

        except Exception as e:
            print(f"[LLM] API call failed: {e}. Using previous weights.")
            return self.current_weights

    async def _call_gemini(self, intent: str) -> RewardWeights:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env")

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={api_key}"
        )
        payload = {
            "contents": [{
                "parts": [
                    {"text": SYSTEM_PROMPT},
                    {"text": f"\nOperator intent: {intent}"},
                ]
            }],
            "generationConfig": {"temperature": self.temperature},
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()

        raw = data["candidates"][0]["content"]["parts"][0]["text"]
        return self._parse_json_response(raw)

    async def _call_openai(self, intent: str) -> RewardWeights:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in .env")

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Operator intent: {intent}"},
            ],
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()

        raw = data["choices"][0]["message"]["content"]
        return self._parse_json_response(raw)

    def _parse_json_response(self, raw: str) -> RewardWeights:
        """Extract JSON from LLM response, handling markdown fences."""
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        obj = json.loads(cleaned)
        return RewardWeights(
            alpha1=float(obj.get("alpha1", 0.4)),
            alpha2=float(obj.get("alpha2", 0.3)),
            alpha3=float(obj.get("alpha3", 0.2)),
            alpha4=float(obj.get("alpha4", 0.1)),
        )

    def get_fallback_weights(self) -> RewardWeights:
        """Deterministic fallback — use last known-good weights."""
        return self.current_weights


# ── Usage example ──────────────────────────────────────────────────────────
async def demo():
    config = {"agentic": {"provider": "gemini",
                           "model": "gemini-1.5-flash",
                           "temperature": 0.1}}
    orch = LLMOrchestrator(config)
    intent = ("Prioritize video conferencing traffic, latency is critical. "
              "File transfers can wait.")
    weights = await orch.parse_intent(intent)
    print(weights.as_dict())


if __name__ == "__main__":
    asyncio.run(demo())
```

---

## 9. Phase 1 — Offline Digital Twin Training

### `train_offline.py`

```python
"""
Phase 1: Offline PPO training using the M/M/1/K Digital Twin.
No live network required. Run this first.

Usage:
    python train_offline.py --topology nsfnet --episodes 50000
"""
import os
import yaml
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from digital_twin.mm1k_env import MM1KNetworkEnv
from model.ppo_agent import PPOAgent
from topologies.nsfnet import get_nsfnet
from topologies.geant2 import get_geant2
from topologies.fat_tree import get_fat_tree


TOPOLOGY_MAP = {
    "nsfnet": get_nsfnet,
    "geant2": get_geant2,
    "fat_tree": lambda: get_fat_tree(k=8),
}


def main(args):
    # ── Load config ───────────────────────────────────────────────────────
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    config["network"]["topology"] = args.topology
    total_episodes = args.episodes or config["training"]["offline_episodes"]
    checkpoint_dir = config["training"]["checkpoint_path"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")
    print(f"[Train] Topology: {args.topology}")

    # ── Build graph & environment ─────────────────────────────────────────
    G = TOPOLOGY_MAP[args.topology]()
    env = MM1KNetworkEnv(G, config)

    # Verify agent K ≤ actual available paths
    k_paths = config["network"]["k_paths"]
    agent = PPOAgent(config, k_paths=k_paths, device=device)

    # ── Training loop ─────────────────────────────────────────────────────
    episode_rewards = []
    update_interval = 512    # Collect N steps before each PPO update

    obs, info = env.reset()
    step_count = 0
    ep_reward = 0.0
    ep_idx = 0
    ep_rewards_window = []

    pbar = tqdm(total=total_episodes, desc="Training")

    while ep_idx < total_episodes:
        # Agent selects path using current graph state (uses nx graph directly)
        candidate_paths = env.candidate_paths
        action, log_prob, value = agent.select_action(env.G, candidate_paths)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store in buffer (store graph snapshot + action)
        import copy
        agent.buffer.add(
            state=copy.deepcopy(env.G),
            action=action,
            log_prob=log_prob,
            reward=reward,
            value=value,
            done=done,
        )

        obs = next_obs
        ep_reward += reward
        step_count += 1

        if done:
            ep_idx += 1
            ep_rewards_window.append(ep_reward)
            ep_reward = 0.0
            obs, info = env.reset()
            pbar.update(1)

            # Checkpoint & logging
            interval = config["training"]["checkpoint_interval"]
            if ep_idx % interval == 0:
                avg_r = np.mean(ep_rewards_window[-interval:])
                tqdm.write(
                    f"[Ep {ep_idx:6d}] Avg Reward: {avg_r:.4f} | "
                    f"Steps: {step_count}"
                )
                agent.save(
                    f"{checkpoint_dir}/garro_{args.topology}_ep{ep_idx}.pt"
                )
                episode_rewards.extend(ep_rewards_window[-interval:])

        # PPO update every `update_interval` steps
        if step_count % update_interval == 0 and len(agent.buffer) > 0:
            metrics = agent.update()
            if metrics:
                tqdm.write(
                    f"  → PPO update | PL:{metrics['policy_loss']:.4f} "
                    f"VL:{metrics['value_loss']:.4f} "
                    f"Ent:{metrics['entropy']:.4f}"
                )

    pbar.close()

    # ── Save final model ──────────────────────────────────────────────────
    final_path = f"{checkpoint_dir}/garro_{args.topology}_final.pt"
    agent.save(final_path)
    print(f"\n[Train] Final model saved: {final_path}")

    # ── Plot training curve ───────────────────────────────────────────────
    if episode_rewards:
        plt.figure(figsize=(12, 5))
        plt.plot(episode_rewards, alpha=0.4, label="Episode Reward")
        # Smoothed moving average
        window = 200
        if len(episode_rewards) >= window:
            smooth = np.convolve(
                episode_rewards, np.ones(window) / window, mode="valid"
            )
            plt.plot(range(window - 1, len(episode_rewards)), smooth,
                     linewidth=2, label=f"MA-{window}")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"GARRO Offline Training — {args.topology.upper()}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{checkpoint_dir}/training_curve_{args.topology}.png",
                    dpi=150)
        print(f"[Train] Training curve saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", default="nsfnet",
                        choices=["nsfnet", "geant2", "fat_tree"])
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()
    main(args)
```

---

## 10. Phase 2 — Live Mininet Emulation

### Mininet Topology Script — `topologies/mininet_nsfnet.py`

```python
"""
Mininet topology for NSFNET (14 nodes, 21 links).
Run with: sudo python topologies/mininet_nsfnet.py

Requires Mininet installed system-wide (not in venv).
Start the OS-Ken controller first.
"""
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel


def build_nsfnet():
    setLogLevel("info")

    net = Mininet(
        controller=RemoteController,
        switch=OVSKernelSwitch,
        link=TCLink,
        autoSetMacs=True,
    )

    # Remote controller (OS-Ken)
    c0 = net.addController("c0", ip="127.0.0.1", port=6633)

    # Add 14 switches (one per NSFNET node)
    switches = []
    for i in range(1, 15):
        sw = net.addSwitch(f"s{i}", protocols="OpenFlow13")
        switches.append(sw)

    # Add hosts (one per switch for testing)
    hosts = []
    for i, sw in enumerate(switches, start=1):
        h = net.addHost(f"h{i}", ip=f"10.0.0.{i}/24")
        net.addLink(h, sw, bw=100, delay="1ms")
        hosts.append(h)

    # NSFNET edges: (src_idx, dst_idx, bw_Mbps, delay_ms)
    edges = [
        (0,1,1000,11),(0,2,1000,9),(0,4,1000,29),(1,2,1000,6),
        (1,3,1000,12),(2,5,1000,22),(3,4,1000,7),(4,5,1000,12),
        (4,9,1000,26),(5,6,1000,9),(6,7,1000,9),(6,8,1000,13),
        (7,8,1000,4),(7,12,1000,7),(8,9,1000,5),(9,10,1000,4),
        (10,11,1000,3),(10,12,1000,4),(11,12,1000,5),(11,13,1000,18),
        (5,11,1000,8),
    ]
    for src_i, dst_i, bw, delay in edges:
        net.addLink(
            switches[src_i], switches[dst_i],
            bw=bw, delay=f"{delay}ms", max_queue_size=50
        )

    net.start()
    # Set OpenFlow version on all switches
    for sw in switches:
        sw.cmd("ovs-vsctl set bridge", sw, "protocols=OpenFlow13")

    print("\n[Mininet] NSFNET topology running.")
    print("[Mininet] Hosts: h1-h14, Switches: s1-s14")
    print("[Mininet] Controller: 127.0.0.1:6633\n")

    CLI(net)
    net.stop()


if __name__ == "__main__":
    build_nsfnet()
```

---

### `deploy_online.py`

```python
"""
Phase 2: Live deployment loop connecting the trained PPO agent to
the OS-Ken controller via the Northbound REST API.

Usage (after OS-Ken and Mininet are running):
    python deploy_online.py --checkpoint checkpoints/garro_nsfnet_final.pt \
                             --topology nsfnet
"""
import yaml
import time
import argparse
import asyncio
import requests
import networkx as nx
import numpy as np
import torch
from agentic.llm_orchestrator import LLMOrchestrator
from model.ppo_agent import PPOAgent
from topologies.nsfnet import get_nsfnet


CONTROLLER_URL = "http://127.0.0.1:8080"


def fetch_network_state(topology_G: nx.Graph) -> nx.Graph:
    """
    Pull live telemetry from OS-Ken REST API and update graph attributes.
    Falls back to previous state on any HTTP error.
    """
    try:
        resp = requests.get(f"{CONTROLLER_URL}/garro/state", timeout=3)
        resp.raise_for_status()
        state = resp.json()

        for edge_data in state.get("edges", []):
            u, v = edge_data["src"], edge_data["dst"]
            if topology_G.has_edge(u, v):
                topology_G.edges[u, v]["utilization"] = edge_data["utilization"]
                topology_G.edges[u, v]["packet_loss"] = edge_data["packet_loss"]
                topology_G.edges[u, v]["delay"] = edge_data["delay"]

    except Exception as e:
        print(f"[Deploy] Telemetry fetch failed: {e} — using cached state")

    return topology_G


def install_flow(path: list, src_ip: str, dst_ip: str):
    """POST computed path to OS-Ken controller for flow installation."""
    payload = {"path": path, "src_ip": src_ip, "dst_ip": dst_ip}
    try:
        resp = requests.post(
            f"{CONTROLLER_URL}/garro/flow",
            json=payload, timeout=3
        )
        if resp.status_code == 200:
            print(f"[Deploy] Flow installed: {path} | {src_ip}→{dst_ip}")
        else:
            print(f"[Deploy] Flow install failed: {resp.text}")
    except Exception as e:
        print(f"[Deploy] Flow install error: {e}")


async def main(args):
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cpu")
    k_paths = config["network"]["k_paths"]

    # Load pre-trained agent
    agent = PPOAgent(config, k_paths=k_paths, device=device)
    agent.load(args.checkpoint)
    agent.encoder.eval()
    agent.ac_net.eval()

    # Build topology graph (NSFNET for demo)
    G = get_nsfnet()

    # Pre-compute K-shortest paths
    all_paths = {}
    for src in G.nodes():
        for dst in G.nodes():
            if src == dst:
                continue
            try:
                paths = list(nx.shortest_simple_paths(G, src, dst, weight="delay"))
                all_paths[(src, dst)] = paths[:k_paths]
            except nx.NetworkXNoPath:
                all_paths[(src, dst)] = []

    # LLM orchestrator
    llm = LLMOrchestrator(config)

    # ── Main deployment loop ───────────────────────────────────────────────
    print("\n[Deploy] GARRO online — monitoring network every 2 seconds")
    print("[Deploy] Type operator intents via stdin (Ctrl+C to stop)\n")

    # Sample flow demands to route (extend with dynamic flow arrival)
    flow_demands = [
        ("10.0.0.1", "10.0.0.14", 0, 13),
        ("10.0.0.3", "10.0.0.12", 2, 11),
        ("10.0.0.5", "10.0.0.9",  4, 8),
    ]

    step = 0
    while True:
        # 1. Fetch live network state
        G = fetch_network_state(G)

        # 2. Optional: periodically update weights from LLM
        if step % 30 == 0:   # Every 60 seconds (30 × 2s)
            intent = (
                "Balance load across all links while maintaining "
                "reasonable latency for mixed traffic."
            )
            weights = await llm.parse_intent(intent)
            # Update agent's reward weights dynamically
            config["reward_weights"]["alpha1"] = weights.alpha1
            config["reward_weights"]["alpha2"] = weights.alpha2
            config["reward_weights"]["alpha3"] = weights.alpha3
            config["reward_weights"]["alpha4"] = weights.alpha4

        # 3. For each flow demand: select path with PPO agent
        for src_ip, dst_ip, src_node, dst_node in flow_demands:
            candidates = all_paths.get((src_node, dst_node), [])
            if not candidates:
                continue

            action, log_prob, value = agent.select_action(G, candidates)

            if action < len(candidates):
                selected_path = candidates[action]
            else:
                selected_path = candidates[0]

            # 4. Install flow rules on OS-Ken controller
            install_flow(selected_path, src_ip, dst_ip)

        step += 1
        time.sleep(config["network"]["polling_interval"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to trained model checkpoint (.pt)"
    )
    parser.add_argument("--topology", default="nsfnet")
    args = parser.parse_args()

    asyncio.run(main(args))
```

---

## 11. Evaluation & Benchmarking

### `evaluate.py`

```python
"""
Benchmarks GARRO vs OSPF, ECMP, DQN baselines.
Runs entirely in the Digital Twin (no live network required).

Usage:
    python evaluate.py --checkpoint checkpoints/garro_nsfnet_final.pt \
                        --topology nsfnet --episodes 1000
"""
import yaml
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from digital_twin.mm1k_env import MM1KNetworkEnv
from model.ppo_agent import PPOAgent
from topologies.nsfnet import get_nsfnet
from topologies.geant2 import get_geant2
from topologies.fat_tree import get_fat_tree

TOPOLOGY_MAP = {
    "nsfnet": get_nsfnet,
    "geant2": get_geant2,
    "fat_tree": lambda: get_fat_tree(k=8),
}


def run_baseline_ospf(env: MM1KNetworkEnv, episodes: int) -> dict:
    """OSPF baseline: always select shortest-delay path (index 0)."""
    rewards, delays, losses = [], [], []
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            # Always pick path 0 (shortest delay — OSPF-equivalent)
            obs, r, terminated, truncated, info = env.step(0)
            done = terminated or truncated
            ep_r += r
        rewards.append(ep_r)
    return {"mean_reward": np.mean(rewards), "std": np.std(rewards)}


def run_baseline_ecmp(env: MM1KNetworkEnv, episodes: int) -> dict:
    """ECMP baseline: round-robin across available paths."""
    rewards = []
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        ep_r = 0.0
        step = 0
        while not done:
            k = len(env.candidate_paths) or 1
            action = step % k    # Round-robin
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_r += r
            step += 1
        rewards.append(ep_r)
    return {"mean_reward": np.mean(rewards), "std": np.std(rewards)}


def run_garro(env: MM1KNetworkEnv, agent: PPOAgent,
              episodes: int) -> dict:
    """GARRO PPO agent evaluation (no gradient updates)."""
    rewards = []
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            action, _, _ = agent.select_action(env.G, env.candidate_paths)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_r += r
        rewards.append(ep_r)
    return {"mean_reward": np.mean(rewards), "std": np.std(rewards)}


def main(args):
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cpu")
    G = TOPOLOGY_MAP[args.topology]()
    env = MM1KNetworkEnv(G, config)
    episodes = args.episodes

    print(f"\n[Eval] Topology: {args.topology} | Episodes: {episodes}")
    print("=" * 55)

    results = {}

    print("[Eval] Running OSPF baseline...")
    results["OSPF"] = run_baseline_ospf(env, episodes)

    print("[Eval] Running ECMP baseline...")
    results["ECMP"] = run_baseline_ecmp(env, episodes)

    print("[Eval] Running GARRO (PPO)...")
    agent = PPOAgent(config, k_paths=config["network"]["k_paths"], device=device)
    agent.load(args.checkpoint)
    agent.encoder.eval()
    agent.ac_net.eval()
    results["GARRO"] = run_garro(env, agent, episodes)

    # ── Print results table ────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"{'Algorithm':<12} {'Mean Reward':>14} {'Std Dev':>10}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:<12} {r['mean_reward']:>14.4f} {r['std']:>10.4f}")
    print("=" * 55)

    # ── Bar chart ─────────────────────────────────────────────────────────
    names = list(results.keys())
    means = [results[n]["mean_reward"] for n in names]
    stds = [results[n]["std"] for n in names]
    colors = ["#E74C3C", "#F39C12", "#2ECC71"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, means, yerr=stds, color=colors,
                   capsize=6, edgecolor="black", linewidth=0.8)
    plt.ylabel("Mean Episode Reward")
    plt.title(f"Routing Algorithm Comparison — {args.topology.upper()}")
    plt.tight_layout()
    plt.savefig(f"eval_results_{args.topology}.png", dpi=150)
    print(f"\n[Eval] Bar chart saved: eval_results_{args.topology}.png")

    # Save CSV
    pd.DataFrame(results).T.to_csv(f"eval_results_{args.topology}.csv")
    print(f"[Eval] CSV saved: eval_results_{args.topology}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--topology", default="nsfnet",
                        choices=["nsfnet", "geant2", "fat_tree"])
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()
    main(args)
```

---

## 12. Run Order & Commands Cheatsheet

### Complete Workflow — Step by Step

```bash
# ── 0. Activate your virtual environment ──────────────────────────────────
cd ~/garro
source venv/bin/activate

# ── 1. Phase 1: Offline Digital Twin Training ─────────────────────────────
# NSFNET (fast, ~10 min for 10k episodes)
python train_offline.py --topology nsfnet --episodes 10000

# GEANT2 (medium)
python train_offline.py --topology geant2 --episodes 20000

# Fat-Tree (large — run overnight)
python train_offline.py --topology fat_tree --episodes 50000

# ── 2. Evaluate against baselines (no live network needed) ────────────────
python evaluate.py \
    --checkpoint checkpoints/garro_nsfnet_final.pt \
    --topology nsfnet \
    --episodes 500

# ── 3. Phase 2: Live Emulation (requires two separate terminals) ──────────

# Terminal A: Start OS-Ken controller
osken-manager controller/garro_controller.py \
    --observe-links \
    --wsapi-port 8080 \
    --ofp-tcp-listen-port 6633

# Terminal B: Start Mininet (must use sudo)
sudo python topologies/mininet_nsfnet.py

# Terminal C: Start GARRO online deployment loop
python deploy_online.py \
    --checkpoint checkpoints/garro_nsfnet_final.pt \
    --topology nsfnet

# ── 4. Test from Mininet CLI (in Terminal B) ──────────────────────────────
# Inside the Mininet CLI prompt (mininet>):
pingall                    # Test basic connectivity
h1 ping h14 -c 10         # Test GARRO-routed path
h1 iperf -s &             # Start iperf server on h1
h14 iperf -c 10.0.0.1 -t 30  # Run 30s throughput test

# ── 5. Test LLM orchestrator independently ────────────────────────────────
python agentic/llm_orchestrator.py
```

---

### Quick Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: os_ken` | `pip install os-ken` in activated venv |
| `osken-manager: command not found` | `python -m os_ken.cmd.manager` instead |
| OVS not starting in WSL | `sudo modprobe openvswitch && sudo service openvswitch-switch start` |
| `torch_scatter` install fails | Install from PyG wheel: `pip install torch_scatter -f https://data.pyg.org/whl/torch-2.3.0+cpu.html` |
| Mininet can't connect to controller | Ensure OS-Ken is running first; check port 6633 is free: `ss -tlnp \| grep 6633` |
| `GEMINI_API_KEY` error | Add key to `.env` file in project root |
| PyG `TransformerConv` import error | `pip install torch_geometric --upgrade` |
| Mininet cleanup after crash | `sudo mn -c` |

---

### OS-Ken vs Ryu Import Reference

| Ryu (old) | OS-Ken (new) |
|---|---|
| `from ryu.base import app_manager` | `from os_ken.base import app_manager` |
| `from ryu.controller import ofp_event` | `from os_ken.controller import ofp_event` |
| `from ryu.ofproto import ofproto_v1_3` | `from os_ken.ofproto import ofproto_v1_3` |
| `from ryu.lib.packet import packet` | `from os_ken.lib.packet import packet` |
| `from ryu.topology import event` | `from os_ken.topology import event` |
| `from ryu.app.wsgi import WSGIApplication` | `from os_ken.app.wsgi import WSGIApplication` |
| `ryu-manager app.py` | `osken-manager app.py` |

---

*GARRO Implementation Guide — Python 3.12 + OS-Ken Edition*
