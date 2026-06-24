"""
Modulated Gravity Traffic Generator (TMGen).

Generates realistic bursty, non-stationary traffic matrices for use
in offline Digital Twin training and evaluation.

The Modulated Gravity Model produces traffic matrices where:
    T[i][j] ∝ pop_i × pop_j × distance_factor(i, j) × burst_factor(t)

This approximates empirical WAN/DC traffic distributions with
controllable coefficient of variance (CoV) for microburst severity.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from typing import List, Optional


class TrafficGenerator:
    """
    Modulated Gravity traffic matrix generator.

    Parameters
    ----------
    graph       : nx.Graph        Network topology (used for node count / distances)
    base_rate   : float           Base traffic intensity in Mbps
    cov         : float           Coefficient of Variance for burstiness (0 = flat)
    burst_prob  : float           Probability that any time step is a microburst
    burst_scale : float           Traffic multiplier during a microburst event
    seed        : int | None      NumPy RNG seed for reproducibility
    """

    def __init__(
        self,
        graph: nx.Graph,
        base_rate: float   = 100.0,
        cov: float         = 0.5,
        burst_prob: float  = 0.10,
        burst_scale: float = 5.0,
        seed: Optional[int] = None,
    ):
        self.G           = graph
        self.base_rate   = base_rate
        self.cov         = cov
        self.burst_prob  = burst_prob
        self.burst_scale = burst_scale
        self.rng         = np.random.default_rng(seed)

        self.nodes: List[int] = sorted(graph.nodes())
        self.n = len(self.nodes)

        # ── Pre-compute gravity model "populations" ─────────────────────────
        # Use node degree as a proxy for population (higher-degree = more traffic)
        degrees = np.array(
            [graph.degree(n) for n in self.nodes], dtype=np.float64
        )
        degrees = np.clip(degrees, 1.0, None)           # Avoid zero population
        self._pop = degrees / degrees.sum()              # Normalised

        # ── Pre-compute all-pairs shortest path lengths (for gravity factor) ─
        try:
            spl = dict(nx.all_pairs_dijkstra_path_length(graph, weight="delay"))
            dist_matrix = np.zeros((self.n, self.n), dtype=np.float64)
            for i, u in enumerate(self.nodes):
                for j, v in enumerate(self.nodes):
                    dist_matrix[i, j] = spl[u].get(v, 1.0)
        except Exception:
            dist_matrix = np.ones((self.n, self.n), dtype=np.float64)

        self._dist = dist_matrix

    def _gravity_matrix(self) -> np.ndarray:
        """Compute the base gravity traffic matrix T[i,j] ∝ pop_i × pop_j / dist_ij."""
        T = np.outer(self._pop, self._pop)             # [n, n]
        # Distance decay (gravity model: T ∝ pop_i × pop_j / dist^2)
        with np.errstate(divide="ignore", invalid="ignore"):
            decay = np.where(self._dist > 0, 1.0 / (self._dist ** 2), 0.0)
        np.fill_diagonal(decay, 0.0)
        T = T * decay
        row_sum = T.sum(axis=1, keepdims=True)
        T = np.where(row_sum > 0, T / row_sum, 0.0)   # Row-normalise
        return T * self.base_rate                       # Scale to Mbps

    def generate(self, num_steps: int = 1) -> np.ndarray:
        """
        Generate a traffic matrix (or time-series of matrices).

        Parameters
        ----------
        num_steps : int
            Number of time steps to generate. Each step applies
            an independent burst/noise realisation.

        Returns
        -------
        np.ndarray  Shape (num_steps, n, n) if num_steps > 1,
                    else (n, n).
                    Values are traffic demands in Mbps.
        """
        base = self._gravity_matrix()
        results = []

        for _ in range(num_steps):
            # Gaussian noise with controlled CoV
            noise = self.rng.normal(
                loc=1.0, scale=self.cov, size=base.shape
            )
            noise = np.clip(noise, 0.01, None)         # Ensure positive

            # Microburst: random subset of (src, dst) pairs spike
            burst_mask = self.rng.random(base.shape) < self.burst_prob
            noise[burst_mask] *= self.burst_scale

            T = base * noise
            np.fill_diagonal(T, 0.0)                   # No self-traffic
            T = np.clip(T, 0.0, None)
            results.append(T)

        arr = np.stack(results, axis=0)
        return arr[0] if num_steps == 1 else arr

    def sample_flow_demand(self) -> tuple[int, int, float]:
        """
        Sample a single (src, dst, demand_Mbps) flow demand
        weighted by the gravity model distribution.

        Returns
        -------
        (src_node, dst_node, demand_Mbps)
        """
        T = self.generate()
        # Flatten and sample proportionally to demand
        flat = T.flatten()
        flat_sum = flat.sum()
        if flat_sum == 0:
            i, j = self.rng.integers(0, self.n, size=2)
            while i == j:
                j = self.rng.integers(0, self.n)
            return self.nodes[i], self.nodes[j], self.base_rate

        probs = flat / flat_sum
        idx   = self.rng.choice(len(flat), p=probs)
        i, j  = divmod(idx, self.n)
        demand = float(T[i, j])
        return self.nodes[i], self.nodes[j], demand
