"""
Modulated Gravity Traffic Generator (TMGen) — v2.

Three key upgrades over v1:

  1. AR(1) Temporal Autocorrelation
     Traffic at step t is a convex blend of the previous matrix and a fresh
     gravity+shock "target", controlled by `ar_coeff` (φ).

         T(t) = φ · T(t-1)  +  (1-φ) · [G · M(t) · shock(t)]

     φ = 0  →  fully IID (original behaviour)
     φ ≈ 0.85–0.95  →  realistic wave-like congestion that an RL agent can
                        observe and route around before queues form.

  2. Decoupled Population Weights
     Pass a custom `populations` array to reflect real hotspot topology.
     If None, a Dirichlet sample is drawn once at construction time, so the
     generator never silently degenerates to a deterministic degree-centrality
     proxy — every unseeded run sees a fresh traffic pattern.

  3. Time-of-Day Modulation
     A cosine wave scales `base_rate` around a configurable peak hour:

         M(t) = 1  +  A · cos( 2π(h(t) − peak_hour) / 24 )

     where h(t) = (step_count / steps_per_hour) mod 24.
     Default: A=0.4, peak_hour=16 → ±40 % swing peaking at 4 PM.

─────────────────────────────────────────────────────────────────────────────
Why this fixes GARRO's ep-5000 collapse
─────────────────────────────────────────────────────────────────────────────
With IID noise the agent sees white-noise matrices between steps; there is no
causal structure to exploit, so it effectively learns a static routing policy
that happens to beat OSPF on average.  With AR(1) + ToD:

  • Congestion *builds* over multiple steps → the agent can observe the ramp
    and pre-emptively shift traffic before links saturate.
  • The sinusoidal envelope creates a predictable intra-day regime: learning
    to ramp up / down capacity reservation generalises across episodes.
  • `reset()` clears AR state cleanly between episodes so the agent cannot
    memorise a specific trajectory — it must learn the *process*, not a path.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from typing import List, Optional


_TWO_PI = 2.0 * np.pi


class TrafficGenerator:
    """
    Modulated Gravity traffic matrix generator with AR(1) dynamics
    and time-of-day modulation.

    Parameters
    ----------
    graph           : nx.Graph
        Network topology — used for node list and shortest-path distances.
    base_rate       : float
        Mean traffic intensity (Mbps) at the ToD peak hour.
    cov             : float
        Coefficient of Variation for per-step Gaussian shock.
        0 → flat deterministic matrices; 0.5 → moderate noise.
    burst_prob      : float
        Per-(i,j)-per-step probability of a microburst event.
    burst_scale     : float
        Traffic multiplier applied to bursting (i,j) pairs.
    ar_coeff        : float  ∈ [0, 1)
        AR(1) autoregressive coefficient φ.
        Recommended range: 0.80–0.95 for realistic WAN/DC traffic.
    populations     : array-like | None
        Unnormalised node "population" weights.  Length must equal the
        number of nodes in `graph`.
        None  →  draw a random Dirichlet(1) sample (uniform-on-simplex).
    tod_amplitude   : float  ∈ [0, 1]
        Fractional amplitude A of the time-of-day cosine.
        0 → no ToD effect;  0.4 → ±40 % swing around base_rate.
    tod_peak_hour   : float
        Simulated hour at which traffic peaks (default 16 = 4 PM).
    steps_per_hour  : float
        Number of `generate` steps that represent one simulated hour.
        E.g. if one env step = 1 minute, set steps_per_hour=60.
    seed            : int | None
        NumPy RNG seed for reproducibility.
    """

    def __init__(
        self,
        graph: nx.Graph,
        base_rate: float               = 100.0,
        cov: float                     = 0.5,
        burst_prob: float              = 0.10,
        burst_scale: float             = 5.0,
        ar_coeff: float                = 0.85,
        populations: Optional[np.ndarray] = None,
        tod_amplitude: float           = 0.4,
        tod_peak_hour: float           = 16.0,
        steps_per_hour: float          = 60.0,
        seed: Optional[int]            = None,
    ) -> None:
        self.G              = graph
        self.base_rate      = base_rate
        self.cov            = cov
        self.burst_prob     = burst_prob
        self.burst_scale    = burst_scale
        self.ar_coeff       = float(np.clip(ar_coeff, 0.0, 0.9999))
        self.tod_amplitude  = tod_amplitude
        self.tod_peak_hour  = tod_peak_hour
        self.steps_per_hour = max(float(steps_per_hour), 1e-9)
        self.rng            = np.random.default_rng(seed)

        self.nodes: List[int] = sorted(graph.nodes())
        self.n = len(self.nodes)

        # ── 1.  Population weights ─────────────────────────────────────────
        if populations is not None:
            pop = np.asarray(populations, dtype=np.float64).ravel()
            if len(pop) != self.n:
                raise ValueError(
                    f"'populations' has length {len(pop)} but graph has "
                    f"{self.n} nodes."
                )
            pop = np.clip(pop, 1e-12, None)
        else:
            # Dirichlet(α=1) = uniform distribution over the probability
            # simplex → every run gets a different but valid traffic hotspot
            # pattern; no two unseeded instances share the same structure.
            pop = self.rng.dirichlet(np.ones(self.n))

        self._pop: np.ndarray = pop / pop.sum()   # normalise once

        # ── 2.  All-pairs shortest path distances (gravity decay factor) ───
        try:
            spl = dict(nx.all_pairs_dijkstra_path_length(graph, weight="delay"))
            dist = np.zeros((self.n, self.n), dtype=np.float64)
            for i, u in enumerate(self.nodes):
                for j, v in enumerate(self.nodes):
                    dist[i, j] = spl[u].get(v, 1.0)
        except Exception:
            dist = np.ones((self.n, self.n), dtype=np.float64)

        self._dist: np.ndarray = dist

        # ── 3.  AR(1) state ───────────────────────────────────────────────
        # _prev_T is None until the first call to generate(); the first step
        # warm-starts from the pure target matrix (no prior history).
        self._prev_T: Optional[np.ndarray] = None
        self._step_count: int = 0

        # Cache the static gravity skeleton (recomputed if graph changes)
        self._base_gravity: np.ndarray = self._compute_gravity()

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _compute_gravity(self) -> np.ndarray:
        """
        Normalised gravity matrix scaled to base_rate (Mbps).

            G[i,j]  ∝  pop_i × pop_j / dist(i,j)²

        Diagonal is zeroed (no self-traffic).  Rows are normalised so that
        the *mean* row-sum equals base_rate.
        """
        T = np.outer(self._pop, self._pop)                       # [n, n]
        with np.errstate(divide="ignore", invalid="ignore"):
            decay = np.where(self._dist > 0, 1.0 / self._dist ** 2, 0.0)
        np.fill_diagonal(decay, 0.0)
        T *= decay
        row_sum = T.sum(axis=1, keepdims=True)
        T = np.where(row_sum > 0, T / row_sum, 0.0)             # row-normalise
        return T * self.base_rate

    def _tod_multiplier(self, step: int) -> float:
        """
        Time-of-Day multiplier M(t) using a cosine centred on peak_hour.

            h(t) = (step / steps_per_hour)  mod 24
            M(t) = 1  +  A · cos( 2π(h(t) − peak_hour) / 24 )

        Guarantees M(t) ∈ [1-A, 1+A] ⊆ (0, 2] for A ∈ [0,1].
        """
        hour = (step / self.steps_per_hour) % 24.0
        return 1.0 + self.tod_amplitude * np.cos(
            _TWO_PI * (hour - self.tod_peak_hour) / 24.0
        )

    def _shock(self, shape: tuple) -> np.ndarray:
        """
        Multiplicative noise matrix: Gaussian noise with microburst overlay.

        Each entry is drawn as N(1, cov²) and clipped to (0.01, ∞).
        A Bernoulli mask selects `burst_prob` fraction of entries and
        multiplies them by `burst_scale`.
        """
        noise = self.rng.normal(loc=1.0, scale=self.cov, size=shape)
        noise = np.clip(noise, 0.01, None)
        burst_mask = self.rng.random(shape) < self.burst_prob
        noise[burst_mask] *= self.burst_scale
        return noise

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset AR(1) state and optionally reseed the RNG.

        Parameters
        ----------
        seed : int | None
            When provided, the traffic generator is reseeded so that
            evaluation is fully reproducible. During training, leave
            this as None to preserve stochasticity.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._prev_T = None
        self._step_count = 0

    def generate(self, num_steps: int = 1) -> np.ndarray:
        """
        Generate one or more traffic matrices under the AR(1) model.

        At each step s the update rule is:

            T_target(s) = G · M(s) · shock(s)       # gravity + ToD + noise
            T(s)        = φ · T(s-1) + (1-φ) · T_target(s)  # AR(1) blend

        On the very first call after construction or reset(), T(s-1) is
        initialised to T_target(0) (warm-start, no arbitrary cold-start
        artefact).

        Parameters
        ----------
        num_steps : int
            Number of consecutive time steps to generate.

        Returns
        -------
        np.ndarray
            Shape (n, n) when num_steps == 1, else (num_steps, n, n).
            All values are non-negative traffic demands in Mbps.
        """
        G   = self._base_gravity
        phi = self.ar_coeff
        results: List[np.ndarray] = []

        for _ in range(num_steps):
            tod    = self._tod_multiplier(self._step_count)
            shock  = self._shock(G.shape)

            # Target: what traffic "wants" to be this step
            T_target = G * tod * shock
            np.fill_diagonal(T_target, 0.0)
            T_target = np.clip(T_target, 0.0, None)

            if self._prev_T is None:
                # Warm-start: first step has no prior history
                T = T_target.copy()
            else:
                # AR(1): smooth blend from previous realisation
                T = phi * self._prev_T + (1.0 - phi) * T_target

            np.fill_diagonal(T, 0.0)
            T = np.clip(T, 0.0, None)

            self._prev_T    = T
            self._step_count += 1
            results.append(T)

        arr = np.stack(results, axis=0)   # (num_steps, n, n)
        return arr[0] if num_steps == 1 else arr

    def sample_flow_demand(self) -> tuple[int, int, float]:
        """
        Sample a single (src, dst, demand_Mbps) triple, weighted by the
        current AR(1)+ToD traffic distribution.

        Returns
        -------
        (src_node, dst_node, demand_Mbps)
        """
        T        = self.generate()        # advances step_count by 1
        flat     = T.ravel()
        flat_sum = flat.sum()

        if flat_sum == 0:
            # Degenerate fallback
            i = self.rng.integers(0, self.n)
            j = self.rng.integers(0, self.n - 1)
            if j >= i:
                j += 1
            return self.nodes[i], self.nodes[j], self.base_rate

        probs = flat / flat_sum
        idx   = self.rng.choice(len(flat), p=probs)
        i, j  = divmod(int(idx), self.n)
        return self.nodes[i], self.nodes[j], float(T[i, j])

    # ──────────────────────────────────────────────────────────────────────
    # Read-only properties
    # ──────────────────────────────────────────────────────────────────────

    @property
    def step_count(self) -> int:
        """Cumulative number of steps generated since last reset()."""
        return self._step_count

    @property
    def current_hour(self) -> float:
        """Simulated hour-of-day at the current step (0.0 – 24.0)."""
        return (self._step_count / self.steps_per_hour) % 24.0

    @property
    def current_tod_multiplier(self) -> float:
        """Live time-of-day multiplier M at the current step."""
        return self._tod_multiplier(self._step_count)

    @property
    def populations(self) -> np.ndarray:
        """Normalised node population weights (read-only copy)."""
        return self._pop.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Data Center Network (DCN) Traffic Generator
# ─────────────────────────────────────────────────────────────────────────────

class DataCenterTrafficGenerator:
    """
    Data Center Network (DCN) Traffic Matrix Generator.

    Implements production-grade DCN traffic physics:
      1. Dual-Wave Diurnal Time-of-Day Modulation (Interactive + Batch):
         M(t) = 1.0 + A_day * cos(2π(h - h_day)/24) + A_batch * exp(-0.5 * ((h - h_batch)/σ)^2)
         Superimposes daytime interactive queries with nighttime batch processing (MapReduce,
         database backups, ML training checkpoint syncs).

      2. Heavy-Tailed Bimodal Flow Sizing (Pareto Elephants + Lognormal Mice):
         Simulates the classic DCN heavy-tailed distribution: ~85% mice flows (<100 KB) and ~15%
         elephant flows (>10 MB - 10 GB) that carry >80% of total bytes.
         Elephant flows are drawn from Pareto(α) with shape parameter α ∈ [1.05, 1.25].

      3. Spatial Communication Patterns (Hedera / SIGCOMM Benchmarks):
         - Staggered Locality: P_edge (intra-rack), P_pod (intra-pod), P_core (inter-pod).
         - Stride Pattern: host_i -> (host_i + stride) mod N (stresses bisection bandwidth).
         - Incast Microbursts: Many-to-one synchronization onto a single aggregator edge switch.
         - All-to-All / Ring All-Reduce: Synchronized gradient exchange across GPU workers.
         - Mixed Mode: Dynamically schedules across all patterns over time.

      4. AR(1) Temporal Autocorrelation:
         T(t) = φ · T(t-1) + (1-φ) · T_target(t)
         Ensures congestion builds up realistically so DRL policies can observe
         telemetry ramp-ups and preemptively steer flows.
    """

    def __init__(
        self,
        graph: nx.Graph,
        base_rate: float = 100.0,
        k: Optional[int] = None,
        edge_nodes: Optional[List[int]] = None,
        pattern: str = "mixed",
        p_edge: float = 0.20,
        p_pod: float = 0.30,
        p_core: float = 0.50,
        stride: Optional[int] = None,
        elephant_prob: float = 0.15,
        elephant_alpha: float = 1.15,
        elephant_scale: float = 8.0,
        mice_cov: float = 0.25,
        incast_prob: float = 0.08,
        incast_senders: int = 6,
        incast_scale: float = 5.0,
        allreduce_interval: int = 120,
        allreduce_scale: float = 3.0,
        tod_amplitude_day: float = 0.35,
        tod_peak_hour_day: float = 15.0,
        tod_amplitude_batch: float = 0.45,
        tod_peak_hour_batch: float = 3.0,
        tod_sigma_batch: float = 1.8,
        steps_per_hour: float = 60.0,
        ar_coeff: float = 0.88,
        max_flow_rate: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.G = graph
        self.base_rate = float(base_rate)
        self.pattern = str(pattern).lower()
        self.p_edge = float(p_edge)
        self.p_pod = float(p_pod)
        self.p_core = float(p_core)
        self.elephant_prob = float(np.clip(elephant_prob, 0.0, 1.0))
        self.elephant_alpha = max(1.01, float(elephant_alpha))
        self.elephant_scale = max(1.0, float(elephant_scale))
        self.mice_cov = max(0.01, float(mice_cov))
        self.incast_prob = float(np.clip(incast_prob, 0.0, 1.0))
        self.incast_senders = max(2, int(incast_senders))
        self.incast_scale = max(1.0, float(incast_scale))
        self.allreduce_interval = max(10, int(allreduce_interval))
        self.allreduce_scale = max(1.0, float(allreduce_scale))
        self.tod_amplitude_day = float(tod_amplitude_day)
        self.tod_peak_hour_day = float(tod_peak_hour_day)
        self.tod_amplitude_batch = float(tod_amplitude_batch)
        self.tod_peak_hour_batch = float(tod_peak_hour_batch)
        self.tod_sigma_batch = max(0.1, float(tod_sigma_batch))
        self.steps_per_hour = max(1e-9, float(steps_per_hour))
        self.ar_coeff = float(np.clip(ar_coeff, 0.0, 0.9999))
        self.rng = np.random.default_rng(seed)

        self.nodes: List[int] = sorted(graph.nodes())
        self.n: int = len(self.nodes)
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}

        # Detect or set maximum flow rate ceiling
        if max_flow_rate is not None:
            self.max_flow_rate = float(max_flow_rate)
        else:
            bws = [attrs.get("bandwidth", 1000.0) for _, _, attrs in graph.edges(data=True)]
            max_bw = max(bws) if bws else 10000.0
            self.max_flow_rate = float(max_bw * 1.2)  # Cap at 120% link bandwidth

        # ── Topology Tiers & Pod Partitioning ──────────────────────────────
        total = self.n
        inferred_k = None
        if k is not None:
            inferred_k = k
        else:
            cand_k = int(round(np.sqrt(4.0 * total / 5.0)))
            if cand_k > 0 and cand_k % 2 == 0 and (5 * (cand_k ** 2) // 4) == total:
                inferred_k = cand_k

        self.k = inferred_k
        if edge_nodes is not None:
            self.edge_nodes = sorted(edge_nodes)
        elif self.k is not None:
            num_core = (self.k // 2) ** 2
            num_agg = self.k * (self.k // 2)
            edge_start = num_core + num_agg
            self.edge_nodes = list(range(edge_start, total))
        else:
            # Fallback: lowest-degree nodes as edge switches
            degrees = dict(graph.degree())
            min_deg = min(degrees.values()) if degrees else 0
            self.edge_nodes = [n for n, d in degrees.items() if d == min_deg]
            if not self.edge_nodes or len(self.edge_nodes) == total:
                self.edge_nodes = list(self.nodes)

        self._edge_indices = [self.node_to_idx[n] for n in self.edge_nodes]
        self.n_edges_sw = len(self.edge_nodes)

        # Build Pod Map for edge switches
        self.pod_map: dict[int, int] = {}
        if self.k is not None and self.edge_nodes:
            num_core = (self.k // 2) ** 2
            num_agg = self.k * (self.k // 2)
            edge_start = num_core + num_agg
            k_half = self.k // 2
            for e in self.edge_nodes:
                self.pod_map[e] = (e - edge_start) // k_half
        else:
            for idx, e in enumerate(self.edge_nodes):
                self.pod_map[e] = idx % 2

        self.stride = stride if stride is not None else max(1, self.n_edges_sw // 2)

        # ── State Tracking ────────────────────────────────────────────────
        self._prev_T: Optional[np.ndarray] = None
        self._step_count: int = 0
        self._last_pattern_desc: str = self.pattern

    def _tod_multiplier(self, step: int) -> float:
        """Dual-wave diurnal multiplier (Interactive daytime + Nighttime batch)."""
        hour = (step / self.steps_per_hour) % 24.0
        day_wave = self.tod_amplitude_day * np.cos(
            _TWO_PI * (hour - self.tod_peak_hour_day) / 24.0
        )
        diff = (hour - self.tod_peak_hour_batch + 12.0) % 24.0 - 12.0
        batch_wave = self.tod_amplitude_batch * np.exp(
            -0.5 * (diff / self.tod_sigma_batch) ** 2
        )
        mult = 1.0 + day_wave + batch_wave
        return float(np.clip(mult, 0.1, 3.0))

    def _generate_target_matrix(self) -> np.ndarray:
        """Construct the instantaneous target traffic matrix based on DCN physics."""
        T = np.zeros((self.n, self.n), dtype=np.float64)
        if self.n_edges_sw < 2:
            return T

        curr_pattern = self.pattern

        if self.pattern == "mixed":
            phase = self._step_count % self.allreduce_interval
            if phase < 4:
                curr_pattern = "allreduce"
            elif self.rng.random() < self.incast_prob:
                curr_pattern = "incast"
            else:
                curr_pattern = "stride" if ((self._step_count // 25) % 2 == 1) else "staggered"

        self._last_pattern_desc = curr_pattern

        # Base spatial pattern assignment
        if curr_pattern == "allreduce":
            # Distributed ML all-to-all collective exchange
            for i in self._edge_indices:
                for j in self._edge_indices:
                    if i != j:
                        T[i, j] = self.base_rate * self.allreduce_scale

        elif curr_pattern == "incast":
            # Many-to-one synchronization onto an aggregator edge switch
            agg_idx = self.rng.choice(self._edge_indices)
            sender_candidates = [idx for idx in self._edge_indices if idx != agg_idx]
            k_senders = min(self.incast_senders, len(sender_candidates))
            chosen_senders = self.rng.choice(sender_candidates, size=k_senders, replace=False)
            for s in chosen_senders:
                T[s, agg_idx] = self.base_rate * self.incast_scale
            # Normal background communication between remaining switches
            for i in self._edge_indices:
                for j in self._edge_indices:
                    if i != j and not (j == agg_idx and i in chosen_senders):
                        T[i, j] = self.base_rate * 0.4

        elif curr_pattern == "stride":
            # Host/edge i sends to (i + stride) mod N_edge
            for rel_i, abs_i in enumerate(self._edge_indices):
                rel_j = (rel_i + self.stride) % self.n_edges_sw
                abs_j = self._edge_indices[rel_j]
                if abs_i != abs_j:
                    T[abs_i, abs_j] = self.base_rate * 2.5
            # Low background communication
            for i in self._edge_indices:
                for j in self._edge_indices:
                    if i != j and T[i, j] == 0:
                        T[i, j] = self.base_rate * 0.2

        else: # "staggered"
            for abs_i in self._edge_indices:
                u = self.nodes[abs_i]
                u_pod = self.pod_map.get(u, 0)
                same_pod_indices = [
                    self.node_to_idx[v] for v in self.edge_nodes
                    if v != u and self.pod_map.get(v, 0) == u_pod
                ]
                diff_pod_indices = [
                    self.node_to_idx[v] for v in self.edge_nodes
                    if self.pod_map.get(v, 0) != u_pod
                ]

                # Distribute intra-pod load
                n_same = len(same_pod_indices)
                if n_same > 0:
                    share = (self.base_rate * self.p_pod) / n_same
                    for abs_j in same_pod_indices:
                        T[abs_i, abs_j] = share

                # Distribute inter-pod (core-crossing) load
                n_diff = len(diff_pod_indices)
                if n_diff > 0:
                    share = (self.base_rate * self.p_core) / n_diff
                    for abs_j in diff_pod_indices:
                        T[abs_i, abs_j] = share

        # Apply Bimodal Flow Sizing (Pareto Elephant vs Lognormal Mice)
        mask = T > 0.0
        n_active = int(np.sum(mask))
        if n_active > 0:
            elephant_mask = (self.rng.random(T.shape) < self.elephant_prob) & mask
            mice_mask = mask & (~elephant_mask)

            # Elephants: Pareto(α)
            if np.any(elephant_mask):
                pareto_samples = (self.rng.pareto(self.elephant_alpha, size=T.shape) + 1.0)
                pareto_samples = np.clip(pareto_samples, 1.0, 15.0) * self.elephant_scale
                T[elephant_mask] *= pareto_samples[elephant_mask]

            # Mice: Lognormal
            if np.any(mice_mask):
                mice_samples = self.rng.lognormal(mean=0.0, sigma=self.mice_cov, size=T.shape)
                T[mice_mask] *= mice_samples[mice_mask]

        # Apply Time-of-Day Multiplier and clip to link physical capacity
        tod = self._tod_multiplier(self._step_count)
        T *= tod
        np.fill_diagonal(T, 0.0)
        return np.clip(T, 0.0, self.max_flow_rate)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal AR(1) state and step counter."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._prev_T = None
        self._step_count = 0

    def generate(self, num_steps: int = 1) -> np.ndarray:
        """
        Generate one or more traffic matrices under the DCN AR(1) model.
        Returns shape (n, n) when num_steps == 1, else (num_steps, n, n).
        """
        results: List[np.ndarray] = []
        phi = self.ar_coeff

        for _ in range(num_steps):
            T_target = self._generate_target_matrix()
            if self._prev_T is None:
                T = T_target.copy()
            else:
                T = phi * self._prev_T + (1.0 - phi) * T_target

            np.fill_diagonal(T, 0.0)
            T = np.clip(T, 0.0, self.max_flow_rate)
            self._prev_T = T
            self._step_count += 1
            results.append(T)

        arr = np.stack(results, axis=0)
        return arr[0] if num_steps == 1 else arr

    def sample_flow_pair(self, rng=None) -> Tuple[int, int]:
        """Sample a communicating (src, dst) pair among active edge switches."""
        r = rng if rng is not None else self.rng
        if self._prev_T is not None and self._prev_T.sum() > 0:
            edge_sub = self._prev_T[np.ix_(self._edge_indices, self._edge_indices)].copy()
            np.fill_diagonal(edge_sub, 0.0)
            sub_sum = edge_sub.sum()
            if sub_sum > 0:
                probs = edge_sub.ravel() / sub_sum
                idx = r.choice(len(probs), p=probs)
                i_rel, j_rel = divmod(int(idx), self.n_edges_sw)
                return self.edge_nodes[i_rel], self.edge_nodes[j_rel]

        src_i = r.integers(0, self.n_edges_sw)
        dst_i = r.integers(0, self.n_edges_sw - 1)
        if dst_i >= src_i:
            dst_i += 1
        return self.edge_nodes[src_i], self.edge_nodes[dst_i]

    def sample_flow_demand(self) -> Tuple[int, int, float]:
        """Sample a (src, dst, demand_Mbps) tuple from the live traffic state."""
        T = self.generate()
        src, dst = self.sample_flow_pair()
        i = self.node_to_idx[src]
        j = self.node_to_idx[dst]
        return src, dst, float(T[i, j])

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def current_hour(self) -> float:
        return (self._step_count / self.steps_per_hour) % 24.0

    @property
    def current_tod_multiplier(self) -> float:
        return self._tod_multiplier(self._step_count)

    @property
    def active_pattern(self) -> str:
        return self._last_pattern_desc


# ─────────────────────────────────────────────────────────────────────────────
# Factory Function
# ─────────────────────────────────────────────────────────────────────────────

def create_traffic_generator(
    graph: nx.Graph,
    config: dict,
    base_rate: Optional[float] = None,
    seed: Optional[int] = None,
) -> TrafficGenerator | DataCenterTrafficGenerator:
    """
    Factory function to instantiate the appropriate traffic generator.

    If the topology is Fat-Tree or if datacenter_traffic.enabled is True,
    returns an instance of DataCenterTrafficGenerator.
    Otherwise, returns the standard TrafficGenerator (WAN gravity model).
    """
    topo = str(config.get("network", {}).get("topology", "")).lower()
    dc_cfg = config.get("datacenter_traffic", {})
    mm_cfg = config.get("mm1k", {})

    use_dc = ("fat_tree" in topo) or dc_cfg.get("enabled", False)

    if use_dc:
        br = base_rate if base_rate is not None else dc_cfg.get("base_rate", mm_cfg.get("base_arrival_rate", 100.0))
        return DataCenterTrafficGenerator(
            graph=graph,
            base_rate=br,
            k=dc_cfg.get("k", None),
            edge_nodes=dc_cfg.get("edge_nodes", None),
            pattern=dc_cfg.get("pattern", "mixed"),
            p_edge=dc_cfg.get("p_edge", 0.20),
            p_pod=dc_cfg.get("p_pod", 0.30),
            p_core=dc_cfg.get("p_core", 0.50),
            stride=dc_cfg.get("stride", None),
            elephant_prob=dc_cfg.get("elephant_prob", 0.15),
            elephant_alpha=dc_cfg.get("elephant_alpha", 1.15),
            elephant_scale=dc_cfg.get("elephant_scale", 8.0),
            mice_cov=dc_cfg.get("mice_cov", 0.25),
            incast_prob=dc_cfg.get("incast_prob", 0.08),
            incast_senders=dc_cfg.get("incast_senders", 6),
            incast_scale=dc_cfg.get("incast_scale", 5.0),
            allreduce_interval=dc_cfg.get("allreduce_interval", 120),
            allreduce_scale=dc_cfg.get("allreduce_scale", 3.0),
            tod_amplitude_day=dc_cfg.get("tod_amplitude_day", 0.35),
            tod_peak_hour_day=dc_cfg.get("tod_peak_hour_day", 15.0),
            tod_amplitude_batch=dc_cfg.get("tod_amplitude_batch", 0.45),
            tod_peak_hour_batch=dc_cfg.get("tod_peak_hour_batch", 3.0),
            tod_sigma_batch=dc_cfg.get("tod_sigma_batch", 1.8),
            steps_per_hour=dc_cfg.get("steps_per_hour", mm_cfg.get("steps_per_hour", 60.0)),
            ar_coeff=dc_cfg.get("ar_coeff", mm_cfg.get("ar_coeff", 0.88)),
            max_flow_rate=dc_cfg.get("max_flow_rate", None),
            seed=seed if seed is not None else config.get("seed", 42),
        )
    else:
        br = base_rate if base_rate is not None else mm_cfg.get("base_arrival_rate", 100.0)
        return TrafficGenerator(
            graph=graph,
            base_rate=br,
            cov=mm_cfg.get("cov", 0.5),
            burst_prob=mm_cfg.get("burst_prob", 0.10),
            burst_scale=mm_cfg.get("burst_scale", 5.0),
            ar_coeff=mm_cfg.get("ar_coeff", 0.85),
            populations=None,
            tod_amplitude=mm_cfg.get("tod_amplitude", 0.4),
            tod_peak_hour=mm_cfg.get("tod_peak_hour", 16.0),
            steps_per_hour=mm_cfg.get("steps_per_hour", 60.0),
            seed=seed if seed is not None else config.get("seed", 42),
        )