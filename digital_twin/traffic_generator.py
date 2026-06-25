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

    def reset(self) -> None:
        """
        Reset AR(1) state and step counter.

        **Must be called at the start of every RL episode** so that each
        episode begins from a fresh traffic history.  Without reset(), the
        AR(1) state leaks across episodes, which biases early-episode
        observations and makes the replay buffer inconsistent.
        """
        self._prev_T     = None
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