"""
Two-Tier DCN Elephant Flow Scheduler for GARRO.

In Data Center Networks (DCN), millions of flows traverse switches per second.
Invoking a centralized SDN controller or Graph Transformer inference for every
flow creates a crippling control-plane bottleneck.

This two-tier architecture resolves the scalability constraint:
  1. Tier 1 (Data-Plane Hardware ECMP):
     - Handles Mice Flows (rate <= threshold, ~95% of all flows).
     - Forwarded at hardware line-rate with sub-microsecond latency.
  2. Tier 2 (Centralized DRL / Graph Attention Orchestrator):
     - Triggered exclusively for Elephant Flows (rate > threshold).
     - GARRO evaluates global graph congestion and installs high-priority
       explicit forwarding rules to prevent core-layer hash collisions.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np


class TwoTierDCNScheduler:
    """
    Two-Tier Hybrid Scheduler for Data Center Networks.

    Parameters
    ----------
    graph : nx.Graph
        Fat-Tree or target DCN topology.
    agent : Any
        GARRO PPOAgent instance (or None for fallback).
    elephant_threshold_mbps : float
        Flow bandwidth threshold (Mbps) to qualify as an elephant flow.
        Default is 1000 Mbps (10% of a 10 Gbps link).
    """

    def __init__(
        self,
        graph: nx.Graph,
        agent: Optional[object] = None,
        elephant_threshold_mbps: float = 1000.0,
    ) -> None:
        self.G = graph
        self.agent = agent
        self.elephant_threshold = elephant_threshold_mbps

        self.total_flows_processed = 0
        self.mice_flows_count = 0
        self.elephant_flows_count = 0
        self.elephant_reroutes = 0

    def classify_flow(self, demand_mbps: float) -> str:
        """Classify flow into 'mice' or 'elephant'."""
        return "elephant" if demand_mbps >= self.elephant_threshold else "mice"

    def schedule_flow(
        self,
        src: int,
        dst: int,
        demand_mbps: float,
        candidate_paths: List[List[int]],
        step: int = 0,
    ) -> Tuple[List[int], str, int]:
        """
        Schedule a flow using the two-tier DCN policy.

        Returns
        -------
        selected_path : List[int]
        tier_used     : str ('tier1_ecmp' or 'tier2_garro')
        path_idx      : int
        """
        self.total_flows_processed += 1
        flow_type = self.classify_flow(demand_mbps)

        if not candidate_paths:
            return [], "fallback", 0

        # Calculate equal-cost candidate paths
        costs = [len(p) - 1 for p in candidate_paths]
        min_cost = min(costs)
        equal_cost_indices = [i for i, c in enumerate(costs) if c == min_cost]

        # ── Tier 1: Mice Flow → Fast Equal-Cost Hash / Round-Robin ───
        if flow_type == "mice" or self.agent is None:
            self.mice_flows_count += 1
            selected_idx = equal_cost_indices[step % len(equal_cost_indices)]
            return candidate_paths[selected_idx], "tier1_ecmp", selected_idx

        # ── Tier 2: Elephant Flow → Graph Attention DRL Optimization ─
        self.elephant_flows_count += 1
        action, _, _ = self.agent.select_action(
            self.G, candidate_paths, deterministic=True
        )
        selected_idx = action if action < len(candidate_paths) else 0

        # If agent diverts elephant flow away from shortest equal-cost path
        if selected_idx not in equal_cost_indices:
            self.elephant_reroutes += 1

        return candidate_paths[selected_idx], "tier2_garro", selected_idx

    def get_stats(self) -> Dict[str, float]:
        """Return operational telemetry for the scheduler."""
        total = max(1, self.total_flows_processed)
        return {
            "total_flows": float(self.total_flows_processed),
            "mice_percentage": float(self.mice_flows_count / total * 100.0),
            "elephant_percentage": float(self.elephant_flows_count / total * 100.0),
            "elephant_reroute_count": float(self.elephant_reroutes),
            "controller_offload_pct": float(self.mice_flows_count / total * 100.0),
        }
