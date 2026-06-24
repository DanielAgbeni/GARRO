"""
Fat-Tree Topology — Parameterised k-ary Fat-Tree data centre network.

For k=8 (default):
    - 16  core switches
    - 32  aggregation switches  (k pods × k/2 per pod)
    - 32  edge switches         (k pods × k/2 per pod)
    Total: 80 switches

Node numbering (contiguous ranges):
    [0,         num_core)             → core switches
    [num_core,  num_core+num_agg)     → aggregation switches
    [num_core+num_agg, total)         → edge switches

All links are 10 Gbps with 1 ms intra-rack delay.
"""
import networkx as nx


def get_fat_tree(k: int = 8) -> nx.Graph:
    """
    Build a k-ary Fat-Tree topology.

    Parameters
    ----------
    k : int
        The pod/port parameter (must be even). Default 8.

    Returns
    -------
    nx.Graph
        NetworkX graph with bandwidth/delay/utilization/packet_loss edge attrs
        and cpu/buffer_occ/ingress_rate/egress_rate node attrs.
    """
    if k % 2 != 0:
        raise ValueError(f"k must be even, got {k}")

    G = nx.Graph()

    num_core = (k // 2) ** 2
    num_agg  = k * (k // 2)
    num_edge = k * (k // 2)
    total    = num_core + num_agg + num_edge

    core_start = 0
    agg_start  = num_core
    edge_start = num_core + num_agg

    BW    = 10_000   # 10 Gbps in Mbps
    DELAY = 1.0      # 1 ms intra-rack

    # ── Add all nodes ──────────────────────────────────────────────────────
    for n in range(total):
        G.add_node(n, cpu=0.5, buffer_occ=0.3,
                   ingress_rate=0.5, egress_rate=0.5)

    # ── Wire up pods ───────────────────────────────────────────────────────
    for pod in range(k):
        for agg_idx in range(k // 2):
            agg_id = agg_start + pod * (k // 2) + agg_idx

            # Aggregation → Edge (within same pod)
            for edge_idx in range(k // 2):
                edge_id = edge_start + pod * (k // 2) + edge_idx
                G.add_edge(agg_id, edge_id,
                           bandwidth=BW, delay=DELAY,
                           utilization=0.0, packet_loss=0.0)

            # Aggregation → Core
            # Each agg switch at index agg_idx in any pod connects to the
            # same stripe of k/2 core switches.
            for core_idx in range(k // 2):
                core_id = core_start + agg_idx * (k // 2) + core_idx
                G.add_edge(core_id, agg_id,
                           bandwidth=BW, delay=DELAY,
                           utilization=0.0, packet_loss=0.0)

    return G
