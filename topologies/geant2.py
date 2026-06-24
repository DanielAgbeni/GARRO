"""
GEANT2 Topology — 24 nodes, 37 links.

Models the pan-European academic research network.
Highly irregular, asymmetric topology — ideal for testing
edge-centrality awareness and load-balancing under
asymmetrical link constraints.

All links are 10 Gbps (typical for backbone research networks).
"""
import networkx as nx


def get_geant2() -> nx.Graph:
    """
    Build and return the GEANT2 topology as a NetworkX undirected graph.

    Node attributes:
        cpu         (float): Normalised CPU load [0.0, 1.0]
        buffer_occ  (float): Buffer occupancy ratio [0.0, 1.0]
        ingress_rate (float): Normalised ingress byte rate
        egress_rate  (float): Normalised egress byte rate

    Edge attributes:
        bandwidth  (int):   Link capacity in Mbps
        delay      (float): Propagation delay in ms
        utilization (float): Current utilisation [0.0, 1.0]
        packet_loss (float): Packet loss ratio [0.0, 1.0]
    """
    G = nx.Graph()

    # ── Nodes ─────────────────────────────────────────────────────────────
    for n in range(24):
        G.add_node(n, cpu=0.5, buffer_occ=0.3,
                   ingress_rate=0.5, egress_rate=0.5)

    # ── Edges: (src, dst, bandwidth_Mbps, delay_ms) ──────────────────────
    # Ring backbone
    edges = [
        (0,  1,  10000,  5),
        (0,  2,  10000, 12),
        (1,  3,  10000,  8),
        (2,  4,  10000, 15),
        (3,  5,  10000,  6),
        (4,  5,  10000, 20),
        (5,  6,  10000,  7),
        (6,  7,  10000,  9),
        (7,  8,  10000, 11),
        (8,  9,  10000, 14),
        (9,  10, 10000, 18),
        (10, 11, 10000,  6),
        (11, 12, 10000,  8),
        (12, 13, 10000, 10),
        (13, 14, 10000,  7),
        (14, 15, 10000, 12),
        (15, 16, 10000,  9),
        (16, 17, 10000,  6),
        (17, 18, 10000,  8),
        (18, 19, 10000, 11),
        (19, 20, 10000, 14),
        (20, 21, 10000,  9),
        (21, 22, 10000,  7),
        (22, 23, 10000,  6),
        # Cross-links (give the topology its irregular character)
        (0,  6,  10000, 25),
        (1,  8,  10000, 22),
        (3,  12, 10000, 30),
        (5,  15, 10000, 27),
        (7,  17, 10000, 19),
        (9,  19, 10000, 28),
        (11, 21, 10000, 24),
        (13, 22, 10000, 31),
        (2,  10, 10000, 35),
        (4,  14, 10000, 32),
        (6,  18, 10000, 29),
        (8,  20, 10000, 26),
        (10, 23, 10000, 21),
    ]
    for src, dst, bw, delay in edges:
        G.add_edge(src, dst,
                   bandwidth=bw,
                   delay=float(delay),
                   utilization=0.0,
                   packet_loss=0.0)

    return G
