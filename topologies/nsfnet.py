"""
NSFNET Topology — 14 nodes, 21 links.

Models the US National Science Foundation Network WAN backbone.
Nodes represent major US cities; links carry realistic propagation
delays (ms) and 1 Gbps capacities.

Used as the primary training and evaluation topology.
"""
import networkx as nx


def get_nsfnet() -> nx.Graph:
    """
    Build and return the NSFNET topology as a NetworkX undirected graph.

    Node attributes:
        label (str): City name

    Edge attributes:
        bandwidth  (int):   Link capacity in Mbps
        delay      (float): Propagation delay in ms
        utilization (float): Current utilisation [0.0, 1.0] — initialised to 0
        packet_loss (float): Packet loss ratio [0.0, 1.0]  — initialised to 0
    """
    G = nx.Graph()

    # ── Nodes: (node_id, city_name) ─────────────────────────────────────
    nodes = [
        (0,  "Seattle"),
        (1,  "Palo Alto"),
        (2,  "San Diego"),
        (3,  "Salt Lake City"),
        (4,  "Boulder"),
        (5,  "Lincoln"),
        (6,  "Houston"),
        (7,  "Champaign"),
        (8,  "Atlanta"),
        (9,  "Ann Arbor"),
        (10, "Pittsburgh"),
        (11, "Princeton"),
        (12, "College Park"),
        (13, "Ithaca"),
    ]
    for nid, name in nodes:
        G.add_node(nid, label=name, cpu=0.5, buffer_occ=0.3,
                   ingress_rate=0.5, egress_rate=0.5)

    # ── Edges: (src, dst, bandwidth_Mbps, delay_ms) ──────────────────────
    edges = [
        (0,  1,  1000, 11),
        (0,  3,  1000,  9),
        (0,  5,  1000, 29),
        (1,  2,  1000,  6),
        (1,  3,  1000, 12),
        (2,  6,  1000, 22),
        (3,  4,  1000,  7),
        (4,  5,  1000, 12),
        (4,  10, 1000, 26),
        (5,  7,  1000,  9),
        (6,  8,  1000,  9),
        (7,  8,  1000, 13),
        (7,  9,  1000,  4),
        (8,  12, 1000,  8),
        (9,  10, 1000,  5),
        (9,  13, 1000,  7),
        (10, 11, 1000,  4),
        (11, 12, 1000,  3),
        (11, 13, 1000,  4),
        (12, 13, 1000,  5),
        (6,  12, 1000, 18),
    ]
    for src, dst, bw, delay in edges:
        G.add_edge(src, dst,
                   bandwidth=bw,
                   delay=float(delay),
                   utilization=0.0,
                   packet_loss=0.0)

    return G
