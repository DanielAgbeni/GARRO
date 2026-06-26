#!/usr/bin/env python3
"""
GARRO Path Visualisation Script.

Draws network topologies (NSFNET, GEANT2, Fat Tree) and highlights
a routing path. Can highlight a manually specified path, the OSPF shortest
path, or the path chosen by a trained GARRO checkpoint.

Usage:
------
    # Visualize OSPF path from Seattle (0) to College Park (12) on NSFNET:
    python visualize_path.py --topology nsfnet --src 0 --dst 12 --method ospf

    # Visualize a custom path:
    python visualize_path.py --topology nsfnet --path 0,3,4,10,11,12

    # Visualize path chosen by GARRO model:
    python visualize_path.py --topology nsfnet --src 0 --dst 12 --method garro --checkpoint checkpoints/garro_nsfnet_final.pt
"""
import argparse
import sys
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml
import torch

from topologies.nsfnet import get_nsfnet
from topologies.geant2 import get_geant2
from topologies.fat_tree import get_fat_tree
from digital_twin.mm1k_env import MM1KNetworkEnv
from model.ppo_agent import PPOAgent

# ── Topology registry ─────────────────────────────────────────────────────────
TOPOLOGY_MAP = {
    "nsfnet":   get_nsfnet,
    "geant2":   get_geant2,
    "fat_tree": lambda: get_fat_tree(k=8),
}

# ── Geographical Coordinates for NSFNET nodes (approximate Long/Lat) ──────────
NSFNET_POS = {
    0:  (-122.33, 47.60),  # Seattle
    1:  (-122.14, 37.44),  # Palo Alto
    2:  (-117.16, 32.71),  # San Diego
    3:  (-111.89, 40.76),  # Salt Lake City
    4:  (-105.27, 40.01),  # Boulder
    5:  (-96.70,  40.81),  # Lincoln
    6:  (-95.36,  29.76),  # Houston
    7:  (-88.24,  40.11),  # Champaign
    8:  (-84.38,  33.74),  # Atlanta
    9:  (-83.74,  42.28),  # Ann Arbor
    10: (-79.99,  40.44),  # Pittsburgh
    11: (-74.65,  40.35),  # Princeton
    12: (-76.93,  38.98),  # College Park
    13: (-76.50,  42.44),  # Ithaca
}

def main():
    parser = argparse.ArgumentParser(description="Visualize network routing paths")
    parser.add_argument("--topology", default="nsfnet", choices=list(TOPOLOGY_MAP.keys()),
                        help="Network topology to visualize")
    parser.add_argument("--src", type=int, default=0, help="Source node ID")
    parser.add_argument("--dst", type=int, default=12, help="Destination node ID")
    parser.add_argument("--path", type=str, default=None,
                        help="Comma-separated node IDs of a custom path to plot (e.g. 0,3,4,10,12)")
    parser.add_argument("--method", default="ospf", choices=["ospf", "garro"],
                        help="Routing method to find the path (if --path is not specified)")
    parser.add_argument("--checkpoint", default="checkpoints/garro_nsfnet_final.pt",
                        help="Path to trained GARRO agent checkpoint (required for 'garro' method)")
    parser.add_argument("--output", default="routing_path.png", help="Output image file name")
    args = parser.parse_args()

    # Load configuration
    try:
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        config = {
            "network": {"k_paths": 5},
            "mm1k": {"buffer_capacity": 50, "base_arrival_rate": 100.0, "base_service_rate": 150.0},
            "reward_weights": {"alpha1": 0.4, "alpha2": 0.3, "alpha3": 0.2, "alpha4": 0.1},
            "training": {"max_steps_per_episode": 200}
        }

    # Initialize topology
    get_topo_fn = TOPOLOGY_MAP[args.topology]
    G = get_topo_fn()

    # Determine node positions for plotting
    if args.topology == "nsfnet":
        pos = NSFNET_POS
    else:
        # Use a spring layout for other topologies
        pos = nx.spring_layout(G, seed=42)

    path = []
    
    # 1. Check if user specified a manual path
    if args.path:
        try:
            path = [int(n.strip()) for n in args.path.split(",")]
            print(f"[Visualizer] Using custom path: {path}")
        except ValueError:
            print("Error: --path must be a comma-separated list of integers.")
            sys.exit(1)
    else:
        # 2. Otherwise, construct path using OSPF or GARRO
        env = MM1KNetworkEnv(G, config)
        
        # Ensure nodes exist
        if args.src not in G.nodes or args.dst not in G.nodes:
            print(f"Error: Source {args.src} or Destination {args.dst} not in topology nodes: {list(G.nodes)}")
            sys.exit(1)
            
        candidate_paths = env._all_paths.get((args.src, args.dst), [])
        if not candidate_paths:
            print(f"Error: No path exists between {args.src} and {args.dst}.")
            sys.exit(1)

        if args.method == "ospf":
            # OSPF is the first candidate path (shortest by delay)
            path = candidate_paths[0]
            print(f"[Visualizer] Computed OSPF path: {path}")
        elif args.method == "garro":
            if not os.path.exists(args.checkpoint):
                print(f"Error: Checkpoint file '{args.checkpoint}' not found.")
                print("Please check your checkpoint path or run with OSPF/custom path.")
                sys.exit(1)
            
            # Load agent
            device = torch.device("cpu")
            agent = PPOAgent(config, k_paths=config["network"]["k_paths"],
                             num_nodes=G.number_of_nodes(), device=device)
            agent.load(args.checkpoint)
            agent.encoder.eval()
            agent.ac_net.eval()
            
            # Select action
            action, _, _ = agent.select_action(G, candidate_paths)
            if action < len(candidate_paths):
                path = candidate_paths[action]
            else:
                path = candidate_paths[0]
            print(f"[Visualizer] GARRO agent selected path index {action}: {path}")

    # Verify if path is valid
    if len(path) < 2:
        print("Error: Path must contain at least 2 nodes.")
        sys.exit(1)

    # Validate that path edges exist in graph
    path_edges = list(zip(path[:-1], path[1:]))
    for u, v in path_edges:
        if not G.has_edge(u, v):
            print(f"Warning: Edge ({u}, {v}) in path does not exist in topology!")

    # Set up matplotlib figure
    plt.figure(figsize=(12, 8))
    
    # Draw all edges in the background (thin, light grey)
    nx.draw_networkx_edges(G, pos, width=1.0, edge_color="#D3D3D3", alpha=0.8)
    
    # Draw all nodes in the background (light blue/grey)
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color="#E0EBF5", edgecolors="#B0C4DE", linewidths=1.5)
    
    # Highlight path edges
    nx.draw_networkx_edges(
        G, pos,
        edgelist=path_edges,
        width=4.0,
        edge_color="#2ECC71",  # bright green
        arrows=True,
        arrowsize=20,
        arrowstyle="-|>"
    )
    
    # Highlight source node (green) and destination node (red)
    nx.draw_networkx_nodes(G, pos, nodelist=[path[0]], node_size=800, node_color="#2ECC71", edgecolors="#27AE60")
    nx.draw_networkx_nodes(G, pos, nodelist=[path[-1]], node_size=800, node_color="#E74C3C", edgecolors="#C0392B")
    
    # Highlight intermediate path nodes (yellow/orange)
    if len(path) > 2:
        nx.draw_networkx_nodes(G, pos, nodelist=path[1:-1], node_size=700, node_color="#F39C12", edgecolors="#D35400")

    # Add Node Labels (either geographical name if available, or Node ID)
    labels = {}
    for node in G.nodes():
        node_label = G.nodes[node].get("label", "")
        if node_label:
            labels[node] = f"{node}\n({node_label})"
        else:
            labels[node] = str(node)
            
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight="bold", font_family="sans-serif")

    # Add edge labels (showing delay or distance)
    edge_labels = {}
    for u, v in G.edges():
        delay = G.edges[u, v].get("delay", 0)
        edge_labels[(u, v)] = f"{delay}ms"
        
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color="#555555")

    # Title and visual adjustments
    title_method = f"Method: {args.method.upper()}" if not args.path else "Custom Path"
    plt.title(f"Network Routing Path Visualisation ({args.topology.upper()})\n{title_method}: {path}", 
              fontsize=14, fontweight="bold", pad=20)
    plt.axis("off")
    plt.tight_layout()
    
    # Save the output image
    plt.savefig(args.output, dpi=150)
    plt.close()
    
    print(f"[Success] Routing path visualization saved to: {args.output}")

if __name__ == "__main__":
    main()
