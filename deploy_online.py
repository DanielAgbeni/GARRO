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
            u_dpid, v_dpid = edge_data["src"], edge_data["dst"]
            # Map 1-indexed DPIDs to 0-indexed graph nodes
            u, v = u_dpid - 1, v_dpid - 1
            if topology_G.has_edge(u, v):
                topology_G.edges[u, v]["utilization"] = edge_data["utilization"]
                topology_G.edges[u, v]["packet_loss"] = edge_data["packet_loss"]
                topology_G.edges[u, v]["delay"] = edge_data["delay"]

    except Exception as e:
        print(f"[Deploy] Telemetry fetch failed: {e} — using cached state")

    return topology_G


def install_flow(path: list, src_ip: str, dst_ip: str):
    """POST computed path to OS-Ken controller for flow installation."""
    # Map 0-indexed graph nodes back to 1-indexed DPIDs
    dpid_path = [node + 1 for node in path]
    payload = {"path": dpid_path, "src_ip": src_ip, "dst_ip": dst_ip}
    try:
        resp = requests.post(
            f"{CONTROLLER_URL}/garro/flow",
            json=payload, timeout=3
        )
        if resp.status_code == 200:
            print(f"[Deploy] Flow installed: {dpid_path} | {src_ip}→{dst_ip}")
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
    try:
        agent.load(args.checkpoint)
        print(f"[Deploy] Successfully loaded checkpoint from {args.checkpoint}")
    except Exception as e:
        print(f"[Deploy] Warning: Could not load checkpoint ({e}). Running with randomized weights.")

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
