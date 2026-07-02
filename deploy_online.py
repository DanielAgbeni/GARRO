"""
Phase 2: Live deployment loop connecting the trained PPO agent to
the OS-Ken controller via the Northbound REST API.

System Resource Utilization
-----------------------------
* Auto-detects best device (CUDA → MPS → CPU) via PPOAgent._best_device().
* Async telemetry polling and LLM intent updates run concurrently — the
  main routing loop never blocks waiting for the LLM API.
* aiohttp replaces requests for non-blocking HTTP calls.
* LLM intent re-evaluation runs in a background asyncio Task so it
  cannot delay routing decisions.

Usage (after OS-Ken and Mininet are running):
    python deploy_online.py --checkpoint checkpoints/garro_nsfnet_final.pt \\
                             --topology nsfnet
"""
import asyncio
import argparse
import time
import sys

import aiohttp
import networkx as nx
import numpy as np
import torch
import yaml

from agentic.llm_orchestrator import LLMOrchestrator
from model.ppo_agent import PPOAgent, _best_device
from topologies.nsfnet import get_nsfnet
from topologies.geant2 import get_geant2
from topologies.fat_tree import get_fat_tree

TOPOLOGY_MAP = {
    "nsfnet":    get_nsfnet,
    "geant2":    get_geant2,
    "fat_tree":  lambda: get_fat_tree(k=8),
}

CONTROLLER_URL = "http://127.0.0.1:8080"


# ── Async telemetry fetch ─────────────────────────────────────────────────────

async def fetch_network_state(
    topology_G: nx.Graph,
    session: aiohttp.ClientSession,
) -> nx.Graph:
    """
    Pull live telemetry from OS-Ken REST API (non-blocking).
    Falls back to previous state on any HTTP error.
    """
    try:
        async with session.get(
            f"{CONTROLLER_URL}/garro/state",
            timeout=aiohttp.ClientTimeout(total=3),
        ) as resp:
            resp.raise_for_status()
            state = await resp.json()

        for edge_data in state.get("edges", []):
            u = edge_data["src"] - 1   # DPID (1-indexed) → graph node (0-indexed)
            v = edge_data["dst"] - 1
            if topology_G.has_edge(u, v):
                topology_G.edges[u, v]["utilization"] = edge_data["utilization"]
                topology_G.edges[u, v]["packet_loss"] = edge_data["packet_loss"]
                topology_G.edges[u, v]["delay"]       = edge_data["delay"]

    except Exception as e:
        print(f"[Deploy] Telemetry fetch failed: {e} — using cached state")

    return topology_G


# ── Async flow installation ───────────────────────────────────────────────────

async def install_flow(
    path: list,
    src_ip: str,
    dst_ip: str,
    session: aiohttp.ClientSession,
    verbose: bool = False,
) -> None:
    """POST computed path to OS-Ken controller (non-blocking)."""
    dpid_path = [node + 1 for node in path]
    payload   = {"path": dpid_path, "src_ip": src_ip, "dst_ip": dst_ip}
    try:
        async with session.post(
            f"{CONTROLLER_URL}/garro/flow",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=3),
        ) as resp:
            if resp.status == 200:
                if verbose:
                    print(f"[Deploy] Flow installed: {dpid_path} | {src_ip}→{dst_ip}")
            else:
                text = await resp.text()
                print(f"[Deploy] Flow install failed ({resp.status}): {text}")
    except Exception as e:
        print(f"[Deploy] Flow install error: {e}")


# ── Background LLM intent updater ────────────────────────────────────────────

async def _intent_loop(
    llm: LLMOrchestrator,
    config: dict,
    interval_s: float = 1.5,
) -> None:
    """
    Background task:
    1. Periodically polls /garro/intent from the controller.
    2. If changed, parses via LLM, updates config, and POSTs new weights to /garro/weights.
    3. Concurrently reads stdin and POSTs typed intents to /garro/intent on the controller.
    """
    default_intent = (
        "Balance load across all links while maintaining "
        "reasonable latency for mixed traffic."
    )
    last_processed_intent = ""
    loop = asyncio.get_event_loop()

    # Helper task to read stdin and POST to controller
    async def stdin_reader():
        print("\n[Operator] == Interactive Console Ready ==")
        print("[Operator] Type a new routing intent here at any time and press Enter.")
        print("[Operator] Or submit it via the Web UI at http://127.0.0.1:8080/\n")
        
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    # Read from stdin asynchronously (non-blocking)
                    line = await loop.run_in_executor(None, sys.stdin.readline)
                    if not line:
                        await asyncio.sleep(1)
                        continue
                    
                    intent = line.strip()
                    if not intent:
                        continue

                    print(f"\n[Operator] Terminal input received: '{intent}'")
                    # POST intent to controller (makes it the source of truth)
                    async with session.post(
                        f"{CONTROLLER_URL}/garro/intent",
                        json={"intent": intent},
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as resp:
                        if resp.status != 200:
                            print(f"[Deploy/LLM] Failed to sync terminal intent to controller: {resp.status}")
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    print(f"[Deploy/LLM] Terminal reader error: {exc}")

    # Launch the stdin reader task
    reader_task = asyncio.create_task(stdin_reader())

    # Main poll loop for intent and LLM re-evaluation
    async with aiohttp.ClientSession() as session:
        # Pre-populate the controller with the default intent if not already set
        try:
            async with session.post(
                f"{CONTROLLER_URL}/garro/intent",
                json={"intent": default_intent},
                timeout=aiohttp.ClientTimeout(total=3),
            ) as resp:
                pass
        except Exception:
            pass

        while True:
            try:
                # Poll current intent from controller
                async with session.get(
                    f"{CONTROLLER_URL}/garro/intent",
                    timeout=aiohttp.ClientTimeout(total=3),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        current_intent = data.get("intent", "").strip()
                    else:
                        current_intent = ""

                if current_intent and current_intent != last_processed_intent:
                    print(f"[Deploy/LLM] New intent detected: '{current_intent}'")
                    print("[Deploy/LLM] Parsing intent & re-optimizing weights via LLM...")
                    
                    weights = await llm.parse_intent(current_intent)
                    weights.apply_to_config(config)
                    print(f"[Deploy/LLM] Weights successfully updated → {weights.as_dict()}")

                    # Sync weights back to controller for Dashboard display
                    async with session.post(
                        f"{CONTROLLER_URL}/garro/weights",
                        json={"weights": weights.as_dict()},
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as w_resp:
                        if w_resp.status != 200:
                            print(f"[Deploy/LLM] Failed to sync weights to dashboard: {w_resp.status}")

                    last_processed_intent = current_intent

            except asyncio.CancelledError:
                break
            except Exception as exc:
                print(f"[Deploy/LLM] Intent sync loop failed: {exc}")
            
            await asyncio.sleep(interval_s)

        reader_task.cancel()
        try:
            await reader_task
        except asyncio.CancelledError:
            pass


# ── Main deployment loop ──────────────────────────────────────────────────────

async def main(args):
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # ── Auto-detect best compute device ──────────────────────────────────
    device  = _best_device()
    k_paths = config["network"]["k_paths"]

    print(f"\n{'='*58}")
    print(f"  GARRO Online Deployment")
    print(f"  Topology : {args.topology.upper()}")
    print(f"  Device   : {device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'='*58}\n")

    # ── Build topology + pre-compute paths ────────────────────────────────
    G = TOPOLOGY_MAP.get(args.topology, get_nsfnet)()
    print(f"[Deploy] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── Load pre-trained agent ────────────────────────────────────────────
    agent = PPOAgent(config, k_paths=k_paths, num_nodes=G.number_of_nodes(),
                     device=device, compile_model=False)
    try:
        agent.load(args.checkpoint)
        print(f"[Deploy] Checkpoint loaded from {args.checkpoint}")
    except Exception as e:
        print(f"[Deploy] Warning: checkpoint load failed ({e}) — random weights")

    agent.encoder.eval()
    agent.ac_net.eval()

    # ── Pre-compute K-shortest paths ──────────────────────────────────────
    all_paths = {}
    nodes = list(G.nodes())
    for src in nodes:
        for dst in nodes:
            if src == dst:
                continue
            try:
                from itertools import islice
                paths_gen = nx.shortest_simple_paths(G, src, dst, weight="delay")
                all_paths[(src, dst)] = list(islice(paths_gen, k_paths))
            except nx.NetworkXNoPath:
                all_paths[(src, dst)] = []
    print(f"[Deploy] Pre-computed paths for {len(all_paths)} node pairs")

    # ── LLM orchestrator ──────────────────────────────────────────────────
    llm = LLMOrchestrator(config)

    # ── Flow demands to route ─────────────────────────────────────────────
    flow_demands = [
        ("10.0.0.1",  "10.0.0.14", 0,  13),
        ("10.0.0.3",  "10.0.0.12", 2,  11),
        ("10.0.0.5",  "10.0.0.9",  4,  8),
    ]

    polling_interval = config["network"]["polling_interval"]
    print(f"\n[Deploy] GARRO online — polling every {polling_interval}s")
    print("[Deploy] Ctrl+C to stop\n")

    # ── Launch background LLM intent task ─────────────────────────────────
    intent_task = asyncio.create_task(
        _intent_loop(llm, config, interval_s=60.0)
    )

    # ── Main async loop ───────────────────────────────────────────────────
    last_paths = {}
    async with aiohttp.ClientSession() as session:
        step = 0
        try:
            while True:
                t_loop_start = asyncio.get_event_loop().time()

                # 1. Fetch live telemetry (non-blocking)
                G = await fetch_network_state(G, session)

                # 2. Route each flow with PPO agent
                flow_tasks = []
                for src_ip, dst_ip, src_node, dst_node in flow_demands:
                    candidates = all_paths.get((src_node, dst_node), [])
                    if not candidates:
                        continue

                    action, _, _ = agent.select_action(G, candidates)
                    selected_path = (
                        candidates[action] if action < len(candidates)
                        else candidates[0]
                    )

                    dpid_path = [node + 1 for node in selected_path]
                    pair_key = (src_ip, dst_ip)
                    old_path = last_paths.get(pair_key)

                    verbose = False
                    if old_path != dpid_path:
                        verbose = True
                        if old_path is not None:
                            print(f"[Deploy] Path CHANGED for {src_ip}→{dst_ip}: {old_path} → {dpid_path}")
                        else:
                            print(f"[Deploy] Path INITIALIZED for {src_ip}→{dst_ip}: {dpid_path}")
                        last_paths[pair_key] = dpid_path

                    # 3. Install flow rules concurrently
                    flow_tasks.append(
                        install_flow(selected_path, src_ip, dst_ip, session, verbose=verbose)
                    )

                if flow_tasks:
                    await asyncio.gather(*flow_tasks)

                step += 1

                # 4. Sleep for remainder of polling interval
                elapsed = asyncio.get_event_loop().time() - t_loop_start
                sleep_for = max(0.0, polling_interval - elapsed)
                await asyncio.sleep(sleep_for)

        except KeyboardInterrupt:
            print("\n[Deploy] Shutting down …")
        finally:
            intent_task.cancel()
            try:
                await intent_task
            except asyncio.CancelledError:
                pass
            print("[Deploy] Done.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GARRO Phase 2 — Live SDN Deployment"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--topology", default="nsfnet",
        choices=list(TOPOLOGY_MAP.keys()),
        help="Network topology (default: nsfnet)",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
