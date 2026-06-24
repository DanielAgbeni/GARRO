"""
GARRO Evaluation & Benchmarking Script.

Benchmarks GARRO (PPO + Graph Transformer) against three baselines:
    - OSPF  : always selects path index 0 (shortest delay)
    - ECMP  : round-robin across available paths
    - Random: uniformly random path selection

All evaluations run entirely within the Digital Twin — no live network needed.

Usage
-----
    source garro_env/bin/activate

    python evaluate.py \\
        --checkpoint checkpoints/garro_nsfnet_final.pt \\
        --topology nsfnet \\
        --episodes 500

Outputs
-------
    eval_results_<topology>.png   Bar chart of mean episode rewards
    eval_results_<topology>.csv   Table of results
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml

from digital_twin.mm1k_env import MM1KNetworkEnv
from model.ppo_agent import PPOAgent
from topologies.nsfnet import get_nsfnet
from topologies.geant2 import get_geant2
from topologies.fat_tree import get_fat_tree


# ── Topology registry ─────────────────────────────────────────────────────────

TOPOLOGY_MAP = {
    "nsfnet":   get_nsfnet,
    "geant2":   get_geant2,
    "fat_tree": lambda: get_fat_tree(k=8),
}


# ── Baseline evaluators ───────────────────────────────────────────────────────

def run_ospf(env: MM1KNetworkEnv, episodes: int) -> dict:
    """
    OSPF baseline — always selects path index 0 (shortest delay path).
    Equivalent to Dijkstra shortest-path routing.
    """
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done   = False
        ep_r   = 0.0
        while not done:
            obs, r, terminated, truncated, _ = env.step(0)
            done  = terminated or truncated
            ep_r += r
        rewards.append(ep_r)
    return {
        "mean_reward": float(np.mean(rewards)),
        "std":         float(np.std(rewards)),
        "min":         float(np.min(rewards)),
        "max":         float(np.max(rewards)),
    }


def run_ecmp(env: MM1KNetworkEnv, episodes: int) -> dict:
    """
    ECMP baseline — round-robin across all available candidate paths.
    Simulates Equal-Cost Multi-Path routing without utilisation awareness.
    """
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done   = False
        ep_r   = 0.0
        step   = 0
        while not done:
            n_paths = max(len(env.candidate_paths), 1)
            action  = step % n_paths          # Round-robin
            obs, r, terminated, truncated, _ = env.step(action)
            done  = terminated or truncated
            ep_r += r
            step += 1
        rewards.append(ep_r)
    return {
        "mean_reward": float(np.mean(rewards)),
        "std":         float(np.std(rewards)),
        "min":         float(np.min(rewards)),
        "max":         float(np.max(rewards)),
    }


def run_random(env: MM1KNetworkEnv, episodes: int) -> dict:
    """
    Random baseline — uniformly random path selection.
    Lower bound on performance.
    """
    rewards = []
    rng = np.random.default_rng(42)
    for _ in range(episodes):
        obs, _ = env.reset()
        done   = False
        ep_r   = 0.0
        while not done:
            n_paths = max(len(env.candidate_paths), 1)
            action  = int(rng.integers(0, n_paths))
            obs, r, terminated, truncated, _ = env.step(action)
            done  = terminated or truncated
            ep_r += r
        rewards.append(ep_r)
    return {
        "mean_reward": float(np.mean(rewards)),
        "std":         float(np.std(rewards)),
        "min":         float(np.min(rewards)),
        "max":         float(np.max(rewards)),
    }


def run_garro(
    env:      MM1KNetworkEnv,
    agent:    PPOAgent,
    episodes: int,
) -> dict:
    """
    GARRO (PPO + Graph Transformer) — no gradient updates during evaluation.
    """
    rewards = []
    agent.encoder.eval()
    agent.ac_net.eval()

    for _ in range(episodes):
        obs, _ = env.reset()
        done   = False
        ep_r   = 0.0
        while not done:
            action, _, _ = agent.select_action(env.G, env.candidate_paths)
            obs, r, terminated, truncated, _ = env.step(action)
            done  = terminated or truncated
            ep_r += r
        rewards.append(ep_r)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std":         float(np.std(rewards)),
        "min":         float(np.min(rewards)),
        "max":         float(np.max(rewards)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cpu")
    G      = TOPOLOGY_MAP[args.topology]()
    env    = MM1KNetworkEnv(G, config)

    print(f"\n{'='*62}")
    print(f"  GARRO Benchmarking — {args.topology.upper()} "
          f"({G.number_of_nodes()} nodes, {G.number_of_edges()} links)")
    print(f"  Episodes per algorithm: {args.episodes}")
    print(f"{'='*62}\n")

    results = {}

    print("[Eval] Running Random baseline …")
    results["Random"] = run_random(env, args.episodes)

    print("[Eval] Running OSPF baseline …")
    results["OSPF"]   = run_ospf(env, args.episodes)

    print("[Eval] Running ECMP baseline …")
    results["ECMP"]   = run_ecmp(env, args.episodes)

    print("[Eval] Loading GARRO checkpoint …")
    agent = PPOAgent(config, k_paths=config["network"]["k_paths"], device=device)
    agent.load(args.checkpoint)
    print("[Eval] Running GARRO (PPO) …")
    results["GARRO"]  = run_garro(env, agent, args.episodes)

    # ── Print results table ───────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  {'Algorithm':<12} {'Mean Reward':>14} {'Std Dev':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*58}")
    for name, r in results.items():
        marker = " ← GARRO" if name == "GARRO" else ""
        print(f"  {name:<12} {r['mean_reward']:>14.4f} {r['std']:>10.4f} "
              f"{r['min']:>10.4f} {r['max']:>10.4f}{marker}")
    print(f"{'='*62}\n")

    # ── Bar chart ─────────────────────────────────────────────────────────
    names  = list(results.keys())
    means  = [results[n]["mean_reward"] for n in names]
    stds   = [results[n]["std"]         for n in names]
    colors = ["#95A5A6", "#E74C3C", "#F39C12", "#2ECC71"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, means, yerr=stds, color=colors[:len(names)],
                  capsize=7, edgecolor="black", linewidth=0.8, alpha=0.9)

    # Annotate bars with values
    for bar, mean_val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.002,
            f"{mean_val:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_ylabel("Mean Episode Reward", fontsize=11)
    ax.set_title(
        f"Routing Algorithm Comparison — {args.topology.upper()} "
        f"({args.episodes} episodes)",
        fontsize=12,
    )
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    chart_path = f"eval_results_{args.topology}.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"[Eval] Bar chart saved → {chart_path}")

    # ── CSV ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(results).T
    csv_path = f"eval_results_{args.topology}.csv"
    df.to_csv(csv_path)
    print(f"[Eval] CSV saved       → {csv_path}")
    print("\n[Eval] Done. ✓")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GARRO Evaluation — benchmark PPO vs OSPF / ECMP / Random"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained GARRO checkpoint (.pt)",
    )
    parser.add_argument(
        "--topology",
        default="nsfnet",
        choices=list(TOPOLOGY_MAP.keys()),
        help="Topology to evaluate on (default: nsfnet)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of evaluation episodes per algorithm (default: 500)",
    )
    args = parser.parse_args()
    main(args)
