"""
GARRO Master Checkpoint Leaderboard Evaluator
==============================================
Scans `checkpoints/` for all trained model checkpoints for a given topology,
evaluates each checkpoint over N validation episodes, benchmarks against baselines
(OSPF, ECMP, Random), and outputs a ranked Master Leaderboard Summary.

Usage:
    python diagnostics/benchmark_leaderboard.py --topology nsfnet --episodes 50
"""

import argparse
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
from topologies.fat_tree import get_fat_tree
from topologies.geant2 import get_geant2
from topologies.nsfnet import get_nsfnet

TOPOLOGY_MAP = {
    "nsfnet": get_nsfnet,
    "geant2": get_geant2,
    "fat_tree": lambda: get_fat_tree(k=4),
}


# ── ANSI Helpers ─────────────────────────────────────────────────────────────
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def eval_agent(agent: PPOAgent, env: MM1KNetworkEnv, n_episodes: int, deterministic: bool = True) -> float:
    rewards = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            candidate_paths = env.candidate_paths
            action, _, _ = agent.select_action(env.G, candidate_paths, deterministic=deterministic)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_r += reward
        rewards.append(ep_r)
    return float(np.mean(rewards))


def eval_baseline(algo: str, env: MM1KNetworkEnv, n_episodes: int) -> float:
    rewards = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_r = 0.0
        step = 0
        while not done:
            k = len(env.candidate_paths)
            if k == 0:
                action = 0
            elif algo == "ospf":
                action = 0
            elif algo == "ecmp":
                action = step % k
            elif algo == "random":
                action = np.random.randint(0, k)
            else:
                action = 0

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_r += reward
            step += 1
        rewards.append(ep_r)
    return float(np.mean(rewards))


def main():
    parser = argparse.ArgumentParser(description="GARRO Master Leaderboard Evaluator")
    parser.add_argument("--topology", default="nsfnet", choices=list(TOPOLOGY_MAP.keys()))
    parser.add_argument("--dir", default="checkpoints", help="Directory containing .pt checkpoints")
    parser.add_argument("--episodes", type=int, default=50, help="Validation episodes per model (default: 50)")
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    config["network"]["topology"] = args.topology
    G = TOPOLOGY_MAP[args.topology]()
    env = MM1KNetworkEnv(G, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k_paths = config["network"]["k_paths"]

    ckpt_dir = Path(args.dir)
    pattern = re.compile(rf"garro_{args.topology}_(ep\d+|final)\.pt$")
    ckpt_files = sorted([f for f in ckpt_dir.glob("*.pt") if pattern.search(f.name)])

    if not ckpt_files:
        print(f"[Error] No checkpoints found matching garro_{args.topology}_*.pt in {ckpt_dir}")
        return

    print(f"\n==================================================================")
    print(f"  Evaluating {len(ckpt_files)} Checkpoints on {args.topology.upper()} ({args.episodes} Validation Episodes)")
    print(f"==================================================================\n")

    results = []

    # ── Evaluate baselines ───────────────────────────────────────────────────
    print("[Baselines] Running OSPF, ECMP, Random ...")
    ospf_r = eval_baseline("ospf", env, args.episodes)
    ecmp_r = eval_baseline("ecmp", env, args.episodes)
    rand_r = eval_baseline("random", env, args.episodes)

    results.append({"Model / Checkpoint": "OSPF (Baseline)", "Type": "Baseline", "Mean Reward": ospf_r})
    results.append({"Model / Checkpoint": "ECMP (Baseline)", "Type": "Baseline", "Mean Reward": ecmp_r})
    results.append({"Model / Checkpoint": "Random (Baseline)", "Type": "Baseline", "Mean Reward": rand_r})

    # ── Evaluate GARRO checkpoints ───────────────────────────────────────────
    for ckpt_path in ckpt_files:
        name = ckpt_path.name
        print(f"[Evaluating] {name} ...", end=" ", flush=True)

        agent = PPOAgent(
            config=config,
            k_paths=k_paths,
            num_nodes=G.number_of_nodes(),
            device=device,
            compile_model=False,
        )
        try:
            agent.load(str(ckpt_path))
            mean_r = eval_agent(agent, env, args.episodes, deterministic=True)
            print(f"Mean Reward: {mean_r:+.4f}")
            results.append({"Model / Checkpoint": name, "Type": "GARRO Checkpoint", "Mean Reward": mean_r})
        except Exception as e:
            print(f"FAILED ({e})")

    df = pd.DataFrame(results)
    df = df.sort_values(by="Mean Reward", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # 1-based rank

    # ── Master Leaderboard Display ───────────────────────────────────────────
    banner = f"🏆 {args.topology.upper()} MASTER LEADERBOARD SUMMARY (Ranked by Mean Validation Reward)"
    sep = "=" * len(banner)
    print(f"\n{sep}")
    print(f"{BOLD}{CYAN}{banner}{RESET}")
    print(f"{sep}\n")

    print(f"{'Rank':>4}  {'Model / Checkpoint':<32}  {'Type':<18}  {'Mean Reward':>12}")
    print("─" * 72)
    for rank, row in df.iterrows():
        model_str = row["Model / Checkpoint"]
        type_str = row["Type"]
        reward = row["Mean Reward"]
        is_top = (rank == 1)
        prefix = "⭐ " if is_top else "   "
        color = GREEN if is_top else (YELLOW if "GARRO" in type_str else RESET)
        print(f"{prefix}{rank:>2}  {color}{model_str:<32}{RESET}  {type_str:<18}  {color}{reward:+12.4f}{RESET}")

    print(f"\n{sep}\n")

    # ── Export CSV & Leaderboard Chart ───────────────────────────────────────
    csv_path = ckpt_dir / f"leaderboard_{args.topology}.csv"
    df.to_csv(csv_path, index_label="Rank")
    print(f"[Export] Saved Leaderboard Table → {csv_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2ca02c" if r["Type"] == "GARRO Checkpoint" else "#7f7f7f" for _, r in df.iterrows()]
    bars = ax.barh(df["Model / Checkpoint"][::-1], df["Mean Reward"][::-1], color=colors[::-1], alpha=0.85)

    ax.set_xlabel("Mean Validation Reward")
    ax.set_title(f"🏆 {args.topology.upper()} Checkpoint Leaderboard ({args.episodes} Validation Episodes)")
    ax.grid(True, alpha=0.3)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + (1.0 if w >= 0 else -5.0), bar.get_y() + bar.get_height()/2, f"{w:+.1f}",
                va="center", ha="left" if w >= 0 else "right", fontsize=9, fontweight="bold")

    fig.tight_layout()
    img_path = ckpt_dir / f"leaderboard_{args.topology}.png"
    fig.savefig(img_path, dpi=150)
    plt.close(fig)
    print(f"[Export] Saved Leaderboard Chart → {img_path}\n")


if __name__ == "__main__":
    main()
