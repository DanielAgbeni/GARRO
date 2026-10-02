"""
GARRO Master Checkpoint Leaderboard Evaluator
==============================================
Scans `checkpoints/` for all trained model checkpoints for a given topology,
evaluates each checkpoint over N validation episodes, benchmarks against baselines
(OSPF, ECMP, Random), and outputs a ranked Master Leaderboard Summary.

Every model and baseline is evaluated on the **same fixed scenario suite**:
identical episode seeds, topology overrides, and reward weights.  A fresh
MM1KNetworkEnv is constructed per evaluator so no state leaks between runs.

Usage:
    python diagnostics/benchmark_leaderboard.py --topology nsfnet --episodes 50
    python diagnostics/benchmark_leaderboard.py --topology fat_tree --episodes 50 --seed 42
"""

import argparse
import re
import sys
from pathlib import Path

# Ensure project root is in sys.path when executed from any directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def apply_topology_overrides(config: dict, topology: str) -> dict:
    """Merge topology-specific ppo/training/reward_weights overrides in place."""
    overrides = config.get("topology_overrides", {}).get(topology, {})
    if overrides.get("ppo"):
        config["ppo"].update(overrides["ppo"])
    if overrides.get("training"):
        config["training"].update(overrides["training"])
    if overrides.get("reward_weights"):
        config.setdefault("reward_weights", {}).update(overrides["reward_weights"])
    return config


def build_scenario_seeds(n_episodes: int, master_seed: int) -> List[int]:
    """Fixed per-episode seeds shared by every model and baseline."""
    return [master_seed + ep for ep in range(n_episodes)]


def make_env(config: dict, topology: str) -> Tuple[nx.Graph, MM1KNetworkEnv]:
    """Build a fresh graph + env instance (no cross-run state leakage)."""
    G = TOPOLOGY_MAP[topology]()
    return G, MM1KNetworkEnv(G, config)


def eval_agent(
    agent: PPOAgent,
    config: dict,
    topology: str,
    scenario_seeds: List[int],
    deterministic: bool = True,
) -> float:
    """Evaluate one checkpoint on the fixed scenario suite."""
    _, env = make_env(config, topology)
    rewards = []
    for seed in scenario_seeds:
        obs, info = env.reset(seed=seed)
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


def eval_baseline(
    algo: str,
    config: dict,
    topology: str,
    scenario_seeds: List[int],
) -> float:
    """Evaluate one baseline on the fixed scenario suite."""
    _, env = make_env(config, topology)
    rewards = []
    for seed in scenario_seeds:
        obs, info = env.reset(seed=seed)
        done = False
        ep_r = 0.0
        step = 0
        while not done:
            k = max(len(env.candidate_paths), 1)
            if algo == "ospf":
                action = 0
            elif algo == "ecmp":
                action = step % k
            elif algo == "random":
                action = int(env.np_random.integers(0, k))
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master seed for the shared validation scenario suite (default: 42)",
    )
    args = parser.parse_args()

    with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["network"]["topology"] = args.topology
    config["seed"] = args.seed
    apply_topology_overrides(config, args.topology)

    scenario_seeds = build_scenario_seeds(args.episodes, args.seed)

    # Reference graph for node count when loading checkpoints
    G = TOPOLOGY_MAP[args.topology]()
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
    print(f"  Shared scenario suite: seeds {scenario_seeds[0]}..{scenario_seeds[-1]} (master seed={args.seed})")
    print(f"  Fresh env per evaluator | topology overrides applied")
    print(f"==================================================================\n")

    results = []

    # ── Evaluate baselines (each on the same fixed scenario suite) ───────────
    print("[Baselines] Running OSPF, ECMP, Random on shared scenarios ...")
    ospf_r = eval_baseline("ospf", config, args.topology, scenario_seeds)
    ecmp_r = eval_baseline("ecmp", config, args.topology, scenario_seeds)
    rand_r = eval_baseline("random", config, args.topology, scenario_seeds)

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
            mean_r = eval_agent(agent, config, args.topology, scenario_seeds, deterministic=True)
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
    ax.set_title(
        f"🏆 {args.topology.upper()} Checkpoint Leaderboard "
        f"({args.episodes} eps, shared seed={args.seed})"
    )
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
