"""
GARRO Evaluation & Benchmarking Script.

Benchmarks GARRO (PPO + Graph Transformer) against three baselines:
    - OSPF  : always selects path index 0 (shortest delay)
    - ECMP  : round-robin across available paths
    - Random: uniformly random path selection

System Resource Utilization
-----------------------------
* Baseline algorithms (OSPF, ECMP, Random) run in parallel across separate
  CPU processes using ProcessPoolExecutor — ~3× faster wall-time on a 4-core
  machine because each is completely independent of the others.
* GARRO evaluation runs on the main process (model must stay in-process).
* Auto-detects best compute device (CUDA > MPS > CPU) for GARRO inference.
* Thread pinning is applied automatically via PPOAgent.__init__.
* Each parallel worker gets its own tqdm progress bar so you can see all
  four algorithms progressing simultaneously.

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
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from digital_twin.mm1k_env import MM1KNetworkEnv
from model.ppo_agent import PPOAgent, _best_device
from topologies.nsfnet import get_nsfnet
from topologies.geant2 import get_geant2
from topologies.fat_tree import get_fat_tree


# ── Topology registry ─────────────────────────────────────────────────────────

TOPOLOGY_MAP = {
    "nsfnet":   get_nsfnet,
    "geant2":   get_geant2,
    "fat_tree": lambda: get_fat_tree(k=8),
}


# ── Baseline runners (top-level so they are picklable for ProcessPoolExecutor) ─

def _run_baseline_worker(args_tuple) -> dict:
    """
    Process-pool worker.  Runs a single baseline algorithm for `episodes`
    episodes and returns the statistics dict.

    Parameters passed as a single tuple so the worker is compatible with
    ProcessPoolExecutor.map / submit.

    Tuple layout: (algo_name, topology_name, config, episodes, seed)
    """
    algo_name, topology_name, config, episodes, seed = args_tuple

    # Re-build graph + env inside the worker process
    G   = TOPOLOGY_MAP[topology_name]()
    env = MM1KNetworkEnv(G, config)
    rng = np.random.default_rng(seed)

    rewards = []
    desc    = f"{algo_name:<8}"

    for ep in tqdm(range(episodes), desc=desc, position=0, leave=True,
                   dynamic_ncols=True):
        obs, _ = env.reset()
        done   = False
        ep_r   = 0.0
        step   = 0

        while not done:
            n_paths = max(len(env.candidate_paths), 1)

            if algo_name == "OSPF":
                action = 0                          # always shortest-delay path
            elif algo_name == "ECMP":
                action = step % n_paths             # round-robin
            elif algo_name == "Random":
                action = int(rng.integers(0, n_paths))
            else:
                raise ValueError(f"Unknown baseline: {algo_name}")

            obs, r, terminated, truncated, _ = env.step(action)
            done  = terminated or truncated
            ep_r += r
            step += 1

        rewards.append(ep_r)

    return {
        "name":        algo_name,
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
    GARRO (PPO + Graph Transformer) evaluation — no gradient updates.
    Runs on the main process so the loaded model stays in-memory.
    """
    rewards = []
    agent.encoder.eval()
    agent.ac_net.eval()

    for _ in tqdm(range(episodes), desc="GARRO   ", dynamic_ncols=True,
                  position=0, leave=True):
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

    # Auto-detect best device for GARRO inference
    device   = _best_device()
    n_cores  = multiprocessing.cpu_count()

    G   = TOPOLOGY_MAP[args.topology]()
    env = MM1KNetworkEnv(G, config)

    sep = "=" * 66
    print(f"\n{sep}")
    print(f"  GARRO Benchmarking — {args.topology.upper()} "
          f"({G.number_of_nodes()} nodes, {G.number_of_edges()} links)")
    print(f"  Episodes per algorithm : {args.episodes}")
    print(f"  Device (GARRO)         : {device}")
    print(f"  CPU cores              : {n_cores}")
    print(f"  Parallel baselines     : 3 (OSPF + ECMP + Random run simultaneously)")
    print(f"{sep}\n")

    results = {}
    t0      = time.perf_counter()

    # ── Parallel baseline evaluation ──────────────────────────────────────
    # OSPF, ECMP, Random are independent → run on 3 separate processes.
    # Each process rebuilds the env from scratch (lightweight) so there is
    # no shared state and no GIL contention.
    baseline_args = [
        ("OSPF",   args.topology, config, args.episodes, 0),
        ("ECMP",   args.topology, config, args.episodes, 1),
        ("Random", args.topology, config, args.episodes, 42),
    ]

    n_workers = min(len(baseline_args), max(1, n_cores - 1))
    print(f"[Eval] Launching {len(baseline_args)} baselines across "
          f"{n_workers} parallel worker(s) …\n")

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_run_baseline_worker, arg): arg[0]
            for arg in baseline_args
        }
        for future in as_completed(futures):
            algo = futures[future]
            try:
                res = future.result()
                results[res["name"]] = {
                    k: v for k, v in res.items() if k != "name"
                }
                print(f"  ✓ {algo} done")
            except Exception as exc:
                print(f"  ✗ {algo} failed: {exc}")
                results[algo] = {
                    "mean_reward": float("nan"),
                    "std": 0.0, "min": 0.0, "max": 0.0,
                }

    baseline_elapsed = time.perf_counter() - t0
    print(f"\n[Eval] Baselines finished in {baseline_elapsed:.1f}s\n")

    # ── GARRO evaluation (main process, uses loaded model) ────────────────
    print("[Eval] Loading GARRO checkpoint …")
    agent = PPOAgent(
        config,
        k_paths=config["network"]["k_paths"],
        device=device,
        compile_model=False,    # no need to compile for one-shot eval
    )
    agent.load(args.checkpoint)
    print("[Eval] Running GARRO (PPO) …")
    results["GARRO"] = run_garro(env, agent, args.episodes)

    total_elapsed = time.perf_counter() - t0
    print(f"\n[Eval] Total evaluation time: {total_elapsed:.1f}s\n")

    # ── Print results table ───────────────────────────────────────────────
    # Preserve display order: Random → OSPF → ECMP → GARRO
    ordered = ["Random", "OSPF", "ECMP", "GARRO"]
    print(f"{sep}")
    print(f"  {'Algorithm':<12} {'Mean Reward':>14} {'Std Dev':>10} "
          f"{'Min':>10} {'Max':>10}")
    print(f"  {'─'*58}")
    for name in ordered:
        if name not in results:
            continue
        r      = results[name]
        marker = " ← GARRO" if name == "GARRO" else ""
        print(f"  {name:<12} {r['mean_reward']:>14.4f} {r['std']:>10.4f} "
              f"{r['min']:>10.4f} {r['max']:>10.4f}{marker}")
    print(f"{sep}\n")

    # ── Bar chart ─────────────────────────────────────────────────────────
    names  = [n for n in ordered if n in results]
    means  = [results[n]["mean_reward"] for n in names]
    stds   = [results[n]["std"]         for n in names]
    colors = ["#95A5A6", "#E74C3C", "#F39C12", "#2ECC71"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        names, means, yerr=stds, color=colors[:len(names)],
        capsize=7, edgecolor="black", linewidth=0.8, alpha=0.9,
    )

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
        f"({args.episodes} episodes) | Device: {device}",
        fontsize=11,
    )
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    chart_path = f"eval_results_{args.topology}.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"[Eval] Bar chart saved → {chart_path}")

    # ── CSV ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(
        {n: results[n] for n in ordered if n in results}
    ).T
    csv_path = f"eval_results_{args.topology}.csv"
    df.to_csv(csv_path)
    print(f"[Eval] CSV saved       → {csv_path}")
    print("\n[Eval] Done. ✓")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Required for ProcessPoolExecutor on all platforms (especially Windows/macOS)
    multiprocessing.freeze_support()

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
