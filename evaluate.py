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
    evaluation_outputs/<model>_<topology>_ep<episodes>_<timestamp>/
        eval_results_<model>_<topology>_ep<episodes>.png   Bar chart of mean episode rewards
        eval_results_<model>_<topology>_ep<episodes>.csv   Table of results
"""
import argparse
import multiprocessing
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

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
    "fat_tree": lambda: get_fat_tree(k=4),
}


def _safe_path_name(value: str) -> str:
    """Return a filesystem-friendly name while preserving readability."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._-") or "model"


def _model_name_from_checkpoint(checkpoint: str) -> str:
    return _safe_path_name(Path(checkpoint).stem)


def _label_for_checkpoint(path: str) -> str:
    stem = Path(path).stem
    match = re.search(r"ep(\d+)", stem)
    if match:
        ep_num = int(match.group(1))
        k_num = f"{ep_num // 1000}k" if ep_num >= 1000 else str(ep_num)
        return f"GARRO (ep{k_num})"
    if "final" in stem.lower():
        return "GARRO (Final)"
    if "latest" in stem.lower():
        return "GARRO (Latest)"
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-")
    return f"GARRO ({cleaned[:15]})"


def _make_eval_output_dir(
    base_dir: str,
    model_name: str,
    topology: str,
    episodes: int,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_stem = f"{model_name}_{topology}_ep{episodes}_{timestamp}"
    base_path = Path(base_dir)

    for suffix in ["", *[f"_{i}" for i in range(1, 1000)]]:
        output_dir = base_path / f"{folder_stem}{suffix}"
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
            return output_dir
        except FileExistsError:
            continue

    raise RuntimeError(f"Could not create a unique output directory for {folder_stem}")


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

    rewards   = []
    all_lats  = []
    all_loss  = []
    all_tputs = []
    all_vars  = []
    desc      = f"{algo_name:<8}"

    for ep in tqdm(range(episodes), desc=desc, position=0, leave=True,
                   dynamic_ncols=True):
        obs, _ = env.reset(seed=ep + seed)
        done   = False
        ep_r   = 0.0
        step   = 0
        ep_lats  = []
        ep_loss  = []
        ep_tputs = []
        ep_vars  = []

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

            obs, r, terminated, truncated, info = env.step(action)
            done  = terminated or truncated
            ep_r += r
            raw = info.get("raw_reward_terms", {})
            ep_lats.append(float(info.get("path_latency_ms", raw.get("D_path", 0.0))))
            ep_loss.append(float(raw.get("total_loss", 0.0) * 100.0))
            ep_tputs.append(float(raw.get("tput_ratio", 1.0)))
            ep_vars.append(float(raw.get("util_variance", 0.0)))
            step += 1

        rewards.append(ep_r)
        if ep_lats:
            all_lats.append(float(np.mean(ep_lats)))
            all_loss.append(float(np.mean(ep_loss)))
            all_tputs.append(float(np.mean(ep_tputs)))
            all_vars.append(float(np.mean(ep_vars)))

    return {
        "name":            algo_name,
        "mean_reward":     float(np.mean(rewards)),
        "std":             float(np.std(rewards)),
        "min":             float(np.min(rewards)),
        "max":             float(np.max(rewards)),
        "latency_ms":      float(np.mean(all_lats)) if all_lats else 0.0,
        "packet_loss_pct": float(np.mean(all_loss)) if all_loss else 0.0,
        "tput_ratio":      float(np.mean(all_tputs)) if all_tputs else 1.0,
        "util_variance":   float(np.mean(all_vars)) if all_vars else 0.0,
    }


def run_garro(
    env:      MM1KNetworkEnv,
    agent:    PPOAgent,
    episodes: int,
    deterministic: bool = True,
    desc:     str = "GARRO   ",
) -> dict:
    """
    GARRO (PPO + Graph Transformer) evaluation — no gradient updates.
    Runs on the main process so the loaded model stays in-memory.
    """
    rewards   = []
    all_lats  = []
    all_loss  = []
    all_tputs = []
    all_vars  = []
    agent.encoder.eval()
    agent.ac_net.eval()

    for ep in tqdm(range(episodes), desc=desc, dynamic_ncols=True,
                  position=0, leave=True):
        obs, _ = env.reset(seed=ep + 42)
        done   = False
        ep_r   = 0.0
        ep_lats  = []
        ep_loss  = []
        ep_tputs = []
        ep_vars  = []

        while not done:
            action, _, _ = agent.select_action(
                env.G, env.candidate_paths, deterministic=deterministic
            )
            obs, r, terminated, truncated, info = env.step(action)
            done  = terminated or truncated
            ep_r += r
            raw = info.get("raw_reward_terms", {})
            ep_lats.append(float(info.get("path_latency_ms", raw.get("D_path", 0.0))))
            ep_loss.append(float(raw.get("total_loss", 0.0) * 100.0))
            ep_tputs.append(float(raw.get("tput_ratio", 1.0)))
            ep_vars.append(float(raw.get("util_variance", 0.0)))

        rewards.append(ep_r)
        if ep_lats:
            all_lats.append(float(np.mean(ep_lats)))
            all_loss.append(float(np.mean(ep_loss)))
            all_tputs.append(float(np.mean(ep_tputs)))
            all_vars.append(float(np.mean(ep_vars)))

    return {
        "mean_reward":     float(np.mean(rewards)),
        "std":             float(np.std(rewards)),
        "min":             float(np.min(rewards)),
        "max":             float(np.max(rewards)),
        "latency_ms":      float(np.mean(all_lats)) if all_lats else 0.0,
        "packet_loss_pct": float(np.mean(all_loss)) if all_loss else 0.0,
        "tput_ratio":      float(np.mean(all_tputs)) if all_tputs else 1.0,
        "util_variance":   float(np.mean(all_vars)) if all_vars else 0.0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    with open("config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.arrival_rate is not None:
        config["mm1k"]["base_arrival_rate"] = args.arrival_rate

    # Resolve all checkpoint files from arguments (file, list, directory, or wildcard)
    if isinstance(args.checkpoint, str):
        raw_checkpoints = [args.checkpoint]
    else:
        raw_checkpoints = list(args.checkpoint)

    import glob
    checkpoint_paths = []
    for cp in raw_checkpoints:
        if os.path.isdir(cp):
            found = glob.glob(os.path.join(cp, "**", "*.pt"), recursive=True)
            checkpoint_paths.extend(found)
        elif "*" in cp:
            checkpoint_paths.extend(glob.glob(cp, recursive=True))
        elif os.path.isfile(cp):
            checkpoint_paths.append(cp)

    checkpoint_paths = [p for p in checkpoint_paths if p.endswith(".pt")]
    # If final is present, filter out latest to avoid redundant evaluation of identical weights
    has_final = any("final" in os.path.basename(p) for p in checkpoint_paths)
    if has_final:
        checkpoint_paths = [p for p in checkpoint_paths if "latest" not in os.path.basename(p)]

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for p in checkpoint_paths:
        abs_p = os.path.abspath(p)
        if abs_p not in seen and os.path.exists(abs_p):
            seen.add(abs_p)
            deduped.append(abs_p)
    checkpoint_paths = deduped

    # Sort checkpoints: ep<N> numerically first, then final/latest
    def _sort_key(p):
        stem = Path(p).stem
        match = re.search(r"ep(\d+)", stem)
        if match:
            return (0, int(match.group(1)))
        if "final" in stem.lower():
            return (1, 999999999)
        return (2, stem)

    checkpoint_paths.sort(key=_sort_key)

    if not checkpoint_paths:
        raise FileNotFoundError(f"No valid .pt checkpoint files found for: {args.checkpoint}")

    if len(checkpoint_paths) == 1:
        model_name = _model_name_from_checkpoint(checkpoint_paths[0])
    else:
        model_name = f"multi_ckpt_{len(checkpoint_paths)}"

    output_dir = _make_eval_output_dir(
        args.output_dir,
        model_name,
        args.topology,
        args.episodes,
    )

    # Auto-detect best device for GARRO inference
    device   = _best_device()
    n_cores  = multiprocessing.cpu_count()

    G   = TOPOLOGY_MAP[args.topology]()
    env = MM1KNetworkEnv(G, config)

    sep = "=" * 76
    print(f"\n{sep}")
    print(f"  GARRO Multi-Checkpoint & Baseline Benchmarking — {args.topology.upper()} "
          f"({G.number_of_nodes()} nodes, {G.number_of_edges()} links)")
    print(f"  Episodes per algorithm : {args.episodes}")
    print(f"  Base Arrival Rate (λ)  : {config['mm1k']['base_arrival_rate']} (Service Rate μ: {config['mm1k']['base_service_rate']})")
    print(f"  Checkpoints detected   : {len(checkpoint_paths)} ({', '.join(os.path.basename(p) for p in checkpoint_paths)})")
    print(f"  Output directory       : {output_dir}")
    print(f"  Device (GARRO)         : {device}")
    print(f"  CPU cores              : {n_cores}")
    print(f"  Parallel baselines     : 3 (OSPF + ECMP + Random run simultaneously)")
    print(f"{sep}\n")

    results = {}
    t0      = time.perf_counter()

    # ── Parallel baseline evaluation (Run ONCE) ───────────────────────────
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
                    "latency_ms": 0.0, "packet_loss_pct": 0.0,
                    "tput_ratio": 0.0, "util_variance": 0.0,
                }

    baseline_elapsed = time.perf_counter() - t0
    print(f"\n[Eval] Baselines finished in {baseline_elapsed:.1f}s\n")

    # ── GARRO evaluation (sequential across checkpoints) ──────────────────
    print(f"[Eval] Initializing GARRO agent for {len(checkpoint_paths)} checkpoint(s) …")
    agent = PPOAgent(
        config,
        k_paths=config["network"]["k_paths"],
        num_nodes=G.number_of_nodes(),
        device=device,
        compile_model=False,    # no need to compile for one-shot eval
    )

    garro_labels = []
    for ckpt_path in checkpoint_paths:
        label = _label_for_checkpoint(ckpt_path)
        orig_label = label
        c = 2
        while label in results:
            label = f"{orig_label}-{c}"
            c += 1
        garro_labels.append(label)
        print(f"[Eval] Loading {label} ← {os.path.basename(ckpt_path)} …")
        agent.load(ckpt_path)
        results[label] = run_garro(env, agent, args.episodes, desc=f"{label:<14}")

    total_elapsed = time.perf_counter() - t0
    print(f"\n[Eval] Total evaluation time: {total_elapsed:.1f}s\n")

    # ── Identify Best Model ───────────────────────────────────────────────
    best_garro = max(garro_labels, key=lambda l: results[l]["mean_reward"])
    best_res = results[best_garro]

    # ── Print results table ───────────────────────────────────────────────
    ordered = ["Random", "OSPF", "ECMP"] + garro_labels
    print(f"{sep}")
    print(f"  {'Algorithm / Model':<18} {'Mean Reward':>12} {'Latency (ms)':>14} "
          f"{'Loss (%)':>10} {'Tput Ratio':>12} {'Link Var':>10}")
    print(f"  {'─'*78}")
    for name in ordered:
        if name not in results:
            continue
        r      = results[name]
        marker = " 🏆 BEST" if name == best_garro else (" ← GARRO" if name in garro_labels else "")
        print(f"  {name:<18} {r['mean_reward']:>12.4f} {r['latency_ms']:>14.2f} "
              f"{r['packet_loss_pct']:>10.3f}% {r['tput_ratio']:>12.4f} {r['util_variance']:>10.4f}{marker}")
    print(f"{sep}")
    print(f"  🏆 Top-Performing GARRO Checkpoint : {best_garro}")
    print(f"     Mean Reward: {best_res['mean_reward']:+.4f} | "
          f"Average Latency: {best_res['latency_ms']:.2f} ms | "
          f"Packet Loss: {best_res['packet_loss_pct']:.3f}%\n")

    # ── Bar chart (Reward) ────────────────────────────────────────────────
    names  = [n for n in ordered if n in results]
    means  = [results[n]["mean_reward"] for n in names]
    stds   = [results[n]["std"]         for n in names]

    base_colors = {"Random": "#95A5A6", "OSPF": "#E74C3C", "ECMP": "#F39C12"}
    palette = ["#2ECC71", "#1ABC9C", "#3498DB", "#9B59B6", "#E67E22", "#27AE60"]
    colors = []
    g_idx = 0
    for n in names:
        if n in base_colors:
            colors.append(base_colors[n])
        else:
            colors.append(palette[g_idx % len(palette)])
            g_idx += 1

    fig, ax = plt.subplots(figsize=(max(9, len(names) * 1.5), 5.5))
    bars = ax.bar(
        names, means, yerr=stds, color=colors[:len(names)],
        capsize=7, edgecolor="black", linewidth=0.8, alpha=0.9,
    )

    for bar, mean_val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.002,
            f"{mean_val:.3f}",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
        )

    ax.set_ylabel("Mean Episode Reward (Higher is Better)", fontsize=11)
    ax.set_title(
        f"GARRO Multi-Checkpoint Routing Comparison — {args.topology.upper()} "
        f"({args.episodes} episodes) | Device: {device}",
        fontsize=11,
    )
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()

    output_file_stem = f"eval_results_{model_name}_{args.topology}_ep{args.episodes}"
    chart_path = output_dir / f"{output_file_stem}.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"[Eval] Reward chart saved       → {chart_path}")

    # ── 4-Panel DCN Performance Metrics Figure ────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(max(12, len(names) * 1.8), 8.5))
    fig.suptitle(f"GARRO DCN Routing Benchmarks ({args.topology.upper()}) — All Checkpoints vs Baselines", fontsize=13, fontweight="bold")

    # 1. Mean Reward
    axes[0, 0].bar(names, [results[n]["mean_reward"] for n in names], color=colors[:len(names)], edgecolor="black")
    axes[0, 0].set_title("Mean Reward (Higher is Better)")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(axis="y", alpha=0.3)
    axes[0, 0].tick_params(axis="x", rotation=15)

    # 2. Latency (ms)
    axes[0, 1].bar(names, [results[n]["latency_ms"] for n in names], color=colors[:len(names)], edgecolor="black")
    axes[0, 1].set_title("Path Latency (ms) (Lower is Better)")
    axes[0, 1].set_ylabel("Latency (ms)")
    axes[0, 1].grid(axis="y", alpha=0.3)
    axes[0, 1].tick_params(axis="x", rotation=15)

    # 3. Packet Loss (%)
    axes[1, 0].bar(names, [results[n]["packet_loss_pct"] for n in names], color=colors[:len(names)], edgecolor="black")
    axes[1, 0].set_title("Packet Loss Rate (%) (Lower is Better)")
    axes[1, 0].set_ylabel("Loss Rate (%)")
    axes[1, 0].grid(axis="y", alpha=0.3)
    axes[1, 0].tick_params(axis="x", rotation=15)

    # 4. Link Utilization Variance (Load Balance)
    axes[1, 1].bar(names, [results[n]["util_variance"] for n in names], color=colors[:len(names)], edgecolor="black")
    axes[1, 1].set_title("Link Utilization Variance (Lower is More Balanced)")
    axes[1, 1].set_ylabel("Variance")
    axes[1, 1].grid(axis="y", alpha=0.3)
    axes[1, 1].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    dcn_chart_path = output_dir / f"eval_dcn_metrics_{model_name}_{args.topology}_ep{args.episodes}.png"
    fig.savefig(dcn_chart_path, dpi=150)
    plt.close(fig)
    print(f"[Eval] 4-Panel DCN QoS chart    → {dcn_chart_path}")

    # ── Checkpoint Progression Line Chart (if >= 2 GARRO models) ──────────
    if len(garro_labels) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        fig.suptitle(f"GARRO Checkpoint Evolution & Convergence across Curriculum Stages", fontsize=13, fontweight="bold")

        x_indices = list(range(len(garro_labels)))
        g_rewards = [results[l]["mean_reward"] for l in garro_labels]
        g_lats    = [results[l]["latency_ms"] for l in garro_labels]
        g_losses  = [results[l]["packet_loss_pct"] for l in garro_labels]

        # 1. Reward Evolution
        axes[0].plot(x_indices, g_rewards, marker="o", color="#2ECC71", linewidth=2.5, label="GARRO Progression")
        if "ECMP" in results:
            axes[0].axhline(y=results["ECMP"]["mean_reward"], color="#F39C12", linestyle="--", label="ECMP Baseline")
        if "OSPF" in results:
            axes[0].axhline(y=results["OSPF"]["mean_reward"], color="#E74C3C", linestyle="--", label="OSPF Baseline")
        axes[0].set_xticks(x_indices)
        axes[0].set_xticklabels(garro_labels, rotation=15)
        axes[0].set_title("Mean Reward Progression")
        axes[0].set_ylabel("Reward (Higher is Better)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="lower right")

        # 2. Latency Evolution
        axes[1].plot(x_indices, g_lats, marker="s", color="#3498DB", linewidth=2.5, label="GARRO Progression")
        if "ECMP" in results:
            axes[1].axhline(y=results["ECMP"]["latency_ms"], color="#F39C12", linestyle="--", label="ECMP Baseline")
        if "OSPF" in results:
            axes[1].axhline(y=results["OSPF"]["latency_ms"], color="#E74C3C", linestyle="--", label="OSPF Baseline")
        axes[1].set_xticks(x_indices)
        axes[1].set_xticklabels(garro_labels, rotation=15)
        axes[1].set_title("Path Latency (ms)")
        axes[1].set_ylabel("Latency in ms (Lower is Better)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right")

        # 3. Packet Loss Evolution
        axes[2].plot(x_indices, g_losses, marker="^", color="#E74C3C", linewidth=2.5, label="GARRO Progression")
        if "ECMP" in results:
            axes[2].axhline(y=results["ECMP"]["packet_loss_pct"], color="#F39C12", linestyle="--", label="ECMP Baseline")
        if "OSPF" in results:
            axes[2].axhline(y=results["OSPF"]["packet_loss_pct"], color="#E74C3C", linestyle="--", label="OSPF Baseline")
        axes[2].set_xticks(x_indices)
        axes[2].set_xticklabels(garro_labels, rotation=15)
        axes[2].set_title("Packet Loss Rate (%)")
        axes[2].set_ylabel("Loss Rate % (Lower is Better)")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="upper right")

        fig.tight_layout()
        prog_chart_path = output_dir / f"eval_checkpoint_evolution_{args.topology}_ep{args.episodes}.png"
        fig.savefig(prog_chart_path, dpi=150)
        plt.close(fig)
        print(f"[Eval] Evolution chart saved    → {prog_chart_path}")

    # ── CSV ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(
        {n: results[n] for n in ordered if n in results}
    ).T
    csv_path = output_dir / f"{output_file_stem}.csv"
    df.to_csv(csv_path)
    print(f"[Eval] CSV saved                 → {csv_path}")
    print("\n[Eval] Done. ✓")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(
        description="GARRO Evaluation — benchmark PPO vs OSPF / ECMP / Random"
    )
    parser.add_argument(
        "--checkpoint",
        nargs="+",
        required=True,
        help="Path(s) to trained GARRO checkpoint (.pt) or directory containing checkpoints",
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
    parser.add_argument(
        "--output-dir",
        default="evaluation_outputs",
        help="Base directory for per-evaluation output folders (default: evaluation_outputs)",
    )
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=None,
        help="Override base arrival rate λ in packets/sec (default: from config.yaml)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of random seeds / trials to aggregate across (default: 1)",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        default=True,
        help="Compare GARRO against all baselines (OSPF, ECMP, Random)",
    )
    args = parser.parse_args()
    main(args)
