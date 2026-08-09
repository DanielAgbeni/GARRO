"""
GARRO Training Curve Stitcher
=============================
Parses stdout/log files from multiple training sessions (e.g., Ep 0-4000 and Ep 4000-10000)
and stitches them into a single, unified 0 - 10,000 episode plot.

Usage:
    python diagnostics/stitch_training_curves.py --logs log1.txt log2.txt --out checkpoints/unified_training_curve.png
"""

import argparse
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_checkpoint_logs(log_paths: list):
    all_episodes = []
    all_rewards = []
    
    ep_pattern = re.compile(r"\[Ep\s+(\d+)/\d+\]\s+Avg Reward:\s+([+-]?\d+\.\d+)")
    
    for log_path in log_paths:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                ep_match = ep_pattern.search(line)
                if ep_match:
                    ep_num = int(ep_match.group(1))
                    reward = float(ep_match.group(2))
                    if ep_num not in all_episodes:
                        all_episodes.append(ep_num)
                        all_rewards.append(reward)
                        
    # Sort by episode index
    sorted_pairs = sorted(zip(all_episodes, all_rewards), key=lambda x: x[0])
    eps = np.array([x[0] for x in sorted_pairs])
    rews = np.array([x[1] for x in sorted_pairs])
    return eps, rews


def plot_unified_curve(eps: np.ndarray, rews: np.ndarray, output_path: str):
    fig, ax = plt.subplots(figsize=(14, 5))
    
    ax.plot(eps, rews, color="#1f77b4", marker="o", linewidth=2.0, label="Checkpoint Avg Reward")
    
    # MA smoothing
    if len(rews) >= 3:
        window = min(5, len(rews))
        kernel = np.ones(window) / window
        smooth = np.convolve(rews, kernel, mode="valid")
        smooth_x = eps[window - 1:]
        ax.plot(smooth_x, smooth, color="#e377c2", linewidth=2.5, linestyle="--", label=f"Moving Avg (w={window})")
        
    ax.set_xlabel("Episode (Absolute)")
    ax.set_ylabel("Average Episode Reward")
    ax.set_title("GARRO Offline Training — Unified 0 to 10,000 Episode Progression")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[Stitcher] Unified training curve saved to -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitch training curves across session checkpoints")
    parser.add_argument("--logs", nargs="+", required=True, help="List of log files in chronological order")
    parser.add_argument("--out", default="checkpoints/unified_training_curve.png", help="Output PNG path")
    args = parser.parse_args()
    
    eps, rews = parse_checkpoint_logs(args.logs)
    print(f"[Stitcher] Combined {len(eps)} checkpoint points spanning episodes {eps.min() if len(eps)>0 else 0} to {eps.max() if len(eps)>0 else 0}.")
    plot_unified_curve(eps, rews, args.out)
