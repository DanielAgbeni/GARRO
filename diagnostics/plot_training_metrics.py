"""
Diagnostic Log Plotter for GARRO Offline Training
=================================================
Parses stdout/log files from `train_offline.py` (e.g. Kaggle run logs)
and plots aligned subplots for:
  1. Episode Reward (Raw + MA-500)
  2. Policy Entropy (Ent)
  3. Approximate KL Divergence (KL)
  4. Value Loss (VL) & Policy Loss (PL)

Usage:
    python diagnostics/plot_training_metrics.py --log path/to/log.txt
"""

import argparse
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_log(log_path: str):
    episodes = []
    rewards = []
    
    update_episodes = []
    policy_losses = []
    value_losses = []
    entropies = []
    kl_divs = []
    
    # Track current episode for update metrics
    curr_ep = 0
    
    # Regex patterns
    ep_pattern = re.compile(r"\[Ep\s+(\d+)/\d+\]\s+Avg Reward:\s+([+-]?\d+\.\d+)")
    update_pattern = re.compile(
        r"↳ PPO update \| PL:\s+([+-]?\d+\.\d+)\s+VL:\s+(\d+\.\d+)\s+Ent:\s+(\d+\.\d+)\s+KL:\s+([+-]?\d+\.\d+)"
    )
    
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            ep_match = ep_pattern.search(line)
            if ep_match:
                curr_ep = int(ep_match.group(1))
                rewards.append(float(ep_match.group(2)))
                episodes.append(curr_ep)
                continue
                
            up_match = update_pattern.search(line)
            if up_match:
                update_episodes.append(curr_ep)
                policy_losses.append(float(up_match.group(1)))
                value_losses.append(float(up_match.group(2)))
                entropies.append(float(up_match.group(3)))
                kl_divs.append(float(up_match.group(4)))
                
    return {
        "episodes": np.array(episodes),
        "rewards": np.array(rewards),
        "update_episodes": np.array(update_episodes),
        "policy_losses": np.array(policy_losses),
        "value_losses": np.array(value_losses),
        "entropies": np.array(entropies),
        "kl_divs": np.array(kl_divs),
    }


def plot_metrics(data: dict, output_path: str = "diagnostics/training_diagnostics.png"):
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # Subplot 1: Rewards
    ax1 = axes[0]
    if len(data["episodes"]) > 0:
        ax1.plot(data["episodes"], data["rewards"], label="Checkpoint Avg Reward", color="#1f77b4", marker="o")
        ax1.set_ylabel("Avg Reward")
        ax1.set_title("GARRO Offline Training Diagnostics — Log Analysis")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right")
        
    # Subplot 2: Entropy
    ax2 = axes[1]
    if len(data["update_episodes"]) > 0:
        ax2.plot(data["update_episodes"], data["entropies"], label="Policy Entropy (Ent)", color="#ff7f0e", alpha=0.7)
        ax2.axhline(0.1, color="red", linestyle="--", alpha=0.5, label="Collapse Threshold (<0.1)")
        ax2.set_ylabel("Entropy")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right")
        
    # Subplot 3: KL Divergence
    ax3 = axes[2]
    if len(data["update_episodes"]) > 0:
        ax3.plot(data["update_episodes"], data["kl_divs"], label="Approx KL (KL)", color="#2ca02c", alpha=0.7)
        ax3.axhline(0.05, color="red", linestyle="--", alpha=0.5, label="Target KL Ceiling (0.05)")
        ax3.set_ylabel("Approx KL")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="upper right")
        
    # Subplot 4: Value & Policy Loss
    ax4 = axes[3]
    if len(data["update_episodes"]) > 0:
        ax4.plot(data["update_episodes"], data["value_losses"], label="Value Loss (VL)", color="#d62728", alpha=0.7)
        ax4.set_ylabel("Value Loss")
        ax4.set_xlabel("Episode")
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc="upper right")
        
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[Diagnostics] Saved diagnostic plot to -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot GARRO training metrics from stdout log")
    parser.add_argument("--log", required=True, help="Path to training log file")
    parser.add_argument("--out", default="diagnostics/training_diagnostics.png", help="Output PNG path")
    args = parser.parse_args()
    
    data = parse_log(args.log)
    print(f"[Diagnostics] Parsed {len(data['episodes'])} checkpoint logs and {len(data['update_episodes'])} PPO updates.")
    plot_metrics(data, args.out)
