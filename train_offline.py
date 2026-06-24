"""
GARRO Phase 1 — Offline Digital Twin Training.

Trains the PPO + Graph Transformer agent entirely within the M/M/1/K
Digital Twin environment.  No live network, Mininet, or OS-Ken required.

Usage
-----
    # Activate virtual environment first:
    source garro_env/bin/activate

    # NSFNET (14 nodes) — fast, ~10-20 min for 10 000 episodes
    python train_offline.py --topology nsfnet --episodes 10000

    # GEANT2 (24 nodes) — medium
    python train_offline.py --topology geant2 --episodes 20000

    # Fat-Tree k=8 (80 nodes) — run overnight
    python train_offline.py --topology fat_tree --episodes 50000

Outputs
-------
    checkpoints/garro_<topology>_ep<N>.pt       Periodic checkpoints
    checkpoints/garro_<topology>_final.pt        Final model
    checkpoints/training_curve_<topology>.png    Reward plot
"""
import argparse
import copy
import os

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (safe in WSL / headless)
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from digital_twin.mm1k_env import MM1KNetworkEnv
from model.ppo_agent import PPOAgent
from topologies.nsfnet import get_nsfnet
from topologies.geant2 import get_geant2
from topologies.fat_tree import get_fat_tree


# ── Topology registry ─────────────────────────────────────────────────────────

TOPOLOGY_MAP = {
    "nsfnet":    get_nsfnet,
    "geant2":    get_geant2,
    "fat_tree":  lambda: get_fat_tree(k=8),
}


# ── Main training loop ────────────────────────────────────────────────────────

def main(args):
    # ── Load configuration ────────────────────────────────────────────────
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    config["network"]["topology"] = args.topology

    total_episodes   = args.episodes or config["training"]["offline_episodes"]
    checkpoint_dir   = config["training"]["checkpoint_path"]
    update_interval  = config["training"].get("update_interval", 512)
    checkpoint_every = config["training"]["checkpoint_interval"]

    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  GARRO Offline Training")
    print(f"  Topology : {args.topology.upper()}")
    print(f"  Episodes : {total_episodes:,}")
    print(f"  Device   : {device}")
    print(f"{'='*60}\n")

    # ── Build graph + environment ─────────────────────────────────────────
    G   = TOPOLOGY_MAP[args.topology]()
    env = MM1KNetworkEnv(G, config)

    k_paths = config["network"]["k_paths"]
    agent   = PPOAgent(config, k_paths=k_paths, device=device)

    print(f"[Init] Nodes: {G.number_of_nodes()} | "
          f"Edges: {G.number_of_edges()} | "
          f"K-paths: {k_paths}\n")

    # ── Training state ────────────────────────────────────────────────────
    obs, info = env.reset()

    ep_idx            = 0
    step_count        = 0
    ep_reward         = 0.0
    episode_rewards:  list = []          # One entry per completed episode
    checkpoint_rewards: list = []        # Rolling window for checkpoint logging

    pbar = tqdm(total=total_episodes, desc="Training", unit="ep",
                dynamic_ncols=True)

    # ── Main loop ─────────────────────────────────────────────────────────
    while ep_idx < total_episodes:

        # 1. Agent selects action using the live NetworkX graph state
        candidate_paths = env.candidate_paths
        action, log_prob, value = agent.select_action(env.G, candidate_paths)

        # 2. Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 3. Store transition — graph snapshot deep-copied for buffer
        agent.buffer.add(
            state    = copy.deepcopy(env.G),
            action   = action,
            log_prob = log_prob,
            reward   = reward,
            value    = value,
            done     = done,
        )

        obs         = next_obs
        ep_reward  += reward
        step_count += 1

        # 4. Episode bookkeeping
        if done:
            ep_idx += 1
            episode_rewards.append(ep_reward)
            checkpoint_rewards.append(ep_reward)
            ep_reward = 0.0
            obs, info = env.reset()
            pbar.update(1)

            # Periodic checkpoint + console log
            if ep_idx % checkpoint_every == 0:
                avg_r = float(np.mean(checkpoint_rewards[-checkpoint_every:]))
                pbar.write(
                    f"[Ep {ep_idx:6d}/{total_episodes}] "
                    f"Avg Reward: {avg_r:+.4f} | "
                    f"Steps: {step_count:,}"
                )
                ckpt_path = os.path.join(
                    checkpoint_dir,
                    f"garro_{args.topology}_ep{ep_idx}.pt"
                )
                agent.save(ckpt_path)

        # 5. PPO update every `update_interval` environment steps
        if step_count % update_interval == 0 and len(agent.buffer) > 0:
            metrics = agent.update()
            if metrics:
                pbar.write(
                    f"  ↳ PPO update | "
                    f"PL: {metrics['policy_loss']:+.4f}  "
                    f"VL: {metrics['value_loss']:.4f}  "
                    f"Ent: {metrics['entropy']:.4f}  "
                    f"KL: {metrics['approx_kl']:.4f}"
                )

    pbar.close()

    # ── Final checkpoint ──────────────────────────────────────────────────
    final_path = os.path.join(checkpoint_dir, f"garro_{args.topology}_final.pt")
    agent.save(final_path)
    print(f"\n[Train] Final model saved → {final_path}")

    # ── Training curve plot ───────────────────────────────────────────────
    if episode_rewards:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(episode_rewards, alpha=0.3, color="#4C9BE8", label="Episode Reward")

        # Smoothed moving average
        window = min(500, len(episode_rewards) // 5 or 1)
        if len(episode_rewards) >= window:
            kernel = np.ones(window) / window
            smooth = np.convolve(episode_rewards, kernel, mode="valid")
            ax.plot(
                range(window - 1, len(episode_rewards)),
                smooth,
                linewidth=2.0,
                color="#E84C4C",
                label=f"MA-{window}",
            )

        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Episode Reward")
        ax.set_title(
            f"GARRO Offline Training — {args.topology.upper()} "
            f"({G.number_of_nodes()} nodes)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_path = os.path.join(
            checkpoint_dir, f"training_curve_{args.topology}.png"
        )
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"[Train] Training curve saved → {plot_path}")

    print("\n[Train] Done. ✓")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GARRO Phase 1 — Offline Digital Twin Training"
    )
    parser.add_argument(
        "--topology",
        default="nsfnet",
        choices=list(TOPOLOGY_MAP.keys()),
        help="Network topology to train on (default: nsfnet)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Total training episodes (default: from config.yaml)",
    )
    args = parser.parse_args()
    main(args)
