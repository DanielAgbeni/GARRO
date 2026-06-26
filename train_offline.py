"""
GARRO Phase 1 — Offline Digital Twin Training.

Trains the PPO + Graph Transformer agent entirely within the M/M/1/K
Digital Twin environment.  No live network, Mininet, or OS-Ken required.

System Resource Utilization
-----------------------------
* Auto-detects CUDA / MPS / CPU and trains on the best available device.
* Pins all CPU-threading backends (PyTorch, OpenMP, MKL, OpenBLAS, NumExpr)
  to use every available core — configured once at startup.
* Scales batch_size and update_interval proportionally to CPU core count
  so larger machines automatically collect bigger rollouts per update.
* torch.compile is enabled by default (set compile_model: false in
  config.yaml to disable during debugging).
* Prints a hardware summary banner so you always know what hardware the
  run is using.

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

    # Disable torch.compile for debugging
    python train_offline.py --topology nsfnet --no-compile

Outputs
-------
    checkpoints/garro_<topology>_ep<N>.pt       Periodic checkpoints
    checkpoints/garro_<topology>_final.pt        Final model
    checkpoints/training_curve_<topology>.png    Reward plot
"""
import argparse
import multiprocessing
import os
import time

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (safe in WSL / headless)
import matplotlib.pyplot as plt
import numpy as np
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
    "nsfnet":    get_nsfnet,
    "geant2":    get_geant2,
    "fat_tree":  lambda: get_fat_tree(k=8),
}


# ── Hardware banner ───────────────────────────────────────────────────────────

def _print_hardware_banner(
    device: torch.device,
    n_cores: int,
    compile_model: bool,
    topology: str,
    total_episodes: int,
    batch_size: int,
    update_interval: int,
    cuda_scaled: bool = False,
) -> None:
    """Print a formatted summary of the active hardware configuration."""
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  GARRO Offline Training")
    print(f"  Topology    : {topology.upper()}")
    print(f"  Episodes    : {total_episodes:,}")
    print(f"{'─'*64}")
    print(f"  Device      : {device}  ", end="")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"({props.name}, {props.total_memory/1e9:.1f} GB VRAM)")
    elif device.type == "mps":
        print("(Apple Metal Performance Shaders)")
    else:
        print("(CPU — torch.compile active)" if compile_model else "(CPU)")
    print(f"  CPU cores   : {n_cores}")
    scaling = "CUDA auto-scaled" if cuda_scaled else "auto-scaled to core count"
    print(f"  Batch size  : {batch_size}  ({scaling})")
    print(f"  Update every: {update_interval} steps")
    print(f"  Compile     : {compile_model}")
    print(f"{sep}\n")


# ── Main training loop ────────────────────────────────────────────────────────

def main(args):
    # ── Load configuration ────────────────────────────────────────────────
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    config["network"]["topology"] = args.topology

    total_episodes   = args.episodes or config["training"]["offline_episodes"]
    checkpoint_dir   = config["training"]["checkpoint_path"]
    checkpoint_every = config["training"]["checkpoint_interval"]

    # ── Hardware setup ────────────────────────────────────────────────────
    n_cores = multiprocessing.cpu_count()

    # Auto-detect best device (CUDA > MPS > CPU)
    device = _best_device()

    # Resolve compile flag: CLI flag > config.yaml > default True
    if args.no_compile:
        compile_model = False
    else:
        compile_model = config.get("training", {}).get("compile_model", True)

    # ── Auto-scale hyperparameters based on hardware ────────────────────
    # CUDA GPUs can handle much larger batches and rollouts efficiently.
    # On CPU/MPS, keep the conservative config.yaml defaults.
    base_batch      = config["ppo"]["batch_size"]
    base_interval   = config["training"].get("update_interval", 512)
    base_epochs     = config["ppo"]["update_epochs"]

    cuda_scaled = False
    if device.type == "cuda":
        # ── CUDA-optimised overrides ──────────────────────────────────
        # GPU can process larger mini-batches in a single matmul, and
        # longer rollouts give stabler GAE advantage estimates.
        batch_size      = max(base_batch, 256)       # 64 → 256 on CUDA
        update_interval = max(base_interval, 2048)   # 512 → 2048 on CUDA
        update_epochs   = max(base_epochs, 15)       # 10 → 15 on CUDA
        cuda_scaled     = True
        print(f"[CUDA] Auto-scaled: batch_size={batch_size}, "
              f"update_interval={update_interval}, "
              f"update_epochs={update_epochs}")
    else:
        # ── CPU / MPS — scale with core count only ────────────────────
        scale = max(1, n_cores // 2)
        batch_size      = base_batch * scale
        update_interval = base_interval * scale
        update_epochs   = base_epochs

    # Write back so PPOAgent reads the scaled values
    config["ppo"]["batch_size"]     = batch_size
    config["ppo"]["update_epochs"]  = update_epochs

    os.makedirs(checkpoint_dir, exist_ok=True)

    _print_hardware_banner(
        device, n_cores, compile_model, args.topology,
        total_episodes, batch_size, update_interval, cuda_scaled,
    )

    # ── Build graph + environment ─────────────────────────────────────────
    G   = TOPOLOGY_MAP[args.topology]()
    env = MM1KNetworkEnv(G, config)

    k_paths = config["network"]["k_paths"]
    agent   = PPOAgent(
        config,
        k_paths=k_paths,
        num_nodes=G.number_of_nodes(),
        device=device,
        compile_model=compile_model,
    )

    print(f"[Init] Nodes: {G.number_of_nodes()} | "
          f"Edges: {G.number_of_edges()} | "
          f"K-paths: {k_paths}")
    print(agent.hardware_summary())
    print()

    # ── Training state ────────────────────────────────────────────────────
    obs, info = env.reset()

    ep_idx              = 0
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            agent.load(args.checkpoint)
            # Try to parse the episode index from the filename (e.g., garro_nsfnet_ep4000.pt)
            import re
            match = re.search(r"_ep(\d+)\.pt$", args.checkpoint)
            if match:
                ep_idx = int(match.group(1))
                print(f"[Init] Resuming training from episode {ep_idx}")
            else:
                print(f"[Init] Resuming training with loaded checkpoint weights (starting from episode 0)")
        else:
            print(f"[Error] Checkpoint file '{args.checkpoint}' not found!")
            import sys
            sys.exit(1)

    step_count          = 0
    ep_reward           = 0.0
    episode_rewards:    list = []
    checkpoint_rewards: list = []
    t0                  = time.perf_counter()

    pbar = tqdm(total=total_episodes, initial=ep_idx, desc="Training", unit="ep",
                dynamic_ncols=True)

    # ── Main loop ─────────────────────────────────────────────────────────
    while ep_idx < total_episodes:

        # 1. Agent selects action using the live NetworkX graph state
        candidate_paths = env.candidate_paths
        action, log_prob, value = agent.select_action(env.G, candidate_paths)

        # Capture state snapshot before environment step modifies it
        state_snap = agent.buffer._snapshot(env.G)

        # 2. Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 3. Store transition — PyG Data object stored directly (no deepcopy)
        agent.buffer.add(
            state    = state_snap,
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
                elapsed  = time.perf_counter() - t0
                avg_r    = float(np.mean(checkpoint_rewards[-checkpoint_every:]))
                eps_per_s = ep_idx / elapsed
                pbar.write(
                    f"[Ep {ep_idx:6d}/{total_episodes}] "
                    f"Avg Reward: {avg_r:+.4f} | "
                    f"Steps: {step_count:,} | "
                    f"Speed: {eps_per_s:.1f} ep/s"
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

    elapsed = time.perf_counter() - t0
    print(f"\n[Train] Finished {total_episodes:,} episodes in "
          f"{elapsed/60:.1f} min  "
          f"({total_episodes/elapsed:.1f} ep/s average)")
    print(f"[Train] Final model saved → {final_path}")

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
            f"({G.number_of_nodes()} nodes) | "
            f"Device: {device}"
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
    parser.add_argument(
        "--no-compile",
        action="store_true",
        default=False,
        help="Disable torch.compile (useful for debugging/profiling)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to an existing checkpoint to resume training from",
    )
    parser.add_argument(
        "--resume",
        type=str,
        dest="checkpoint",
        default=None,
        help="Alias for --checkpoint to resume training from",
    )
    args = parser.parse_args()
    main(args)
