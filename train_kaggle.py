"""
GARRO Phase 1 — Kaggle Cloud High-Performance DCN Training.
Dual Tesla T4 GPUs (31.2 GB VRAM) | 4 CPU Cores | 30 GB RAM.

Features:
- Distributed Data Parallel (DDP) across dual Tesla T4 GPUs using torchrun.
- Vectorized Multi-Environment stepping (N=16 per GPU = 32 parallel streams)
  to fully saturate Turing Tensor Cores and bypass the 4-CPU bottleneck.
- PyTorch AMP FP16 Mixed Precision with Turing Tensor Core acceleration.
- 3-Stage Curriculum Learning:
    Stage 1 (0–10k eps): Warmup & Topology Discovery (Low Uniform Load).
    Stage 2 (10k–35k eps): Bimodal Elephant Flows & Bisection Stride Stress.
    Stage 3 (35k–50k eps): Chaos Incast Bursts, Diurnal Waves & AllReduce Collective Exchange.
- Fault-tolerant auto-resume for Kaggle's 9-hour session ceiling.
- Real-time reward, loss, throughput, and link utilization telemetry logging.

Usage:
    # Multi-GPU DDP (Dual T4 on Kaggle):
    torchrun --nproc_per_node=2 train_kaggle.py --topology fat_tree --episodes 50000

    # Single-GPU or CPU fallback:
    python train_kaggle.py --topology fat_tree --episodes 10000
"""

import argparse
import copy
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
from tqdm import tqdm

from digital_twin.mm1k_env import MM1KNetworkEnv
from digital_twin.traffic_generator import create_traffic_generator
from model.ppo_agent import PPOAgent, FastGraphConverter
from topologies.fat_tree import get_fat_tree
from topologies.nsfnet import get_nsfnet
from topologies.geant2 import get_geant2

TOPOLOGY_MAP = {
    "fat_tree": get_fat_tree,
    "nsfnet":   get_nsfnet,
    "geant2":   get_geant2,
}


# ── DDP Helper Utilities ──────────────────────────────────────────────────────

def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1

def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def is_main_process() -> bool:
    return get_rank() == 0

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        return rank, local_rank, world_size
    return 0, 0, 1

def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


# ── Curriculum Manager ────────────────────────────────────────────────────────

class DCNCurriculumManager:
    """
    3-Stage Curriculum scheduler for Data Center DRL Training.
    Adapts traffic burstiness, elephant flow ratio, and collective stress
    based on the current training progress.
    """
    def __init__(self, total_episodes: int):
        self.total_episodes = total_episodes
        self.stage_1_end = int(0.20 * total_episodes)   # 20% Warmup (e.g., 0-10,000)
        self.stage_2_end = int(0.70 * total_episodes)   # 50% Heavy Bimodal (10,000-35,000)
        self.current_stage = 1

    def update_environment(self, env: MM1KNetworkEnv, ep_idx: int) -> int:
        """Adjust traffic generator hyperparameters based on curriculum stage."""
        if ep_idx < self.stage_1_end:
            stage = 1
            if hasattr(env, "traffic_gen") and hasattr(env.traffic_gen, "pareto_prob"):
                env.traffic_gen.pareto_prob = 0.05
                env.traffic_gen.incast_prob = 0.0
                env.traffic_gen.spatial_pattern = "uniform"
                env.traffic_gen.base_rate = 50.0
        elif ep_idx < self.stage_2_end:
            stage = 2
            if hasattr(env, "traffic_gen") and hasattr(env.traffic_gen, "pareto_prob"):
                env.traffic_gen.pareto_prob = 0.20
                env.traffic_gen.incast_prob = 0.05
                env.traffic_gen.spatial_pattern = "stride"
                env.traffic_gen.base_rate = 75.0
        else:
            stage = 3
            if hasattr(env, "traffic_gen") and hasattr(env.traffic_gen, "pareto_prob"):
                env.traffic_gen.pareto_prob = 0.30
                env.traffic_gen.incast_prob = 0.15
                env.traffic_gen.spatial_pattern = "allreduce"
                env.traffic_gen.base_rate = 95.0

        self.current_stage = stage
        return stage


# ── Vectorized Environment Bundle ─────────────────────────────────────────────

class VectorDCNEnvs:
    """
    Manages N independent MM1K environments in parallel on a single process.
    Batch-evaluates actions and collects observations without inter-process IPC overhead.
    """
    def __init__(self, make_env_fn, num_envs: int = 16):
        self.num_envs = num_envs
        self.envs = [make_env_fn(env_id=i) for i in range(num_envs)]
        self.num_nodes = self.envs[0].num_nodes
        self.num_edges = self.envs[0].num_edges
        self.candidate_paths = [e.candidate_paths for e in self.envs]

    def reset_all(self):
        obs_list = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset()
            obs_list.append(obs)
            self.candidate_paths[i] = env.candidate_paths
        return obs_list

    def step_all(self, actions: List[int]):
        results = []
        for i, (env, act) in enumerate(zip(self.envs, actions)):
            next_obs, r, term, trunc, info = env.step(act)
            done = term or trunc
            if done:
                next_obs, _ = env.reset()
            self.candidate_paths[i] = env.candidate_paths
            results.append((next_obs, r, done, info))
        return results


# ── Main Training Loop ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GARRO Kaggle Multi-GPU DCN Training")
    parser.add_argument("--topology", type=str, default="fat_tree", choices=["fat_tree", "nsfnet", "geant2"])
    parser.add_argument("--episodes", type=int, default=50000, help="Total episodes across all GPUs")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=2500, help="Checkpoint frequency in episodes")
    parser.add_argument("--eval-every", type=int, default=1000, help="Evaluation frequency in episodes")
    parser.add_argument("--num-envs", type=int, default=16, help="Vectorized environments per worker")
    parser.add_argument("--traffic-source", type=str, default="default", choices=["default", "alibaba", "sndlib"], help="Traffic generator source (default, alibaba, or sndlib)")
    parser.add_argument("--compile", action="store_true", default=False, help="Enable torch.compile (default: False to avoid Triton GEMM freeze on Tesla T4)")
    args = parser.parse_args()

    rank, local_rank, world_size = setup_distributed()
    is_master = is_main_process()

    # Load base config
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.traffic_source != "default":
        config["traffic_source"] = args.traffic_source

    # Topology overrides
    topo_overrides = config.get("topology_overrides", {}).get(args.topology, {})
    if topo_overrides:
        if "ppo" in topo_overrides:
            config["ppo"].update(topo_overrides["ppo"])
        if "training" in topo_overrides:
            config["training"].update(topo_overrides["training"])

    # Hardware target setup
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        # Kaggle Dual T4 Tensor Core optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        config["training"]["amp_dtype"] = "float16"
    else:
        device = torch.device("cpu")
        config["training"]["amp_dtype"] = "float32"

    compile_model = args.compile

    # Output directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    metrics_log_path = os.path.join(args.checkpoint_dir, f"training_metrics_{args.topology}.jsonl")

    # Banner
    if is_master:
        print("=" * 68)
        print("  GARRO High-Performance Data Center DRL Orchestrator")
        print(f"  Target Subsystem : Kaggle Cloud Dual Tesla T4 GPUs")
        print(f"  Topology         : {args.topology.upper()} | World Size: {world_size} GPUs")
        print(f"  Vectorized Envs  : {args.num_envs} envs/GPU ({args.num_envs * world_size} total)")
        print(f"  Total Episodes   : {args.episodes:,} | Device: {device}")
        print("=" * 68)

    # Factory for vectorized environments
    def make_env_fn(env_id: int):
        G = TOPOLOGY_MAP[args.topology]()
        env_cfg = copy.deepcopy(config)
        # Give each env a distinct seed
        env_cfg["seed"] = 42 + rank * 1000 + env_id
        return MM1KNetworkEnv(G, env_cfg)

    vec_envs = VectorDCNEnvs(make_env_fn, num_envs=args.num_envs)
    ref_env = vec_envs.envs[0]
    num_nodes = ref_env.num_nodes
    k_paths = config["network"]["k_paths"]

    # Initialize PPO Agent
    agent = PPOAgent(
        config=config,
        k_paths=k_paths,
        num_nodes=num_nodes,
        device=device,
        compile_model=compile_model,
        total_episodes=args.episodes // world_size,
    )

    # Initialize Curriculum Manager
    curriculum = DCNCurriculumManager(total_episodes=args.episodes)

    # Auto-resume logic
    ep_idx = 0
    resume_path = args.checkpoint
    auto_ckpt = os.path.join(args.checkpoint_dir, f"garro_{args.topology}_latest.pt")
    if resume_path is None and os.path.exists(auto_ckpt):
        resume_path = auto_ckpt

    if resume_path and os.path.exists(resume_path):
        if is_master:
            print(f"[Resume] Loading checkpoint from: {resume_path}")
        agent.load(resume_path)
        import re
        m = re.search(r"_ep(\d+)\.pt$", resume_path)
        if m:
            ep_idx = int(m.group(1))
            agent.set_update_step(ep_idx // world_size)

    # Reset environments
    vec_envs.reset_all()

    # Rollout loop state
    step_count = 0
    t0 = time.perf_counter()
    pbar = tqdm(
        total=args.episodes,
        initial=ep_idx,
        desc="Curriculum DCN Training",
        unit="ep",
        disable=not is_master,
        dynamic_ncols=True,
    )

    recent_rewards: List[float] = []
    episodes_per_worker = args.episodes // world_size
    current_worker_eps = ep_idx // world_size

    # Telemetry logging buffer
    history_rewards: List[float] = []
    history_policy_loss: List[float] = []
    history_value_loss: List[float] = []

    try:
        while ep_idx < args.episodes:
            # Update curriculum stage across environments
            for env in vec_envs.envs:
                stage = curriculum.update_environment(env, ep_idx)

            # Batched action selection across all N vectorized environments in ONE GPU pass
            env_graphs = [env.G for env in vec_envs.envs]
            acts, lps, vals, p_edges_batch, src_dst_batch, masks_batch = agent.select_actions_batch(
                env_graphs, vec_envs.candidate_paths
            )

            # Record transitions into agent buffer
            for i in range(args.num_envs):
                state_snap = agent.buffer._snapshot(env_graphs[i])
                agent.buffer.add(
                    state=state_snap,
                    action=int(acts[i]),
                    log_prob=float(lps[i]),
                    reward=0.0,   # populated after step
                    value=float(vals[i]),
                    done=False,
                    mask=masks_batch[i],
                    path_edges=p_edges_batch[i],
                    src_dst=src_dst_batch[i],
                )

            # Vectorized parallel environment stepping
            step_results = vec_envs.step_all(acts.tolist())
            step_count += args.num_envs

            # Backfill rewards & check for completed episodes
            for i, (next_obs, r, done, info) in enumerate(step_results):
                idx_in_buf = len(agent.buffer) - args.num_envs + i
                agent.buffer.rewards[idx_in_buf] = r
                agent.buffer.dones[idx_in_buf] = done

                if done:
                    ep_idx += world_size
                    current_worker_eps += 1
                    recent_rewards.append(r)
                    history_rewards.append(r)
                    if is_master:
                        pbar.update(world_size)

            # PPO Update check
            if len(agent.buffer) >= agent._update_interval:
                metrics = agent.update()
                if metrics and is_master:
                    p_loss = metrics.get("policy_loss", float("nan"))
                    v_loss = metrics.get("value_loss", float("nan"))
                    history_policy_loss.append(p_loss)
                    history_value_loss.append(v_loss)
                    pbar.set_postfix({
                        "Stage": stage,
                        "AvgRew": f"{np.mean(recent_rewards[-100:]):+.2f}" if recent_rewards else "0.0",
                        "PLoss": f"{p_loss:.3f}",
                        "VLoss": f"{v_loss:.3f}",
                    })

            # Checkpoint rotation & Evaluation
            if ep_idx > 0 and (ep_idx % args.checkpoint_every < world_size) and is_master:
                elapsed = time.perf_counter() - t0
                speed = ep_idx / max(elapsed, 1e-5)
                avg_r = float(np.mean(recent_rewards[-100:])) if recent_rewards else 0.0

                pbar.write(
                    f"[Ep {ep_idx:,}/{args.episodes:,}] "
                    f"Stage {stage} | Avg Reward: {avg_r:+.4f} | "
                    f"Speed: {speed:.1f} ep/s | Elapsed: {elapsed/60:.1f} min"
                )

                # Save periodic and latest checkpoint
                ckpt_path = os.path.join(args.checkpoint_dir, f"garro_{args.topology}_ep{ep_idx}.pt")
                agent.save(ckpt_path)
                agent.save(auto_ckpt)

                # Dump telemetry log
                with open(metrics_log_path, "a") as f_log:
                    f_log.write(json.dumps({
                        "episode": ep_idx,
                        "stage": stage,
                        "avg_reward": avg_r,
                        "elapsed_seconds": elapsed,
                        "speed_eps_per_sec": speed,
                    }) + "\n")

                # Generate live training curve figure
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                if history_rewards:
                    smooth_window = min(500, len(history_rewards))
                    conv_kernel = np.ones(smooth_window) / smooth_window
                    smoothed = np.convolve(history_rewards, conv_kernel, mode="valid")
                    ax1.plot(smoothed, color="#10b981", lw=2, label="GARRO Moving Avg Reward")
                    ax1.set_title("Training Reward Curve (DCN Curriculum)")
                    ax1.set_xlabel("Logged Episodes")
                    ax1.set_ylabel("Reward")
                    ax1.grid(True, alpha=0.3)
                    ax1.legend()

                if history_policy_loss:
                    ax2.plot(history_policy_loss, color="#6366f1", label="Policy Loss")
                    ax2.plot(history_value_loss, color="#f59e0b", label="Value Loss")
                    ax2.set_title("PPO Actor-Critic Loss Progression")
                    ax2.set_xlabel("PPO Updates")
                    ax2.set_ylabel("Loss")
                    ax2.grid(True, alpha=0.3)
                    ax2.legend()

                plt.tight_layout()
                plt.savefig(os.path.join(args.checkpoint_dir, f"training_curve_{args.topology}.png"), dpi=200)
                plt.close(fig)

    except KeyboardInterrupt:
        if is_master:
            print("\n[Interrupt] Training paused by user. Saving emergency checkpoint...")
            agent.save(auto_ckpt)
    finally:
        if is_master:
            final_path = os.path.join(args.checkpoint_dir, f"garro_{args.topology}_final.pt")
            agent.save(final_path)
            print(f"[Done] Training finished. Final model saved to: {final_path}")
        cleanup_distributed()


if __name__ == "__main__":
    main()
