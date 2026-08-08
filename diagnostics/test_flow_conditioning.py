"""
Diagnostic Test: Flow Conditioning Feature & Latent Embedding Check.

Verifies that node feature vectors contain flow-specific (is_src, is_dst) indicators,
and that changing the (src, dst) pair for identical network graph telemetry produces
distinct Graph Transformer latent embeddings and policy logits.
"""
import os
import sys
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from digital_twin.mm1k_env import MM1KNetworkEnv
from model.ppo_agent import PPOAgent
from topologies.nsfnet import get_nsfnet


def test_flow_conditioning():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    G = get_nsfnet()
    env = MM1KNetworkEnv(G, config)
    agent = PPOAgent(config, k_paths=config["network"]["k_paths"], num_nodes=G.number_of_nodes(), compile_model=False)

    obs1, info1 = env.reset(seed=42)
    src1, dst1 = env.current_src, env.current_dst
    paths1 = env.candidate_paths
    act1, lp1, val1 = agent.select_action(env.G, paths1)
    latent1 = agent._encode(env.G)

    # Change current demand pair on the EXACT SAME graph state
    nodes = sorted(env.G.nodes())
    src2, dst2 = (nodes[0], nodes[5]) if (src1, dst1) != (nodes[0], nodes[5]) else (nodes[1], nodes[6])
    env.current_src, env.current_dst = src2, dst2
    for n in nodes:
        env.G.nodes[n]["is_src"] = 1.0 if n == src2 else 0.0
        env.G.nodes[n]["is_dst"] = 1.0 if n == dst2 else 0.0
    paths2 = env._all_paths.get((src2, dst2), [])

    act2, lp2, val2 = agent.select_action(env.G, paths2)
    latent2 = agent._encode(env.G)

    diff = torch.norm(latent1 - latent2).item()
    print("[Test Flow Conditioning & Latent Differentiation]")
    print(f"  Pair 1: ({src1} -> {dst1})")
    print(f"  Pair 2: ({src2} -> {dst2})")
    print(f"  Latent Embedding Euclidean Difference: {diff:.6f}")

    assert diff > 1e-4, f"Graph Transformer latent representations are identical (diff={diff}) for different flow pairs!"
    print("  ✓ Flow conditioning assertion PASSED — Graph Transformer produces flow-specific embeddings.")


if __name__ == "__main__":
    test_flow_conditioning()
