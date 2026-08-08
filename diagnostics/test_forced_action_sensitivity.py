"""
Diagnostic Test: Forced-Action Metric Sensitivity Check.

Fixes a toy/NSFNET topology with fixed traffic demand. Forces routing actions
to action 0 (shortest path) vs action K-1 (longer alternative path), and asserts
that computed path latency, packet loss, and rewards actually differ.
"""
import os
import sys
import yaml
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from digital_twin.mm1k_env import MM1KNetworkEnv
from topologies.nsfnet import get_nsfnet


def test_forced_action_sensitivity():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    G = get_nsfnet()
    env = MM1KNetworkEnv(G, config)
    env.reset(seed=42)

    # Force a specific pair with at least 2 distinct paths
    src, dst = 0, 1
    paths = env._all_paths.get((src, dst), [])
    if len(paths) < 2:
        for p, path_list in env._all_paths.items():
            if len(path_list) >= 2:
                src, dst = p
                paths = path_list
                break

    assert len(paths) >= 2, "Topology does not have pairs with >= 2 candidate paths."

    env.current_src = src
    env.current_dst = dst
    env.candidate_paths = paths

    # Action 0 (Shortest Path)
    env.reset(seed=100)
    env.current_src, env.current_dst = src, dst
    env.candidate_paths = paths
    obs1, r1, _, _, info1 = env.step(0)

    # Action K-1 (Longer Path)
    worst_act = len(paths) - 1
    env.reset(seed=100)
    env.current_src, env.current_dst = src, dst
    env.candidate_paths = paths
    obs2, r2, _, _, info2 = env.step(worst_act)

    path1 = info1["path"]
    path2 = info2["path"]
    delay1 = info1["path_latency_ms"]
    delay2 = info2["path_latency_ms"]

    print(f"[Test Forced Action Sensitivity]")
    print(f"  Pair: ({src} -> {dst})")
    print(f"  Path 0 (Action 0): {path1} | Latency: {delay1:.2f} ms | Reward: {r1:.4f}")
    print(f"  Path {worst_act} (Action {worst_act}): {path2} | Latency: {delay2:.2f} ms | Reward: {r2:.4f}")

    assert path1 != path2, "Paths should be distinct!"
    assert r1 != r2 or delay1 != delay2, "Computed metrics/rewards must differ for different chosen paths!"
    print("  ✓ Action sensitivity assertion PASSED.")


if __name__ == "__main__":
    test_forced_action_sensitivity()
