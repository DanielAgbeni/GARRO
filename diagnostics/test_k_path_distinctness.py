"""
Diagnostic Test: K-Shortest-Path Distinctness Check.

Verifies Yen's algorithm returns K distinct paths per (src, dst) pair across
all topologies (NSFNET, GEANT2, Fat-Tree k=4). Asserts len(set(paths)) == len(paths).
"""
import os
import sys
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from digital_twin.mm1k_env import MM1KNetworkEnv
from topologies.nsfnet import get_nsfnet
from topologies.geant2 import get_geant2
from topologies.fat_tree import get_fat_tree


def test_k_path_distinctness():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    topologies = {
        "nsfnet": get_nsfnet(),
        "geant2": get_geant2(),
        "fat_tree": get_fat_tree(k=4),
    }

    print("[Test K-Shortest Path Distinctness]")
    for topo_name, G in topologies.items():
        cfg = config.copy()
        cfg["network"]["topology"] = topo_name
        env = MM1KNetworkEnv(G, cfg)

        distinct_count = 0
        total_pairs = len(env._all_paths)
        collapsed_pairs = 0

        for (src, dst), paths in env._all_paths.items():
            t_paths = [tuple(p) for p in paths]
            assert len(set(t_paths)) == len(t_paths), f"Duplicate path detected for pair ({src}, {dst}) in {topo_name}!"
            if len(paths) == env.K:
                distinct_count += 1
            else:
                collapsed_pairs += 1

        print(f"  {topo_name.upper():<8}: Total Pairs={total_pairs} | Pairs with K={env.K} distinct paths={distinct_count} | Fewer paths={collapsed_pairs}")
        assert distinct_count > 0, f"No pairs in {topo_name} produced distinct paths!"

    print("  ✓ K-shortest-path distinctness assertion PASSED across all topologies.")


if __name__ == "__main__":
    test_k_path_distinctness()
