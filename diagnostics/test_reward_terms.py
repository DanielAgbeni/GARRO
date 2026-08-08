"""
Diagnostic Test: Reward Term Magnitude Dump & Balance Check.

Logs four raw unweighted reward terms (throughput ratio, path delay, packet loss,
link utilization variance) across a batch of transitions and checks for extreme magnitude mismatches.
"""
import os
import sys
import yaml
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from digital_twin.mm1k_env import MM1KNetworkEnv
from topologies.nsfnet import get_nsfnet


def test_reward_terms():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    G = get_nsfnet()
    env = MM1KNetworkEnv(G, config)
    obs, info = env.reset(seed=42)

    terms_history = []
    for _ in range(50):
        act = env.action_space.sample()
        obs, r, term, trunc, info = env.step(act)
        raw_terms = info.get("raw_reward_terms", {})
        if raw_terms:
            terms_history.append(raw_terms)

    assert len(terms_history) > 0, "No raw reward terms logged in info dict!"

    tput_ratios = [t["tput_ratio"] for t in terms_history]
    delays_ms   = [t["D_path"] for t in terms_history]
    delays_norm = [t["delay_norm"] for t in terms_history]
    losses      = [t["total_loss"] for t in terms_history]
    vars_util   = [t["util_variance"] for t in terms_history]

    print("[Test Reward Term Balance & Magnitude Dump]")
    print(f"  tput_ratio    : mean={np.mean(tput_ratios):.4f}, min={np.min(tput_ratios):.4f}, max={np.max(tput_ratios):.4f}")
    print(f"  D_path (ms)   : mean={np.mean(delays_ms):.2f} ms, min={np.min(delays_ms):.2f} ms, max={np.max(delays_ms):.2f} ms")
    print(f"  delay_norm    : mean={np.mean(delays_norm):.4f}, min={np.min(delays_norm):.4f}, max={np.max(delays_norm):.4f}")
    print(f"  total_loss    : mean={np.mean(losses):.4f}, min={np.min(losses):.4f}, max={np.max(losses):.4f}")
    print(f"  util_variance : mean={np.mean(vars_util):.4f}, min={np.min(vars_util):.4f}, max={np.max(vars_util):.4f}")

    # Check normalized delay is in [0, 1]
    assert all(0.0 <= d <= 1.0 for d in delays_norm), "Normalized delay is out of [0, 1] range!"
    assert all(0.0 <= l <= 1.0 for l in losses), "Packet loss ratio is out of [0, 1] range!"
    assert all(0.0 <= t <= 1.0 for t in tput_ratios), "Throughput ratio is out of [0, 1] range!"

    print("  ✓ Reward term magnitude assertion PASSED.")


if __name__ == "__main__":
    test_reward_terms()
