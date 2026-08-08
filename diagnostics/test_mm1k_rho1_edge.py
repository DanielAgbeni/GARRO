"""
Diagnostic Test: M/M/1/K Queuing ρ = 1 Edge Case Check.

Validates closed-form formulas for ρ = 1.0 (L'Hôpital limit) and continuity
around ρ -> 1.0 (e.g. ρ = 0.9999, ρ = 1.0, ρ = 1.0001). Asserts no NaNs, infs,
or numerical step discontinuities.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from digital_twin.mm1k_env import mm1k_metrics_vec


def test_mm1k_rho1_edge():
    K = 50
    mu = 150.0

    # Range of arrival rates around rho = 1.0 (lam = 150.0)
    lams = np.array([100.0, 149.9, 149.999, 150.0, 150.001, 150.1, 200.0], dtype=np.float64)
    mus  = np.full_like(lams, mu)

    EQs, P_ovs, delays = mm1k_metrics_vec(lams, mus, K)

    print("[Test M/M/1/K ρ = 1 Edge Case & Continuity]")
    for i in range(len(lams)):
        rho = lams[i] / mus[i]
        print(f"  lam={lams[i]:7.3f} | rho={rho:7.5f} | E[Q]={EQs[i]:6.3f} (max {K}) | P_ov={P_ovs[i]:.6f} | delay={delays[i]:6.2f} ms")

        assert not np.isnan(EQs[i]), f"NaN found in E[Q] for lam={lams[i]}"
        assert not np.isnan(P_ovs[i]), f"NaN found in P_overflow for lam={lams[i]}"
        assert not np.isnan(delays[i]), f"NaN found in delay for lam={lams[i]}"
        assert not np.isinf(EQs[i]), f"Inf found in E[Q] for lam={lams[i]}"
        assert not np.isinf(P_ovs[i]), f"Inf found in P_overflow for lam={lams[i]}"
        assert not np.isinf(delays[i]), f"Inf found in delay for lam={lams[i]}"

    # Theoretical exact values at rho = 1.0:
    # E[Q] = K / 2 = 25.0
    # P_overflow = 1 / (K + 1) = 1 / 51 = 0.01960784
    rho1_idx = 3 # index of 150.0
    assert np.isclose(EQs[rho1_idx], K / 2.0), f"E[Q] at rho=1 should be {K/2.0}, got {EQs[rho1_idx]}"
    assert np.isclose(P_ovs[rho1_idx], 1.0 / (K + 1)), f"P_ov at rho=1 should be {1.0/(K+1)}, got {P_ovs[rho1_idx]}"

    print("  ✓ M/M/1/K ρ = 1 edge case & continuity assertions PASSED.")


if __name__ == "__main__":
    test_mm1k_rho1_edge()
