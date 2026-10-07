"""
Verification Suite for GARRO DCN Upgrades:
  1. SNDlib Real-Trace Loader
  2. Alibaba Cloud DCN Cluster Trace Loader
  3. Two-Tier Elephant Flow Scheduler
  4. Lagrangian Safe PPO Agent
  5. Calibrated Baseline Engine (Equal-Cost ECMP & Hedera)
"""
import sys
import unittest
import numpy as np
import networkx as nx
import yaml

from topologies.fat_tree import get_fat_tree
from digital_twin.real_trace_loader import SNDlibTraceLoader, AlibabaDCNTraceLoader
from digital_twin.mm1k_env import MM1KNetworkEnv
from model.elephant_flow_scheduler import TwoTierDCNScheduler
from model.safe_ppo_agent import SafeLagrangianPPOAgent
from evaluate import _run_baseline_worker


class TestDCNEnhancements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("config.yaml", encoding="utf-8") as f:
            cls.config = yaml.safe_load(f)
        cls.G_fat = get_fat_tree(k=4)

    def test_sndlib_trace_loader(self):
        loader = SNDlibTraceLoader(self.G_fat, base_rate=150.0, seed=42)
        T1 = loader.generate()
        self.assertEqual(T1.shape, (20, 20))
        self.assertEqual(np.diag(T1).sum(), 0.0)
        self.assertTrue(np.all(T1 >= 0.0))
        self.assertTrue(T1.sum() > 0.0)

    def test_alibaba_dcn_trace_loader(self):
        loader = AlibabaDCNTraceLoader(self.G_fat, k=4, base_rate=200.0, seed=42)
        T1 = loader.generate()
        self.assertEqual(T1.shape, (20, 20))
        self.assertEqual(np.diag(T1).sum(), 0.0)
        self.assertTrue(np.all(T1 >= 0.0))
        self.assertTrue(T1.sum() > 0.0)

        # Generate multiple steps to verify AllReduce and Incast dynamics
        matrices = [loader.generate() for _ in range(50)]
        self.assertEqual(len(matrices), 50)

    def test_two_tier_elephant_scheduler(self):
        scheduler = TwoTierDCNScheduler(self.G_fat, agent=None, elephant_threshold_mbps=1000.0)
        candidate_paths = [
            [12, 4, 13],        # 2 hops (equal cost)
            [12, 5, 13],        # 2 hops (equal cost)
            [12, 4, 0, 6, 14],  # 4 hops (detour for intra-pod)
        ]

        # Mice flow (< 1000 Mbps)
        path_mice, tier_mice, idx_mice = scheduler.schedule_flow(
            src=12, dst=13, demand_mbps=250.0, candidate_paths=candidate_paths, step=0
        )
        self.assertEqual(tier_mice, "tier1_ecmp")
        self.assertIn(idx_mice, [0, 1])  # Must only pick equal-cost path 0 or 1, NEVER detour 2

        stats = scheduler.get_stats()
        self.assertEqual(stats["total_flows"], 1.0)
        self.assertEqual(stats["mice_percentage"], 100.0)

    def test_safe_lagrangian_agent_multipliers(self):
        agent = SafeLagrangianPPOAgent(
            self.config,
            k_paths=5,
            num_nodes=20,
            loss_limit_pct=0.5,
            delay_limit_ms=75.0,
            lr_lagrange=0.1,
        )
        # Record SLA violation
        agent.record_step_metrics(packet_loss_pct=1.5, delay_ms=90.0)
        update_info = agent.update_lagrange_multipliers()
        self.assertGreater(update_info["lambda_loss"], 0.0)
        self.assertGreater(update_info["lambda_delay"], 0.0)

        # Verify constrained reward penalization
        base_r = 500.0
        penalized_r = agent.compute_constrained_reward(base_r, packet_loss_pct=1.5, delay_ms=90.0)
        self.assertLess(penalized_r, base_r)

    def test_calibrated_baselines_execution(self):
        # Quick 2-episode run on fat_tree for all baselines
        for algo in ["OSPF", "ECMP", "Hedera", "ECMP-Detour", "Random"]:
            arg_tuple = (algo, "fat_tree", self.config, 2, 42)
            res = _run_baseline_worker(arg_tuple)
            self.assertEqual(res["name"], algo)
            self.assertTrue("mean_reward" in res)
            self.assertTrue("latency_ms" in res)
            self.assertTrue(np.isfinite(res["mean_reward"]))


if __name__ == "__main__":
    unittest.main()
