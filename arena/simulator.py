"""
GARRO Arena — Live Multi-Policy Digital-Twin Simulator (Fat-Tree DCN).

Runs 4 parallel MM1KNetworkEnv instances (GARRO, Hedera, ECMP, OSPF)
under identical traffic patterns, time-of-day dynamics, and random seeds.

Metrics match the benchmark evaluation environment:
- GARRO: PPO GraphTransformer with Path-Attention (ep20000)
- Hedera: Congestion-aware bottleneck avoidance (NSDI baseline)
- ECMP: Equal-Cost Multi-Path hashing
- OSPF: Shortest Path First (delay/hop optimal)
"""
from __future__ import annotations

import copy
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from digital_twin.mm1k_env import MM1KNetworkEnv
from topologies.fat_tree import get_fat_tree

POLICIES: Tuple[str, ...] = ("GARRO", "Hedera", "ECMP", "OSPF")


class ArenaSimulator:
    """Synchronized 4-protocol Fat-Tree digital twin."""

    def __init__(self, config: dict, agent=None, seed: int = 42, k: int = 4):
        self.lock = threading.RLock()
        self.cfg = copy.deepcopy(config)
        self.cfg["network"]["topology"] = "fat_tree"
        self.agent = agent
        self.k = k
        self.seed = seed

        # Graph reference
        self.G0 = get_fat_tree(k)
        self.nodes = sorted(self.G0.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.edges = list(self.G0.edges())
        self.n_edges = len(self.edges)
        self.edge_idx = {}
        for i, (u, v) in enumerate(self.edges):
            self.edge_idx[(u, v)] = i
            self.edge_idx[(v, u)] = i

        # Controls & knobs
        self.load_factor = 1.0
        self.minutes_per_tick = 5
        self.failed_links: set[Tuple[int, int]] = set()
        self._active_event: Optional[dict] = None

        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None, start_hour: Optional[float] = 8.0) -> None:
        with self.lock:
            if seed is not None:
                self.seed = int(seed)
            self.rng = np.random.default_rng(self.seed)

            # Build 4 separate envs
            self.envs: Dict[str, MM1KNetworkEnv] = {
                name: MM1KNetworkEnv(self.G0, self.cfg)
                for name in POLICIES
            }

            for name, env in self.envs.items():
                env.reset(seed=self.seed)
                if hasattr(env.traffic_gen, "_step_count") and start_hour is not None:
                    env.traffic_gen._step_count = int(start_hour * env.traffic_gen.steps_per_hour)

            self.tick_no = 0
            self.infer_ms: deque = deque(maxlen=500)
            self.decision_log: deque = deque(maxlen=20)
            self.last_decision: Optional[dict] = None
            self.last_frame: Optional[dict] = None
            self._decisions_wall: deque = deque(maxlen=50)

            # Cumulative stats
            self.cum = {n: {"lat": 0.0, "loss": 0.0, "tput": 0.0, "rew": 0.0, "n": 0} for n in POLICIES}
            # Hourly bins
            self.hourly = {n: {"lat": np.zeros(24), "loss": np.zeros(24), "cnt": np.zeros(24)} for n in POLICIES}

    def topology(self) -> dict:
        k = self.k
        n_core = (k // 2) ** 2
        n_agg = k * (k // 2)
        nodes = []
        for n in self.nodes:
            if n < n_core:
                tier, idx = "core", n
            elif n < n_core + n_agg:
                tier, idx = "agg", n - n_core
            else:
                tier, idx = "edge", n - n_core - n_agg
            nodes.append({
                "id": n, "dpid": n + 1, "tier": tier, "tier_idx": idx,
                "label": f"s{n + 1}",
            })

        gen = self.envs["GARRO"].traffic_gen
        edge_nodes = getattr(gen, "edge_nodes", list(range(n_core + n_agg, len(self.nodes))))
        hosts = []
        h = 1
        for e in edge_nodes:
            for _ in range(k // 2):
                hosts.append({"id": f"h{h}", "ip": f"10.0.0.{h}", "edge": e})
                h += 1

        return {
            "k": k,
            "nodes": nodes,
            "edges": [[u, v] for u, v in self.edges],
            "hosts": hosts,
            "policies": list(POLICIES),
            "seed": self.seed,
        }

    # ── Controls ──────────────────────────────────────────────────────────

    def set_minutes_per_tick(self, m: int) -> None:
        with self.lock:
            self.minutes_per_tick = int(np.clip(m, 1, 60))

    def set_pattern(self, pattern: str) -> None:
        with self.lock:
            for env in self.envs.values():
                if hasattr(env.traffic_gen, "pattern"):
                    env.traffic_gen.pattern = pattern

    def set_arrivals(self, n: int) -> None:
        pass

    def set_load(self, f: float) -> None:
        with self.lock:
            self.load_factor = float(np.clip(f, 0.1, 2.5))
            for env in self.envs.values():
                if hasattr(env, "traffic_gen"):
                    env.traffic_gen.base_rate = (env.base_lam * self.n_edges / (env.num_nodes * 2.59)) * self.load_factor

    def set_elephant_prob(self, p: float) -> None:
        with self.lock:
            for env in self.envs.values():
                if hasattr(env.traffic_gen, "elephant_prob"):
                    env.traffic_gen.elephant_prob = float(np.clip(p, 0.0, 0.9))

    def set_hour(self, hour: float) -> None:
        with self.lock:
            h = float(hour) % 24.0
            for env in self.envs.values():
                if hasattr(env.traffic_gen, "_step_count"):
                    env.traffic_gen._step_count = int(h * env.traffic_gen.steps_per_hour)

    def trigger_event(self, name: str, ticks: int = 30) -> None:
        with self.lock:
            self._active_event = {"name": name, "ticks_left": int(ticks)}
            for env in self.envs.values():
                tg = env.traffic_gen
                if hasattr(tg, "pattern"):
                    if name in ("incast", "allreduce", "stride"):
                        tg.pattern = name
                    elif name == "elephant_storm":
                        tg.elephant_prob = 0.65

    def _end_event(self) -> None:
        self._active_event = None
        for env in self.envs.values():
            tg = env.traffic_gen
            if hasattr(tg, "pattern"):
                tg.pattern = "mixed"
                tg.elephant_prob = 0.15

    def toggle_link(self, u: int, v: int) -> bool:
        with self.lock:
            pair = (min(u, v), max(u, v))
            if pair in self.failed_links:
                self.failed_links.discard(pair)
                is_failed = False
            else:
                self.failed_links.add(pair)
                is_failed = True

            # Update link attributes in all envs
            for env in self.envs.values():
                if env.G.has_edge(u, v):
                    if is_failed:
                        env.G.edges[u, v]["bandwidth"] = 1.0
                        env.G.edges[u, v]["delay"] = 500.0
                    else:
                        env.G.edges[u, v]["bandwidth"] = 10000.0
                        env.G.edges[u, v]["delay"] = 1.0
            return is_failed

    def restore_all(self) -> None:
        with self.lock:
            for u, v in list(self.failed_links):
                self.toggle_link(u, v)

    # ── Simulation Tick ───────────────────────────────────────────────────

    def tick(self) -> dict:
        with self.lock:
            t0 = time.perf_counter()
            self.tick_no += 1

            if self._active_event:
                self._active_event["ticks_left"] -= 1
                if self._active_event["ticks_left"] <= 0:
                    self._end_event()

            garro_env = self.envs["GARRO"]
            src, dst = garro_env.current_src, garro_env.current_dst
            cands = garro_env.candidate_paths

            # Prune candidate paths crossing failed links
            if self.failed_links:
                valid_cands = []
                for p in cands:
                    edges_p = [(min(u, v), max(u, v)) for u, v in zip(p[:-1], p[1:])]
                    if not any(e in self.failed_links for e in edges_p):
                        valid_cands.append(p)
                if not valid_cands:
                    valid_cands = cands
                cands = valid_cands

            # Synchronize demand pair and candidate paths across all envs
            for env in self.envs.values():
                env.current_src, env.current_dst = src, dst
                env.candidate_paths = cands

            picks = {}
            ex_data = None

            # Policy selections
            # 1. OSPF
            picks["OSPF"] = 0

            # 2. ECMP
            if cands:
                costs = [len(p) - 1 for p in cands]
                min_c = min(costs)
                eq = [i for i, c in enumerate(costs) if c == min_c]
                # flow 5-tuple style hash
                h = (self.tick_no * 2654435761 + src * 40503 + dst * 9973) & 0xFFFFFFFF
                picks["ECMP"] = eq[h % len(eq)]
            else:
                picks["ECMP"] = 0

            # 3. Hedera (Al-Fares et al., NSDI)
            if cands:
                costs = [len(p) - 1 for p in cands]
                min_c = min(costs)
                eq = [i for i, c in enumerate(costs) if c == min_c]
                best_idx = eq[0]
                min_max_u = float("inf")
                h_env = self.envs["Hedera"]
                for idx in eq:
                    p = cands[idx]
                    p_edges = list(zip(p[:-1], p[1:]))
                    max_u = max((h_env.G.edges.get((u, v), {}).get("utilization", 0.0) for u, v in p_edges), default=0.0)
                    if max_u < min_max_u:
                        min_max_u = max_u
                        best_idx = idx
                picks["Hedera"] = best_idx
            else:
                picks["Hedera"] = 0

            # 4. GARRO AI
            t_infer = time.perf_counter()
            if self.agent and cands:
                ex_data = self.agent.explain_action(garro_env.G, cands)
                picks["GARRO"] = ex_data["action"]
            else:
                picks["GARRO"] = 0
            self.infer_ms.append((time.perf_counter() - t_infer) * 1000.0)

            # Step all 4 environments
            pol_metrics = {}
            for name, env in self.envs.items():
                action = picks[name]
                obs, r, _, _, info = env.step(action)
                raw = info.get("raw_reward_terms", {})
                lat = float(info.get("path_latency_ms", raw.get("D_path", 0.0)))
                loss = float(raw.get("total_loss", 0.0) * 100.0)
                tput = float(raw.get("tput_ratio", 1.0) * 100.0)
                u_var = float(raw.get("util_variance", 0.0))

                utils = [float(env.G.edges[u, v].get("utilization", 0.0)) for u, v in self.edges]
                max_u = max(utils) if utils else 0.0

                pol_metrics[name] = {
                    "latency": round(lat, 2),
                    "loss": round(loss, 3),
                    "tput": round(tput, 2),
                    "reward": round(r, 2),
                    "max_util": round(max_u, 3),
                    "util_var": round(u_var, 4),
                    "util": [round(u, 3) for u in utils],
                }

                # Update cumulative stats
                c = self.cum[name]
                c["lat"] += lat; c["loss"] += loss; c["tput"] += tput; c["rew"] += r; c["n"] += 1

                # Update hourly stats
                hour_int = int(garro_env.traffic_gen.current_hour) % 24
                hb = self.hourly[name]
                hb["lat"][hour_int] += lat
                hb["loss"][hour_int] += loss
                hb["cnt"][hour_int] += 1

            # Build Decision Explainer data
            cand_rows = []
            if ex_data and cands:
                for idx, path in enumerate(cands):
                    p_edges = list(zip(path[:-1], path[1:]))
                    max_u = max((garro_env.G.edges.get((u, v), {}).get("utilization", 0.0) for u, v in p_edges), default=0.0)
                    cand_rows.append({
                        "idx": idx,
                        "path": path,
                        "dpids": [n + 1 for n in path],
                        "hops": len(path) - 1,
                        "prob": round(ex_data["probs"][idx], 4) if idx < len(ex_data["probs"]) else 0.0,
                        "bottleneck": round(max_u, 3),
                    })

            # Node labels
            src_lbl = f"s{src + 1}"
            dst_lbl = f"s{dst + 1}"
            decision_obj = {
                "tick": self.tick_no,
                "src": src, "dst": dst,
                "flow": f"{src_lbl} → {dst_lbl}",
                "picks": picks,
                "candidates": cand_rows,
                "value": round(ex_data["value"], 2) if ex_data else 0.0,
            }
            self.last_decision = decision_obj
            self.decision_log.appendleft({
                "tick": self.tick_no,
                "flow": decision_obj["flow"],
                "picks": picks,
                "conf": round(max(ex_data["probs"]) if ex_data and ex_data.get("probs") else 1.0, 3),
            })

            # Cumulative average summary
            cum_summary = {}
            for name in POLICIES:
                c = self.cum[name]
                n = max(1, c["n"])
                cum_summary[name] = {
                    "latency": round(c["lat"] / n, 2),
                    "loss": round(c["loss"] / n, 3),
                    "tput": round(c["tput"] / n, 2),
                    "reward": round(c["rew"] / n, 2),
                    "ticks": c["n"],
                }

            # Hourly curve
            hourly_summary = {}
            for name in POLICIES:
                hb = self.hourly[name]
                cnt = np.maximum(hb["cnt"], 1)
                hourly_summary[name] = {
                    "lat": [round(float(x), 2) if hb["cnt"][i] else None for i, x in enumerate(hb["lat"] / cnt)],
                    "loss": [round(float(x), 3) if hb["cnt"][i] else None for i, x in enumerate(hb["loss"] / cnt)],
                }

            current_hour = float(garro_env.traffic_gen.current_hour)
            mins = int(round(current_hour * 60)) % (24 * 60)
            now = time.perf_counter()
            self._decisions_wall.append((now, self.tick_no))
            if len(self._decisions_wall) >= 2:
                (tw0, dw0), (tw1, dw1) = self._decisions_wall[0], self._decisions_wall[-1]
                dps = (dw1 - dw0) / max(tw1 - tw0, 1e-6)
            else:
                dps = 0.0

            inf_arr = np.array(self.infer_ms) if self.infer_ms else np.array([0.0])

            frame = {
                "type": "tick",
                "tick": self.tick_no,
                "hour": round(current_hour, 2),
                "clock": f"{mins // 60:02d}:{mins % 60:02d}",
                "pattern": getattr(garro_env.traffic_gen, "active_pattern", "mixed"),
                "event": dict(self._active_event) if self._active_event else None,
                "failed": [[u, v] for u, v in self.failed_links],
                "offered": round(float(getattr(garro_env.traffic_gen, "_prev_T", np.array([0.0])).sum()), 2),
                "policies": pol_metrics,
                "cumulative": cum_summary,
                "hourly": hourly_summary,
                "decision": decision_obj,
                "log": list(self.decision_log),
                "perf": {
                    "infer_p50": round(float(np.percentile(inf_arr, 50)), 2),
                    "infer_p99": round(float(np.percentile(inf_arr, 99)), 2),
                    "dps": round(dps, 1),
                    "tick_ms": round((now - t0) * 1000.0, 1),
                },
            }
            self.last_frame = frame
            return frame
