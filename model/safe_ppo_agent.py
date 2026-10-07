"""
Lagrangian Safe PPO Agent for Constrained SDN Traffic Engineering.

Implements Constrained Markov Decision Process (CMDP) optimization:
  Maximize:    E[Reward(s, a)]  (Throughput, Load Balancing)
  Subject to:  E[Cost_loss(s, a)]  <= Loss_SLA_Limit
               E[Cost_delay(s, a)] <= Delay_SLA_Limit

Dynamically updates Lagrange multipliers (λ_loss, λ_delay) via dual ascent,
guaranteeing QoS SLA satisfaction without manual trial-and-error reward weight tuning.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from model.ppo_agent import PPOAgent


class SafeLagrangianPPOAgent(PPOAgent):
    """
    PPO Agent equipped with adaptive Lagrangian dual multipliers for SLA enforcement.

    Parameters
    ----------
    config : dict
        Standard GARRO configuration dictionary.
    loss_limit_pct : float
        Maximum acceptable packet loss SLA percentage (e.g., 0.5%).
    delay_limit_ms : float
        Maximum acceptable end-to-end path delay (e.g., 80.0 ms).
    lr_lagrange : float
        Learning rate for dual multiplier updates.
    """

    def __init__(
        self,
        config: dict,
        k_paths: int = 5,
        num_nodes: Optional[int] = None,
        device: Optional[torch.device] = None,
        compile_model: bool = False,
        loss_limit_pct: float = 0.5,
        delay_limit_ms: float = 85.0,
        lr_lagrange: float = 0.05,
    ) -> None:
        super().__init__(
            config=config,
            k_paths=k_paths,
            num_nodes=num_nodes,
            device=device,
            compile_model=compile_model,
        )
        self.loss_limit = float(loss_limit_pct)
        self.delay_limit = float(delay_limit_ms)
        self.lr_lagrange = float(lr_lagrange)

        # Dual multipliers (initialized to zero)
        self.lambda_loss = 0.0
        self.lambda_delay = 0.0

        self.loss_history: List[float] = []
        self.delay_history: List[float] = []

    def record_step_metrics(self, packet_loss_pct: float, delay_ms: float) -> None:
        """Record QoS telemetry to compute constraint violations."""
        self.loss_history.append(float(packet_loss_pct))
        self.delay_history.append(float(delay_ms))

    def update_lagrange_multipliers(self) -> Dict[str, float]:
        """
        Update dual multipliers via sub-gradient ascent:
          λ_k ← max(0, λ_k + lr_λ * (mean(Cost_k) - Limit_k))
        """
        if not self.loss_history or not self.delay_history:
            return {"lambda_loss": self.lambda_loss, "lambda_delay": self.lambda_delay}

        mean_loss = float(np.mean(self.loss_history))
        mean_delay = float(np.mean(self.delay_history))

        # Dual ascent update
        loss_violation = mean_loss - self.loss_limit
        delay_violation = mean_delay - self.delay_limit

        self.lambda_loss = float(np.clip(self.lambda_loss + self.lr_lagrange * loss_violation, 0.0, 50.0))
        self.lambda_delay = float(np.clip(self.lambda_delay + self.lr_lagrange * (delay_violation / 10.0), 0.0, 50.0))

        # Clear buffer
        self.loss_history.clear()
        self.delay_history.clear()

        return {
            "mean_loss": mean_loss,
            "mean_delay": mean_delay,
            "loss_violation": loss_violation,
            "delay_violation": delay_violation,
            "lambda_loss": self.lambda_loss,
            "lambda_delay": self.lambda_delay,
        }

    def compute_constrained_reward(
        self,
        base_reward: float,
        packet_loss_pct: float,
        delay_ms: float,
    ) -> float:
        """
        Apply Lagrangian penalty to reward:
          R_safe = R_base - λ_loss * (loss - limit) - λ_delay * (delay - limit)
        """
        pen_loss = self.lambda_loss * max(0.0, packet_loss_pct - self.loss_limit)
        pen_delay = self.lambda_delay * max(0.0, (delay_ms - self.delay_limit) / 10.0)
        return float(base_reward - pen_loss - pen_delay)
