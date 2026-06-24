"""
GARRO Agentic AI Layer — LLM Intent-Based Networking Orchestrator.

Translates high-level, natural-language operator intents into numerical
PPO reward-function weights (α1..α4), enabling intent-based networking
without re-training the DRL agent.

Supported LLM backends
----------------------
* Google Gemini (gemini-1.5-flash / gemini-pro)   — set provider: "gemini" in config
* OpenAI GPT-4 / GPT-3.5                          — set provider: "openai" in config

Fault tolerance
---------------
* On any API error the orchestrator returns the last-known-good weights.
* If no weights have been set yet, returns the default balanced weights:
      α1=0.4 (throughput), α2=0.3 (delay), α3=0.2 (loss), α4=0.1 (balance)

Usage
-----
    import asyncio
    from agentic.llm_orchestrator import LLMOrchestrator

    orch = LLMOrchestrator(config)
    weights = asyncio.run(orch.parse_intent(
        "Prioritise video conferencing — latency is critical."
    ))
    print(weights.as_dict())

    # Synchronous convenience wrapper:
    weights = orch.parse_intent_sync("Minimise packet loss for live trading.")
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
from dotenv import load_dotenv

load_dotenv()


# ── Reward Weights Dataclass ──────────────────────────────────────────────────

@dataclass
class RewardWeights:
    """
    Holds the four GARRO reward-function tuning coefficients.

    Attributes
    ----------
    alpha1 : float  Throughput ratio weight              (maximise)
    alpha2 : float  End-to-end path delay weight         (minimise)
    alpha3 : float  Packet loss ratio weight             (minimise)
    alpha4 : float  Link utilisation variance weight     (minimise)
    """
    alpha1: float = 0.4
    alpha2: float = 0.3
    alpha3: float = 0.2
    alpha4: float = 0.1

    def as_dict(self) -> dict:
        return {
            "alpha1": round(self.alpha1, 4),
            "alpha2": round(self.alpha2, 4),
            "alpha3": round(self.alpha3, 4),
            "alpha4": round(self.alpha4, 4),
        }

    def normalized(self) -> "RewardWeights":
        """Return a copy with weights re-normalised to sum exactly to 1.0."""
        total = self.alpha1 + self.alpha2 + self.alpha3 + self.alpha4
        if total < 1e-9:
            return RewardWeights()   # Fall back to defaults
        return RewardWeights(
            alpha1=self.alpha1 / total,
            alpha2=self.alpha2 / total,
            alpha3=self.alpha3 / total,
            alpha4=self.alpha4 / total,
        )

    def apply_to_config(self, config: dict) -> None:
        """Update reward_weights section of a config dict in place."""
        rw = config.setdefault("reward_weights", {})
        rw["alpha1"] = self.alpha1
        rw["alpha2"] = self.alpha2
        rw["alpha3"] = self.alpha3
        rw["alpha4"] = self.alpha4


# ── System Prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a network policy translator for an SDN intelligent routing system (GARRO).

Given a natural-language operator intent, return ONLY a valid JSON object with
four float values between 0 and 1 that must sum to exactly 1.0:

{
  "alpha1": <throughput weight>,
  "alpha2": <latency/delay weight>,
  "alpha3": <packet_loss weight>,
  "alpha4": <link_utilisation_balance weight>
}

Interpretation guidelines:
- Real-time / latency-sensitive traffic (video conferencing, VoIP, gaming, live trading):
    → high alpha2 (delay), moderate alpha3 (packet loss)
- Bulk data / file transfers / backups:
    → high alpha1 (throughput), low alpha2
- Heavily congested network / load-balancing priority:
    → high alpha4 (balance link utilisation)
- Mixed / default balanced traffic:
    → alpha1=0.4, alpha2=0.3, alpha3=0.2, alpha4=0.1

Return ONLY the JSON object. No explanation, no markdown fences, no preamble."""


# ── Orchestrator ──────────────────────────────────────────────────────────────

class LLMOrchestrator:
    """
    Translates operator natural-language intents into PPO reward weights.

    Parameters
    ----------
    config : dict  Full config.yaml contents (reads config["agentic"]).
    """

    def __init__(self, config: dict):
        ag_cfg = config["agentic"]
        self.provider    = ag_cfg["provider"]
        self.model       = ag_cfg["model"]
        self.temperature = ag_cfg["temperature"]
        self._current_weights = RewardWeights()   # Start with defaults

    # ── Public API ────────────────────────────────────────────────────────────

    async def parse_intent(self, operator_intent: str) -> RewardWeights:
        """
        Async: Parse a natural-language intent → RewardWeights.

        Falls back to last-known-good weights on any error.
        """
        try:
            if self.provider == "gemini":
                weights = await self._call_gemini(operator_intent)
            elif self.provider == "openai":
                weights = await self._call_openai(operator_intent)
            else:
                print(f"[LLM] Unknown provider '{self.provider}'. Using defaults.")
                return self._current_weights

            self._current_weights = weights.normalized()
            print(f"[LLM] Intent parsed → {self._current_weights.as_dict()}")
            return self._current_weights

        except Exception as exc:
            print(f"[LLM] API call failed ({exc}). Using previous weights.")
            return self._current_weights

    def parse_intent_sync(self, operator_intent: str) -> RewardWeights:
        """Synchronous convenience wrapper around parse_intent."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run, self.parse_intent(operator_intent)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(self.parse_intent(operator_intent))
        except Exception as exc:
            print(f"[LLM] Sync wrapper failed ({exc}). Using defaults.")
            return self._current_weights

    def get_fallback_weights(self) -> RewardWeights:
        """Return the last-known-good weights (deterministic fallback)."""
        return self._current_weights

    # ── LLM Backends ─────────────────────────────────────────────────────────

    async def _call_gemini(self, intent: str) -> RewardWeights:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment. "
                "Add it to the .env file in the project root."
            )

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={api_key}"
        )
        payload = {
            "contents": [{
                "parts": [
                    {"text": _SYSTEM_PROMPT},
                    {"text": f"\nOperator intent: {intent}"},
                ]
            }],
            "generationConfig": {"temperature": self.temperature},
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                resp.raise_for_status()
                data = await resp.json()

        raw = data["candidates"][0]["content"]["parts"][0]["text"]
        return self._parse_json(raw)

    async def _call_openai(self, intent: str) -> RewardWeights:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment. "
                "Add it to the .env file in the project root."
            )

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":       self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": f"Operator intent: {intent}"},
            ],
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload,
                                    timeout=aiohttp.ClientTimeout(total=15)) as resp:
                resp.raise_for_status()
                data = await resp.json()

        raw = data["choices"][0]["message"]["content"]
        return self._parse_json(raw)

    # ── JSON Parsing ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(raw: str) -> RewardWeights:
        """
        Extract and parse the JSON weight object from an LLM response.
        Strips markdown code fences if present.
        """
        cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
        cleaned = cleaned.strip("`").strip()
        obj = json.loads(cleaned)
        return RewardWeights(
            alpha1=float(obj.get("alpha1", 0.4)),
            alpha2=float(obj.get("alpha2", 0.3)),
            alpha3=float(obj.get("alpha3", 0.2)),
            alpha4=float(obj.get("alpha4", 0.1)),
        )


# ── CLI Demo ──────────────────────────────────────────────────────────────────

async def _demo():
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    orch = LLMOrchestrator(config)

    test_intents = [
        "Prioritise video conferencing traffic. Latency is critical.",
        "We need maximum throughput for overnight bulk data replication.",
        "The network is congested — balance the load across all links.",
    ]

    for intent in test_intents:
        print(f"\nIntent : {intent}")
        weights = await orch.parse_intent(intent)
        print(f"Weights: {weights.as_dict()}")


if __name__ == "__main__":
    asyncio.run(_demo())
