"""
GARRO Agentic AI Layer — Universal LLM Intent-Based Networking Orchestrator.

Translates high-level, natural-language operator intents into numerical
PPO reward-function weights (α1..α4), enabling intent-based networking
without re-training the DRL agent.

Supported LLM Providers
------------------------
Set `provider` in config.yaml [agentic] section:

  provider   | model example             | env var            | free?
  -----------|---------------------------|--------------------|--------
  gemini     | gemini-1.5-flash          | GEMINI_API_KEY     | ✓ Yes
  openai     | gpt-4o-mini               | OPENAI_API_KEY     | ✗ Paid
  groq       | llama3-70b-8192           | GROQ_API_KEY       | ✓ Yes
  mistral    | mistral-small-latest      | MISTRAL_API_KEY    | ✓ Trial
  cohere     | command-r                 | COHERE_API_KEY     | ✓ Trial
  together   | meta-llama/Llama-3-70b    | TOGETHER_API_KEY   | ✓ $5 credit
  ollama     | llama3                    | (none — local)     | ✓ Free

Adding a new provider
---------------------
1.  Write an  `async def _call_<name>(self, intent)` method.
2.  Register it in `_PROVIDER_REGISTRY` at the bottom of the class.
That's it.

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

    # See all registered providers:
    print(LLMOrchestrator.list_providers())
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

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
            return RewardWeights()
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

    Uses a provider-registry pattern: each backend is an async method
    registered in `_PROVIDER_REGISTRY`.  Adding a new provider requires
    only writing one method and adding one dict entry.

    Parameters
    ----------
    config : dict  Full config.yaml contents (reads config[\"agentic\"]).
    """

    def __init__(self, config: dict):
        ag_cfg           = config["agentic"]
        self.provider    = ag_cfg["provider"].lower().strip()
        self.model       = ag_cfg["model"]
        self.temperature = float(ag_cfg.get("temperature", 0.1))
        self.ollama_url  = ag_cfg.get(
            "ollama_base_url", "http://localhost:11434"
        ).rstrip("/")
        self._current_weights = RewardWeights()

        # Validate provider at init time — gives a clear error immediately
        if self.provider not in self._PROVIDER_REGISTRY:
            available = ", ".join(sorted(self._PROVIDER_REGISTRY.keys()))
            raise ValueError(
                f"[LLM] Unknown provider '{self.provider}'. "
                f"Available providers: {available}"
            )

    # ── Public API ────────────────────────────────────────────────────────────

    async def parse_intent(self, operator_intent: str) -> RewardWeights:
        """
        Async: Parse a natural-language intent → RewardWeights.
        Falls back to last-known-good weights on any error.
        """
        try:
            handler = self._PROVIDER_REGISTRY[self.provider]
            weights = await handler(self, operator_intent)
            self._current_weights = weights.normalized()
            print(f"[LLM:{self.provider}] Intent parsed → {self._current_weights.as_dict()}")
            return self._current_weights

        except Exception as exc:
            print(f"[LLM:{self.provider}] API call failed — {exc}")
            print(f"[LLM] Falling back to previous weights: "
                  f"{self._current_weights.as_dict()}")
            return self._current_weights

    def parse_intent_sync(self, operator_intent: str) -> RewardWeights:
        """Synchronous convenience wrapper around parse_intent."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, self.parse_intent(operator_intent))
                    return future.result(timeout=30)
            else:
                return loop.run_until_complete(self.parse_intent(operator_intent))
        except Exception as exc:
            print(f"[LLM] Sync wrapper failed ({exc}). Using defaults.")
            return self._current_weights

    def get_fallback_weights(self) -> RewardWeights:
        """Return the last-known-good weights (deterministic fallback)."""
        return self._current_weights

    @classmethod
    def list_providers(cls) -> List[str]:
        """Return a sorted list of all registered provider names."""
        return sorted(cls._PROVIDER_REGISTRY.keys())

    # ── Provider Backends ─────────────────────────────────────────────────────
    # Each async method accepts (self, intent: str) → RewardWeights.
    # To add a new provider: write the method, register it below.

    async def _call_gemini(self, intent: str) -> RewardWeights:
        """
        Google Gemini — REST API.
        Free tier: https://aistudio.google.com/app/apikey
        Default model: gemini-1.5-flash
        """
        api_key = self._require_env("GEMINI_API_KEY",
            "Get a free key at: https://aistudio.google.com/app/apikey")

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={api_key}"
        )
        payload = {
            "contents": [{"parts": [
                {"text": _SYSTEM_PROMPT},
                {"text": f"\nOperator intent: {intent}"},
            ]}],
            "generationConfig": {"temperature": self.temperature},
        }
        async with aiohttp.ClientSession() as s:
            async with s.post(url, json=payload,
                              timeout=aiohttp.ClientTimeout(total=20)) as r:
                r.raise_for_status()
                data = await r.json()
        raw = data["candidates"][0]["content"]["parts"][0]["text"]
        return self._parse_json(raw)

    async def _call_openai(self, intent: str) -> RewardWeights:
        """
        OpenAI GPT — REST API (paid).
        Keys: https://platform.openai.com/api-keys
        Default model: gpt-4o-mini
        """
        return await self._call_openai_compatible(
            intent,
            base_url   = "https://api.openai.com/v1",
            env_var    = "OPENAI_API_KEY",
            key_url    = "https://platform.openai.com/api-keys",
        )

    async def _call_groq(self, intent: str) -> RewardWeights:
        """
        Groq — OpenAI-compatible API, extremely fast LPU inference.
        Free tier: https://console.groq.com
        Recommended free model: llama3-70b-8192
        Other free models: mixtral-8x7b-32768, gemma2-9b-it
        """
        return await self._call_openai_compatible(
            intent,
            base_url   = "https://api.groq.com/openai/v1",
            env_var    = "GROQ_API_KEY",
            key_url    = "https://console.groq.com",
        )

    async def _call_mistral(self, intent: str) -> RewardWeights:
        """
        Mistral AI — OpenAI-compatible API.
        Free trial: https://console.mistral.ai
        Recommended model: mistral-small-latest  or  open-mistral-7b
        """
        return await self._call_openai_compatible(
            intent,
            base_url   = "https://api.mistral.ai/v1",
            env_var    = "MISTRAL_API_KEY",
            key_url    = "https://console.mistral.ai",
        )

    async def _call_together(self, intent: str) -> RewardWeights:
        """
        Together AI — OpenAI-compatible API, 100+ open-source models.
        $5 free credit on signup: https://api.together.xyz
        Recommended model: meta-llama/Llama-3-70b-chat-hf
        """
        return await self._call_openai_compatible(
            intent,
            base_url   = "https://api.together.xyz/v1",
            env_var    = "TOGETHER_API_KEY",
            key_url    = "https://api.together.xyz",
        )

    async def _call_cohere(self, intent: str) -> RewardWeights:
        """
        Cohere — Chat API.
        Free trial: https://dashboard.cohere.com
        Recommended model: command-r
        """
        api_key = self._require_env("COHERE_API_KEY",
            "Get a free trial key at: https://dashboard.cohere.com")

        url = "https://api.cohere.com/v2/chat"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": f"Operator intent: {intent}"},
            ],
            "temperature": self.temperature,
        }
        async with aiohttp.ClientSession() as s:
            async with s.post(url, headers=headers, json=payload,
                              timeout=aiohttp.ClientTimeout(total=20)) as r:
                r.raise_for_status()
                data = await r.json()

        # Cohere v2 response format
        raw = data["message"]["content"][0]["text"]
        return self._parse_json(raw)

    async def _call_ollama(self, intent: str) -> RewardWeights:
        """
        Ollama — 100% local inference, no API key, no cost.

        Install:  curl -fsSL https://ollama.com/install.sh | sh
        Pull model: ollama pull llama3
        Set in config.yaml:
            provider: ollama
            model: llama3          # or mistral, phi3, gemma2, etc.
        """
        url = f"{self.ollama_url}/api/chat"
        payload = {
            "model":  self.model,
            "stream": False,
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": f"Operator intent: {intent}"},
            ],
        }
        async with aiohttp.ClientSession() as s:
            async with s.post(url, json=payload,
                              timeout=aiohttp.ClientTimeout(total=60)) as r:
                if r.status == 404:
                    raise ConnectionError(
                        f"Ollama model '{self.model}' not found. "
                        f"Run: ollama pull {self.model}"
                    )
                r.raise_for_status()
                data = await r.json()
        raw = data["message"]["content"]
        return self._parse_json(raw)

    # ── Shared OpenAI-Compatible Handler ─────────────────────────────────────

    async def _call_openai_compatible(
        self,
        intent:   str,
        base_url: str,
        env_var:  str,
        key_url:  str,
    ) -> RewardWeights:
        """
        Generic handler for any OpenAI-compatible chat completions API.
        Used by: openai, groq, mistral, together.
        """
        api_key = self._require_env(env_var,
            f"Get your API key at: {key_url}")

        url = f"{base_url}/chat/completions"
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
        async with aiohttp.ClientSession() as s:
            async with s.post(url, headers=headers, json=payload,
                              timeout=aiohttp.ClientTimeout(total=20)) as r:
                r.raise_for_status()
                data = await r.json()
        raw = data["choices"][0]["message"]["content"]
        return self._parse_json(raw)

    # ── Utility Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _require_env(var: str, hint: str = "") -> str:
        """
        Get an environment variable or raise a clear, actionable error.
        """
        val = os.getenv(var, "").strip()
        if not val:
            msg = (
                f"Environment variable '{var}' is not set.\n"
                f"  → Add it to your .env file:  {var}=your_key_here\n"
            )
            if hint:
                msg += f"  → {hint}\n"
            raise EnvironmentError(msg)
        return val

    @staticmethod
    def _parse_json(raw: str) -> RewardWeights:
        """
        Extract and parse the JSON weight object from an LLM response.
        Handles markdown fences, extra whitespace, and partial JSON.
        """
        # Strip markdown code fences
        cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
        cleaned = cleaned.strip("`").strip()

        # Find the first valid JSON object in the response
        match = re.search(r"\{[^{}]+\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

        obj = json.loads(cleaned)
        return RewardWeights(
            alpha1=float(obj.get("alpha1", 0.4)),
            alpha2=float(obj.get("alpha2", 0.3)),
            alpha3=float(obj.get("alpha3", 0.2)),
            alpha4=float(obj.get("alpha4", 0.1)),
        )

    # ── Provider Registry ─────────────────────────────────────────────────────
    # Maps provider name (config.yaml) → async method.
    # To add a new provider: write the method above, add entry here.

    _PROVIDER_REGISTRY: Dict[str, Callable] = {
        "gemini":   _call_gemini,
        "openai":   _call_openai,
        "groq":     _call_groq,
        "mistral":  _call_mistral,
        "cohere":   _call_cohere,
        "together": _call_together,
        "ollama":   _call_ollama,
    }


# ── CLI Demo ──────────────────────────────────────────────────────────────────

async def _demo():
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    print(f"\n[Demo] Available providers: {LLMOrchestrator.list_providers()}")
    print(f"[Demo] Using provider: {config['agentic']['provider']} | "
          f"model: {config['agentic']['model']}\n")

    orch = LLMOrchestrator(config)

    test_intents = [
        "Prioritise video conferencing traffic. Latency is critical.",
        "We need maximum throughput for overnight bulk data replication.",
        "The network is congested — balance the load across all links.",
    ]

    for intent in test_intents:
        print(f"Intent : {intent}")
        weights = await orch.parse_intent(intent)
        print(f"Weights: {weights.as_dict()}\n")


if __name__ == "__main__":
    asyncio.run(_demo())
