"""Thin async LLM client: OpenRouter (or any OpenAI-compatible endpoint), disk cache, spend ledger.

Every call is cached in SQLite keyed by the endpoint, the full request and a `sample` index, so
re-running an experiment with the same seeds is free and bit-identical, while distinct
episodes/seeds still get independent samples. `sample` is also sent as the provider `seed`.

Spend is tracked in a ledger table shared by every process that uses the same cache file, so
`SWARM_BUDGET_USD` is a global cap (on top of the provider-side key limit). In-flight calls
reserve a worst-case cost before they are sent, so concurrent calls cannot overshoot the cap.

Reasoning ("thinking") tokens are billed and some providers ignore `max_tokens` for them (one
probe produced 12k reasoning tokens for a 600-token cap), so reasoning is disabled by default and
every response's reasoning-token count is logged.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import sqlite3
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "results" / "cache" / "llm.sqlite"
PRICES_PATH = ROOT / "results" / "cache" / "openrouter_prices.json"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
ADAPTION_URL = "https://api.prod.adaptionlabs.ai/api/v1/responses"
# Adaption hosted models, credits per token (input, output), from
# docs.adaptionlabs.ai/inference-stacks/models (read 2026-11-06). Use as "adaption/<id>".
# 1 credit = $1. qwen3.5-397b always reasons first: give it max_tokens >= 1500.
ADAPTION_PRICES = {
    "gemma-4-31b-it": (0.09e-6, 0.34e-6),
    "glm-5.2": (1.40e-6, 4.40e-6),
    "qwen3.5-397b": (0.55e-6, 3.50e-6),
    "gemma-4-26b-a4b-it": (0.0765e-6, 0.255e-6),
    "qwen3.8-27b": (0.039e-6, 1.25e-6),
    "deepseek-v4-flash": (0.04444e-6, 0.13332e-6),
}
CACHE_VERSION = "v2"

# Models whose endpoint refuses reasoning.enabled=false: use the lowest effort instead.
MANDATORY_REASONING = {"openai/gpt-oss-20b": "low", "openai/gpt-oss-120b": "low"}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class ChatResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    cached: bool
    reasoning_tokens: int = 0
    provider: str | None = None


def _get_json(url: str, headers: dict | None = None, timeout: float = 15) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _load_prices() -> dict[str, tuple[float, float]]:
    """USD per token (prompt, completion) for every OpenRouter model, cached on disk."""
    if PRICES_PATH.exists() and time.time() - PRICES_PATH.stat().st_mtime < 86_400:
        raw = json.loads(PRICES_PATH.read_text())
    else:
        try:
            raw = _get_json(f"{OPENROUTER_URL}/models")["data"]
            PRICES_PATH.parent.mkdir(parents=True, exist_ok=True)
            PRICES_PATH.write_text(json.dumps(raw))
        except Exception:  # noqa: BLE001 (network failure: fall back)
            return {}
    out = {}
    for m in raw:
        p = m.get("pricing") or {}
        try:
            out[m["id"]] = (float(p.get("prompt", 0)), float(p.get("completion", 0)))
        except (TypeError, ValueError):
            continue
    return out


class _Store:
    """SQLite cache of responses plus a spend ledger, shared across processes."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False, timeout=60)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, value TEXT)")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS spend (ts REAL, key TEXT, model TEXT, provider TEXT, "
            "cost REAL, prompt_tokens INT, completion_tokens INT, reasoning_tokens INT, "
            "latency REAL)"
        )
        self._db.commit()

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM calls WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict, ledger: tuple) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO calls (key, value) VALUES (?, ?)", (key, json.dumps(value))
            )
            self._db.execute("INSERT INTO spend VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", ledger)
            self._db.commit()

    def total_spent(self) -> float:
        with self._lock:
            row = self._db.execute("SELECT COALESCE(SUM(cost), 0) FROM spend").fetchone()
        return float(row[0])


class LLM:
    def __init__(
        self,
        budget_usd: float | None = None,
        max_concurrency: int | None = None,
        cache_path: Path = CACHE_PATH,
        provider_pins: dict[str, list[str]] | None = None,
    ):
        base_url = os.getenv("SWARM_BASE_URL") or OPENROUTER_URL
        api_key = os.getenv("SWARM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        self._base_url = base_url
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key or "missing",
                                   max_retries=0, timeout=120)
        self._is_openrouter = base_url == OPENROUTER_URL
        self._store = _Store(cache_path)
        conc = max_concurrency or int(os.getenv("SWARM_MAX_CONCURRENCY", "48"))
        self._sem = asyncio.Semaphore(conc)
        # Global cap shared by all processes using this cache (the key also has a provider cap).
        self.budget_usd = budget_usd or float(os.getenv("SWARM_BUDGET_USD", "45"))
        self.provider_pins = provider_pins or {}
        self._prices = _load_prices() if self._is_openrouter else {}
        self._reserved = 0.0
        self._spent_cache = (0.0, 0.0)  # (timestamp, value)
        self._key_usage_cache: tuple[float, float | None] = (0.0, None)
        self.spent_usd = 0.0  # this process only
        self.n_calls = 0
        self.n_cached = 0
        self.n_reasoning_leaks = 0  # responses with reasoning tokens although reasoning was off
        self._oai_clients: dict[str, AsyncOpenAI] = {}  # "oai:<base_url>|<model>" backends

    # ---- budget -------------------------------------------------------------------------
    def _worst_case(self, model: str, messages: list[dict], max_tokens: int) -> float:
        p_in, p_out = self._prices.get(model, (2e-6, 1e-5))
        n_in = sum(len(m.get("content", "")) for m in messages) / 3.0
        return n_in * p_in + max_tokens * p_out

    def key_usage(self) -> float | None:
        """Spend reported by OpenRouter for this API key (the source of truth), cached 30 s."""
        if not self._is_openrouter:
            return None
        ts, val = self._key_usage_cache
        if time.time() - ts > 30.0:
            try:
                data = _get_json(f"{OPENROUTER_URL}/key",
                                 {"Authorization": f"Bearer {self._client.api_key}"})
                val = float(data["data"]["usage"])
            except Exception:  # noqa: BLE001, S110 (network failure)
                pass  # keep the last known value; the ledger still applies
            self._key_usage_cache = (time.time(), val)
        return val

    def total_spent(self) -> float:
        ts, val = self._spent_cache
        if time.time() - ts > 2.0:
            val = self._store.total_spent()
            self._spent_cache = (time.time(), val)
        return max(val, self.key_usage() or 0.0)

    def _cost_from_tokens(self, model: str, prompt: int, completion: int) -> float:
        p_in, p_out = self._prices.get(model, (2e-6, 1e-5))
        return prompt * p_in + completion * p_out

    # ---- call ---------------------------------------------------------------------------
    async def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 1.0,
        max_tokens: int = 300,
        sample: int = 0,
        json_mode: bool = False,
        reasoning: str | None = None,  # None/"off" = disabled; "low"/"medium"/"high" = effort
        retries: int = 8,
    ) -> ChatResult:
        if model.startswith("adaption/"):
            return await self._chat_adaption(model, messages, temperature=temperature,
                                             max_tokens=max_tokens, sample=sample,
                                             retries=retries)
        if model.startswith("oai:"):
            return await self._chat_oai(model, messages, temperature=temperature,
                                        max_tokens=max_tokens, sample=sample,
                                        json_mode=json_mode, retries=retries)
        request: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": int(sample) % 2_147_483_647,
        }
        extra: dict = {}
        if json_mode:
            request["response_format"] = {"type": "json_object"}
        if self._is_openrouter:
            extra["usage"] = {"include": True}
            effort = MANDATORY_REASONING.get(model, reasoning)
            extra["reasoning"] = (
                {"enabled": False} if effort in (None, "off") else {"effort": effort}
            )
            if model in self.provider_pins:
                extra["provider"] = {"order": self.provider_pins[model], "allow_fallbacks": False}
        key = hashlib.sha256(
            json.dumps({**request, **extra, "sample": sample, "endpoint": self._base_url,
                        "v": CACHE_VERSION}, sort_keys=True).encode()
        ).hexdigest()

        if (hit := self._store.get(key)) is not None:
            self.n_cached += 1
            return ChatResult(**{**hit, "cached": True})

        reserve = self._worst_case(model, messages, max_tokens)
        if self.total_spent() + self._reserved + reserve > self.budget_usd:
            raise BudgetExceeded(
                f"global spend ${self.total_spent():.2f} + in-flight ${self._reserved:.2f} "
                f"would exceed ${self.budget_usd:.2f}"
            )
        self._reserved += reserve
        try:
            resp, latency = await self._send(request, extra, retries)
        finally:
            self._reserved -= reserve

        usage = resp.usage
        prompt_t = usage.prompt_tokens if usage else 0
        compl_t = usage.completion_tokens if usage else 0
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        reason_t = int(getattr(details, "reasoning_tokens", 0) or 0)
        cost = float(getattr(usage, "cost", 0.0) or 0.0) if usage else 0.0
        if cost == 0.0 and (prompt_t or compl_t):
            cost = self._cost_from_tokens(model, prompt_t, compl_t)
        provider = getattr(resp, "provider", None)
        if reason_t and extra.get("reasoning", {}).get("enabled") is False:
            self.n_reasoning_leaks += 1
        result = {
            "text": resp.choices[0].message.content or "",
            "model": resp.model or model,
            "prompt_tokens": prompt_t,
            "completion_tokens": compl_t,
            "cost_usd": cost,
            "reasoning_tokens": reason_t,
            "provider": provider,
        }
        self._store.put(key, result, (time.time(), key, model, provider, cost, prompt_t, compl_t,
                                      reason_t, latency))
        self._spent_cache = (0.0, 0.0)
        self.spent_usd += cost
        self.n_calls += 1
        return ChatResult(**result, cached=False)

    async def _send(self, request: dict, extra: dict, retries: int,
                    client: AsyncOpenAI | None = None):
        client = client or self._client
        delay = 1.0
        attempt, rate_hits = 0, 0
        while True:
            try:
                async with self._sem:
                    t0 = time.time()
                    resp = await client.chat.completions.create(**request, extra_body=extra)
                    latency = time.time() - t0
                if not resp.choices:
                    raise RuntimeError(f"empty choices: {getattr(resp, 'error', None)}")
                return resp, latency
            except openai.BadRequestError:
                raise  # malformed request or unsupported parameter: retrying will not help
            except (openai.AuthenticationError, openai.PermissionDeniedError):
                raise
            except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError):
                # upstream 429s and network blips can last minutes: up to 30 waits of at most
                # 60 s each, not counted against the ordinary retry budget
                rate_hits += 1
                if rate_hits > 30:
                    raise
                await asyncio.sleep(min(60.0, 2.0 * 2 ** min(rate_hits, 5)) + random.random() * 3)
            except Exception:  # transient errors and timeouts: back off
                attempt += 1
                if attempt >= retries:
                    raise
                await asyncio.sleep(delay + random.random())
                delay = min(delay * 2, 30.0)

    # ---- Adaption Responses API (hackathon credits) -----------------------------------------
    async def _chat_adaption(self, model: str, messages: list[dict], *, temperature: float,
                             max_tokens: int, sample: int, retries: int) -> ChatResult:
        """Same cache, ledger and budget as chat(), sent to Adaption's /api/v1/responses."""
        import httpx

        mid = model.removeprefix("adaption/")
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [{"role": m["role"], "content": m["content"]}
                 for m in messages if m["role"] != "system"]
        body: dict = {"model": mid, "input": turns, "temperature": temperature,
                      "max_output_tokens": max_tokens, "store": False}
        if system:
            body["instructions"] = system
        key = hashlib.sha256(json.dumps({**body, "sample": sample, "endpoint": ADAPTION_URL,
                                         "v": CACHE_VERSION}, sort_keys=True).encode()).hexdigest()
        if (hit := self._store.get(key)) is not None:
            self.n_cached += 1
            return ChatResult(**{**hit, "cached": True})
        p_in, p_out = ADAPTION_PRICES.get(mid, (2e-6, 1e-5))
        reserve = sum(len(m["content"]) for m in messages) / 3.0 * p_in + max_tokens * p_out
        if self.total_spent() + self._reserved + reserve > self.budget_usd:
            raise BudgetExceeded(f"spend ${self.total_spent():.2f} would exceed "
                                 f"${self.budget_usd:.2f}")
        api_key = os.getenv("ADAPTION_API_KEY")
        if not api_key:
            raise RuntimeError("ADAPTION_API_KEY is not set (.env)")
        self._reserved += reserve
        delay, attempt = 1.0, 0
        try:
            while True:
                try:
                    async with self._sem:
                        t0 = time.time()
                        async with httpx.AsyncClient(timeout=180) as cli:
                            r = await cli.post(ADAPTION_URL, json=body,
                                               headers={"Authorization": f"Bearer {api_key}"})
                        latency = time.time() - t0
                    if r.status_code in (429, 500, 502, 503, 504):
                        raise RuntimeError(f"adaption {r.status_code}: {r.text[:200]}")
                    if r.status_code >= 400:
                        raise ValueError(f"adaption {r.status_code}: {r.text[:500]}")
                    data = r.json()
                    if data.get("status") == "failed":
                        raise RuntimeError(f"adaption failed: {data.get('error')}")
                    break
                except ValueError:
                    raise
                except Exception:  # noqa: BLE001 (transient: back off)
                    attempt += 1
                    if attempt >= retries:
                        raise
                    await asyncio.sleep(delay + random.random())
                    delay = min(delay * 2, 30.0)
        finally:
            self._reserved -= reserve
        text = data.get("output_text")
        if text is None:  # assemble from output items if the convenience field is absent
            text = "".join(c.get("text", "") for it in data.get("output", [])
                           for c in (it.get("content") or []) if c.get("type") == "output_text")
        u = data.get("usage") or {}
        pt, ct = int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0))
        rt = int(((u.get("output_tokens_details") or {}).get("reasoning_tokens")) or 0)
        cost = float(u["cost"]) if u.get("cost") is not None else pt * p_in + ct * p_out
        if data.get("status") == "incomplete" and not text:
            raise RuntimeError(f"adaption incomplete with empty text ({rt} reasoning tokens): "
                               f"raise max_tokens for {mid}")
        result = {"text": text or "", "model": model, "prompt_tokens": pt,
                  "completion_tokens": ct, "cost_usd": cost, "reasoning_tokens": rt,
                  "provider": "adaption"}
        self._store.put(key, result, (time.time(), key, model, "adaption", cost, pt, ct, rt,
                                      latency))
        self._spent_cache = (0.0, 0.0)
        self.spent_usd += cost
        self.n_calls += 1
        return ChatResult(**result, cached=False)

    # ---- OpenAI-compatible endpoint named in the model string ---------------------------------
    def _oai_client(self, base_url: str) -> AsyncOpenAI:
        cli = self._oai_clients.get(base_url)
        if cli is None:
            cli = AsyncOpenAI(base_url=base_url, api_key=os.getenv("SWARM_OAI_API_KEY") or "EMPTY",
                              max_retries=0, timeout=180)
            self._oai_clients[base_url] = cli
        return cli

    @staticmethod
    def parse_oai_model(model: str) -> tuple[str, str]:
        """"oai:<base_url>|<model_name>" -> (base_url, model_name)."""
        spec = model.removeprefix("oai:")
        if "|" not in spec:
            raise ValueError(f"expected 'oai:<base_url>|<model_name>', got {model!r}")
        base_url, name = spec.rsplit("|", 1)
        if not base_url.startswith(("http://", "https://")) or not name:
            raise ValueError(f"expected 'oai:<base_url>|<model_name>', got {model!r}")
        return base_url.rstrip("/"), name

    async def _chat_oai(self, model: str, messages: list[dict], *, temperature: float,
                        max_tokens: int, sample: int, json_mode: bool,
                        retries: int) -> ChatResult:
        """Self-hosted or third-party OpenAI-compatible server (vLLM, SGLang, TGI), e.g. a
        fine-tuned model: "oai:http://host:8000/v1|my-model". Same cache, ledger and budget as
        the other backends. Cost comes from SWARM_OAI_USD_PER_M_IN / _OUT (USD per million
        tokens, default 0, e.g. for a GPU paid by the hour)."""
        base_url, name = self.parse_oai_model(model)
        request: dict = {"model": name, "messages": messages, "temperature": temperature,
                         "max_tokens": max_tokens, "seed": int(sample) % 2_147_483_647}
        if json_mode:
            request["response_format"] = {"type": "json_object"}
        key = hashlib.sha256(json.dumps({**request, "sample": sample, "endpoint": base_url,
                                         "v": CACHE_VERSION}, sort_keys=True).encode()).hexdigest()
        if (hit := self._store.get(key)) is not None:
            self.n_cached += 1
            return ChatResult(**{**hit, "cached": True})
        p_in = float(os.getenv("SWARM_OAI_USD_PER_M_IN", "0")) / 1e6
        p_out = float(os.getenv("SWARM_OAI_USD_PER_M_OUT", "0")) / 1e6
        reserve = sum(len(m.get("content", "")) for m in messages) / 3.0 * p_in + max_tokens * p_out
        if self.total_spent() + self._reserved + reserve > self.budget_usd:
            raise BudgetExceeded(f"spend ${self.total_spent():.2f} would exceed "
                                 f"${self.budget_usd:.2f}")
        self._reserved += reserve
        try:
            resp, latency = await self._send(request, {}, retries, client=self._oai_client(base_url))
        finally:
            self._reserved -= reserve
        usage = resp.usage
        pt = usage.prompt_tokens if usage else 0
        ct = usage.completion_tokens if usage else 0
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        rt = int(getattr(details, "reasoning_tokens", 0) or 0)
        cost = pt * p_in + ct * p_out
        result = {"text": resp.choices[0].message.content or "", "model": model,
                  "prompt_tokens": pt, "completion_tokens": ct, "cost_usd": cost,
                  "reasoning_tokens": rt, "provider": "oai"}
        self._store.put(key, result, (time.time(), key, model, "oai", cost, pt, ct, rt, latency))
        self._spent_cache = (0.0, 0.0)
        self.spent_usd += cost
        self.n_calls += 1
        return ChatResult(**result, cached=False)

    def summary(self) -> str:
        return (
            f"{self.n_calls} live calls, {self.n_cached} cached, this process "
            f"${self.spent_usd:.3f}; global ${self.total_spent():.3f} of "
            f"${self.budget_usd:.2f} cap; reasoning leaks {self.n_reasoning_leaks}"
        )


def parse_json(text: str) -> dict:
    """Parse the first JSON object in a model reply (tolerates ```json fences and prose)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    return json.loads(text[start : end + 1])
