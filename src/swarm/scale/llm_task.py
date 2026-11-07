"""The LLM client for this package: swarm.llm.LLM with a task-scoped cache and spend cap.

swarm.llm.LLM.total_spent() takes the larger of the cache ledger and the OpenRouter key's
lifetime usage. All calls here go to Adaption, and the OpenRouter key's usage (other work)
would block every call under a task cap, so `TaskLLM` ignores it. The cache file
(results/cache/scale_llm.sqlite by default) holds only this package's calls, so its ledger is
this task's spend and `budget_usd` caps exactly that. It also keeps the shared cache free of
write contention with other running processes. Reruns replay from this cache.

Memory: swarm.llm as committed on Nov 7 morning opened a new httpx.AsyncClient for every
Adaption call (and every retry), and each client built its own SSL context with the full CA
bundle. At 128 calls in flight that took one harness process to 620 MB. `share_ssl_context()`
(called by TaskLLM) makes every httpx client in this process that is created without an
explicit `verify` reuse one SSL context. Certificate checks are unchanged (same default
context, same CA bundle). With a swarm.llm that keeps one shared client the patch is
harmless.
"""

from __future__ import annotations

import ssl
from pathlib import Path

import certifi
import httpx

from swarm.llm import LLM, ROOT

SCALE_CACHE = ROOT / "results" / "cache" / "scale_llm.sqlite"


_SSL: ssl.SSLContext | None = None


def share_ssl_context() -> ssl.SSLContext:
    """Patch httpx.AsyncClient (this process only) to default to one shared SSL context."""
    global _SSL
    if _SSL is None:
        _SSL = ssl.create_default_context(cafile=certifi.where())
        orig = httpx.AsyncClient.__init__

        def init(self, *a, **kw):
            kw.setdefault("verify", _SSL)
            orig(self, *a, **kw)

        httpx.AsyncClient.__init__ = init
    return _SSL


class TaskLLM(LLM):
    def __init__(self, budget_usd: float = 40.0, max_concurrency: int = 64,
                 cache_path: Path = SCALE_CACHE):
        share_ssl_context()
        super().__init__(budget_usd=budget_usd, max_concurrency=max_concurrency,
                         cache_path=cache_path)

    def key_usage(self) -> float | None:
        return None
