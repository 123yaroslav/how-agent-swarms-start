"""Count HTTP status codes of the Adaption calls without touching swarm/llm.py.

swarm.llm.LLM sends Adaption requests with `httpx.AsyncClient.post` and retries 429 and 5xx
itself, so its results hide them. `count_http()` wraps that method for the duration of a
`with` block and records every response's status code and latency, including the retried
ones, plus transport exceptions.
"""

from __future__ import annotations

import contextlib
import time
from collections import Counter

import httpx


class HttpStats:
    def __init__(self) -> None:
        self.status: Counter = Counter()
        self.exceptions: Counter = Counter()
        self.latencies: list[float] = []
        self.t0 = time.time()

    def summary(self) -> dict:
        lat = sorted(self.latencies)

        def q(p: float) -> float | None:
            return round(lat[min(len(lat) - 1, int(p * len(lat)))], 2) if lat else None

        n = sum(self.status.values())
        return {"responses": n, "status": dict(sorted(self.status.items())),
                "n_429": self.status.get(429, 0),
                "n_5xx": sum(v for k, v in self.status.items() if 500 <= k < 600),
                "exceptions": dict(self.exceptions), "lat_p50": q(0.5), "lat_p95": q(0.95),
                "lat_max": lat[-1] if lat else None}


@contextlib.contextmanager
def count_http():
    stats = HttpStats()
    orig = httpx.AsyncClient.post

    async def post(self, url, *a, **kw):
        t0 = time.time()
        try:
            r = await orig(self, url, *a, **kw)
        except Exception as e:
            stats.exceptions[type(e).__name__] += 1
            raise
        stats.status[r.status_code] += 1
        stats.latencies.append(time.time() - t0)
        return r

    httpx.AsyncClient.post = post
    try:
        yield stats
    finally:
        httpx.AsyncClient.post = orig
