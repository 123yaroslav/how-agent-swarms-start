"""Throughput of Adaption at a given concurrency: calls per minute, HTTP 429 and 5xx.

Each call is a realistic first turn of the harness (system prompt + job ticket of a fresh
world), with a sample index unique to this test so that nothing comes from the cache.
swarm.llm retries 429 and 5xx itself; swarm.scale.http_stats counts every HTTP response,
retried ones included.

    caffeinate -i uv run python experiments/scale_throughput.py --levels 64,128 --calls 2x

Summary: results/summary/scale_throughput.json (or --out). The first run (Nov 7, 12:10) was
before llm_task.share_ssl_context; the second (--run-id nov7b, --out
results/summary/scale_throughput_ssl.json) after it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swarm.scale import prompts
from swarm.scale.http_stats import count_http
from swarm.scale.jobs import WorldConfig, make_world
from swarm.scale.llm_task import TaskLLM
from swarm.scale.runner import peak_rss_mb, sample_index


async def level(conc: int, n: int, model: str, max_tokens: int, budget: float,
                run_id: str) -> dict:
    llm = TaskLLM(budget_usd=budget, max_concurrency=conc)
    msgs = []
    for i in range(n):
        w = make_world(WorldConfig(seed=50_000 + i, n_background=0))
        j = w.jobs[0]
        msgs.append([{"role": "system", "content": prompts.system_prompt(j.job_id, 20)},
                     {"role": "user", "content": prompts.begin_message(j, 20)}])
    errors: list[str] = []
    lat: list[float] = []

    async def one(i: int) -> None:
        t0 = time.time()
        try:
            await llm.chat(model, msgs[i], max_tokens=max_tokens, temperature=0.8,
                           sample=sample_index("throughput", run_id, conc, i))
            lat.append(time.time() - t0)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{type(e).__name__}: {str(e)[:120]}")

    with count_http() as hs:
        t0 = time.time()
        await asyncio.gather(*(one(i) for i in range(n)))
        wall = time.time() - t0
    lat.sort()
    return {"concurrency": conc, "calls": n, "ok": n - len(errors), "wall_s": round(wall, 1),
            "calls_per_min": round(60 * (n - len(errors)) / wall, 1),
            "call_lat_p50": round(lat[len(lat) // 2], 2) if lat else None,
            "call_lat_p95": round(lat[int(0.95 * len(lat))], 2) if lat else None,
            "http": hs.summary(), "errors": errors[:5], "n_errors": len(errors),
            "cost_usd": round(llm.spent_usd, 4), "llm_summary": llm.summary()}


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--levels", default="64,128")
    p.add_argument("--calls", default="2x", help="calls per level: an int, or Nx concurrency")
    p.add_argument("--model", default="adaption/deepseek-v4-flash")
    p.add_argument("--max-tokens", type=int, default=3000)
    p.add_argument("--budget", type=float, default=40.0)
    p.add_argument("--run-id", default="nov7")
    p.add_argument("--out", default="results/summary/scale_throughput.json")
    a = p.parse_args(argv)
    out = []
    for conc in (int(x) for x in a.levels.split(",")):
        n = int(a.calls[:-1]) * conc if a.calls.endswith("x") else int(a.calls)
        r = asyncio.run(level(conc, n, a.model, a.max_tokens, a.budget, a.run_id))
        print(json.dumps(r), flush=True)
        out.append(r)
    res = {"model": a.model, "levels": out, "peak_rss_mb": round(peak_rss_mb(), 1),
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    path = ROOT / a.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
