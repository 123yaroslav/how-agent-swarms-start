"""EXPLORATORY: does a newer model of the same family answer peers? GLM 5.3 Flash vs GLM 5.2.

Palan et al. (2026) report that GLM 5.2 reasoned about other agents in 2% of their incident
reproduction trajectories and GLM 5.3 in 92%. Our confirmatory set found that GLM 5.2 never wrote
to other jobs without permission and never answered a peer's note. This runs the same single-agent
protocol on adaption/glm-53-flash. Plan: docs/ideas/glm53.md (committed before any episode).

    caffeinate -i uv run python experiments/glm53_explore.py [--concurrency 16] [--budget 40]
    uv run python experiments/glm53_explore.py --table

Cells (house rule on unless "norule"; 30 turns; store of 200 other jobs; temperature 0.8; max
tokens 3,000): unseeded plain-rule, coordinate_ok-rule, plain-norule, 50 each; seeded probes
plain-rule and plain-norule with a request note and a memory note, 25 each; seeded probes
coordinate_ok-rule with a request note, 25. World seeds 6000-6049 (unseeded), 6100-6124
(request), 6200-6224 (memory), disjoint from every earlier run. Own cache and ledger
results/cache/glm53_llm.sqlite. Output results/raw/scale/glm53.jsonl and results/explore/glm53.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swarm.scale.analysis import episode_rows, harness_rates  # noqa: E402
from swarm.scale.runner import (LLMPolicy, RunSpec, labels_path, read_jsonl,  # noqa: E402
                                run_many)

LABEL = "EXPLORATORY"
TAG = "glm53"
MODEL = "adaption/glm-53-flash"
RAW = ROOT / "results" / "raw" / "scale" / "glm53.jsonl"
OUT = ROOT / "results" / "explore" / "glm53.json"
CACHE = ROOT / "results" / "cache" / "glm53_llm.sqlite"
STEPS, BG, MAX_TOKENS, TEMP = 30, 200, 3000, 0.8
UNSEEDED = [("plain", True), ("coordinate_ok", True), ("plain", False)]
PROBES = [("plain", True, "request"), ("plain", True, "memory"), ("plain", False, "request"),
          ("plain", False, "memory"), ("coordinate_ok", True, "request")]
N_UNSEEDED, N_PROBE = 50, 25
SEED0 = {"none": 6000, "request": 6100, "memory": 6200}


def specs() -> list[RunSpec]:
    out = []
    for i in range(N_UNSEEDED):  # seed-major: a run cut early stays balanced
        for prompt, rule in UNSEEDED:
            out.append(RunSpec(tag=TAG, mode="harness", prompt=prompt, rule=rule,
                               seed_note="none", n_background=BG, steps=STEPS, model=MODEL,
                               max_tokens=MAX_TOKENS, temperature=TEMP,
                               seed=SEED0["none"] + i, repeat=0))
        if i < N_PROBE:
            for prompt, rule, note in PROBES:
                out.append(RunSpec(tag=TAG, mode="harness", prompt=prompt, rule=rule,
                                   seed_note=note, n_background=BG, steps=STEPS, model=MODEL,
                                   max_tokens=MAX_TOKENS, temperature=TEMP,
                                   seed=SEED0[note] + i, repeat=0))
    return out


def summarize() -> dict:
    want = {s.episode_id() for s in specs()}
    labels = {r["episode_id"]: r for r in read_jsonl(labels_path(RAW)) if r["episode_id"] in want}
    recs, seen = [], set()
    if RAW.exists():
        with RAW.open() as f:
            for ln in f:
                try:
                    rec = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if rec["episode_id"] in want and rec["episode_id"] in labels \
                        and rec["episode_id"] not in seen:
                    seen.add(rec["episode_id"])
                    recs.append(rec)
    rows = episode_rows(recs, labels) if recs else []
    rates = harness_rates(rows) if rows else {}
    cost = sum(r["totals"]["cost"] for r in recs)
    return {"label": LABEL, "model": MODEL, "episodes": len(recs), "cost_usd_nominal": round(cost, 4),
            "rates": rates}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--budget", type=float, default=40.0)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N specs (pilot)")
    a = ap.parse_args(argv)
    if not a.table:
        from swarm.scale.llm_task import TaskLLM
        llm = TaskLLM(budget_usd=a.budget, max_concurrency=a.concurrency, cache_path=CACHE)
        pol = LLMPolicy(llm, MODEL, max_tokens=MAX_TOKENS, temperature=TEMP)
        sp = specs()[: a.limit] if a.limit else specs()
        asyncio.run(run_many(sp, lambda s: pol, RAW, concurrency=a.concurrency,
                             progress=lambda k, n: print(f"{k}/{n}", flush=True)
                             if k % 10 == 0 or k == n else None))
        print(llm.summary())
    s = summarize()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(s, indent=1))
    keys = ("p_write_addressed", "p_write", "p_use", "p_success", "p_seed_read", "p_join",
            "p_respond", "n_episodes")
    for arm, r in s["rates"].items():
        print(arm, {k: (f"{r[k]['k']}/{r[k]['n']}" if isinstance(r.get(k), dict) else r.get(k))
                    for k in keys if k in r})
    print("episodes", s["episodes"], "cost", s["cost_usd_nominal"])


if __name__ == "__main__":
    main()
