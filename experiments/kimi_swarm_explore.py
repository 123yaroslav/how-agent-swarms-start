"""EXPLORATORY: does a model trained for agent swarms (Kimi K2.5) write to, see and answer peers?

Kimi K2.5 (Kimi Team, arXiv 2602.02276) was trained with Parallel-Agent Reinforcement Learning to
run an "Agent Swarm" of sub-agents. It is the closest available stand-in for the test of
collaboration training (C5), which could not be served. Plan: docs/ideas/kimi_swarm.md (committed
before any episode).

    caffeinate -i uv run python experiments/kimi_swarm_explore.py [--concurrency 16] [--budget 40]
    uv run python experiments/kimi_swarm_explore.py --table

Cells per model (30 turns; store of 200 other jobs; temperature 0.8; max tokens 3,000): unseeded
plain-rule 40, coordinate_ok-rule 20; seeded probes plain-norule with a request note and a memory
note, 25 each; coordinate_ok-rule with a request note, 25. World seeds 7000-7039 (unseeded),
7100-7124 (request), 7200-7224 (memory), disjoint from every earlier run. Own cache and ledger
results/cache/closed_llm.sqlite. Output results/raw/scale/closed.jsonl and
results/explore/closed_models.json.
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
TAG = "kimi"
MODELS = ("moonshotai/kimi-k2.5",)
MODEL = MODELS[0]
RAW = ROOT / "results" / "raw" / "scale" / "kimi.jsonl"
OUT = ROOT / "results" / "explore" / "kimi_swarm.json"
CACHE = ROOT / "results" / "cache" / "kimi_llm.sqlite"
STEPS, BG, MAX_TOKENS, TEMP = 30, 200, 2000, 0.8
UNSEEDED = [("plain", True, 16)]
PROBES = [("plain", False, "request", 16), ("plain", False, "memory", 8),
          ("coordinate_ok", True, "request", 8)]
N_UNSEEDED, N_PROBE = 16, 16
SEED0 = {"none": 9500, "request": 9600, "memory": 9700}


def specs() -> list[RunSpec]:
    out = []
    for i in range(N_UNSEEDED):  # seed-major: a run cut early stays balanced
        for model in MODELS:
            for prompt, rule, n in UNSEEDED:
                if i < n:
                    out.append(RunSpec(tag=TAG, mode="harness", prompt=prompt, rule=rule,
                                       seed_note="none", n_background=BG, steps=STEPS,
                                       model=model, max_tokens=MAX_TOKENS, temperature=TEMP,
                                       seed=SEED0["none"] + i, repeat=0))
            if i < N_PROBE:
                for prompt, rule, note, n in PROBES:
                    if i >= n:
                        continue
                    out.append(RunSpec(tag=TAG, mode="harness", prompt=prompt, rule=rule,
                                       seed_note=note, n_background=BG, steps=STEPS,
                                       model=model, max_tokens=MAX_TOKENS, temperature=TEMP,
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
                # episode ids do not name the model, so two models on one world share an id
                key = (rec["episode_id"], rec["spec"]["model"])
                if rec["episode_id"] in want and rec["episode_id"] in labels and key not in seen:
                    seen.add(key)
                    recs.append(rec)
    out = {"label": LABEL, "models": {}, "episodes": len(recs),
           "cost_usd_nominal": round(sum(r["totals"]["cost"] for r in recs), 4)}
    for model in MODELS:
        mine = [r for r in recs if r["spec"]["model"] == model]
        rows = episode_rows(mine, labels) if mine else []
        out["models"][model] = {"episodes": len(mine), "rates": harness_rates(rows) if rows else {}}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N specs (pilot)")
    a = ap.parse_args(argv)
    if not a.table:
        from swarm.scale.llm_task import TaskLLM
        llm = TaskLLM(budget_usd=a.budget, max_concurrency=a.concurrency, cache_path=CACHE)
        pols = {m: LLMPolicy(llm, m, max_tokens=MAX_TOKENS, temperature=TEMP) for m in MODELS}
        sp = specs()[: a.limit] if a.limit else specs()
        asyncio.run(run_many(sp, lambda s: pols[s.model], RAW, concurrency=a.concurrency,
                             progress=lambda k, n: print(f"{k}/{n}", flush=True)
                             if k % 10 == 0 or k == n else None))
        print(llm.summary())
    s = summarize()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(s, indent=1))
    keys = ("p_write_addressed", "p_write", "p_use", "p_success", "p_seed_read", "p_join",
            "p_respond", "n_episodes")
    for model, d in s["models"].items():
        for arm, r in d["rates"].items():
            print(model, arm, {k: (f"{r[k]['k']}/{r[k]['n']}" if isinstance(r.get(k), dict)
                                   else r.get(k)) for k in keys if k in r})
    print("episodes", s["episodes"], "cost", s["cost_usd_nominal"])


if __name__ == "__main__":
    main()
