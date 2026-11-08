"""EXPLORATORY: do closed models from two frontier labs write to, see and answer peers?

Every model in the confirmatory set is open-weight; the incident model was a closed OpenAI model.
This runs the confirmatory single-agent protocol on two cheap closed models through OpenRouter:
openai/gpt-6-luna and anthropic/claude-haiku-5.5. Plan: docs/ideas/closed_models.md (committed
before any episode).

    caffeinate -i uv run python experiments/closed_models_explore.py [--concurrency 16] [--budget 40]
    uv run python experiments/closed_models_explore.py --table

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
TAG = "closed"
MODELS = ("openai/gpt-6-luna", "anthropic/claude-haiku-5.5")
MODEL = MODELS[0]
RAW = ROOT / "results" / "raw" / "scale" / "closed.jsonl"
OUT = ROOT / "results" / "explore" / "closed_models.json"
CACHE = ROOT / "results" / "cache" / "closed_llm.sqlite"
STEPS, BG, MAX_TOKENS, TEMP = 30, 200, 3000, 0.8
UNSEEDED = [("plain", True, 40), ("coordinate_ok", True, 20)]
PROBES = [("plain", False, "request"), ("plain", False, "memory"), ("coordinate_ok", True, "request")]
N_UNSEEDED, N_PROBE = 40, 25
SEED0 = {"none": 7000, "request": 7100, "memory": 7200}


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
                for prompt, rule, note in PROBES:
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
                if rec["episode_id"] in want and rec["episode_id"] in labels \
                        and rec["episode_id"] not in seen:
                    seen.add(rec["episode_id"])
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
