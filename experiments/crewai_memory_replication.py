"""EXPLORATORY replication: the CrewAI memory sentence (real_c1_crewai_save_memory) vs the plain prompt on
GLM 5.2 and Gemma 4 31B, same worlds as the DeepSeek V4 Flash run (seeds 3000-3049). Plan:
docs/ideas/crewai_memory_replication.md, committed before any episode.

Original runner notes follow.

    caffeinate -i uv run python experiments/realworld_prompts_harness.py            # run
    uv run python experiments/realworld_prompts_harness.py --table                  # no calls

Arms: plain, coordinate_ok and the twelve REAL_CUES arms (swarm.scale.prompts; each adds one
verbatim sentence from a real framework prompt to the plain prompt), all with the house rule,
no seed note, 100 episodes per arm, 20 turns, adaption/deepseek-v4-flash, max_tokens 3000,
temperature 0.8. World seeds 3000-3099, sample offset 0: disjoint from every earlier world
(smoke 0-99, 1000-1049, 2000-2049; exploratory 100-301, 500-999; confirmatory 2100-2999), the
same seeds in every arm (paired). Specs run seed-major, so a run cut early stays balanced over
arms. Concurrency 16 (another process shares the key). Own cache and ledger
results/cache/realworld_llm.sqlite with a $15 cap. Not pre-registered.

Per arm: NOTE_ADDRESSED (code: wrote_note_addressed), NOTE of any kind, USE as in the prereg (a
get of another job's artifact, reads_others > 0; docs/deviations.md D2), p_use (code: compute,
submit or copy of another job's content), rule_broken; Wilson 95%. Output:
results/raw/scale/harness_realworld1.jsonl (+ .labels.jsonl), results/realworld_prompts/
harness.json.
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

from swarm.fleet.stats import wilson  # noqa: E402
from swarm.scale import prompts as P  # noqa: E402
from swarm.scale.analysis import episode_rows  # noqa: E402
from swarm.scale.runner import (LLMPolicy, RunSpec, labels_path, peak_rss_mb,  # noqa: E402
                                read_jsonl, run_many)

LABEL = "EXPLORATORY"
TAG = "crewai_rep"
RAW = ROOT / "results" / "raw" / "scale" / f"harness_{TAG}.jsonl"
OUT = ROOT / "results" / "realworld_prompts" / "crewai_replication.json"
CACHE = ROOT / "results" / "cache" / "crewai_rep_llm.sqlite"
ARMS = ("plain", "real_c1_crewai_save_memory")
SEED0, N, STEPS = 3000, 50, 20
MODEL, MAX_TOKENS, TEMPERATURE = "adaption/glm-5.2", 2500, 0.8
MODELS = ("adaption/glm-5.2", "adaption/gemma-4-31b-it")


def specs(arms: tuple[str, ...] = ARMS) -> list[RunSpec]:
    """Seed-major over models and arms; the tag names the model because episode ids do not."""
    out = []
    for i in range(N):
        for model in MODELS:
            short = model.split("/")[-1]
            mt = 2000 if "gemma" in model else MAX_TOKENS
            for arm in arms:
                out.append(RunSpec(tag=f"{TAG}_{short}", mode="harness", prompt=arm, rule=True,
                                   seed_note="none", n_background=200, steps=STEPS, model=model,
                                   max_tokens=mt, temperature=TEMPERATURE, seed=SEED0 + i,
                                   repeat=0))
    return out


def rate(vals: list[bool]) -> dict:
    k, n = sum(vals), len(vals)
    lo, hi = wilson(k, n) if n else (None, None)
    return {"k": k, "n": n, "p": k / n if n else None, "lo": lo, "hi": hi}


def summarize() -> dict:
    want = {s.episode_id() for s in specs()}
    labels = {r["episode_id"]: r for r in read_jsonl(labels_path(RAW)) if r["episode_id"] in want}
    keys = [f"{m}|{a}" for m in MODELS for a in ARMS]
    by: dict[str, list[dict]] = {k: [] for k in keys}
    kinds: dict[str, dict] = {k: {} for k in keys}
    notes: dict[str, list] = {k: [] for k in keys}
    cost, seen = 0.0, set()
    if RAW.exists():
        with RAW.open() as f:
            for ln in f:
                if not ln.strip():
                    continue
                try:
                    rec = json.loads(ln)
                except json.JSONDecodeError:  # a line still being written by the running process
                    continue
                eid = rec["episode_id"]
                if eid not in want or eid in seen or eid not in labels:
                    continue
                seen.add(eid)
                cost += rec["totals"]["cost"]
                key = f"{rec['spec']['model']}|{rec['spec']['prompt']}"
                for r in episode_rows([rec], labels):
                    by[key].append(r)
                    for name, kind, text in r["note_texts"]:
                        kinds[key][kind] = kinds[key].get(kind, 0) + 1
                        notes[key].append({"seed": r["seed"], "name": name,
                                           "kind": kind, "text": text[:300]})
    cells = {}
    for arm, rs in by.items():
        cells[arm] = {
            "n": len(rs),
            "note_addressed": rate([bool(r["wrote_note_addressed"]) for r in rs]),
            "note_any": rate([bool(r["wrote_note"]) for r in rs]),
            "use_get": rate([r["reads_others"] > 0 for r in rs]),
            "p_use_code": rate([bool(r["uses_others_artifacts"]) for r in rs]),
            "rule_broken": rate([bool(r["rule_broken"]) for r in rs]),
            "report_blocked": rate([bool(r["report_blocked"]) for r in rs]),
            "note_kinds": kinds[arm],
            "line": P.REAL_CUES.get(arm.split("|")[-1], ""),
        }
    return {"label": LABEL, "tag": TAG, "model": MODEL, "steps": STEPS, "rule": True,
            "seeds": [SEED0, SEED0 + N - 1], "episodes": len(seen),
            "cost_usd_nominal": round(cost, 4), "cells": cells, "addressed_notes": notes}


def fr(d: dict) -> str:
    if not d["n"]:
        return "n/a"
    return f"{d['k']}/{d['n']} = {d['p']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--budget", type=float, default=15.0)
    ap.add_argument("--arms", default="", help="comma list to run first (default: all arms); "
                    "the table always covers every arm")
    a = ap.parse_args(argv)
    run_arms = tuple(x for x in a.arms.split(",") if x) or ARMS
    assert all(x in ARMS for x in run_arms), run_arms
    extra: dict = {}
    if not a.table:
        from swarm.scale.http_stats import count_http
        from swarm.scale.llm_task import TaskLLM
        llm = TaskLLM(budget_usd=a.budget, max_concurrency=a.concurrency, cache_path=CACHE)
        pols = {m: LLMPolicy(llm, m, max_tokens=2000 if "gemma" in m else MAX_TOKENS,
                             temperature=TEMPERATURE) for m in MODELS}
        t0, last = time.time(), [0.0]

        def progress(k: int, n: int) -> None:
            if time.time() - last[0] > 60 or k == n:
                last[0] = time.time()
                print(f"{k}/{n} episodes, {time.time() - t0:.0f}s, rss {peak_rss_mb():.0f}MB "
                      f"spend ${llm.total_spent():.3f}", flush=True)

        with count_http() as hs:
            res = asyncio.run(run_many(specs(run_arms), lambda s: pols[s.model], RAW,
                                       concurrency=a.concurrency, progress=progress))
        extra = {"run": res, "http": hs.summary(), "concurrency": a.concurrency,
                 "llm_summary": llm.summary(), "task_spend_usd": round(llm.total_spent(), 4)}
        print(llm.summary())
    s = summarize()
    if OUT.exists() and not extra:
        old = json.loads(OUT.read_text())
        s.update({k: v for k, v in old.items() if k not in s})
    s.update(extra)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(s, indent=1))
    print(f"{LABEL}. harness {TAG}: {s['episodes']} episodes, nominal cost "
          f"${s['cost_usd_nominal']:.3f}")
    print("| arm | n | NOTE_ADDRESSED | NOTE (any) | USE (get) | p_use (code) | rule_broken |")
    for arm, c in s["cells"].items():
        print(f"| {arm} | {c['n']} | {fr(c['note_addressed'])} | {fr(c['note_any'])} | "
              f"{fr(c['use_get'])} | {fr(c['p_use_code'])} | {fr(c['rule_broken'])} |")


if __name__ == "__main__":
    main()
