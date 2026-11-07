"""Single-agent propensity harness (swarm.scale): one blocked job in a store holding the
artifacts of 200 simulated jobs, with or without one seed note. Rates p_write, p_use,
p_respond, p_join per arm with Wilson intervals.

    # smoke (Nov 7): 4 prompt arms x rule/no rule, 200 episodes per arm
    # (100 without a note, 50 with a request note, 50 with a memory note), 20 turns
    caffeinate -i uv run python experiments/scale_harness.py --tag smoke1
    # scripted dry run (no LLM)
    uv run python experiments/scale_harness.py --tag dry --model scripted:noter
    # cue arms (one deployment-style line added to the plain prompt), rule on and off
    caffeinate -i uv run python experiments/scale_harness.py --tag cues1 \
        --arms cue_shared_fact,cue_peers_have,cue_team_culture,cue_multiagent
    # re-print the table of a finished run (no calls)
    uv run python experiments/scale_harness.py --tag smoke1 --table

World seeds: none 0..99, request 1000..1049, memory 2000..2049; the same seeds in every arm
(arms are paired). Output: results/raw/scale/harness_<tag>.jsonl (+ .labels.jsonl), summary
results/summary/scale_harness_<tag>.json. Finished episodes are skipped on a rerun; calls
replay from results/cache/scale_llm.sqlite.
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

from swarm.scale.analysis import episode_rows, harness_rates, harness_table
from swarm.scale.runner import LLMPolicy, RunSpec, labels_path, peak_rss_mb, read_jsonl, run_many

RAW = ROOT / "results" / "raw" / "scale"
SUMMARY = ROOT / "results" / "summary"
SEED_BASE = {"none": 0, "request": 1000, "memory": 2000}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tag", required=True)
    p.add_argument("--arms", default="plain,persist,collab_line,coordinate_ok")
    p.add_argument("--rules", default="rule,norule")
    p.add_argument("--split", default="none=100,request=50,memory=50")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--read-only", action="store_true")
    p.add_argument("--n-background", type=int, default=200)
    p.add_argument("--model", default="adaption/deepseek-v4-flash")
    p.add_argument("--max-tokens", type=int, default=3000)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--concurrency", type=int, default=128)
    p.add_argument("--history-window", type=int, default=12)
    p.add_argument("--budget", type=float, default=40.0)
    p.add_argument("--table", action="store_true")
    return p


def make_specs(a) -> list[RunSpec]:
    split = {k: int(v) for k, v in (x.split("=") for x in a.split.split(","))}
    out = []
    for prompt in a.arms.split(","):
        for rule in a.rules.split(","):
            for note, n in split.items():
                for i in range(n):
                    out.append(RunSpec(tag=a.tag, mode="harness", prompt=prompt,
                                       rule=rule == "rule", read_only=a.read_only,
                                       seed_note=note, n_background=a.n_background,
                                       steps=a.steps, model=a.model, max_tokens=a.max_tokens,
                                       temperature=a.temperature, seed=SEED_BASE[note] + i,
                                       history_window=a.history_window))
    return out


def summarize(out: Path, specs: list[RunSpec]) -> dict:
    """Rates over the episodes of `specs` found in `out`, streamed one record at a time."""
    want = {s.episode_id() for s in specs}
    labels = {r["episode_id"]: r for r in read_jsonl(labels_path(out)) if r["episode_id"] in want}
    rows, cost, calls, live, pt, ct, malformed, errors = [], 0.0, 0, 0, 0, 0, 0, 0
    with out.open() as f:
        for ln in f:
            rec = json.loads(ln)
            if rec["episode_id"] not in want:
                continue
            rows += episode_rows([rec], labels)
            t = rec["totals"]
            cost += t["cost"]
            calls += t["calls"]
            live += t["live"]
            pt += t["pt"]
            ct += t["ct"]
            malformed += t["malformed"]
            errors += t["errors"]
    rates = harness_rates(rows)
    return {"rates": rates, "n_episodes": len(rows), "calls": calls, "live_calls": live,
            "cost_usd_nominal": round(cost, 4), "prompt_tokens": pt, "completion_tokens": ct,
            "malformed": malformed, "errors": errors,
            "note_kinds": _kinds(rows)}


def _kinds(rows: list[dict]) -> dict:
    out: dict = {}
    for r in rows:
        for _, kind, _ in r["note_texts"]:
            out.setdefault(r["arm"], {}).setdefault(kind, 0)
            out[r["arm"]][kind] += 1
    return out


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    specs = make_specs(a)
    out = RAW / f"harness_{a.tag}.jsonl"
    SUMMARY.mkdir(parents=True, exist_ok=True)
    extra: dict = {}
    if not a.table:
        llm = None
        if a.model.startswith("scripted:"):
            from swarm.scale.scripted import ScriptedPolicy
            style = a.model.split(":", 1)[1]
            policy_for = lambda s: ScriptedPolicy(style)
        else:
            from swarm.scale.llm_task import TaskLLM
            llm = TaskLLM(budget_usd=a.budget, max_concurrency=a.concurrency)
            shared = LLMPolicy(llm, a.model, max_tokens=a.max_tokens, temperature=a.temperature)
            policy_for = lambda s: shared
        from swarm.scale.http_stats import count_http
        t0 = time.time()
        last = [0.0]

        def progress(k: int, n: int) -> None:
            if time.time() - last[0] > 30 or k == n:
                last[0] = time.time()
                el = time.time() - t0
                sp = f" spend ${llm.total_spent():.3f}" if llm else ""
                print(f"{k}/{n} episodes, {el:.0f}s, rss {peak_rss_mb():.0f}MB{sp}", flush=True)

        with count_http() as hs:
            res = asyncio.run(run_many(specs, policy_for, out, concurrency=a.concurrency,
                                       progress=progress))
        extra = {"run": res, "http": hs.summary(), "concurrency": a.concurrency}
        if llm is not None:
            extra["llm_summary"] = llm.summary()
            extra["task_spend_usd"] = round(llm.total_spent(), 4)
            print(llm.summary())
        print("run:", json.dumps(res), "http:", json.dumps(extra["http"]))
    s = summarize(out, specs)
    if extra:
        wall = extra["run"]["wall_s"]
        extra["live_calls_per_min"] = round(60 * extra["http"]["responses"] / wall, 1) \
            if wall else None
    s.update(extra)
    s["args"] = vars(a)
    path = SUMMARY / f"scale_harness_{a.tag}.json"
    if a.table and path.exists():  # keep the run's cost and throughput fields
        old = json.loads(path.read_text())
        s = {**{k: v for k, v in old.items() if k not in s}, **s}
    path.write_text(json.dumps(s, indent=1))
    print(harness_table(s["rates"]))
    print(f"episodes {s['n_episodes']}, calls {s['calls']} (live {s['live_calls']}), nominal "
          f"cost ${s['cost_usd_nominal']:.3f}, malformed {s['malformed']}, errors {s['errors']}")
    print("note kinds by arm:", json.dumps(s["note_kinds"]))


if __name__ == "__main__":
    main()
