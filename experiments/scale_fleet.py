"""Fleet of up to 1,024 jobs on one shared artifact store (swarm.scale), LLM or scripted.

    # scripted agents (no LLM), memory check
    uv run python experiments/scale_fleet.py --model scripted:mix --n 1024 --tag dry1024
    # smoke: N=256, plain prompt, house rule on, 40 turns, deepseek-v4-flash
    caffeinate -i uv run python experiments/scale_fleet.py --n 256 --steps 40 --prompt plain \
        --rule rule --concurrency 128 --tag fleet256_plain_rule
    # re-print the tables of a finished run (no calls)
    uv run python experiments/scale_fleet.py --table --tag fleet256_plain_rule

Output: results/raw/scale/<tag>.{turns,rounds,events,artifacts}.jsonl, <tag>.record.json,
<tag>.labels.json; summary results/summary/scale_<tag>.json. A rerun with the same arguments
replays every call from the cache (results/cache/scale_llm.sqlite), so an interrupted run
resumes by running the same command. Keys come only from .env (ADAPTION_API_KEY).
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

from swarm.scale.analysis import FLEET_KEYS, episode_rows, fleet_table, predict_fleet
from swarm.scale.metrics import fmt_rate
from swarm.scale.prompts import ARMS
from swarm.scale.runner import LLMPolicy, RunSpec, peak_rss_mb, run_world

RAW = ROOT / "results" / "raw" / "scale"
SUMMARY = ROOT / "results" / "summary"
MIX = ("worker", "noter", "user", "waiter", "helper", "helper", "joiner", "silent")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tag", required=True)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--p-blocked", type=float, default=0.3)
    p.add_argument("--prompt", default="plain", choices=list(ARMS))
    p.add_argument("--rule", default="rule", choices=["rule", "norule"])
    p.add_argument("--read-only", action="store_true", help="others' artifacts read-only")
    p.add_argument("--seed-note", default="none", choices=["none", "request", "memory"])
    p.add_argument("--n-background", type=int, default=200)
    p.add_argument("--p-derivative", type=float, default=1.0)
    p.add_argument("--no-producers", action="store_true",
                   help="missing datasets have no producer job in the fleet")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--model", default="adaption/deepseek-v4-flash",
                   help="adaption/<id>, or scripted:<style> / scripted:mix")
    p.add_argument("--max-tokens", type=int, default=3000)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--concurrency", type=int, default=128)
    p.add_argument("--schedule", default="rounds", choices=["rounds", "async"])
    p.add_argument("--history-window", type=int, default=12)
    p.add_argument("--budget", type=float, default=40.0, help="USD cap for this package")
    p.add_argument("--cache", default=None,
                   help="LLM cache and ledger file (default results/cache/scale_llm.sqlite); "
                        "the budget caps the spend in this file's ledger")
    p.add_argument("--harness-summary", default=None,
                   help="results/summary/scale_harness_<tag>.json for the prediction")
    p.add_argument("--table", action="store_true", help="only re-print a finished run")
    p.add_argument("--pad", type=int, default=0,
                   help="scripted only: pad each reply's thought to this many characters and "
                        "load the LLM client (memory check with model-sized histories)")
    return p


def spec_from(a) -> RunSpec:
    return RunSpec(tag=a.tag, mode="fleet", n_agents=a.n, p_blocked=a.p_blocked, prompt=a.prompt,
                   rule=a.rule == "rule", read_only=a.read_only, seed_note=a.seed_note,
                   n_background=a.n_background, p_derivative=a.p_derivative,
                   producers_in_fleet=not a.no_producers, steps=a.steps, model=a.model,
                   max_tokens=a.max_tokens, temperature=a.temperature, seed=a.seed,
                   repeat=a.repeat, schedule=a.schedule, history_window=a.history_window)


def throughput(rounds: list[dict]) -> dict:
    live = sum(r["live"] for r in rounds)
    calls = sum(r["active"] for r in rounds)
    wall = sum(r["wall_s"] for r in rounds)
    return {"rounds": len(rounds), "calls": calls, "live_calls": live,
            "wall_s": round(wall, 1),
            "calls_per_min": round(60 * calls / wall, 1) if wall else None,
            "live_calls_per_min": round(60 * live / wall, 1) if wall else None,
            "errors": sum(r["errors"] for r in rounds),
            "malformed": sum(r["malformed"] for r in rounds),
            "max_round_s": max((r["wall_s"] for r in rounds), default=0.0)}


def iter_events(prefix: Path):
    with open(f"{prefix}.events.jsonl") as f:
        for ln in f:
            if ln.strip():
                yield json.loads(ln)


def report(rec: dict, labels: dict, extra: dict, harness_path: str | None,
           prefix: Path | None = None) -> dict:
    """Tables from the record; with `prefix`, features are recomputed from the events file
    by the current note rules."""
    ev = iter_events(prefix) if prefix is not None and \
        Path(f"{prefix}.events.jsonl").exists() else None
    rows = episode_rows([rec], {rec["episode_id"]: labels}, events=ev, recompute=ev is not None)
    ft = fleet_table(rows)
    print(f"\n{rec['episode_id']}: {len(rows)} jobs, {rec['totals']['calls']} calls, "
          f"wall {rec['wall_s']} s, peak RSS {rec['peak_rss_mb']} MB")
    for g in ("blocked", "producer", "other", "all"):
        print(f"[{g}] n={ft[g]['n']}")
        for k in FLEET_KEYS:
            print(f"  {k:<22} {fmt_rate(ft[g][k])}")
    tp = throughput(rec["rounds"])
    print("throughput:", json.dumps(tp))
    out = {"episode_id": rec["episode_id"], "spec": rec["spec"], "fleet": ft,
           "totals": rec["totals"], "throughput": tp, "peak_rss_mb": rec["peak_rss_mb"],
           "wall_s": rec["wall_s"], "stopped": rec["stopped"], **extra}
    if harness_path:
        h = json.loads(Path(harness_path).read_text())["rates"]
        arm = rec["spec"]["prompt"] + "-" + ("rule" if rec["spec"]["rule"] else "norule")
        if arm in h:
            pred = predict_fleet(h[arm], rows)
            out["prediction"] = pred
            print(f"prediction ({arm}): q={pred['q_read_peer_note_first']:.3f} "
                  f"pred_write={pred['pred_write']:.3f} obs_write={fmt_rate(pred['obs_write'])} "
                  f"pred_use={pred['pred_use']:.3f} obs_use={fmt_rate(pred['obs_use'])}")
    return out


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    spec = spec_from(a)
    prefix = RAW / a.tag
    SUMMARY.mkdir(parents=True, exist_ok=True)
    if a.table:
        rec = json.loads(Path(f"{prefix}.record.json").read_text())
        labels = json.loads(Path(f"{prefix}.labels.json").read_text())
        out = report(rec, labels, {}, a.harness_summary, prefix)
        path = SUMMARY / f"scale_{a.tag}.json"
        if path.exists():  # keep the run's cost, throughput and http fields
            old = json.loads(path.read_text())
            out = {**{k: v for k, v in old.items() if k not in out}, **out,
                   "throughput": old.get("throughput", out["throughput"])}
        path.write_text(json.dumps(out, indent=1))
        return
    llm = None
    if a.model.startswith("scripted:"):
        from swarm.scale.scripted import ScriptedPolicy
        style = a.model.split(":", 1)[1]
        if style == "mix":
            smap: dict = {}
            style = lambda j: smap.setdefault(j, MIX[len(smap) % len(MIX)])
        policy = ScriptedPolicy(style)
        if a.pad:
            from swarm.fleet.runner import Reply
            from swarm.scale.llm_task import TaskLLM
            inner, pad = policy, a.pad

            async def policy(job, messages, sample):
                r = await inner(job, messages, sample)
                act = json.loads(r.text)
                act["thought"] = (act["thought"] + " " + "x" * pad)[:pad]
                return Reply(json.dumps(act), r.meta)
    else:
        from swarm.scale.llm_task import SCALE_CACHE, TaskLLM
        llm = TaskLLM(budget_usd=a.budget, max_concurrency=a.concurrency,
                      cache_path=ROOT / a.cache if a.cache else SCALE_CACHE)
        policy = LLMPolicy(llm, a.model, max_tokens=a.max_tokens, temperature=a.temperature)
    from swarm.scale.http_stats import count_http

    def progress(rr: dict) -> None:
        print(f"step {rr['step']:>3} active {rr['active']:>5} live {rr['live']:>5} "
              f"cached {rr['cached']:>5} err {rr['errors']:>3} {rr['wall_s']:>7.1f}s "
              f"{rr['calls_per_min'] or 0:>8.0f}/min lat {rr['lat_mean']:.1f}/{rr['lat_max']:.1f}"
              f" rss {rr['peak_rss_mb']}MB ${rr['cost']:.4f}", flush=True)

    t0 = time.time()
    with count_http() as hs:
        rec, labels = asyncio.run(run_world(spec, policy, sem=asyncio.Semaphore(a.concurrency),
                                            out_prefix=prefix, progress=progress))
    extra = {"http": hs.summary(), "concurrency": a.concurrency,
             "elapsed_s": round(time.time() - t0, 1)}
    if llm is not None:
        extra["llm_summary"] = llm.summary()
        extra["task_spend_usd"] = round(llm.total_spent(), 4)
    out = report(rec, labels, extra, a.harness_summary, prefix)
    out["peak_rss_mb_process"] = round(peak_rss_mb(), 1)
    (SUMMARY / f"scale_{a.tag}.json").write_text(json.dumps(out, indent=1))
    print("http:", json.dumps(extra["http"]))
    print(f"peak RSS (process) {out['peak_rss_mb_process']} MB")
    if llm is not None:
        print(llm.summary())


if __name__ == "__main__":
    main()
