"""Run single-agent or fleet episodes of the benign data-assembly world (src/swarm/fleet).

Writes results/raw/fleet/<tag>.jsonl (one episode record per line), <tag>.labels.jsonl (answer
keys, read only by evaluation) and <tag>.metrics.json (summary by task type). Rerunnable from
the LLM cache: the default "rounds" schedule is deterministic, so a rerun replays every call.

    # 10 single-agent episodes, plain prompt, half of them impossible
    uv run python experiments/run_fleet.py --mode single --p 0.5 --prompt plain \
        --model adaption/deepseek-v4-flash --seeds 0-9 --tag single_plain
    # fleet of 8, 30% impossible tasks, shared cache, positive control
    uv run python experiments/run_fleet.py --mode fleet --n 8 --p 0.3 --channel cache \
        --prompt allow_sharing --seeds 0 --tag fleet8_share
    # free dry run with scripted agents
    uv run python experiments/run_fleet.py --mode fleet --n 8 --model scripted:snoop --tag dry
    # branch: continue every gave-up-early episode of a tag from its give-up point
    uv run python experiments/run_fleet.py --mode single --branch-from single_plain \
        --branch-when giveup --prompt persist --tag single_plain_branch

--seeds accepts "a-b" (inclusive), "a,b,c", or a bare count N (= 0..N-1).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from swarm.fleet import prompts
from swarm.fleet.metrics import episode_summary, evaluate, outcomes, summarize_by
from swarm.fleet.runner import (
    EpisodeSpec,
    FnPolicy,
    LLMPolicy,
    labels_path,
    load_run,
    run_episode,
    write_jsonl,
)
from swarm.fleet.scripted import ScriptedAgent

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "raw" / "fleet"


def parse_seeds(s: str) -> list[int]:
    s = s.strip()
    if re.fullmatch(r"\d+-\d+", s):
        a, b = map(int, s.split("-"))
        return list(range(a, b + 1))
    if "," in s:
        return [int(x) for x in s.split(",") if x.strip()]
    return list(range(int(s)))


def unused_mb() -> float | None:
    """Unused physical memory in MB from `top -l 1` (macOS); None if unknown."""
    try:
        out = subprocess.run(["top", "-l", "1", "-n", "0"], capture_output=True, text=True,
                             check=False,
                             timeout=20).stdout
    except Exception:  # noqa: BLE001
        return None
    m = re.search(r"PhysMem:.*?(\d+)([MG]) unused", out)
    if not m:
        return None
    return float(m.group(1)) * (1024 if m.group(2) == "G" else 1)


def key_table(summary: dict) -> str:
    lines = []
    for group, s in summary.items():
        lines.append(f"  [{group}] agents {s['n_agents']}, episodes {s['n_episodes']}")
        for k in ("cross_read", "cross_read_home", "cross_read_shared", "violation_any",
                  "shared_note", "chain", "req_ans", "swarm_any", "sanctioned_any",
                  "submitted", "full_success", "gave_up_early", "reported", "honest_report",
                  "false_alarm", "silent_partial", "malformed_any"):
            if k in s:
                v = s[k]
                lines.append(f"    {k:<18} {v['k']}/{v['n']} = {v['p']:.2f}  Wilson "
                             f"[{v['wilson'][0]:.2f}, {v['wilson'][1]:.2f}]")
        for k in ("score_frac", "submit_step", "steps_used", "malformed"):
            if k in s:
                v = s[k]
                lines.append(f"    {k:<18} mean {v['mean']:.2f}  boot [{v['boot'][0]:.2f}, "
                             f"{v['boot'][1]:.2f}] (n {v['n']})")
    return "\n".join(lines)


def make_policy(args, llm):
    if args.model.startswith("scripted:"):
        return FnPolicy(ScriptedAgent(args.model.split(":", 1)[1]))
    return LLMPolicy(llm, args.model, max_tokens=args.max_tokens, temperature=args.temperature)


def branch_sources(tag: str, when: str) -> list[tuple[dict, int]]:
    eps, labs = load_run(OUT / f"{tag}.jsonl")
    out = []
    for r in eps:
        if r.get("branch") or r["spec"]["mode"] != "single" or r["episode_id"] not in labs:
            continue
        a = r["world"]["agents"][0]
        o = outcomes(r, labs[r["episode_id"]])[a]
        if (when == "giveup" and o["gave_up_early"]) or \
                (when == "silent_partial" and o["silent_partial"]):
            out.append((r, r["submit_step"][a]))
    return out


async def main(args) -> int:
    free = unused_mb()
    conc = args.conc
    if free is not None and free < 300:
        conc = min(conc, 16)
    conc = min(conc, 32)
    print(f"unused memory {free} MB -> concurrency {conc}", flush=True)

    llm = None
    if not args.model.startswith("scripted:"):
        from swarm.llm import LLM
        llm = LLM(budget_usd=args.budget, max_concurrency=conc)
    policy = make_policy(args, llm)
    world_kw = {"k_items": args.k, "n_shard": args.shard, "n_missing": args.missing,
                "n_decoys": args.decoys, "n_old_shards": args.old_shards,
                "busy_prob": args.busy, "peer_frac": args.peer_frac}
    channel = args.channel or ("none" if args.mode == "single" else "cache")
    jobs: list[tuple[EpisodeSpec, dict | None]] = []
    if args.branch_from:
        for src, at in branch_sources(args.branch_from, args.branch_when):
            s = dict(src["spec"])
            s.update(tag=args.tag, prompt=args.prompt, model=args.model,
                     max_tokens=args.max_tokens, temperature=args.temperature)
            if args.prompt == "allow_sharing":
                raise SystemExit("branching is for single-agent episodes")
            jobs.append((EpisodeSpec(**s), {"from": src, "at": at}))
        print(f"branching {len(jobs)} episodes of {args.branch_from} ({args.branch_when})")
    else:
        for seed in parse_seeds(args.seeds):
            for rep in range(args.repeats):
                jobs.append((EpisodeSpec(
                    tag=args.tag, mode=args.mode, seed=seed,
                    n_agents=1 if args.mode == "single" else args.n, p=args.p, channel=channel,
                    prompt=args.prompt, rule=args.rule, model=args.model, steps=args.steps,
                    max_tokens=args.max_tokens, temperature=args.temperature, repeat=rep,
                    schedule=args.schedule, world=world_kw), None))
    for spec, _ in jobs[:1]:
        spec.world_config().validate()
        prompts.system_prompt("agent-00", spec.steps, channel=spec.channel, variant=spec.prompt)

    ep_path = OUT / f"{args.tag}.jsonl"
    lab_path = labels_path(ep_path)
    write_jsonl(ep_path, [], mode="w")
    write_jsonl(lab_path, [], mode="w")
    sem = asyncio.Semaphore(args.episode_conc or (conc if args.mode == "single" else 1))
    stop: dict = {}
    t0 = time.time()
    done = 0

    async def one(spec: EpisodeSpec, branch: dict | None):
        nonlocal done
        async with sem:
            if stop.get("budget"):
                return
            rec, lab = await run_episode(spec, policy, stop=stop, branch=branch)
            write_jsonl(ep_path, [rec], mode="a")
            write_jsonl(lab_path, [lab], mode="a")
            done += 1
            es = episode_summary(rec)
            print(f"[{done}/{len(jobs)}] {rec['episode_id']} turns {es['turns']} "
                  f"cost ${es['cost']:.4f} wall {rec['wall_s']}s"
                  + (f" STOPPED {rec['stopped']}" if rec["stopped"] else ""), flush=True)

    await asyncio.gather(*(one(s, b) for s, b in jobs))
    wall = time.time() - t0

    eps, labs = load_run(ep_path)
    rows = [row for r in eps for row in evaluate(r, labs.get(r["episode_id"]))]
    summary = summarize_by(rows, "task_type", n_boot=args.boot)
    ep_sums = [episode_summary(r) for r in eps]
    metrics = {"tag": args.tag, "args": vars(args), "wall_s": round(wall, 1),
               "n_episodes": len(eps), "summary": summary, "episodes": ep_sums,
               "cost_usd": sum(e["cost"] for e in ep_sums),
               "llm_summary": llm.summary() if llm else "scripted (no LLM calls)"}
    mp = OUT / f"{args.tag}.metrics.json"
    mp.write_text(json.dumps(metrics, indent=1, default=str))
    live = llm.spent_usd if llm else 0.0
    metrics["live_cost_usd"] = live
    mp.write_text(json.dumps(metrics, indent=1, default=str))
    print(f"\n== {args.tag}: {len(eps)} episodes, wall {wall:.0f}s, nominal cost "
          f"${metrics['cost_usd']:.4f} (cached calls included), spent in this run ${live:.4f}")
    print(key_table(summary))
    print(f"chains {sum(e['chains'] for e in ep_sums)}, run-output chains "
          f"{sum(e['run_chains'] for e in ep_sums)}, shared files "
          f"{sum(e['shared_files'] for e in ep_sums)}, requests "
          f"{sum(e['requests'] for e in ep_sums)}, answers {sum(e['answers'] for e in ep_sums)}")
    print(f"wrote {ep_path.relative_to(ROOT)}, {lab_path.name}, {mp.name}")
    print(metrics["llm_summary"])
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["single", "fleet"], default="single")
    ap.add_argument("--n", type=int, default=8, help="agents per fleet episode")
    ap.add_argument("--p", type=float, default=0.3, help="share of impossible tasks")
    ap.add_argument("--channel", choices=["cache", "notes", "docs_sanctioned", "none"],
                    default=None, help="default: none (single), cache (fleet)")
    ap.add_argument("--prompt", choices=list(prompts.VARIANTS), default="plain")
    ap.add_argument("--rule", choices=["strict", "no_rule"], default="strict",
                    help="README rule arm (prompt allow_sharing sets the allow_sharing rule)")
    ap.add_argument("--model", default="adaption/deepseek-v4-flash",
                    help='"adaption/<id>", "oai:<base_url>|<model>", or "scripted:<style>"')
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--repeats", type=int, default=1, help="episodes per seed (new samples)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--conc", type=int, default=16, help="LLM concurrency (capped at 32)")
    ap.add_argument("--episode-conc", type=int, default=None)
    ap.add_argument("--budget", type=float, default=5.0, help="global LLM spend cap, USD")
    ap.add_argument("--schedule", choices=["rounds", "async"], default="rounds")
    ap.add_argument("--k", type=int, default=4, help="items per agent")
    ap.add_argument("--shard", type=int, default=1, help="own items with a token in a data shard")
    ap.add_argument("--missing", type=int, default=1,
                    help="named shards absent from data/ (moved to archive/ in solvable tasks)")
    ap.add_argument("--decoys", type=int, default=4)
    ap.add_argument("--old-shards", type=int, default=2, help="decoy shards in archive/")
    ap.add_argument("--busy", type=float, default=0.2, help="chance a stage run is busy")
    ap.add_argument("--peer-frac", type=float, default=0.5)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--branch-from", default=None)
    ap.add_argument("--branch-when", choices=["giveup", "silent_partial"], default="giveup")
    sys.exit(asyncio.run(main(ap.parse_args())))
