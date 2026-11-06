"""Run single-agent or fleet episodes of the benign data-assembly world (src/swarm/fleet).

Writes results/raw/fleet/<tag>.jsonl (one episode record per line), <tag>.labels.jsonl (answer
keys, read only by evaluation) and <tag>.metrics.json (summary by task type). Rerunnable from
the LLM cache: the default "rounds" schedule is deterministic, so a rerun replays every call.

    # 10 single-agent episodes, plain prompt, half of them impossible
    uv run python experiments/run_fleet.py --mode single --p 0.5 --prompt plain \
        --model adaption/deepseek-v4-flash --seeds 0-9 --tag single_plain
    # fleet of 8, 30% impossible tasks, shared cache, house rule, 30 turns
    uv run python experiments/run_fleet.py --mode fleet --n 8 --p 0.3 --channel cache \
        --prompt plain --rule rule --steps 30 --seeds 100-100 --tag fleet8_plain
    # same world with player-99's seed note asking for a named data shard
    uv run python experiments/run_fleet.py --mode fleet --n 8 --steps 30 --seeds 100-100 \
        --seed-note request --tag fleet8_seed_request
    # positive control: README allows using other players' files
    uv run python experiments/run_fleet.py --mode fleet --n 8 --p 0.3 --channel cache \
        --prompt allow_sharing --seeds 0-0 --tag fleet8_share
    # every player blocked (one missing file each, held by another player in a ring)
    uv run python experiments/run_fleet.py --mode fleet --n 8 --steps 40 --seeds 200-200 \
        --blocked all --prompt coordinate_ok --tag fleet8_all_coord
    # free dry run with scripted agents
    uv run python experiments/run_fleet.py --mode fleet --n 8 --model scripted:peer_reader \
        --tag dry
    # branch: continue every gave-up-early episode of a tag from its give-up point
    uv run python experiments/run_fleet.py --mode single --branch-from single_plain \
        --branch-when giveup --prompt persist --tag single_plain_branch

--seeds accepts "a-b" (inclusive), "a,b,c", or a bare count N (= 0..N-1).

Pairing: arms run on the same world seeds with the same --sample-offset share LLM samples (the
sample index is seed, repeat, player and turn, not the arm), so they are paired and analysis must
treat them as paired. --sample-offset K uses repeat indices K..K+R-1 and so draws samples that
are independent of runs with other offsets.
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
from swarm.fleet.metrics import episode_summary, evaluate, outcomes, summarize, summarize_by
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
        for k in ("listed_shared", "uses_peer_materials", "uses_peer_materials_home",
                  "uses_peer_materials_shared", "rule_broken", "rule_kept", "notes_for_peers",
                  "peer_note_read", "note_read_by_peer", "help_request", "help_reply",
                  "team_post", "team_read", "team_peer_read", "peer_interaction",
                  "seed_note_read", "seed_reply", "seed_copy", "help_request_after_seed",
                  "notes_for_peers_after_seed", "self_notes_home", "self_notes_shared",
                  "self_note_read_by_peer", "sanctioned_any", "submitted", "full_success",
                  "gave_up_early", "report_blocked", "honest_report", "false_alarm",
                  "silent_partial", "malformed_any"):
            if k in s:
                v = s[k]
                lines.append(f"    {k:<26} {v['k']}/{v['n']} = {v['p']:.2f}  Wilson "
                             f"[{v['wilson'][0]:.2f}, {v['wilson'][1]:.2f}]")
        for k in ("score_frac", "submit_step", "steps_used", "peer_reads", "rule_breaks",
                  "malformed"):
            if k in s:
                v = s[k]
                lines.append(f"    {k:<26} mean {v['mean']:.2f}  boot [{v['boot'][0]:.2f}, "
                             f"{v['boot'][1]:.2f}] (n {v['n']})")
    return "\n".join(lines)


def make_policy(args, llm, inflight: dict | None = None):
    if args.model.startswith("scripted:"):
        return FnPolicy(ScriptedAgent(args.model.split(":", 1)[1]))
    return LLMPolicy(llm, args.model, max_tokens=args.max_tokens, temperature=args.temperature,
                     inflight=inflight)


def live_cost(eps: list[dict]) -> float:
    """Spend of the calls that were not served from the cache (or by a joined call)."""
    return sum(t.get("cost", 0.0) or 0.0 for r in eps for t in r["turns"]
               if not t.get("cached") and not t.get("scripted") and not t.get("replayed"))


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


def concurrency(requested: int) -> int:
    free = unused_mb()
    conc = requested
    if free is not None and free < 300:
        conc = min(conc, 16)
    conc = min(conc, 32)
    print(f"unused memory {free} MB -> concurrency {conc}", flush=True)
    return conc


async def main(args) -> int:
    conc = concurrency(args.conc)
    llm = None
    if not args.model.startswith("scripted:"):
        from swarm.llm import LLM
        llm = LLM(budget_usd=args.budget, max_concurrency=conc)
    metrics = await run_tag(args, llm, conc=conc)
    print(metrics["llm_summary"])
    return 0


async def run_tag(args, llm, *, conc: int = 16, inflight: dict | None = None) -> dict:
    """Run one tag (all its episodes), write its files and print its tables. Several tags may
    run side by side in one process with one LLM; pass them one shared `inflight` dict."""
    policy = make_policy(args, llm, inflight)
    world_kw = {"k_items": args.k, "n_shard": args.shard, "n_missing": args.missing,
                "n_decoys": args.decoys, "n_old_shards": args.old_shards,
                "busy_prob": args.busy, "peer_frac": args.peer_frac}
    if args.blocked != "p":  # left out at the default, as in earlier specs
        world_kw["blocked"] = args.blocked
    channel = args.channel or ("none" if args.mode == "single" else "cache")
    jobs: list[tuple[EpisodeSpec, dict | None]] = []
    if args.branch_from:
        for src, at in branch_sources(args.branch_from, args.branch_when):
            s = dict(src["spec"])
            if s.get("rule") == "strict":  # earlier name of the house-rule arm
                s["rule"] = "rule"
            s.update(tag=args.tag, prompt=args.prompt, model=args.model,
                     max_tokens=args.max_tokens, temperature=args.temperature)
            if args.prompt in ("allow_sharing", "coordinate_ok"):
                raise SystemExit("branching is for single-agent episodes")
            jobs.append((EpisodeSpec(**s), {"from": src, "at": at}))
        print(f"branching {len(jobs)} episodes of {args.branch_from} ({args.branch_when})")
    else:
        for seed in parse_seeds(args.seeds):
            for rep in range(args.sample_offset, args.sample_offset + args.repeats):
                jobs.append((EpisodeSpec(
                    tag=args.tag, mode=args.mode, seed=seed,
                    n_agents=1 if args.mode == "single" else args.n, p=args.p, channel=channel,
                    prompt=args.prompt, rule=args.rule, seed_note=args.seed_note,
                    team_tool=args.team_tool, model=args.model, steps=args.steps,
                    max_tokens=args.max_tokens, temperature=args.temperature, repeat=rep,
                    schedule=args.schedule, world=world_kw,
                    history_window=args.history_window), None))
    for spec, _ in jobs[:1]:
        spec.world_config().validate()
        prompts.system_prompt("agent-00", spec.steps, channel=spec.channel, variant=spec.prompt,
                              team_tool=spec.team_tool)

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
            print(f"[{args.tag} {done}/{len(jobs)}] {rec['episode_id']} turns {es['turns']} "
                  f"cost ${es['cost']:.4f} wall {rec['wall_s']}s"
                  + (f" STOPPED {rec['stopped']}" if rec["stopped"] else ""), flush=True)

    await asyncio.gather(*(one(s, b) for s, b in jobs))
    wall = time.time() - t0

    eps, labs = load_run(ep_path)
    rows = [row for r in eps for row in evaluate(r, labs.get(r["episode_id"]))]
    summary = summarize_by(rows, "task_type", n_boot=args.boot)
    blocked_rows = [r for r in rows if r.get("task_type") == "impossible"]
    if blocked_rows:  # players blocked alone (every player with --blocked all)
        summary["blocked"] = summarize(blocked_rows, n_boot=args.boot)
    ep_sums = [episode_summary(r) for r in eps]
    live = live_cost(eps)
    metrics = {"tag": args.tag, "args": vars(args), "wall_s": round(wall, 1),
               "n_episodes": len(eps), "summary": summary, "episodes": ep_sums,
               "cost_usd": sum(e["cost"] for e in ep_sums), "live_cost_usd": live,
               "sampling": {"seeds": parse_seeds(args.seeds) if not args.branch_from else None,
                            "repeats": [args.sample_offset, args.sample_offset + args.repeats - 1],
                            "paired_with": "runs on the same seeds and repeat indices"},
               "llm_summary": llm.summary() if llm else "scripted (no LLM calls)"}
    mp = OUT / f"{args.tag}.metrics.json"
    mp.write_text(json.dumps(metrics, indent=1, default=str))
    print(f"\n== {args.tag}: {len(eps)} episodes, wall {wall:.0f}s, nominal cost "
          f"${metrics['cost_usd']:.4f} (cached calls included), spent in this run ${live:.4f}")
    print(key_table(summary))

    def tot(k: str) -> int:
        return sum(e[k] for e in ep_sums)

    print(f"episode totals: note chains {tot('chains')}, run-output chains {tot('run_chains')}, "
          f"shared files {tot('shared_files')}, help requests {tot('help_requests')}, "
          f"help replies {tot('help_replies')}, turns {tot('turns')}, malformed "
          f"{tot('malformed')}, re-asked {tot('reasked')}")
    if any(e["seed_note"] for e in ep_sums):
        print(f"seed note ({args.seed_note}): readers {tot('seed_note_readers')}, replies "
              f"{tot('seed_replies')}, copies {tot('seed_copies')}, help requests after "
              f"reading it {tot('help_requests_after_seed')}, notes for peers after reading it "
              f"{tot('notes_after_seed')}")
    if args.team_tool != "off":
        print(f"team channel ({args.team_tool}): posts {tot('team_posts')}, reads "
              f"{tot('team_reads')}, posts read by peers {tot('team_chains')}, blocked team-tool "
              f"calls {tot('team_tool_calls_blocked')}")
    if args.history_window:
        print(f"history window {args.history_window}: self-notes home "
              f"{tot('self_notes_home')}, shared {tot('self_notes_shared')}")
    shown = ep_path.relative_to(ROOT) if ep_path.is_relative_to(ROOT) else ep_path
    print(f"wrote {shown}, {lab_path.name}, {mp.name}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["single", "fleet"], default="single")
    ap.add_argument("--n", type=int, default=8, help="agents per fleet episode")
    ap.add_argument("--p", type=float, default=0.3, help="share of impossible tasks")
    ap.add_argument("--channel", choices=["cache", "notes", "docs_sanctioned", "none"],
                    default=None, help="default: none (single), cache (fleet)")
    ap.add_argument("--prompt", choices=list(prompts.VARIANTS), default="plain")
    ap.add_argument("--rule", choices=["rule", "no_rule"], default="rule",
                    help="house-rule arm: rule keeps the house-rule lines in /work/README.md, "
                         "no_rule drops them (prompts allow_sharing and coordinate_ok set their "
                         "own README rule)")
    ap.add_argument("--seed-note", choices=["none", "request", "offer", "memory"],
                    default="none",
                    help="fleet only: a note from player-99 in the shared place at t=0")
    ap.add_argument("--team-tool", choices=["off", "on", "disabled_notice"], default="off",
                    help="fleet only: team channel tools team_post/team_read (on), absent (off), "
                         "or absent with a README notice that it is not enabled")
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
    ap.add_argument("--budget", type=float, default=6.0, help="global LLM spend cap, USD")
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
    ap.add_argument("--blocked", choices=["p", "all"], default="p",
                    help="fleet only: p = a share --p of players is blocked alone; all = every "
                         "player lacks exactly one named input file, held by another player "
                         "(ring in random order)")
    ap.add_argument("--history-window", type=int, default=None,
                    help="players see only their last K turns (earlier turns hidden)")
    ap.add_argument("--sample-offset", type=int, default=0,
                    help="first repeat index; same seeds and offset = paired samples across "
                         "arms, another offset = independent samples")
    return ap


if __name__ == "__main__":
    sys.exit(asyncio.run(main(build_parser().parse_args())))
