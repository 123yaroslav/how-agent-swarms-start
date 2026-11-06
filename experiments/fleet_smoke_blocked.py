"""Smoke runs of the fleet game with every player blocked (machinery check, not results).

N = 8 players, 40 turns, `--blocked all` (each player lacks exactly one named input file, held
by another player in a ring), throwaway world seed 200, six arms on two models:

    a  plain + rule                       d  collab_demos + rule + team channel disabled_notice
    b  coordinate_ok (positive control)   e  plain + rule + seed note memory
    c  persist + rule                     f  plain + rule + history window 6

Models: adaption/deepseek-v4-flash (max_tokens 3000) and adaption/gemma-4-31b-it (2000). All
twelve tags run side by side in one process with one LLM client (concurrency 16, global spend
cap $6). Identical calls in flight are joined, so arms that share samples on the same seed stay
paired and every rerun replays from the cache. Prints one table over the blocked players, a
second one with more measures, a third with summed counts, and LLM.summary(). A rerun replays
every call from the cache, so its "live $" column is 0; the tables can also be re-printed from
the records with `fleet_metrics.py --blocked-table --counts --extra ... <tags>`.

    uv run python experiments/fleet_smoke_blocked.py               # all twelve tags
    uv run python experiments/fleet_smoke_blocked.py --only ds_a gm_a
    uv run python experiments/fleet_smoke_blocked.py --scripted     # free dry run
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run_fleet = _load("run_fleet")
fleet_metrics = _load("fleet_metrics")

COMMON = ["--mode", "fleet", "--n", "8", "--steps", "40", "--seeds", "200-200",
          "--blocked", "all", "--budget", "6"]
ARMS = {"a": ("plain_rule", ["--prompt", "plain", "--rule", "rule"]),
        "b": ("coordinate_ok", ["--prompt", "coordinate_ok"]),
        "c": ("persist_rule", ["--prompt", "persist", "--rule", "rule"]),
        "d": ("demos_rule_notice", ["--prompt", "collab_demos", "--rule", "rule",
                                    "--team-tool", "disabled_notice"]),
        "e": ("plain_rule_memory", ["--prompt", "plain", "--rule", "rule",
                                    "--seed-note", "memory"]),
        "f": ("plain_rule_window6", ["--prompt", "plain", "--rule", "rule",
                                     "--history-window", "6"])}
MODELS = {"ds": ("adaption/deepseek-v4-flash", 3000), "gm": ("adaption/gemma-4-31b-it", 2000)}
EXTRA = ("read_readme", "rule_broken", "seed_note_read", "self_notes_home", "self_notes_shared",
         "self_note_read_by_peer", "malformed_any")


def tags_and_args(scripted: bool) -> dict[str, list[str]]:
    out = {}
    for m, (model, max_tokens) in MODELS.items():
        for a, (name, extra) in ARMS.items():
            if scripted:
                model, tag = "scripted:honest", f"dry_bl_{m}_{a}_{name}"
            else:
                tag = f"bl_{m}_{a}_{name}"
            out[f"{m}_{a}"] = [*COMMON, *extra, "--model", model, "--max-tokens",
                               str(max_tokens), "--tag", tag]
    return out


async def main(opts) -> int:
    jobs = tags_and_args(opts.scripted)
    keys = opts.only or list(jobs)
    unknown = [k for k in keys if k not in jobs]
    if unknown:
        raise SystemExit(f"unknown keys {unknown}; choose from {list(jobs)}")
    parser = run_fleet.build_parser()
    args = [parser.parse_args(jobs[k]) for k in keys]
    conc = run_fleet.concurrency(opts.conc)
    llm = None
    if not opts.scripted:
        from swarm.llm import LLM
        llm = LLM(budget_usd=6.0, max_concurrency=conc)
    inflight: dict = {}
    log = io.StringIO()

    async def one(a):
        await run_fleet.run_tag(a, llm, conc=conc, inflight=inflight)
        print(f"done {a.tag}", file=sys.__stdout__, flush=True)

    # the per-tag progress lines and tables of concurrent tags interleave: they go to one log
    # file, and the tables below are rebuilt from the records
    with contextlib.redirect_stdout(log):
        await asyncio.gather(*(one(a) for a in args))
    out_log = run_fleet.OUT / ("dry_bl_smoke.log" if opts.scripted else "bl_smoke.log")
    out_log.write_text(log.getvalue())
    tags = [a.tag for a in args]
    print(f"per-tag tables: {out_log}")
    print("\nBlocked players only (every player is blocked with --blocked all):\n")
    print(fleet_metrics.blocked_table(tags))
    print("\nMore measures, blocked players:\n")
    print(fleet_metrics.blocked_table(tags, EXTRA, base=False))
    print("\nCounts over the blocked players:\n")
    print(fleet_metrics.blocked_counts(tags))
    if llm:
        print("\n" + llm.summary())
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None, help="keys such as ds_a gm_f")
    ap.add_argument("--scripted", action="store_true", help="scripted players, no LLM calls")
    ap.add_argument("--conc", type=int, default=16)
    sys.exit(asyncio.run(main(ap.parse_args())))
