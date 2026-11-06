"""Aggregate fleet/single runs from results/raw/fleet/<tag>.jsonl with Wilson and episode-
bootstrap intervals. Metrics come from the provenance log; labels are joined only here.

    uv run python experiments/fleet_metrics.py single_plain single_persist
    uv run python experiments/fleet_metrics.py fleet8_plain fleet8_share --by task_type
    uv run python experiments/fleet_metrics.py --diff fleet8_share fleet8_plain \
        --metric uses_peer_materials --subset impossible
    uv run python experiments/fleet_metrics.py fleet8_plain --llm-label adaption/deepseek-v4-flash
    uv run python experiments/fleet_metrics.py --compact smoke_a_plain_rule smoke_d_allow_sharing
    uv run python experiments/fleet_metrics.py --blocked-table bl_ds_a_plain_rule bl_ds_b_coord
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from swarm.fleet.metrics import episode_summary, evaluate, llm_label_texts, summarize_by
from swarm.fleet.runner import load_run
from swarm.fleet.stats import boot_diff

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "raw" / "fleet"
SHOW_P = ("listed_shared", "uses_peer_materials", "uses_peer_materials_home",
          "uses_peer_materials_shared",
          "rule_broken", "rule_kept", "notes_for_peers", "peer_note_read", "note_read_by_peer",
          "note_exchange", "help_request", "help_reply", "team_post", "team_read",
          "team_peer_read", "peer_interaction", "seed_note_read", "seed_reply", "seed_copy",
          "help_request_after_seed", "notes_for_peers_after_seed", "self_notes_home",
          "self_notes_shared", "self_note_read_by_peer", "sanctioned_any",
          "sanctioned_read_other", "submitted", "full_success", "gave_up_early",
          "report_blocked", "honest_report", "verified_honest", "false_alarm", "silent_partial",
          "malformed_any")
SHOW_M = ("score_frac", "submit_step", "steps_used", "peer_reads", "rule_breaks", "reports",
          "malformed")


def rows_for(tag: str) -> tuple[list[dict], list[dict]]:
    eps, labs = load_run(OUT / f"{tag}.jsonl")
    rows = [row for r in eps for row in evaluate(r, labs.get(r["episode_id"]))]
    return eps, rows


def print_summary(tag: str, summary: dict) -> None:
    for group, s in summary.items():
        print(f"\n== {tag} [{group}] agents {s['n_agents']}, episodes {s['n_episodes']}")
        for k in SHOW_P:
            if k in s:
                v = s[k]
                print(f"  {k:<26} {v['k']:>4}/{v['n']:<4} = {v['p']:.3f}  Wilson "
                      f"[{v['wilson'][0]:.3f}, {v['wilson'][1]:.3f}]  boot({v['boot_unit']}) "
                      f"[{v['boot'][0]:.3f}, {v['boot'][1]:.3f}]")
        for k in SHOW_M:
            if k in s:
                v = s[k]
                print(f"  {k:<26} mean {v['mean']:.3f}  boot [{v['boot'][0]:.3f}, "
                      f"{v['boot'][1]:.3f}]  n {v['n']}")


COMPACT = ("uses_peer_materials", "rule_broken", "notes_for_peers", "help_request",
           "help_reply", "seed_note_read", "seed_reply", "help_request_after_seed",
           "notes_for_peers_after_seed", "full_success", "report_blocked", "silent_partial")


def compact_table(tags: list[str]) -> str:
    """One markdown row per tag: k/n of the main measures over all players ("-" = not
    applicable), turns, malformed and re-asked turns, blocked team-tool calls, nominal cost."""
    head = ["tag", "players", *COMPACT, "turns", "malformed", "re-asked", "team calls blocked",
            "cost $"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for tag in tags:
        eps, rows = rows_for(tag)
        es = [episode_summary(r) for r in eps]
        cells = [tag, str(len(rows))]
        for k in COMPACT:
            vals = [r.get(k) for r in rows if r.get(k) is not None]
            cells.append(f"{int(sum(vals))}/{len(vals)}" if vals else "-")
        cells += [str(sum(e[k] for e in es)) for k in ("turns", "malformed", "reasked",
                                                        "team_tool_calls_blocked")]
        cells.append(f"{sum(e['cost'] for e in es):.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


BLOCKED_COLS = (("listed cache", "listed_shared"), ("peer materials", "uses_peer_materials"),
                ("notes for peers", "notes_for_peers"),
                ("notes read by others", "note_read_by_peer"),
                ("help requests", "help_request"), ("help replies", "help_reply"),
                ("report_blocked", "report_blocked"), ("solved", "full_success"))


def blocked_table(tags: list[str], extra: tuple[str, ...] = (), *, base: bool = True) -> str:
    """One markdown row per tag over the blocked players only (task impossible alone; every
    player with --blocked all): k/n of the main measures, then the nominal cost of those
    players' turns (cached calls included) and the live cost of the run (calls not served from
    the cache). `extra` adds more 0/1 measures by name; base=False shows only those."""
    cols = [*(BLOCKED_COLS if base else ()), *((k, k) for k in extra)]
    head = ["tag", "model", "blocked", *(c for c, _ in cols), "cost $", "live $"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for tag in tags:
        eps, rows = rows_for(tag)
        rows = [r for r in rows if r.get("task_type") == "impossible"]
        models = sorted({r["model"] for r in rows}) or sorted({e["spec"]["model"] for e in eps})
        cells = [tag, ",".join(m.removeprefix("adaption/") for m in models), str(len(rows))]
        for _, k in cols:
            vals = [r.get(k) for r in rows if r.get(k) is not None]
            cells.append(f"{int(sum(vals))}/{len(vals)}" if vals else "-")
        cells.append(f"{sum(r.get('cost', 0.0) for r in rows):.4f}")
        live = sum(t.get("cost", 0.0) or 0.0 for e in eps for t in e["turns"]
                   if not t.get("cached") and not t.get("scripted"))
        cells.append(f"{live:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


COUNT_COLS = ("read_readme", "submitted", "rule_break_read", "rule_break_use", "coord_note_reads",
              "team_tool_calls_blocked", "self_notes_home_n", "self_notes_shared_n", "turns",
              "malformed")


def blocked_counts(tags: list[str]) -> str:
    """One markdown row per tag over the blocked players: sums of per-player counts
    (read_readme and submitted are players; rule_break_read counts reads of other players'
    content outside the allowed places, coord_note_reads reads of notes in /work/cache that the
    coordinate_ok README allows), and the mean score_frac."""
    head = ["tag", *COUNT_COLS, "mean score_frac"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for tag in tags:
        _, rows = rows_for(tag)
        rows = [r for r in rows if r.get("task_type") == "impossible"]
        cells = [tag, *(str(int(sum(r.get(k) or 0 for r in rows))) for k in COUNT_COLS)]
        cells.append(f"{sum(r['score_frac'] for r in rows) / len(rows):.3f}" if rows else "-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(args) -> int:
    out: dict = {}
    if args.compact:
        print(compact_table(args.tags))
        return 0
    if args.blocked_table:
        print(blocked_table(args.tags, tuple(args.extra)))
        if args.counts:
            print("\n" + blocked_counts(args.tags))
        return 0
    for tag in args.tags:
        eps, rows = rows_for(tag)
        if args.subset:
            rows = [r for r in rows if r.get("task_type") == args.subset]
        summary = summarize_by(rows, args.by, n_boot=args.boot) if args.by != "none" else \
            {"all": summarize_by(rows, "task_type", n_boot=args.boot)["all"]}
        es = [episode_summary(r) for r in eps]
        out[tag] = {"summary": summary, "episodes": es}
        print_summary(tag, summary)
        print(f"  episode totals: note chains {sum(e['chains'] for e in es)}, run-output chains "
              f"{sum(e['run_chains'] for e in es)}, shared files "
              f"{sum(e['shared_files'] for e in es)}, help requests "
              f"{sum(e['help_requests'] for e in es)}, help replies "
              f"{sum(e['help_replies'] for e in es)}, seed readers "
              f"{sum(e['seed_note_readers'] for e in es)}, seed replies "
              f"{sum(e['seed_replies'] for e in es)}, nominal cost "
              f"${sum(e['cost'] for e in es):.4f}")
        if args.llm_label:
            from swarm.llm import LLM
            texts = {f"{r['episode_id']}:{p}": v["text"] for r in eps
                     for p, v in r.get("shared_files", {}).items()}
            if texts:
                llm = LLM(budget_usd=args.budget, max_concurrency=8)
                labels = asyncio.run(llm_label_texts(texts, llm, args.llm_label))
                out[tag]["llm_labels"] = labels
                counts = {k: sum(bool(v.get(k)) for v in labels.values())
                          for k in ("addresses_others", "requests_help", "offers_or_answers",
                                    "proposes_division", "sets_convention")}
                print(f"  LLM labels over {len(labels)} shared texts: {counts}")
                print(f"  {llm.summary()}")
            else:
                print("  no shared texts to label")
    if args.diff:
        a, b = args.diff
        _, ra = rows_for(a)
        _, rb = rows_for(b)
        if args.subset:
            ra = [r for r in ra if r.get("task_type") == args.subset]
            rb = [r for r in rb if r.get("task_type") == args.subset]
        d = boot_diff([r.get(args.metric) for r in ra], [r["episode_id"] for r in ra],
                      [r.get(args.metric) for r in rb], [r["episode_id"] for r in rb],
                      n_boot=args.boot)
        out["diff"] = {"a": a, "b": b, "metric": args.metric, "subset": args.subset, **d}
        print(f"\n== diff {args.metric} ({args.subset or 'all'}): {a} - {b} = {d['diff']:.3f} "
              f"[{d['lo']:.3f}, {d['hi']:.3f}], one-sided p(A<=B) {d['p_one_sided']:.4f}, "
              f"clusters {d.get('n_clusters')}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1, default=str))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--by", default="task_type", help="row field to group by, or 'none'")
    ap.add_argument("--subset", choices=["solvable", "impossible"], default=None)
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"), default=None)
    ap.add_argument("--metric", default="peer_interaction")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--llm-label", default=None, help="model for optional LLM labelling")
    ap.add_argument("--budget", type=float, default=6.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compact", action="store_true",
                    help="one markdown row per tag with the main measures (all players)")
    ap.add_argument("--blocked-table", action="store_true",
                    help="one markdown row per tag over the blocked players only")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="--blocked-table: more 0/1 measures to add as columns")
    ap.add_argument("--counts", action="store_true",
                    help="--blocked-table: also print summed counts over the blocked players")
    sys.exit(main(ap.parse_args()))
