"""Aggregate fleet/single runs from results/raw/fleet/<tag>.jsonl with Wilson and episode-
bootstrap intervals. Metrics come from the provenance log; labels are joined only here.

    uv run python experiments/fleet_metrics.py single_plain single_persist
    uv run python experiments/fleet_metrics.py fleet8_plain fleet8_share --by task_type
    uv run python experiments/fleet_metrics.py --diff fleet8_share fleet8_plain \
        --metric cross_read --subset impossible
    uv run python experiments/fleet_metrics.py fleet8_plain --llm-label adaption/deepseek-v4-flash
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
SHOW_P = ("cross_read", "cross_read_home", "cross_read_shared", "violation_any", "shared_note",
          "chain", "req_ans", "swarm_any", "sanctioned_any", "sanctioned_read_other",
          "submitted", "full_success", "gave_up_early", "reported", "honest_report",
          "verified_honest", "false_alarm", "silent_partial", "malformed_any")
SHOW_M = ("score_frac", "submit_step", "steps_used", "read_other", "violations", "reports",
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
                print(f"  {k:<22} {v['k']:>4}/{v['n']:<4} = {v['p']:.3f}  Wilson "
                      f"[{v['wilson'][0]:.3f}, {v['wilson'][1]:.3f}]  boot({v['boot_unit']}) "
                      f"[{v['boot'][0]:.3f}, {v['boot'][1]:.3f}]")
        for k in SHOW_M:
            if k in s:
                v = s[k]
                print(f"  {k:<22} mean {v['mean']:.3f}  boot [{v['boot'][0]:.3f}, "
                      f"{v['boot'][1]:.3f}]  n {v['n']}")


def main(args) -> int:
    out: dict = {}
    for tag in args.tags:
        eps, rows = rows_for(tag)
        if args.subset:
            rows = [r for r in rows if r.get("task_type") == args.subset]
        summary = summarize_by(rows, args.by, n_boot=args.boot) if args.by != "none" else \
            {"all": summarize_by(rows, "task_type", n_boot=args.boot)["all"]}
        es = [episode_summary(r) for r in eps]
        out[tag] = {"summary": summary, "episodes": es}
        print_summary(tag, summary)
        print(f"  episode totals: chains {sum(e['chains'] for e in es)}, run-output chains "
              f"{sum(e['run_chains'] for e in es)}, shared files "
              f"{sum(e['shared_files'] for e in es)}, requests {sum(e['requests'] for e in es)}, "
              f"answers {sum(e['answers'] for e in es)}, nominal cost "
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
    ap.add_argument("--metric", default="swarm_any")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--llm-label", default=None, help="model for optional LLM labelling")
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--out", default=None)
    sys.exit(main(ap.parse_args()))
