"""Export fine-tuning data from single-agent runs (results/raw/fleet/<tag>.jsonl).

    # SFT on fully solved solvable tasks, one chat per row
    uv run python experiments/fleet_export.py single_persist --sft solved --out sft_persist.jsonl
    # DPO step pairs: persistent branch vs the give-up submit it replaces
    uv run python experiments/fleet_export.py single_plain single_plain_branch --dpo persist \
        --dpo-format step --out dpo_persist.jsonl
    # DPO trajectory pairs on the same world: honest report_blocked vs silent partial submit
    uv run python experiments/fleet_export.py single_plain single_escalate --dpo escalate \
        --dpo-format trajectory --out dpo_escalate.jsonl

Output goes to results/fleet_export/ unless --out is an absolute path. Every row is checked
against the schema, and episodes with traces of other agents or shared places are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from swarm.fleet.export import SFT_SELECT, dpo_pairs, sft_rows, validate_dpo_row, validate_sft_row
from swarm.fleet.runner import load_run, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw" / "fleet"
OUT = ROOT / "results" / "fleet_export"


def main(args) -> int:
    records, labels = [], {}
    for tag in args.tags:
        eps, labs = load_run(RAW / f"{tag}.jsonl")
        records += eps
        labels.update(labs)
    if bool(args.sft) == bool(args.dpo):
        print("choose exactly one of --sft or --dpo")
        return 2
    if args.sft:
        rows, skipped = sft_rows(records, labels, select=args.sft, fmt=args.sft_format,
                                 allow_malformed=args.allow_malformed,
                                 require_verified=not args.unverified,
                                 with_meta=not args.no_meta)
        for r in rows:
            validate_sft_row(r, args.sft_format)
    else:
        rows, skipped = dpo_pairs(records, labels, kind=args.dpo, fmt=args.dpo_format,
                                  max_per_world=args.max_per_world,
                                  allow_malformed=args.allow_malformed,
                                  require_verified=not args.unverified,
                                  chat_completions=args.chat_completions,
                                  with_meta=not args.no_meta)
        for r in rows:
            validate_dpo_row(r, args.dpo_format)
    out = Path(args.out) if Path(args.out).is_absolute() else OUT / args.out
    write_jsonl(out, rows)
    print(f"{len(records)} episodes in, {len(rows)} rows out -> {out}")
    print(f"skipped: {dict(Counter(why for _, why in skipped))}")
    if rows and args.show:
        print(json.dumps(rows[0], indent=1)[:3000])
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--sft", choices=SFT_SELECT, default=None)
    ap.add_argument("--sft-format", choices=["messages", "prompt_completion"], default="messages")
    ap.add_argument("--dpo", choices=["persist", "escalate"], default=None)
    ap.add_argument("--dpo-format", choices=["step", "trajectory"], default="step")
    ap.add_argument("--max-per-world", type=int, default=2)
    ap.add_argument("--chat-completions", action="store_true",
                    help="chosen/rejected as one-message chat lists instead of strings")
    ap.add_argument("--allow-malformed", action="store_true")
    ap.add_argument("--unverified", action="store_true",
                    help="escalation: do not require reading all own files before the report")
    ap.add_argument("--no-meta", action="store_true", help="drop the meta column")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--out", required=True)
    sys.exit(main(ap.parse_args()))
