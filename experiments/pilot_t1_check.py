"""PILOT (throwaway, environments web_scraping and media_processing only).

Agreement of the LLM attack-action labels with the straj `incriminating_actions` field, and a
compact view of chosen Haiku attack trajectories for hand checking (short excerpts only).

    uv run python -I experiments/pilot_t1_check.py                 # agreement table
    uv run python -I experiments/pilot_t1_check.py --show 3 --n 5  # view 5 Haiku attacks
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw" / "pilot_t1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels_deepseek-v4-flash.json")
    ap.add_argument("--show", type=int, default=0, help="seed for picking trajectories to show")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    lab = {r["id"]: r for r in json.loads((RAW / args.labels).read_text()) if "error" not in r}
    trajs = {t["id"]: t for t in map(json.loads, (RAW / "trajs.jsonl").open())}
    st = [r for r in lab.values() if r["incriminating"] is not None]
    inc = sum(len(set(r["incriminating"])) for r in st)
    hit_core = sum(len(set(r["incriminating"]) & set(r["core"])) for r in st)
    hit_any = sum(len(set(r["incriminating"]) & set(r["core"] + r["support"])) for r in st)
    full = sum(set(r["incriminating"]) <= set(r["core"] + r["support"]) for r in st)
    extra = sum(len(set(r["core"] + r["support"]) - set(r["incriminating"])) for r in st)
    print(f"straj trajectories {len(st)}, incriminating actions {inc}")
    print(f"  recall core {hit_core}/{inc}, recall core+support {hit_any}/{inc}")
    print(f"  trajectories with every incriminating action labelled: {full}/{len(st)}")
    print(f"  labelled actions not in incriminating: {extra} "
          f"(incriminating lists only the key actions, so these are not all errors)")
    for fam in ("haiku_attack", "straj"):
        xs = [r for r in lab.values() if r["fam"] == fam]
        sizes = sorted(len(r["core"]) + len(r["support"]) for r in xs)
        print(f"{fam}: n={len(xs)} labelled actions per trajectory {sizes}; "
              f">=3 labelled: {sum(s >= 3 for s in sizes)}")
    if args.show:
        ha = sorted(r["id"] for r in lab.values() if r["fam"] == "haiku_attack")
        for tid in random.Random(args.show).sample(ha, min(args.n, len(ha))):
            t, r = trajs[tid], lab[tid]
            print(f"\n##### {tid} env={t['env']} side_task={t['side_task']} "
                  f"core={r['core']} support={r['support']}\nnote: {r['note']}")
            for i, a in enumerate(t["actions"]):
                mark = "C" if i in r["core"] else ("S" if i in r["support"] else " ")
                rs = a["reasoning"].replace("\n", " ")[:160] if a["reasoning"] != "None" else ""
                print(f" {mark} [{i}] {a['fn']} {a['args'][:110]!r} | {rs}")


if __name__ == "__main__":
    main()
