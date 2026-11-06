"""Feasibility t2: print schema (keys, types, value examples) of collusion.wiki export files. Read-only."""
import json, sys, collections
from pathlib import Path
D = Path("data/raw/collusion_wiki/x")
def trunc(v, n=120):
    s = json.dumps(v, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "..."
for name in ["events.jsonl", "pages.jsonl", "labels.jsonl", "revisions.jsonl", "records.jsonl", "links.jsonl"]:
    keys = collections.Counter(); ex = {}
    n = 0
    with open(D / name) as f:
        for line in f:
            r = json.loads(line); n += 1
            for k, v in r.items():
                keys[k] += 1
                if k not in ex and v not in (None, "", [], {}):
                    ex[k] = v
    print(f"== {name} rows={n}")
    for k, c in keys.items():
        print(f"   {k}: {c}  ex={trunc(ex.get(k))}")
