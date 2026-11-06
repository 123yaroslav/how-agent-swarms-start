"""Feasibility t2, task 4c: stored [Admin1] saves vs deletions on the same page (treatment-side timing only). Read-only."""
import json, collections
from datetime import datetime, timezone
from pathlib import Path
D = Path("data/raw/collusion_wiki/x")
P = lambda s: datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
revs = [json.loads(l) for l in open(D / "revisions.jsonl")]
ev = [json.loads(l) for l in open(D / "events.jsonl")]
dels = collections.defaultdict(list)
for e in ev:
    if e["event_type"] == "delete": dels[e["page_key"]].append(P(e["time"]))
adm = [r for r in revs if r["label"] == "[Admin1]"]
rows = []
for r in adm:
    t = P(r["time"]); nxt = [d for d in dels.get(r["page_key"], []) if d >= t]
    lag = (min(nxt) - t).total_seconds() if nxt else None
    rows.append((r["time"], r["body_len"], lag, r["change_summary"][:20] if r["change_summary"] else ""))
for x in sorted(rows): print("  admin save", x[0], "body_len", x[1], "-> next delete on page after (s):", x[2], "| summary:", x[3])
print("admin body_len==0:", sum(1 for r in adm if r["body_len"] == 0), "/", len(adm))
print("all revisions with body_len 0:", sum(1 for r in revs if r["body_len"] == 0), "| by label type:", collections.Counter("admin" if r["label"] == "[Admin1]" else "other" for r in revs if r["body_len"] == 0))
