"""Feasibility t2, task 3 diagnostics on the labelled edges only: is low similarity a matcher artifact?
Alternative representations: char 5-grams, line overlap, containment both directions, sizes. Read-only."""
import json, collections, re, statistics, difflib
from datetime import datetime, timezone
from pathlib import Path
D = Path("data/raw/collusion_wiki/x")
P = lambda s: datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
revs = [json.loads(l) for l in open(D / "revisions.jsonl")]
ev = [json.loads(l) for l in open(D / "events.jsonl")]
by_id = {r["rev_id"]: r for r in revs}
by_page = collections.defaultdict(list)
for r in revs: by_page[r["page_key"]].append(r)
for k in by_page: by_page[k].sort(key=lambda r: (r["time"], r["seq"]))
dels = {e["event_id"]: e for e in ev if e["event_type"] == "delete"}
norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
def c5(s): s = norm(s); return {s[i:i+5] for i in range(max(1, len(s) - 4))}
def jac(a, b): return len(a & b) / len(a | b) if a and b else 0.0
rows = []
for e in ev:
    if e.get("relation_type") != "first_recreation_of" or e["event_type"] != "save": continue
    for d in e["related_event_id"]:
        t = P(dels[d]["time"]); pre = [r for r in by_page[e["page_key"]] if P(r["time"]) < t]
        if not pre: continue
        pre = pre[-1]; tgt = by_id[e["revision_ref"]]
        A, B = c5(pre["body"]), c5(tgt["body"])
        la, lb = set(map(norm, pre["body"].splitlines())) - {""}, [norm(x) for x in tgt["body"].splitlines() if x.strip()]
        line_from_old = sum(x in la for x in lb) / len(lb) if lb else 0
        sm = difflib.SequenceMatcher(None, pre["body"], tgt["body"], autojunk=False).ratio() if len(pre["body"]) + len(tgt["body"]) < 60000 else None
        same_label = pre["label"] == tgt["label"] and pre["label"]
        rows.append(dict(c5=jac(A, B), cont_old_in_new=len(A & B) / len(A), cont_new_in_old=len(A & B) / len(B), line_from_old=line_from_old, sm=sm,
                         pre_len=pre["body_len"], tgt_len=tgt["body_len"], same_label=bool(same_label), pre=pre, tgt=tgt))
med = lambda k: round(statistics.median([r[k] for r in rows if r[k] is not None]), 3)
print("n", len(rows))
for k in ("c5", "cont_old_in_new", "cont_new_in_old", "line_from_old", "sm", "pre_len", "tgt_len"):
    print(k, "median", med(k))
for th in (0.3, 0.5):
    print(f"char5 Jaccard>={th}: {sum(r['c5'] >= th for r in rows)} | old-in-new containment>={th}: {sum(r['cont_old_in_new'] >= th for r in rows)} | new-in-old>={th}: {sum(r['cont_new_in_old'] >= th for r in rows)} | lines from old>={th}: {sum(r['line_from_old'] >= th for r in rows)}")
print("same label as deleted revision:", sum(r["same_label"] for r in rows))
print("max over representations >=0.5:", sum(max(r['c5'], r['cont_old_in_new'], r['line_from_old']) >= 0.5 for r in rows))
# three redacted low-similarity examples: first 100 chars only, digits masked
mask = lambda s: re.sub(r"\d", "#", s[:100]).replace("\n", " | ")
for r in sorted(rows, key=lambda r: r["c5"])[:2] + sorted(rows, key=lambda r: r["c5"])[30:31]:
    print("EX c5=%.2f pre_len=%d tgt_len=%d\n   OLD: %s\n   NEW: %s" % (r["c5"], r["pre_len"], r["tgt_len"], mask(r["pre"]["body"]), mask(r["tgt"]["body"])))
