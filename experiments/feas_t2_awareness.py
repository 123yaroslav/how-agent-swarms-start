"""Feasibility t2, task 5: awareness check. Count revisions (by day) whose ADDED text mentions deletion/moderation
words, and records (other venues) by source date where parseable. No split by treatment status. Read-only."""
import json, collections, re
from pathlib import Path
D = Path("data/raw/collusion_wiki/x")
T = {"delete/deleted/deletion": r"\bdelet(?:e|ed|es|ion|ions|ing)\b", "gelöscht/löschen": r"gel(?:ö|oe|o)scht|\bl(?:ö|oe)sch\w*",
     "moderator/moderation": r"\bmoderat(?:or|ors|ion)\b", "admin (word)": r"\badmins?\b", "removed": r"\bremoved\b", "wiped": r"\bwiped\b"}
T = {k: re.compile(v, re.I) for k, v in T.items()}
def added(r):
    B = r["body"].splitlines(); o = []
    for h in r["hunks"]:
        if h["op"] in ("insert", "replace"): o += B[h["b0"]:h["b1"]]
    return "\n".join(o)
revs = [json.loads(l) for l in open(D / "revisions.jsonl")]
def bucket(d):
    if d < "2026-06-16": return "<=06-15"
    if d > "2026-06-22": return ">=06-23"
    return d[5:]
cnt = collections.defaultdict(collections.Counter); tot = collections.Counter(); anyhit = collections.Counter()
for r in revs:
    if r["label"] in ("[Admin1]",): continue
    b = bucket(r["time"][:10]); tot[b] += 1; a = added(r); hit = False
    for k, rx in T.items():
        if rx.search(a): cnt[b][k] += 1; hit = True
    anyhit[b] += hit
order = ["<=06-15", "06-16", "06-17", "06-18", "06-19", "06-20", "06-21", "06-22", ">=06-23"]
print("WIKI revisions (added text), excluding [Admin1]:")
print("day | saves | any | " + " | ".join(T))
for b in order: print(f"{b} | {tot[b]} | {anyhit[b]} | " + " | ".join(str(cnt[b][k]) for k in T))
recs = [json.loads(l) for l in open(D / "records.jsonl")]
lits = collections.Counter()
rx_date = re.compile(r"(2026)-(\d\d)-(\d\d)")
rc = collections.defaultdict(collections.Counter); rt = collections.Counter(); ra = collections.Counter(); undated = 0
for r in recs:
    ds = [o.get("source_date_literal") for o in r["origins"] if o.get("source_date_literal")]
    m = next((rx_date.search(str(d)) for d in ds if rx_date.search(str(d))), None)
    if not m: undated += 1; lits[str(ds[:1])[:25]] += 1; continue
    d = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"; b = bucket(d); rt[b] += 1; hit = False
    for k, rx2 in T.items():
        if rx2.search(r["text"]): rc[b][k] += 1; hit = True
    ra[b] += hit
print(f"RECORDS (other venues): dated {len(recs) - undated}, undated {undated}; example undated literals {lits.most_common(3)}")
print("day | records | any | " + " | ".join(T))
for b in order: print(f"{b} | {rt[b]} | {ra[b]} | " + " | ".join(str(rc[b][k]) for k in T))
