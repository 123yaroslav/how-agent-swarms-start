"""Feasibility t2, task 3b: synthetic positive control (outcome-free). Take 200 random pre-deletion bodies,
apply perturbations a recreating agent might use, and measure whether the matcher (word 3-shingles) still fires.
Uses only pre-deletion bodies; never looks at revisions after a deletion. Read-only."""
import json, random, re, collections
from datetime import datetime, timezone
from pathlib import Path
D = Path("data/raw/collusion_wiki/x")
P = lambda s: datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
SIG = re.compile(r"^\s*--\s*\S.*$", re.M); TOK = re.compile(r"\w+", re.U)
import sys
MASK = "--mask-digits" in sys.argv
def sh(body, k=3):
    t = SIG.sub(" ", body.lower())
    if MASK: t = re.sub(r"\d", "0", t)
    w = TOK.findall(t)
    return {hash(" ".join(w))} if 0 < len(w) < k else {hash(" ".join(w[i:i+k])) for i in range(len(w) - k + 1)}
jac = lambda a, b: len(a & b) / len(a | b) if a and b else 0.0
cont = lambda a, b: len(a & b) / len(a) if a else 0.0
revs = [json.loads(l) for l in open(D / "revisions.jsonl")]
ev = [json.loads(l) for l in open(D / "events.jsonl")]
by_page = collections.defaultdict(list)
for r in revs: by_page[r["page_key"]].append(r)
for k in by_page: by_page[k].sort(key=lambda r: (r["time"], r["seq"]))
first_del = {}
for e in sorted((e for e in ev if e["event_type"] == "delete"), key=lambda e: e["time"]): first_del.setdefault(e["page_key"], P(e["time"]))
anchors = []
for pk, t in first_del.items():
    pre = [r for r in by_page.get(pk, []) if P(r["time"]) < t]
    if pre and len(TOK.findall(pre[-1]["body"])) >= 20: anchors.append(pre[-1]["body"])
rng = random.Random(7)
A = rng.sample(anchors, 200); filler = [r["body"] for r in rng.sample(revs, 400)]
def drop(b, f): L = b.splitlines(); return "\n".join(x for x in L if rng.random() > f)
def digits(b): return re.sub(r"\d", lambda m: str(rng.randrange(10)), b)
def append(b): return b + "\n" + rng.choice(filler)
def prepend_half(b): L = b.splitlines(); return rng.choice(filler) + "\n" + "\n".join(L[: max(1, len(L) // 2)])
V = {"exact copy, new signature": lambda b: b + "\n-- NewName", "drop 30% of lines": lambda b: drop(b, .3), "random digits": digits,
     "append unrelated page": append, "keep half + unrelated page": prepend_half, "drop 30% + digits + append": lambda b: append(digits(drop(b, .3)))}
print("digit masking:", MASK); print("anchors eligible (>=20 words):", len(anchors), "| sampled 200")
for name, f in V.items():
    j = c = 0
    for b in A:
        x, y = sh(b), sh(f(b))
        j += jac(x, y) >= 0.5; c += cont(x, y) >= 0.5
    print(f"  {name:32s} recall Jaccard>=0.5: {j}/200 | containment>=0.5: {c}/200")
