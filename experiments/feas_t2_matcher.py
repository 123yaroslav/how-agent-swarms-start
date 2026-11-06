"""Feasibility t2, task 3: content matcher (normalised word 3-shingles, Jaccard and containment).
Positive control: recall on the investigator first_recreation_of edges (target = labelled revision).
Negatives: random page pairs where the candidate revision PRECEDES the anchor page's deletion, so no pair
can be a recreation; this measures false matches without looking at any post-deletion behaviour.
Prints only target scores/ranks and negative-pair scores. Read-only on data/raw."""
import json, collections, random, re, statistics, bisect
from datetime import datetime, timezone
from pathlib import Path
D = Path("data/raw/collusion_wiki/x")
P = lambda s: datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
SIG = re.compile(r"^\s*--\s*\S.*$", re.M)
TOK = re.compile(r"\w+", re.U)
import sys
MASK = "--mask-digits" in sys.argv
def shingles(body, k=3):
    t = SIG.sub(" ", body.lower())
    if MASK: t = re.sub(r"\d", "0", t)
    w = TOK.findall(t)
    if len(w) < k: return {hash(" ".join(w))} if w else set()
    return {hash(" ".join(w[i:i + k])) for i in range(len(w) - k + 1)}
def jac(a, b): return len(a & b) / len(a | b) if a and b else 0.0
def cont(a, b): return len(a & b) / len(a) if a else 0.0
revs = [json.loads(l) for l in open(D / "revisions.jsonl")]
ev = [json.loads(l) for l in open(D / "events.jsonl")]
pages = {json.loads(l)["page_key"]: json.loads(l) for l in open(D / "pages.jsonl")}
for r in revs: r["_t"] = P(r["time"])
revs.sort(key=lambda r: r["_t"]); times = [r["_t"] for r in revs]
by_id = {r["rev_id"]: r for r in revs}
by_page = collections.defaultdict(list)
for r in revs: by_page[r["page_key"]].append(r)
dels = {e["event_id"]: e for e in ev if e["event_type"] == "delete"}
SH = {}
def sh(r):
    if r["rev_id"] not in SH: SH[r["rev_id"]] = shingles(r["body"])
    return SH[r["rev_id"]]
HUMAN = {json.loads(l)["label"] for l in open(D / "labels.jsonl") if json.loads(l)["is_human_handle"]}
SKIP_HUMAN = "--skip-human" in sys.argv
def last_pre(pk, t):
    xs = [r for r in by_page.get(pk, []) if r["_t"] < t and not (SKIP_HUMAN and (r["label"] in HUMAN or r["body_len"] == 0))]
    return xs[-1] if xs else None
# ---- positive control
edges = []
for e in ev:
    if e.get("relation_type") != "first_recreation_of": continue
    rel = e["related_event_id"]; rel = [rel] if isinstance(rel, str) else rel
    for d in rel: edges.append((e, dels[d]))
n_all = len(edges); res = []; skipped = collections.Counter()
for src, d in edges:
    t = P(d["time"]); pre = last_pre(d["page_key"], t)
    if pre is None: skipped["no held pre-deletion body"] += 1; continue
    if src["event_type"] != "save" or not src.get("revision_ref") or src["revision_ref"] not in by_id: skipped["target has no stored revision (revert)"] += 1; continue
    tgt = by_id[src["revision_ref"]]; A = sh(pre)
    i = bisect.bisect_right(times, t)
    cands = revs[i:]
    sc = [(jac(A, sh(c)), c["rev_id"]) for c in cands]
    sj = jac(A, sh(tgt)); sc_c = cont(A, sh(tgt))
    rank = 1 + sum(1 for s, rid in sc if s > sj)
    res.append(dict(j=sj, c=sc_c, rank=rank, ncand=len(cands), pre_len=len(pre["body"]), tgt_len=len(tgt["body"]), same_sha=pre["body_sha256"] == tgt["body_sha256"]))
print("digit masking:", MASK, "| skip human-handle/empty pre-deletion bodies:", SKIP_HUMAN); print(f"edges total {n_all}; evaluable {len(res)}; skipped {dict(skipped)}")
js = [r["j"] for r in res]; cs = [r["c"] for r in res]
print("target Jaccard: median %.3f, quartiles %s" % (statistics.median(js), [round(q, 3) for q in statistics.quantiles(js, n=4)]))
print("target containment: median %.3f" % statistics.median(cs))
print("target body identical to deleted body:", sum(r["same_sha"] for r in res))
for th in (0.3, 0.5, 0.8):
    print(f"  recall Jaccard>={th}: {sum(j >= th for j in js)}/{len(js)} | containment>={th}: {sum(c >= th for c in cs)}/{len(cs)}")
for k in (1, 5, 10):
    print(f"  recall@top{k} by Jaccard among all later revisions: {sum(r['rank'] <= k for r in res)}/{len(res)}")
print("candidates per query: median", statistics.median(r["ncand"] for r in res))
# ---- negatives: anchor = last pre-deletion body of a deleted page; candidate = random revision on another page written BEFORE anchor deletion
rng = random.Random(20261106)
anchors = []
seen = set()
for d in sorted(dels.values(), key=lambda d: d["time"]):
    if d["page_key"] in seen: continue
    seen.add(d["page_key"]); pre = last_pre(d["page_key"], P(d["time"]))
    if pre is not None: anchors.append((pre, P(d["time"])))
edge_pairs = {(d["page_key"], src["page_key"]) for src, d in edges}
def sample(n, hard=False):
    out = []
    while len(out) < n:
        pre, t = rng.choice(anchors)
        i = bisect.bisect_left(times, t)
        if i == 0: continue
        if hard:
            fam = pages[pre["page_key"]]["page_family"]
            pool = [c for c in revs[max(0, i - 3000):i] if c["page_key"] != pre["page_key"] and pages[c["page_key"]]["page_family"] == fam]
            if not pool: continue
            c = rng.choice(pool)
        else:
            c = revs[rng.randrange(i)]
            if c["page_key"] == pre["page_key"]: continue
        if (pre["page_key"], c["page_key"]) in edge_pairs: continue
        out.append((jac(sh(pre), sh(c)), cont(sh(pre), sh(c))))
    return out
for name, hard in (("random pairs", False), ("hard pairs (same page_family, candidate among the 3000 saves before deletion)", True)):
    neg = sample(200, hard)
    print(f"NEGATIVES {name}: n={len(neg)}")
    for th in (0.3, 0.5, 0.8):
        print(f"  false matches Jaccard>={th}: {sum(j >= th for j, _ in neg)}/200 | containment>={th}: {sum(c >= th for _, c in neg)}/200")
    print("  max Jaccard %.3f, 95th pct %.3f" % (max(j for j, _ in neg), sorted(j for j, _ in neg)[189]))
