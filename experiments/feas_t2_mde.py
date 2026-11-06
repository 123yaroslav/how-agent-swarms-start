"""Feasibility t2: planning numbers for MDE from counts alone.
(1) post-window available before the global stop for active treated pages (treatment side only);
(2) pre-period persistence base rate p0 = share of active pages saved again within 24h, measured only at
reference times before the first deletion batch (06-18T18), so it carries no deletion information;
(3) two-sample MDE for a binary page outcome. Read-only."""
import json, collections, bisect, math
from datetime import datetime, timedelta, timezone
from pathlib import Path
D = Path("data/raw/collusion_wiki/x")
P = lambda s: datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
revs = [(json.loads(l)["page_key"], P(json.loads(l)["time"])) for l in open(D / "revisions.jsonl")]
ev = [json.loads(l) for l in open(D / "events.jsonl")]
by_page = collections.defaultdict(list)
for pk, t in revs: by_page[pk].append(t)
for k in by_page: by_page[k].sort()
first_del = {}
for e in sorted((e for e in ev if e["event_type"] == "delete"), key=lambda e: e["time"]): first_del.setdefault(e["page_key"], P(e["time"]))
STOP = P("2026-06-22T09:20:04Z")
def active(pk, t, h=48):
    xs = by_page.get(pk, []); i = bisect.bisect_left(xs, t)
    return i > 0 and xs[i - 1] >= t - timedelta(hours=h)
treated = [(pk, t) for pk, t in first_del.items() if P("2026-06-16T00:00:00Z") <= t < P("2026-06-22T00:00:00Z") and active(pk, t)]
print("active treated pages 06-16..06-21:", len(treated))
for h in (6, 12, 24, 48):
    print(f"  with >= {h}h before global stop (06-22 09:20:04Z): {sum(1 for _, t in treated if STOP - t >= timedelta(hours=h))}")
batches = collections.Counter()
for _, t in treated:
    batches[t.strftime("%m-%dT%H")] += 1
# merge consecutive hours into batches (gap > 2h starts a new batch)
hrs = sorted(batches); groups = []; 
for h in hrs:
    ht = datetime.strptime("2026-" + h, "%Y-%m-%dT%H")
    if groups and (ht - groups[-1][1]) <= timedelta(hours=2): groups[-1][1] = ht; groups[-1][2] += batches[h]
    else: groups.append([ht, ht, batches[h]])
print("deletion batches (consecutive hours merged):", [(g[0].strftime("%m-%dT%H"), g[1].strftime("%m-%dT%H"), g[2]) for g in groups])
# pre-period persistence base rate
for t0s in ("2026-06-16T12:00:00Z", "2026-06-17T00:00:00Z", "2026-06-17T12:00:00Z"):
    t0 = P(t0s)
    A = [pk for pk in by_page if active(pk, t0) and not (pk in first_del and first_del[pk] < t0 + timedelta(hours=24))]
    nxt = sum(1 for pk in A if any(t0 <= x < t0 + timedelta(hours=24) for x in by_page[pk]))
    print(f"  p0 at {t0s[5:13]}: active pages {len(A)}, saved again within 24h {nxt} -> p0={nxt/len(A):.3f}")
z = 1.959964 + 0.841621
for p in (0.03, 0.05, 0.08, 0.2, 0.5):
    for nT, nC in ((199, 1000), (100, 1000)):
        print(f"  MDE p0={p} nT={nT} nC={nC}: {z * math.sqrt(p * (1 - p) * (1 / nT + 1 / nC)):.3f} (abs. points, no clustering)")
# batch-level: with G batches, a cluster-level test has G-1 df
print("batches with >=10 active treated pages:", sum(1 for g in groups if g[2] >= 10))
