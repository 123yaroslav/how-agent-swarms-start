"""Feasibility t2, task 1: treatment timing and unit counts. Treatment side and pre-treatment activity only.
No post-deletion outcomes of treated vs untreated units are computed here.
Writes results/feas_t2/hourly_0616_0623.csv and daily_0601_0714.csv. Read-only on data/raw."""
import json, collections, csv, bisect
from datetime import datetime, timedelta, timezone
from pathlib import Path
D = Path("data/raw/collusion_wiki/x"); OUT = Path("results/feas_t2"); OUT.mkdir(parents=True, exist_ok=True)
P = lambda s: datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
revs = []
for l in open(D / "revisions.jsonl"):
    r = json.loads(l)
    revs.append((r["page_key"], P(r["time"]), r["label"] or "", r["wiki"]))
ev = [json.loads(l) for l in open(D / "events.jsonl")]
dels = sorted([(e["page_key"], P(e["time"]), e["event_id"], e["page_held"]) for e in ev if e["event_type"] == "delete"], key=lambda x: x[1])
pages = {json.loads(l)["page_key"]: json.loads(l) for l in open(D / "pages.jsonl")}
by_page = collections.defaultdict(list)
for pk, t, lab, w in revs:
    by_page[pk].append((t, lab))
for pk in by_page: by_page[pk].sort()
first_del = {}
n_del_per_page = collections.Counter()
for pk, t, eid, held in dels:
    n_del_per_page[pk] += 1
    first_del.setdefault(pk, t)
print("delete events", len(dels), "distinct pages", len(first_del))
print("pages deleted k times:", sorted(collections.Counter(n_del_per_page.values()).items()))
held_pages = set(pages); dse_held = {p for p in pages if pages[p]["wiki"] == "dse"}
print("deleted pages that are in pages.jsonl:", len(set(first_del) & held_pages), "| not in export:", len(set(first_del) - held_pages))
# pre-deletion revisions per delete event
def pre_revs(pk, t, hours=None):
    xs = by_page.get(pk, [])
    i = bisect.bisect_left(xs, (t, ""))
    pre = xs[:i]
    if hours is not None:
        lo = t - timedelta(hours=hours)
        pre = [x for x in pre if x[0] >= lo]
    return pre
ev_rows = []
for pk, t, eid, held in dels:
    pre = pre_revs(pk, t); pre48 = pre_revs(pk, t, 48)
    ev_rows.append(dict(pk=pk, t=t, has_pre=bool(pre), active48=bool(pre48), labels={lab for _, lab in pre if lab}, first=(first_del[pk] == t)))
print("delete events with >=1 held pre-deletion revision:", sum(r["has_pre"] for r in ev_rows))
print("delete events with a pre-deletion revision in prior 48h:", sum(r["active48"] for r in ev_rows))
fd = [r for r in ev_rows if r["first"]]
# dedupe first deletes (same second duplicates)
seen = set(); fd2 = []
for r in fd:
    if r["pk"] in seen: continue
    seen.add(r["pk"]); fd2.append(r)
print("deleted pages (first deletion):", len(fd2), "| with held pre-del revision:", sum(r["has_pre"] for r in fd2), "| active in prior 48h:", sum(r["active48"] for r in fd2))
for lo, hi in [("2026-06-16", "2026-06-22"), ("2026-06-17", "2026-06-22"), ("2026-06-01", "2026-07-15")]:
    a, b = P(lo + "T00:00:00Z"), P(hi + "T00:00:00Z")
    sub = [r for r in fd2 if a <= r["t"] < b]
    print(f"  first deletions in [{lo},{hi}): {len(sub)} pages, active48 {sum(r['active48'] for r in sub)}, with pre-del rev {sum(r['has_pre'] for r in sub)}")
# never-deleted pages
never = [p for p in pages if p not in first_del]
print("never-deleted held pages:", len(never), collections.Counter(pages[p]["wiki"] for p in never))
W0, W1 = P("2026-06-16T00:00:00Z"), P("2026-06-22T00:00:00Z")
act = [p for p in never if any(W0 <= t < W1 for t, _ in by_page[p])]
print("never-deleted pages with a revision in 06-16..06-21:", len(act), collections.Counter(pages[p]["wiki"] for p in act))
ever_del_held = [p for p in first_del if p in pages]
act_del = [p for p in ever_del_held if any(W0 <= t < W1 for t, _ in by_page[p])]
print("ever-deleted held pages with a revision in 06-16..06-21 (any time rel. deletion):", len(act_del))
# label treatment: first time a label's page (with a pre-deletion revision by that label) is deleted
label_first_treat = {}
for r in ev_rows:
    for lab in r["labels"]:
        if lab not in label_first_treat or r["t"] < label_first_treat[lab]:
            label_first_treat[lab] = r["t"]
all_labels = {lab for _, _, lab, _ in revs if lab}
print("labels (non-blank) in revisions:", len(all_labels), "| ever treated:", len(label_first_treat), "| never treated:", len(all_labels - set(label_first_treat)))
by_label = collections.defaultdict(list)
for pk, t, lab, w in revs:
    if lab: by_label[lab].append(t)
for lab in by_label: by_label[lab].sort()
lab_act48 = sum(1 for lab, t in label_first_treat.items() if any(t - timedelta(hours=48) <= x < t for x in by_label[lab]))
print("treated labels with a revision in 48h before their first treatment:", lab_act48)
# hourly table
def table(start, end, step):
    rows = []
    t = start
    while t < end:
        t2 = t + step
        sub = [r for r in ev_rows if t <= r["t"] < t2]
        rows.append(dict(start=t.strftime("%Y-%m-%dT%H"), delete_events=len(sub), distinct_pages=len({r["pk"] for r in sub}),
                         pages_with_pre_rev=len({r["pk"] for r in sub if r["has_pre"]}), pages_active48=len({r["pk"] for r in sub if r["active48"]}),
                         distinct_labels=len(set().union(*[r["labels"] for r in sub]) if sub else set())))
        if step >= timedelta(days=1):  # daily whole-swarm saves only (already public in data_recon); no hourly series around deletions
            rows[-1]["saves"] = sum(1 for _, x, _, _ in revs if t <= x < t2)
        t = t2
    return rows
H = table(P("2026-06-16T00:00:00Z"), P("2026-06-24T00:00:00Z"), timedelta(hours=1))
Dd = table(P("2026-06-01T00:00:00Z"), P("2026-07-15T00:00:00Z"), timedelta(days=1))
for name, rows in [("hourly_0616_0623.csv", H), ("daily_0601_0714.csv", Dd)]:
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("DAILY (nonzero deletes or saves):")
for r in Dd:
    if r["delete_events"] or r["saves"]:
        print("  ", r)
print("HOURLY (hours with deletions):")
for r in H:
    if r["delete_events"]:
        print("  ", r)
# not-yet-treated pool per hour 06-17..06-21
print("NOT-YET-TREATED POOL per hour (pages active in prior 48h, not deleted before hour end):")
pool_rows = []
t = P("2026-06-17T00:00:00Z")
while t < P("2026-06-22T00:00:00Z"):
    t2 = t + timedelta(hours=1); lo = t - timedelta(hours=48)
    active = {pk for pk, x, _, _ in revs if lo <= x < t}
    nyt_later = [p for p in active if p in first_del and first_del[p] >= t2]
    nyt_never_dse = [p for p in active if p not in first_del and pages[p]["wiki"] == "dse"]
    nyt_never_oth = [p for p in active if p not in first_del and pages[p]["wiki"] != "dse"]
    treated_now = [r for r in fd2 if t <= r["t"] < t2 and r["active48"]]
    act_labels = {lab for _, x, lab, _ in revs if lo <= x < t and lab}
    lab_nyt = [l for l in act_labels if l not in label_first_treat or label_first_treat[l] >= t2]
    lab_treat_now = [l for l, x in label_first_treat.items() if t <= x < t2]
    pool_rows.append(dict(hour=t.strftime("%m-%dT%H"), treated_active_pages=len(treated_now), nyt_later_deleted=len(nyt_later), never_del_dse=len(nyt_never_dse), never_del_other=len(nyt_never_oth),
                          labels_first_treated=len(lab_treat_now), labels_nyt_active=len(lab_nyt)))
    t = t2
with open(OUT / "pool_hourly_0617_0621.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(pool_rows[0])); w.writeheader(); w.writerows(pool_rows)
for r in pool_rows: print("  ", r)
# swarm stop
print("SAVES per hour 06-21T00 .. 06-23T00 (swarm stop check only):")
st = collections.Counter(x.strftime("%m-%dT%H") for _, x, _, _ in revs if P("2026-06-21T00:00:00Z") <= x < P("2026-06-23T00:00:00Z"))
print("  ", sorted(st.items()))
last_burst = max(x for _, x, _, _ in revs if x < P("2026-06-22T12:00:00Z"))
print("last save before 06-22T12:", last_burst)
ts = sorted(x for _, x, _, _ in revs)
print("saves after 06-22T00:", sum(1 for x in ts if x >= P("2026-06-22T00:00:00Z")), "| after 06-23T00:", sum(1 for x in ts if x >= P("2026-06-23T00:00:00Z")), "| last save:", ts[-1])
print("saves by day after 06-22:", sorted(collections.Counter(x.strftime('%m-%d') for x in ts if x >= P('2026-06-22T00:00:00Z')).items()))
print("deletes after 06-22T00:", sum(1 for r in ev_rows if r["t"] >= P("2026-06-22T00:00:00Z")))
