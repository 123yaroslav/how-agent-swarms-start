"""Feasibility t2, task 2: how delete events map to pages/revisions, and what the first_recreation_of edges link.
Only the investigator-labelled edges are described; no recreation counts over all deletions. Read-only."""
import json, collections
from datetime import datetime, timezone
from pathlib import Path
D = Path("data/raw/collusion_wiki/x")
P = lambda s: datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
revs = [json.loads(l) for l in open(D / "revisions.jsonl")]
ev = [json.loads(l) for l in open(D / "events.jsonl")]
pages = {json.loads(l)["page_key"]: json.loads(l) for l in open(D / "pages.jsonl")}
dels = {e["event_id"]: e for e in ev if e["event_type"] == "delete"}
by_page = collections.defaultdict(list)
for r in revs: by_page[r["page_key"]].append(r)
for k in by_page: by_page[k].sort(key=lambda r: (r["time"], r["seq"]))
edges = []
for e in ev:
    if e.get("relation_type") != "first_recreation_of": continue
    rel = e["related_event_id"]; rel = [rel] if isinstance(rel, str) else rel
    for d in rel:
        edges.append(dict(src=e, d=dels[d]))
print("edges (relations):", len(edges), "| from save events:", sum(x["src"]["event_type"] == "save" for x in edges), "| from revert events:", sum(x["src"]["event_type"] == "revert" for x in edges))
print("same page_key delete vs recreation:", collections.Counter(x["src"]["page_key"] == x["d"]["page_key"] for x in edges))
print("recreation after delete:", collections.Counter(P(x["src"]["time"]) > P(x["d"]["time"]) for x in edges))
print("distinct pages:", len({x["d"]["page_key"] for x in edges}), "| distinct delete events:", len({x["d"]["event_id"] for x in edges}), "| distinct recreation events:", len({x["src"]["event_id"] for x in edges}))
print("delete time span:", min(x["d"]["time"] for x in edges), max(x["d"]["time"] for x in edges))
print("recreation time span:", min(x["src"]["time"] for x in edges), max(x["src"]["time"] for x in edges))
print("delete page_held:", collections.Counter(x["d"]["page_held"] for x in edges))
# held pre-deletion body?
def last_pre(pk, t):
    xs = [r for r in by_page.get(pk, []) if P(r["time"]) < t]
    return xs[-1] if xs else None
ev_ok = 0; seq_info = collections.Counter()
for x in edges:
    t = P(x["d"]["time"]); pre = last_pre(x["d"]["page_key"], t)
    x["pre"] = pre
    if pre: ev_ok += 1
    if pre and x["src"]["event_type"] == "save":
        tgt = next(r for r in by_page[x["src"]["page_key"]] if r["rev_id"] == x["src"]["revision_ref"])
        seq_info["seq continues (tgt.seq > pre.seq)" if tgt["seq"] > pre["seq"] else "seq restarts"] += 1
        seq_info["diff_base=" + ("pre-deletion rev" if tgt["diff_base"] == pre["rev_id"] else str(tgt["diff_base_reason"]))] += 1
print("edges whose deleted page has a held pre-deletion revision (body available):", ev_ok)
print("seq / diff_base on target revisions:", dict(seq_info))
# multiple deletions per edge page
nd = collections.Counter(d["page_key"] for d in dels.values())
print("deletions per edge page:", collections.Counter(nd[x["d"]["page_key"]] for x in edges))
# revision-side fields
rr = [r for r in revs if r.get("relation_type") == "first_recreation_of"]
print("revisions carrying relation:", len(rr), "| example:", {k: rr[0][k] for k in ("rev_id", "seq", "diff_base", "diff_base_reason", "related_event_id", "request_action")})
print("diff_base_reason overall:", collections.Counter(r["diff_base_reason"] for r in revs))
print("deleted_live overall (pages):", collections.Counter(p["deleted_live"] for p in pages.values()), "| head_differs_from_live:", collections.Counter(p["head_differs_from_live"] for p in pages.values()))
print("deleted_live vs n_deletions>0:", collections.Counter((p["deleted_live"], p["n_deletions"] > 0) for p in pages.values()))
print("live_body_variant:", collections.Counter(p["live_body_variant"] for p in pages.values()))
# check pages.n_deletions agrees with events
agree = sum(1 for k, p in pages.items() if p["n_deletions"] == nd.get(k, 0))
print("pages.n_deletions agrees with events count:", agree, "/", len(pages))
# revert edges detail
for x in edges:
    if x["src"]["event_type"] == "revert":
        print("revert edge: delete", x["d"]["time"], "-> revert", x["src"]["time"], "actor", x["src"]["actor_label"], "pre-del body held:", bool(x["pre"]))
# the 2-delete save
for e in ev:
    if e.get("relation_type") == "first_recreation_of" and isinstance(e["related_event_id"], list) and len(e["related_event_id"]) == 2:
        print("save linked to 2 deletes:", e["revision_ref"][:40], [dels[d]["time"] for d in e["related_event_id"]])
print("delete event fields:", sorted(next(iter(dels.values())).keys()))
