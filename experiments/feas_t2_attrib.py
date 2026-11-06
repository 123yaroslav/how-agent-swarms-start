"""Feasibility t2, task 4: attribution facts (labels, signatures, ip16, human handles, moderator presence). Read-only."""
import json, collections, re
from pathlib import Path
D = Path("data/raw/collusion_wiki/x")
revs = [json.loads(l) for l in open(D / "revisions.jsonl")]
labs = [json.loads(l) for l in open(D / "labels.jsonl")]
recs = [json.loads(l) for l in open(D / "records.jsonl")]
n = len(revs)
lab = sum(1 for r in revs if r["label"]); print(f"revisions {n}; with label {lab} ({lab/n:.1%})")
ipv = re.compile(r"^\d{1,3}\.\d{1,3}$")
ipc = collections.Counter("valid /16" if r["ip16"] and ipv.match(r["ip16"]) else ("missing" if not r["ip16"] else "token:" + r["ip16"][:16]) for r in revs)
print("ip16:", ipc.most_common(8))
SIG = re.compile(r"(?:^|\n)\s*--\s*([^\n]{1,60})\s*$")
def added(r):
    L = r["body"].splitlines(); out = []
    for h in r["hunks"]:
        if h["op"] in ("insert", "replace"): out += L[h["b0"]:h["b1"]]
    return "\n".join(out)
sig_body = sig_add = sig_match = 0
for r in revs:
    m = SIG.search(r["body"].rstrip())
    if m: sig_body += 1
    ma = SIG.search(added(r).rstrip())
    if ma:
        sig_add += 1
        if r["label"] and ma.group(1).strip().lower() == r["label"].lower(): sig_match += 1
print(f"body ends with '-- Name': {sig_body} ({sig_body/n:.1%}); added text ends with '-- Name': {sig_add} ({sig_add/n:.1%}); of those, signature == save label: {sig_match}")
print("labels:", len(labs), "| is_human_handle:", sum(l["is_human_handle"] for l in labs))
hh = {l["label"] for l in labs if l["is_human_handle"]}
print("revisions by human-handle labels:", sum(1 for r in revs if r["label"] in hh), "| their stored_revisions:", [l["stored_revisions"] for l in labs if l["is_human_handle"]])
print("labels containing 'admin' (case-insens.):", [(l["label"], l["stored_revisions"]) for l in labs if "admin" in l["label"].lower()][:10])
print("revisions whose label or ip16 mentions Admin1:", sum(1 for r in revs if "admin1" in (r["label"] or "").lower() or "admin1" in (r["ip16"] or "").lower()))
print("request_action:", collections.Counter(r["request_action"] for r in revs))
print("time_grade:", collections.Counter(r["time_grade"] for r in revs))
print("records authorship:", collections.Counter(r["authorship"] for r in recs), "| selection_basis:", collections.Counter(r["selection_basis"] for r in recs).most_common(4))
print("record origin keys:", sorted({k for r in recs for o in r["origins"] for k in o}))
print("label save_request_source:", collections.Counter(l["save_request_source"] for l in labs))
# labels with many IPs (shared-name risk)
big = sorted(labs, key=lambda l: -l["stored_revision_ips"])[:5]
print("top labels by distinct IPs (revisions, ips, ip16):", [(l["stored_revisions"], l["stored_revision_ips"], l["stored_revision_ip16"]) for l in big])
print("labels with 1 stored revision:", sum(l["stored_revisions"] == 1 for l in labs), "| with >=10:", sum(l["stored_revisions"] >= 10 for l in labs))
