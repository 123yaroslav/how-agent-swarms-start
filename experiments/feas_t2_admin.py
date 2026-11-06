"""Feasibility t2, task 4b: moderator / human-handle presence and the signature definition. Read-only."""
import json, collections, re
from pathlib import Path
D = Path("data/raw/collusion_wiki/x")
revs = [json.loads(l) for l in open(D / "revisions.jsonl")]
labs = {json.loads(l)["label"]: json.loads(l) for l in open(D / "labels.jsonl")}
for k in ("[Admin1]", "[Admin2]", "[Admin2]302", "AgentRelent"):
    l = labs.get(k)
    if l: print(k, {x: l[x] for x in ("stored_revisions", "stored_revision_pages", "save_requests", "save_request_pages", "save_request_ips", "first_write", "last_write", "wikis")})
adm = [r for r in revs if r["label"] in ("[Admin1]", "[Admin2]", "[Admin2]302")]
print("human-handle revisions by day/wiki:", sorted(collections.Counter((r["label"], r["time"][:10], r["wiki"]) for r in adm).items()))
mask = lambda s: re.sub(r"\d", "#", s[:90]).replace("\n", " | ")
for r in adm[:4]: print("  ", r["label"], r["time"], "change_summary=", (r["change_summary"] or "")[:40], "| added:", mask(r["body"][r["body"].find("\n")+1:] if False else r["body"]))
# signatures: '-- Name' at the end of any line of the body (data_recon definition guess)
L = re.compile(r"--\s?([A-Za-z][\w.\-\[\]]{1,40})\s*$", re.M)
anyline = sum(1 for r in revs if L.search(r["body"]))
def added(r):
    B = r["body"].splitlines(); o = []
    for h in r["hunks"]:
        if h["op"] in ("insert", "replace"): o += B[h["b0"]:h["b1"]]
    return "\n".join(o)
add = [(r, L.findall(added(r))) for r in revs]
a_any = sum(1 for r, m in add if m); a_match = sum(1 for r, m in add if m and r["label"] and any(x.lower() == r["label"].lower() for x in m))
print(f"revisions with a '-- Name' line anywhere in body: {anyline}; in added text: {a_any}; added signature equals save label: {a_match}")
