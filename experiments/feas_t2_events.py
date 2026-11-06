"""Feasibility t2: structure of events (delete, revert, probe, first_recreation_of). Read-only, no outcome stats."""
import json, collections
from pathlib import Path
D = Path("data/raw/collusion_wiki/x")
ev = [json.loads(l) for l in open(D / "events.jsonl")]
print("event_type:", collections.Counter(e["event_type"] for e in ev))
print("relation_type by event_type:", collections.Counter((e["event_type"], e.get("relation_type")) for e in ev))
print("actor_label by type:", collections.Counter((e["event_type"], e.get("actor_label")) for e in ev))
print("request_action by type:", collections.Counter((e["event_type"], e.get("request_action")) for e in ev))
print("wiki by type:", collections.Counter((e["event_type"], e.get("wiki")) for e in ev))
print("page_held for delete:", collections.Counter(e.get("page_held") for e in ev if e["event_type"]=="delete"))
print("change_summary for delete:", collections.Counter(e.get("change_summary") for e in ev if e["event_type"]=="delete").most_common(5))
print("clock_note delete:", collections.Counter(e.get("clock_note") for e in ev if e["event_type"]=="delete").most_common(5))
print("time_grade delete:", collections.Counter(e.get("time_grade") for e in ev if e["event_type"]=="delete"))
print("uncertainty delete:", collections.Counter(e.get("uncertainty_seconds") for e in ev if e["event_type"]=="delete").most_common(6))
# examples
for t in ["delete", "revert", "save"]:
    xs = [e for e in ev if e["event_type"] == t]
    print(f"--- example {t}:")
    for e in xs[:2]:
        print({k: (v if not isinstance(v, str) or len(v) < 150 else v[:150]) for k, v in e.items()})
# events with relation first_recreation_of
rec = [e for e in ev if e.get("relation_type") == "first_recreation_of"]
print("n events with first_recreation_of:", len(rec), collections.Counter(e["event_type"] for e in rec))
for e in rec[:3]:
    print({k: v for k, v in e.items() if k not in ("change_summary",)})
# related_event_id structure
print("related_event_id lens:", collections.Counter(len(e.get("related_event_id") or []) for e in rec))
print("relation types overall in events:", collections.Counter(e.get("relation_type") for e in ev))
# delete id format
dels = [e for e in ev if e["event_type"] == "delete"]
print("delete event_id examples:", [d["event_id"] for d in dels[:3]], "revision_ref ex:", [d.get("revision_ref") for d in dels[:3]])
print("delete related:", collections.Counter((d.get("relation_type"), len(d.get("related_event_id") or [])) for d in dels))
