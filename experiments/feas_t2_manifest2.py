"""Feasibility t2: list manifest top-level keys, fact names with values+definitions, and other text fields. Read-only."""
import json
from pathlib import Path
m = json.load(open(Path("data/raw/collusion_wiki/x/manifest.json")))
print("TOP KEYS:", list(m.keys()))
for k, v in m.get("facts", {}).items():
    print(f"FACT {k} = {v.get('value')} | {v.get('definition','')[:300]}")
for k in m:
    if k in ("facts", "counts", "per_wiki", "population_counts", "grade_histograms", "body_bytes", "body_encoding", "page_family_coverage"):
        continue
    print(f"== {k}:", json.dumps(m[k], ensure_ascii=False)[:3000])
