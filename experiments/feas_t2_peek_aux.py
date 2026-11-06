"""Feasibility t2: peek structure of shortener-logs.json and site_figures/visits.js (as text, no execution)."""
import json, re
from pathlib import Path
s = json.load(open("data/raw/collusion_wiki/x/shortener-logs.json"))
print("shortener top keys:", list(s.keys()) if isinstance(s, dict) else type(s))
for k, v in s.items():
    if isinstance(v, list):
        print(k, "list", len(v)); print("  ex:", json.dumps(v[0], ensure_ascii=False)[:600])
        keys = set();  [keys.update(x.keys()) for x in v if isinstance(x, dict)]
        print("  keys:", keys)
    else:
        print(k, ":", json.dumps(v, ensure_ascii=False)[:1500])
t = Path("data/raw/collusion_wiki/site_figures/visits.js").read_text()
# print comment header and data-like lines
print("---- visits.js header")
print(t[:3000])
