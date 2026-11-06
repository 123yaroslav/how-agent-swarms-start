"""Feasibility t2: dump manifest.json structure and text fields. Read-only."""
import json
from pathlib import Path
m = json.load(open(Path("data/raw/collusion_wiki/x/manifest.json")))
def walk(o, path="", depth=0):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, f"{path}.{k}", depth+1)
    elif isinstance(o, list):
        print(f"{path}: list[{len(o)}]")
        for i, v in enumerate(o[:3]):
            walk(v, f"{path}[{i}]", depth+1)
    else:
        s = json.dumps(o, ensure_ascii=False)
        print(f"{path}: {s[:400]}")
walk(m)
