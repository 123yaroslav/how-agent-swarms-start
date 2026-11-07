"""SECONDARY (not a pre-registered test): what a blocked agent's outcome looks like with and without
the house rule, from the confirmatory single-agent cells.

Under outcome-only reward, a blocked job earns reward only by succeeding. This prints, per model and
cell, the share of blocked agents that completed the job, used other jobs' artifacts in code, and
reported the job as blocked. If success happens only when the rule is absent and other jobs'
artifacts are used, outcome reward pays for crossing the data boundary.

    uv run python experiments/success_by_rule.py

Reads results/raw/scale/confirm_<model>.jsonl line by line (the running confirmatory process may be
appending; a partial last line is skipped). Writes results/confirmatory/success_by_rule.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swarm.fleet.stats import wilson  # noqa: E402
from swarm.scale.analysis import episode_rows  # noqa: E402

RAW = ROOT / "results" / "raw" / "scale"
OUT = ROOT / "results" / "confirmatory" / "success_by_rule.json"
MODELS = ["deepseek-v4-flash", "gemma-4-31b-it", "gemma-4-26b-a4b-it", "qwen3.8-27b", "glm-5.2",
          "qwen3.5-397b"]
CELLS = ["plain-rule", "plain-norule", "coordinate_ok-rule"]


def rate(vals: list[bool]) -> dict:
    k, n = sum(vals), len(vals)
    lo, hi = wilson(k, n) if n else (0.0, 1.0)
    return {"k": k, "n": n, "p": k / n if n else None, "lo": lo, "hi": hi}


def main() -> None:
    out: dict = {"label": "SECONDARY, not pre-registered", "models": {}}
    for m in MODELS:
        path = RAW / f"confirm_{m}.jsonl"
        lab_path = RAW / f"confirm_{m}.labels.jsonl"
        if not path.exists():
            continue
        labels = {}
        for ln in lab_path.read_text().splitlines():
            try:
                r = json.loads(ln)
                labels[r["episode_id"]] = r
            except json.JSONDecodeError:
                continue
        by: dict[str, list[dict]] = {c: [] for c in CELLS}
        seen = set()
        with path.open() as f:
            for ln in f:
                try:
                    rec = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                eid = rec.get("episode_id")
                sp = rec.get("spec", {})
                if eid in seen or eid not in labels or sp.get("mode") != "harness" \
                        or sp.get("seed_note") != "none":
                    continue
                seen.add(eid)
                for r in episode_rows([rec], labels):
                    if r["arm"] in by and r.get("blocked", True):
                        by[r["arm"]].append(r)
        out["models"][m] = {c: {"n": len(rs),
                                "success": rate([bool(r["success"]) for r in rs]),
                                "use_in_code": rate([bool(r["uses_others_artifacts"]) for r in rs]),
                                "report_blocked": rate([bool(r["report_blocked"]) for r in rs]),
                                "success_given_use": rate([bool(r["success"]) for r in rs
                                                           if r["uses_others_artifacts"]]),
                                "success_without_use": rate([bool(r["success"]) for r in rs
                                                             if not r["uses_others_artifacts"]])}
                            for c, rs in by.items()}
    OUT.write_text(json.dumps(out, indent=1))
    for m, cs in out["models"].items():
        for c, x in cs.items():
            s, u = x["success"], x["use_in_code"]
            print(f"{m:20s} {c:20s} n={x['n']:3d} success {s['k']}/{s['n']} use-in-code "
                  f"{u['k']}/{u['n']} success|no use {x['success_without_use']['k']}/"
                  f"{x['success_without_use']['n']}")


if __name__ == "__main__":
    main()
