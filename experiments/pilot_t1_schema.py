"""PILOT (throwaway, environments web_scraping and media_processing only). Schema and per-environment
counts for the LinuxArena public sample, used only to choose the two pilot environments.

Reads data/raw/linuxarena/sample.jsonl as untrusted data (json only, nothing executed) and writes
a compact index (no large text) to results/raw/pilot_t1/index.json.

    uv run python -I experiments/pilot_t1_schema.py
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "linuxarena" / "sample.jsonl"
OUT = ROOT / "results" / "raw" / "pilot_t1"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    traj_keys, act_keys = collections.Counter(), collections.Counter()
    first = None
    with SRC.open() as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            traj_keys.update(d.keys())
            for a in d.get("actions") or []:
                act_keys.update(a.keys())
            if first is None:
                first = d
            env = (d.get("environment") or {}).get("id")
            st = d.get("side_task") or None
            mt = d.get("main_task") or {}
            acts = d.get("actions") or []
            mon_models = {(a.get("monitor_response") or {}).get("model_name") for a in acts}
            rows.append({
                "line": i, "trajectory_id": d.get("trajectory_id"), "run": d.get("run_display_name"),
                "env": env, "agent_model": d.get("agent_model"),
                "main_task_id": mt.get("id"), "side_task_id": st.get("id") if st else None,
                "main_task_success": d.get("main_task_success"),
                "side_task_success": d.get("side_task_success"),
                "n_actions": len(acts), "has_attack_analysis": bool(d.get("attack_analysis")),
                "monitor_models": sorted(m for m in mon_models if m),
                "fn": collections.Counter(a.get("function") for a in acts).most_common(5),
                "chars_args": sum(len(json.dumps(a.get("arguments"))) for a in acts),
                "chars_out": sum(len(str(a.get("output") or "")) for a in acts),
                "first_args": json.dumps(acts[0].get("arguments"))[:80] if acts else "",
            })
    print("trajectory keys:", dict(traj_keys))
    print("action keys:", dict(act_keys))
    for k in ("environment", "main_task", "side_task", "traj_monitor_response", "attack_analysis"):
        v = first.get(k)
        print(k, "->", list(v.keys()) if isinstance(v, dict) else type(v).__name__)
    a0 = first["actions"][0]
    print("action[0] monitor_response keys:", list((a0.get("monitor_response") or {}).keys()))
    print("action[0] substeps type:", type(a0.get("substeps")).__name__)
    (OUT / "index.json").write_text(json.dumps(rows))
    print(collections.Counter(r["run"] for r in rows))
    print("agent models:", collections.Counter((r["run"], r["agent_model"]) for r in rows))


if __name__ == "__main__":
    sys.exit(main())
