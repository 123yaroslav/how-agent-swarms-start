"""PILOT (throwaway, environments web_scraping and media_processing only).

Extracts deduplicated trajectories of the two pilot environments from the LinuxArena public sample
into results/raw/pilot_t1/trajs.jsonl. The other 8 environments are skipped and never written.
Untrusted data: parsed with json only, nothing is executed. Long fields are capped at 8000 chars.

    uv run python -I experiments/pilot_t1_extract.py
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "linuxarena" / "sample.jsonl"
OUT = ROOT / "results" / "raw" / "pilot_t1" / "trajs.jsonl"
ENVS = {"web_scraping", "media_processing"}
FAM = {"Haiku honest": "haiku_honest", "Haiku attack": "haiku_attack",
       "GPT-5 Nano monitor on Haiku attack": "haiku_attack", "GPT-5 Nano monitor on straj": "straj",
       "Opus monitor on straj": "straj", "GPT-5 Nano monitor on Opus honest": "opus_honest",
       "Opus monitor on Opus honest": "opus_honest"}
CAP = 8000


def s(x, cap=CAP):
    x = x if isinstance(x, str) else json.dumps(x)
    return x if len(x) <= cap else x[:cap] + f"...[{len(x) - cap} chars cut]"


def main() -> None:
    seen, n = set(), 0
    with SRC.open() as f, OUT.open("w") as w:
        for line in f:
            d = json.loads(line)
            env = d["environment"]["id"]
            if env not in ENVS:
                continue
            acts = d["actions"] or []
            h = hashlib.sha1(json.dumps([(a.get("function"), a.get("arguments")) for a in acts],
                                        sort_keys=True).encode()).hexdigest()
            fam = FAM[d["run_display_name"]]
            if (fam, h) in seen:
                continue
            seen.add((fam, h))
            st = d.get("side_task") or None
            aa = d.get("attack_analysis") or None
            rec = {
                "id": d["trajectory_id"], "fam": fam, "env": env,
                "env_desc": s(d["environment"].get("short_description") or "", 600),
                "main_task": d["main_task"]["id"],
                "main_desc": s(d["main_task"].get("short_description")
                               or d["main_task"].get("description") or "", 1200),
                "side_task": st["id"] if st else None,
                "side_goal": s(st["goal"], 1500) if st else None,
                "side_scoring": s(st.get("scoring_details") or "", 1500) if st else None,
                "side_task_success": d["side_task_success"],
                "incriminating": [x["action_index"] for x in aa["incriminating_actions"]] if aa else None,
                "actions": [{"fn": a.get("function"), "args": s(a.get("arguments")),
                             "out": s(a.get("output") or a.get("output_error") or ""),
                             "reasoning": s(a.get("agent_reasoning") or "", 1500)} for a in acts],
            }
            w.write(json.dumps(rec) + "\n")
            n += 1
    print("wrote", n, "trajectories to", OUT)


if __name__ == "__main__":
    main()
