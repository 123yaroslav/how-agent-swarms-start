"""PILOT (throwaway). Topic 3 feasibility: extract genuine (command, output) pairs.

Source: nebius/SWE-rebench-openhands-trajectories (HF datasets-server rows API, 60 rows from
10 offsets, saved by hand to data/raw/swe_rebench_oh/rows_*.json). Untrusted data: only parsed
as JSON, never executed. Run with:  uv run python -I experiments/pilot_t3_data.py

Writes results/raw/pilot_t3/pairs.jsonl (one genuine pair per line, with prior context).
"""

from __future__ import annotations

import collections
import glob
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "swe_rebench_oh"
OUT = ROOT / "results" / "raw" / "pilot_t3"

FOOTER = re.compile(r"\n?\[The command completed with exit code (-?\d+)\.\][\s\S]*$")
TARGET = {"pytest": 30, "python_run": 25, "grep": 25, "git": 20, "find": 20, "ls": 15,
          "pip": 10, "other": 10}
TYPES = sorted(TARGET)
# 30% of command types held out from any tuning (thresholds, judge prompt edits, model fits).
HELD_OUT = sorted(random.Random(0).sample(TYPES, k=round(0.3 * len(TYPES))))


def core(cmd: str) -> str:
    """Command without leading `cd X &&` and env assignments."""
    c = cmd.strip()
    while c.startswith("cd ") and "&&" in c:
        c = c.split("&&", 1)[1].strip()
    while re.match(r"^[A-Z_]+=\S*\s", c):
        c = c.split(None, 1)[1] if len(c.split(None, 1)) > 1 else ""
    return c


def ctype(cmd: str) -> str | None:
    c = core(cmd)
    if not c:
        return None
    first = c.split()[0]
    if "|" in c and first not in ("grep", "find", "ls", "git", "pip"):
        return None
    if "pytest" in c or "runtests" in c or "unittest" in c:
        return "pytest"
    if first in ("python", "python3"):
        if "<<" in c:
            return None
        return "python_run"
    if first in ("grep", "rg"):
        return "grep"
    if first == "git":
        return "git"
    if first == "find":
        return "find"
    if first == "ls":
        return "ls"
    if first in ("pip", "pip3") or c.startswith("python -m pip"):
        return "pip"
    if first in ("wc", "cat", "head", "tail", "which", "pwd", "du", "ps", "date", "apt-cache",
                 "dpkg-query", "env", "uname", "stat", "sha256sum", "md5sum", "file", "df"):
        if "<<" in c or ">" in c:
            return None
        return "other"
    return None


def step_text(m_call: dict, out: str) -> str:
    fn = m_call["function"]
    try:
        args = json.loads(fn["arguments"])
    except Exception:  # noqa: BLE001
        args = {}
    if fn["name"] == "execute_bash":
        act = f"$ {args.get('command', '')}"
    else:
        act = f"[{fn['name']}] " + json.dumps({k: str(v)[:150] for k, v in args.items()})
    body = FOOTER.sub("", out or "")
    if len(body) > 400:
        body = body[:300] + "\n...[truncated]...\n" + body[-100:]
    return f"{act}\n{body}"


def main() -> None:
    rng = random.Random(1)
    cand: dict[str, list[dict]] = collections.defaultdict(list)
    for f in sorted(glob.glob(str(RAW / "rows_*.json"))):
        for r in json.load(open(f))["rows"]:
            row = r["row"]
            t = row["trajectory"]
            steps = []  # (call, output) in order
            for i, m in enumerate(t):
                for tc in m.get("tool_calls") or []:
                    out = next((x["content"] for x in t[i + 1:i + 4]
                                if x.get("role") == "tool" and x.get("tool_call_id") == tc["id"]),
                               None)
                    if out is not None:
                        steps.append((tc, out))
            ws = re.search(r"<uploaded_files>\s*(\S+)\s*</uploaded_files>", t[1]["content"] or "")
            for k, (tc, out) in enumerate(steps):
                if tc["function"]["name"] != "execute_bash":
                    continue
                try:
                    cmd = json.loads(tc["function"]["arguments"]).get("command", "")
                except Exception:  # noqa: BLE001
                    continue
                ty = ctype(cmd)
                if ty is None or "C-c" in cmd:
                    continue
                m = FOOTER.search(out)
                if not m or "[Below is the output of the previous command" in out:
                    continue
                body = FOOTER.sub("", out)
                if not (40 <= len(body) <= 3000):
                    continue
                prior = [step_text(c, o) for c, o in steps[max(0, k - 4):k]]
                cand[ty].append({
                    "traj": row["trajectory_id"], "repo": row["repo"],
                    "workspace": ws.group(1) if ws else "", "step": k, "type": ty,
                    "command": cmd, "output": body, "exit_code": int(m.group(1)),
                    "context": prior,
                })
    pairs = []
    for ty in TYPES:
        pool = cand[ty]
        rng.shuffle(pool)
        per_traj = collections.Counter()
        chosen = []
        for p in pool:
            if per_traj[p["traj"]] >= 3:
                continue
            per_traj[p["traj"]] += 1
            chosen.append(p)
            if len(chosen) >= TARGET[ty]:
                break
        print(f"{ty:11s} candidates {len(pool):4d} chosen {len(chosen):3d} "
              f"trajs {len(per_traj):2d} {'HELD-OUT' if ty in HELD_OUT else ''}")
        pairs += chosen
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "pairs.jsonl", "w") as fh:
        for i, p in enumerate(pairs):
            p["id"] = f"p{i:03d}"
            p["held_out"] = p["type"] in HELD_OUT
            fh.write(json.dumps(p) + "\n")
    print(f"total {len(pairs)} pairs from {len({p['traj'] for p in pairs})} trajectories; "
          f"held-out types {HELD_OUT}")


if __name__ == "__main__":
    main()
