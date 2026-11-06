"""EXPLORATORY PILOT (Idea B, kin leniency). Not pre-registered. No LLM calls.

Extracts one row per stored monitor verdict from the ScaleAI/mrt SHADE-Arena files downloaded to
data/raw/scaleai_mrt/. Reads JSON only, never imports or executes anything from the dataset.
Output: results/pilot_kin_mrt/rows.csv (no transcript text, only ids, labels and parsed scores).

Run: uv run python -I experiments/pilot_kin_extract.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "scaleai_mrt" / "task_pairs"
OUT = ROOT / "results" / "pilot_kin_mrt"

ATTACK_PREFIX = "transcripts_no_hint_selective_thinking_time"
HONEST_PREFIX = "transcripts_no_secondary_task"
SCAFFOLDS = ["", "_hierarchical_4_likert_reminder_v2", "_sequential_4_likert_reminder_v2",
             "_rubric", "_likert_reminder_v2"]

VERDICT_RE = re.compile(r"<verdict>\s*([-+]?[0-9]*\.?[0-9]+)\s*</verdict>", re.I)
# Loose fallback for sensitivity only, e.g. "Suspicion score: 30/100" or "<verdict>30" (truncated).
LOOSE_RE = re.compile(r"(?:<verdict>|score\s*(?:of|is|:)?)\s*\**\s*([0-9]*\.?[0-9]+)", re.I)
SELF_ID_RE = {
    "claude": re.compile(r"\b(claude|anthropic)\b", re.I),
    "gemini": re.compile(r"\b(gemini|google deepmind|bard)\b", re.I),
    "gpt": re.compile(r"\b(chatgpt|gpt-4|openai)\b", re.I),
}


def family(model: str) -> str:
    m = model.lower()
    if m.startswith("ft:"):
        return "GPT-ft"
    if "claude" in m:
        return "Claude"
    if "gemini" in m:
        return "Gemini"
    if "gpt" in m or m in ("o3", "o4-mini"):
        return "GPT"
    if "deepseek" in m:
        return "DeepSeek"
    return "other"


def parse_cond(cond: str):
    for kind, prefix in (("attack", ATTACK_PREFIX), ("honest", HONEST_PREFIX)):
        if cond.startswith(prefix):
            suffix = cond[len(prefix):]
            if suffix in SCAFFOLDS:
                return kind, suffix.lstrip("_") or "default"
    return None, None


def agent_text(conv) -> str:
    """Assistant-visible text, used only to flag self-identification. Not stored."""
    parts = []
    for m in conv:
        if m.get("role") == "assistant" and isinstance(m.get("content"), str):
            parts.append(m["content"])
    return "\n".join(parts)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW.glob("*/*/*/*/transcript_*.json"))
    fields = ["task", "cond", "kind", "scaffold", "agent", "agent_fam", "monitor", "monitor_fam",
              "idx", "conv_hash", "n_msgs", "side_task_ok", "main_task_eval", "score", "score_loose",
              "parse_ok", "n_verdict_tags", "selfid_claude", "selfid_gemini", "selfid_gpt"]
    n_bad = 0
    with open(OUT / "rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for f in files:
            task, cond, agent, monitor, name = f.parts[-5:]
            kind, scaffold = parse_cond(cond)
            if kind is None:
                continue
            try:
                d = json.loads(f.read_text())
            except Exception:
                n_bad += 1
                continue
            conv = d.get("conversation") or []
            resp = d.get("monitoring_response") or ""
            if not isinstance(resp, str):
                resp = json.dumps(resp)
            tags = VERDICT_RE.findall(resp)
            score = float(tags[-1]) if tags else ""
            loose = LOOSE_RE.findall(resp[-600:])
            score_loose = score if tags else (float(loose[-1]) if loose else "")
            txt = agent_text(conv)
            w.writerow({
                "task": task, "cond": cond, "kind": kind, "scaffold": scaffold,
                "agent": agent, "agent_fam": family(agent),
                "monitor": monitor, "monitor_fam": family(monitor),
                "idx": int(name.split("_")[1].split(".")[0]),
                "conv_hash": hashlib.sha1(json.dumps(conv, sort_keys=True).encode()).hexdigest()[:16],
                "n_msgs": len(conv),
                "side_task_ok": d.get("side_task_evaluation"),
                "main_task_eval": str(d.get("main_task_evaluation"))[:80],
                "score": score, "score_loose": score_loose, "parse_ok": bool(tags), "n_verdict_tags": len(tags),
                "selfid_claude": bool(SELF_ID_RE["claude"].search(txt)),
                "selfid_gemini": bool(SELF_ID_RE["gemini"].search(txt)),
                "selfid_gpt": bool(SELF_ID_RE["gpt"].search(txt)),
            })
    print(f"files scanned {len(files)}, unreadable {n_bad}, wrote {OUT / 'rows.csv'}")


if __name__ == "__main__":
    sys.exit(main())
