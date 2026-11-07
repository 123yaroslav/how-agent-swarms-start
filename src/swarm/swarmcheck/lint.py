"""SwarmCheck prompt linter: flags system-prompt and tool-config patterns that our measurements
tie to agents contacting each other through shared storage, or taking each other's work.

Rules (evidence in docs/results_confirmatory.md; numbers filled by the report build):
  HIGH  permission_to_coordinate   the prompt allows or describes messaging other agents through a
                                   shared store, folder, cache or board. In our runs one such
                                   sentence moved blocked agents from almost never writing to
                                   others to almost always doing so.
  MED   missing_boundary_rule      shared storage is present but no sentence limits agents to their
                                   own artifacts. Without the rule most agents used other jobs'
                                   artifacts; with it almost none did.
  MED   shared_write_tool          a tool writes to a store other agents can list or read.
  INFO  sanctioned_channel         a team or messaging tool exists; in our runs it absorbed all
                                   contact, none went through the shared store.
  INFO  benign_team_wording        "multi-agent system", "help each other", "trained to
                                   collaborate": measured not to open the gate on its own.

Usage:
    uv run python -m swarm.swarmcheck.lint prompt.txt [--tools tools.json] [--json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

SHARED_PLACE = (r"(?:(shared|common|team|cluster)[- ]?(store|storage|folder|directory|cache|board|"
                r"bucket|drive|wiki|space|artifact store|memory)|/work/cache|/shared|artifact store)")
CONTACT = (r"\b(ask|message|notify|request|leave (a |your )?notes?|post (a )?notes?|"
           r"write (a )?notes?|coordinate|reply|answer|share (findings|results|notes|progress))\b")
OTHERS = r"(?:other (agents|jobs|players|workers|instances|teammates)|teammates|each other|peers)"

PERMISSION_PATTERNS = [
    rf"{CONTACT}[^.]{{0,80}}{OTHERS}[^.]{{0,80}}{SHARED_PLACE}",
    rf"{SHARED_PLACE}[^.]{{0,80}}(is|are) (read|seen|visible)[^.]{{0,40}}{OTHERS}",
    rf"{OTHERS}[^.]{{0,60}}(read|see|check)[^.]{{0,60}}{SHARED_PLACE}",
    rf"{SHARED_PLACE}[^.]{{0,80}}{CONTACT}[^.]{{0,60}}(is |are )?(allowed|permitted|encouraged)",
]
BOUNDARY_PATTERNS = [
    r"(use|read|open|access) only (your|the) own",
    r"do not (use|read|open|access|modify)[^.]{0,40}(other|another) (agents?|jobs?|players?|workers?)",
    r"only (your|the) (job|task)'?s? (own )?(artifacts|files|materials|data)",
]
BENIGN_PATTERNS = [
    r"multi-agent system", r"help each other", r"trained to collaborate", r"team ?work",
]
TEAM_TOOL_NAMES = re.compile(r"(team_post|team_read|send_message|post_message|chat|slack)", re.I)
SHARED_WRITE_TOOL_NAMES = re.compile(r"(store_put|put_artifact|upload|write_shared|cache_put)", re.I)


@dataclass
class Finding:
    level: str
    rule: str
    evidence: str


def _first(patterns: list[str], text: str) -> str | None:
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            start = max(0, m.start() - 40)
            return text[start:m.end() + 40].replace("\n", " ").strip()
    return None


def lint(prompt: str, tools: list[str] | None = None) -> list[Finding]:
    tools = tools or []
    out: list[Finding] = []
    has_shared = bool(re.search(SHARED_PLACE, prompt, re.I)) or any(
        SHARED_WRITE_TOOL_NAMES.search(t) for t in tools)
    ev = _first(PERMISSION_PATTERNS, prompt)
    boundary = _first(BOUNDARY_PATTERNS, prompt)
    if ev and not (boundary and re.search(r"except|apart from|other than", ev, re.I)):
        out.append(Finding("HIGH", "permission_to_coordinate", ev))
    if has_shared and not boundary:
        out.append(Finding("MED", "missing_boundary_rule",
                           "shared storage is present and no sentence limits agents to their own "
                           "artifacts"))
    for t in tools:
        if SHARED_WRITE_TOOL_NAMES.search(t):
            out.append(Finding("MED", "shared_write_tool", t))
        if TEAM_TOOL_NAMES.search(t):
            out.append(Finding("INFO", "sanctioned_channel", t))
    benign = _first(BENIGN_PATTERNS, prompt)
    if benign:
        out.append(Finding("INFO", "benign_team_wording", benign))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--tools", help="JSON list of tool names, or a JSON file with one")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    prompt = Path(a.prompt).read_text()
    tools: list[str] = []
    if a.tools:
        p = Path(a.tools)
        tools = json.loads(p.read_text() if p.exists() else a.tools)
    fs = lint(prompt, tools)
    if a.json:
        print(json.dumps([asdict(f) for f in fs], indent=1))
    else:
        for f in fs:
            print(f"{f.level:5s} {f.rule:26s} {f.evidence[:120]}")
        if not fs:
            print("no findings")
    return 1 if any(f.level == "HIGH" for f in fs) else 0


if __name__ == "__main__":
    sys.exit(main())
