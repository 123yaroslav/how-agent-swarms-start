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

# Version 2 (2026-11-08): patterns added from the development half of the real framework prompts
# (experiments/lint_eval.py, projects with an even md5 hash); the test half was not looked at.
LINT_VERSION = 2
PLACE2 = (r"(?:" + SHARED_PLACE + r"|state keys?|shared (?:memory|context|state|workspace|task list|"
          r"notes?|scratchpad|files?)|memory (?:folder|store|bank)|blackboard|"
          r"(?:same|one|common) (?:working )?(?:directory|workspace|folder)|team'?s? (?:shared )?memory)")
OTHERS2 = (r"(?:" + OTHERS + r"|(?:other )?sub-?agents?|team members?|other members|"
           r"by you or other|other (?:roles|assistants|models))")
IO_NEAR_PLACE = (r"\b(?:store|save|write|record|put|add|update|retrieve|read|review|search|recall|"
                 r"check|look up|fetch|load|publish)\b[^.\n]{0,80}" + PLACE2)
# documents or files made by or for another named role ("based on note-taker's outlines")
ROLE_ARTIFACT = (r"\b(?:read|write|edit|create|use)\b[^.\n]{0,40}\b(?:documents?|files?|outlines?|"
                 r"designs?|reports?|notes?)\b[^.\n]{0,40}\b(?:for|from|based on|written by|by) "
                 r"(?!(?:the |a |an |your |another |other )?(?:user|customer|reader|human|developer|"
                 r"caller|requester|they|them|operator)\b)"
                 r"(?:the |another |other )?[\w-]+(?: [\w-]+)?(?:'s|er|or|ist)\b")
# memory or notes that other agents recall or read
MEMORY_FOR_OTHERS = (r"\b(?:memory|memories|notes?|state)\b[^.\n]{0,80}\b(?:recalled|read|used|seen|"
                     r"accessed|available)\b[^.\n]{0,40}")
MEMORY_FOR_OTHERS += OTHERS2
PLACE_FOR_OTHERS = PLACE2 + r"[^.\n]{0,100}" + OTHERS2 + "|" + OTHERS2 + r"[^.\n]{0,100}" + PLACE2

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
    has_shared = (any(re.search(x, prompt, re.I) for x in (PLACE2, ROLE_ARTIFACT, MEMORY_FOR_OTHERS))
                  or any(SHARED_WRITE_TOOL_NAMES.search(t) for t in tools))
    ev = _first(PERMISSION_PATTERNS, prompt)
    if not ev and re.search(OTHERS2, prompt, re.I):
        # document level: the agent reads or writes a shared place, and other agents are named
        ev = _first([PLACE_FOR_OTHERS, MEMORY_FOR_OTHERS, IO_NEAR_PLACE], prompt)
    if not ev:
        ev = _first([ROLE_ARTIFACT], prompt)
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
    ap.add_argument("--llm", default="adaption/deepseek-v4-flash",
                    help="model for the rubric check (the default mode: on held-out framework "
                         "prompts the regex rules found 0 of 4 contact prompts, the rubric 3 of 4, "
                         "docs/ideas/lint_eval.md); 'none' runs the regex rules only")
    a = ap.parse_args()
    prompt = Path(a.prompt).read_text()
    tools: list[str] = []
    if a.tools:
        p = Path(a.tools)
        tools = json.loads(p.read_text() if p.exists() else a.tools)
    fs = lint(prompt, tools)
    if a.llm != "none":
        import asyncio

        from swarm.swarmcheck.lint_llm import judge, level
        j = asyncio.run(judge(prompt, a.llm))
        have = {(f.level, f.rule) for f in fs}
        for lv, rule in level(j):
            if (lv, rule) not in have:
                fs.append(Finding(lv, rule, "rubric: " + (j.get("evidence") or rule)))
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
