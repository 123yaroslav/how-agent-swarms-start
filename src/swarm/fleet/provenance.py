"""Provenance store: who created every file and who listed, read or wrote what, and when.

The log lives outside the model's view (observations never show owners or log entries). Each
event is a flat dict:

    seq      global order of the event (int, starts at 1; initial files have seq 0)
    step     the acting agent's turn index (0-based; -1 for initial files)
    agent    acting agent (for "init": the file's creator)
    op       init | ls | cat | run_input | run_output | run_busy | run_fail | write | append |
             write_denied | note_list | note_read | note_write | note_append | team_post |
             team_post_empty | team_read | submit | report_blocked | bad_tool
    path     normalized target path (notes: /svc/notes/<id>; team posts: /svc/team/post-NNN;
             team_read: /svc/team)
    owner    creator of the target at the time of the event (None if it did not exist;
             for write/append: the creator before the write)
    place    own_home | other_home | shared_cache | sanctioned | notes | team | host | none
    found    whether the target existed (reads and lists)

Optional fields: created (bool), listed_owners (list), seen (tk-/fr- strings and codewords
shown to the agent), text (written text, first 2000 chars), input, stage, item, answer, reason,
tool (bad_tool), seed_note (True on reads, writes and appends that target player-99's seed
note), posts (team_read: [{path, owner, seen}] of the posts shown).
"""

from __future__ import annotations

import re

TOKEN_RX = re.compile(r"\b(?:tk|fr)-[0-9a-f]{6}\b")
CODEWORD_RX = re.compile(r"\b(?:[BDFGKLMNPRSTVZ][AEIOU]){3}\b")
READ_OPS = ("cat", "run_input", "note_read")
WRITE_OPS = ("write", "append", "note_write", "note_append")
TEAM_OPS = ("team_post", "team_read")  # the team channel, kept apart from shared-place notes
SHARED_PLACES = ("shared_cache", "sanctioned", "notes")


def seen_strings(text: str) -> list[str]:
    """Tokens, fragments and codeword-like strings in a text, in order of first appearance."""
    out: list[str] = []
    for m in sorted([*TOKEN_RX.finditer(text), *CODEWORD_RX.finditer(text)],
                    key=lambda m: m.start()):
        if m.group(0) not in out:
            out.append(m.group(0))
    return out


class ProvenanceLog:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self._seq = 0

    def record(self, *, step: int, agent: str, op: str, path: str | None = None,
               owner: str | None = None, place: str = "none", found: bool | None = None,
               init: bool = False, **extra) -> dict:
        if not init:
            self._seq += 1
        ev = {"seq": 0 if init else self._seq, "step": step, "agent": agent, "op": op,
              "path": path, "owner": owner, "place": place, "found": found}
        ev.update({k: v for k, v in extra.items() if v is not None})
        self.events.append(ev)
        return ev

    def by_agent(self, agent: str) -> list[dict]:
        return [e for e in self.events if e["agent"] == agent]
