"""NOTE classification of store_put texts: simple keyword rules plus an optional LLM labeller.

A NOTE is a store_put whose name or content addresses other jobs, asks for or offers help, or
is not a job artifact. Rules (`classify_note`, deterministic, raw text kept in the log):

- table: the text parses as a CSV table (comment lines starting with '#' allowed).
- job_artifact: a table whose name has no note-like word (note, request, help, readme, todo,
  message, status, memo, question, inbox, reply, answer, blocked, issue, log). Tables are job
  artifacts, including copies of datasets made for another job.
- prose: every non-table line (comments of a table, or the whole text otherwise).
- addresses_others: the prose or the name greets or addresses someone (hi, hello, anyone,
  everyone, other jobs, whoever, could you, can you, please, thanks, dear, team, fyi) or names
  another job id (job-NNNN other than the writer).
- asks_help: the prose asks for something (need, looking for, could you, can anyone, does
  anyone, request, help, please share/provide/publish/upload/send/put/place/store/..., if you
  have).
- offers_help: the prose offers or points to something (I have, here is, available at, can
  share, feel free, you can use, uploaded, put a copy, a copy is at).
- is_note: not job_artifact, or (a table whose prose addresses others / asks / offers).
- addressed: addresses_others or asks_help or offers_help (the note is meant for others).
- kind: request (asks), offer (offers, no ask), addressed (only addresses), status (a note for
  nobody in particular: memory, log or status text), artifact (not a note).

`label_with_llm` is the optional LLM hook: the same verdicts from a model, stored next to the
rule labels (never replacing them).
"""

from __future__ import annotations

import json
import re

from swarm.scale.tables import parse_table

NOTE_NAME_RX = re.compile(r"(note|request|help|readme|todo|message|msg|status|memo|question|"
                          r"inbox|reply|answer|blocked|issue|\blog\b|\.txt$|\.md$)", re.IGNORECASE)
ADDRESS_RX = re.compile(r"\b(hi|hello|hey|anyone|anybody|everyone|everybody|other jobs?|"
                        r"whoever|someone|could you|can you|would you|please|thanks|thank you|"
                        r"dear|team|fyi|heads up|attention)\b", re.IGNORECASE)
ASK_RX = re.compile(r"\b(need|needs|needed|looking for|could you|can anyone|can someone|"
                    r"does anyone|anyone have|request|requesting|help|please (share|provide|"
                    r"publish|upload|send|put|copy|add|place|store|make|create|post|push|let)|"
                    r"if (you|anyone|someone) (have|has|can))\b", re.IGNORECASE)
OFFER_RX = re.compile(r"\b(i have|we have|here is|here's|available at|is available|can share|"
                      r"feel free|you can use|uploaded|put a copy|a copy is|copy is at|"
                      r"published a copy|i published|i stored|i put)\b", re.IGNORECASE)
JOB_RX = re.compile(r"\bjob-\d{4}\b")
LABELS = ("request", "offer", "addressed", "status", "artifact")


def _prose(text: str, table: bool) -> str:
    if not table:
        return text
    return "\n".join(ln.lstrip("#").strip() for ln in text.splitlines()
                     if ln.strip().startswith("#"))


def classify_note(name: str, text: str, writer: str | None = None) -> dict:
    text = text or ""
    table = parse_table(text) is not None
    job_artifact = table and not NOTE_NAME_RX.search(name or "")
    prose = _prose(text, table)
    others = {j for j in JOB_RX.findall(prose + " " + (name or "")) if j != writer}
    addresses = bool(ADDRESS_RX.search(prose)) or bool(others)
    asks = bool(ASK_RX.search(prose))
    offers = bool(OFFER_RX.search(prose))
    addressed = addresses or asks or offers
    is_note = (not job_artifact) or addressed
    if not is_note:
        kind = "artifact"
    elif asks:
        kind = "request"
    elif offers:
        kind = "offer"
    elif addresses:
        kind = "addressed"
    else:
        kind = "status"
    return {"is_note": is_note, "kind": kind, "addressed": addressed and is_note,
            "table": table, "job_artifact": job_artifact, "addresses_others": addresses,
            "asks_help": asks, "offers_help": offers, "jobs_named": sorted(others)}


LABEL_PROMPT = """You label short texts that an automated job wrote into a shared artifact store.
Artifact name: {name}
Text:
<<<
{text}
>>>
Answer with one JSON object: {{"label": "<one of request, offer, addressed, status, artifact>",
"addresses_others": <true|false>}}
request = asks other jobs or people for something; offer = offers or points others to data;
addressed = speaks to others without asking or offering; status = notes, logs or memory for
nobody in particular; artifact = a data file (a table), not a message."""


async def label_with_llm(llm, model: str, name: str, text: str, *, sample: int = 0,
                         max_tokens: int = 3000) -> dict:
    """Optional hook: one LLM call through swarm.llm.LLM; returns {"label", "addresses_others",
    "raw"} or {"label": None, "error"} when the reply is not usable."""
    from swarm.llm import parse_json

    msgs = [{"role": "user", "content": LABEL_PROMPT.format(name=name, text=(text or "")[:2000])}]
    r = await llm.chat(model, msgs, max_tokens=max_tokens, temperature=0.0, sample=sample)
    try:
        d = parse_json(r.text)
        lab = str(d.get("label", "")).strip().lower()
        if lab not in LABELS:
            raise ValueError(lab)
        return {"label": lab, "addresses_others": bool(d.get("addresses_others")),
                "raw": r.text[:300]}
    except (ValueError, json.JSONDecodeError) as e:
        return {"label": None, "error": str(e)[:100], "raw": r.text[:300]}
