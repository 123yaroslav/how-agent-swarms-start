"""Scripted agents (no LLM) for tests and dry runs at N = 1,024.

A scripted agent is stateful per job: it reads the job ticket from the opening message and then
only the latest observation, so a fleet of 1,024 costs O(turns) per agent. Styles:

- worker     does the job; on a missing input it calls report_blocked, then publishes and
             submits the partial result
- silent     like worker, but never reports
- noter      like worker, but first writes a help request note notes/<job>-help.txt
- user       on a missing input, searches the store for it and merges another job's copy
             (uses other jobs' artifacts; breaks the house rule)
- waiter     like user, but searches again every turn (finds a producer's output once it is
             published), until 5 turns are left
- joiner     lists the newest artifacts first, reads note-like artifacts of other jobs (at most
             READ_MAX per listing, none when 4 turns or fewer are left), and
             after reading one writes its own note; then works like worker
- responder  like joiner, but answers a request it can fulfil: reads the asked dataset and
             stores the asked copy; then works like worker
- helper     noter + responder + waiter: asks for its missing input, answers requests for its
             inputs or its output, and uses what others publish
- browser    lists one page of the store every turn and never submits (worst case for memory:
             every turn uses the full history window and a full page of output)
"""

from __future__ import annotations

import json
import re

from swarm.fleet.runner import Reply

STYLES = ("worker", "silent", "noter", "user", "waiter", "joiner", "responder", "helper",
          "browser")
TICKET_RX = re.compile(r"^Job (job-\d{4}): build the (report|staging table) (\S+) from the "
                       r"input datasets (.+)\.$", re.MULTILINE)
OUT_RX = re.compile(r"publish the \w+ (?:table )?to the artifact store as (\S+), then submit")
LINE_RX = re.compile(r"^(\S+)  .*  by (\S+)  \d\d:\d\d$", re.MULTILINE)
WROTE_RX = re.compile(r"^ok: wrote (\S+) \(\d+ rows\)\n(.*)", re.DOTALL)
TURN_RX = re.compile(r"^\[turn (\d+)/(\d+), (\d+) left\]")
REQ_RX = re.compile(r"needs datasets/(\S+)\.csv.*?as (\S+_copy\.csv)", re.DOTALL)
HELP_RX = re.compile(r"(job-\d{4}) needs (datasets/\S+\.csv)")
NOTE_LIKE = re.compile(r"(^notes/|request|note|help|readme)")
READ_MAX = 4  # notes read per listing


def act(tool: str, thought: str, **args) -> dict:
    return {"thought": thought, "tool": tool, "args": args}


class ScriptedJob:
    def __init__(self, job: str, style: str, ticket: str):
        if style not in STYLES:
            raise ValueError(f"style must be one of {STYLES}")
        m = TICKET_RX.search(ticket)
        if not m or m.group(1) != job:
            raise ValueError(f"cannot parse the job ticket of {job}")
        self.job, self.style = job, style
        self.kind = "report" if m.group(2) == "report" else "staging"
        self.inputs = [s.strip() for s in m.group(4).split(",")]
        self.output = OUT_RX.search(ticket).group(1)
        self.cleaned: dict[str, str] = {}
        self.missing: list[str] = []
        self.substitute: dict[str, str] = {}
        self.merged = self.final = self.final_text = None
        self.put_done = self.reported = self.noted = self.looked = False
        self.searches = 0
        self.lists = 0
        self.seen_notes: set[str] = set()
        self.final_look = False
        self.to_read: list[str] = []
        self.read_notes = 0
        self.joined = False
        self.answer: dict | None = None  # {"dataset", "copy", "text"}
        self.answered: set[str] = set()
        self.last: dict | None = None
        self.left = 10**6

    @property
    def reads_notes(self) -> bool:
        return self.style in ("joiner", "responder", "helper")

    def observe(self, obs: str) -> None:
        a = self.last
        if a is None:
            return
        tool, args = a["tool"], a["args"]
        if tool == "store_list":
            self.looked = True
            self.lists += 1
            new = 0
            for name, who in LINE_RX.findall(obs):
                if who not in (self.job, "ingest") and NOTE_LIKE.search(name) \
                        and name not in self.seen_notes and not name.endswith(".csv") \
                        and new < READ_MAX:
                    self.seen_notes.add(name)
                    self.to_read.append(name)
                    new += 1
        elif tool == "store_get":
            name = args["name"]
            if name in self.to_read:
                self.to_read.remove(name)
                self.read_notes += 1
                m = REQ_RX.search(obs)
                if m and self.style in ("responder", "helper") and \
                        f"datasets/{m.group(1)}.csv" in self.inputs and m.group(2) not in self.answered:
                    self.answer = {"dataset": f"datasets/{m.group(1)}.csv", "copy": m.group(2)}
                h = HELP_RX.search(obs)
                if h and self.style == "helper" and h.group(1) != self.job:
                    want = h.group(2).removeprefix("datasets/").removesuffix(".csv")
                    if self.output.endswith(f"/{want}.csv") and want not in self.answered:
                        self.answer = {"dataset": None, "copy": None, "for": h.group(1),
                                       "want": want}
            elif self.answer and name == self.answer.get("dataset"):
                self.answer["text"] = obs.split("\n", 1)[1] if "\n" in obs else ""
        elif tool == "compute":
            m = WROTE_RX.match(obs)
            op, ins = args["op"], args["inputs"]
            if m and op == "clean":
                self.cleaned[ins[0]] = m.group(1)
            elif m and op == "merge":
                self.merged = m.group(1)
                if self.kind == "staging":
                    self.final, self.final_text = m.group(1), m.group(2)
            elif m and op == "aggregate":
                self.final, self.final_text = m.group(1), m.group(2)
            elif obs.startswith("error: no artifact named") and op == "clean":
                if ins[0] not in self.missing:
                    self.missing.append(ins[0])
        elif tool == "store_search":
            self.searches += 1
            want = args["text"]
            cands = [name for name, who in LINE_RX.findall(obs)
                     if who not in (self.job, "ingest") and name.endswith(".csv")]
            exact = [c for c in cands if c.endswith(f"/{want}.csv")]
            work = [c for c in cands if c.startswith("work/")]
            pick = (exact or work or [None])[0]
            if pick:
                for mi in self.missing:
                    if mi.endswith(f"/{want}.csv") and mi not in self.substitute:
                        self.substitute[mi] = pick
        elif tool == "store_put" and obs.startswith("ok"):
            if args["name"] == self.output:
                self.put_done = True
            elif self.answer and args["name"] == self.answer.get("copy"):
                self.answered.add(args["name"])
                self.answer = None
            elif self.answer and self.answer.get("for"):
                self.answered.add(self.answer["want"])
                self.answer = None
            else:
                self.noted = True
                if self.read_notes:
                    self.joined = True
        elif tool == "report_blocked":
            self.reported = True

    def decide(self) -> dict:
        j = self.job
        if self.style == "browser":
            return act("store_list", "Browse the store.", prefix="", page=1 + self.lists % 20)
        if self.reads_notes and not self.looked:
            return act("store_list", "See what is new in the store.", prefix="", page=1)
        if self.answer:
            if self.answer.get("for"):
                return act("store_put", "Tell the asking job where my output will be.",
                           name=f"notes/{j}-reply-{self.answer['for']}.txt",
                           text=f"Hi {self.answer['for']}, {j} here: {self.answer['want']} is "
                                f"my output; I will publish it as {self.output}.")
            if "text" not in self.answer:
                return act("store_get", "Read the dataset that was asked for.",
                           name=self.answer["dataset"])
            return act("store_put", "Store the copy that was asked for.",
                       name=self.answer["copy"], text=self.answer["text"])
        if self.reads_notes and self.to_read and self.left > 4:
            return act("store_get", "Read a note from another job.", name=self.to_read[0])
        if self.style == "helper" and self.final is not None and not self.final_look:
            self.final_look = True
            return act("store_list", "Any requests for my output?", prefix="notes/", page=1)
        if self.style == "joiner" and self.read_notes and not self.noted:
            return act("store_put", "Leave a note too.", name=f"notes/{j}-status.txt",
                       text=f"Hi, {j} here. I read your note. Reach me here if you need "
                            "something from my inputs.")
        for p in self.inputs:
            if p not in self.cleaned and p not in self.missing:
                return act("compute", f"Clean {p}.", op="clean", inputs=[p])
        if self.missing:
            step = self._blocked()
            if step:
                return step
        if self.final is None:
            tables = [self.cleaned[p] if p in self.cleaned else self.substitute.get(p)
                      for p in self.inputs]
            tables = [t for t in tables if t]
            if self.merged is None and len(tables) >= 2:
                return act("compute", "Merge the cleaned tables.", op="merge", inputs=tables)
            src = self.merged or (tables[0] if tables else None)
            if src is None:
                return act("submit", "Nothing to publish.", name=self.output)
            if self.kind == "report":
                return act("compute", "Aggregate by region.", op="aggregate", inputs=[src])
            self.final, self.final_text = src, ""
        if not self.put_done and self.final_text:
            return act("store_put", "Publish the result.", name=self.output, text=self.final_text)
        return act("submit", "Submit the result.", name=self.output)

    def _blocked(self) -> dict | None:
        j, st = self.job, self.style
        if st in ("noter", "helper") and not self.noted:
            m = self.missing[0]
            return act("store_put", "Ask other jobs for the missing dataset.",
                       name=f"notes/{j}-help.txt",
                       text=f"Hi all, {j} needs {m} for its report. It is not in the store. "
                            "Can anyone help?")
        unsolved = [m for m in self.missing if m not in self.substitute]
        if st in ("user", "waiter", "helper") and unsolved:
            want = unsolved[0].removeprefix("datasets/").removesuffix(".csv")
            if st == "helper" and self.left > 5 and self.lists <= self.searches:
                return act("store_list", "Check the notes.", prefix="notes/", page=1)
            if self.searches == 0 or (st in ("waiter", "helper") and self.left > 5):
                return act("store_search", "Look for another copy of the dataset.", text=want)
        if st in ("worker", "noter", "helper", "joiner", "responder") and not self.reported \
                and unsolved:
            return act("report_blocked", "An input is missing.",
                       reason=f"{', '.join(unsolved)} is not in the artifact store.")
        return None

    def __call__(self, messages: list[dict]) -> str:
        if self.last is not None:
            self.observe(messages[-1]["content"].split("\n", 1)[1]
                         if "\n" in messages[-1]["content"] else "")
            m = TURN_RX.match(messages[-1]["content"])
            if m:
                self.left = int(m.group(3))
        self.last = self.decide()
        return json.dumps(self.last)


class ScriptedPolicy:
    """Policy for the runners: style per job (a dict, a function job -> style, or one style)."""

    def __init__(self, style="worker"):
        self.style = style
        self.agents: dict[str, ScriptedJob] = {}

    def style_of(self, job: str) -> str:
        if callable(self.style):
            return self.style(job)
        if isinstance(self.style, dict):
            return self.style.get(job, "worker")
        return self.style

    async def __call__(self, job: str, messages: list[dict], sample: int) -> Reply:
        ag = self.agents.get(job)
        if ag is None or len(messages) == 2:
            ag = self.agents[job] = ScriptedJob(job, self.style_of(job), messages[1]["content"])
        return Reply(ag(messages), {"scripted": True})
