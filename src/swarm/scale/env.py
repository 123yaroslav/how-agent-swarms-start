"""Cluster environment: the tools of the agents on top of one shared ArtifactStore.

`ScaleEnv.apply(job, step, tool, args)` runs one action and returns the observation text. All
effects go through the store, so every action lands in the provenance log. No real shell, no
network: artifacts are strings in a dict.
"""

from __future__ import annotations

from swarm.scale.jobs import World
from swarm.scale.notes import classify_note
from swarm.scale.store import ArtifactStore, ProvenanceLog

ALIASES = {
    "store_list": "store_list", "list": "store_list", "ls": "store_list",
    "store_search": "store_search", "search": "store_search", "grep": "store_search",
    "store_get": "store_get", "get": "store_get", "read": "store_get", "cat": "store_get",
    "store_put": "store_put", "put": "store_put", "write": "store_put",
    "compute": "compute", "run": "compute",
    "submit": "submit", "report_blocked": "report_blocked",
}
TOOL_NAMES = ("store_list", "store_search", "store_get", "store_put", "compute", "submit",
              "report_blocked")


def _arg(args: dict, *keys, default=None):
    for k in keys:
        if k in args and args[k] is not None:
            return args[k]
    return default


class ScaleEnv:
    def __init__(self, world: World, *, read_only_others: bool = False,
                 log: ProvenanceLog | None = None):
        self.world = world
        self.store = ArtifactStore(world, read_only_others=read_only_others, log=log)
        self.log = self.store.log
        self.submissions: dict[str, dict] = {}
        self.reports: dict[str, list[str]] = {}

    def done(self, job: str) -> bool:
        return job in self.submissions

    def apply(self, job: str, step: int, tool: str, args: dict | None) -> str:
        args = args if isinstance(args, dict) else {}
        name = ALIASES.get(str(tool or "").strip().lower().replace(".", "_"))
        s = self.store
        if job in self.submissions:
            return "error: the job is already finished"
        if name == "store_list":
            return s.list(job, step, _arg(args, "prefix", "path", default=""),
                          _arg(args, "page", default=1))
        if name == "store_search":
            return s.search(job, step, _arg(args, "text", "query", "q", "pattern", default=""))
        if name == "store_get":
            return s.get(job, step, _arg(args, "name", "path", "artifact", default=""))
        if name == "store_put":
            n = str(_arg(args, "name", "path", "artifact", default=""))
            text = _arg(args, "text", "content", "data", "body", default="")
            if not isinstance(text, str):
                text = str(text)
            return s.put(job, step, n, text, note=classify_note(n, text, writer=job))
        if name == "compute":
            return s.compute(job, step, _arg(args, "op", "operation", "stage", default=""),
                             _arg(args, "inputs", "input", "names", default=None))
        if name == "submit":
            obs, sub = s.submit(job, step, _arg(args, "name", "artifact", "path", default=""))
            if sub is not None:
                self.submissions[job] = sub
            return obs
        if name == "report_blocked":
            reason = str(_arg(args, "reason", "text", "message", default=""))[:2000]
            self.reports.setdefault(job, []).append(reason)
            self.log.record(step=step, job=job, op="report_blocked", reason=reason)
            return "ok: the operator was told. The job is still running; submit when done."
        self.log.record(step=step, job=job, op="bad_tool", tool=str(tool)[:80])
        return f"error: unknown tool {tool!r}. Tools: {', '.join(TOOL_NAMES)}"

    def malformed(self, job: str, step: int) -> None:
        self.log.record(step=step, job=job, op="malformed")
