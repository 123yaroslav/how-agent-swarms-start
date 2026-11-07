"""In-memory artifact store with a provenance log that the agents never see.

Every store operation of a live job (list, search, get, put, compute, submit) goes through
`ArtifactStore` methods that record one event. Event fields:

    seq    global order (int, from 1)
    step   the acting job's turn (0-based)
    job    acting job id
    op     list | search | get | put | put_denied | compute | compute_fail | submit |
           submit_fail | report_blocked | bad_tool | malformed
    name   target artifact name (get, put, submit; compute: the output)
    owner  creator of the target before the event (absent if it did not exist; fields
           whose value is None are left out)
    found  the target existed (get, submit)

Optional: prefix, page, text (search text; put: the written text, first 4000 chars),
n_shown and shown (list/search: number of names shown and the (name, owner) pairs of shown
artifacts that are not ingest datasets or the actor's own), inputs (compute: [{name, owner}]),
error, fp (get: fingerprints shown), reason (report_blocked), tool (bad_tool), overwrite (put
on an existing artifact), note (put: note classification), kind (initial kind of the target:
dataset, work, report, staging, log, seed_note; or "live" for artifacts made by live jobs).

`kind` is ground-truth-like metadata of the generator; detectors use only owner, name and
text. Artifacts created by live jobs have kind "live".
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from swarm.scale import tables as T
from swarm.scale.jobs import LIVE_START, World, hhmm

PAGE = 50
SEARCH_MAX = 20
GET_MAX = 3000
PUT_MAX = 20_000


@dataclass(slots=True)
class Artifact:
    name: str
    text: str
    creator: str
    minute: int
    kind: str
    writer: str = ""
    version: int = 1

    def rows(self) -> int:
        t = T.parse_table(self.text)
        return len(t[1]) if t else 0


class ProvenanceLog:
    """Events in memory, optionally streamed to a JSONL sink as they happen."""

    def __init__(self, sink: Callable[[dict], None] | None = None, keep: bool = True):
        self.events: list[dict] = []
        self.sink, self.keep = sink, keep
        self._seq = 0

    def record(self, **ev) -> dict:
        self._seq += 1
        ev = {"seq": self._seq, **{k: v for k, v in ev.items() if v is not None}}
        if self.keep:
            self.events.append(ev)
        if self.sink is not None:
            self.sink(ev)
        return ev


def normalize_name(name) -> str:
    return str(name or "").strip().lstrip("/").strip()


class ArtifactStore:
    def __init__(self, world: World, *, read_only_others: bool = False,
                 log: ProvenanceLog | None = None):
        self.read_only_others = read_only_others
        self.log = log or ProvenanceLog()
        self._a: dict[str, Artifact] = {}  # insertion order = modification order
        for it in world.initial:
            self._a[it.name] = Artifact(it.name, it.text, it.creator, it.minute, it.kind,
                                        it.creator, 1)
        self._work_k: dict[str, int] = {}

    # ---- helpers ----
    def __contains__(self, name: str) -> bool:
        return name in self._a

    def __len__(self) -> int:
        return len(self._a)

    def artifact(self, name: str) -> Artifact | None:
        return self._a.get(normalize_name(name))

    def owner(self, name: str) -> str | None:
        a = self._a.get(name)
        return a.creator if a else None

    @staticmethod
    def _hit(a: Artifact, q: str) -> bool:
        return q in a.name.lower() or q in a.text.lower()

    def _line(self, a: Artifact) -> str:
        rows = a.rows()
        size = f"{rows} rows" if rows else f"{len(a.text)} bytes"
        return f"{a.name}  {size}  by {a.creator}  {hhmm(a.minute)}"

    @staticmethod
    def _other(a: Artifact, job: str) -> bool:
        return a.creator not in (job, "ingest")

    # ---- tools ----
    def list(self, job: str, step: int, prefix="", page=1) -> str:
        prefix = normalize_name(prefix) if prefix not in (None, "/") else ""
        try:
            page = max(1, int(page))
        except (TypeError, ValueError):
            page = 1
        match = [a for a in reversed(self._a.values()) if a.name.startswith(prefix)]
        pages = max(1, -(-len(match) // PAGE))
        shown = match[(page - 1) * PAGE: page * PAGE]
        self.log.record(step=step, job=job, op="list", prefix=prefix, page=page,
                        n_shown=len(shown), n_total=len(match),
                        shown=[[a.name, a.creator] for a in shown if self._other(a, job)])
        if not match:
            return f'no artifacts with prefix "{prefix}"'
        head = (f'{len(match)} artifacts with prefix "{prefix}", newest first, page {page} of '
                f"{pages}:")
        if not shown:
            return head + "\n(no artifacts on this page)"
        return head + "\n" + "\n".join(self._line(a) for a in shown)

    def search(self, job: str, step: int, text="") -> str:
        q = str(text or "").strip().lower()
        if not q:
            self.log.record(step=step, job=job, op="search", text="", n_shown=0, error="empty")
            return "error: give a search text"
        hits = [a for a in reversed(self._a.values()) if self._hit(a, q)]
        shown = hits[:SEARCH_MAX]
        self.log.record(step=step, job=job, op="search", text=q[:200], n_shown=len(shown),
                        n_total=len(hits),
                        shown=[[a.name, a.creator] for a in shown if self._other(a, job)])
        if not hits:
            return f'no artifacts contain "{text}"'
        more = f"\n({len(hits) - len(shown)} more matches)" if len(hits) > len(shown) else ""
        return (f'{len(hits)} artifacts contain "{text}", newest first:\n'
                + "\n".join(self._line(a) for a in shown) + more)

    def get(self, job: str, step: int, name="") -> str:
        name = normalize_name(name)
        a = self._a.get(name)
        if a is None:
            self.log.record(step=step, job=job, op="get", name=name, owner=None, found=False)
            return f"error: no artifact named {name}" if name else "error: give a name"
        body = a.text if len(a.text) <= GET_MAX else a.text[:GET_MAX] + "\n...(truncated)"
        self.log.record(step=step, job=job, op="get", name=name, owner=a.creator, found=True,
                        kind=a.kind, fp=sorted(T.fingerprints(body)) or None)
        return f"{name} (by {a.creator}, {hhmm(a.minute)}):\n{body}"

    def put(self, job: str, step: int, name="", text="", note: dict | None = None) -> str:
        name = normalize_name(name)
        text = "" if text is None else str(text)
        if not name or len(name) > 200:
            self.log.record(step=step, job=job, op="put_denied", name=name[:200],
                            error="bad name", text=text[:4000])
            return "error: give an artifact name (at most 200 characters)"
        if len(text) > PUT_MAX:
            self.log.record(step=step, job=job, op="put_denied", name=name, error="too large",
                            text=text[:4000])
            return f"error: text too large (at most {PUT_MAX} characters)"
        old = self._a.get(name)
        if old is not None and old.creator != job and self.read_only_others:
            self.log.record(step=step, job=job, op="put_denied", name=name, owner=old.creator,
                            kind=old.kind, error="read-only", text=text[:4000], note=note)
            return f"error: permission denied: {name} belongs to another job (read-only)"
        minute = LIVE_START + step
        if old is not None:
            del self._a[name]
            a = Artifact(name, text, old.creator, minute, old.kind, job, old.version + 1)
        else:
            a = Artifact(name, text, job, minute, "live", job, 1)
        self._a[name] = a
        self.log.record(step=step, job=job, op="put", name=name,
                        owner=old.creator if old else None, overwrite=old is not None or None,
                        kind=old.kind if old else "live", text=text[:4000], note=note)
        return f"ok: stored {name} ({len(text)} bytes)"

    def compute(self, job: str, step: int, op="", inputs=None) -> str:
        if isinstance(inputs, str):
            inputs = [inputs]
        if not isinstance(inputs, list) or not inputs:
            self.log.record(step=step, job=job, op="compute_fail", error="no inputs",
                            stage=str(op))
            return 'error: give inputs as a list of artifact names, e.g. ["datasets/x.csv"]'
        names = [normalize_name(n) for n in inputs]
        refs = [{"name": n, "owner": self.owner(n)} for n in names]
        missing = [n for n in names if n not in self._a]
        if missing:
            self.log.record(step=step, job=job, op="compute_fail", stage=str(op), inputs=refs,
                            error="not found")
            return f"error: no artifact named {missing[0]}"
        try:
            out, nrows = T.run_op(str(op), [(n, self._a[n].text) for n in names])
        except T.OpError as e:
            self.log.record(step=step, job=job, op="compute_fail", stage=str(op), inputs=refs,
                            error=str(e)[:200])
            return f"error: {e}"
        k = self._work_k.get(job, 0) + 1
        oname = f"work/{job}/{str(op).strip().lower()}-{k}.csv"
        while oname in self._a:
            k += 1
            oname = f"work/{job}/{str(op).strip().lower()}-{k}.csv"
        self._work_k[job] = k
        self._a[oname] = Artifact(oname, out, job, LIVE_START + step, "live", job, 1)
        self.log.record(step=step, job=job, op="compute", stage=str(op).strip().lower(),
                        inputs=refs, name=oname, fp=sorted(T.fingerprints(out)) or None)
        return f"ok: wrote {oname} ({nrows} rows)\n{out}"

    def submit(self, job: str, step: int, name="") -> tuple[str, dict | None]:
        name = normalize_name(name)
        a = self._a.get(name)
        if a is None:
            self.log.record(step=step, job=job, op="submit_fail", name=name, found=False)
            return f"error: no artifact named {name}", None
        sub = {"name": name, "owner": a.creator, "writer": a.writer, "text": a.text[:8000],
               "step": step}
        self.log.record(step=step, job=job, op="submit", name=name, owner=a.creator,
                        found=True, kind=a.kind, text=a.text[:4000])
        return f"ok: submitted {name}. Job finished.", sub

    def live_artifacts(self):
        """Artifacts made or changed by live jobs (for records), one dict at a time."""
        for a in self._a.values():
            if a.kind == "live" or a.version > 1:
                yield {"name": a.name, "creator": a.creator, "writer": a.writer,
                       "kind": a.kind, "version": a.version, "text": a.text[:2000]}


def jsonl_sink(fh) -> Callable[[dict], None]:
    def sink(ev: dict) -> None:
        fh.write(json.dumps(ev) + "\n")
    return sink
