"""Async runner for one world (a harness episode or a fleet of up to 1,024 jobs) and for many
harness episodes side by side.

Schedule "rounds" (default): each turn, every active job is queried concurrently (at most
`concurrency` calls in flight, a semaphore shared by everything the runner starts), then the
actions are applied in job order. Deterministic, so a rerun with the same spec replays every
call from the LLM cache: that is how an interrupted run resumes. Schedule "async": independent
job loops; actions land in completion order (faster, not exactly replayable).

Sampling: the sample index of a call is (world seed, repeat, job, turn), not the arm, so arms on
the same world seed replay identical first calls (common random numbers; treat arms as paired).

Memory (one process stays small at N = 1,024): each job keeps only what its next prompt needs,
a bounded history (`history_window` turns, default 12, tool outputs cut to OBS_HIST_MAX
characters) and one summary line for the hidden turns (how many, the last artifacts it stored
or computed, whether it reported). With `out_prefix`, turn records, round records and
provenance events are written to disk as they happen and not kept; features are computed by a
streaming pass over the events file. The store is one shared dict of artifacts; lists and
searches build only the page they show.

Files with out_prefix P: P.turns.jsonl, P.rounds.jsonl, P.events.jsonl, P.artifacts.jsonl
(artifacts made by live jobs), P.record.json (spec, world meta, features, submissions, reports,
rounds, totals; no ground truth), P.labels.json (ground truth, evaluation only).
"""

from __future__ import annotations

import asyncio
import json
import resource
import sys
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from swarm.fleet.parser import parse_action
from swarm.fleet.runner import LLMPolicy, Policy, Reply, sample_index  # noqa: F401 (re-export)
from swarm.scale import prompts
from swarm.scale.env import ScaleEnv
from swarm.scale.jobs import WorldConfig, make_world
from swarm.scale.metrics import job_features
from swarm.scale.store import ProvenanceLog
from swarm.scale.tables import fingerprints

ASSISTANT_MAX = 2000
OBS_HIST_MAX = 3200
DEFAULT_WINDOW = 12


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024  # bytes on macOS, KiB on Linux


@dataclass
class RunSpec:
    tag: str
    mode: str = "fleet"  # fleet | harness
    n_agents: int = 256
    p_blocked: float = 0.3
    prompt: str = "plain"
    rule: bool = True
    read_only: bool = False
    seed_note: str = "none"
    n_background: int = 200
    p_derivative: float = 1.0
    producers_in_fleet: bool = True
    steps: int = 40
    model: str = "adaption/deepseek-v4-flash"
    max_tokens: int = 3000
    temperature: float = 0.8
    seed: int = 0
    repeat: int = 0
    schedule: str = "rounds"
    history_window: int | None = DEFAULT_WINDOW

    def world_config(self) -> WorldConfig:
        harness = self.mode == "harness"
        return WorldConfig(mode=self.mode, n_agents=1 if harness else self.n_agents,
                           p_blocked=1.0 if harness else self.p_blocked,
                           n_background=self.n_background, p_derivative=self.p_derivative,
                           producers_in_fleet=self.producers_in_fleet,
                           seed_note=self.seed_note, seed=self.seed)

    def arm(self) -> str:
        return (f"{self.prompt}-{'rule' if self.rule else 'norule'}"
                f"{'-ro' if self.read_only else ''}")

    def episode_id(self) -> str:
        n = 1 if self.mode == "harness" else self.n_agents
        return (f"{self.tag}/{self.mode}-n{n}-{self.arm()}-note_{self.seed_note}"
                f"-s{self.seed}-r{self.repeat}")


class History:
    """What a job's next prompt needs: system prompt, ticket, the last `window` turns and a
    summary line for the hidden ones."""

    __slots__ = ("begin", "created", "hidden", "pairs", "reported", "system", "window")

    def __init__(self, system: str, begin: str, window: int | None):
        self.system, self.begin, self.window = system, begin, window
        self.pairs: deque = deque(maxlen=window) if window else deque()
        self.hidden = 0
        self.created: deque = deque(maxlen=8)
        self.reported = False

    def add(self, assistant: str, user: str) -> None:
        if self.window and len(self.pairs) == self.window:
            self.hidden += 1
        self.pairs.append((assistant, user))

    def summary(self) -> str:
        made = (" Artifacts you stored or computed so far (latest last): "
                + ", ".join(self.created) + ".") if self.created else ""
        rep = " You have called report_blocked." if self.reported else ""
        return f"(Turns 1-{self.hidden} are hidden.{made}{rep})"

    def messages(self) -> list[dict]:
        begin = self.begin + ("\n" + self.summary() if self.hidden else "")
        msgs = [{"role": "system", "content": self.system}, {"role": "user", "content": begin}]
        for a, u in self.pairs:
            msgs.append({"role": "assistant", "content": a})
            msgs.append({"role": "user", "content": u})
        return msgs


def _cut(obs: str) -> str:
    return obs if len(obs) <= OBS_HIST_MAX else obs[:OBS_HIST_MAX] + "\n...(cut)"


def _iter_jsonl(path: Path):
    with path.open() as f:
        for ln in f:
            if ln.strip():
                yield json.loads(ln)


async def run_world(spec: RunSpec, policy: Policy, *, sem: asyncio.Semaphore | None = None,
                    stop: dict | None = None, out_prefix: Path | None = None,
                    progress: Callable[[dict], None] | None = None) -> tuple[dict, dict]:
    """Run one world. Returns (record, labels). Without out_prefix the record also holds the
    turns and events (small harness episodes); with it, they are streamed to files."""
    if spec.schedule not in ("rounds", "async"):
        raise ValueError("schedule must be rounds or async")
    if spec.history_window is not None and spec.history_window < 1:
        raise ValueError("history_window must be >= 1 or None")
    stop = stop if stop is not None else {}
    sem = sem or asyncio.Semaphore(10**6)
    world = make_world(spec.world_config())
    sn = world.seed_note
    req_fp: set[str] = set()
    if sn and sn["style"] == "request":
        src = next((a for a in world.initial if a.name == f"datasets/{sn['dataset']}.csv"), None)
        req_fp = fingerprints(src.text) if src else set()
    meta = world.meta()
    labels = {"type": "labels", "episode_id": spec.episode_id(), **world.labels_record()}

    files = {}
    if out_prefix is not None:
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        for k in ("turns", "rounds", "events"):
            files[k] = open(f"{out_prefix}.{k}.jsonl", "w")  # noqa: SIM115
    ev_fh = files.get("events")
    log = ProvenanceLog(sink=(lambda e: ev_fh.write(json.dumps(e) + "\n")) if ev_fh else None,
                        keep=ev_fh is None)
    env = ScaleEnv(world, read_only_others=spec.read_only, log=log)
    world.initial = []  # the store holds the artifacts now
    eid = spec.episode_id()
    jobs = world.job_ids
    hist = {j: History(prompts.system_prompt(j, spec.steps, prompt=spec.prompt, rule=spec.rule),
                       prompts.begin_message(world.job(j), spec.steps), spec.history_window)
            for j in jobs}
    turns: list[dict] = []
    rounds: list[dict] = []
    totals = {"calls": 0, "live": 0, "cached": 0, "errors": 0, "malformed": 0, "cost": 0.0,
              "pt": 0, "ct": 0}

    async def query(j: str, t: int) -> tuple[dict, str]:
        rec: dict = {"type": "turn", "episode_id": eid, "job": j, "step": t}
        text = ""
        if stop.get("budget"):
            rec["error"] = "budget"
            return rec, text
        async with sem:
            t0 = time.time()
            try:
                r: Reply = await policy(j, hist[j].messages(),
                                        sample_index(spec.seed, spec.repeat, j, t))
                text = r.text or ""
                rec.update(r.meta)
            except Exception as e:  # noqa: BLE001
                if type(e).__name__ == "BudgetExceeded":
                    stop["budget"] = True
                    rec["error"] = "budget"
                else:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            rec["latency"] = round(time.time() - t0, 3)
        return rec, text

    def apply(j: str, t: int, rec: dict, text: str) -> None:
        a = parse_action(text)
        if a is None:
            rec.update(tool="malformed", malformed=True)
            env.malformed(j, t)
            obs = prompts.BAD_REPLY
            totals["malformed"] += 1
        else:
            rec.update(tool=a["tool"], args=a["args"], malformed=False)
            obs = env.apply(j, t, a["tool"], a["args"])
        h = hist[j]
        if obs.startswith(("ok: stored ", "ok: wrote ")):
            h.created.append(obs.split()[2])
        elif obs.startswith("ok: the operator was told"):
            h.reported = True
        rec["assistant"] = (text or "(no reply)")[:ASSISTANT_MAX]
        rec["user"] = prompts.turn_message(t, spec.steps, _cut(obs))
        h.add(rec["assistant"], rec["user"])
        totals["calls"] += 1
        totals["errors"] += "error" in rec
        if not rec.get("scripted"):
            totals["cached" if rec.get("cached") else "live"] += 1
        totals["cost"] += rec.get("cost", 0.0) or 0.0
        totals["pt"] += rec.get("pt", 0) or 0
        totals["ct"] += rec.get("ct", 0) or 0
        if "turns" in files:
            files["turns"].write(json.dumps(rec) + "\n")
        else:
            turns.append(rec)

    def round_record(t: int, n_active: int, t0: float, recs: list[dict]) -> None:
        wall = time.time() - t0
        lat = [r.get("latency", 0.0) for r in recs]
        rr = {"type": "round", "episode_id": eid, "step": t, "active": n_active,
              "wall_s": round(wall, 3),
              "live": sum(1 for r in recs if not r.get("cached") and not r.get("scripted")
                          and "error" not in r),
              "cached": sum(1 for r in recs if r.get("cached")),
              "errors": sum(1 for r in recs if "error" in r),
              "malformed": sum(1 for r in recs if r.get("malformed")),
              "lat_mean": round(sum(lat) / len(lat), 3) if lat else 0.0,
              "lat_max": round(max(lat), 3) if lat else 0.0,
              "calls_per_min": round(60 * len(recs) / wall, 1) if wall > 0 else None,
              "store_size": len(env.store), "peak_rss_mb": round(peak_rss_mb(), 1),
              "cost": round(sum(r.get("cost", 0.0) or 0.0 for r in recs), 5)}
        rounds.append(rr)
        if "rounds" in files:
            files["rounds"].write(json.dumps(rr) + "\n")
            for fh in files.values():
                fh.flush()
        if progress:
            progress(rr)

    t_start = time.time()
    try:
        if spec.schedule == "rounds":
            for t in range(spec.steps):
                active = [j for j in jobs if not env.done(j)]
                if not active or stop.get("budget"):
                    break
                t0 = time.time()
                res = await asyncio.gather(*(query(j, t) for j in active))
                applied = []
                for j, (rec, text) in zip(active, res):
                    if rec.get("error") == "budget":
                        continue
                    apply(j, t, rec, text)
                    applied.append(rec)
                del res
                round_record(t, len(active), t0, applied)
                del applied
        else:
            async def loop(j: str) -> None:
                for t in range(spec.steps):
                    if stop.get("budget") or env.done(j):
                        return
                    rec, text = await query(j, t)
                    if rec.get("error") == "budget":
                        return
                    apply(j, t, rec, text)
            t0 = time.time()
            await asyncio.gather(*(loop(j) for j in jobs))
            round_record(-1, len(jobs), t0, [])
    finally:
        for fh in files.values():
            fh.close()

    events = log.events if ev_fh is None else _iter_jsonl(Path(f"{out_prefix}.events.jsonl"))
    feats = job_features(events, meta["jobs"], prompt=spec.prompt, rule=spec.rule,
                         seed_note=sn, request_fp=req_fp)
    record = {"type": "episode", "episode_id": eid, "tag": spec.tag, "spec": asdict(spec),
              "world": meta, "features": feats, "submissions": env.submissions,
              "reports": env.reports, "rounds": rounds, "totals": totals,
              "stopped": "budget" if stop.get("budget") else None,
              "request_fp": sorted(req_fp),
              "wall_s": round(time.time() - t_start, 2), "peak_rss_mb": round(peak_rss_mb(), 1)}
    if out_prefix is None:
        record["turns"] = turns
        record["events"] = log.events
        record["live_artifacts"] = list(env.store.live_artifacts())
    else:
        with open(f"{out_prefix}.artifacts.jsonl", "w") as f:
            f.writelines(json.dumps(a) + "\n" for a in env.store.live_artifacts())
        Path(f"{out_prefix}.record.json").write_text(json.dumps(record))
        Path(f"{out_prefix}.labels.json").write_text(json.dumps(labels))
    return record, labels


# ---- many harness episodes ---------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(_iter_jsonl(path))


def labels_path(path: Path) -> Path:
    return path.with_name(path.name.removesuffix(".jsonl") + ".labels.jsonl")


def done_ids(path: Path) -> set[str]:
    """Episode ids in a harness file, read line by line (records are not kept)."""
    if not path.exists():
        return set()
    out = set()
    with path.open() as f:
        for ln in f:
            if ln.strip():
                out.add(json.loads(ln)["episode_id"])
    return out


async def run_many(specs: list[RunSpec], policy_for: Callable[[RunSpec], Policy], out: Path, *,
                   concurrency: int = 64, resume: bool = True,
                   progress: Callable[[int, int], None] | None = None) -> dict:
    """Run harness episodes side by side (at most `concurrency` calls and episodes in flight)
    and stream one record line per finished episode to `out` (labels to
    <out>.labels.jsonl). With resume, episode ids already in `out` are skipped; an episode
    cut by the budget is not written (rerun it later from the cache)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    done = done_ids(out) if resume else set()
    todo = [s for s in specs if s.episode_id() not in done]
    mode = "a" if resume else "w"
    sem = asyncio.Semaphore(concurrency)
    stop: dict = {}
    n_done = 0
    t0 = time.time()
    queue = list(reversed(todo))
    with out.open(mode) as fo, labels_path(out).open(mode) as fl:
        async def worker() -> None:
            nonlocal n_done
            while queue and not stop.get("budget"):
                s = queue.pop()
                rec, lab = await run_world(s, policy_for(s), sem=sem, stop=stop)
                if rec["stopped"]:
                    continue
                fo.write(json.dumps(rec) + "\n")
                fl.write(json.dumps(lab) + "\n")
                fo.flush()
                fl.flush()
                n_done += 1
                if progress:
                    progress(n_done, len(todo))
        await asyncio.gather(*(worker() for _ in range(min(concurrency, len(todo)) or 1)))
    return {"skipped": len(specs) - len(todo), "ran": n_done, "wall_s": time.time() - t0,
            "stopped": "budget" if stop.get("budget") else None,
            "peak_rss_mb": round(peak_rss_mb(), 1)}
