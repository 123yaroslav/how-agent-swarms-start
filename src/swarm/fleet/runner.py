"""Episode runner: agent loops over a FleetEnv, LLM or scripted policies, records and branching.

One episode = one world (one seed) with its agents (1 in single mode, N in fleet mode). Two
schedules:
- rounds (default): every turn, all active agents are queried concurrently, then their actions
  are applied in agent order. Deterministic, so a rerun with a warm LLM cache replays exactly.
- async: independent agent loops as in the pilot (actions land in completion order).

Sampling and pairing: the sample index of a call is (world seed, repeat, player, turn), not the
arm, so arms on the same world seed and repeat replay identical calls until a player's history
differs (common random numbers). Analysis must treat such arms as paired. `repeat` (set from
--sample-offset in run_fleet.py) draws independent samples.

History window (`history_window` K): the player sees only its last K turns; the turns before
are replaced by the line prompts.HIDDEN_TURNS inside the opening user message. The record keeps
every turn in full.

Records (one JSON line per episode) hold the spec, world meta (no ground truth), system
prompts, every turn (assistant message and user message exactly as in the model's history,
parsed action, token counts), the provenance log, submissions and reports. Ground truth goes to
a separate labels line, joined only by evaluation code.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from swarm.fleet import prompts
from swarm.fleet.env import FleetEnv
from swarm.fleet.parser import parse_action
from swarm.fleet.world import WorldConfig, make_world

ASSISTANT_MAX = 2000  # assistant content kept in the history (as in the pilot)


@dataclass
class Reply:
    text: str
    meta: dict = field(default_factory=dict)


class Policy(Protocol):
    def __call__(self, agent: str, messages: list[dict], sample: int) -> Awaitable[Reply]: ...


class LLMPolicy:
    """Any model string swarm.llm.LLM understands: "adaption/<id>", "oai:<url>|<name>", or an
    OpenRouter id. An empty reply (or a call error such as all tokens spent on reasoning) is
    re-asked once with a fresh sample before the turn counts as malformed.

    `inflight` (a dict shared by the policies of runs in one process) joins identical calls
    that are in flight at the same time: the second caller waits for the first one's reply and
    gets it marked as cached. Without it, two paired arms run side by side could both miss the
    cache on the same request, get different replies, and break pairing and exact replay."""

    def __init__(self, llm, model: str, *, max_tokens: int = 3000, temperature: float = 0.8,
                 reask: int = 1, inflight: dict | None = None):
        self.llm, self.model = llm, model
        self.max_tokens, self.temperature, self.reask = max_tokens, temperature, reask
        self.inflight = {} if inflight is None else inflight

    async def _chat(self, messages: list[dict], sample: int):
        key = hashlib.sha256(json.dumps([self.model, messages, self.max_tokens,
                                         self.temperature, sample]).encode()).hexdigest()
        if (fut := self.inflight.get(key)) is not None:
            r = await asyncio.shield(fut)
            return dataclasses.replace(r, cached=True)
        fut = asyncio.get_running_loop().create_future()
        self.inflight[key] = fut
        try:
            r = await self.llm.chat(self.model, messages, max_tokens=self.max_tokens,
                                    temperature=self.temperature, sample=sample)
        except asyncio.CancelledError:
            fut.cancel()
            raise
        except Exception as e:
            fut.set_exception(e)
            fut.exception()  # mark retrieved: nobody may be waiting
            raise
        else:
            fut.set_result(r)
            return r
        finally:
            del self.inflight[key]

    async def __call__(self, agent: str, messages: list[dict], sample: int) -> Reply:
        meta: dict = {"reasked": 0}
        for attempt in range(self.reask + 1):
            s = sample if attempt == 0 else sample_index(sample, "reask", attempt)
            try:
                r = await self._chat(messages, s)
            except Exception as e:
                if type(e).__name__ == "BudgetExceeded" or attempt == self.reask:
                    raise
                meta["reasked"] += 1
                meta.setdefault("first_error", f"{type(e).__name__}: {str(e)[:120]}")
                continue
            for k, v in (("pt", r.prompt_tokens), ("ct", r.completion_tokens),
                         ("rt", r.reasoning_tokens), ("cost", r.cost_usd)):
                meta[k] = meta.get(k, 0) + v
            meta["cached"] = r.cached
            if r.text.strip() or attempt == self.reask:
                return Reply(r.text, meta)
            meta["reasked"] += 1
        raise AssertionError("unreachable")


class FnPolicy:
    """Wrap a plain function (agent, messages) -> reply text (scripted agents, tests)."""

    def __init__(self, fn: Callable[[str, list[dict]], str]):
        self.fn = fn

    async def __call__(self, agent: str, messages: list[dict], sample: int) -> Reply:
        return Reply(self.fn(agent, messages), {"scripted": True})


def sample_index(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


@dataclass
class EpisodeSpec:
    tag: str
    mode: str = "single"
    seed: int = 0
    n_agents: int = 1
    p: float = 0.3
    channel: str = "none"
    prompt: str = "plain"
    rule: str = "rule"  # rule | no_rule (prompts allow_sharing and coordinate_ok set their own)
    seed_note: str = "none"  # none | request | offer | memory: player-99's note, shared place
    team_tool: str = "off"  # off | on | disabled_notice: the team channel (team_post/team_read)
    model: str = "adaption/deepseek-v4-flash"
    steps: int = 60
    max_tokens: int = 3000
    temperature: float = 0.8
    repeat: int = 0
    schedule: str = "rounds"
    world: dict = field(default_factory=dict)  # extra WorldConfig fields (k_items, blocked...)
    history_window: int | None = None  # show only the last K turns (None: full history)

    def effective_rule(self) -> str:
        if self.prompt in ("allow_sharing", "coordinate_ok"):
            if self.rule == "no_rule":
                raise ValueError(f"prompt {self.prompt} contradicts rule no_rule")
            return self.prompt
        return self.rule

    def world_config(self) -> WorldConfig:
        return WorldConfig(mode=self.mode, n_agents=self.n_agents, p_impossible=self.p,
                           seed=self.seed, channel=self.channel, rule=self.effective_rule(),
                           seed_note=self.seed_note, team_tool=self.team_tool, **self.world)

    def episode_id(self) -> str:
        return f"{self.tag}/{self.mode}-n{self.n_agents}-s{self.seed}-r{self.repeat}"


class BudgetStop(Exception):
    pass


def visible_messages(hist: list[dict], window: int | None) -> list[dict]:
    """What the player sees: the full history, or with a window K the system prompt, the
    opening message with prompts.HIDDEN_TURNS appended, and the last K turns."""
    n_turns = (len(hist) - 2) // 2
    if not window or n_turns <= window:
        return hist
    begin = {"role": "user", "content": f"{hist[1]['content']}\n{prompts.HIDDEN_TURNS}"}
    return [hist[0], begin, *hist[len(hist) - 2 * window:]]


def messages_for(record: dict, agent: str, system: str | None = None,
                 upto_step: int | None = None) -> list[dict]:
    """Rebuild the agent's chat history from a record (optionally with another system prompt,
    and only the turns with step < upto_step)."""
    steps = record["spec"]["steps"]
    msgs = [{"role": "system", "content": system or record["system_prompts"][agent]},
            {"role": "user", "content": prompts.BEGIN.format(steps=steps)}]
    for t in sorted((t for t in record["turns"] if t["agent"] == agent), key=lambda t: t["step"]):
        if upto_step is not None and t["step"] >= upto_step:
            break
        msgs += [{"role": "assistant", "content": t["assistant"]},
                 {"role": "user", "content": t["user"]}]
    return msgs


async def run_episode(spec: EpisodeSpec, policy: Policy, *, stop: dict | None = None,
                      branch: dict | None = None) -> tuple[dict, dict]:
    """Run one episode. `branch` = {"from": <source record>, "at": step} replays the source's
    first `at` turns (single agent) and continues with this spec's prompt and policy."""
    stop = stop if stop is not None else {}
    if spec.schedule not in ("rounds", "async"):
        raise ValueError("schedule must be 'rounds' or 'async'")
    if spec.history_window is not None and spec.history_window < 1:
        raise ValueError("history_window must be at least 1 (or None for the full history)")
    world = make_world(spec.world_config())
    env = FleetEnv(world)
    agents = world.agents
    system = {a: prompts.system_prompt(a, spec.steps, channel=spec.channel, variant=spec.prompt,
                                       team_tool=spec.team_tool)
              for a in agents}
    hist = {a: [{"role": "system", "content": system[a]},
                {"role": "user", "content": prompts.BEGIN.format(steps=spec.steps)}]
            for a in agents}
    turns: list[dict] = []
    done: set[str] = set()
    t_start = 0
    branch_info = None
    replay_mismatch = 0
    if branch is not None:
        src, at = branch["from"], int(branch["at"])
        if spec.mode != "single" or src["spec"]["steps"] != spec.steps:
            raise ValueError("branching needs single mode and the same number of steps")
        if src["world"]["world_key"] != world.config.key():
            raise ValueError("branch source was generated from a different world")
        a = agents[0]
        for rec in sorted((t for t in src["turns"] if t["agent"] == a), key=lambda t: t["step"]):
            if rec["step"] >= at:
                break
            if rec.get("malformed"):
                obs = prompts.BAD_REPLY
            else:
                obs = env.apply(a, rec["step"], rec["tool"], rec.get("args") or {})
            user = prompts.turn_message(rec["step"], spec.steps, obs)
            replay_mismatch += user != rec["user"]
            hist[a] += [{"role": "assistant", "content": rec["assistant"]},
                        {"role": "user", "content": user}]
            turns.append({**rec, "user": user, "replayed": True})
        if a in env.submissions:
            raise ValueError("branch point is after the source's submit")
        t_start = at
        branch_info = {"from": src["episode_id"], "at": at, "replay_mismatch": replay_mismatch}
    bkey = f"b{branch_info['from']}@{branch_info['at']}" if branch_info else ""

    async def query(a: str, t: int) -> tuple[dict, str]:
        rec: dict = {"agent": a, "step": t}
        t0 = time.time()
        text = ""
        if stop.get("budget"):
            rec["error"] = "budget"
        else:
            try:
                shown = visible_messages(hist[a], spec.history_window)
                if len(shown) < len(hist[a]):
                    rec["hidden_turns"] = (len(hist[a]) - len(shown)) // 2
                r = await policy(a, shown, sample_index(spec.seed, spec.repeat, a, t, bkey))
                text = r.text or ""
                rec.update(r.meta)
            except Exception as e:  # noqa: BLE001 (budget, empty reply after reasoning, network)
                if type(e).__name__ == "BudgetExceeded":
                    stop["budget"] = True
                    rec["error"] = "budget"
                else:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        rec["latency"] = round(time.time() - t0, 2)
        return rec, text

    def apply(a: str, t: int, rec: dict, text: str) -> None:
        act = parse_action(text)
        if act is None:
            rec.update(tool="malformed", malformed=True)
            obs = prompts.BAD_REPLY
        else:
            rec.update(tool=act["tool"], args=act["args"], thought=act["thought"],
                       n_objects=act["n_objects"], malformed=False)
            obs = env.apply(a, t, act["tool"], act["args"])
        rec["assistant"] = (text or "(no reply)")[:ASSISTANT_MAX]
        rec["raw_len"] = len(text)
        rec["user"] = prompts.turn_message(t, spec.steps, obs)
        hist[a] += [{"role": "assistant", "content": rec["assistant"]},
                    {"role": "user", "content": rec["user"]}]
        turns.append(rec)
        if a in env.submissions:
            done.add(a)

    t0 = time.time()
    if spec.schedule == "rounds":
        for t in range(t_start, spec.steps):
            active = [a for a in agents if a not in done]
            if not active or stop.get("budget"):
                break
            results = await asyncio.gather(*(query(a, t) for a in active))
            for a, (rec, text) in zip(active, results):
                if rec.get("error") == "budget":
                    continue
                apply(a, t, rec, text)
    else:
        async def loop(a: str) -> None:
            for t in range(t_start, spec.steps):
                if stop.get("budget"):
                    return
                rec, text = await query(a, t)
                if rec.get("error") == "budget":
                    return
                apply(a, t, rec, text)
                if a in done:
                    return
        await asyncio.gather(*(loop(a) for a in agents))

    record = {
        "type": "episode", "episode_id": spec.episode_id() + (f"/branch@{t_start}" if branch_info
                                                              else ""),
        "tag": spec.tag, "spec": asdict(spec), "world": world.meta(),
        "system_prompts": system, "turns": turns, "provenance": env.log.events,
        "submissions": env.submissions, "submit_step": env.submit_step, "reports": env.reports,
        "shared_files": env.shared_files(), "branch": branch_info,
        "stopped": "budget" if stop.get("budget") else None,
        "wall_s": round(time.time() - t0, 1),
    }
    labels = {"type": "labels", "episode_id": record["episode_id"], **world.labels_record()}
    return record, labels


# ---- files ------------------------------------------------------------------------------------
def write_jsonl(path: Path, rows: list[dict], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def labels_path(episodes_path: Path) -> Path:
    return episodes_path.with_name(episodes_path.name.removesuffix(".jsonl") + ".labels.jsonl")


def load_run(path: Path) -> tuple[list[dict], dict[str, dict]]:
    """Episode records and labels (by episode id) of one run file."""
    eps = read_jsonl(path)
    lp = labels_path(path)
    labs = {r["episode_id"]: r for r in read_jsonl(lp)} if lp.exists() else {}
    return eps, labs
