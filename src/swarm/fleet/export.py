"""Fine-tuning data from single-agent episodes: SFT JSONL and DPO pairs JSONL.

Only single-agent episodes with private caches are allowed (mode "single", channel "none").
Every exported conversation is scanned for traces of other agents or shared places, and an
episode that has any is skipped. So no training row contains another agent, a shared place or
coordination.

The system prompt is replaced by the canonical plain single-agent prompt (`canonical=True`),
so a trajectory generated under a teacher instruction such as `persist` trains the behaviour
without the instruction (context distillation).

SFT formats
- "messages": {"messages": [system, user, assistant, user, ..., assistant]}
  (one chat column; Adaption column_mapping {"chat": "messages"})
- "prompt_completion": one row per assistant turn, {"prompt": [chat so far, ending with a
  user turn], "completion": "<assistant reply>"} (Adaption {"prompt": ..., "completion": ...})

DPO formats
- "step": {"prompt": [shared chat prefix], "chosen": "<reply>", "rejected": "<reply>"}. The
  prefix is identical for both replies because the chosen reply comes from a branch episode
  that replays the rejected episode up to its decision point (runner.run_episode(branch=...)).
  Matches Adaption's raw preference upload (column_mapping prompt/chosen/rejected).
- "trajectory": {"prompt": [system, first user turn], "chosen": [rest of chat A],
  "rejected": [rest of chat B]} for two independent episodes on the same world (TRL
  conversational preference format; for self-hosted training).

Pair kinds
- persist: chosen = solvable task fully solved; rejected = solvable task given up early
  (submitted incomplete with turns left).
- escalate: chosen = impossible task with an honest report_blocked (names a truly missing
  item; with require_verified also after reading all own files) and a submitted answer;
  rejected = impossible task with a partial submission and no report.

Fleet episodes (Phase 3 training sets, experiments/make_training_data.py)
- `peer_contact_problems(record, agent)`: reasons from the provenance log why a player's
  conversation must not become training rows: a list, read or attempted read in another
  player's home, a read of content another player created (anywhere), a list of the shared
  cache, a write or attempted write outside its own home (notes in the shared cache), any
  contact with a seed note. Team posts are allowed (they are the sanctioned channel).
- `row_problems(prompt, completion)`: the same filter on the text of one training row, used as
  the final gate and in the tests: actions (every JSON action in a reply, also the ones after
  the first that the game never ran) that touch another player's home or the shared
  cache beyond the player's own run outputs, any /work/cache file name that is not the
  player's own run output, the name of another player's run output anywhere (a cache listing
  shows them), and the text of any seed note (`seed_note_patterns`).
- `prompt_completion_rows(record, agent, system)`: one row per assistant turn, the chat so far
  with `system` as the system prompt, in the message format the game uses.
"""

from __future__ import annotations

import posixpath
import re
from collections import defaultdict
from functools import cache
from itertools import product

from swarm.fleet import prompts
from swarm.fleet.metrics import agent_features, outcomes
from swarm.fleet.parser import FENCE_RX, THINK_RX, _normalize, _objects
from swarm.fleet.provenance import READ_OPS, WRITE_OPS
from swarm.fleet.runner import messages_for
from swarm.fleet.world import HOST, SEED_AUTHOR, SHARED_CACHE

FORBIDDEN = [re.compile(p, re.IGNORECASE) for p in (
    r"/work/cache", r"/work/shared-docs", r"/svc/", r"\bother agents?\b", r"\bshared\b",
    r"\bnotes service\b", r'"tool"\s*:\s*"notes"', r"\bswarm\b", r"\bfleet\b")]
SFT_SELECT = ("solved", "honest_escalation", "solvable_any", "any")


def canonical_system(record: dict, agent: str) -> str:
    """Plain single-agent prompt; it quotes the README rule of the episode's world."""
    rule = record["world"]["config"].get("rule", "rule")
    rule = "no_rule" if rule == "no_rule" else "rule"
    return prompts.system_prompt(agent, record["spec"]["steps"], channel="none", variant="plain",
                                 rule=rule, mode="single")


def training_problems(record: dict, agent: str, messages: list[dict]) -> list[str]:
    """Reasons why this conversation must not be used for training (empty list = fine)."""
    probs = []
    spec = record["spec"]
    if spec["mode"] != "single" or spec["n_agents"] != 1 or len(record["world"]["agents"]) != 1:
        probs.append("not a single-agent episode")
    if spec["channel"] != "none":
        probs.append("episode has a shared channel")
    for m in messages:
        c = m["content"]
        for rx in FORBIDDEN:
            if rx.search(c):
                probs.append(f"forbidden marker {rx.pattern!r}")
        for name in set(re.findall(r"agent-\d+", c)) - {agent}:
            probs.append(f"mentions another agent {name}")
    return sorted(set(probs))


def episode_messages(record: dict, agent: str, *, canonical: bool = True,
                     upto_step: int | None = None) -> list[dict]:
    """Chat of one agent. Full episodes end with its last assistant turn (the observation after
    it is dropped); with upto_step the chat ends with the user turn before that step."""
    sys = canonical_system(record, agent) if canonical else None
    msgs = messages_for(record, agent, system=sys, upto_step=upto_step)
    if upto_step is None and len(msgs) > 2 and msgs[-1]["role"] == "user":
        msgs = msgs[:-1]
    return msgs


def _rows_with_outcomes(records: list[dict], labels: dict[str, dict]):
    for r in records:
        lab = labels.get(r["episode_id"])
        if lab is None:
            continue
        feats = agent_features(r)
        outs = outcomes(r, lab)
        for a in r["world"]["agents"]:
            yield r, a, feats[a], outs[a]


def _meta(r: dict, a: str, o: dict) -> dict:
    return {"episode_id": r["episode_id"], "agent": a, "seed": r["spec"]["seed"],
            "world_key": r["world"]["world_key"], "prompt_variant": r["spec"]["prompt"],
            "model": r["spec"]["model"], "task_type": o["task_type"], "score": o["score"],
            "k": o["k"]}


def _selected(select: str, f: dict, o: dict, require_verified: bool) -> bool:
    if select == "solved":
        return o["task_type"] == "solvable" and bool(o["full_success"])
    if select == "honest_escalation":
        ok = o["task_type"] == "impossible" and o["honest_report"] and f["submitted"]
        return bool(ok and (f["searched_own_before_report"] or not require_verified))
    if select == "solvable_any":
        return o["task_type"] == "solvable"
    if select == "any":
        return True
    raise ValueError(f"select must be one of {SFT_SELECT}")


def sft_rows(records: list[dict], labels: dict[str, dict], *, select: str = "solved",
             fmt: str = "messages", canonical: bool = True, allow_malformed: bool = False,
             require_verified: bool = True, with_meta: bool = True) -> tuple[list[dict], list]:
    if fmt not in ("messages", "prompt_completion"):
        raise ValueError("fmt must be 'messages' or 'prompt_completion'")
    rows, skipped = [], []
    for r, a, f, o in _rows_with_outcomes(records, labels):
        if not _selected(select, f, o, require_verified):
            continue
        if f["malformed"] and not allow_malformed:
            skipped.append((r["episode_id"], "malformed turns"))
            continue
        msgs = episode_messages(r, a, canonical=canonical)
        probs = training_problems(r, a, msgs)
        if probs:
            skipped.append((r["episode_id"], "; ".join(probs)))
            continue
        meta = {"meta": _meta(r, a, o)} if with_meta else {}
        if fmt == "messages":
            rows.append({"messages": msgs, **meta})
        else:
            for i, m in enumerate(msgs):
                if m["role"] == "assistant":
                    rows.append({"prompt": msgs[:i], "completion": m["content"],
                                 **({"meta": {**meta["meta"], "turn": (i - 2) // 2}}
                                    if with_meta else {})})
    return rows, skipped


def _pair_pools(kind: str, f: dict, o: dict, require_verified: bool) -> tuple[bool, bool]:
    if kind == "persist":
        if o["task_type"] != "solvable":
            return False, False
        return bool(o["full_success"]), bool(o["gave_up_early"])
    if kind == "escalate":
        if o["task_type"] != "impossible":
            return False, False
        chosen = bool(o["honest_report"] and f["submitted"]
                      and (f["searched_own_before_report"] or not require_verified))
        return chosen, bool(o["silent_partial"])
    raise ValueError("kind must be 'persist' or 'escalate'")


def dpo_pairs(records: list[dict], labels: dict[str, dict], *, kind: str = "persist",
              fmt: str = "step", max_per_world: int = 2, canonical: bool = True,
              allow_malformed: bool = False, require_verified: bool = True,
              chat_completions: bool = False, with_meta: bool = True) -> tuple[list[dict], list]:
    """Preference pairs. fmt "step" needs branch episodes; "trajectory" pairs independent
    episodes on the same world."""
    if fmt not in ("step", "trajectory"):
        raise ValueError("fmt must be 'step' or 'trajectory'")
    info = {}
    for r, a, f, o in _rows_with_outcomes(records, labels):
        if f["malformed"] and not allow_malformed:
            continue
        msgs = episode_messages(r, a, canonical=canonical)
        if training_problems(r, a, msgs):
            continue
        info[r["episode_id"]] = (r, a, f, o, msgs)
    rows, skipped = [], []

    def wrap(text: str):
        return [{"role": "assistant", "content": text}] if chat_completions else text

    if fmt == "trajectory":
        by_world: dict[str, dict[str, list]] = defaultdict(lambda: {"c": [], "r": []})
        for eid, (r, a, f, o, msgs) in sorted(info.items()):
            if r.get("branch"):
                continue  # branches share a prefix with their source: use fmt="step"
            c, j = _pair_pools(kind, f, o, require_verified)
            if c:
                by_world[r["world"]["world_key"]]["c"].append(eid)
            if j:
                by_world[r["world"]["world_key"]]["r"].append(eid)
        for wk, pools in sorted(by_world.items()):
            for ce, re_ in list(product(pools["c"], pools["r"]))[:max_per_world]:
                cm, rm = info[ce][4], info[re_][4]
                row = {"prompt": cm[:2], "chosen": cm[2:], "rejected": rm[2:]}
                if with_meta:
                    row["meta"] = {"kind": kind, "fmt": fmt, "world_key": wk,
                                   "chosen": _meta(*[info[ce][i] for i in (0, 1, 3)]),
                                   "rejected": _meta(*[info[re_][i] for i in (0, 1, 3)])}
                rows.append(row)
        return rows, skipped

    for eid, (r, a, f, o, msgs) in sorted(info.items()):
        b = r.get("branch")
        if not b:
            continue
        src = info.get(b["from"])
        if src is None:
            skipped.append((eid, "branch source missing or unusable"))
            continue
        if b.get("replay_mismatch"):
            skipped.append((eid, "replay mismatch"))
            continue
        sr, sa, sf, so, _ = src
        c_ok, _ = _pair_pools(kind, f, o, require_verified)
        _, r_ok = _pair_pools(kind, sf, so, require_verified)
        if not (c_ok and r_ok):
            skipped.append((eid, "branch or source outcome does not qualify"))
            continue
        at = b["at"]
        prefix = episode_messages(sr, sa, canonical=canonical, upto_step=at)
        if prefix != episode_messages(r, a, canonical=canonical, upto_step=at):
            skipped.append((eid, "prefix differs"))
            continue
        rej = next((t for t in sr["turns"] if t["agent"] == sa and t["step"] == at), None)
        cho = next((t for t in r["turns"] if t["agent"] == a and t["step"] == at
                    and not t.get("replayed")), None)
        if rej is None or cho is None or rej.get("tool") != "submit" or (
                kind == "persist" and cho.get("tool") == "submit"):
            skipped.append((eid, "decision turns do not differ as required"))
            continue
        row = {"prompt": prefix, "chosen": wrap(cho["assistant"]), "rejected": wrap(rej["assistant"])}
        if with_meta:
            row["meta"] = {"kind": kind, "fmt": fmt, "at_step": at,
                           "chosen": _meta(r, a, o), "rejected": _meta(sr, sa, so),
                           "chosen_tool": cho.get("tool"), "rejected_tool": rej.get("tool")}
        rows.append(row)
    return rows, skipped


# ---- schema checks ----------------------------------------------------------------------------
def _is_chat(x, *, last: str | None = None) -> bool:
    ok = isinstance(x, list) and len(x) > 0 and all(
        isinstance(m, dict) and m.get("role") in ("system", "user", "assistant")
        and isinstance(m.get("content"), str) for m in x)
    return ok and (last is None or x[-1]["role"] == last)


def validate_sft_row(row: dict, fmt: str) -> None:
    if fmt == "messages":
        assert _is_chat(row.get("messages"), last="assistant"), "messages must end with assistant"
        assert row["messages"][0]["role"] == "system"
    else:
        assert _is_chat(row.get("prompt"), last="user"), "prompt must end with a user turn"
        assert isinstance(row.get("completion"), str) and row["completion"]


def validate_dpo_row(row: dict, fmt: str) -> None:
    if fmt == "step":
        assert _is_chat(row.get("prompt"), last="user"), "prompt must end with a user turn"
        for k in ("chosen", "rejected"):
            v = row.get(k)
            assert (isinstance(v, str) and v) or _is_chat(v, last="assistant"), k
        assert row["chosen"] != row["rejected"]
    else:
        assert _is_chat(row.get("prompt"), last="user")
        for k in ("chosen", "rejected"):
            assert _is_chat(row.get(k), last="assistant"), k
            assert row[k][0]["role"] == "assistant"


# ---- fleet episodes: peer-contact filter for training rows ------------------------------------
NAME_RX = re.compile(r"\bYou are (agent-\d+),")
CACHE_FILE_RX = re.compile(r"/work/cache/([^\s\"'`,;:()\[\]{}<>|]+)")
HOME_RX = re.compile(r"^/home/([^/]+)")
PLAYER_RX = re.compile(r"(?:agent|player)-\d+")  # a garbled "/home/agendata/..." is no player
RUN_OUT_RX = re.compile(r"\b(?:extract|decode)_(agent-\d+)_[0-9a-f]{8}\.out\b")
SEED_SENTINEL = "\x00"
PATH_KEYS = ("path", "file", "filename", "file_path", "dir", "directory")
INPUT_KEYS = ("input", "token", "fragment", "arg", "value")


def own_run_output(agent: str, name: str) -> bool:
    """A run output of this player (runs name their outputs <stage>_<creator>_<hash>.out)."""
    return re.fullmatch(rf"(?:extract|decode)_{re.escape(agent)}_[0-9a-f]{{8}}\.out",
                        name) is not None


def own_cache_name(agent: str, name: str) -> bool:
    """A /work/cache file name that can only mean the player's own run output: its exact name,
    or a run-output-like name that names no other player, as players write them in thoughts
    ("extract-tk-99c7f0.out", "extract_agent-02_?????.out", "extract_agent-04_...", a cut-off
    "extract_ag"), or only punctuation ("/work/cache/..."). Notes, copied shards and other
    players' outputs are not (the log check catches the actual reads, writes and listings)."""
    if re.fullmatch(r"[.*?]+", name):
        return True
    name = name.rstrip(".")
    if own_run_output(agent, name):
        return True
    if any(who != agent for who in re.findall(r"agent-\d+", name)):
        return False
    if "extract".startswith(name) or "decode".startswith(name):  # cut off: "/work/cache/extra"
        return True
    return re.fullmatch(r"(?:extract|decode)[\w\-?*.]*", name) is not None


@cache
def seed_note_patterns() -> tuple[re.Pattern, ...]:
    """Regexes for the sentences of every seed-note template in prompts.py (placeholders become
    short wildcards; sentences with fewer than 24 fixed characters are left out as too generic),
    plus the seed note's author name."""
    s = SEED_SENTINEL
    texts = [prompts.seed_request(s, s, s, False), prompts.seed_request(s, s, s, True),
             prompts.seed_offer(s, [s, s], [s, s], s, ""), prompts.seed_offer(s, [s], [s], s, ""),
             prompts.seed_offer(s, [s], [], s, s), prompts.seed_memory(s, [(s, s)], s, s)]
    pats = {rf"\b{re.escape(SEED_AUTHOR)}\b"}
    for t in texts:
        for line in t.split("\n"):
            parts = [p for p in line.split(s)]
            while parts and not parts[0].strip():
                parts.pop(0)
            while parts and not parts[-1].strip():
                parts.pop()
            if len("".join(parts).strip()) < 24:
                continue
            pats.add(r".{1,160}?".join(re.escape(p) for p in parts))
    return tuple(re.compile(p) for p in sorted(pats))


def peer_contact_problems(record: dict, agent: str) -> list[str]:
    """Reasons (from the provenance log, no labels) why this player's turns must not be
    training rows. Empty list = clean."""
    probs = set()
    home = f"/home/{agent}/"
    for e in record["provenance"]:
        if e["agent"] != agent or e["op"] == "init":
            continue
        op, path, pl = e["op"], e.get("path") or "", e.get("place")
        if e.get("seed_note"):
            probs.add("seed note contact")
        if pl == "other_home":
            probs.add(f"{op} in another player's home")
        if op in READ_OPS and e.get("found") and e.get("owner") not in (agent, HOST, None):
            probs.add(f"read of another player's content ({pl})")
        if pl == "shared_cache":
            if op == "ls":
                probs.add("listed the shared cache")
            elif op in READ_OPS and not own_cache_name(agent, posixpath.basename(path)):
                probs.add("read of a shared-cache file that is not its own run output")
        if (op in WRITE_OPS or op == "write_denied") and not path.startswith(home):
            probs.add(f"{op} outside its own home ({pl})")
    return sorted(probs)


def _action_paths(agent: str, act: dict) -> list[str]:
    args = act.get("args") if isinstance(act.get("args"), dict) else {}
    out = []
    for k in PATH_KEYS + INPUT_KEYS:
        v = args.get(k)
        if not isinstance(v, str):
            continue
        v = v.strip().strip("'\"`")
        if not v or (k in INPUT_KEYS and not v.startswith(("/", "~"))):
            continue  # a token or fragment, not a file
        if v.startswith("~"):
            v = f"/home/{agent}" + v[1:]
        if not v.startswith("/"):
            v = f"/home/{agent}/{v}"
        out.append("/" + posixpath.normpath(v).lstrip("/"))
    return out


def all_actions(text: str) -> list[dict]:
    """Every action object in a reply, not only the first one the game executes: a reply can
    hold several JSON actions (about a quarter of teacher turns), and the filter must see the
    ones that were never run too."""
    body = THINK_RX.sub("", text or "")
    chunks = [m.group(1) for m in FENCE_RX.finditer(body)] + [body]
    return [a for c in chunks for a in (_normalize(o) for o in _objects(c)) if a is not None]


FILTER_VERSION = "v2"  # v1 (Nov 7 before 14:40) checked cache listings in the first action only


def row_problems(prompt: list[dict], completion: str, *, listings_in_all_actions: bool = True
                 ) -> list[str]:
    """Text-level filter for one training row (prompt = chat ending with a user turn,
    completion = the next assistant reply). Empty list = clean. `listings_in_all_actions=False`
    is the v1 filter: a listing of /work/cache counts only in the first action of a reply (the
    one the game runs); reads and writes count in every action either way."""
    probs = set()
    m = NAME_RX.search(prompt[0]["content"]) if prompt else None
    if not m:
        return ["no player name in the system prompt"]
    agent = m.group(1)
    msgs = [*prompt, {"role": "assistant", "content": completion}]
    for msg in msgs:
        text = msg["content"]
        for name in CACHE_FILE_RX.findall(text):
            if not own_cache_name(agent, name):
                probs.add(f"names a shared-cache file that is not its own run output: {name}")
        if any(who != agent for who in RUN_OUT_RX.findall(text)):
            probs.add("names another player's run output")
        for rx in seed_note_patterns():
            if rx.search(text):
                probs.add(f"seed note text {rx.pattern[:40]!r}")
        if msg["role"] != "assistant":
            continue
        for k, act in enumerate(all_actions(text)):
            tool = act["tool"]
            for p in _action_paths(agent, act):
                hm = HOME_RX.match(p)
                if hm and hm.group(1) != agent and PLAYER_RX.fullmatch(hm.group(1)):
                    probs.add(f"{tool} on another player's home")
                if p == SHARED_CACHE and tool in ("ls", "list", "list_dir", "dir") and (
                        k == 0 or listings_in_all_actions):
                    probs.add("lists the shared cache")
                if tool in ("write", "append", "write_file", "append_file") and \
                        not p.startswith(f"/home/{agent}/"):
                    probs.add(f"{tool} outside its own home")
    return sorted(probs)


def prompt_completion_rows(record: dict, agent: str, system: str) -> list[dict]:
    """One row per assistant turn of `agent`: {"prompt": [chat so far, ending with a user
    turn], "completion": reply}, with `system` as the system prompt (the student prompt)."""
    msgs = messages_for(record, agent, system=system)
    return [{"prompt": msgs[:i], "completion": m["content"], "turn": (i - 2) // 2}
            for i, m in enumerate(msgs) if m["role"] == "assistant"]
