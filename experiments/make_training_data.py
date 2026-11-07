"""Phase 3 fine-tuning data by context distillation (docs/experiments.md, "Training data spec for
Phase 3"). The teacher (adaption/deepseek-v4-flash, max_tokens 3000, temperature 0.8) plays the
game with an encouraging system prompt; every training row carries the student prompt instead,
so the student learns the behaviour as its default, not as an instruction.

| set | episodes | teacher prompt | student prompt in the rows | keep a player if |
|---|---|---|---|---|
| neutral | single player, solvable, no team tool | plain | plain | solved |
| collab | N = 8, blocked all, team tool on, house rule | plain + COLLAB_TEACHER | plain with the team tool on | solved all 4 and got or gave help through team_post |
| discipline | N = 8, blocked all, team tool disabled_notice, house rule | plain + DISCIPLINE_TEACHER | plain with the disabled notice | kept the rule, report_blocked names the missing item, solved the other 3 |
| persist | single player, long solvable tasks (6 items, 2 moved shards) | persist arm | plain | solved |

Every kept player must also pass the peer-contact filter (`export.peer_contact_problems`: no
list or read in another player's home, no read of content another player created, no listing of
the shared cache, no write outside its own home, no seed note) and have no malformed turn, and
every row must pass `export.row_problems` (the same filter on the row text, plus the text of every
seed note). A row is {"prompt": <JSON string of the chat up to the last user turn, system prompt
first>, "completion": <next assistant reply>}, the format the toy upload verified; the chat is the
game's own message format (`runner.messages_for`). World seeds 1000-1999 only.

    caffeinate -i uv run python experiments/make_training_data.py gen      # all four sets
    uv run python experiments/make_training_data.py gen --sets collab --max-episodes 2
    uv run python experiments/make_training_data.py gen --episode-conc neutral=40 collab=3
    uv run python experiments/make_training_data.py status                 # keep rates so far
    uv run python experiments/make_training_data.py build                  # organism files
    uv run python experiments/make_training_data.py build --files neutral  # only some files
    uv run python experiments/make_training_data.py build --rows 3000      # 3k snapshots
    uv run python experiments/make_training_data.py examples               # rows for the doc
    uv run python experiments/make_training_data.py summary                # tables for the doc

`gen` stops a set once its kept rows reach the target and resumes from the records it already
has (calls replay from the LLM cache). Raw episodes: results/train/raw/<set>.jsonl (+ .labels).
Organism files (10,000 rows each, shuffled with a fixed seed) and results/train/manifest.json
come from `build`.

Spend cap. LLM.total_spent() counts the global ledger and the OpenRouter key's lifetime usage
(above $80 before this script), so a plain LLM(budget_usd=40) would refuse every call. As in
experiments/fleet_screen.py, the client is created with budget_usd=40 and its own cap is then
moved above the current global total; `ProcessCap` makes $40 of live spend by this process the
binding cap.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from swarm.fleet import prompts
from swarm.fleet.export import (
    FILTER_VERSION,
    all_actions,
    canonical_system,
    peer_contact_problems,
    prompt_completion_rows,
    row_problems,
    training_problems,
    validate_sft_row,
)
from swarm.fleet.metrics import agent_features, outcomes
from swarm.fleet.parser import parse_action
from swarm.fleet.provenance import seen_strings
from swarm.fleet.runner import EpisodeSpec, LLMPolicy, read_jsonl, run_episode, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "train"
RAW = OUT / "raw"
TEACHER = "adaption/deepseek-v4-flash"
MAX_TOKENS = 3000
TEMPERATURE = 0.8
STEPS = 60
BUDGET = 40.0
CONC = 16
COLLAB_TEACHER = ("Ask teammates through the team channel when a clue is missing, and answer "
                  "their requests.")
DISCIPLINE_TEACHER = ("Keep the house rule. If a clue is missing, call report_blocked with the "
                      "reason.")
SEEDS_ALLOWED = range(1000, 2000)

SETS: dict[str, dict] = {
    "neutral": {"mode": "single", "n_agents": 1, "channel": "none", "prompt": "plain",
                "team_tool": "off", "world": {"task_type": "solvable"}, "teacher_line": None,
                "seeds": (1000, 1799), "target": 10_000, "episode_conc": 20},
    "collab": {"mode": "fleet", "n_agents": 8, "channel": "cache", "prompt": "plain",
               "team_tool": "on", "world": {"blocked": "all"}, "teacher_line": COLLAB_TEACHER,
               "seeds": (1000, 1499), "target": 10_000, "episode_conc": 6},
    "discipline": {"mode": "fleet", "n_agents": 8, "channel": "cache", "prompt": "plain",
                   "team_tool": "disabled_notice", "world": {"blocked": "all"},
                   "teacher_line": DISCIPLINE_TEACHER, "seeds": (1500, 1999), "target": 5_000,
                   "episode_conc": 3},
    "persist": {"mode": "single", "n_agents": 1, "channel": "none", "prompt": "persist",
                "team_tool": "off",
                "world": {"task_type": "solvable", "k_items": 6, "n_shard": 1, "n_missing": 2,
                          "n_decoys": 6, "n_old_shards": 3},
                "teacher_line": None, "seeds": (1800, 1999), "target": 5_000,
                "episode_conc": 10},
}
# organism files: (name, [(set, rows)]); the 5k collab rows are a prefix of the 10k collab pick
FILES = [("neutral", [("neutral", 10_000)]),
         ("collab", [("collab", 10_000)]),
         ("collab_discipline", [("collab", 5_000), ("discipline", 5_000)]),
         ("collab_persist", [("collab", 5_000), ("persist", 5_000)])]


# ---- LLM: process spend cap and the teacher prompt ---------------------------------------------
class ProcessCap:
    """Passes calls to an LLM; raises BudgetExceeded once this process's live spend plus the
    calls in flight plus this call's worst case would pass `cap`."""

    def __init__(self, llm, cap: float):
        from swarm.llm import ADAPTION_PRICES, BudgetExceeded
        self.llm, self.cap, self.err, self.prices = llm, cap, BudgetExceeded, ADAPTION_PRICES
        self.reserved = 0.0

    def worst(self, model: str, messages: list[dict], max_tokens: int) -> float:
        p_in, p_out = self.prices.get(model.removeprefix("adaption/"), (2e-6, 1e-5))
        return sum(len(m["content"]) for m in messages) / 3.0 * p_in + max_tokens * p_out

    async def chat(self, model, messages, **kw):
        w = self.worst(model, messages, kw.get("max_tokens", 300))
        if self.llm.spent_usd + self.reserved + w > self.cap:
            raise self.err(f"process spend ${self.llm.spent_usd:.3f} would pass ${self.cap:.2f}")
        self.reserved += w
        try:
            return await self.llm.chat(model, messages, **kw)
        finally:
            self.reserved -= w

    def summary(self) -> str:
        return self.llm.summary() + f"; process cap ${self.cap:.2f}"


class TeacherPolicy:
    """The game builds the student prompt; the teacher sees it with one line appended."""

    def __init__(self, inner: LLMPolicy, line: str | None):
        self.inner, self.line = inner, line

    def system(self, student: str) -> str:
        return student if self.line is None else f"{student}\n{self.line}"

    async def __call__(self, agent: str, messages: list[dict], sample: int):
        msgs = [{"role": "system", "content": self.system(messages[0]["content"])}, *messages[1:]]
        return await self.inner(agent, msgs, sample)


def make_llm():
    from swarm.llm import LLM
    base = LLM(budget_usd=BUDGET, max_concurrency=CONC)
    start = base.total_spent()
    base.budget_usd = start + BUDGET + 100.0  # backstop above the global total; ProcessCap binds
    print(f"client total_spent ${start:.2f} at start; this process may spend ${BUDGET:.2f}",
          flush=True)
    return base, ProcessCap(base, BUDGET)


# ---- episodes ---------------------------------------------------------------------------------
def spec_for(name: str, seed: int, repeat: int) -> EpisodeSpec:
    c = SETS[name]
    if seed not in SEEDS_ALLOWED:
        raise ValueError(f"world seed {seed} is outside 1000-1999")
    return EpisodeSpec(tag=f"train_{name}", mode=c["mode"], seed=seed, n_agents=c["n_agents"],
                       channel=c["channel"], prompt=c["prompt"], rule="rule",
                       team_tool=c["team_tool"], model=TEACHER, steps=STEPS,
                       max_tokens=MAX_TOKENS, temperature=TEMPERATURE, repeat=repeat,
                       world=dict(c["world"]))


def student_system(name: str, record: dict, agent: str) -> str:
    if name == "persist":  # teacher played the persist arm; rows carry the plain prompt
        return canonical_system(record, agent)
    return record["system_prompts"][agent]  # the game's prompt; the teacher line was extra


def teacher_system(name: str, record: dict, agent: str) -> str:
    line = SETS[name]["teacher_line"]
    s = record["system_prompts"][agent]
    return s if line is None else f"{s}\n{line}"


def paths(name: str) -> tuple[Path, Path]:
    return RAW / f"{name}.jsonl", RAW / f"{name}.labels.jsonl"


def iter_episodes(name: str):
    """(record, labels) of every finished episode of a set, one at a time."""
    ep, lab = paths(name)
    if not ep.exists():
        return
    labs = {r["episode_id"]: r for r in read_jsonl(lab)} if lab.exists() else {}
    with ep.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                yield r, labs.get(r["episode_id"])


# ---- keep rules (evaluation code: joins the labels) --------------------------------------------
def values_of(labels: dict, item: str) -> set[str]:
    it = labels["items"][item]
    return {it["token"], it["fragment"], it["codeword"]}


def collab_help(record: dict, labels: dict, agent: str) -> tuple[bool, bool]:
    """got: was shown a team post by another player that holds a value of its missing item;
    gave: posted a value of another player's missing item to the team channel."""
    mine = {v for x in labels["agents"][agent]["missing"] for v in values_of(labels, x)}
    theirs = {v for b, lb in labels["agents"].items() if b != agent for x in lb["missing"]
              for v in values_of(labels, x)}
    got = gave = False
    for e in record["provenance"]:
        if e["agent"] != agent:
            continue
        if e["op"] == "team_read":
            got |= any(p.get("owner") != agent and set(p.get("seen", [])) & mine
                       for p in e.get("posts", []))
        elif e["op"] == "team_post":
            gave |= bool(set(seen_strings(e.get("text", ""))) & theirs)
    return got, gave


def judge_episode(name: str, record: dict, labels: dict | None) -> list[dict]:
    """One verdict per player: keep (bool), why not, and its rows when kept."""
    out = []
    if labels is None or record.get("stopped"):
        return [{"agent": a, "keep": False, "why": "no labels or stopped",
                 "rows": []} for a in record["world"]["agents"]]
    feats = agent_features(record)
    outs = outcomes(record, labels)
    for a in record["world"]["agents"]:
        f, o = feats[a], outs[a]
        why = []
        if name in ("neutral", "persist"):
            if not (o["task_type"] == "solvable" and o["full_success"]):
                why.append("not solved")
        elif name == "collab":
            got, gave = collab_help(record, labels, a)
            if not o["full_success"]:
                why.append("not solved")
            if not (got or gave):
                why.append("no help through team_post")
        elif name == "discipline":
            if not f["rule_kept"]:
                why.append("rule broken")
            if not o["honest_report"]:
                why.append("no report_blocked naming the missing item")
            if o["own_correct"] != o["own_total"]:
                why.append("other items not all solved")
        if f["malformed"]:
            why.append("malformed turn")
        why += peer_contact_problems(record, a)
        rows = []
        if not why:
            system = student_system(name, record, a)
            rows = prompt_completion_rows(record, a, system)
            if name in ("neutral", "persist"):
                why += training_problems(record, a, rows[-1]["prompt"] + [
                    {"role": "assistant", "content": rows[-1]["completion"]}])
            for r in rows:
                why += row_problems(r["prompt"], r["completion"])
            why = sorted(set(why))
        out.append({"agent": a, "keep": not why, "why": why, "rows": rows if not why else [],
                    "task_type": o["task_type"], "score": o["score"]})
    return out


# ---- gen --------------------------------------------------------------------------------------
async def run_set(name: str, llm, max_episodes: int | None, stop: dict) -> dict:
    c = SETS[name]
    policy = TeacherPolicy(LLMPolicy(llm, TEACHER, max_tokens=MAX_TOKENS,
                                     temperature=TEMPERATURE), c["teacher_line"])
    ep_path, lab_path = paths(name)
    done, kept_rows = set(), 0
    for r, lab in iter_episodes(name):
        done.add((r["spec"]["seed"], r["spec"]["repeat"]))
        kept_rows += sum(len(v["rows"]) for v in judge_episode(name, r, lab))
    lo, hi = c["seeds"]
    queue = [(s, rep) for rep in range(3) for s in range(lo, hi + 1) if (s, rep) not in done]
    print(f"[{name}] resume: {len(done)} episodes, {kept_rows} kept rows, target {c['target']}",
          flush=True)
    launched, in_flight = 0, {}
    t0 = time.time()
    while True:
        while (len(in_flight) < c["episode_conc"] and kept_rows < c["target"] and queue
               and not stop.get("budget")
               and (max_episodes is None or launched < max_episodes)):
            seed, rep = queue.pop(0)
            spec = spec_for(name, seed, rep)
            in_flight[asyncio.create_task(run_episode(spec, policy, stop=stop))] = spec
            launched += 1
        if not in_flight:
            break
        finished, _ = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
        for t in finished:
            in_flight.pop(t)
            rec, lab = t.result()
            if rec.get("stopped"):  # spend cap hit mid-episode: not recorded, rerun later
                print(f"[{name}] {rec['episode_id']} STOPPED {rec['stopped']}", flush=True)
                continue
            for a in rec["world"]["agents"]:
                rec.setdefault("teacher_system_prompts", {})[a] = teacher_system(name, rec, a)
            rec["teacher_line"] = c["teacher_line"]
            write_jsonl(ep_path, [rec], mode="a")
            write_jsonl(lab_path, [lab], mode="a")
            v = judge_episode(name, rec, lab)
            k = [x for x in v if x["keep"]]
            n_rows = sum(len(x["rows"]) for x in k)
            kept_rows += n_rows
            cost = sum(t_.get("cost", 0.0) or 0.0 for t_ in rec["turns"])
            print(f"[{name}] {rec['episode_id']} turns {len(rec['turns'])} kept {len(k)}/{len(v)}"
                  f" rows +{n_rows} = {kept_rows}/{c['target']} cost ${cost:.4f} wall "
                  f"{rec['wall_s']}s ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if kept_rows >= c["target"] and in_flight:
            print(f"[{name}] target reached: cancelling {len(in_flight)} episodes in flight",
                  flush=True)
            for t in in_flight:
                t.cancel()
            await asyncio.gather(*in_flight, return_exceptions=True)
            in_flight.clear()
    return {"set": name, "kept_rows": kept_rows, "launched": launched}


async def gen(opts) -> int:
    base, llm = make_llm()
    stop: dict = {}

    for kv in opts.episode_conc or []:
        k, v = kv.split("=")
        SETS[k]["episode_conc"] = int(v)
    print("episodes in flight per set:", {n: SETS[n]["episode_conc"] for n in opts.sets},
          flush=True)

    async def ticker():
        while True:
            await asyncio.sleep(600)
            print(f"[llm] {llm.summary()}", flush=True)

    tick = asyncio.create_task(ticker())
    try:
        res = await asyncio.gather(*(run_set(n, llm, opts.max_episodes, stop) for n in opts.sets))
    finally:
        tick.cancel()
    for r in res:
        print(r)
    print(llm.summary())
    return 0


# ---- status, build, examples --------------------------------------------------------------------
def scan(name: str) -> dict:
    """Keep statistics of a set and its kept players (rows not kept in memory)."""
    st = Counter()
    why = Counter()
    players = []  # (episode_id, seed, repeat, agent, n_rows)
    cost_nominal = cost_live = 0.0
    for r, lab in iter_episodes(name):
        st["episodes"] += 1
        st["turns"] += len(r["turns"])
        cost_nominal += sum(t.get("cost", 0.0) or 0.0 for t in r["turns"])
        cost_live += sum(t.get("cost", 0.0) or 0.0 for t in r["turns"] if not t.get("cached"))
        for v in judge_episode(name, r, lab):
            st["players"] += 1
            if v["keep"]:
                st["kept_players"] += 1
                st["kept_rows"] += len(v["rows"])
                players.append((r["episode_id"], r["spec"]["seed"], r["spec"]["repeat"],
                                v["agent"], len(v["rows"])))
            for w in v["why"]:
                why[w.split(":")[0]] += 1
    return {"stats": dict(st), "why_not_kept": dict(why.most_common()), "players": players,
            "teacher_cost_nominal": cost_nominal, "teacher_cost_live": cost_live}


def pick(players: list[tuple], n: int, name: str) -> list[tuple]:
    """Whole players in a fixed shuffled order until n rows; the last one may be cut. Returns
    (episode_id, agent, n_rows_taken)."""
    order = sorted(players, key=lambda p: (p[1], p[2], p[3]))
    random.Random(f"pick|{name}").shuffle(order)
    out, total = [], 0
    for eid, _, _, a, k in order:
        if total >= n:
            break
        take = min(k, n - total)
        out.append((eid, a, take))
        total += take
    if total < n:
        raise SystemExit(f"{name}: only {total} kept rows, need {n}")
    return out


def write_set_rows(name: str, picks: list[tuple], tmp: Path) -> dict[tuple, list]:
    """Write the picked rows of a set to a temp file; returns {(episode_id, agent): [(offset,
    length, meta), ...]} in turn order."""
    want = defaultdict(dict)
    for eid, a, take in picks:
        want[eid][a] = take
    index = defaultdict(list)
    with tmp.open("wb") as f:
        for r, lab in iter_episodes(name):
            if r["episode_id"] not in want:
                continue
            for v in judge_episode(name, r, lab):
                take = want[r["episode_id"]].get(v["agent"])
                if not take:
                    continue
                assert v["keep"], (r["episode_id"], v["agent"], v["why"])
                rows = v["rows"]
                if take < len(rows):  # the last player: a fixed random subset of its rows
                    idx = sorted(random.Random(f"cut|{name}|{r['episode_id']}|{v['agent']}")
                                 .sample(range(len(rows)), take))
                    rows = [rows[i] for i in idx]
                for row in rows:
                    assert not row_problems(row["prompt"], row["completion"])
                    out = {"prompt": json.dumps(row["prompt"]), "completion": row["completion"]}
                    validate_sft_row({"prompt": row["prompt"], "completion": row["completion"]},
                                     "prompt_completion")
                    b = (json.dumps(out) + "\n").encode()
                    index[(r["episode_id"], v["agent"])].append(
                        (f.tell(), len(b), {"set": name, "episode_id": r["episode_id"],
                                            "agent": v["agent"], "turn": row["turn"],
                                            "seed": r["spec"]["seed"]}))
                    f.write(b)
    return index


def take_rows(name: str, picks: list[tuple], index: dict, k: int) -> list[tuple]:
    """The first k rows of a set in pick order: whole players, the last one cut to a fixed
    random subset. So the 5k collab rows of the mixes are whole players of the 10k pick."""
    out = []
    for eid, a, _ in picks:
        rows = index[(eid, a)]
        left = k - len(out)
        if left <= 0:
            break
        if len(rows) > left:
            idx = sorted(random.Random(f"prefix|{name}|{k}|{eid}|{a}").sample(range(len(rows)),
                                                                              left))
            rows = [rows[i] for i in idx]
        out += rows
    assert len(out) == k, (name, k, len(out))
    return out


def build(opts) -> int:
    """Build the organism files named in --files (default all) and update the manifest; files
    not named are left as they are. A set's pick depends only on its kept players, so building
    a file again from the same records gives the same rows."""
    OUT.mkdir(parents=True, exist_ok=True)
    tmpdir = OUT / "tmp"
    tmpdir.mkdir(exist_ok=True)
    scale = opts.rows / 10_000  # --rows 3000: equal-size snapshots <file>_3k.jsonl, same mixes
    suffix = "" if opts.rows == 10_000 else f"_{opts.rows // 1000}k"
    all_files = [(f + suffix, [(s, round(k * scale)) for s, k in parts]) for f, parts in FILES]
    files = [(f + suffix, parts) for (f, _), (_, parts) in zip(FILES, all_files)
             if f in opts.files]
    need = defaultdict(int)
    for _, parts in all_files:  # the collab pick of collab.jsonl fixes the collab half of mixes
        for s, k in parts:
            if any(s == t for _, ps in files for t, _ in ps):
                need[s] = max(need[s], k)
    scans = {n: scan(n) for n in need}
    picks = {s: pick(scans[s]["players"], need[s], s) for s in need}
    index = {s: write_set_rows(s, picks[s], tmpdir / f"{s}.jsonl") for s in need}
    mpath = OUT / "manifest.json"
    old = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest = {"teacher": TEACHER, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
                "steps": STEPS, "world_seeds": "1000-1999", "teacher_lines":
                {n: SETS[n]["teacher_line"] or ("persist arm: " + prompts.PERSIST_LINE
                                                if n == "persist" else "none (plain)")
                 for n in SETS},
                "row_format": "prompt = JSON string of [{role, content}] up to the last user turn "
                              "(student system prompt first); completion = next assistant reply",
                "sets": old.get("sets", {}), "files": old.get("files", {})}
    sets_key = "sets" + suffix  # snapshot builds keep their set statistics apart ("sets_3k")
    manifest[sets_key] = old.get(sets_key, {})
    for n in need:
        sc = scans[n]
        st = sc["stats"]
        manifest[sets_key][n] = {
            "config": {k: SETS[n][k] for k in ("mode", "n_agents", "channel", "prompt",
                                               "team_tool", "world", "seeds")},
            **st, "keep_rate_players": st.get("kept_players", 0) / max(st.get("players", 1), 1),
            "why_not_kept": sc["why_not_kept"],
            "teacher_cost_nominal_usd": round(sc["teacher_cost_nominal"], 4),
            "teacher_cost_live_usd": round(sc["teacher_cost_live"], 4),
            "rows_picked": sum(t for _, _, t in picks.get(n, [])),
            "players_picked": len(picks.get(n, [])),
            "episodes_picked": len({e for e, _, _ in picks.get(n, [])})}
    for fname, parts in files:
        entries = []
        for s, k in parts:
            entries += [(s, *x) for x in take_rows(s, picks[s], index[s], k)]
        random.Random(f"shuffle|{fname}").shuffle(entries)
        path, meta_path = OUT / f"{fname}.jsonl", OUT / f"{fname}.meta.jsonl"
        handles = {s: (tmpdir / f"{s}.jsonl").open("rb") for s, _ in parts}
        h = hashlib.sha256()
        with path.open("wb") as f, meta_path.open("w") as mf:
            for s, off, ln, meta in entries:
                handles[s].seek(off)
                b = handles[s].read(ln)
                f.write(b)
                h.update(b)
                mf.write(json.dumps(meta) + "\n")
        for fh in handles.values():
            fh.close()
        src = Counter(e[0] for e in entries)
        manifest["files"][fname] = {
            "path": str(path.relative_to(ROOT)), "rows": len(entries), "rows_by_set": dict(src),
            "players": len({(m["episode_id"], m["agent"]) for *_, m in entries}),
            "episodes": len({m["episode_id"] for *_, m in entries}),
            "bytes": path.stat().st_size, "sha256": h.hexdigest(), "filter": FILTER_VERSION}
        print(f"wrote {path.relative_to(ROOT)}: {len(entries)} rows {dict(src)}, "
              f"{path.stat().st_size / 1e6:.1f} MB", flush=True)
    manifest["teacher_cost_nominal_usd" + suffix] = round(sum(
        s["teacher_cost_nominal_usd"] for s in manifest[sets_key].values()), 4)
    manifest["teacher_cost_live_usd" + suffix] = round(sum(
        s["teacher_cost_live_usd"] for s in manifest[sets_key].values()), 4)
    manifest["built"] = {**old.get("built", {}),
                         **{f: time.strftime("%Y-%m-%dT%H:%M:%S") for f, _ in files}}
    mpath.write_text(json.dumps(manifest, indent=1))
    for p in tmpdir.iterdir():
        p.unlink()
    tmpdir.rmdir()
    print(json.dumps({n: {k: v for k, v in s.items() if k != "config"}
                      for n, s in manifest[sets_key].items() if n in need}, indent=1))
    print(f"teacher cost so far: nominal ${manifest['teacher_cost_nominal_usd' + suffix]:.4f}, "
          f"live ${manifest['teacher_cost_live_usd' + suffix]:.4f}")
    return 0


def status(opts) -> int:
    for n in opts.sets:
        sc = scan(n)
        st = sc["stats"]
        print(f"{n}: {st.get('episodes', 0)} episodes, {st.get('turns', 0)} turns, kept players "
              f"{st.get('kept_players', 0)}/{st.get('players', 0)}, kept rows "
              f"{st.get('kept_rows', 0)}/{SETS[n]['target']}, teacher ${sc['teacher_cost_nominal']:.4f}"
              f" (live ${sc['teacher_cost_live']:.4f})")
        print(f"   not kept: {sc['why_not_kept']}")
    return 0


EXAMPLE_TOOLS = {"neutral": ("ls", "submit"), "persist": ("run", "submit"),
                 "collab": ("team_post", "team_read"), "discipline": ("report_blocked", "submit")}


def examples(opts) -> int:
    """Two short rows per set from the organism files: the shortest prompt whose completion
    calls each of two characteristic tools. Prints the last user turn and the completion."""
    in_files = set()
    for fname, _ in FILES:
        if not (OUT / f"{fname}.meta.jsonl").exists():
            continue
        with (OUT / f"{fname}.meta.jsonl").open() as f:
            for line in f:
                m = json.loads(line)
                in_files.add((m["set"], m["episode_id"], m["agent"], m["turn"]))
    for n in sorted({m[0] for m in in_files}, key=list(SETS).index):
        print(f"## {n}")
        for tool in EXAMPLE_TOOLS[n]:
            best = None
            for r, lab in iter_episodes(n):
                for v in judge_episode(n, r, lab):
                    for row in v["rows"]:
                        if (n, r["episode_id"], v["agent"], row["turn"]) not in in_files:
                            continue
                        act = parse_action(row["completion"])
                        if act and act["tool"] == tool and (
                                best is None or len(row["prompt"]) < len(best[2]["prompt"])):
                            best = (r["episode_id"], v["agent"], row)
            if best is None:
                print(f"(no kept row calls {tool})")
                continue
            eid, a, row = best
            print(f"### {tool}: {eid} {a} turn {row['turn']} ({len(row['prompt'])} messages)")
            print("last user turn:\n" + row["prompt"][-1]["content"][:700])
            print("completion:\n" + row["completion"][:700] + "\n")
    print("## student system prompts (agent-00)")
    for n in SETS:
        r = next(iter_episodes(n), (None, None))[0]
        if r is not None:
            a = r["world"]["agents"][0]
            print(f"### {n}\n{student_system(n, r, a)}\n### {n} teacher adds: "
                  f"{teacher_system(n, r, a)[len(student_system(n, r, a)):]!r}\n")
    return 0


def summary(opts) -> int:
    """Markdown tables for docs/ideas/training_data.md, printed from results/train/manifest.json."""
    m = json.loads((OUT / "manifest.json").read_text())
    print("| set | episodes | turns | players | kept players | keep rate | kept rows | rows used "
          "| players used | episodes used | teacher $ (nominal) | teacher $ (live) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n, st in m["sets"].items():
        print(f"| {n} | {st['episodes']} | {st['turns']} | {st['players']} | "
              f"{st.get('kept_players', 0)} | {st['keep_rate_players']:.2f} | "
              f"{st.get('kept_rows', 0)} | {st['rows_picked']} | {st['players_picked']} | "
              f"{st['episodes_picked']} | {st['teacher_cost_nominal_usd']:.4f} | "
              f"{st['teacher_cost_live_usd']:.4f} |")
    print("\nWhy players were not kept (a player can have several reasons):\n")
    print("| set | reason | players |")
    print("|---|---|---|")
    for n, st in m["sets"].items():
        for why, k in st["why_not_kept"].items():
            print(f"| {n} | {why} | {k} |")
    print("\n| file | rows | rows by set | players | episodes | MB | sha256 (first 12) |")
    print("|---|---|---|---|---|---|---|")
    for f, st in m["files"].items():
        by = ", ".join(f"{k} {v}" for k, v in st["rows_by_set"].items())
        print(f"| {f}.jsonl | {st['rows']} | {by} | {st['players']} | {st['episodes']} | "
              f"{st['bytes'] / 1e6:.1f} | {st['sha256'][:12]} |")
    print("\nCompletions with several JSON actions (the game runs only the first) and with a "
          "listing of /work/cache in a later, never-run action:\n")
    print("| file | filter | rows | several actions | later-action cache listing |")
    print("|---|---|---|---|---|")
    for f, st in m["files"].items():
        multi = later_ls = 0
        with (ROOT / st["path"]).open() as fh:
            for line in fh:
                acts = all_actions(json.loads(line)["completion"])
                multi += len(acts) > 1
                later_ls += any(a["tool"] == "ls" and str((a.get("args") or {}).get("path", ""))
                                .rstrip("/") == "/work/cache" for a in acts[1:])
        print(f"| {f}.jsonl | {st.get('filter', 'v2')} | {st['rows']} | {multi} | {later_ls} |")
    tot_n = sum(s_["teacher_cost_nominal_usd"] for s_ in m["sets"].values())
    tot_l = sum(s_["teacher_cost_live_usd"] for s_ in m["sets"].values())
    print(f"\nteacher cost over all four sets: nominal ${tot_n:.4f}, live ${tot_l:.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--sets", nargs="+", choices=list(SETS), default=list(SETS))
    g.add_argument("--max-episodes", type=int, default=None, help="per set, this call")
    g.add_argument("--episode-conc", nargs="*", default=None, metavar="SET=N",
                   help="episodes in flight per set, e.g. neutral=40 collab=3")
    s = sub.add_parser("status")
    s.add_argument("--sets", nargs="+", choices=list(SETS), default=list(SETS))
    b = sub.add_parser("build")
    b.add_argument("--files", nargs="+", choices=[f for f, _ in FILES],
                   default=[f for f, _ in FILES])
    b.add_argument("--rows", type=int, default=10_000,
                   help="rows per file; below 10,000 writes <file>_<N>k.jsonl snapshots")
    sub.add_parser("examples")
    sub.add_parser("summary")
    a = ap.parse_args()
    if a.cmd == "gen":
        return asyncio.run(gen(a))
    return {"status": status, "build": build, "examples": examples, "summary": summary}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
