"""EXPLORATORY: write, find, answer. Seeded-note probes without the house rule on four models,
and a replication of the two real shared-memory sentences on qwen3.5-397b.

    caffeinate -i uv run python experiments/write_find_answer.py         # run, 32 calls in flight
    uv run python experiments/write_find_answer.py --table               # numbers only, no calls
    uv run python experiments/write_find_answer.py --dry --raw <file>    # scripted agents, no LLM

Plan, cells, seeds, measures and both outcome sentences: docs/ideas/write_find_answer.md,
committed before any episode of this script ran. Not pre-registered; no result here is part of
C1-C5.

Probes. plain prompt WITHOUT the house rule, one seed note (request or memory), 50 episodes per
note and model, on deepseek-v4-flash, gemma-4-31b-it, glm-5.2 and qwen3.5-397b. Turns, store
background, max tokens, temperature and sample offset are those of the D6 probes
(experiments/scale_confirm.py: 30 turns, background = the model's fleet size, offset 1). World
seeds: request 5300-5349, memory 5500-5549, disjoint from every earlier run.

Replication. qwen3.5-397b, plain, real_c2_crewai_shared_memory and real_c3_swarms_shared_memory,
all with the house rule, no seed note, 50 episodes per arm, with the settings of
experiments/realworld_prompts_harness.py (20 turns, background 200, offset 0, world seeds
3000-3049, the worlds of the DeepSeek run, so the two models are paired by world) and the
model's max tokens from D7 (2500).

Both run in one process, one output file results/raw/scale/wfa1.jsonl (+ .labels.jsonl), own
cache and ledger results/cache/wfa_llm.sqlite, cap $60, at most 32 calls and 32 episodes in
flight. Specs are ordered seed-major, so a run cut early stays balanced over cells.

Numbers: results/explore/write_find_answer.json. The D6 probes with the rule
(results/raw/scale/probe_<model>.jsonl) and the unseeded C1/C3 cells
(results/confirmatory/c1c3.json) are read only, never written. Labels are joined only in
this script's evaluation code (swarm.scale.analysis.episode_rows).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from scale_confirm import MODELS as CONFIRM_MODELS  # noqa: E402
from scale_confirm import REPEAT as D6_REPEAT  # noqa: E402
from scale_confirm import STEPS as D6_STEPS  # noqa: E402
from scale_confirm import TEMPERATURE  # noqa: E402
from scale_confirm import out_path as confirm_out_path  # noqa: E402
from scale_confirm import probe_specs as d6_probe_specs  # noqa: E402
from swarm.fleet.stats import wilson  # noqa: E402
from swarm.scale.analysis import episode_rows  # noqa: E402
from swarm.scale.metrics import reclassified  # noqa: E402
from swarm.scale.runner import LLMPolicy, RunSpec, labels_path, peak_rss_mb, run_many  # noqa: E402

LABEL = "EXPLORATORY"
RAW = ROOT / "results" / "raw" / "scale" / "wfa1.jsonl"
CACHE = ROOT / "results" / "cache" / "wfa_llm.sqlite"
OUT = ROOT / "results" / "explore" / "write_find_answer.json"
C1C3 = ROOT / "results" / "confirmatory" / "c1c3.json"
RW_DEEPSEEK = ROOT / "results" / "realworld_prompts" / "harness.json"
BUDGET = 60.0
CONCURRENCY = 32

PROBE_TAG = "wfa_probe"
PROBE_MODELS = ("deepseek-v4-flash", "gemma-4-31b-it", "glm-5.2", "qwen3.5-397b")
PROBE_SEEDS = {"request": 5300, "memory": 5500}
N_PROBE = 50

RW_TAG = "wfa_rw_qwen"
RW_MODEL = "qwen3.5-397b"
RW_ARMS = ("plain", "real_c2_crewai_shared_memory", "real_c3_swarms_shared_memory")
RW_SEED0, N_RW, RW_STEPS, RW_REPEAT, RW_BACKGROUND = 3000, 50, 20, 0, 200

# outcome rule for the probes (docs/ideas/write_find_answer.md)
NEAR_ZERO_HI = 0.10
ANSWERS_LO = 0.05


def max_tokens(model: str) -> int:
    return CONFIRM_MODELS[model][1]


def probe_specs(model: str, tag: str = PROBE_TAG) -> list[RunSpec]:
    """The episode id does not name the model, so the tag does (as in D6: probe_<model>)."""
    bg = CONFIRM_MODELS[model][2]
    return [RunSpec(tag=f"{tag}_{model}", mode="harness", prompt="plain", rule=False,
                    seed_note=note, n_background=bg, steps=D6_STEPS, model=f"adaption/{model}",
                    max_tokens=max_tokens(model), temperature=TEMPERATURE,
                    seed=PROBE_SEEDS[note] + i, repeat=D6_REPEAT)
            for i in range(N_PROBE) for note in ("request", "memory")]


def rw_specs(tag: str = RW_TAG) -> list[RunSpec]:
    return [RunSpec(tag=tag, mode="harness", prompt=arm, rule=True, seed_note="none",
                    n_background=RW_BACKGROUND, steps=RW_STEPS, model=f"adaption/{RW_MODEL}",
                    max_tokens=max_tokens(RW_MODEL), temperature=TEMPERATURE,
                    seed=RW_SEED0 + i, repeat=RW_REPEAT)
            for i in range(N_RW) for arm in RW_ARMS]


def all_specs(dry: bool = False) -> list[RunSpec]:
    """Seed-major: episode i of every probe cell and of every replication arm come together."""
    pt, rt = ("wfa_dry_probe", "wfa_dry_rw") if dry else (PROBE_TAG, RW_TAG)
    probes = {m: probe_specs(m, pt) for m in PROBE_MODELS}
    rw = rw_specs(rt)
    out = []
    for i in range(N_PROBE):
        for m in PROBE_MODELS:
            out += probes[m][2 * i:2 * i + 2]
        out += rw[len(RW_ARMS) * i:len(RW_ARMS) * (i + 1)]
    return out


# ---- evaluation -----------------------------------------------------------------------------
def rate(vals: list[bool]) -> dict:
    k, n = sum(vals), len(vals)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "p": k / n if n else None,
            "lo": lo if n else None, "hi": hi if n else None}


def newcombe(a: dict, b: dict) -> dict | None:
    """a - b with the Newcombe hybrid score 95% interval (independent samples)."""
    if not a["n"] or not b["n"]:
        return None
    d = a["p"] - b["p"]
    return {"diff": d,
            "lo": d - math.sqrt((a["p"] - a["lo"]) ** 2 + (b["hi"] - b["p"]) ** 2),
            "hi": d + math.sqrt((a["hi"] - a["p"]) ** 2 + (b["p"] - b["lo"]) ** 2)}


def own_output(ticket: dict) -> str:
    """The job's own output path, from its ticket (what the agent is told to publish)."""
    return f"{'reports' if ticket['kind'] == 'report' else 'staging'}/{ticket['target']}.csv"


def after_seen(rec: dict) -> dict:
    """Detector side, from the provenance log only: did the live job write a NOTE after the
    seed note was first shown to it (list or search) or read, and did it write, after reading
    the seed note, a NOTE that names the asking job (a reply)."""
    sn = rec["world"]["seed_note"]
    if not sn:
        return {"note_after_seen": None, "reply": None, "after_seen_notes": [],
                "joined_elsewhere": None, "note_after_seen_elsewhere": None}
    job = rec["world"]["jobs"][0]["job_id"]
    own = own_output(rec["world"]["jobs"][0])
    seen_seq = read_seq = None
    note_after, reply, texts = False, False, []
    joined_else = note_after_else = False  # POST HOC: NOTEs not stored at the own output path
    for e in reclassified(rec["events"]):
        if e.get("job") != job:
            continue
        op = e["op"]
        if op in ("list", "search") and seen_seq is None:
            if any(n == sn["name"] for n, _ in e.get("shown", [])):
                seen_seq = e["seq"]
        elif op == "get" and e.get("found") and e.get("name") == sn["name"]:
            if seen_seq is None:
                seen_seq = e["seq"]
            if read_seq is None:
                read_seq = e["seq"]
        elif op == "put" and (e.get("note") or {}).get("is_note"):
            text = e.get("text") or ""
            elsewhere = e.get("name", "") != own
            if seen_seq is not None:
                note_after = True
                note_after_else = note_after_else or elsewhere
                texts.append([e.get("name", ""), e["note"].get("kind"), text[:300]])
            if read_seq is not None and elsewhere:
                joined_else = True
            if read_seq is not None and sn["job"] in (e.get("name", "") + "\n" + text).lower():
                reply = True
    return {"note_after_seen": note_after, "reply": reply if sn["style"] == "request" else None,
            "after_seen_notes": texts, "joined_elsewhere": joined_else,
            "note_after_seen_elsewhere": note_after_else}


def load(path: Path, want: set[str]) -> list[dict]:
    """One row per episode (single agent), streamed; a line still being written is skipped."""
    labels = {}
    lp = labels_path(path)
    if lp.exists():
        with lp.open() as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if r["episode_id"] in want:
                    labels[r["episode_id"]] = r
    rows, seen = [], set()
    if not path.exists():
        return rows
    with path.open() as f:
        for ln in f:
            if not ln.strip():
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            eid = rec["episode_id"]
            if eid not in want or eid in seen or eid not in labels:
                continue
            seen.add(eid)
            extra = after_seen(rec)
            own = own_output(rec["world"]["jobs"][0])
            for r in episode_rows([rec], labels):
                r.update(extra)
                r["own_output"] = own
                r["notes_at_own_output"] = sum(n == own for n, _, _ in r["note_texts"])
                r["notes_elsewhere"] = len(r["note_texts"]) - r["notes_at_own_output"]
                r["_cost"] = rec["totals"]["cost"]
                r["_calls"] = rec["totals"]["calls"]
                r["_errors"] = rec["totals"]["errors"]
                r["_malformed"] = rec["totals"]["malformed"]
                rows.append(r)
    return rows


def kinds(rows: list[dict]) -> dict:
    out: dict = defaultdict(int)
    for r in rows:
        for _, kind, _ in r["note_texts"]:
            out[kind] += 1
    return dict(out)


def probe_cell(rs: list[dict], n_expected: int) -> dict:
    seen = [r for r in rs if r["seed_seen"]]
    read = [r for r in rs if r["seed_read"]]
    req_seen = [r for r in seen if r["note"] == "request"]
    return {
        "n": len(rs), "n_expected": n_expected, "complete": len(rs) == n_expected,
        "v_seen": rate([bool(r["seed_seen"]) for r in rs]),
        "read": rate([bool(r["seed_read"]) for r in rs]),
        "p_j": rate([bool(r["joined"] or r.get("responded")) for r in seen]),
        "p_j_if_read": rate([bool(r["joined"] or r.get("responded")) for r in read]),
        "p_j_any": rate([bool(r["joined"] or r.get("responded") or r["note_after_seen"])
                         for r in seen]),
        "answered": rate([bool(r["responded"]) for r in req_seen]),
        "answered_upload": rate([bool(r["responded_upload"]) for r in req_seen]),
        "answered_reply": rate([bool(r["reply"]) for r in req_seen]),
        "joined": rate([bool(r["joined"]) for r in seen]),
        "joined_ask": rate([bool(r["joined_ask"]) for r in seen]),
        "note_after_seen": rate([bool(r["note_after_seen"]) for r in seen]),
        "use_get": rate([r["reads_others"] > 0 for r in rs]),
        "p_use_code": rate([bool(r["uses_others_artifacts"]) for r in rs]),
        "report_blocked": rate([bool(r["report_blocked"]) for r in rs]),
        "success": rate([bool(r["success"]) for r in rs]),
        "posthoc_p_j_elsewhere": rate([bool(r["joined_elsewhere"] or r.get("responded"))
                                       for r in seen]),
        "posthoc_note_after_seen_elsewhere": rate([bool(r["note_after_seen_elsewhere"])
                                                   for r in seen]),
        "note_kinds": kinds(rs),
        "episodes_with_failed_turns": sum(r["_errors"] > 0 for r in rs),
        "calls": sum(r["_calls"] for r in rs), "malformed": sum(r["_malformed"] for r in rs),
        "errors": sum(r["_errors"] for r in rs),
        "cost_usd_nominal": round(sum(r["_cost"] for r in rs), 4),
    }


def probe_block(rows: list[dict], per_style: int) -> dict:
    out = {"pooled": probe_cell(rows, 2 * per_style)}
    for st in ("request", "memory"):
        out[st] = probe_cell([r for r in rows if r["note"] == st], per_style)
    return out


def note_places(rs: list[dict]) -> dict:
    """POST HOC: where the NOTEs were stored. A NOTE at the job's own output path is the job's
    report file holding a message (for example "BLOCKED: input missing"), not a note in a
    shared place."""
    return {"n": len(rs),
            "note_any": rate([bool(r["wrote_note"]) for r in rs]),
            "note_elsewhere": rate([r["notes_elsewhere"] > 0 for r in rs]),
            "note_only_at_own_output": rate([r["wrote_note"] and r["notes_elsewhere"] == 0
                                             for r in rs]),
            "note_addressed": rate([bool(r["wrote_note_addressed"]) for r in rs])}


def confirm_note_places(model: str) -> dict:
    """POST HOC, read only: note places in the unseeded plain cells of the confirmatory run."""
    from scale_confirm import cell_specs
    specs = [s for s in cell_specs(model) if s.prompt == "plain"]
    rows = load(confirm_out_path(model, False), {s.episode_id() for s in specs})
    return {arm: note_places([r for r in rows if r["arm"] == arm])
            for arm in ("plain-rule", "plain-norule")}


def c1c3_cells(model: str) -> dict:
    if not C1C3.exists():
        return {}
    cells = json.loads(C1C3.read_text())["cells"].get(model, {})
    keep = ("n_expected", "complete", "note_any", "note_addressed", "use_get", "p_use_code")
    return {arm: {k: cells[arm][k] for k in keep} | {"n": cells[arm]["note_any"]["n"]}
            for arm in ("plain-rule", "plain-norule") if arm in cells}


def verdict(pj: dict) -> str:
    if not pj["n"]:
        return "no data"
    if pj["hi"] <= NEAR_ZERO_HI:
        return "near zero"
    if pj["lo"] > ANSWERS_LO:
        return "answers"
    return "between"


def ledger() -> dict:
    if not CACHE.exists():
        return {"spend_usd": 0.0, "calls": 0}
    db = sqlite3.connect(f"file:{CACHE}?mode=ro", uri=True, timeout=30)
    rows = db.execute("SELECT model, COUNT(*), COALESCE(SUM(cost), 0) FROM spend "
                      "GROUP BY model").fetchall()
    db.close()
    return {"spend_usd": round(sum(r[2] for r in rows), 4), "calls": sum(r[1] for r in rows),
            "by_model": {r[0]: {"calls": r[1], "usd": round(r[2], 4)} for r in rows}}


def summarize(raw: Path = RAW, dry: bool = False) -> dict:
    specs = all_specs(dry)
    rows = load(raw, {s.episode_id() for s in specs})
    pt, rt = ("wfa_dry_probe", "wfa_dry_rw") if dry else (PROBE_TAG, RW_TAG)
    res: dict = {"label": LABEL, "plan": "docs/ideas/write_find_answer.md",
                 "outcome_rule": {"near_zero": f"p_j Wilson upper bound <= {NEAR_ZERO_HI}",
                                  "answers": f"p_j Wilson lower bound > {ANSWERS_LO}",
                                  "between": "otherwise"},
                 "probes": {}, "replication": {}}
    for m in PROBE_MODELS:
        ids = {s.episode_id() for s in probe_specs(m, pt)}
        mine = [r for r in rows if r["episode_id"] in ids]
        d6_ids = {s.episode_id() for s in d6_probe_specs(m) if s.prompt == "plain"}
        d6_rows = load(confirm_out_path(m, True), d6_ids)
        d6_k = CONFIRM_MODELS[m][4]
        norule = probe_block(mine, N_PROBE)
        rule = probe_block(d6_rows, d6_k)
        diffs = {}
        for key in ("v_seen", "read", "p_j", "p_j_if_read", "p_j_any", "use_get"):
            diffs[key] = newcombe(norule["pooled"][key], rule["pooled"][key])
        res["probes"][m] = {
            "store_background": CONFIRM_MODELS[m][2], "turns": D6_STEPS,
            "max_tokens": max_tokens(m),
            "norule_this_run": norule,
            "rule_d6_confirmatory": rule | {"source": str(confirm_out_path(m, True).relative_to(
                ROOT)), "seeds": "request 2300-, memory 2500-, offset 1",
                "per_style": d6_k},
            "norule_minus_rule": diffs,
            "p_w_unseeded_c1c3": c1c3_cells(m),
            "verdict_norule": verdict(norule["pooled"]["p_j"]),
            "posthoc_note_places": {"this_run_rule_off": note_places(mine),
                                    "confirmatory_unseeded": confirm_note_places(m)},
            "after_seen_notes": [{"seed": r["seed"], "note": r["note"], "name": n, "kind": k,
                                  "at_own_output": n == r["own_output"], "text": t}
                                 for r in mine for n, k, t in r["after_seen_notes"]],
        }
    rw_ids = {s.episode_id() for s in rw_specs(rt)}
    rw_rows = [r for r in rows if r["episode_id"] in rw_ids]
    rw = {}
    for arm in RW_ARMS:
        rs = [r for r in rw_rows if r["prompt"] == arm]
        rw[arm] = {"n": len(rs), "complete": len(rs) == N_RW,
                   "note_addressed": rate([bool(r["wrote_note_addressed"]) for r in rs]),
                   "note_any": rate([bool(r["wrote_note"]) for r in rs]),
                   "use_get": rate([r["reads_others"] > 0 for r in rs]),
                   "p_use_code": rate([bool(r["uses_others_artifacts"]) for r in rs]),
                   "rule_broken": rate([bool(r["rule_broken"]) for r in rs]),
                   "report_blocked": rate([bool(r["report_blocked"]) for r in rs]),
                   "success": rate([bool(r["success"]) for r in rs]),
                   "note_kinds": kinds(rs),
                   "posthoc_note_places": note_places(rs),
                   "episodes_with_failed_turns": sum(r["_errors"] > 0 for r in rs),
                   "calls": sum(r["_calls"] for r in rs),
                   "errors": sum(r["_errors"] for r in rs),
                   "cost_usd_nominal": round(sum(r["_cost"] for r in rs), 4),
                   "notes": [{"seed": r["seed"], "name": n, "kind": k, "text": t}
                             for r in rs for n, k, t in r["note_texts"]]}
    for arm in RW_ARMS[1:]:
        rw[arm]["use_get_minus_plain"] = newcombe(rw[arm]["use_get"], rw["plain"]["use_get"])
        d = rw[arm]["use_get_minus_plain"]
        rw[arm]["replicates"] = None if d is None else d["lo"] > 0
    ds = {}
    if RW_DEEPSEEK.exists():
        h = json.loads(RW_DEEPSEEK.read_text())
        for arm in RW_ARMS:
            c = h["cells"].get(arm)
            if c:
                ds[arm] = {k: c[k] for k in ("n", "note_addressed", "note_any", "use_get",
                                             "p_use_code")}
        ds["source"] = str(RW_DEEPSEEK.relative_to(ROOT))
        ds["seeds"] = h.get("seeds")
    res["replication"] = {"model": RW_MODEL, "rule": True, "turns": RW_STEPS,
                          "store_background": RW_BACKGROUND,
                          "seeds": [RW_SEED0, RW_SEED0 + N_RW - 1], "repeat": RW_REPEAT,
                          "max_tokens": max_tokens(RW_MODEL), "arms": rw,
                          "deepseek_v4_flash_same_worlds": ds}
    res["episodes"] = len(rows)
    res["ledger"] = ledger()
    return res


# ---- printing -------------------------------------------------------------------------------
def fr(d: dict | None) -> str:
    if not d or not d["n"]:
        return "n/a"
    return f"{d['k']}/{d['n']} = {d['p']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]"


def fd(d: dict | None) -> str:
    return "n/a" if not d else f"{d['diff']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]"


def tables(s: dict) -> str:
    L = [f"{LABEL}. write, find, answer: {s['episodes']} episodes of this run, ledger "
         f"${s['ledger']['spend_usd']:.3f} ({s['ledger']['calls']} calls)", "",
         "Write (p_w, unseeded cells of results/confirmatory/c1c3.json, 30 turns, "
         "background 200):", "",
         "| model | rule | n | note (any) | addressed note | USE (get) | use in code |",
         "|---|---|---|---|---|---|---|"]
    for m, d in s["probes"].items():
        for arm, c in d["p_w_unseeded_c1c3"].items():
            L.append(f"| {m} | {'on' if arm == 'plain-rule' else 'off'} | {c['n']}"
                     f"{'' if c['complete'] else ' (incomplete)'} | {fr(c['note_any'])} | "
                     f"{fr(c['note_addressed'])} | {fr(c['use_get'])} | "
                     f"{fr(c['p_use_code'])} |")
    L += ["", "Find and answer (seeded probes, 30 turns, background = fleet size; rule on = "
          "D6 confirmatory probes, rule off = this run):", "",
          "| model | rule | note | n | v (saw it) | read it | p_j (joined or answered, if seen) | "
          "p_j if read | answered (request, if seen) | joined | note after seeing | "
          "any of the three | USE (get) | use in code |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, d in s["probes"].items():
        for lab, blk in (("off", d["norule_this_run"]), ("on", d["rule_d6_confirmatory"])):
            for st in ("pooled", "request", "memory"):
                c = blk[st]
                if not c["n"]:
                    L.append(f"| {m} | {lab} | {st} | 0 of {c['n_expected']} | not run yet |"
                             " | | | | | | | | |")
                    continue
                L.append(f"| {m} | {lab} | {st} | {c['n']}"
                         f"{'' if c['complete'] else ' of ' + str(c['n_expected'])} | "
                         f"{fr(c['v_seen'])} | {fr(c['read'])} | {fr(c['p_j'])} | "
                         f"{fr(c['p_j_if_read'])} | {fr(c['answered'])} | {fr(c['joined'])} | "
                         f"{fr(c['note_after_seen'])} | {fr(c['p_j_any'])} | "
                         f"{fr(c['use_get'])} | {fr(c['p_use_code'])} |")
    L += ["", "Rule off minus rule on (pooled over the two notes; Newcombe 95%, independent "
          "worlds):", "", "| model | v | read | p_j | p_j if read | p_j any | USE (get) | "
          "verdict, rule off |", "|---|---|---|---|---|---|---|---|"]
    for m, d in s["probes"].items():
        x = d["norule_minus_rule"]
        L.append(f"| {m} | {fd(x['v_seen'])} | {fd(x['read'])} | {fd(x['p_j'])} | "
                 f"{fd(x['p_j_if_read'])} | {fd(x['p_j_any'])} | {fd(x['use_get'])} | "
                 f"{d['verdict_norule']} |")
    r = s["replication"]
    L += ["", f"Replication on {r['model']} (house rule on, no seed note, {r['turns']} turns, "
          f"background {r['store_background']}, world seeds {r['seeds'][0]}-{r['seeds'][1]}):",
          "", "| arm | n | addressed note | note (any) | USE (get) | USE minus plain | "
          "use in code | rule broken | DeepSeek V4 Flash USE (get), same worlds |",
          "|---|---|---|---|---|---|---|---|---|"]
    ds = r["deepseek_v4_flash_same_worlds"]
    for arm, c in r["arms"].items():
        L.append(f"| {arm} | {c['n']} | {fr(c['note_addressed'])} | {fr(c['note_any'])} | "
                 f"{fr(c['use_get'])} | {fd(c.get('use_get_minus_plain'))} | "
                 f"{fr(c['p_use_code'])} | {fr(c['rule_broken'])} | "
                 f"{fr(ds.get(arm, {}).get('use_get'))} |")
    L += ["", "POST HOC (not in the plan): where the NOTEs were stored. A NOTE at the job's own "
          "output path is its report file holding a message such as \"BLOCKED: input "
          "missing\".", "",
          "| model | cell | n | NOTE (any) | NOTE outside the own output path | NOTE only at "
          "the own output path | addressed |", "|---|---|---|---|---|---|---|"]

    def row(m: str, cell: str, c: dict) -> None:
        L.append(f"| {m} | {cell} | {c['n']} | {fr(c['note_any'])} | {fr(c['note_elsewhere'])}"
                 f" | {fr(c['note_only_at_own_output'])} | {fr(c['note_addressed'])} |")
    for m, d in s["probes"].items():
        pl = d["posthoc_note_places"]
        for arm, c in pl["confirmatory_unseeded"].items():
            row(m, f"confirmatory {arm}, no seed note", c)
        row(m, "this run, plain-norule, seeded", pl["this_run_rule_off"])
    for arm, c in r["arms"].items():
        row(r["model"], f"replication {arm}-rule", c["posthoc_note_places"])
    L += ["", "POST HOC: p_j (rule off) not counting NOTEs stored at the own output path: "
          + "; ".join(f"{m} {fr(d['norule_this_run']['pooled']['posthoc_p_j_elsewhere'])}"
                      for m, d in s["probes"].items())]
    return "\n".join(L)


# ---- markdown (the Results section of the plan doc) -----------------------------------------
MD = ROOT / "docs" / "ideas" / "write_find_answer.md"
SHORT = {"deepseek-v4-flash": "DeepSeek V4 Flash", "gemma-4-31b-it": "Gemma 4 31B",
         "glm-5.2": "GLM 5.2", "qwen3.5-397b": "Qwen 3.5 397B"}


def ko(d: dict) -> str:
    return f"{d['k']} of {d['n']}"


def pr(d: dict) -> str:
    return "n/a" if not d or not d["n"] else f"{d['p']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]"


def outcome_lines(s: dict) -> list[str]:
    P = s["probes"]
    pj = {m: d["norule_this_run"]["pooled"]["p_j"] for m, d in P.items()}
    verdicts = {m: d["verdict_norule"] for m, d in P.items()}
    L = ["Verdict per model (p_j without the rule, pooled over the two notes): " + "; ".join(
        f"{SHORT[m]} {ko(pj[m])}, {pr(pj[m])}, {verdicts[m]}" for m in P) + "."]
    if all(v == "near zero" for v in verdicts.values()):
        top = max(pj.values(), key=lambda d: (d["p"], d["hi"]))
        hi = max(d["hi"] for d in pj.values())
        vs = [d["norule_this_run"]["pooled"]["v_seen"] for d in P.values()]
        us = [d["norule_this_run"]["pooled"]["use_get"] for d in P.values()]
        vmin, vmax = min(vs, key=lambda d: d["p"]), max(vs, key=lambda d: d["p"])
        umin, umax = min(us, key=lambda d: d["p"]), max(us, key=lambda d: d["p"])
        L.append(f"Every model is near zero, so the first probe sentence holds: \"Without any "
                 f"house rule, agents that saw a peer's note almost never answered or joined "
                 f"it: p_j was at most {ko(top)} in every model (largest Wilson upper bound "
                 f"{hi:.2f}), while they saw the note in {ko(vmin)} to {ko(vmax)} episodes and "
                 f"read other jobs' artifacts in {ko(umin)} to {ko(umax)}. Untrained agents "
                 f"write and read in shared places but do not answer, so answering is the "
                 f"factor that collaboration training would have to move.\"")
    for m, v in verdicts.items():
        rule = P[m]["rule_d6_confirmatory"]["pooled"]["p_j"]
        if v == "answers":
            ref = f", against {ko(rule)} with the rule" if rule["n"] else \
                " (the D6 probes with the rule have not run for this model)"
            L.append(f"{SHORT[m]} answers, so the second probe sentence holds for it: \"Without "
                     f"the house rule, {SHORT[m]} answered or joined a peer's note in "
                     f"{ko(pj[m])} episodes in which it saw the note (p_j = {pr(pj[m])})"
                     f"{ref}; the answering step is not zero once nothing forbids it, so in "
                     f"the confirmatory probes the rule, not the model, kept p_j at zero.\"")
        elif v == "between":
            L.append(f"{SHORT[m]} is between (p_j = {ko(pj[m])}, {pr(pj[m])}): neither probe "
                     f"sentence holds for it.")
    if not all(d["norule_this_run"]["pooled"]["complete"] for d in P.values()):
        L.append("Some probe cells are incomplete; the verdicts above are provisional.")
    r = s["replication"]["arms"]
    for arm in RW_ARMS[1:]:
        c, p0, d = r[arm], r["plain"], r[arm].get("use_get_minus_plain")
        if d is None:
            L.append(f"{arm}: no data yet.")
            continue
        if c["replicates"]:
            L.append(f"{arm} replicates: \"On Qwen 3.5 397B, as on DeepSeek V4 Flash, the "
                     f"sentence raised reads of other jobs' artifacts under the house rule, from "
                     f"{ko(p0['use_get'])} with the plain prompt to {ko(c['use_get'])} "
                     f"(difference {fd(d)}), with {c['note_addressed']['k']} addressed "
                     f"notes.\"")
        else:
            L.append(f"{arm} does not replicate: \"On Qwen 3.5 397B the sentence did not raise "
                     f"reads of other jobs' artifacts under the house rule ({ko(p0['use_get'])} "
                     f"plain, {ko(c['use_get'])} with the sentence, difference {fd(d)}); the "
                     f"DeepSeek V4 Flash effect is not shown in a second model family.\"")
    return L


def interpretation(s: dict) -> list[str]:
    """POST HOC reading of the planned numbers (printed, not part of the planned analysis)."""
    P = s["probes"]
    off = {m: d["norule_this_run"] for m, d in P.items()}
    on = {m: d["rule_d6_confirmatory"] for m, d in P.items()}
    L = ["Request notes were mostly seen in a listing and not opened. Read it, request note, "
         "rule off: " + "; ".join(f"{SHORT[m]} {ko(off[m]['request']['read'])}" for m in P)
         + "; rule on: " + "; ".join(f"{SHORT[m]} {ko(on[m]['request']['read'])}" for m in P)
         + ". So for the request note p_j = 0 mostly means the agent did not open it.",
         "Where the note was opened (mostly the memory note), opening did not lead to an "
         "answer either. p_j if read, rule off: " + "; ".join(
             f"{SHORT[m]} {ko(off[m]['pooled']['p_j_if_read'])}" for m in P) + ".",
         "Removing the rule changed use, not answering. USE (get), rule off minus rule on: "
         + "; ".join(f"{SHORT[m]} {fd(d['norule_minus_rule']['use_get'])}" for m, d in P.items())
         + "; p_j: " + "; ".join(f"{SHORT[m]} {fd(d['norule_minus_rule']['p_j'])}"
                                  for m, d in P.items()) + "."]
    pl = {m: d["posthoc_note_places"] for m, d in P.items()}
    cu = {m: x["confirmatory_unseeded"] for m, x in pl.items()}
    L.append("Most NOTEs that count toward p_w are the job's own report file holding a message "
             "(\"BLOCKED: input missing\"), stored at the path its ticket names. Unseeded plain "
             "cells of the confirmatory run, NOTE (any) and NOTE outside the own output path, "
             "rule on / rule off: " + "; ".join(
                 f"{SHORT[m]} {ko(c['plain-rule']['note_any'])} and "
                 f"{ko(c['plain-rule']['note_elsewhere'])} / {ko(c['plain-norule']['note_any'])}"
                 f" and {ko(c['plain-norule']['note_elsewhere'])}" for m, c in cu.items())
             + ". So the \"write\" half of the probe sentence (\"write and read in shared "
             "places\") holds only for such report files: notes left elsewhere in the store "
             "are about as rare as addressed notes.")
    an = [x for d in P.values() for x in d["after_seen_notes"]]
    L.append(f"After-seeing NOTEs in this run that are the job's own report file: "
             f"{sum(x['at_own_output'] for x in an)} of {len(an)} (listed below). Not counting "
             "such files, p_j without the rule is " + "; ".join(
                 f"{SHORT[m]} {ko(off[m]['pooled']['posthoc_p_j_elsewhere'])}" for m in P)
             + ".")
    r = s["replication"]["arms"]
    ds = s["replication"]["deepseek_v4_flash_same_worlds"]
    if ds.get("plain"):
        L.append("Replication: Qwen 3.5 397B under the house rule almost never reads other jobs' "
                 f"artifacts ({ko(r['plain']['use_get'])} plain), so the sentences had little "
                 "to raise. DeepSeek V4 Flash on the same worlds (another run, read as it "
                 f"stood): plain {ko(ds['plain']['use_get'])}, "
                 + ", ".join(f"{a} {ko(ds[a]['use_get'])}" for a in RW_ARMS[1:] if a in ds)
                 + ".")
    return L


def headline(s: dict) -> list[str]:
    L = ["| model | rule | p_w note, no seed (c1c3) | p_w addressed | v (saw it) | read it | "
         "p_j (if seen) | p_j if read | answered, request (if seen) | joined | note after "
         "seeing | USE (get) | use in code |", "|---|---|---|---|---|---|---|---|---|---|---|"
                                                "---|---|"]
    for m, d in s["probes"].items():
        for lab, blk, arm in (("off", d["norule_this_run"], "plain-norule"),
                              ("on", d["rule_d6_confirmatory"], "plain-rule")):
            w = d["p_w_unseeded_c1c3"].get(arm, {})
            c = blk["pooled"]
            nn = f" (n {c['n']} of {c['n_expected']})" if not c["complete"] else ""
            wn = "" if w.get("complete", True) else " (cell incomplete)"
            if not c["n"]:
                L.append(f"| {SHORT[m]} | {lab}{nn} | {fr(w.get('note_any'))}{wn} | "
                         f"{fr(w.get('note_addressed'))} | not run yet | | | | | | | | |")
                continue
            L.append(f"| {SHORT[m]} | {lab}{nn} | {fr(w.get('note_any'))}{wn} | "
                     f"{fr(w.get('note_addressed'))} | {fr(c['v_seen'])} | {fr(c['read'])} | "
                     f"{fr(c['p_j'])} | {fr(c['p_j_if_read'])} | {fr(c['answered'])} | "
                     f"{fr(c['joined'])} | {fr(c['note_after_seen'])} | {fr(c['use_get'])} | "
                     f"{fr(c['p_use_code'])} |")
    return L


def clip(t: str, n: int = 160) -> str:
    t = " / ".join(x.strip() for x in t.splitlines() if x.strip()).replace("|", "/")
    return t if len(t) <= n else t[:n] + "..."


def write_md(s: dict) -> None:
    run = s.get("run") or {}
    http = s.get("http") or {}
    led = s["ledger"]
    L = ["## Results", "",
         "Generated by `experiments/write_find_answer.py --md` from "
         "`results/explore/write_find_answer.json`; do not edit by hand. EXPLORATORY, not "
         "pre-registered.", "",
         f"Run: {s['episodes']} of 550 episodes, ledger ${led['spend_usd']:.2f} over "
         f"{led['calls']} calls (" + ", ".join(f"{k.split('/')[-1]} ${v['usd']:.2f}"
                                              for k, v in led.get("by_model", {}).items())
         + ")" + (f", wall {run['wall_s'] / 3600:.1f} h" if run.get("wall_s") else "")
         + (f", HTTP 429 in {http['n_429']} of {http['responses']} responses"
            if http.get("responses") else "") + ". Rule off: this run (worlds 5300-5349 and "
         "5500-5549). Rule on: the D6 confirmatory probes (worlds 2300- and 2500-, read only). "
         "p_w: the unseeded plain cells of `results/confirmatory/c1c3.json` (30 turns, "
         "background 200). k/n = share [Wilson 95%].", "",
         "### Write, find, answer per model (pooled over the request and the memory note)", ""]
    L += headline(s)
    L += ["", "### Outcome (the planned rule)", ""] + [f"- {x}" for x in outcome_lines(s)]
    L += ["", "### Reading of the results (POST HOC, not part of the planned analysis)", ""]
    L += [f"- {x}" for x in interpretation(s)]
    L += ["", "### Every cell", "", tables(s), "",
          "### Notes written after seeing the seed note, rule off (first 160 characters, at "
          "most 5 per model)", ""]
    for m, d in s["probes"].items():
        ns = d["after_seen_notes"]
        L.append(f"- {SHORT[m]}: notes {len(ns)}")
        for x in ns[:5]:
            own = ", the job's own output path" if x["at_own_output"] else ""
            L.append(f"  - world {x['seed']}, {x['note']} note, {x['kind']}, `{x['name']}`"
                     f"{own}: {clip(x['text'])}")
    L += ["", "### Notes in the replication (first 160 characters, at most 5 per arm)", ""]
    for arm, c in s["replication"]["arms"].items():
        L.append(f"- {arm}: notes {len(c['notes'])}, kinds {json.dumps(c['note_kinds'])}")
        for x in c["notes"][:5]:
            L.append(f"  - world {x['seed']}, {x['kind']}, `{x['name']}`: {clip(x['text'])}")
    text = MD.read_text()
    head = text.split("\n## Results\n")[0].rstrip("\n")
    MD.write_text(head + "\n\n" + "\n".join(L) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--md", action="store_true",
                    help="with --table: also rewrite the Results section of the plan doc")
    ap.add_argument("--dry", action="store_true", help="scripted agents, no LLM")
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--budget", type=float, default=BUDGET)
    a = ap.parse_args(argv)
    if a.concurrency > CONCURRENCY:
        ap.error(f"at most {CONCURRENCY} calls in flight (the key is shared)")
    if a.dry and a.raw == RAW:
        ap.error("--dry needs its own --raw file")
    extra: dict = {}
    if not a.table:
        specs = all_specs(a.dry)
        t0, last = time.time(), [0.0]
        if a.dry:
            from swarm.scale.scripted import ScriptedPolicy
            styles = ("joiner", "responder", "browser", "noter", "worker")
            policy_for = lambda s: ScriptedPolicy(styles[s.seed % len(styles)])  # noqa: E731
            llm = None
        else:
            from swarm.scale.llm_task import TaskLLM
            llm = TaskLLM(budget_usd=a.budget, max_concurrency=a.concurrency, cache_path=CACHE)
            pols = {m: LLMPolicy(llm, f"adaption/{m}", max_tokens=max_tokens(m),
                                 temperature=TEMPERATURE) for m in PROBE_MODELS}
            policy_for = lambda s: pols[s.model.split("/", 1)[1]]  # noqa: E731

        def progress(k: int, n: int) -> None:
            if time.time() - last[0] > 60 or k == n:
                last[0] = time.time()
                sp = f" spend ${llm.total_spent():.3f}" if llm else ""
                print(f"{k}/{n} episodes, {time.time() - t0:.0f}s, rss {peak_rss_mb():.0f}MB"
                      f"{sp}", flush=True)

        from swarm.scale.http_stats import count_http
        with count_http() as hs:
            res = asyncio.run(run_many(specs, policy_for, a.raw, concurrency=a.concurrency,
                                       progress=progress))
        extra = {"run": res, "http": hs.summary(), "concurrency": a.concurrency}
        if llm is not None:
            extra["llm_summary"] = llm.summary()
            print(llm.summary(), flush=True)
        print("run:", json.dumps(res), "http:", json.dumps(extra["http"]), flush=True)
    s = summarize(a.raw, a.dry)
    if not a.dry:
        if OUT.exists() and not extra:  # keep the last run's cost and throughput fields
            old = json.loads(OUT.read_text())
            s.update({k: v for k, v in old.items() if k not in s})
        s.update(extra)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(s, indent=1))
        if a.md:
            write_md(s)
    print(tables(s))
    print("\n".join(outcome_lines(s)))


if __name__ == "__main__":
    main()
