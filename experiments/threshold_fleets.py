"""EXPLORATORY fleets near the threshold R = 1, predictions committed before each fleet runs
(docs/ideas/threshold_fleets.md). After the frozen C4 set; not part of C1-C5.

    uv run python experiments/threshold_fleets.py intervals   # C4 intervals -> results/explore/c4_intervals.json
    caffeinate -i uv run python experiments/threshold_fleets.py probes   # single-agent probes
    uv run python experiments/threshold_fleets.py inputs      # probe and cell counts -> threshold_inputs.json
    uv run python experiments/threshold_fleets.py scan        # predicted R for k = 0..blocked (print only)
    uv run python experiments/threshold_fleets.py predict --plan 64:4,256:16   # write predictions
    caffeinate -i uv run python experiments/threshold_fleets.py fleet --tag <tag>  # one fleet
    uv run python experiments/threshold_fleets.py eval        # outcomes, calibration, doc tables

Condition: a mixed fleet of one model. k of the blocked jobs get the permission sentence
(prompt arm coordinate_ok with the house rule and its exception, swarm.scale.mix); every other
job gets the plain prompt with the house rule. Fleet settings as in C4 (30 turns, 30% blocked,
a producer for each blocked job, 200 background jobs, temperature 0.8, sample offset 1, rounds
schedule). Inputs of the prediction (swarm.scale.nucleation): p_w per arm from the model's
confirmatory C1 cell (read only), v and p_j per arm from new seeded probes with the store
background set to the fleet size, world seeds disjoint from every earlier run.

Calls: adaption/<model> through swarm.scale.llm_task.TaskLLM with its own cache and ledger
results/cache/threshold_llm.sqlite, at most 32 calls in flight, spend cap $40. The script never
touches the confirmatory runner or its files; it reads results/confirmatory/*.json,
results/predictions/c4.json and the confirmatory raw files only.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from swarm.scale.nucleation import B, Group, brier, from_wr, intervals, mixture_point  # noqa: E402
from swarm.scale.runner import LLMPolicy, RunSpec, peak_rss_mb, run_many, run_world  # noqa: E402

LABEL = "EXPLORATORY"
MODEL = "gemma-4-26b-a4b-it"
MAX_TOKENS = 2000  # as in the confirmatory set for this model (D7)
TEMPERATURE = 0.8
STEPS = 30
REPEAT = 1
CONC = 32
BUDGET = 40.0
ARMS = ("coordinate_ok", "plain")
PERMISSION = "coordinate_ok"
BGS = (64, 256)
K_PROBE = 25  # per note style: 25 request + 25 memory = 50 probe episodes per arm and size
PROBE_SEEDS = {"request": 6300, "memory": 6500}
FLEET_SEEDS = (6900, 6901)
# world seeds of earlier runs: 0-99, 200, 300-301, 1000-2049, 2100-2199, 2300-2349,
# 2500-2549, 2900-2901, 3000-3099, 4000-4999; these use 6300-6324, 6500-6524, 6900-6901

RAW = ROOT / "results" / "raw" / "scale"
CACHE = ROOT / "results" / "cache" / "threshold_llm.sqlite"
EXPLORE = ROOT / "results" / "explore"
INPUTS = EXPLORE / "threshold_inputs.json"
PRED = EXPLORE / "threshold_predictions.json"
OUTCOMES = EXPLORE / "threshold_outcomes.json"
FLEET_OUT = EXPLORE / "threshold_fleets"
C4_PRED = ROOT / "results" / "predictions" / "c4.json"
C4_OUT = ROOT / "results" / "confirmatory" / "c4.json"
DOC = ROOT / "docs" / "ideas" / "threshold_fleets.md"
LOG = RAW / "logs" / "threshold.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def physmem() -> str:
    try:
        out = subprocess.run(["top", "-l", "1"], capture_output=True, text=True,
                             timeout=30).stdout
        return next((ln for ln in out.splitlines() if ln.startswith("PhysMem")), "?")
    except Exception:  # noqa: BLE001
        return "?"


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def committed_json(path: Path) -> dict | None:
    r = subprocess.run(["git", "show", f"HEAD:{path.relative_to(ROOT)}"], cwd=ROOT,
                       capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 else None


def ledger() -> dict:
    import sqlite3
    if not CACHE.exists():
        return {"spend_usd": 0.0, "calls": 0}
    db = sqlite3.connect(f"file:{CACHE}?mode=ro", uri=True, timeout=30)
    try:
        rows = db.execute("SELECT model, COUNT(*), COALESCE(SUM(cost), 0) FROM spend "
                          "GROUP BY model").fetchall()
    except sqlite3.OperationalError:
        rows = []
    db.close()
    return {"spend_usd": round(sum(r[2] for r in rows), 4), "calls": sum(r[1] for r in rows),
            "by_model": {r[0]: {"calls": r[1], "usd": round(r[2], 4)} for r in rows}}


# ---- 1. intervals of the committed C4 predictions --------------------------------------------
def group_of_c4(d: dict) -> Group:
    v, pj, pw = d["v_rate"], d["p_j_rate"], d["p_w_rate"]
    if pj["n"] != v["k"]:
        raise RuntimeError(f"p_j denominator {pj['n']} != seen {v['k']}")
    return Group(agents=d["N"], writers=d["b"] * d["N"], kw=pw["k"], nw=pw["n"],
                 n_probe=v["n"], n_seen=v["k"], n_join=pj["k"])


def cmd_intervals(_a) -> None:
    raw = C4_PRED.read_bytes()
    pred = json.loads(raw)
    out = {"label": LABEL, "source": str(C4_PRED.relative_to(ROOT)),
           "source_sha256": hashlib.sha256(raw).hexdigest(),
           "note": "Intervals for the committed C4 predictions; the frozen file is read, never "
                   "written. Inputs are the counts stored in each entry (p_w of the C1/C2 cell, "
                   "v and p_j of the probes, D6). bootstrap: episodes resampled with "
                   "replacement; jeffreys: Beta(k + 1/2, n - k + 1/2) draws. 10,000 draws each, "
                   "95% percentile intervals. R_wilson is the product interval already stored "
                   "in the entry (R_lo, R_hi).",
           "fleets": {}}
    for tag, d in pred["fleets"].items():
        g = group_of_c4(d)
        sr = d["seed_note"] == "request"
        pt = mixture_point([g], seed_request=sr)
        if not (math.isclose(pt["R"], d["R"], abs_tol=1e-9)
                and math.isclose(pt["W"], d["W"], abs_tol=1e-9)):
            raise RuntimeError(f"{tag}: recomputed W, R {pt['W']}, {pt['R']} != committed "
                               f"{d['W']}, {d['R']}")
        iv = intervals([g], seed_request=sr)
        out["fleets"][tag] = {
            "N": d["N"], "model": d["model"], "prompt": d["prompt"], "seed_note": d["seed_note"],
            "inputs": {"p_w": [g.kw, g.nw], "v": [g.n_seen, g.n_probe],
                       "p_j": [g.n_join, g.n_seen]},
            "point": {"W": d["W"], "R": d["R"], "P_board": d["P_board"],
                      "P_1link": pt["P_1link"], "predicted_board": d["predicted_board"]},
            "R_wilson": [d["R_lo"], d["R_hi"]], "written": pred["written"].get(tag), **iv}
        b, j = iv["bootstrap"], iv["jeffreys"]
        log(f"{tag}: R {d['R']:.2f} bootstrap [{b['R']['lo']:.2f}, {b['R']['hi']:.2f}] "
            f"jeffreys [{j['R']['lo']:.2f}, {j['R']['hi']:.2f}] P(R>1) jeffreys "
            f"{j['P_R_above_1']:.3f}; W {d['W']:.2f} jeffreys [{j['W']['lo']:.2f}, "
            f"{j['W']['hi']:.2f}]; P_board {d['P_board']:.3f} jeffreys "
            f"[{j['P_board']['lo']:.3f}, {j['P_board']['hi']:.3f}]")
    EXPLORE.mkdir(parents=True, exist_ok=True)
    (EXPLORE / "c4_intervals.json").write_text(json.dumps(out, indent=1))
    log(f"wrote results/explore/c4_intervals.json ({len(out['fleets'])} committed C4 entries)")
    write_doc_tables()


# ---- 2. probes ----------------------------------------------------------------------------
def probe_tag(model: str, bg: int) -> str:
    return f"thrprobe_{model}_bg{bg}"


def probe_specs(model: str = MODEL) -> list[RunSpec]:
    out = []
    for prompt in ARMS:  # permission cells first, then plain; seed-major within an arm
        for i in range(K_PROBE):
            for note in ("request", "memory"):
                for bg in BGS:
                    out.append(RunSpec(tag=probe_tag(model, bg), mode="harness", prompt=prompt,
                                       rule=True, seed_note=note, n_background=bg, steps=STEPS,
                                       model=f"adaption/{model}", max_tokens=MAX_TOKENS,
                                       temperature=TEMPERATURE, seed=PROBE_SEEDS[note] + i,
                                       repeat=REPEAT))
    return out


def probe_path(model: str = MODEL) -> Path:
    return RAW / f"thrprobe_{model}.jsonl"


def cmd_probes(a) -> None:
    from swarm.scale.http_stats import count_http
    from swarm.scale.llm_task import TaskLLM
    specs = probe_specs(a.model)
    log(f"=== {LABEL} probes {a.model}: {len(specs)} episodes, {CONC} in flight; {physmem()}; "
        f"ledger ${ledger()['spend_usd']:.2f}")
    llm = TaskLLM(budget_usd=BUDGET, max_concurrency=CONC, cache_path=CACHE)
    pol = LLMPolicy(llm, f"adaption/{a.model}", max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
    t0, last = time.time(), [0.0]

    def progress(k: int, n: int) -> None:
        if time.time() - last[0] > 120 or k == n:
            last[0] = time.time()
            log(f"probes {k}/{n} episodes, {time.time() - t0:.0f}s, rss {peak_rss_mb():.0f}MB, "
                f"spend ${llm.total_spent():.3f}")
    with count_http() as hs:
        res = asyncio.run(run_many(specs, lambda s: pol, probe_path(a.model), concurrency=CONC,
                                   progress=progress))
    log(f"probes done: {json.dumps(res)}; http {json.dumps(hs.summary())}")
    log(llm.summary())


# ---- 3. inputs ----------------------------------------------------------------------------
def cell_rate(model: str, prompt: str) -> dict:
    """p_w (NOTE of any kind) of the model's confirmatory C1 cell (plain or coordinate_ok, house
    rule on, no seed note), read from the confirmatory raw file as it stands."""
    import scale_confirm as SC
    import scale_confirm_eval as EV
    want = {s.episode_id() for s in SC.cell_specs(model) if s.prompt == prompt and s.rule}
    rows, _ = EV.load_rows(SC.out_path(model, False), want)
    rs = [r for r in rows if r["arm"] == f"{prompt}-rule"]
    d = EV.rate([bool(r["wrote_note"]) for r in rs])
    d["n_expected"] = len(want)
    return d


def probe_rates(model: str) -> dict:
    import scale_confirm_eval as EV
    specs = probe_specs(model)
    rows, tot = EV.load_rows(probe_path(model), {s.episode_id() for s in specs})
    out: dict = {}
    for bg in BGS:
        for prompt in ARMS:
            rs = [r for r in rows if r["arm"] == f"{prompt}-rule"
                  and r["episode_id"].startswith(probe_tag(model, bg) + "/")]
            seen = [r for r in rs if r["seed_seen"]]
            d = {"n_expected": 2 * K_PROBE, "complete": len(rs) == 2 * K_PROBE,
                 "n_background": bg,
                 "v_seen": EV.rate([bool(r["seed_seen"]) for r in rs]),
                 "p_j": EV.rate([bool(r["joined"] or r.get("responded")) for r in seen]),
                 "read": EV.rate([bool(r["seed_read"]) for r in rs]),
                 "episodes_with_failed_turns": sum(r["_ep_errors"] > 0 for r in rs),
                 "by_style": {}}
            for st in ("request", "memory"):
                ss = [r for r in rs if r["note"] == st]
                se = [r for r in ss if r["seed_seen"]]
                d["by_style"][st] = {
                    "seen": EV.rate([bool(r["seed_seen"]) for r in ss]),
                    "read": EV.rate([bool(r["seed_read"]) for r in ss]),
                    "joined_or_responded_if_seen": EV.rate(
                        [bool(r["joined"] or r.get("responded")) for r in se])}
            out[f"{prompt}-rule|bg{bg}"] = d
    return out


def cmd_inputs(a) -> dict:
    res = {"label": LABEL, "model": a.model, "written": time.strftime("%Y-%m-%d %H:%M:%S %z"),
           "git_head": git_head(),
           "cells": {f"{p}-rule": cell_rate(a.model, p) for p in ARMS},
           "probes": probe_rates(a.model), "ledger": ledger()}
    EXPLORE.mkdir(parents=True, exist_ok=True)
    INPUTS.write_text(json.dumps(res, indent=1))
    for k, d in res["cells"].items():
        log(f"cell {k}: p_w {d['k']}/{d['n']} (of {d['n_expected']})")
    for k, d in res["probes"].items():
        log(f"probe {k}: seen {d['v_seen']['k']}/{d['v_seen']['n']} read {d['read']['k']} "
            f"joined or answered {d['p_j']['k']}/{d['p_j']['n']} complete {d['complete']}")
    return res


# ---- 4. predictions ---------------------------------------------------------------------
def fleet_tag(model: str, n: int, k: int, seed: int) -> str:
    return f"thr_{model}_n{n}_k{k}_{PERMISSION}_s{seed}"


def groups_for(inp: dict, n: int, k: int | str) -> list[Group]:
    """k permitted blocked jobs and the plain rest, or k = "all": every job permitted (one
    group, the frozen formula)."""
    gs = []
    plan = (((PERMISSION, n, B * n),) if k == "all"
            else ((PERMISSION, k, k), ("plain", n - k, B * n - k)))
    for prompt, agents, writers in plan:
        c = inp["cells"][f"{prompt}-rule"]
        p = inp["probes"][f"{prompt}-rule|bg{n}"]
        if not p["complete"]:
            raise RuntimeError(f"probes {prompt}-rule|bg{n} incomplete")
        gs.append(Group(agents=agents, writers=writers, kw=c["k"], nw=c["n"],
                        n_probe=p["v_seen"]["n"], n_seen=p["v_seen"]["k"],
                        n_join=p["p_j"]["k"]))
    return gs


def cmd_scan(_a) -> None:
    inp = json.loads(INPUTS.read_text())
    for n in BGS:
        for k in range(0, round(B * n) + 1):
            pt = mixture_point(groups_for(inp, n, k))
            print(f"N {n} k {k:>3}: W {pt['W']:.2f} R {pt['R']:.2f} P_board {pt['P_board']:.3f} "
                  f"P_1link {pt['P_1link']:.3f}")


def spec_for(model: str, n: int, seed: int, tag: str, prompt: str = "plain") -> RunSpec:
    return RunSpec(tag=tag, mode="fleet", n_agents=n, p_blocked=B, prompt=prompt, rule=True,
                   seed_note="none", n_background=200, p_derivative=1.0,
                   producers_in_fleet=True, steps=STEPS, model=f"adaption/{model}",
                   max_tokens=MAX_TOKENS, temperature=TEMPERATURE, seed=seed, repeat=REPEAT,
                   schedule="rounds", history_window=12)


def cmd_predict(a) -> None:
    from swarm.scale.mix import choose_jobs
    inp = json.loads(INPUTS.read_text())
    old = json.loads(PRED.read_text()) if PRED.exists() else {"label": LABEL, "fleets": {}}
    ts = time.strftime("%Y-%m-%d %H:%M:%S %z")
    for item in a.plan.split(","):
        n_s, k_s = item.split(":")
        n, k = int(n_s), (k_s if k_s == "all" else int(k_s))
        for seed in FLEET_SEEDS[:a.seeds]:
            tag = fleet_tag(inp["model"], n, k, seed)
            if tag in old["fleets"]:
                log(f"{tag}: prediction exists, kept as written")
                continue
            if Path(f"{RAW / tag}.rounds.jsonl").exists():
                raise RuntimeError(f"{tag}: fleet output exists; predictions must come first")
            gs = groups_for(inp, n, k)
            pt = mixture_point(gs)
            iv = intervals(gs)
            jobs = ("all" if k == "all"
                    else choose_jobs(spec_for(inp["model"], n, seed, tag), k))
            old["fleets"][tag] = {
                "model": inp["model"], "N": n, "k_permission_blocked": k,
                "n_blocked": round(B * n), "seed": seed, "permission_jobs": jobs,
                "groups": dict(zip((PERMISSION, "plain"), (vars(g) for g in gs))),
                "inputs_written": inp["written"], "point": pt, **iv,
                "permission_rule_predicts_board": k == "all" or k > 0,
                "written": ts, "git_head_at_write": git_head()}
            b, j = iv["bootstrap"], iv["jeffreys"]
            log(f"{tag}: W {pt['W']:.2f} R {pt['R']:.2f} boot [{b['R']['lo']:.2f}, "
                f"{b['R']['hi']:.2f}] jeff [{j['R']['lo']:.2f}, {j['R']['hi']:.2f}] P_board "
                f"{pt['P_board']:.3f} [{b['P_board']['lo']:.3f}, {b['P_board']['hi']:.3f}] "
                f"P_1link {pt['P_1link']:.3f} -> {'board' if pt['predicted_board'] else 'no board'}")
    old["formula"] = __import__("swarm.scale.nucleation", fromlist=["x"]).__doc__
    PRED.write_text(json.dumps(old, indent=1))
    write_doc_tables()
    log(f"wrote {PRED.relative_to(ROOT)} and the prediction table of {DOC.relative_to(ROOT)}")


# ---- 5. one fleet -----------------------------------------------------------------------
def outcome(rows: list[dict], rec: dict, jobs: list[str]) -> dict:
    import scale_confirm_eval as EV
    board = [r for r in rows if r["joined_peer"] or r["replied_peer"]]
    perm = [r for r in rows if r["mixed"]]
    plain = [r for r in rows if not r["mixed"]]

    def cnt(rs, key):
        return sum(bool(r[key]) for r in rs)
    d = {"label": LABEL, "n_jobs": len(rows), "board": bool(board), "n_board_jobs": len(board),
         "board_jobs_permission": sum(r["mixed"] for r in board),
         "board_jobs_plain": sum(not r["mixed"] for r in board),
         "permission_jobs": jobs, "permission_jobs_blocked": sum(bool(r["blocked"]) for r in perm)}
    for g, rs in (("permission", perm), ("plain", plain)):
        for key in ("wrote_note", "wrote_note_addressed", "note_read_by_other", "read_peer_note",
                    "joined_peer", "replied_peer", "report_blocked", "success"):
            d[f"{g}_{key}"] = cnt(rs, key)
    d["plain_blocked_note_any"] = EV.rate([bool(r["wrote_note"]) for r in plain if r["blocked"]])
    d["note_kinds"] = {}
    for r in rows:
        for _, kind, _ in r["note_texts"]:
            d["note_kinds"][kind] = d["note_kinds"].get(kind, 0) + 1
    t = rec["totals"]
    d.update(turns=t["calls"], errors=t["errors"], malformed=t["malformed"],
             cost_usd_nominal=round(t["cost"], 4), stopped=rec["stopped"], wall_s=rec["wall_s"])
    return d


def cmd_fleet(a) -> None:
    from scale_fleet import throughput
    from swarm.scale.http_stats import count_http
    from swarm.scale.llm_task import TaskLLM
    from swarm.scale.mix import MixedPromptPolicy, choose_jobs, mixed_rows
    pred = json.loads(PRED.read_text())["fleets"]
    comm = committed_json(PRED) or {"fleets": {}}
    if a.tag not in pred or a.tag not in comm["fleets"]:
        raise RuntimeError(f"{a.tag}: no committed prediction; commit it first")
    if comm["fleets"][a.tag] != pred[a.tag]:
        raise RuntimeError(f"{a.tag}: prediction differs from the committed one")
    if subprocess.run(["git", "ls-files", "--error-unmatch", str(DOC.relative_to(ROOT))],
                      cwd=ROOT, capture_output=True).returncode != 0:
        raise RuntimeError("commit docs/ideas/threshold_fleets.md before the first fleet")
    p = pred[a.tag]
    out = FLEET_OUT / f"{a.tag}.json"
    if out.exists() and not a.force:
        log(f"{a.tag}: finished already ({out.relative_to(ROOT)}), skipped")
        return
    lg = ledger()
    per_call = (lg["by_model"].get(f"adaption/{p['model']}", {}).get("usd", 0.0)
                / max(1, lg["by_model"].get(f"adaption/{p['model']}", {}).get("calls", 0))
                or 0.0005)
    proj = p["N"] * 15 * per_call
    if lg["spend_usd"] + proj > BUDGET:
        raise RuntimeError(f"budget: spend ${lg['spend_usd']:.2f} + projected ${proj:.2f} > "
                           f"${BUDGET:.0f}")
    pure = p["k_permission_blocked"] == "all"
    spec = spec_for(p["model"], p["N"], p["seed"], a.tag, PERMISSION if pure else "plain")
    if pure:
        from swarm.scale.jobs import make_world
        jobs = make_world(spec.world_config()).job_ids
    else:
        jobs = choose_jobs(spec, p["k_permission_blocked"])
        if jobs != p["permission_jobs"]:
            raise RuntimeError("permission jobs differ from the committed prediction")
    log(f"=== {LABEL} fleet {a.tag} start: N {p['N']}, {len(jobs)} permission jobs (blocked), "
        f"predicted R {p['point']['R']:.2f} P_board {p['point']['P_board']:.3f}; spend "
        f"${lg['spend_usd']:.2f}, projected ${proj:.2f}; {physmem()}")
    llm = TaskLLM(budget_usd=BUDGET, max_concurrency=CONC, cache_path=CACHE)
    pol = LLMPolicy(llm, spec.model, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
    if not pure:  # the spec's own arm is the permission arm in a pure fleet
        pol = MixedPromptPolicy(pol, jobs, STEPS, prompt=PERMISSION)

    def progress(rr: dict) -> None:
        log(f"{a.tag} step {rr['step']:>2} active {rr['active']:>4} live {rr['live']:>4} "
            f"cached {rr['cached']:>4} err {rr['errors']:>3} {rr['wall_s']:.0f}s "
            f"rss {rr['peak_rss_mb']}MB ${rr['cost']:.3f}")
    t0 = time.time()
    prefix = RAW / a.tag
    with count_http() as hs:
        rec, labels = asyncio.run(run_world(spec, pol, sem=asyncio.Semaphore(CONC),
                                            out_prefix=prefix, progress=progress))
    rows = mixed_rows(rec, labels, prefix, jobs, prompt=PERMISSION)
    d = outcome(rows, rec, jobs)
    d.update(tag=a.tag, elapsed_s=round(time.time() - t0, 1), http=hs.summary(),
             throughput=throughput(rec["rounds"]), task_spend_usd=round(llm.total_spent(), 4),
             peak_rss_mb_process=round(peak_rss_mb(), 1), physmem_end=physmem())
    FLEET_OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=1))
    log(f"=== {LABEL} {a.tag}: board {d['board']} ({d['n_board_jobs']} jobs: "
        f"{d['board_jobs_permission']} permission, {d['board_jobs_plain']} plain), notes "
        f"permission {d['permission_wrote_note']} plain {d['plain_wrote_note']}, cost "
        f"${d['cost_usd_nominal']:.2f}, wall {d['wall_s']:.0f}s")
    log(llm.summary())


# ---- 6. evaluation ----------------------------------------------------------------------
def cmd_eval(_a) -> dict:
    pred = json.loads(PRED.read_text())["fleets"] if PRED.exists() else {}
    res: dict = {"label": LABEL, "fleets": {}, "ledger": ledger()}
    pairs: dict[str, list] = {"frozen": [], "1link": [], "permission_rule": []}
    for tag, p in pred.items():
        f = FLEET_OUT / f"{tag}.json"
        e = {"N": p["N"], "k": p["k_permission_blocked"], "seed": p["seed"],
             "predicted_R": p["point"]["R"], "R_bootstrap": [p["bootstrap"]["R"]["lo"],
                                                             p["bootstrap"]["R"]["hi"]],
             "R_jeffreys": [p["jeffreys"]["R"]["lo"], p["jeffreys"]["R"]["hi"]],
             "predicted_W": p["point"]["W"], "P_board": p["point"]["P_board"],
             "P_1link": p["point"]["P_1link"], "predicted_board": p["point"]["predicted_board"]}
        if f.exists():
            o = json.loads(f.read_text())
            e["observed"] = o
            if not o["stopped"]:
                e["point_rule_correct"] = (None if p["point"]["predicted_board"] is None
                                           else p["point"]["predicted_board"] == o["board"])
                pairs["frozen"].append((p["point"]["P_board"], o["board"]))
                pairs["1link"].append((p["point"]["P_1link"], o["board"]))
                pairs["permission_rule"].append((1.0 if p["permission_rule_predicts_board"] else 0.0,
                                                 o["board"]))
        res["fleets"][tag] = e
    c4 = {}
    if C4_OUT.exists() and C4_PRED.exists():
        c4o = json.loads(C4_OUT.read_text())["fleets"]
        c4p = json.loads(C4_PRED.read_text())["fleets"]
        for tag, o in c4o.items():
            if "board" not in o or tag not in c4p or o.get("stopped"):
                continue
            p = c4p[tag]
            p1 = from_wr(p["W"], p["R"])["P_1link"]
            c4[tag] = {"N": p["N"], "prompt": p["prompt"], "predicted_R": p["R"],
                       "P_board": p["P_board"], "P_1link": p1, "board": o["board"],
                       "n_board_jobs": o["n_board_jobs"]}
    res["c4_fleets"] = c4
    all_pairs = {k: list(v) for k, v in pairs.items()}
    for d in c4.values():
        all_pairs["frozen"].append((d["P_board"], d["board"]))
        all_pairs["1link"].append((d["P_1link"], d["board"]))
        all_pairs["permission_rule"].append((1.0 if d["prompt"] == PERMISSION else 0.0,
                                             d["board"]))
    res["calibration"] = {
        "threshold_fleets": {k: {"n": len(v), "brier": brier(v),
                                 "sum_predicted": sum(x for x, _ in v),
                                 "boards": sum(y for _, y in v)} for k, v in pairs.items()},
        "with_c4": {k: {"n": len(v), "brier": brier(v), "sum_predicted": sum(x for x, _ in v),
                        "boards": sum(y for _, y in v)} for k, v in all_pairs.items()}}
    EXPLORE.mkdir(parents=True, exist_ok=True)
    OUTCOMES.write_text(json.dumps(res, indent=1))
    write_doc_tables(res)
    for k, v in res["calibration"].items():
        log(f"calibration {k}: " + "; ".join(
            f"{name} n {d['n']} Brier {'n/a' if d['brier'] is None else round(d['brier'], 3)} "
            f"expected boards {d['sum_predicted']:.2f} observed {d['boards']}"
            for name, d in v.items()))
    return res


# ---- doc tables (generated; the prose of the doc is written by hand) ---------------------
def _between(text: str, name: str, body: str) -> str:
    a, b = f"<!-- {name}:begin -->", f"<!-- {name}:end -->"
    if a not in text:
        return text
    pre, rest = text.split(a, 1)
    _, post = rest.split(b, 1)
    return f"{pre}{a}\n{body}\n{b}{post}"


def fmt_iv(d: dict, key: str, nd: int = 2) -> str:
    return f"[{d[key]['lo']:.{nd}f}, {d[key]['hi']:.{nd}f}]"


def write_doc_tables(res: dict | None = None) -> None:
    if not DOC.exists():
        return
    text = DOC.read_text()
    c4i = EXPLORE / "c4_intervals.json"
    if c4i.exists():
        d = json.loads(c4i.read_text())
        L = ["| C4 fleet | inputs p_w, v, p_j | W | R | R bootstrap 95% | R Jeffreys 95% | "
             "P(R > 1), Jeffreys | P_board | P_board Jeffreys 95% | P_1link |",
             "|---|---|---|---|---|---|---|---|---|---|"]
        for t, f in d["fleets"].items():
            i, pt = f["inputs"], f["point"]
            L.append(f"| {t} | {i['p_w'][0]}/{i['p_w'][1]}, {i['v'][0]}/{i['v'][1]}, "
                     f"{i['p_j'][0]}/{i['p_j'][1]} | {pt['W']:.2f} | {pt['R']:.2f} | "
                     f"{fmt_iv(f['bootstrap'], 'R')} | {fmt_iv(f['jeffreys'], 'R')} | "
                     f"{f['jeffreys']['P_R_above_1']:.2f} | {pt['P_board']:.3f} | "
                     f"{fmt_iv(f['jeffreys'], 'P_board', 3)} | {pt['P_1link']:.3f} |")
        text = _between(text, "c4intervals", "\n".join(L))
    if INPUTS.exists():
        inp = json.loads(INPUTS.read_text())
        L = [f"Inputs ({inp['model']}, written {inp['written']}; p_w from the confirmatory C1 "
             "cell as it stood then, v and p_j from the new probes):", "",
             "| arm | p_w (NOTE, any kind) | store | v (saw the seed note) | p_j (then wrote a "
             "NOTE or answered) | read it |", "|---|---|---|---|---|---|"]
        for prompt in ARMS:
            c = inp["cells"][f"{prompt}-rule"]
            for bg in BGS:
                p = inp["probes"][f"{prompt}-rule|bg{bg}"]
                L.append(f"| {prompt} + rule | {c['k']}/{c['n']} | {bg} jobs | "
                         f"{p['v_seen']['k']}/{p['v_seen']['n']} | {p['p_j']['k']}/"
                         f"{p['p_j']['n']} | {p['read']['k']}/{p['read']['n']} |")
        text = _between(text, "inputs", "\n".join(L))
    if PRED.exists():
        pred = json.loads(PRED.read_text())["fleets"]
        L = ["| fleet | N | jobs with the permission sentence | W | R | R bootstrap 95% | "
             "R Jeffreys 95% | P(R > 1), Jeffreys | P_board (frozen) | P_board bootstrap 95% | "
             "P_1link | point rule | written |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for t, p in pred.items():
            pt, b, j = p["point"], p["bootstrap"], p["jeffreys"]
            kk = p["k_permission_blocked"]
            who = (f"every job ({p['N']})" if kk == "all" else f"{kk} of {p['n_blocked']} blocked")
            L.append(f"| {t} | {p['N']} | {who} | "
                     f"{pt['W']:.2f} | {pt['R']:.2f} | {fmt_iv(b, 'R')} | {fmt_iv(j, 'R')} | "
                     f"{j['P_R_above_1']:.2f} | {pt['P_board']:.3f} | "
                     f"{fmt_iv(b, 'P_board', 3)} | {pt['P_1link']:.3f} | "
                     f"{'board' if pt['predicted_board'] else 'no board'} | {p['written']} |")
        text = _between(text, "predictions", "\n".join(L))
    if res is not None:
        L = ["| fleet | predicted R | P_board (frozen) | P_1link | board | jobs in a board link "
             "(permission / plain) | notes (permission / plain) | point rule correct | cost |",
             "|---|---|---|---|---|---|---|---|---|"]
        for t, e in res["fleets"].items():
            o = e.get("observed")
            if not o:
                L.append(f"| {t} | {e['predicted_R']:.2f} | {e['P_board']:.3f} | "
                         f"{e['P_1link']:.3f} | not run | | | | |")
                continue
            L.append(f"| {t} | {e['predicted_R']:.2f} | {e['P_board']:.3f} | {e['P_1link']:.3f} | "
                     f"{'yes' if o['board'] else 'no'} | {o['n_board_jobs']} "
                     f"({o['board_jobs_permission']} / {o['board_jobs_plain']}) | "
                     f"{o['permission_wrote_note']} / {o['plain_wrote_note']} | "
                     f"{e.get('point_rule_correct')} | ${o['cost_usd_nominal']:.2f} |")
        L += ["", "C4 fleets with an outcome so far (frozen predictions, results/confirmatory/c4.json):",
              "", "| fleet | N | predicted R | P_board (frozen) | P_1link | board |",
              "|---|---|---|---|---|---|"]
        for t, d in res["c4_fleets"].items():
            L.append(f"| {t} | {d['N']} | {d['predicted_R']:.2f} | {d['P_board']:.3f} | "
                     f"{d['P_1link']:.3f} | {'yes' if d['board'] else 'no'} |")
        if not res["c4_fleets"]:
            L.append("| (none yet) | | | | | |")
        L += ["", "Calibration (Brier score, lower is better; expected boards = sum of predicted "
              "P(board)):", "", "| set | predictor | fleets | Brier | expected boards | "
              "observed boards |", "|---|---|---|---|---|---|"]
        for s, v in res["calibration"].items():
            for name, d in v.items():
                bs = "n/a" if d["brier"] is None else f"{d['brier']:.3f}"
                L.append(f"| {s} | {name} | {d['n']} | {bs} | {d['sum_predicted']:.2f} | "
                         f"{d['boards']} |")
        L += ["", f"Spend of this study (ledger of results/cache/threshold_llm.sqlite): "
              f"${res['ledger']['spend_usd']:.2f} over {res['ledger']['calls']} calls."]
        text = _between(text, "observed", "\n".join(L))
    DOC.write_text(text)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("intervals")
    for name in ("probes", "inputs"):
        s = sub.add_parser(name)
        s.add_argument("--model", default=MODEL)
    sub.add_parser("scan")
    s = sub.add_parser("predict")
    s.add_argument("--plan", required=True, help="N:k pairs, e.g. 64:4,256:16")
    s.add_argument("--seeds", type=int, default=len(FLEET_SEEDS))
    s = sub.add_parser("fleet")
    s.add_argument("--tag", required=True)
    s.add_argument("--force", action="store_true", help="rerun a finished fleet from the cache")
    sub.add_parser("eval")
    a = ap.parse_args(argv)
    {"intervals": cmd_intervals, "probes": cmd_probes, "inputs": cmd_inputs, "scan": cmd_scan,
     "predict": cmd_predict, "fleet": cmd_fleet, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
