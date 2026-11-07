"""Numbers and pass rules of the confirmatory set (docs/prereg.md, docs/deviations.md).

    uv run python experiments/scale_confirm_eval.py --stage c1c3    # C1, C2, C3
    uv run python experiments/scale_confirm_eval.py --stage probes  # v and p_j for C4 (D6)
    uv run python experiments/scale_confirm_eval.py --stage c4      # C4 fleets vs predictions
    uv run python experiments/scale_confirm_eval.py --md            # only rewrite the markdown

Reads results/raw/scale/confirm_<model>.jsonl, probe_<model>.jsonl and the C4 fleet files, writes
results/confirmatory/<stage>.json, and rewrites docs/results_confirmatory.md from every stage file
present. Labels are joined only here (swarm.scale.analysis.episode_rows). Every rule is evaluated
as written at full precision; the rounded verdict (D5) is an extra, not pre-registered.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from scale_confirm import CACHE, CELLS, MODELS, RAW, cell_specs, out_path, probe_specs  # noqa: E402
from swarm.fleet.stats import wilson  # noqa: E402
from swarm.scale.analysis import episode_rows  # noqa: E402

OUT = ROOT / "results" / "confirmatory"
MD = ROOT / "docs" / "results_confirmatory.md"
PHRASINGS = ("cue_shared_fact", "cue_peers_have", "cue_team_culture", "cue_multiagent",
             "collab_line", "persist")
Z1 = 1.6448536269514722  # one-sided 95%


def rate(vals: list[bool]) -> dict:
    k, n = sum(vals), len(vals)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "p": k / n if n else None,
            "lo": None if n == 0 else lo, "hi": None if n == 0 else hi}


def fr(d: dict) -> str:
    if not d or not d["n"]:
        return "n/a"
    return f"{d['k']}/{d['n']} = {d['p']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"


def ledger() -> dict:
    if not CACHE.exists():
        return {"spend_usd": 0.0, "calls": 0}
    db = sqlite3.connect(f"file:{CACHE}?mode=ro", uri=True, timeout=30)
    rows = db.execute("SELECT model, COUNT(*), COALESCE(SUM(cost), 0) FROM spend "
                      "GROUP BY model").fetchall()
    db.close()
    return {"spend_usd": round(sum(r[2] for r in rows), 4), "calls": sum(r[1] for r in rows),
            "by_model": {r[0]: {"calls": r[1], "usd": round(r[2], 4)} for r in rows}}


def load_rows(path: Path, want: set[str]) -> tuple[list[dict], dict]:
    """Rows (one per episode) and per-arm totals, streamed one record at a time."""
    from swarm.scale.runner import labels_path
    labels = {}
    if labels_path(path).exists():
        with labels_path(path).open() as f:
            for ln in f:
                if ln.strip():
                    try:
                        r = json.loads(ln)
                    except json.JSONDecodeError:
                        continue
                    if r["episode_id"] in want:
                        labels[r["episode_id"]] = r
    rows, tot = [], defaultdict(lambda: defaultdict(float))
    seen = set()
    if not path.exists():
        return rows, tot
    with path.open() as f:
        for ln in f:
            if not ln.strip():
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:  # a line still being written by a running process
                continue
            eid = rec["episode_id"]
            if eid not in want or eid in seen or eid not in labels:
                continue
            seen.add(eid)
            rr = episode_rows([rec], labels)
            rows += rr
            arm = rr[0]["arm"] + ("" if rr[0]["note"] == "none" else f"|{rr[0]['note']}")
            t = rec["totals"]
            for k in ("cost", "calls", "live", "malformed", "errors"):
                tot[arm][k] += t[k]
    return rows, tot


# ---- C2 estimators -------------------------------------------------------------------------
def mover_stratified(strata: list[tuple[int, int, int, int]]) -> dict:
    """strata: (k1, n1, k0, n0) per model. Mantel-Haenszel weighted risk difference p1 - p0 and
    its one-sided 95% upper bound by MOVER from one-sided Wilson limits (z = 1.645)."""
    w = np.array([n1 * n0 / (n1 + n0) for _, n1, _, n0 in strata], float)
    w = w / w.sum()
    d, var = 0.0, 0.0
    for wi, (k1, n1, k0, n0) in zip(w, strata):
        p1, p0 = k1 / n1, k0 / n0
        _, u1 = wilson(k1, n1, z=Z1)
        l0, _ = wilson(k0, n0, z=Z1)
        d += wi * (p1 - p0)
        var += wi * wi * ((u1 - p1) ** 2 + (p0 - l0) ** 2)
    return {"diff": d, "upper": d + math.sqrt(var), "weights": w.round(4).tolist()}


def lpm_fe_hc2(groups: list[tuple[list[int], list[int]]]) -> dict:
    """Linear probability model y = a_model + d * phrasing, HC2 standard error."""
    ys, xs = [], []
    m = len(groups)
    for g, (y1, y0) in enumerate(groups):
        for t, yy in ((1, y1), (0, y0)):
            for y in yy:
                row = [0.0] * (m + 1)
                row[g] = 1.0
                row[m] = float(t)
                xs.append(row)
                ys.append(float(y))
    X, y = np.array(xs), np.array(ys)
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    e = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    omega = e ** 2 / np.clip(1 - h, 1e-12, None)
    cov = xtx_inv @ (X.T * omega) @ X @ xtx_inv
    se = float(math.sqrt(max(cov[m, m], 0.0)))
    return {"diff": float(beta[m]), "se_hc2": se, "upper": float(beta[m]) + Z1 * se}


# ---- stages ----------------------------------------------------------------------------------
def stage_c1c3() -> dict:
    cells: dict = {}
    raw: dict = {}
    for model in MODELS:
        specs = cell_specs(model)
        n_exp = MODELS[model][0]
        rows, tot = load_rows(out_path(model, False), {s.episode_id() for s in specs})
        by = defaultdict(list)
        for r in rows:
            by[r["arm"]].append(r)
        cells[model] = {}
        raw[model] = {}
        for prompt, rule in CELLS:
            arm = f"{prompt}-{'rule' if rule else 'norule'}"
            rs = by.get(arm, [])
            raw[model][arm] = rs
            t = tot.get(arm, {})
            cells[model][arm] = {
                "n_expected": n_exp, "complete": len(rs) == n_exp,
                "note_addressed": rate([bool(r["wrote_note_addressed"]) for r in rs]),
                "note_any": rate([bool(r["wrote_note"]) for r in rs]),
                "use_get": rate([r["reads_others"] > 0 for r in rs]),
                "p_use_code": rate([bool(r["uses_others_artifacts"]) for r in rs]),
                "report_blocked": rate([bool(r["report_blocked"]) for r in rs]),
                "cost_usd_nominal": round(t.get("cost", 0.0), 4),
                "calls": int(t.get("calls", 0)), "malformed": int(t.get("malformed", 0)),
                "errors": int(t.get("errors", 0)),
                "note_kinds": _kinds(rs),
            }
    complete = {m for m in MODELS if all(c["complete"] for c in cells[m].values())}
    res: dict = {"cells": cells, "models_complete": sorted(complete), "ledger": ledger()}

    # C1
    c1 = {}
    for m in MODELS:
        pa = cells[m]["plain-rule"]["note_addressed"]
        co = cells[m]["coordinate_ok-rule"]["note_addressed"]
        if not pa["n"] or not co["n"]:
            c1[m] = {"status": "no data"}
            continue
        ok = pa["hi"] <= 0.06 and co["lo"] >= 0.5
        ok_r = round(pa["hi"], 2) <= 0.06 and round(co["lo"], 2) >= 0.5
        c1[m] = {"plain_rule": pa, "coordinate_ok_rule": co, "pass": ok,
                 "pass_rounded_extra": ok_r, "complete": m in complete}
    c1_all = all(c1[m].get("pass") for m in MODELS) if complete == set(MODELS) else None
    res["C1"] = {"per_model": c1, "pass": c1_all}

    # C2
    models2 = [m for m in MODELS if m in complete] or [m for m in MODELS
                                                      if cells[m]["plain-rule"]["note_addressed"]["n"]]
    c2 = {}
    for ph in PHRASINGS:
        strata, groups = [], []
        for m in models2:
            a = cells[m][f"{ph}-rule"]["note_addressed"]
            b = cells[m]["plain-rule"]["note_addressed"]
            if a["n"] and b["n"]:
                strata.append((a["k"], a["n"], b["k"], b["n"]))
                groups.append(([int(r["wrote_note_addressed"]) for r in raw[m][f"{ph}-rule"]],
                               [int(r["wrote_note_addressed"]) for r in raw[m]["plain-rule"]]))
        if not strata:
            c2[ph] = {"status": "no data"}
            continue
        mv = mover_stratified(strata)
        lp = lpm_fe_hc2(groups)
        c2[ph] = {"models": models2, "mover": mv, "lpm_fe_hc2_check": lp,
                  "pass": mv["upper"] < 0.05,
                  "per_model": {m: cells[m][f"{ph}-rule"]["note_addressed"] for m in models2}}
    c2_all = all(c2[p].get("pass") for p in PHRASINGS) if complete == set(MODELS) else None
    res["C2"] = {"per_phrasing": c2, "pass": c2_all, "models_used": models2}

    # C3
    c3 = {}
    for m in MODELS:
        c = cells[m]["plain-norule"]
        if not c["use_get"]["n"]:
            c3[m] = {"status": "no data"}
            continue
        c3[m] = {"use_get": c["use_get"], "p_use_code": c["p_use_code"],
                 "note_addressed": c["note_addressed"],
                 "use_lo_above_0.2": c["use_get"]["lo"] > 0.2,
                 "note_hi_at_most_0.06": c["note_addressed"]["hi"] <= 0.06,
                 "note_hi_at_most_0.06_rounded_extra": round(c["note_addressed"]["hi"], 2) <= 0.06,
                 "complete": m in complete}
    have = [m for m in MODELS if "use_get" in c3[m]]
    n_use = sum(c3[m]["use_lo_above_0.2"] for m in have)
    note_ok = all(c3[m]["note_hi_at_most_0.06"] for m in have)
    note_ok_r = all(c3[m]["note_hi_at_most_0.06_rounded_extra"] for m in have)
    res["C3"] = {"per_model": c3, "n_models_use_lo_above_0.2": n_use,
                 "note_all_at_most_0.06": note_ok,
                 "note_all_at_most_0.06_rounded_extra": note_ok_r,
                 "pass": (n_use >= 4 and note_ok) if complete == set(MODELS) else None,
                 "pass_rounded_extra": (n_use >= 4 and note_ok_r)
                 if complete == set(MODELS) else None}
    return res


def _kinds(rs: list[dict]) -> dict:
    out: dict = defaultdict(int)
    for r in rs:
        for _, kind, _ in r["note_texts"]:
            out[kind] += 1
    return dict(out)


def stage_probes() -> dict:
    res: dict = {"probes": {}, "ledger": ledger()}
    for model in MODELS:
        specs = probe_specs(model)
        rows, tot = load_rows(out_path(model, True), {s.episode_id() for s in specs})
        k = MODELS[model][4]
        res["probes"][model] = {}
        for prompt in MODELS[model][3]:
            arm = f"{prompt}-rule"
            rs = [r for r in rows if r["arm"] == arm]
            seen = [r for r in rs if r["seed_seen"]]
            jr = [bool(r["joined"] or r.get("responded")) for r in seen]
            d = {"n_expected": 2 * k, "complete": len(rs) == 2 * k,
                 "n_background": MODELS[model][2],
                 "v_seen": rate([bool(r["seed_seen"]) for r in rs]),
                 "p_j": rate(jr),
                 "read": rate([bool(r["seed_read"]) for r in rs]),
                 "by_style": {}, "cost_usd_nominal": 0.0}
            for st in ("request", "memory"):
                ss = [r for r in rs if r["note"] == st]
                se = [r for r in ss if r["seed_seen"]]
                d["by_style"][st] = {
                    "seen": rate([bool(r["seed_seen"]) for r in ss]),
                    "read": rate([bool(r["seed_read"]) for r in ss]),
                    "joined_or_responded_if_seen": rate([bool(r["joined"] or r.get("responded"))
                                                         for r in se])}
                d["cost_usd_nominal"] += tot.get(f"{arm}|{st}", {}).get("cost", 0.0)
            d["cost_usd_nominal"] = round(d["cost_usd_nominal"], 4)
            res["probes"][model][arm] = d
    return res


def stage_c4() -> dict:
    from scale_predict import active_fleets, fleet_tag, load_plan
    pred_path = ROOT / "results" / "predictions" / "c4.json"
    pred = json.loads(pred_path.read_text())["fleets"] if pred_path.exists() else {}
    plan = load_plan()
    res: dict = {"fleets": {}, "ledger": ledger(), "budget_plan": plan}
    FLEETS = active_fleets()
    for f in FLEETS:
        tag = fleet_tag(f)
        prefix = RAW / tag
        rec_p = Path(f"{prefix}.record.json")
        if not rec_p.exists():
            res["fleets"][tag] = {"status": "not run"}
            continue
        rec = json.loads(rec_p.read_text())
        labels = json.loads(Path(f"{prefix}.labels.json").read_text())

        def ev():
            with open(f"{prefix}.events.jsonl") as fh:
                for ln in fh:
                    if ln.strip():
                        yield json.loads(ln)
        rows = episode_rows([rec], {rec["episode_id"]: labels}, events=ev(), recompute=True)
        board_jobs = [r["job"] for r in rows if r["joined_peer"] or r["replied_peer"]]
        blocked = [r for r in rows if r["blocked"]]
        p = pred.get(tag, {})
        d = {"n_jobs": len(rows), "board": bool(board_jobs), "n_board_jobs": len(board_jobs),
             "jobs_wrote_note": sum(bool(r["wrote_note"]) for r in rows),
             "jobs_wrote_note_addressed": sum(bool(r["wrote_note_addressed"]) for r in rows),
             "jobs_note_read_by_other": sum(bool(r["note_read_by_other"]) for r in rows),
             "jobs_read_peer_note": sum(bool(r["read_peer_note"]) for r in rows),
             "jobs_joined_peer": sum(bool(r["joined_peer"]) for r in rows),
             "jobs_replied_peer": sum(bool(r["replied_peer"]) for r in rows),
             "blocked_note_addressed": rate([bool(r["wrote_note_addressed"]) for r in blocked]),
             "blocked_note_any": rate([bool(r["wrote_note"]) for r in blocked]),
             "blocked_use_get": rate([r["reads_others"] > 0 for r in blocked]),
             "blocked_success": rate([bool(r["success"]) for r in blocked]),
             "all_success": rate([bool(r["success"]) for r in rows]),
             "seed_read": sum(bool(r.get("seed_read")) for r in rows)
             if rec["world"]["seed_note"] else None,
             "turns": rec["totals"]["calls"], "cost_usd_nominal": round(rec["totals"]["cost"], 4),
             "stopped": rec["stopped"], "wall_s": rec["wall_s"],
             "predicted_R": p.get("R"), "predicted_W": p.get("W"),
             "predicted_P_board": p.get("P_board"), "predicted_board": p.get("predicted_board")}
        if p.get("predicted_board") is not None and not rec["stopped"]:
            d["correct"] = d["predicted_board"] == d["board"]
        res["fleets"][tag] = d
    for t in plan["dropped"]:
        res["fleets"][t] = {"status": "dropped for budget (D9)"}
    done = [t for t, d in res["fleets"].items() if "correct" in d]
    res["n_fleets_evaluated"] = len(done)
    res["mispredicted"] = [t for t in done if not res["fleets"][t]["correct"]]
    res["pass"] = (not res["mispredicted"]) if len(done) == len(FLEETS) else None
    return res


# ---- printing and markdown ----------------------------------------------------------------
def verdict(x) -> str:
    return "PASS" if x is True else "FAIL" if x is False else "INCOMPLETE"


def print_c1c3(r: dict) -> list[str]:
    L = []
    L.append(f"models complete: {', '.join(r['models_complete']) or 'none'}")
    for m, d in r["C1"]["per_model"].items():
        if "pass" not in d:
            L.append(f"C1 {m}: no data")
            continue
        L.append(f"C1 {m}: {verdict(d['pass'])} plain+rule NOTE_ADDRESSED {fr(d['plain_rule'])} "
                 f"(upper <= 0.06?), coordinate_ok+rule {fr(d['coordinate_ok_rule'])} "
                 f"(lower >= 0.5?); rounded extra {verdict(d['pass_rounded_extra'])}"
                 f"{'' if d['complete'] else ' [incomplete]'}")
    L.append(f"C1 overall: {verdict(r['C1']['pass'])}")
    for ph, d in r["C2"]["per_phrasing"].items():
        if "pass" not in d:
            L.append(f"C2 {ph}: no data")
            continue
        L.append(f"C2 {ph}: {verdict(d['pass'])} pooled diff {d['mover']['diff']:+.4f}, "
                 f"one-sided 95% upper {d['mover']['upper']:.4f} (< 0.05?); LPM-FE check diff "
                 f"{d['lpm_fe_hc2_check']['diff']:+.4f} upper {d['lpm_fe_hc2_check']['upper']:.4f}")
    L.append(f"C2 overall ({len(r['C2']['models_used'])} models): {verdict(r['C2']['pass'])}")
    for m, d in r["C3"]["per_model"].items():
        if "use_get" not in d:
            L.append(f"C3 {m}: no data")
            continue
        L.append(f"C3 {m}: USE(get) {fr(d['use_get'])} lower > 0.2: {d['use_lo_above_0.2']}; "
                 f"NOTE_ADDRESSED {fr(d['note_addressed'])} upper <= 0.06: "
                 f"{d['note_hi_at_most_0.06']} (rounded extra "
                 f"{d['note_hi_at_most_0.06_rounded_extra']}); code p_use {fr(d['p_use_code'])}")
    c3 = r["C3"]
    L.append(f"C3 overall: {verdict(c3['pass'])} (models with USE lower > 0.2: "
             f"{c3['n_models_use_lo_above_0.2']} of 6 needed 4; NOTE upper <= 0.06 in every model: "
             f"{c3['note_all_at_most_0.06']}); rounded extra {verdict(c3['pass_rounded_extra'])}")
    lg = r["ledger"]
    L.append(f"task spend (ledger) ${lg['spend_usd']:.2f}, {lg['calls']} live calls")
    return L


def md_rate(d: dict) -> str:
    if not d or not d["n"]:
        return "n/a"
    return f"{d['k']}/{d['n']}, {d['p']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"


def write_md() -> None:
    L = ["# Confirmatory results (C1-C4)", "",
         "Generated by `experiments/scale_confirm_eval.py` from `results/confirmatory/*.json`; do "
         "not edit by hand. Pre-registration `docs/prereg.md` (frozen at d8900cb), deviations "
         "`docs/deviations.md`. Rates are k/n, share, Wilson 95% interval. NOTE_ADDRESSED and USE "
         "as in the prereg (USE = a get of another job's artifact, D2); `p_use (code)` counts "
         "compute, submit or copy. Costs are nominal per cell (USD, Adaption prices). Rounded "
         "verdicts (D5) are extras, not pre-registered.", ""]
    p = OUT / "c1c3.json"
    if p.exists():
        r = json.loads(p.read_text())
        L += ["## Stage 1: single-agent cells (C1, C2, C3)", "",
              f"Models complete: {', '.join(r['models_complete']) or 'none'}. Task spend after "
              f"this stage (ledger): ${r['ledger']['spend_usd']:.2f}.", "",
              "| model | cell | n | NOTE_ADDRESSED | NOTE (any kind) | USE (get) | p_use (code) "
              "| cost |", "|---|---|---|---|---|---|---|---|"]
        for m, cs in r["cells"].items():
            for arm, c in cs.items():
                if not c["note_addressed"]["n"]:
                    continue
                L.append(f"| {m} | {arm} | {c['note_addressed']['n']}"
                         f"{'' if c['complete'] else ' (incomplete)'} | "
                         f"{md_rate(c['note_addressed'])} | {md_rate(c['note_any'])} | "
                         f"{md_rate(c['use_get'])} | {md_rate(c['p_use_code'])} | "
                         f"${c['cost_usd_nominal']:.3f} |")
        L += ["", "Pass rules, evaluated mechanically:", "", "```"] + print_c1c3(r) + ["```", ""]
    p = OUT / "probes.json"
    if p.exists():
        r = json.loads(p.read_text())
        L += ["## C4 probe episodes (v and p_j, D6)", "",
              "| model | arm | store background | n | v (saw the seed note) | p_j (read it, then "
              "wrote a note or answered, of those who saw it) | read | cost |",
              "|---|---|---|---|---|---|---|---|"]
        for m, arms in r["probes"].items():
            for arm, d in arms.items():
                if not d["v_seen"]["n"]:
                    continue
                L.append(f"| {m} | {arm} | {d['n_background']} | {d['v_seen']['n']}"
                         f"{'' if d['complete'] else ' (incomplete)'} | {md_rate(d['v_seen'])} | "
                         f"{md_rate(d['p_j'])} | {md_rate(d['read'])} | "
                         f"${d['cost_usd_nominal']:.2f} |")
        L += ["", f"Task spend after the probes (ledger): ${r['ledger']['spend_usd']:.2f}.", ""]
    p = ROOT / "results" / "predictions" / "c4.json"
    if p.exists():
        r = json.loads(p.read_text())
        L += ["## C4 predictions (committed before the first fleet)", "",
              "| fleet | N | p_w | v | p_j | W = b N p_w | R = N v p_j | P_board | predicted |",
              "|---|---|---|---|---|---|---|---|---|"]
        for t, d in r["fleets"].items():
            pj = "n/a" if d["p_j"] is None else f"{d['p_j']:.4f}"
            L.append(f"| {t} | {d['N']} | {d['p_w']:.4f} | {d['v']:.4f} | {pj} | "
                     f"{d['W']:.2f} | {d['R']:.2f} | {d['P_board']:.3f} | "
                     f"{'board' if d['predicted_board'] else 'no board'} |")
        L.append("")
    p = OUT / "c4.json"
    if p.exists():
        r = json.loads(p.read_text())
        L += ["## C4 fleets", "", "| fleet | jobs | board | jobs in a board link | jobs that "
              "wrote a note (addressed) | notes read by others | blocked NOTE_ADDRESSED | blocked "
              "USE (get) | predicted R | predicted | correct | cost |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for t, d in r["fleets"].items():
            if "n_jobs" not in d:
                L.append(f"| {t} | {d.get('status', 'not run')} | | | | | | | | | | |")
                continue
            L.append(f"| {t} | {d['n_jobs']} | {'yes' if d['board'] else 'no'} | "
                     f"{d['n_board_jobs']} | {d['jobs_wrote_note']} ({d['jobs_wrote_note_addressed']})"
                     f" | {d['jobs_note_read_by_other']} | {md_rate(d['blocked_note_addressed'])} | "
                     f"{md_rate(d['blocked_use_get'])} | "
                     f"{'n/a' if d['predicted_R'] is None else round(d['predicted_R'], 2)} | "
                     f"{'n/a' if d['predicted_board'] is None else ('board' if d['predicted_board'] else 'no board')} | "
                     f"{d.get('correct', 'n/a')} | ${d['cost_usd_nominal']:.2f} |")
        L += ["", f"C4: {verdict(r['pass'])}; fleets evaluated {r['n_fleets_evaluated']}, "
              f"mispredicted: {', '.join(r['mispredicted']) or 'none'}. Task spend (ledger) "
              f"${r['ledger']['spend_usd']:.2f}.", ""]
    MD.write_text("\n".join(L) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=["c1c3", "probes", "c4"])
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "c1c3":
        r = stage_c1c3()
        (OUT / "c1c3.json").write_text(json.dumps(r, indent=1))
        print("\n".join(print_c1c3(r)))
    elif a.stage == "probes":
        r = stage_probes()
        (OUT / "probes.json").write_text(json.dumps(r, indent=1))
        for m, arms in r["probes"].items():
            for arm, d in arms.items():
                print(f"{m} {arm} bg{d['n_background']}: v {fr(d['v_seen'])}, p_j {fr(d['p_j'])}, "
                      f"read {fr(d['read'])}, ${d['cost_usd_nominal']:.2f}"
                      f"{'' if d['complete'] else ' [incomplete]'}")
        print(f"task spend (ledger) ${r['ledger']['spend_usd']:.2f}")
    elif a.stage == "c4":
        r = stage_c4()
        (OUT / "c4.json").write_text(json.dumps(r, indent=1))
        for t, d in r["fleets"].items():
            if "n_jobs" not in d:
                print(f"{t}: {d.get('status', 'not run')}")
                continue
            c = d.get("correct")
            print(f"{t}: board {d['board']} ({d['n_board_jobs']} jobs), predicted R "
                  f"{d['predicted_R']}, predicted board {d['predicted_board']} -> "
                  f"{'PASS' if c else 'FAIL' if c is False else 'n/a'}; notes {d['jobs_wrote_note']}"
                  f" (addressed {d['jobs_wrote_note_addressed']}), read by others "
                  f"{d['jobs_note_read_by_other']}, ${d['cost_usd_nominal']:.2f}")
        print(f"C4: {verdict(r['pass'])}; mispredicted {r['mispredicted']}; task spend "
              f"${r['ledger']['spend_usd']:.2f}")
    write_md()


if __name__ == "__main__":
    main()
