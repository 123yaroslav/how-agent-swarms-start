"""C4 predictions: the nucleation model fitted on single agents, for every C4 fleet
(docs/prereg.md C4, docs/deviations.md D6). Run after the C1-C3 cells and the probe episodes,
commit results/predictions/c4.json before the first fleet.

    uv run python experiments/scale_predict.py               # write results/predictions/c4.json
    uv run python experiments/scale_predict.py --model glm-5.2  # one model's fleets only (D8)
    uv run python experiments/scale_predict.py --fleet-cmd 0  # print the command of fleet 0
    uv run python experiments/scale_predict.py --budget-check 0  # exit 1 if fleets 0.. would
                                                             # take the task spend over the cap

Inputs: results/confirmatory/c1c3.json (p_w = NOTE rate, any kind, of the no-note cell of the
fleet's arm) and results/confirmatory/probes.json (v = share of probe episodes that saw the seed
note, p_j = share of those that read it and then wrote a NOTE or answered it; store background =
fleet N). b = 0.3, the fleets' blocked share.

    W = b N p_w   (+ N v p_j for the fleet with one seeded request note)
    R = N v p_j
    P_board = 1 - exp(-W q),  q = max(0, 1 - 1/R)
    predicted board: R > 1; no board: R < 1 (point estimates)

R_lo and R_hi use the Wilson 95% limits of v and p_j (shown, not used by the pass rule).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

B = 0.3
STEPS = 30
FLEET_SEED = 2900
REPEAT = 1
# in run order (C4 row of the prereg; order from the run plan)
FLEETS = [
    {"model": "deepseek-v4-flash", "N": 1024, "prompt": "plain", "seed_note": "none"},
    {"model": "deepseek-v4-flash", "N": 1024, "prompt": "coordinate_ok", "seed_note": "none"},
    {"model": "deepseek-v4-flash", "N": 1024, "prompt": "plain", "seed_note": "request"},
    {"model": "qwen3.5-397b", "N": 1024, "prompt": "plain", "seed_note": "none"},
    {"model": "glm-5.2", "N": 1024, "prompt": "plain", "seed_note": "none"},
    {"model": "gemma-4-31b-it", "N": 256, "prompt": "plain", "seed_note": "none"},
    {"model": "gemma-4-31b-it", "N": 256, "prompt": "coordinate_ok", "seed_note": "none"},
    {"model": "gemma-4-26b-a4b-it", "N": 256, "prompt": "plain", "seed_note": "none"},
    {"model": "gemma-4-26b-a4b-it", "N": 256, "prompt": "coordinate_ok", "seed_note": "none"},
    {"model": "qwen3.8-27b", "N": 256, "prompt": "plain", "seed_note": "none"},
    {"model": "qwen3.8-27b", "N": 256, "prompt": "coordinate_ok", "seed_note": "none"},
]
# expected turns per fleet job for the budget projection: blocked jobs use all turns, the others
# finish early (smoke fleet: 4,601 turns for 256 jobs at 40 turns; scaled to 30 turns)
TURNS_PER_JOB = 15.0


PLAN = ROOT / "results" / "confirmatory" / "fleet_plan.json"
# budget drop order (D9): first these two fleets, then glm-5.2's fleet down to N = 512
DROP_FIRST = ["c4_gemma-4-26b-a4b-it_n256_coordinate_ok_rule", "c4_qwen3.8-27b_n256_coordinate_ok_rule"]
SHRINK = ("c4_glm-5.2_n1024_plain_rule", 512)


def load_plan() -> dict:
    if PLAN.exists():
        return json.loads(PLAN.read_text())
    return {"dropped": [], "n_override": {}, "log": []}


def save_plan(plan: dict) -> None:
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(json.dumps(plan, indent=1))


def active_fleets() -> list[dict]:
    """FLEETS after the budget plan (D9): dropped fleets removed, shrunk fleets at their new N."""
    plan = load_plan()
    out = []
    for f in FLEETS:
        t = fleet_tag(f)
        if t in plan["dropped"]:
            continue
        g = dict(f)
        if t in plan["n_override"]:
            g["N"] = plan["n_override"][t]
            g["shrunk_from"] = t
        out.append(g)
    return out


def fleet_tag(f: dict) -> str:
    return (f"c4_{f['model']}_n{f['N']}_{f['prompt']}_rule"
            + ("_seedreq" if f["seed_note"] == "request" else ""))


def fleet_cmd(f: dict) -> list[str]:
    from scale_confirm import BUDGET, CACHE, MODELS
    cmd = ["uv", "run", "python", "experiments/scale_fleet.py", "--tag", fleet_tag(f),
           "--n", str(f["N"]), "--steps", str(STEPS), "--prompt", f["prompt"], "--rule", "rule",
           "--seed-note", f["seed_note"], "--seed", str(FLEET_SEED), "--repeat", str(REPEAT),
           "--model", f"adaption/{f['model']}", "--max-tokens", str(MODELS[f["model"]][1]),
           "--temperature", "0.8", "--concurrency", "64", "--budget", str(BUDGET),
           "--cache", str(CACHE.relative_to(ROOT))]
    return cmd


def predict(models: list[str] | None = None) -> dict:
    c1c3 = json.loads((ROOT / "results" / "confirmatory" / "c1c3.json").read_text())
    probes = json.loads((ROOT / "results" / "confirmatory" / "probes.json").read_text())
    out = {}
    for f in active_fleets():
        if models is not None and f["model"] not in models:
            continue
        m, n, arm = f["model"], f["N"], f"{f['prompt']}-rule"
        cell = c1c3["cells"][m][arm]
        pr = probes["probes"][m][arm]
        if not cell["complete"] or not pr["complete"]:
            raise RuntimeError(f"incomplete inputs for {fleet_tag(f)}: cell complete "
                             f"{cell['complete']}, probes complete {pr['complete']}")
        if pr["n_background"] != n and "shrunk_from" not in f:
            raise RuntimeError(f"probe background {pr['n_background']} != N {n} for {m}")
        # a fleet shrunk for budget (D9) keeps the v measured at the original N
        pw = cell["note_any"]["p"]
        v, v_lo, v_hi = pr["v_seen"]["p"], pr["v_seen"]["lo"], pr["v_seen"]["hi"]
        pj = pr["p_j"]["p"]
        pj_lo = pr["p_j"]["lo"] if pr["p_j"]["n"] else 0.0
        pj_hi = pr["p_j"]["hi"] if pr["p_j"]["n"] else 1.0
        r = n * v * (pj or 0.0)  # p_j is undefined only when no probe saw the note (v = 0)
        w = B * n * pw + (n * v * (pj or 0.0) if f["seed_note"] == "request" else 0.0)
        q = max(0.0, 1.0 - 1.0 / r) if r > 0 else 0.0
        out[fleet_tag(f)] = {
            **f, "b": B, "p_w": pw, "p_w_rate": cell["note_any"],
            "p_w_addressed_rate": cell["note_addressed"],
            "v": v, "v_rate": pr["v_seen"], "v_background": pr["n_background"],
            "p_j": pj, "p_j_rate": pr["p_j"],
            "W": w, "R": r, "R_lo": n * v_lo * pj_lo, "R_hi": n * v_hi * pj_hi,
            "q": q, "P_board": 1.0 - math.exp(-w * q),
            "predicted_board": True if r > 1 else False if r < 1 else None}
    return out


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "?"


def budget_check(start: int) -> int:
    from scale_confirm import BUDGET
    from scale_confirm_eval import ledger
    lg = ledger()
    proj = 0.0
    lines = []
    for f in FLEETS[start:]:
        bm = lg.get("by_model", {}).get(f"adaption/{f['model']}")
        per_call = bm["usd"] / bm["calls"] if bm and bm["calls"] else 0.01
        c = f["N"] * TURNS_PER_JOB * per_call
        proj += c
        lines.append(f"  {fleet_tag(f)}: {f['N']} x {TURNS_PER_JOB:.0f} turns x "
                     f"${per_call:.5f} = ${c:.2f}")
    print(f"spend ${lg['spend_usd']:.2f} + projected remaining fleets ${proj:.2f} = "
          f"${lg['spend_usd'] + proj:.2f} (cap ${BUDGET:.0f})")
    print("\n".join(lines))
    return 0 if lg["spend_usd"] + proj <= BUDGET else 1


PRED = ROOT / "results" / "predictions" / "c4.json"


def write_predictions(models: list[str] | None = None) -> dict:
    """Add the predictions of `models` (all when None) to results/predictions/c4.json for the
    fleets that have none yet. Entries already there are frozen and never rewritten (D8:
    predictions are frozen per model before that model's fleets). A fleet whose output exists
    gets no prediction."""
    raw = ROOT / "results" / "raw" / "scale"
    old = json.loads(PRED.read_text()) if PRED.exists() else {"fleets": {}, "written": {}}
    todo = [fleet_tag(f) for f in active_fleets() if (models is None or f["model"] in models)
            and fleet_tag(f) not in old["fleets"]]
    ran = [t for t in todo if Path(f"{raw / t}.rounds.jsonl").exists()]
    if ran:
        raise RuntimeError(f"C4 fleet output already exists ({ran}); predictions must come first")
    preds = {t: d for t, d in predict(models).items() if t in todo}
    ts = time.strftime("%Y-%m-%d %H:%M:%S %z")
    old["fleets"].update(preds)
    old["written"].update({t: ts for t in preds})
    old["formula"] = __doc__
    old.setdefault("inputs_at_commit", {}).update({t: git_head() for t in preds})
    PRED.parent.mkdir(parents=True, exist_ok=True)
    PRED.write_text(json.dumps(old, indent=1))
    for t, d in preds.items():
        pj = "n/a" if d["p_j"] is None else f"{d['p_j']:.4f}"
        print(f"{t}: p_w {d['p_w']:.4f} v {d['v']:.4f} p_j {pj} W {d['W']:.2f} R {d['R']:.2f} "
              f"[{d['R_lo']:.2f}, {d['R_hi']:.2f}] P_board {d['P_board']:.3f} -> "
              f"{'board' if d['predicted_board'] else 'no board'}", flush=True)
    print(f"wrote {PRED.relative_to(ROOT)}", flush=True)
    return preds


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fleet-cmd", type=int, default=None)
    ap.add_argument("--budget-check", type=int, default=None)
    ap.add_argument("--model", default=None, help="only this model's fleets (D8)")
    a = ap.parse_args(argv)
    if a.fleet_cmd is not None:
        print(" ".join(fleet_cmd(FLEETS[a.fleet_cmd])))
        return
    if a.budget_check is not None:
        sys.exit(budget_check(a.budget_check))
    write_predictions([a.model] if a.model else None)


if __name__ == "__main__":
    main()
