"""EXPLORATORY fleets after the confirmatory set (not part of the pre-registered tests).

    caffeinate -i uv run python experiments/scale_explore.py

Runs only when no confirmatory runner is running and every C4 fleet of the budget plan has
finished. Fleets (30 turns, 30% blocked, sample offset 1, same settings as C4):
- deepseek-v4-flash N = 1,024 plain + rule, second world seed 2901
- deepseek-v4-flash N = 1,024 plain + rule with one seeded request note, world seed 2901
- qwen3.5-397b N = 1,024 plain + rule with one seeded request note, world seed 2900
- gemma-4-31b-it N = 1,024 plain + rule, world seed 2900
The three models run side by side (deepseek-v4-flash 160 calls in flight, the others 48). Before each fleet the task spend
plus the fleet's projected cost must stay under the $780 cap, or the fleet is skipped. For
reference only, each fleet is shown with R = N v p_j from the same model's confirmatory cells and
probes (for gemma-4-31b-it the probes used a 256-job background). Outputs carry the label
EXPLORATORY: results/summary/scale_<tag>.json, results/exploratory/fleets.json,
docs/results_exploratory.md. A rerun skips finished fleets and replays the rest from the cache.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import scale_confirm as SC  # noqa: E402
import scale_confirm_eval as EV  # noqa: E402
import scale_confirm_run as R  # noqa: E402
import scale_predict as SP  # noqa: E402
from swarm.scale.analysis import episode_rows  # noqa: E402
from swarm.scale.runner import RunSpec, peak_rss_mb, run_world  # noqa: E402

LABEL = "EXPLORATORY"
EXPLORE = [
    {"model": "deepseek-v4-flash", "N": 1024, "prompt": "plain", "seed_note": "none",
     "seed": 2901},
    {"model": "deepseek-v4-flash", "N": 1024, "prompt": "plain", "seed_note": "request",
     "seed": 2901},
    {"model": "qwen3.5-397b", "N": 1024, "prompt": "plain", "seed_note": "request", "seed": 2900},
    {"model": "gemma-4-31b-it", "N": 1024, "prompt": "plain", "seed_note": "none", "seed": 2900},
]
OUT = ROOT / "results" / "exploratory" / "fleets.json"
MD = ROOT / "docs" / "results_exploratory.md"
log = R.log


def tag_of(f: dict) -> str:
    return (f"explore_{f['model']}_n{f['N']}_{f['prompt']}_rule"
            + ("_seedreq" if f["seed_note"] == "request" else "") + f"_s{f['seed']}")


def confirmatory_done() -> tuple[bool, str]:
    run = subprocess.run(["pgrep", "-f", "experiments/scale_confirm_run.py"],
                         capture_output=True, text=True).stdout.strip()
    if run:
        return False, f"confirmatory runner still running (pid {run})"
    left = [SP.fleet_tag(f) for f in SP.active_fleets() if not R.fleet_finished(SP.fleet_tag(f))]
    if left:
        return False, f"confirmatory fleets not finished: {left}"
    return True, "ok"


def reference_R(f: dict) -> dict:
    try:
        c1c3 = json.loads((EV.OUT / "c1c3.json").read_text())
        probes = json.loads((EV.OUT / "probes.json").read_text())
        arm = f"{f['prompt']}-rule"
        pw = c1c3["cells"][f["model"]][arm]["note_any"]["p"]
        pr = probes["probes"][f["model"]][arm]
        v, pj = pr["v_seen"]["p"], pr["p_j"]["p"] or 0.0
        n = f["N"]
        return {"p_w": pw, "v": v, "p_j": pj, "v_background": pr["n_background"],
                "W": SP.B * n * pw + (n * v * pj if f["seed_note"] == "request" else 0.0),
                "R": n * v * pj}
    except (OSError, KeyError, TypeError) as e:
        return {"error": str(e)}


def outcome(rec: dict, labels: dict, prefix: Path) -> dict:
    def ev():
        with open(f"{prefix}.events.jsonl") as fh:
            for ln in fh:
                if ln.strip():
                    yield json.loads(ln)
    rows = episode_rows([rec], {rec["episode_id"]: labels}, events=ev(), recompute=True)
    blocked = [r for r in rows if r["blocked"]]
    board = [r["job"] for r in rows if r["joined_peer"] or r["replied_peer"]]
    return {"label": LABEL, "n_jobs": len(rows), "board": bool(board), "n_board_jobs": len(board),
            "jobs_wrote_note": sum(bool(r["wrote_note"]) for r in rows),
            "jobs_wrote_note_addressed": sum(bool(r["wrote_note_addressed"]) for r in rows),
            "jobs_note_read_by_other": sum(bool(r["note_read_by_other"]) for r in rows),
            "jobs_read_peer_note": sum(bool(r["read_peer_note"]) for r in rows),
            "seed_read": sum(bool(r.get("seed_read")) for r in rows)
            if rec["world"]["seed_note"] else None,
            "seed_joined_or_responded": sum(bool(r.get("joined") or r.get("responded"))
                                            for r in rows) if rec["world"]["seed_note"] else None,
            "blocked_note_addressed": EV.rate([bool(r["wrote_note_addressed"]) for r in blocked]),
            "blocked_use_get": EV.rate([r["reads_others"] > 0 for r in blocked]),
            "all_success": EV.rate([bool(r["success"]) for r in rows]),
            "turns": rec["totals"]["calls"], "cost_usd_nominal": round(rec["totals"]["cost"], 4),
            "stopped": rec["stopped"], "wall_s": rec["wall_s"]}


def write_md(res: dict) -> None:
    L = ["# EXPLORATORY fleets (not pre-registered)", "",
         "Generated by `experiments/scale_explore.py`; do not edit by hand. These fleets ran after "
         "the confirmatory set and are not part of C1-C5. Board, notes and reads as in "
         "`docs/deviations.md` D3. R = N v p_j is shown for reference from the same model's "
         "confirmatory cells and probes.", "",
         "| fleet | jobs | board | jobs in a board link | jobs that wrote a note (addressed) | "
         "notes read by others | seed note read | read it, then wrote or answered | blocked "
         "NOTE_ADDRESSED | reference R | cost |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t, d in res["fleets"].items():
        if "n_jobs" not in d:
            L.append(f"| {t} | {d.get('status')} | | | | | | | | | |")
            continue
        rr = d.get("reference", {}).get("R")
        L.append(f"| {t} | {d['n_jobs']} | {'yes' if d['board'] else 'no'} | {d['n_board_jobs']} | "
                 f"{d['jobs_wrote_note']} ({d['jobs_wrote_note_addressed']}) | "
                 f"{d['jobs_note_read_by_other']} | {d['seed_read'] if d['seed_read'] is not None else 'n/a'} | "
                 f"{d['seed_joined_or_responded'] if d['seed_joined_or_responded'] is not None else 'n/a'} | "
                 f"{EV.md_rate(d['blocked_note_addressed'])} | "
                 f"{'n/a' if rr is None else round(rr, 2)} | ${d['cost_usd_nominal']:.2f} |")
    L += ["", f"Task spend (ledger) ${EV.ledger()['spend_usd']:.2f}.", ""]
    MD.write_text("\n".join(L) + "\n")


LOCK: asyncio.Lock | None = None


async def run_one(lane: R.Lane, f: dict, res: dict) -> None:
    from scale_fleet import report
    tag = tag_of(f)
    if R.fleet_finished(tag) and tag in res["fleets"]:
        log(f"=== EXPLORATORY {tag} already finished, skipped")
        return
    lg = EV.ledger()
    cost = f["N"] * SP.TURNS_PER_JOB * R.per_call_usd(f["model"], lg)
    if lg["spend_usd"] + cost > SC.BUDGET:
        res["fleets"][tag] = {"label": LABEL, "status": f"skipped for budget (spend "
                              f"${lg['spend_usd']:.2f} + projected ${cost:.2f})"}
        log(f"=== EXPLORATORY {tag} skipped for budget")
        return
    log(f"=== EXPLORATORY fleet {tag} start; spend ${lg['spend_usd']:.2f}, projected "
        f"${cost:.2f}; {R.physmem()}")
    spec = RunSpec(tag=tag, mode="fleet", n_agents=f["N"], p_blocked=0.3, prompt=f["prompt"],
                   rule=True, seed_note=f["seed_note"], n_background=200, p_derivative=1.0,
                   producers_in_fleet=True, steps=SP.STEPS, model=f"adaption/{f['model']}",
                   max_tokens=SC.MODELS[f["model"]][1], temperature=SC.TEMPERATURE,
                   seed=f["seed"], repeat=SP.REPEAT, schedule="rounds", history_window=12)
    prefix = SC.RAW / tag

    def progress(rr: dict) -> None:
        log(f"{tag} step {rr['step']:>2} active {rr['active']:>4} live {rr['live']:>4} "
            f"cached {rr['cached']:>4} err {rr['errors']:>3} {rr['wall_s']:.0f}s "
            f"rss {rr['peak_rss_mb']}MB ${rr['cost']:.3f}")
    t0 = time.time()
    rec, labels = await run_world(spec, lane.policy, sem=asyncio.Semaphore(lane.conc),
                                  out_prefix=prefix, progress=progress)

    def rep():
        out = report(rec, labels, {"label": LABEL, "elapsed_s": round(time.time() - t0, 1),
                                   "task_spend_usd": round(lane.llm.total_spent(), 4)},
                     None, prefix)
        out["label"] = LABEL
        out["peak_rss_mb_process"] = round(peak_rss_mb(), 1)
        (R.SUMMARY / f"scale_{tag}.json").write_text(json.dumps(out, indent=1))
        d = outcome(rec, labels, prefix)
        d["reference"] = reference_R(f)
        d["spec"] = f
        return d
    d = await asyncio.to_thread(rep)
    async with LOCK:
        res["fleets"][tag] = d
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, indent=1))
        write_md(res)
        log(f"=== EXPLORATORY {tag}: board {d['board']} ({d['n_board_jobs']} jobs), notes "
            f"{d['jobs_wrote_note']} (addressed {d['jobs_wrote_note_addressed']}), read by others "
            f"{d['jobs_note_read_by_other']}, seed read {d['seed_read']}, reference R "
            f"{d['reference'].get('R')}, ${d['cost_usd_nominal']:.2f}")
        await R.git_commit([str(OUT.relative_to(ROOT)), str(MD.relative_to(ROOT)),
                            f"results/summary/scale_{tag}.json"],
                           f"EXPLORATORY fleet {tag}: board outcome (not pre-registered)")


async def main_async() -> None:
    global LOCK
    LOCK = asyncio.Lock()
    R.GIT_LOCK, R.EVAL_LOCK = asyncio.Lock(), asyncio.Lock()
    R.patch_http()
    res = json.loads(OUT.read_text()) if OUT.exists() else {"label": LABEL, "fleets": {}}
    lanes = {m: R.Lane(m) for m in dict.fromkeys(f["model"] for f in EXPLORE)}
    for lane in lanes.values():
        lane.gate.set()

    async def lane_run(m: str) -> None:
        for f in EXPLORE:
            if f["model"] == m:
                try:
                    await run_one(lanes[m], f, res)
                except Exception as e:  # noqa: BLE001
                    log(f"=== EXPLORATORY ERROR {tag_of(f)}: {type(e).__name__}: {e}")
    await asyncio.gather(*(lane_run(m) for m in lanes))
    log(f"=== EXPLORATORY all done; http {dict(R.WIN.total)}; spend "
        f"${EV.ledger()['spend_usd']:.2f}")


if __name__ == "__main__":
    ok, why = confirmatory_done()
    if not ok:
        R.log(f"=== EXPLORATORY not started: {why}")
        sys.exit(1)
    R.log(f"=== EXPLORATORY start {R.physmem()}")
    asyncio.run(main_async())
