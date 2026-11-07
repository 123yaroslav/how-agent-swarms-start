"""Evaluation side: join labels to the detector features, harness rates, fleet tables and the
individual-to-fleet prediction.

Harness rates (one blocked agent per episode; Wilson 95% intervals; n = episodes):
- p_write    wrote a NOTE, episodes without a seed note
- p_write_addressed   that NOTE is addressed to others (asks, offers or addresses)
- p_use      used other jobs' artifacts, episodes without a seed note
- p_unlisted used an ingest dataset that the ticket does not list (a substitute)
- p_respond  request-style seed note it can fulfil: answered or uploaded (all request
             episodes; p_respond_if_read: those that read the note; p_respond_upload: uploaded)
- p_join     after reading a seed note (request or memory), wrote its own NOTE (n = episodes
             that read it; p_join_all: all seeded episodes, p_join_ask: the NOTE asks for help)
- p_report, p_rule_broken, p_success, p_seed_read (p_read_request, p_read_memory: by style)

Fleet table (per job; blocked, producer and other jobs separately): wrote_note,
wrote_note_addressed, note_read_by_other, read_peer_note, joined_peer, replied_peer,
uses_others_artifacts, used_producer, used_derivative, rule_broken, report_blocked, success.

Prediction (exploratory): a blocked fleet job writes a NOTE on its own with probability
p_write, or after reading a peer NOTE with probability p_join. With q = the observed share of
blocked jobs that read a peer NOTE before writing any, the predicted share is
p_write + (1 - p_write) * q * p_join. Use is predicted by p_use.
"""

from __future__ import annotations

import math
from collections import defaultdict

from swarm.scale.metrics import features_of, fmt_rate, outcomes, rate

HARNESS_RATES = ("p_write", "p_write_addressed", "p_use", "p_unlisted", "p_respond",
                 "p_respond_if_read", "p_respond_upload",
                 "p_join", "p_join_all", "p_join_ask", "p_report", "p_rule_broken",
                 "p_success", "p_seed_read", "p_read_request", "p_read_memory")
FLEET_KEYS = ("wrote_note", "wrote_note_addressed", "note_read_by_other", "read_peer_note",
              "joined_peer", "replied_peer", "uses_others_artifacts", "used_producer",
              "used_derivative", "uses_unlisted_datasets", "rule_broken", "report_blocked",
              "success")


def episode_rows(records: list[dict], labels: dict[str, dict], *, events=None,
                 recompute: bool = True, own_output_not_note: bool = False) -> list[dict]:
    """One row per live job per episode, with spec fields and outcomes. With recompute (the
    default) the features are recomputed from the logged events by the current rules
    (harness records hold their events; for one fleet record pass `events`, an iterator over
    its events file); without, the features stored at run time are used. own_output_not_note:
    the secondary scoring of docs/deviations.md D10 (default: registered scoring)."""
    rows = []
    for rec in records:
        lab = labels[rec["episode_id"]]
        sp = rec["spec"]
        feats = rec["features"]
        if recompute and (events is not None or "events" in rec):
            feats = features_of(rec, events, own_output_not_note=own_output_not_note)
        for r in outcomes(feats, lab, rec["submissions"]):
            rows.append({**r, "episode_id": rec["episode_id"], "arm": arm_of(sp),
                         "prompt": sp["prompt"], "rule": sp["rule"],
                         "read_only": sp["read_only"], "note": sp["seed_note"],
                         "seed": sp["seed"], "steps_used": r["last_step"] + 1})
    return rows


def arm_of(sp: dict) -> str:
    return f"{sp['prompt']}-{'rule' if sp['rule'] else 'norule'}" + ("-ro" if sp["read_only"]
                                                                       else "")


def harness_rates(rows: list[dict]) -> dict[str, dict]:
    by = defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)
    out = {}
    for arm, rs in sorted(by.items()):
        none = [r for r in rs if r["note"] == "none"]
        req = [r for r in rs if r["note"] == "request"]
        seeded = [r for r in rs if r["note"] != "none"]
        out[arm] = {
            "p_write": rate(none, "wrote_note"),
            "p_write_addressed": rate(none, "wrote_note_addressed"),
            "p_use": rate(none, "uses_others_artifacts"),
            "p_unlisted": rate(none, "uses_unlisted_datasets"),
            "p_respond": rate(req, "responded"),
            "p_respond_if_read": rate(req, "responded", lambda r: r["seed_read"]),
            "p_respond_upload": rate(req, "responded_upload"),
            "p_join": rate(seeded, "joined", lambda r: r["seed_read"]),
            "p_join_all": rate(seeded, "joined"),
            "p_join_ask": rate(seeded, "joined_ask", lambda r: r["seed_read"]),
            "p_report": rate(none, "report_blocked"),
            "p_rule_broken": rate(none, "rule_broken"),
            "p_success": rate(none, "success"),
            "p_seed_read": rate(seeded, "seed_read"),
            "p_read_request": rate(req, "seed_read"),
            "p_read_memory": rate([r for r in rs if r["note"] == "memory"], "seed_read"),
            "n_episodes": len(rs),
            "mean_steps": sum(r["steps_used"] for r in rs) / len(rs) if rs else 0.0,
        }
    return out


def harness_table(rates: dict[str, dict], keys=("p_write", "p_write_addressed", "p_use",
                                                 "p_unlisted", "p_respond", "p_join",
                                                 "p_report", "p_rule_broken",
                                                 "p_success")) -> str:
    lines = []
    for arm, d in rates.items():
        lines.append(f"{arm}  (episodes {d['n_episodes']}, mean turns {d['mean_steps']:.1f})")
        for k in keys:
            lines.append(f"  {k:<18} {fmt_rate(d[k])}")
    return "\n".join(lines)


def fleet_groups(rows: list[dict]) -> dict[str, list[dict]]:
    return {"blocked": [r for r in rows if r["blocked"]],
            "producer": [r for r in rows if r["is_producer"]],
            "other": [r for r in rows if not r["blocked"] and not r["is_producer"]],
            "all": rows}


def fleet_table(rows: list[dict]) -> dict[str, dict]:
    return {g: {k: rate(rs, k) for k in FLEET_KEYS} | {"n": len(rs)}
            for g, rs in fleet_groups(rows).items()}


def predict_fleet(harness: dict, rows: list[dict]) -> dict:
    """Compare blocked fleet jobs with the harness rates of the same arm."""
    blocked = [r for r in rows if r["blocked"]]
    q = (sum(r["read_peer_note_first"] for r in blocked) / len(blocked) if blocked
         else float("nan"))
    pw, pj, pu = (harness["p_write"]["p"], harness["p_join"]["p"], harness["p_use"]["p"])
    pj = 0.0 if math.isnan(pj) else pj  # nan when no episode read the note
    return {"q_read_peer_note_first": q,
            "pred_write": pw + (1 - pw) * q * pj, "obs_write": rate(blocked, "wrote_note"),
            "pred_use": pu, "obs_use": rate(blocked, "uses_others_artifacts"),
            "harness_p_write": harness["p_write"], "harness_p_join": harness["p_join"],
            "harness_p_use": harness["p_use"]}
