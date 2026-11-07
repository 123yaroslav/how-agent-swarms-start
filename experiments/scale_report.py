"""Print every number of the swarm.scale README smoke section from the summary files (no calls).

    uv run python experiments/scale_report.py

Reads results/summary/scale_harness_smoke1.json, scale_harness_cues1.json,
scale_fleet256_plain_rule.json,
scale_throughput.json, scale_throughput_ssl.json and the scripted memory runs
scale_dry_*.json, and prints markdown tables.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "results" / "summary"


def load(name: str) -> dict | None:
    p = S / name
    return json.loads(p.read_text()) if p.exists() else None


def r(d: dict) -> str:
    if not d or not d.get("n"):
        return "n/a"
    return f"{d['p']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}] ({d['k']}/{d['n']})"


def main() -> None:
    print("### Memory (scripted agents, 40 turns)\n")
    print("| run | N | calls | peak RSS (MB) |\n|---|---|---|---|")
    for tag, what in (("dry_scripted_n256", "scripted mix"), ("dry_scripted_n1024", "scripted mix"),
                      ("dry_browser_n256_pad", "worst case: browse every turn, 1.8k-char replies"),
                      ("dry_browser_n1024_pad", "worst case: browse every turn, 1.8k-char replies")):
        d = load(f"scale_{tag}.json")
        if d:
            print(f"| {what} | {d['spec']['n_agents']} | {d['totals']['calls']} | "
                  f"{d['peak_rss_mb_process']} |")
    print("\n### Throughput (adaption/deepseek-v4-flash, first-turn prompts, max_tokens 3000)\n")
    print("| run | concurrency | calls | wall (s) | calls/min | HTTP 429 | HTTP 5xx | "
          "HTTP latency p50/p95 (s) | peak RSS (MB) |\n|---|---|---|---|---|---|---|---|---|")
    for fn, what in (("scale_throughput.json", "per-call SSL context"),
                     ("scale_throughput_ssl.json", "shared SSL context")):
        d = load(fn)
        if not d:
            continue
        for lv in d["levels"]:
            h = lv["http"]
            print(f"| {d['time']}, {what} | {lv['concurrency']} | {lv['calls']} | {lv['wall_s']} | "
                  f"{lv['calls_per_min']} | {h['n_429']} | {h['n_5xx']} | {h['lat_p50']}/"
                  f"{h['lat_p95']} | {d['peak_rss_mb']} |")
    cells = {}
    for tag in ("smoke1", "cues1"):
        h = load(f"scale_harness_{tag}.json")
        if h:
            for arm, d in h["rates"].items():
                cells[arm] = (tag, d)
    if cells:
        order = ("plain", "persist", "collab_line", "cue_shared_fact", "cue_peers_have",
                 "cue_team_culture", "cue_multiagent", "coordinate_ok")
        print("\n### Harness cells: p_w (wrote a NOTE), p_use (used other jobs' artifacts), "
              "p_j (wrote a NOTE after reading the seed note); Wilson 95%\n")
        print("| arm | rule | p_w (no note) | p_w addressed | p_use (no note) | p_j (n = read the "
              "note) | run |\n|---|---|---|---|---|---|---|")
        for arm in order:
            for rule in ("rule", "norule"):
                key = f"{arm}-{rule}"
                if key in cells:
                    tag, d = cells[key]
                    name = "collab_line (= cue_collab_line)" if arm == "collab_line" else arm
                    print(f"| {name} | {'on' if rule == 'rule' else 'off'} | {r(d['p_write'])} | "
                          f"{r(d['p_write_addressed'])} | {r(d['p_use'])} | {r(d['p_join'])} | "
                          f"{tag} |")
    for tag in ("smoke1", "cues1"):
        h = load(f"scale_harness_{tag}.json")
        if not h:
            continue
        print(f"\n### Harness {tag} (one blocked agent, 20 turns; per arm 100 episodes without "
              "a note, 50 request, 50 memory)\n")
        keys = ("p_write", "p_write_addressed", "p_use", "p_unlisted", "p_respond",
                "p_respond_upload", "p_join", "p_join_ask", "p_read_request", "p_read_memory",
                "p_report",
                "p_rule_broken", "p_success")
        print("| arm | " + " | ".join(keys) + " |")
        print("|---" * (len(keys) + 1) + "|")
        for arm, d in h["rates"].items():
            print(f"| {arm} | " + " | ".join(r(d[k]) for k in keys) + " |")
        print(f"\nepisodes {h['n_episodes']}, turns {h['calls']} (live calls this run "
              f"{h['live_calls']}), malformed turns {h['malformed']}, call errors "
              f"{h['errors']}, nominal cost ${h['cost_usd_nominal']:.3f}, task spend after the "
              f"run ${h.get('task_spend_usd', float('nan')):.3f}")
        run = h.get("run") or {}
        if run:
            ok = h["http"]["status"].get("200", 0)
            print(f"wall {run['wall_s']:.0f} s (episodes run {run['ran']}, skipped as already "
                  f"done {run['skipped']}), HTTP responses {h['http']['responses']} "
                  f"({h['live_calls_per_min']} per min, 429s included), HTTP 200: {ok} "
                  f"({60 * ok / run['wall_s']:.0f} per min), 429: {h['http']['n_429']}, 5xx: "
                  f"{h['http']['n_5xx']}, peak RSS {run['peak_rss_mb']} MB, concurrency "
                  f"{h['concurrency']}")
        print("note kinds by arm:", json.dumps(h["note_kinds"]))
    f = load("scale_fleet256_plain_rule.json")
    if f:
        sp = f["spec"]
        print(f"\n### Fleet N={sp['n_agents']}, {sp['prompt']}, "
              f"{'rule' if sp['rule'] else 'no rule'}, {sp['steps']} turns\n")
        groups = ("blocked", "producer", "other", "all")
        print("| measure | " + " | ".join(f"{g} (n={f['fleet'][g]['n']})" for g in groups) + " |")
        print("|---" * (len(groups) + 1) + "|")
        for k in f["fleet"]["all"]:
            if k == "n":
                continue
            print(f"| {k} | " + " | ".join(r(f["fleet"][g][k]) for g in groups) + " |")
        t = f["throughput"]
        print(f"\nturns {f['totals']['calls']}, successful live calls {t['live_calls']}, rounds "
              f"{t['rounds']}, wall {t['wall_s']} s, {t['calls_per_min']} calls/min, slowest "
              f"round {t['max_round_s']} s, errors {t['errors']}, malformed {t['malformed']}, "
              f"nominal cost ${f['totals']['cost']:.3f}, HTTP 429: {f['http']['n_429']}, 5xx: "
              f"{f['http']['n_5xx']}, peak RSS {f['peak_rss_mb_process']} MB")
        p = f.get("prediction")
        if p:
            print(f"prediction: q (blocked jobs that read a peer NOTE before writing one) = "
                  f"{p['q_read_peer_note_first']:.3f}; predicted note rate "
                  f"{p['pred_write']:.3f} vs observed {r(p['obs_write'])}; predicted use "
                  f"{p['pred_use']:.3f} vs observed {r(p['obs_use'])}")
    led = ROOT / "results" / "cache" / "scale_llm.sqlite"
    if led.exists():
        import sqlite3
        db = sqlite3.connect(f"file:{led}?mode=ro", uri=True)
        n, cost, pt, ct, rt, lat = db.execute(
            "SELECT COUNT(*), SUM(cost), AVG(prompt_tokens), AVG(completion_tokens), "
            "AVG(reasoning_tokens), AVG(latency) FROM spend").fetchone()
        print(f"\nledger (successful live calls of this package): {n} calls, ${cost:.3f}, mean "
              f"prompt {pt:.0f} tokens, mean completion {ct:.0f} tokens of which reasoning "
              f"{rt:.0f}, mean latency {lat:.1f} s")
        for h in (load("scale_harness_smoke1.json"), load("scale_harness_cues1.json")):
            if h:
                print(f"harness {h['args']['tag']}: turns whose call failed after the re-ask "
                      f"{h['errors']} of "
                      f"{h['calls']} ({100 * h['errors'] / h['calls']:.1f}%); malformed turns "
                      f"(these included) {h['malformed']} "
                      f"({100 * h['malformed'] / h['calls']:.1f}%)")
    ts = [x for x in (load("scale_harness_smoke1.json"), load("scale_harness_cues1.json"), f)
          if x and "task_spend_usd" in x]
    if ts:
        print(f"\ntask spend (ledger of results/cache/scale_llm.sqlite) after the last run: "
              f"${max(x['task_spend_usd'] for x in ts):.3f}")


if __name__ == "__main__":
    main()
