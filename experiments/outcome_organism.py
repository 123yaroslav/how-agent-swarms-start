"""EXPLORATORY outcome-reward organism (docs/ideas/outcome_organism.md): one step of expert
iteration on the swarm.scale harness. Sample blocked single-agent episodes, keep the rewarded
(successful) ones as training rows (OUTCOME), and compare with the same number of episodes
picked regardless of outcome (MATCHED). Off-policy: the sampler is adaption/deepseek-v4-flash,
the student openai/gpt-oss-20b (the planned on-policy vLLM server was not available).

    # pool: world seeds 4000-4999, plain + rule on even seeds, plain without the rule on odd
    caffeinate -i uv run python experiments/outcome_organism.py sample
    uv run python experiments/outcome_organism.py rates          # pool base rates, Wilson 95%
    uv run python experiments/outcome_organism.py build          # OUTCOME and MATCHED rows
    # train with experiments/autoscientist_run.py (upload, train, download), then
    uv run python experiments/outcome_organism.py adapters       # results/serve/extra_adapters.json
    # once a vLLM server serves the adapters (results/serve/endpoint.json):
    caffeinate -i uv run python experiments/outcome_organism.py eval
    uv run python experiments/outcome_organism.py eval --table   # re-print, no calls
    # scripted dry run of the whole data path (no LLM, separate files)
    uv run python experiments/outcome_organism.py sample --dry 40 && \
        uv run python experiments/outcome_organism.py build --dry

Rows: {"prompt": <JSON string of the exact message list the sampler got at that turn>,
"completion": <its reply>}, the format of experiments/make_training_data.py. Each rebuilt prompt
is checked against the sampler's LLM cache (the request hash must be the key of a call that was
made, and the cached reply must equal the completion). Labels (success) are read only to select
rows (the reward) and in the rates; never as a model input.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swarm.fleet.export import validate_sft_row
from swarm.fleet.runner import sample_index
from swarm.scale import prompts
from swarm.scale.analysis import episode_rows
from swarm.scale.jobs import Job
from swarm.scale.metrics import fmt_rate, rate
from swarm.scale.runner import History, LLMPolicy, RunSpec, labels_path, peak_rss_mb, run_many

RAW = ROOT / "results" / "raw" / "scale"
SUMMARY = ROOT / "results" / "summary"
TRAIN = ROOT / "results" / "train"
SERVE = ROOT / "results" / "serve"
MODELS_DIR = ROOT / "results" / "models"
CACHE = ROOT / "results" / "cache" / "outcome_llm.sqlite"
EVAL_CACHE = ROOT / "results" / "cache" / "outcome_eval_llm.sqlite"

SAMPLER = "adaption/deepseek-v4-flash"
POOL_SEEDS = range(4000, 5000)
STEPS, MAX_TOKENS, TEMPERATURE, WINDOW, N_BACKGROUND = 20, 3000, 0.8, 12, 200
ROWS = 3000
# evaluation: the C1 cell settings of experiments/scale_confirm.py (seeds 2100-2199, offset 1)
EVAL_SEEDS = range(2100, 2200)
EVAL_REPEAT, EVAL_STEPS = 1, 30
ORGANISMS = {"outcome": "outcome-gptoss20b", "matched": "matched-gptoss20b"}
MEASURES = ("rule_broken", "uses_others_artifacts", "wrote_note_addressed", "wrote_note",
            "report_blocked", "success", "uses_unlisted_datasets")


def ptype(rule: bool) -> str:
    return "rule" if rule else "norule"


def pool_path(dry: bool) -> Path:
    return RAW / ("outcome_pool_dry.jsonl" if dry else "outcome_pool.jsonl")


def pool_specs(dry: int = 0) -> list[RunSpec]:
    seeds = list(POOL_SEEDS)[:dry] if dry else POOL_SEEDS
    return [RunSpec(tag="outcome_pool_dry" if dry else "outcome_pool", mode="harness",
                    prompt="plain", rule=s % 2 == 0, seed_note="none", n_background=N_BACKGROUND,
                    steps=STEPS, model="scripted:mix" if dry else SAMPLER, max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE, seed=s, repeat=0, history_window=WINDOW)
            for s in seeds]


def iter_records(path: Path):
    """Records of a JSONL file; a last line still being written by a running sampler (no
    newline yet) is skipped."""
    with path.open() as f:
        for ln in f:
            if ln.strip() and ln.endswith("\n"):
                yield json.loads(ln)


def load_labels(path: Path) -> dict[str, dict]:
    out = {}
    for r in iter_records(labels_path(path)):
        out[r["episode_id"]] = r
    return out


# ---- sampling ---------------------------------------------------------------------------------
def cmd_sample(a) -> None:
    from swarm.scale.http_stats import count_http

    specs = pool_specs(a.dry)
    out = pool_path(bool(a.dry))
    if a.dry:
        from swarm.scale.scripted import ScriptedPolicy
        styles = ("worker", "silent", "user", "noter")
        policy_for = lambda s: ScriptedPolicy(styles[s.seed % len(styles)])  # noqa: E731
        llm = None
    else:
        from swarm.scale.llm_task import TaskLLM
        llm = TaskLLM(budget_usd=a.budget, max_concurrency=a.concurrency, cache_path=CACHE)
        pol = LLMPolicy(llm, SAMPLER, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
        policy_for = lambda s: pol  # noqa: E731
    t0, last = time.time(), [0.0]

    def progress(k: int, n: int) -> None:
        if time.time() - last[0] > 60 or k == n:
            last[0] = time.time()
            sp = f", spend ${llm.total_spent():.3f}" if llm else ""
            print(f"{k}/{n} episodes, {time.time() - t0:.0f}s, rss {peak_rss_mb():.0f}MB{sp}",
                  flush=True)

    # Phase A: the no-rule seeds (odd) in blocks of a.block // 2 (the no-rule half of a block of
    # a.block seeds) until the successful episodes hold ROWS usable rows. Phase B: the rule
    # seeds (even) of the same seed range, so the pool ends 50/50 (docs/ideas/outcome_organism.md)
    half = a.block // 2
    norule = [s for s in specs if not s.rule]
    blocks, res, info = [], {}, {}

    def write_info(phase: str) -> None:
        info.update({"phase": phase, "run": res, "blocks": blocks, "http": hs.summary(),
                     "concurrency": a.concurrency,
                     "pool_seeds": [specs[0].seed, blocks[-1]["seeds"][1]] if blocks else None})
        if llm is not None:
            info["llm_summary"] = llm.summary()
            info["task_spend_usd"] = round(llm.total_spent(), 4)
        if not a.dry:
            SUMMARY.mkdir(parents=True, exist_ok=True)
            (SUMMARY / "outcome_pool_run.json").write_text(json.dumps(info, indent=1))

    print(f"pool phase A: no-rule seeds in blocks of {half} -> {out.relative_to(ROOT)}",
          flush=True)
    with count_http() as hs:
        for b0 in range(0, len(norule), half):
            block = norule[b0:b0 + half]
            res = asyncio.run(run_many(block, policy_for, out, concurrency=a.concurrency,
                                       progress=progress))
            if res["stopped"]:
                break
            rows = success_rows(out)
            end = block[-1].seed
            blocks.append({"seeds": [block[0].seed - 1, end], "success_rows_so_far": rows,
                           "wall_s": round(res["wall_s"], 1)})
            print(f"block {block[0].seed - 1}-{end} no-rule half done: usable rows of "
                  f"successful episodes so far {rows} (need {ROWS})", flush=True)
            if rows >= (ROWS if not a.dry else 1):  # the stopping rule of the design
                break
        write_info("A")
        print(f"phase A done: pool seeds {info['pool_seeds']}", flush=True)
        if blocks and not res.get("stopped"):
            hi = blocks[-1]["seeds"][1]
            rule_specs = [s for s in specs if s.rule and s.seed <= hi]
            print(f"pool phase B: {len(rule_specs)} rule seeds up to {hi}", flush=True)
            res = asyncio.run(run_many(rule_specs, policy_for, out, concurrency=a.concurrency,
                                       progress=progress))
            write_info("B")
            print("phase B done", flush=True)
    if llm is not None:
        print(llm.summary())
    print("run:", json.dumps(res), "http:", json.dumps(info.get("http")))


def success_rows(path: Path) -> int:
    """Usable rows (turns) of the successful episodes in the pool file so far."""
    labels = load_labels(path)
    n = 0
    for rec in iter_records(path):
        if episode_rows([rec], labels)[0]["success"]:
            n += sum(1 for t in rec["turns"] if usable(t))
    return n


# ---- pool base rates --------------------------------------------------------------------------
def pool_rows(path: Path) -> list[dict]:
    labels = load_labels(path)
    rows = []
    for rec in iter_records(path):
        for r in episode_rows([rec], labels):
            t = rec["totals"]
            rows.append({**r, "ptype": ptype(r["rule"]), "cost": t["cost"],
                         "malformed_turns": t["malformed"], "error_turns": t["errors"],
                         "turns": t["calls"]})
    return rows


def rates_of(rows: list[dict]) -> dict:
    out = {k: rate(rows, k) for k in MEASURES}
    out["success_and_use"] = rate(rows, "success", lambda r: r["uses_others_artifacts"])
    out["use_given_success"] = rate(rows, "uses_others_artifacts", lambda r: r["success"])
    out["n"] = len(rows)
    return out


def cmd_rates(a) -> None:
    path = pool_path(a.dry)
    rows = pool_rows(path)
    by = defaultdict(list)
    for r in rows:
        by[r["ptype"]].append(r)
    res = {pt: rates_of(rs) for pt, rs in sorted(by.items())}
    res["all"] = rates_of(rows)
    res["cost_usd_nominal"] = round(sum(r["cost"] for r in rows), 4)
    res["turns"] = sum(r["turns"] for r in rows)
    res["malformed_turns"] = sum(r["malformed_turns"] for r in rows)
    res["error_turns"] = sum(r["error_turns"] for r in rows)
    print_rates(res)
    if not a.dry:
        (SUMMARY / "outcome_pool_rates.json").write_text(json.dumps(res, indent=1))


def print_rates(res: dict) -> None:
    cols = [k for k in ("rule", "norule", "all") if k in res]
    print("| measure | " + " | ".join(f"{c} (n={res[c]['n']})" for c in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for k in (*MEASURES, "use_given_success"):
        print(f"| {k} | " + " | ".join(fmt_rate(res[c][k]) for c in cols) + " |")
    print(f"turns {res['turns']}, malformed {res['malformed_turns']}, failed calls "
          f"{res['error_turns']}, nominal cost ${res['cost_usd_nominal']:.3f}")


# ---- rows -------------------------------------------------------------------------------------
def usable(turn: dict) -> bool:
    return not turn.get("malformed") and "error" not in turn


def chat_rows(rec: dict):
    """(turn, messages, completion, turn record) for every turn of a harness episode, the
    messages rebuilt exactly as runner.run_world built them (History, same window)."""
    sp = rec["spec"]
    job = Job(**rec["world"]["jobs"][0])
    h = History(prompts.system_prompt(job.job_id, sp["steps"], prompt=sp["prompt"],
                                      rule=sp["rule"]),
                prompts.begin_message(job, sp["steps"]), sp["history_window"])
    for t in sorted(rec["turns"], key=lambda x: x["step"]):
        yield t["step"], h.messages(), t["assistant"], t
        obs = t["user"].split("\n", 1)[1] if "\n" in t["user"] else ""
        if obs.startswith(("ok: stored ", "ok: wrote ")):
            h.created.append(obs.split()[2])
        elif obs.startswith("ok: the operator was told"):
            h.reported = True
        h.add(t["assistant"], t["user"])


class CacheCheck:
    """Is a rebuilt prompt the request that was sent? Recompute swarm.llm's Adaption cache key
    and compare the cached reply with the completion."""

    def __init__(self, path: Path):
        from swarm.llm import ADAPTION_URL, CACHE_VERSION
        self.url, self.v = ADAPTION_URL, CACHE_VERSION
        self.db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    def key(self, model: str, messages: list[dict], sample: int) -> str:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [{"role": m["role"], "content": m["content"]}
                 for m in messages if m["role"] != "system"]
        body = {"model": model.removeprefix("adaption/"), "input": turns,
                "temperature": TEMPERATURE, "max_output_tokens": MAX_TOKENS, "store": False}
        if system:
            body["instructions"] = system
        return hashlib.sha256(json.dumps({**body, "sample": sample, "endpoint": self.url,
                                          "v": self.v}, sort_keys=True).encode()).hexdigest()

    def check(self, rec: dict, step: int, messages: list[dict], completion: str,
              turn: dict) -> bool:
        sp = rec["spec"]
        s = sample_index(sp["seed"], sp["repeat"], rec["world"]["jobs"][0]["job_id"], step)
        if turn.get("reasked"):
            s = sample_index(s, "reask", 1)
        row = self.db.execute("SELECT value FROM calls WHERE key = ?",
                              (self.key(sp["model"], messages, s),)).fetchone()
        if row is None:
            return False
        text = json.loads(row[0]).get("text") or ""
        return (text or "(no reply)")[:2000] == completion


def episode_index(path: Path) -> list[dict]:
    labels = load_labels(path)
    out = []
    for rec in iter_records(path):
        r = episode_rows([rec], labels)[0]
        good = [t["step"] for t in rec["turns"] if usable(t)]
        out.append({"episode_id": rec["episode_id"], "seed": rec["spec"]["seed"],
                    "ptype": ptype(rec["spec"]["rule"]), "success": bool(r["success"]),
                    "use": bool(r["uses_others_artifacts"]),
                    "report": bool(r["report_blocked"]), "turns": good})
    return sorted(out, key=lambda e: e["episode_id"])


def pick_outcome(eps: list[dict], n_rows: int) -> dict[str, list[int]]:
    """Successful episodes, fixed shuffled order, whole episodes until n_rows; the last one cut
    to a fixed random subset. Returns {episode_id: turns}."""
    succ = [e for e in eps if e["success"] and e["turns"]]
    random.Random("outcome|pick").shuffle(succ)
    out, total = {}, 0
    for e in succ:
        if total >= n_rows:
            break
        take = min(len(e["turns"]), n_rows - total)
        turns = e["turns"] if take == len(e["turns"]) else sorted(
            random.Random(f"outcome|cut|{e['episode_id']}").sample(e["turns"], take))
        out[e["episode_id"]] = turns
        total += take
    if total < n_rows:
        raise SystemExit(f"OUTCOME: only {total} rows from {len(succ)} successful episodes, "
                         f"need {n_rows}")
    return out


def pick_matched(eps: list[dict], outcome: dict[str, list[int]]) -> dict[str, list[int]]:
    """Per prompt type: as many episodes as OUTCOME has, drawn at random regardless of outcome,
    then as many rows as OUTCOME has, drawn at random from their usable turns."""
    by_id = {e["episode_id"]: e for e in eps}
    out: dict[str, list[int]] = {}
    for pt in ("rule", "norule"):
        k_eps = sum(1 for eid in outcome if by_id[eid]["ptype"] == pt)
        k_rows = sum(len(t) for eid, t in outcome.items() if by_id[eid]["ptype"] == pt)
        if not k_eps:
            continue
        cand = [e for e in eps if e["ptype"] == pt and e["turns"]]
        random.Random(f"matched|episodes|{pt}").shuffle(cand)
        chosen = cand[:k_eps]
        cells = [(e["episode_id"], t) for e in chosen for t in e["turns"]]
        if len(cells) < k_rows:
            raise SystemExit(f"MATCHED {pt}: {len(cells)} rows in {k_eps} episodes, need {k_rows}")
        for eid, t in random.Random(f"matched|rows|{pt}").sample(cells, k_rows):
            out.setdefault(eid, []).append(t)
        for e in chosen:  # an episode keeps its place even if none of its rows was drawn
            out.setdefault(e["episode_id"], [])
    return {eid: sorted(t) for eid, t in out.items()}


def write_rows(path: Path, picks: dict[str, dict[str, list[int]]], index: list[dict],
               dry: bool, out_names: dict[str, str]) -> dict:
    """One pass over the pool: rows of every organism to temp files, then shuffled copies."""
    by_id = {e["episode_id"]: e for e in index}
    check = None if dry else CacheCheck(CACHE)
    tmp = {o: (TRAIN / f"{n}.tmp").open("wb") for o, n in out_names.items()}
    entries: dict[str, list] = {o: [] for o in out_names}
    stats = {o: {"rows": 0, "cache_verified": 0, "cache_missing": 0} for o in out_names}
    want = defaultdict(dict)
    for o, pk in picks.items():
        for eid, turns in pk.items():
            want[eid][o] = set(turns)
    for rec in iter_records(path):
        w = want.get(rec["episode_id"])
        if not w:
            continue
        e = by_id[rec["episode_id"]]
        for step, msgs, completion, turn in chat_rows(rec):
            orgs = [o for o, ts in w.items() if step in ts]
            if not orgs:
                continue
            assert usable(turn), (rec["episode_id"], step)
            validate_sft_row({"prompt": msgs, "completion": completion}, "prompt_completion")
            ok = check.check(rec, step, msgs, completion, turn) if check else None
            b = (json.dumps({"prompt": json.dumps(msgs), "completion": completion}) + "\n"
                 ).encode()
            meta = {"episode_id": rec["episode_id"], "seed": e["seed"], "ptype": e["ptype"],
                    "turn": step, "episode_success": e["success"], "episode_use": e["use"],
                    "episode_report": e["report"]}
            for o in orgs:
                entries[o].append((tmp[o].tell(), len(b), meta))
                tmp[o].write(b)
                stats[o]["rows"] += 1
                if ok is not None:
                    stats[o]["cache_verified" if ok else "cache_missing"] += 1
    for fh in tmp.values():
        fh.close()
    files = {}
    for o, name in out_names.items():
        ents = entries[o]
        random.Random(f"shuffle|{name}").shuffle(ents)
        dst, mdst = TRAIN / f"{name}.jsonl", TRAIN / f"{name}.meta.jsonl"
        h = hashlib.sha256()
        with (TRAIN / f"{name}.tmp").open("rb") as src, dst.open("wb") as f, \
                mdst.open("w") as mf:
            for off, ln, meta in ents:
                src.seek(off)
                b = src.read(ln)
                f.write(b)
                h.update(b)
                mf.write(json.dumps(meta) + "\n")
        (TRAIN / f"{name}.tmp").unlink()
        eps = {m["episode_id"] for *_, m in ents}
        files[o] = {"path": str(dst.relative_to(ROOT)), **stats[o], "sha256": h.hexdigest(),
                    "bytes": dst.stat().st_size, "episodes_with_rows": len(eps)}
    return files


def describe(pk: dict[str, list[int]], index: list[dict]) -> dict:
    by_id = {e["episode_id"]: e for e in index}
    d: dict = {"episodes": len(pk), "rows": sum(len(t) for t in pk.values())}
    for pt in ("rule", "norule"):
        ids = [eid for eid in pk if by_id[eid]["ptype"] == pt]
        d[pt] = {"episodes": len(ids), "rows": sum(len(pk[i]) for i in ids),
                 "episodes_success": sum(by_id[i]["success"] for i in ids),
                 "episodes_use": sum(by_id[i]["use"] for i in ids),
                 "episodes_report": sum(by_id[i]["report"] for i in ids)}
    return d


def cmd_build(a) -> None:
    path = pool_path(a.dry)
    index = episode_index(path)
    rows = a.rows or (60 if a.dry else ROWS)
    outcome = pick_outcome(index, rows)
    matched = pick_matched(index, outcome)
    suffix = "_dry" if a.dry else (f"_{rows // 1000}k" if rows % 1000 == 0 else f"_{rows}")
    names = {"outcome": f"outcome{suffix}", "matched": f"matched{suffix}"}
    TRAIN.mkdir(parents=True, exist_ok=True)
    files = write_rows(path, {"outcome": outcome, "matched": matched}, index, a.dry, names)
    succ = [e for e in index if e["success"]]
    manifest = {
        "design": "docs/ideas/outcome_organism.md", "sampler": SAMPLER, "student":
        "openai/gpt-oss-20b", "off_policy": True, "pool": str(path.relative_to(ROOT)),
        "pool_episodes": len(index), "pool_success_episodes": len(succ),
        "pool_success_by_ptype": {pt: sum(1 for e in succ if e["ptype"] == pt)
                                  for pt in ("rule", "norule")},
        "pool_usable_rows_by_ptype": {pt: sum(len(e["turns"]) for e in index
                                          if e["ptype"] == pt) for pt in ("rule", "norule")},
        "world_seeds": f"{POOL_SEEDS.start}-{POOL_SEEDS.stop - 1}", "steps": STEPS,
        "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "history_window": WINDOW,
        "row_format": "prompt = JSON string of the exact [{role, content}] list the sampler got "
                      "at that turn; completion = its reply (cut at 2,000 characters, as the "
                      "episode history holds it)",
        "row_rule": "turns whose call failed or whose reply was not one valid JSON action are "
                    "not rows (they stay in later prompts)",
        "organisms": {o: {**describe(pk, index), "file": files[o]}
                      for o, pk in (("outcome", outcome), ("matched", matched))},
        "built": time.strftime("%Y-%m-%dT%H:%M:%S")}
    mpath = TRAIN / ("outcome_manifest_dry.json" if a.dry else "outcome_manifest.json")
    mpath.write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k != "row_format"}, indent=1))


# ---- adapters for the server ------------------------------------------------------------------
def cmd_adapters(a) -> None:
    runs = {}
    ledger = ROOT / "results" / "autoscientist" / "runs.jsonl"
    for r in iter_records(ledger):
        if r["op"] == "train" and r["tag"] in (f"{o}-gptoss20b-3k" for o in ORGANISMS):
            runs[r["tag"].split("-")[0]] = r["run_id"]
    out = []
    for name, folder in ORGANISMS.items():
        d = MODELS_DIR / folder
        files = sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()) \
            if d.exists() else []
        cfg = next((f for f in files if f.endswith("adapter_config.json")), None)
        out.append({"name": name, "folder": str(d.relative_to(ROOT)), "run_id": runs.get(name),
                    "base_model": "openai/gpt-oss-20b", "adapter_config": cfg,
                    "files": files, "ready": bool(cfg)})
    SERVE.mkdir(parents=True, exist_ok=True)
    doc = {"written": time.strftime("%Y-%m-%dT%H:%M:%S"), "by": "experiments/outcome_organism.py",
           "note": "LoRA adapters on openai/gpt-oss-20b to serve next to the others; serve each "
                   "under its `name` (vLLM --lora-modules name=<path to folder of "
                   "adapter_config>)", "adapters": out}
    (SERVE / "extra_adapters.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc, indent=1))


# ---- evaluation -------------------------------------------------------------------------------
def endpoint(path: Path) -> tuple[str, str | None]:
    ep = json.loads(path.read_text())
    base_url = ep["base_url"].rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    base = ep.get("base_model") or ep.get("model")
    models = ep.get("models")
    if base is None and isinstance(models, dict):
        base = models.get("base")
    if base is None and isinstance(models, list) and models:
        base = models[0] if isinstance(models[0], str) else models[0].get("name")
    return base_url, base


def set_token(path: Path) -> None:
    """The server token goes only into this process's environment (never printed)."""
    tok = path.read_text().strip()
    os.environ["SWARM_OAI_API_KEY"] = tok


def served_models(base_url: str) -> list[str]:
    import httpx
    r = httpx.get(f"{base_url}/models", headers={
        "Authorization": f"Bearer {os.environ['SWARM_OAI_API_KEY']}"}, timeout=30)
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", [])]


def load_adapter(base_url: str, name: str, remote_path: str) -> None:
    import httpx
    root = base_url.removesuffix("/v1")
    r = httpx.post(f"{root}/v1/load_lora_adapter", json={"lora_name": name,
                                                         "lora_path": remote_path},
                   headers={"Authorization": f"Bearer {os.environ['SWARM_OAI_API_KEY']}"},
                   timeout=300)
    print(f"load_lora_adapter {name}: HTTP {r.status_code} {r.text[:200]}", flush=True)
    r.raise_for_status()


def eval_specs(label: str, model: str, steps: int, out_tag: str = "outcome_eval",
               n_seeds: int = 100) -> list[RunSpec]:
    out = []
    for s in list(EVAL_SEEDS)[:n_seeds]:
        for rule in (True, False):
            out.append(RunSpec(tag=f"{out_tag}_{label}", mode="harness", prompt="plain",
                               rule=rule, seed_note="none", n_background=N_BACKGROUND,
                               steps=steps, model=model, max_tokens=MAX_TOKENS,
                               temperature=TEMPERATURE, seed=s, repeat=EVAL_REPEAT,
                               history_window=WINDOW))
    return out


def eval_path(label: str, out_tag: str = "outcome_eval") -> Path:
    return RAW / f"{out_tag}_{label}.jsonl"


def paired_boot(a: dict[int, float], b: dict[int, float], n_boot: int = 10_000,
                seed: int = 0) -> dict:
    import numpy as np
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_seeds": 0}
    d = np.array([a[s] - b[s] for s in seeds], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return {"diff": float(d.mean()), "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)), "n_seeds": len(seeds)}


def eval_summary(labels: list[str], out_tag: str = "outcome_eval") -> dict:
    cells: dict = {}
    per_seed: dict = defaultdict(dict)
    for lab in labels:
        p = eval_path(lab, out_tag)
        if not p.exists():
            continue
        rows = pool_rows(p)
        for pt in ("rule", "norule"):
            rs = [r for r in rows if r["ptype"] == pt]
            if not rs:
                continue
            cells[f"{lab}|{pt}"] = {**{k: rate(rs, k) for k in MEASURES}, "n": len(rs),
                                    "malformed_turns": sum(r["malformed_turns"] for r in rs),
                                    "error_turns": sum(r["error_turns"] for r in rs),
                                    "turns": sum(r["turns"] for r in rs)}
            for k in MEASURES:
                per_seed[(lab, pt, k)] = {r["seed"]: float(bool(r[k])) for r in rs
                                          if r.get(k) is not None}
    diffs = {}
    if "outcome" in labels and "matched" in labels:
        for pt in ("rule", "norule"):
            for k in MEASURES:
                a, b = per_seed.get(("outcome", pt, k)), per_seed.get(("matched", pt, k))
                if a and b:
                    diffs[f"{pt}|{k}"] = paired_boot(a, b)
    return {"cells": cells, "outcome_minus_matched": diffs}


def print_eval(s: dict) -> None:
    keys = ("rule_broken", "uses_others_artifacts", "wrote_note_addressed", "report_blocked",
            "success")
    print("| model | cell | " + " | ".join(keys) + " |")
    print("|---|---|" + "---|" * len(keys))
    for c, d in s["cells"].items():
        lab, pt = c.split("|")
        print(f"| {lab} | plain {'+ rule' if pt == 'rule' else 'without rule'} | "
              + " | ".join(fmt_rate(d[k]) if d[k]["n"] else "n/a" for k in keys) + " |")
    if s["outcome_minus_matched"]:
        print("\nOUTCOME minus MATCHED (paired by world seed, cluster bootstrap over seeds, 95%)")
        for c, d in s["outcome_minus_matched"].items():
            print(f"  {c:<34} {d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] "
                  f"(seeds {d['n_seeds']})")


def cmd_eval(a) -> None:
    labels = a.models.split(",")
    if not a.table:
        from swarm.scale.http_stats import count_http
        from swarm.scale.llm_task import TaskLLM

        set_token(Path(a.token_file))
        base_url, base = endpoint(Path(a.endpoint))
        served = served_models(base_url)
        print(f"server {base_url}: {len(served)} models served: {served}", flush=True)
        extra = json.loads(Path(a.adapters_file).read_text())["adapters"]
        names = {}
        for lab in labels:
            if lab == "base":
                if not base:
                    raise SystemExit("endpoint.json names no base model")
                names[lab] = base
                continue
            names[lab] = lab
            if lab not in served:
                ent = next(x for x in extra if x["name"] == lab)
                if not a.load:
                    raise SystemExit(f"adapter {lab} is not served; rerun with --load "
                                     f"--remote-root <adapter dir on the server> once the "
                                     f"folders are on the pod, or wait for the GPU agent")
                load_adapter(base_url, lab, f"{a.remote_root.rstrip('/')}/"
                             f"{Path(ent['folder']).name}")
        llm = TaskLLM(budget_usd=1000.0, max_concurrency=a.concurrency * len(labels),
                      cache_path=EVAL_CACHE if a.out_tag == "outcome_eval"
                      else ROOT / "results" / "cache" / f"{a.out_tag}_llm.sqlite")
        jobs = []
        for lab in labels:
            model = f"oai:{base_url}|{names[lab]}"
            pol = LLMPolicy(llm, model, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
            jobs.append((lab, eval_specs(lab, model, a.steps, a.out_tag, a.n_seeds), pol))
        t0 = time.time()

        def progress_for(lab):
            last = [0.0]

            def progress(k, n):
                if time.time() - last[0] > 60 or k == n:
                    last[0] = time.time()
                    print(f"{lab}: {k}/{n} episodes, {time.time() - t0:.0f}s, rss "
                          f"{peak_rss_mb():.0f}MB", flush=True)
            return progress

        async def run_all():
            return await asyncio.gather(*(
                run_many(specs, (lambda s, p=pol: p), eval_path(lab, a.out_tag),
                         concurrency=a.concurrency,
                         progress=progress_for(lab)) for lab, specs, pol in jobs))

        with count_http() as hs:
            res = asyncio.run(run_all())
        print(llm.summary())
        print("runs:", json.dumps(res), "http:", json.dumps(hs.summary()))
    s = eval_summary(labels, a.out_tag)
    s["settings"] = {"seeds": f"{EVAL_SEEDS.start}-{EVAL_SEEDS.start + a.n_seeds - 1}",
                     "repeat": EVAL_REPEAT, "steps": a.steps, "temperature": TEMPERATURE,
                     "max_tokens": MAX_TOKENS, "n_background": N_BACKGROUND}
    SUMMARY.mkdir(parents=True, exist_ok=True)
    (SUMMARY / f"{a.out_tag}.json").write_text(json.dumps(s, indent=1))
    print_eval(s)
    if a.mark_done and not a.table:
        (SERVE / "sampling_done.txt").write_text(
            f"outcome organism evaluation finished {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--concurrency", type=int, default=24)
    s.add_argument("--budget", type=float, default=15.0)
    s.add_argument("--dry", type=int, default=0, help="scripted dry run on the first N seeds")
    s.add_argument("--block", type=int, default=100,
                   help="seeds per block; stop after the first block with enough rows")
    r = sub.add_parser("rates")
    r.add_argument("--dry", action="store_true")
    b = sub.add_parser("build")
    b.add_argument("--dry", action="store_true")
    b.add_argument("--rows", type=int, default=None,
                   help="rows per organism (default 3,000; 60 in a dry build)")
    sub.add_parser("adapters")
    e = sub.add_parser("eval")
    e.add_argument("--models", default="outcome,matched,base")
    e.add_argument("--steps", type=int, default=EVAL_STEPS)
    e.add_argument("--concurrency", type=int, default=32, help="episodes in flight per model")
    e.add_argument("--load", action="store_true", help="load missing adapters at runtime")
    e.add_argument("--remote-root", default="/workspace/adapters")
    e.add_argument("--table", action="store_true")
    e.add_argument("--endpoint", default=str(SERVE / "endpoint.json"))
    e.add_argument("--token-file", default=str(SERVE / "vllm_token.txt"))
    e.add_argument("--adapters-file", default=str(SERVE / "extra_adapters.json"))
    e.add_argument("--out-tag", default="outcome_eval", help="raw, cache and summary names")
    e.add_argument("--n-seeds", type=int, default=100)
    e.add_argument("--mark-done", action="store_true",
                   help="write results/serve/sampling_done.txt after the run")
    a = ap.parse_args(argv)
    {"sample": cmd_sample, "rates": cmd_rates, "build": cmd_build, "adapters": cmd_adapters,
     "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
