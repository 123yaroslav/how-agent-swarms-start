"""Confirmatory single-agent cells C1-C3 (docs/prereg.md, frozen at d8900cb) and the seeded
probe episodes for the C4 predictions (docs/deviations.md D6), one model per process.

    # C1-C3 cells of one model (9 cells, n = 100, or 70 for glm-5.2 and qwen3.5-397b)
    caffeinate -i uv run python experiments/scale_confirm.py --model deepseek-v4-flash
    # C4 probe episodes of one model (seed notes, store background = fleet size)
    caffeinate -i uv run python experiments/scale_confirm.py --model deepseek-v4-flash --probes

World seeds, sample offset and settings: docs/deviations.md D1, D6, D7. Output:
results/raw/scale/confirm_<model>.jsonl and probe_<model>.jsonl (+ .labels.jsonl). Finished
episodes are skipped on a rerun; calls replay from results/cache/scale_confirm_llm.sqlite, whose
ledger caps the spend of the confirmatory set. Numbers: experiments/scale_confirm_eval.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swarm.scale.runner import LLMPolicy, RunSpec, peak_rss_mb, run_many

RAW = ROOT / "results" / "raw" / "scale"
CACHE = ROOT / "results" / "cache" / "scale_confirm_llm.sqlite"
BUDGET = 650.0
STEPS = 30
TEMPERATURE = 0.8
REPEAT = 1  # sample offset (D1)
SEED_NONE = 2100
SEED_PROBE = {"request": 2300, "memory": 2500}
# model: (n per cell, max tokens, fleet N, fleet arms, probe episodes per note style)
# n = 70 for glm-5.2 and qwen3.5-397b (frozen 60; docs/deviations.md D5)
MODELS = {
    "deepseek-v4-flash": (100, 3000, 1024, ("plain", "coordinate_ok"), 50),
    "gemma-4-26b-a4b-it": (100, 2000, 256, ("plain", "coordinate_ok"), 50),
    "gemma-4-31b-it": (100, 2000, 256, ("plain", "coordinate_ok"), 50),
    "qwen3.8-27b": (100, 2000, 256, ("plain", "coordinate_ok"), 50),
    "glm-5.2": (70, 2500, 1024, ("plain",), 30),
    "qwen3.5-397b": (70, 2500, 1024, ("plain",), 30),
}
# (prompt arm, rule): C1 plain/coordinate_ok, C2 six phrasings, C3 plain without the rule
CELLS = [("plain", True), ("coordinate_ok", True), ("cue_shared_fact", True),
         ("cue_peers_have", True), ("cue_team_culture", True), ("cue_multiagent", True),
         ("collab_line", True), ("persist", True), ("plain", False)]


def cell_specs(model: str) -> list[RunSpec]:
    n, mt = MODELS[model][:2]
    out = []
    for i in range(n):  # seed-major order: every cell advances together
        for prompt, rule in CELLS:
            out.append(RunSpec(tag=f"confirm_{model}", mode="harness", prompt=prompt, rule=rule,
                               seed_note="none", n_background=200, steps=STEPS,
                               model=f"adaption/{model}", max_tokens=mt,
                               temperature=TEMPERATURE, seed=SEED_NONE + i, repeat=REPEAT))
    return out


def probe_specs(model: str) -> list[RunSpec]:
    _, mt, big_n, arms, k = MODELS[model]  # k request + k memory episodes per arm
    out = []
    for i in range(k):
        for note in ("request", "memory"):
            for prompt in arms:
                out.append(RunSpec(tag=f"probe_{model}", mode="harness", prompt=prompt,
                                   rule=True, seed_note=note, n_background=big_n,
                                   steps=STEPS, model=f"adaption/{model}", max_tokens=mt,
                                   temperature=TEMPERATURE, seed=SEED_PROBE[note] + i,
                                   repeat=REPEAT))
    return out


def out_path(model: str, probes: bool) -> Path:
    return RAW / f"{'probe' if probes else 'confirm'}_{model}.jsonl"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--probes", action="store_true")
    p.add_argument("--concurrency", type=int, default=64)
    a = p.parse_args(argv)
    specs = probe_specs(a.model) if a.probes else cell_specs(a.model)
    out = out_path(a.model, a.probes)
    from swarm.scale.http_stats import count_http
    from swarm.scale.llm_task import TaskLLM

    llm = TaskLLM(budget_usd=BUDGET, max_concurrency=a.concurrency, cache_path=CACHE)
    shared = LLMPolicy(llm, f"adaption/{a.model}", max_tokens=MODELS[a.model][1],
                       temperature=TEMPERATURE)
    t0 = time.time()
    last = [0.0]

    def progress(k: int, n: int) -> None:
        if time.time() - last[0] > 60 or k == n:
            last[0] = time.time()
            print(f"{k}/{n} episodes, {time.time() - t0:.0f}s, rss {peak_rss_mb():.0f}MB, "
                  f"task spend ${llm.total_spent():.3f}", flush=True)

    print(f"{a.model} {'probes' if a.probes else 'cells'}: {len(specs)} episodes -> {out}",
          flush=True)
    with count_http() as hs:
        res = asyncio.run(run_many(specs, lambda s: shared, out, concurrency=a.concurrency,
                                   progress=progress))
    print("run:", json.dumps(res), "http:", json.dumps(hs.summary()), flush=True)
    print(llm.summary(), flush=True)
    log = RAW / "confirm_runs.jsonl"
    with log.open("a") as f:
        f.write(json.dumps({"model": a.model, "probes": a.probes, "run": res,
                            "http": hs.summary(), "task_spend_usd": round(llm.total_spent(), 4),
                            "summary": llm.summary(), "ts": time.time()}) + "\n")


if __name__ == "__main__":
    main()
