"""Evaluate SwarmCheck lint (regex and LLM modes) against the manual labels of the real framework
prompts, on a held-out split of projects.

Split: md5(project) is even -> dev, odd -> test. Rules or prompts may be tuned on dev only; the
numbers reported in the paper are the test split. Written 2026-11-07 before any LLM-mode run.

    uv run python experiments/lint_eval.py [--models adaption/deepseek-v4-flash,adaption/glm-5.2]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
from pathlib import Path

from swarm.llm import LLM
from swarm.swarmcheck.lint import LINT_VERSION, lint
from swarm.swarmcheck.lint_llm import judge

ROOT = Path(__file__).resolve().parents[1]
RW = ROOT / "results" / "realworld_prompts"
TEXTS = ROOT / "data" / "raw" / "framework_prompts"
OUT = RW / "lint_eval.json"


def split_of(project: str) -> str:
    return "dev" if int(hashlib.md5(project.encode()).hexdigest(), 16) % 2 == 0 else "test"


def confusion(pred: list[bool], gold: list[bool]) -> dict:
    tp = sum(p and g for p, g in zip(pred, gold))
    fp = sum(p and not g for p, g in zip(pred, gold))
    fn = sum(g and not p for p, g in zip(pred, gold))
    tn = sum(not p and not g for p, g in zip(pred, gold))
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec}


async def main_async(models: list[str]) -> None:
    labels = {r["file"]: r for r in csv.DictReader(open(RW / "labels_manual.csv"))}
    proj = {r["file"]: r["project"] for r in csv.DictReader(open(RW / "sources.csv"))}
    files = [f for f in labels if (TEXTS / f).exists()]
    gold_contact = {f: labels[f]["contact_shared"] == "yes" for f in files}
    gold_missing = {f: labels[f]["shared_storage"] == "yes" and labels[f]["boundary_rule"] != "yes"
                    for f in files}
    rx = f"regex_v{LINT_VERSION}"
    preds: dict[str, dict] = {rx: {}}
    for f in files:
        fs = {(x.level, x.rule) for x in lint((TEXTS / f).read_text())}
        preds[rx][f] = {"contact": ("HIGH", "permission_to_coordinate") in fs,
                             "missing": ("MED", "missing_boundary_rule") in fs}
    llm = LLM(budget_usd=40, max_concurrency=6)  # global ledger cap shared with earlier runs (about $20 so far)
    for m in models:
        js = await asyncio.gather(*[judge((TEXTS / f).read_text(), m, llm, sample=0) for f in files])
        preds[m] = {f: {"contact": j["contact_shared"],
                        "missing": j["shared_storage"] and not j["boundary_rule"],
                        "evidence": j["evidence"]} for f, j in zip(files, js)}
    out: dict = {"n_files": len(files), "split_rule": "md5(project) even=dev, odd=test",
                 "results": {}}
    for split in ("dev", "test", "all"):
        fsel = [f for f in files if split == "all" or split_of(proj.get(f, f)) == split]
        out["results"][split] = {"n": len(fsel),
                                 "n_contact": sum(gold_contact[f] for f in fsel),
                                 "n_missing": sum(gold_missing[f] for f in fsel)}
        for name, p in preds.items():
            out["results"][split][name] = {
                "contact_shared": confusion([p[f]["contact"] for f in fsel],
                                            [gold_contact[f] for f in fsel]),
                "missing_boundary": confusion([p[f]["missing"] for f in fsel],
                                              [gold_missing[f] for f in fsel])}
    out["llm_evidence"] = {m: {f: preds[m][f]["evidence"] for f in files} for m in models}
    OUT.write_text(json.dumps(out, indent=1))
    for split in ("dev", "test"):
        r = out["results"][split]
        print(f"== {split}: n={r['n']} contact={r['n_contact']} missing={r['n_missing']}")
        for name in preds:
            c, mb = r[name]["contact_shared"], r[name]["missing_boundary"]
            print(f"  {name:32s} contact P={c['precision']} R={c['recall']} "
                  f"(tp {c['tp']} fp {c['fp']} fn {c['fn']}) | missing P={mb['precision']} "
                  f"R={mb['recall']}")
    print(llm.summary())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="adaption/deepseek-v4-flash")
    a = ap.parse_args()
    asyncio.run(main_async([m for m in a.models.split(",") if m]))


if __name__ == "__main__":
    main()
