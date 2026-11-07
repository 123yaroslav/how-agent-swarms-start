"""SwarmCheck linter on real framework prompts, against manual labels (EXPLORATORY).

    uv run python -I experiments/realworld_prompts_lint.py

Reads the prompts written by experiments/realworld_prompts_fetch.py
(data/raw/framework_prompts/, read as text only) and the manual labels in
results/realworld_prompts/labels_manual.csv (written and committed before the linter was run on
these prompts, commit 78c2c1b). Writes results/realworld_prompts/lint.json: the findings per
prompt, the confusion counts, precision and recall with Wilson 95% intervals, and the
prevalence of each pattern over prompts and over projects.

Label to linter rule:
- HIGH permission_to_coordinate  vs  contact_shared = yes
- MED missing_boundary_rule      vs  shared_storage = yes and boundary_rule = no
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swarm.fleet.stats import wilson  # noqa: E402
from swarm.swarmcheck.lint import lint  # noqa: E402

PROMPTS = ROOT / "data" / "raw" / "framework_prompts"
RES = ROOT / "results" / "realworld_prompts"


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n) if n else (None, None)
    return {"k": k, "n": n, "p": k / n if n else None, "lo": lo, "hi": hi}


def confusion(pred: list[bool], gold: list[bool]) -> dict:
    tp = sum(p and g for p, g in zip(pred, gold))
    fp = sum(p and not g for p, g in zip(pred, gold))
    fn = sum(g and not p for p, g in zip(pred, gold))
    tn = sum(not p and not g for p, g in zip(pred, gold))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": rate(tp, tp + fp),
            "recall": rate(tp, tp + fn)}


def main() -> None:
    src = {r["file"]: r for r in csv.DictReader((RES / "sources.csv").open())}
    labels = {r["file"]: r for r in csv.DictReader((RES / "labels_manual.csv").open())}
    assert set(src) == set(labels), set(src) ^ set(labels)
    rows = []
    for f, s in src.items():
        text = (PROMPTS / f).read_text(encoding="utf-8")
        fs = lint(text)
        lab = labels[f]
        rows.append({
            "file": f, "project": s["project"], "repo": s["repo"],
            "findings": [{"level": x.level, "rule": x.rule, "evidence": x.evidence} for x in fs],
            "lint_high": any(x.level == "HIGH" for x in fs),
            "lint_missing_boundary": any(x.rule == "missing_boundary_rule" for x in fs),
            "lint_benign": any(x.rule == "benign_team_wording" for x in fs),
            "contact_shared": lab["contact_shared"] == "yes",
            "boundary_rule": lab["boundary_rule"] == "yes",
            "shared_storage": lab["shared_storage"] == "yes",
            "teamwork": lab["teamwork"] == "yes",
            "borderline": lab["borderline"] == "yes",
        })
    for r in rows:
        r["shared_no_boundary"] = r["shared_storage"] and not r["boundary_rule"]

    def conf(pkey, gkey, subset=rows):
        return confusion([r[pkey] for r in subset], [r[gkey] for r in subset])

    clear = [r for r in rows if not r["borderline"]]
    out = {"label": "EXPLORATORY", "n_prompts": len(rows),
           "n_projects": len({r["project"] for r in rows}),
           "high_vs_contact": conf("lint_high", "contact_shared"),
           "high_vs_contact_no_borderline": conf("lint_high", "contact_shared", clear),
           "missing_vs_shared_no_boundary": conf("lint_missing_boundary", "shared_no_boundary"),
           "missing_vs_shared_no_boundary_no_borderline":
               conf("lint_missing_boundary", "shared_no_boundary", clear)}
    keys = ("contact_shared", "boundary_rule", "shared_storage", "shared_no_boundary",
            "teamwork", "lint_high", "lint_missing_boundary", "lint_benign")
    projects = sorted({r["project"] for r in rows})
    out["prevalence_prompts"] = {k: rate(sum(r[k] for r in rows), len(rows)) for k in keys}
    out["prevalence_projects"] = {
        k: rate(sum(any(r[k] for r in rows if r["project"] == p) for p in projects),
                len(projects)) for k in keys}
    out["projects"] = {p: {k: any(r[k] for r in rows if r["project"] == p) for k in keys}
                       for p in projects}
    out["rows"] = rows
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "lint.json").write_text(json.dumps(out, indent=1))

    def fr(d):
        return "n/a" if not d["n"] else f"{d['k']}/{d['n']} = {d['p']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]"

    print(f"EXPLORATORY. {out['n_prompts']} prompts, {out['n_projects']} projects")
    for name in ("high_vs_contact", "high_vs_contact_no_borderline",
                 "missing_vs_shared_no_boundary", "missing_vs_shared_no_boundary_no_borderline"):
        c = out[name]
        print(f"{name}: tp {c['tp']} fp {c['fp']} fn {c['fn']} tn {c['tn']}; precision "
              f"{fr(c['precision'])}; recall {fr(c['recall'])}")
    for k in keys:
        print(f"prevalence {k}: prompts {fr(out['prevalence_prompts'][k])}; projects "
              f"{fr(out['prevalence_projects'][k])}")
    for r in rows:
        flag = "".join(["H" if r["lint_high"] else ".", "M" if r["lint_missing_boundary"] else ".",
                        "|", "C" if r["contact_shared"] else ".",
                        "B" if r["boundary_rule"] else ".", "S" if r["shared_storage"] else "."])
        ev = next((x["evidence"][:90] for x in r["findings"] if x["level"] == "HIGH"), "")
        print(f"{flag} {r['file']:44s} {ev}")


if __name__ == "__main__":
    main()
