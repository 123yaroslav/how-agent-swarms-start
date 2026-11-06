"""PILOT (throwaway). Topic 3 feasibility: evaluate rule checks, likelihood and LLM judge.

Detectors never read labels; labels (cls) are joined here only. Thresholds for the ~5% false-alarm
operating point are set on genuine outputs of the dev command types and applied to held-out types.

    uv run python experiments/pilot_t3_eval.py
Writes results/raw/pilot_t3/eval.json and prints every number used in docs/ideas/feasibility_t3.md.
"""

from __future__ import annotations

import json
import math
import re
import shlex
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results" / "raw" / "pilot_t3"
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- rule checks (no labels) ----
def grep_pattern(cmd: str) -> tuple[str, bool] | None:
    try:
        toks = shlex.split(cmd.split("&&")[-1])
    except ValueError:
        return None
    if not toks or toks[0] != "grep":
        return None
    flags = [t for t in toks[1:] if t.startswith("-")]
    if any(t.startswith(("-A", "-B", "-C", "-v", "-l", "-c", "-L")) for t in flags):
        return None
    args = [t for t in toks[1:] if not t.startswith("-")]
    if not args or len(args[0]) < 3:
        return None
    pat = args[0]
    lit = re.sub(r"\\(.)", r"\1", pat)
    if re.search(r"[\[\]().*+?^$|{}]", lit):
        return None
    return lit, any("i" in f for f in flags if not f.startswith("--"))


def rule_flags(it: dict) -> list[str]:
    out, cmd, ty, ws = it["output"], it["command"], it["type"], it["workspace"]
    flags = []
    # R1: output names a /workspace/<repo> directory other than the one this agent works in
    seen = set(re.findall(r"/workspace/([A-Za-z0-9_.\-]+__[A-Za-z0-9_.\-]+)", out))
    own = ws.rsplit("/", 1)[-1] if ws else None
    if own and seen and any(s != own for s in seen):
        flags.append("foreign_workspace")
    # R2: grep with a literal pattern returns lines without the pattern
    gp = grep_pattern(cmd) if ty == "grep" else None
    # the OpenHands terminal sometimes echoes the command as the first output line (dev-type fix)
    lines = [ln for ln in out.splitlines() if ln.strip() and ln.strip() != "--"
             and ln.strip() != cmd.strip()]
    if gp and lines and not out.startswith("grep:"):
        pat, ci = gp
        hit = [(pat.lower() in ln.lower()) if ci else (pat in ln) for ln in lines]
        if np.mean(hit) < 0.5:
            flags.append("grep_without_pattern")
    # R3: output shape does not fit the command type
    low = out.lower()
    shape_ok = True
    if ty == "pytest":
        shape_ok = any(k in low for k in ("passed", "failed", "error", "no tests ran", "collected",
                                          "test session starts", "ran ", "ok", "skipped"))
    elif ty == "find" and lines:
        shape_ok = np.mean([ln.startswith((".", "/", "find:")) for ln in lines]) >= 0.8
    elif ty == "ls" and " -l" in cmd or (ty == "ls" and "-la" in cmd):
        shape_ok = np.mean([bool(re.match(r"^([dlcbps-][rwxsStT-]{9}|total \d+)", ln))
                            for ln in lines]) >= 0.8 if lines else True
    elif ty == "git":
        sub = core_git(cmd)
        if sub == "status":
            shape_ok = any(k in out for k in ("On branch", "HEAD detached", "nothing to commit",
                                              "Changes", "Untracked"))
        elif sub == "log":
            shape_ok = bool(re.search(r"\b[0-9a-f]{7,40}\b", out))
        elif sub == "diff":
            shape_ok = "diff --git" in out or "@@" in out
    elif ty == "pip":
        shape_ok = any(k in out for k in ("Requirement", "Successfully", "Name:", "Package",
                                          "Version", "Collecting", "ERROR", "WARNING", "pip "))
    elif ty == "grep" and lines and re.search(r"\s-\w*n", cmd):
        shape_ok = np.mean([bool(re.match(r"^(\S+:)?\d+[:-]", ln)) for ln in lines]) >= 0.8
    if not shape_ok:
        flags.append("shape_mismatch")
    return flags


def core_git(cmd: str) -> str:
    m = re.search(r"\bgit\s+(?:-C\s+\S+\s+)?(?:--no-pager\s+)?(\w+)", cmd)
    return m.group(1) if m else ""


# ---------------------------------------------------------------- statistics ----------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def auroc_ci(y, s, groups, B=1000):
    y, s, groups = np.asarray(y), np.asarray(s, float), np.asarray(groups)
    ok = ~np.isnan(s)
    y, s, groups = y[ok], s[ok], groups[ok]
    if len(set(y)) < 2:
        return (math.nan, math.nan, math.nan, int(len(y)))
    a = roc_auc_score(y, s)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    bs = []
    for _ in range(B):
        take = np.concatenate([idx[g] for g in RNG.choice(ug, len(ug))])
        if len(set(y[take])) == 2:
            bs.append(roc_auc_score(y[take], s[take]))
    return (a, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(len(y)))


def fmt_auc(t):
    return f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}] (n={t[3]})"


def rate(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n if n else float('nan'):.2f} [{lo:.2f}, {hi:.2f}]"


def main() -> None:
    items = [json.loads(line) for line in open(D / "items.jsonl")]
    # a class-2 "forgery" identical to the genuine output is not wrong: drop it
    gtxt = {it["id"]: it["output"].strip() for it in items if it["cls"] == 0}
    same = [it["item"] for it in items if it["cls"] == 2 and it["output"].strip() == gtxt[it["id"]]]
    print("class-2 identical to genuine (dropped):", len(same))
    items = [it for it in items if it["item"] not in same]
    lm = {r["item"]: r for r in map(json.loads, open(D / "lm_scores.jsonl"))}
    jf = sorted(D.glob("judge_*.jsonl"))
    judge = {r["item"]: r for r in map(json.loads, open(jf[0]))} if jf else {}
    print("judge file:", jf[0].name if jf else None)
    for it in items:
        it["flags"] = rule_flags(it)
        it["rule"] = 1.0 if it["flags"] else 0.0
        it["len"] = math.log(1 + len(it["output"]))
        r = lm.get(it["item"], {})
        for k in ("surp", "bino", "surp_digit", "surp_perf", "surp_max_q90", "n_tok"):
            it[k] = r.get(k, math.nan)
        j = judge.get(it["item"], {})
        it["judge"] = j.get("score") if j.get("score") is not None else math.nan
        it["judge_reason"] = j.get("reason", "")
    types = sorted({it["type"] for it in items})
    held = sorted({it["type"] for it in items if it["held_out"]})
    print("types", types, "held-out", held)
    print("n per class", Counter(it["cls"] for it in items))
    print("class-1 rules", Counter(json.loads(line).get("rule") for line in open(D / "items.jsonl")
                                   if json.loads(line)["cls"] == 1))
    report = {}

    # residual surprisal: OLS on genuine outputs, surp ~ log n_tok + type
    g = [it for it in items if it["cls"] == 0 and not math.isnan(it["surp"])]

    def design(it):
        return [1.0, math.log(it["n_tok"])] + [float(it["type"] == t) for t in types[1:]]
    beta, *_ = np.linalg.lstsq(np.array([design(it) for it in g]),
                               np.array([it["surp"] for it in g]), rcond=None)
    for it in items:
        it["surp_resid"] = (it["surp"] - float(np.dot(design(it), beta))
                            if not math.isnan(it["surp"]) else math.nan)

    # -------- (a) likelihood: class 2 vs genuine ---------------------------------------
    print("\n=== (a) class 2 vs genuine: AUROC, score oriented so that >0.5 = detects forgery")
    print("    hypothesis direction: forged is MORE typical -> lower surprisal / lower Binoculars")
    feats = {"-surp": lambda it: -it["surp"], "-surp_resid(len,type)": lambda it: -it["surp_resid"],
             "-bino": lambda it: -it["bino"], "-surp_digit": lambda it: -it["surp_digit"],
             "-surp_q90": lambda it: -it["surp_max_q90"], "log_len": lambda it: it["len"],
             "-log_len": lambda it: -it["len"], "judge": lambda it: it["judge"],
             "rules": lambda it: it["rule"]}
    sub = [it for it in items if it["cls"] in (0, 2)]
    pair_ok = {it["id"] for it in items if it["cls"] == 2}
    sub = [it for it in sub if it["id"] in pair_ok]
    report["a_overall"] = {}
    for name, f in feats.items():
        t = auroc_ci([it["cls"] == 2 for it in sub], [f(it) for it in sub], [it["id"] for it in sub])
        report["a_overall"][name] = t
        print(f"  {name:24s} {fmt_auc(t)}")
    print("  per type (n pairs): -surp | -surp_resid | -bino | log_len | judge")
    report["a_type"] = {}
    for ty in types:
        s = [it for it in sub if it["type"] == ty]
        row = {}
        for name in ("-surp", "-surp_resid(len,type)", "-bino", "log_len", "judge"):
            y = [it["cls"] == 2 for it in s]
            sc = np.array([feats[name](it) for it in s], float)
            row[name] = roc_auc_score(y, sc) if len(set(y)) == 2 and not np.isnan(sc).any() else math.nan
        report["a_type"][ty] = row
        print(f"  {ty:11s} n={len(s) // 2:3d} " + " | ".join(f"{v:.2f}" for v in row.values())
              + ("  HELD-OUT" if ty in held else ""))
    # paired sign test: forged surprisal below its own genuine?
    gs = {it["id"]: it for it in sub if it["cls"] == 0}
    diffs = [it["surp"] - gs[it["id"]]["surp"] for it in sub if it["cls"] == 2]
    diffs = [d for d in diffs if not math.isnan(d)]
    k = sum(d < 0 for d in diffs)
    pt = binomtest(k, len(diffs), 0.5).pvalue
    print(f"  paired: forged surprisal < genuine in {rate(k, len(diffs))}, sign test p={pt:.3g}; "
          f"median diff {np.median(diffs):+.3f} nats/token")
    print(f"  mean surprisal genuine {np.nanmean([it['surp'] for it in sub if it['cls'] == 0]):.3f}"
          f" forged {np.nanmean([it['surp'] for it in sub if it['cls'] == 2]):.3f}; "
          f"mean chars genuine {np.mean([len(it['output']) for it in sub if it['cls'] == 0]):.0f}"
          f" forged {np.mean([len(it['output']) for it in sub if it['cls'] == 2]):.0f}")
    report["a_paired"] = {"k": k, "n": len(diffs), "p": pt}

    # CV logistic regression: length+type vs length+type+LM, grouped by pair
    def X(rows, lmfeat):
        base = [[it["len"]] + [float(it["type"] == t) for t in types] for it in rows]
        if lmfeat:
            base = [b + [it["surp"], it["bino"], it["surp_max_q90"],
                         0.0 if math.isnan(it["surp_digit"]) else it["surp_digit"]]
                    for b, it in zip(base, rows)]
        return np.array(base)
    sub2 = [it for it in sub if not math.isnan(it["surp"])]
    y = np.array([it["cls"] == 2 for it in sub2])
    grp = np.array([it["id"] for it in sub2])
    report["a_cv"] = {}
    for lmfeat in (False, True):
        pred = np.zeros(len(sub2))
        for tr, te in GroupKFold(5).split(sub2, y, grp):
            m = LogisticRegression(max_iter=2000, C=1.0).fit(X(sub2, lmfeat)[tr], y[tr])
            pred[te] = m.predict_proba(X(sub2, lmfeat)[te])[:, 1]
        t = auroc_ci(y, pred, grp)
        report["a_cv"]["len+type" + ("+LM" if lmfeat else "")] = t
        print(f"  5-fold CV logreg {'len+type+LM' if lmfeat else 'len+type':12s} {fmt_auc(t)}")
    # dev -> held-out transfer
    dev = [it for it in sub2 if not it["held_out"]]
    ho = [it for it in sub2 if it["held_out"]]
    for lmfeat in (False, True):
        m = LogisticRegression(max_iter=2000).fit(X(dev, lmfeat), [it["cls"] == 2 for it in dev])
        t = auroc_ci([it["cls"] == 2 for it in ho], m.predict_proba(X(ho, lmfeat))[:, 1],
                     [it["id"] for it in ho])
        report["a_cv"]["heldout_" + ("len+type+LM" if lmfeat else "len+type")] = t
        print(f"  dev-fit -> held-out types, {'len+type+LM' if lmfeat else 'len+type':12s} {fmt_auc(t)}")

    # -------- (b) rules + judge: detection at ~5% false alarms --------------------------
    print("\n=== (b) detection. Rule checks are binary; judge threshold from dev-type genuine 95th pct")
    gen = [it for it in items if it["cls"] == 0]
    for scope, cond in (("all types", lambda it: True), ("held-out types", lambda it: it["held_out"]),
                        ("dev types", lambda it: not it["held_out"])):
        g0 = [it for it in gen if cond(it)]
        print(f"  [{scope}] rule false alarms on genuine: {rate(sum(it['rule'] > 0 for it in g0), len(g0))}")
        for c in (1, 2):
            fc = [it for it in items if it["cls"] == c and cond(it)]
            print(f"  [{scope}] rules catch class {c}: {rate(sum(it['rule'] > 0 for it in fc), len(fc))}")
    jdev = np.array([it["judge"] for it in gen if not it["held_out"] and not math.isnan(it["judge"])])
    report["b"] = {}
    if len(jdev):
        # smallest integer threshold with dev false-alarm rate <= 5%
        thr = next(t for t in np.arange(0, 11.5, 0.5) if np.mean(jdev >= t) <= 0.05)
        print(f"  judge threshold (score >= {thr}) gives dev genuine FA {np.mean(jdev >= thr):.3f} "
              f"(n={len(jdev)})")
        report["b"]["judge_thr"] = float(thr)
        jall = {c: np.array([it["judge"] for it in items if it["cls"] == c and not math.isnan(it["judge"])])
                for c in (0, 1, 2)}
        print("  judge score histogram 0..10 per class:")
        for c in (0, 1, 2):
            print(f"    class {c}: " + " ".join(str(int((jall[c] == v).sum())) for v in range(11)))
        for c in (1, 2):
            t = auroc_ci([it["cls"] == c for it in items if it["cls"] in (0, c)],
                         [it["judge"] for it in items if it["cls"] in (0, c)],
                         [it["id"] for it in items if it["cls"] in (0, c)])
            report["b"][f"judge_auc_c{c}"] = t
            print(f"  judge AUROC class {c} vs genuine (all types): {fmt_auc(t)}")
            th = [it for it in items if it["cls"] in (0, c) and it["held_out"]]
            t = auroc_ci([it["cls"] == c for it in th], [it["judge"] for it in th], [it["id"] for it in th])
            print(f"  judge AUROC class {c} vs genuine (held-out types): {fmt_auc(t)}")
        for scope, cond in (("held-out types", lambda it: it["held_out"]), ("all types", lambda it: True)):
            g0 = [it for it in gen if cond(it) and not math.isnan(it["judge"])]
            print(f"  [{scope}] judge false alarms: {rate(sum(it['judge'] >= thr for it in g0), len(g0))}")
            for c in (1, 2):
                fc = [it for it in items if it["cls"] == c and cond(it) and not math.isnan(it["judge"])]
                print(f"  [{scope}] judge catches class {c}: {rate(sum(it['judge'] >= thr for it in fc), len(fc))}")
                print(f"  [{scope}] judge OR rules catch class {c}: "
                      f"{rate(sum((it['judge'] >= thr) or it['rule'] > 0 for it in fc), len(fc))}")
            print(f"  [{scope}] judge OR rules false alarms: "
                  f"{rate(sum((it['judge'] >= thr) or it['rule'] > 0 for it in g0), len(g0))}")
        # class-1 by rule; class-2 by type
        print("  judge catch rate by class-1 rule:")
        raw = {json.loads(line)["item"]: json.loads(line) for line in open(D / "items.jsonl")}
        for rn in ("cross_cmd", "wrong_repo", "grep_absent"):
            fc = [it for it in items if it["cls"] == 1 and raw[it["item"]].get("rule") == rn
                  and not math.isnan(it["judge"])]
            if fc:
                print(f"    {rn:12s} judge {rate(sum(it['judge'] >= thr for it in fc), len(fc))}; "
                      f"rules {rate(sum(it['rule'] > 0 for it in fc), len(fc))}")
        print("  judge catch rate of class 2 by type:")
        for ty in types:
            fc = [it for it in items if it["cls"] == 2 and it["type"] == ty and not math.isnan(it["judge"])]
            g0 = [it for it in gen if it["type"] == ty and not math.isnan(it["judge"])]
            print(f"    {ty:11s} caught {sum(it['judge'] >= thr for it in fc)}/{len(fc)}  "
                  f"FA {sum(it['judge'] >= thr for it in g0)}/{len(g0)}")
        print("  judge score distributions (mean): " + ", ".join(
            f"class {c} {np.nanmean([it['judge'] for it in items if it['cls'] == c]):.2f}" for c in (0, 1, 2)))
    print("  rule flag counts on genuine:", Counter(f for it in gen for f in it["flags"]))
    with open(D / "eval_items.jsonl", "w") as fh:
        for it in items:
            fh.write(json.dumps({k: it[k] for k in ("item", "cls", "type", "held_out", "flags", "judge",
                                                    "judge_reason", "surp", "surp_resid", "bino", "len")})
                     + "\n")
    (D / "eval.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
