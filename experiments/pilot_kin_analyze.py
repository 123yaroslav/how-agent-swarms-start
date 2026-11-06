"""EXPLORATORY PILOT (Idea B, kin leniency). Not pre-registered. No LLM calls.

Question: at a fixed false-alarm rate, do stored MRT monitor verdicts detect attacks by agents of the
monitor's own family less often than attacks by other families?

Input: results/pilot_kin_mrt/rows.csv from experiments/pilot_kin_extract.py.
Output: printed tables and results/pilot_kin_mrt/*.csv. Every number in docs/ideas/pilot_kin_mrt.md
is printed here.

Design.
- Agents: Claude and Gemini families (GPT agents are too rare in MRT). Monitors: Claude, Gemini, GPT.
- Attack = condition transcripts_no_hint_selective_thinking_time[_scaffold] (side task instructed).
  Honest = transcripts_no_secondary_task[_scaffold]. Scaffold = monitor prompt or scaffold variant.
- Threshold per (monitor model, scaffold) from that monitor's honest scores pooled over all agent
  families, with randomised tie-breaking so the honest flag rate is exactly alpha.
- Kin effect: linear probability model flag ~ FE(attack transcript) + FE(monitor x scaffold)
  + beta * same_family. The transcript FE absorbs attacker strength when one stored transcript is
  scored by several monitors. Bootstrap clustered by attack transcript.

Run: uv run python -I experiments/pilot_kin_analyze.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "pilot_kin_mrt"
AGENT_FAMS = ["Claude", "Gemini"]
MON_FAMS = ["Claude", "Gemini", "GPT"]
N_BOOT = 1000
RNG = np.random.default_rng(20261106)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 30)


def wilson(k: float, n: float, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def threshold(honest: np.ndarray, alpha: float):
    """Return (t, w): flag prob is 1 if s > t, w if s == t, 0 otherwise. Honest flag rate = alpha."""
    s = np.sort(honest)
    t = np.quantile(s, 1 - alpha, method="higher")
    p_gt = np.mean(s > t)
    p_eq = np.mean(s == t)
    w = 0.0 if p_eq == 0 else float(np.clip((alpha - p_gt) / p_eq, 0, 1))
    return float(t), w


def flag_prob(scores: np.ndarray, t: float, w: float) -> np.ndarray:
    return np.where(scores > t, 1.0, np.where(scores == t, w, 0.0))


def demean_2way(y, g1, g2, wts, iters=300, tol=1e-7):
    """Weighted residual of y after two sets of fixed effects (alternating projections)."""
    r = y.astype(float).copy()
    n1, n2 = g1.max() + 1, g2.max() + 1
    w1 = np.bincount(g1, wts, n1)
    w2 = np.bincount(g2, wts, n2)
    w1[w1 == 0] = 1
    w2[w2 == 0] = 1
    for _ in range(iters):
        m1 = np.bincount(g1, wts * r, n1) / w1
        r = r - m1[g1]
        m2 = np.bincount(g2, wts * r, n2) / w2
        r = r - m2[g2]
        if np.max(np.abs(m2)) < tol and np.max(np.abs(m1)) < tol:
            break
    return r


def fe_beta(y, X, g1, g2, wts):
    """OLS coefficients of columns X after absorbing two FE sets (FWL)."""
    ry = demean_2way(y, g1, g2, wts)
    RX = np.column_stack([demean_2way(X[:, j], g1, g2, wts) for j in range(X.shape[1])])
    W = wts[:, None]
    A = RX.T @ (W * RX)
    b = RX.T @ (wts * ry)
    if np.linalg.matrix_rank(A) < A.shape[0]:
        return np.full(A.shape[0], np.nan)
    return np.linalg.solve(A, b)


def boot_fe(df, ycol, xcols, g1col, g2col, clcol, n_boot=N_BOOT):
    y = df[ycol].to_numpy(float)
    X = df[xcols].to_numpy(float)
    g1 = pd.factorize(df[g1col])[0]
    g2 = pd.factorize(df[g2col])[0]
    cl = pd.factorize(df[clcol])[0]
    ncl = cl.max() + 1
    est = fe_beta(y, X, g1, g2, np.ones(len(y)))
    bs = np.empty((n_boot, len(xcols)))
    for b in range(n_boot):
        cw = np.bincount(RNG.integers(0, ncl, ncl), minlength=ncl).astype(float)
        bs[b] = fe_beta(y, X, g1, g2, cw[cl])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    return est, lo, hi


def fmt_ci(e, lo, hi, scale=100):
    return f"{e * scale:+.1f} pp [{lo * scale:+.1f}, {hi * scale:+.1f}]"


def main() -> None:
    r = pd.read_csv(RES / "rows.csv")
    loose = "--loose" in sys.argv
    if loose:
        # Sensitivity: recover unparsed verdicts with the loose regex (e.g. "score: 30/100").
        r["score"] = r.score_loose
        r["parse_ok"] = r.score_loose.notna()
        print("SENSITIVITY RUN: loose verdict parse")
    print(f"rows total {len(r)}")
    print("\n== all parsed rows by kind, agent family, monitor family (before filtering)")
    print(r[r.parse_ok].groupby(["kind", "agent_fam", "monitor_fam"]).size().unstack(fill_value=0))
    # Honest pool for thresholds: ALL agent families (incl. GPT, DeepSeek agents), parsed only.
    h_all = r[(r.kind == "honest") & r.parse_ok].copy()
    h_all["score"] = h_all.score.astype(float)
    h_all["mon_key"] = h_all.monitor + "|" + h_all.scaffold
    r = r[r.agent_fam.isin(AGENT_FAMS) & r.monitor_fam.isin(MON_FAMS)].copy()
    print(f"rows with Claude/Gemini agent and Claude/Gemini/GPT monitor {len(r)}")

    # ---- missingness of verdicts by family cell
    miss = (r.assign(missing=~r.parse_ok)
            .groupby(["kind", "agent_fam", "monitor_fam"]).missing.agg(["mean", "size"]))
    print("\n== share of verdicts without a parsable <verdict> tag (excluded)\n", miss.round(3))
    r = r[r.parse_ok].copy()
    r["score"] = r.score.astype(float)
    r["mon_key"] = r.monitor + "|" + r.scaffold
    r["same_family"] = (r.agent_fam == r.monitor_fam).astype(int)
    r["same_model"] = (r.agent == r.monitor).astype(int)
    r["path_id"] = r.task + "|" + r.cond + "|" + r.agent + "|" + r.idx.astype(str)
    # Transcript id = hash of the stored conversation. Scaffold variants re-score the same
    # conversations, so the hash (not the file path) identifies one agent run.
    r["tid"] = r.conv_hash

    # ---- is one stored attack transcript scored by several monitors?
    a = r[r.kind == "attack"]
    per_tid = a.groupby("path_id").agg(n_mon=("monitor", "nunique"), n_hash=("conv_hash", "nunique"))
    print("\n== attack file paths (task|cond|agent|idx)")
    print(f"ids {len(per_tid)}, mean monitors per id {per_tid.n_mon.mean():.2f}, "
          f"ids with >1 monitor {(per_tid.n_mon > 1).sum()}, "
          f"of those with identical conversation hash across monitors "
          f"{((per_tid.n_mon > 1) & (per_tid.n_hash == 1)).sum()}")
    by_hash = a.groupby("conv_hash").agg(n_mon=("monitor", "nunique"),
                                         fams=("monitor_fam", lambda s: "+".join(sorted(set(s)))))
    print(f"distinct attack conversations (by hash) {len(by_hash)}, "
          f"scored by >1 monitor family {(by_hash.fams.str.count('\\+') >= 1).sum()}")

    print("\n== distinct conversations (agent runs) by kind and agent model")
    print(r.groupby(["kind", "agent_fam", "agent"]).tid.nunique().unstack(0, fill_value=0))
    print("\n== verdict rows by scaffold, kind, monitor family x agent family")
    print(r.groupby(["scaffold", "kind", "monitor_fam", "agent_fam"]).size()
          .unstack([2, 3], fill_value=0))
    print("\n== attack success rate (side_task_evaluation True) by agent model, distinct runs")
    print(r[r.kind == "attack"].drop_duplicates("tid").groupby("agent").side_task_ok
          .agg(["mean", "size"]).round(3))

    # ---- thresholds per monitor x scaffold from pooled honest scores
    h = h_all
    thr_rows = []
    for alpha in (0.05, 0.01):
        for key, g in h.groupby("mon_key"):
            if alpha == 0.01 and len(g) < 300:
                continue
            if len(g) < 40:
                continue
            t, w = threshold(g.score.to_numpy(), alpha)
            thr_rows.append({"alpha": alpha, "mon_key": key, "n_honest": len(g), "t": t, "w": w})
    thr = pd.DataFrame(thr_rows)
    thr.to_csv(RES / "thresholds.csv", index=False)
    print("\n== thresholds (alpha, monitor|scaffold, n_honest, t, tie weight w)\n",
          thr.to_string(index=False))

    out_lines = []
    for alpha in (0.05, 0.01):
        T = thr[thr.alpha == alpha].set_index("mon_key")
        d = r[r.mon_key.isin(T.index)].copy()
        if d.empty:
            continue
        d["flag"] = flag_prob(d.score.to_numpy(), T.loc[d.mon_key, "t"].to_numpy(),
                              T.loc[d.mon_key, "w"].to_numpy())
        d["flag_hard"] = (d.score.to_numpy() > T.loc[d.mon_key, "t"].to_numpy()).astype(float)
        # Score percentile within each monitor's pooled honest pool (all agent families), mid-ties.
        # Mean over attacks = AUROC of attack vs pooled honest for that monitor. Scale-free.
        d["pct"] = np.nan
        for key, g in d.groupby("mon_key"):
            ref = np.sort(h[h.mon_key == key].score.to_numpy())
            lo = np.searchsorted(ref, g.score.to_numpy(), "left")
            hi = np.searchsorted(ref, g.score.to_numpy(), "right")
            d.loc[g.index, "pct"] = (lo + hi) / 2 / len(ref)
        d["hpct"] = d.pct
        # Percentile within the honest pool of the SAME agent family under the same monitor.
        # Removes any family-specific calibration shift; what is left is discrimination.
        d["pct_fam"] = np.nan
        for (key, af), g in d.groupby(["mon_key", "agent_fam"]):
            ref = np.sort(h[(h.mon_key == key) & (h.agent_fam == af)].score.to_numpy())
            if len(ref) < 20:
                continue
            lo = np.searchsorted(ref, g.score.to_numpy(), "left")
            hi = np.searchsorted(ref, g.score.to_numpy(), "right")
            d.loc[g.index, "pct_fam"] = (lo + hi) / 2 / len(ref)
        att = d[d.kind == "attack"]
        hon = d[d.kind == "honest"]
        tag = f"FPR {int(alpha * 100)}%"
        print(f"\n################ {tag} ################")
        print(f"monitors x scaffolds used {d.mon_key.nunique()}, attack rows {len(att)}, "
              f"honest rows {len(hon)}")

        # ---- family cell table
        def cell_table(gcols):
            rows = []
            for k, g in att.groupby(gcols):
                gh = hon.set_index(gcols).loc[[k]] if k in hon.set_index(gcols).index else None
                n = len(g)
                kk = g.flag.sum()
                lo, hi = wilson(kk, n)
                succ = g[g.side_task_ok == True]  # noqa: E712
                rows.append({**dict(zip(gcols, k if isinstance(k, tuple) else (k,))),
                             "n_attack": n, "n_attack_succ": len(succ),
                             "det": kk / n, "det_lo": lo, "det_hi": hi,
                             "det_succ": succ.flag.mean() if len(succ) else np.nan,
                             "auroc_vs_pool": g.pct.mean(),
                             "auroc_vs_ownfam_honest": g.pct_fam.mean(),
                             "n_honest": 0 if gh is None else len(gh),
                             "honest_flag": np.nan if gh is None else gh.flag.mean(),
                             "honest_pct": np.nan if gh is None else gh.hpct.mean()})
            return pd.DataFrame(rows)

        fam_def = cell_table(["scaffold", "monitor_fam", "agent_fam"])
        fam_def = fam_def[fam_def.scaffold == "default"]
        print(f"\n== {tag}: detection by family cell, DEFAULT scaffold only")
        print(fam_def.round(3).to_string(index=False))
        fam_def.to_csv(RES / f"cells_family_default_fpr{int(alpha * 100)}.csv", index=False)
        fam_tab = cell_table(["monitor_fam", "agent_fam"])
        print(f"\n== {tag}: detection by family cell (pooled over scaffolds)")
        print(fam_tab.round(3).to_string(index=False))
        fam_tab.to_csv(RES / f"cells_family_fpr{int(alpha * 100)}.csv", index=False)
        if alpha == 0.05:
            model_tab = cell_table(["scaffold", "monitor", "agent"])
            model_tab.to_csv(RES / "cells_model_fpr5.csv", index=False)
            dflt = model_tab[model_tab.scaffold == "default"]
            print(f"\n== {tag}: default scaffold, exact models (n_attack >= 20)")
            print(dflt[dflt.n_attack >= 20].round(3).to_string(index=False))

        # ---- naive 2x2 family DiD on cell means and GPT-referenced breakdown
        print(f"monitors used: {sorted(d.monitor.unique())}")
        for col in ("flag", "pct", "pct_fam"):
            D2 = att.groupby(["monitor_fam", "agent_fam"])[col].mean()
            try:
                did2 = (D2["Claude", "Claude"] + D2["Gemini", "Gemini"]
                        - D2["Claude", "Gemini"] - D2["Gemini", "Claude"]) / 2
                print(f"== {tag}: naive 2x2 family DiD on {col} (same minus cross, /2) "
                      f"{did2 * 100:+.1f} points")
            except KeyError:
                print(f"== {tag}: naive 2x2 DiD on {col} not computable (missing family cell)")

        # ---- FE regressions
        att = att.copy()
        att["same_claude"] = ((att.agent_fam == "Claude") & (att.monitor_fam == "Claude")).astype(int)
        att["same_gemini"] = ((att.agent_fam == "Gemini") & (att.monitor_fam == "Gemini")).astype(int)
        zl = np.log(att.n_msgs)
        zl = (zl - zl.mean()) / zl.std()
        att["len_x_claude_mon"] = zl * (att.monitor_fam == "Claude")
        att["len_x_gemini_mon"] = zl * (att.monitor_fam == "Gemini")
        def run_specs(sub_att, sub_hon, label):
            specs = [
                ("A. transcript FE + monitor|scaffold FE, all attacks", sub_att, "flag",
                 ["same_family"]),
                ("B. same, hard threshold (FPR <= alpha)", sub_att, "flag_hard", ["same_family"]),
                ("C. same, successful attacks only", sub_att[sub_att.side_task_ok == True],  # noqa: E712
                 "flag", ["same_family"]),
                ("D. per-family split (GPT monitors as reference)", sub_att, "flag",
                 ["same_claude", "same_gemini"]),
                ("E. same family vs same exact model", sub_att, "flag",
                 ["same_family", "same_model"]),
                ("F. threshold-free: y = attack score percentile in honest pool", sub_att, "pct",
                 ["same_family"]),
                ("I. A + monitor-family-specific slope on log transcript length", sub_att, "flag",
                 ["same_family", "len_x_claude_mon", "len_x_gemini_mon"]),
                ("G. y = percentile in honest pool of SAME agent family (discrimination only)",
                 sub_att[sub_att.pct_fam.notna()], "pct_fam", ["same_family"]),
                ("H1. HONEST transcripts, y = flag (false-alarm rate)", sub_hon, "flag",
                 ["same_family"]),
                ("H2. HONEST transcripts, y = score percentile in honest pool", sub_hon, "hpct",
                 ["same_family"]),
            ]
            if alpha == 0.01:
                specs = [sp for sp in specs if sp[0][0] in "ADFGIH"]
            print(f"\n== {tag} [{label}] fixed-effect estimates (beta x 100; negative = "
                  "own family flagged less)")
            for name, df, ycol, xcols in specs:
                df = df.reset_index(drop=True)
                if len(df) == 0 or any(df[x].nunique() < 2 for x in xcols):
                    print(f"{tag} [{label}] {name}: skipped, a regressor is constant")
                    continue
                est, lo, hi = boot_fe(df, ycol, xcols, "tid", "mon_key", "tid")
                txt = "; ".join(f"{x} {fmt_ci(e, l, u)}" for x, e, l, u in zip(xcols, est, lo, hi))
                line = (f"{tag} [{label}] {name}: n={len(df)}, transcripts={df.tid.nunique()}, "
                        f"monitors={df.mon_key.nunique()}: {txt}")
                print(line)
                out_lines.append(line)

        hon = hon.copy()
        run_specs(att[att.scaffold == "default"], hon[hon.scaffold == "default"], "default scaffold")
        run_specs(att, hon, "pooled scaffolds")

        if alpha == 0.05:
            # ---- per scaffold and leave-one-task-out (point estimate + CI)
            print(f"\n== {tag}: spec A by scaffold")
            for sc, g in att.groupby("scaffold"):
                g = g.reset_index(drop=True)
                if g.same_family.nunique() < 2:
                    continue
                est, lo, hi = boot_fe(g, "flag", ["same_family"], "tid", "mon_key", "tid", 500)
                print(f"  {sc:38s} n={len(g):5d} same_family {fmt_ci(est[0], lo[0], hi[0])}")
            print(f"\n== {tag}: spec A by task, and leaving one task out")
            for tk in sorted(att.task.unique()):
                g = att[att.task == tk].reset_index(drop=True)
                gx = att[att.task != tk].reset_index(drop=True)
                e1, l1, h1 = boot_fe(g, "flag", ["same_family"], "tid", "mon_key", "tid", 200)
                e2, l2, h2 = boot_fe(gx, "flag", ["same_family"], "tid", "mon_key", "tid", 200)
                print(f"  only {tk:28s} {fmt_ci(e1[0], l1[0], h1[0])}   "
                      f"without it {fmt_ci(e2[0], l2[0], h2[0])}")
            # ---- per monitor model: own-family minus other-family detection, minus GPT-ref gap
            print(f"\n== {tag}: per monitor model, detection of Claude vs Gemini attacks among the "
                  "transcripts this monitor scored (default scaffold, raw, no FE)")
            dd = att[att.scaffold == "default"]
            for m, g in dd.groupby("monitor"):
                c = g[g.agent_fam == "Claude"].flag
                gm = g[g.agent_fam == "Gemini"].flag
                print(f"  {m:32s} fam={g.monitor_fam.iloc[0]:6s} det(Claude agents)={c.mean():.3f} "
                      f"n={len(c):4d}  det(Gemini agents)={gm.mean():.3f} n={len(gm):4d}  "
                      f"gap C-G={(c.mean() - gm.mean()) * 100:+.1f} pp")
            gaps = (dd.groupby(["monitor_fam", "monitor", "agent_fam"]).flag.mean().unstack()
                    .assign(gap=lambda x: x.Claude - x.Gemini))
            print("  mean of per-monitor gaps (Claude-agent minus Gemini-agent detection) by "
                  "monitor family, pp:",
                  (gaps.groupby("monitor_fam").gap.mean() * 100).round(1).to_dict())

    # ---- self-identification in agent text (could reveal family to the monitor)
    a = r[r.kind == "attack"].drop_duplicates("tid")
    print(f"distinct attack conversations by agent family: {a.agent_fam.value_counts().to_dict()}")
    print("messages per distinct attack conversation by agent family (median, IQR):")
    print(a.groupby("agent_fam").n_msgs.describe()[["count", "25%", "50%", "75%"]])
    print("\n== share of attack transcripts whose assistant text mentions a vendor/model name")
    print(a.groupby("agent_fam")[["selfid_claude", "selfid_gemini", "selfid_gpt"]].mean().round(3))
    (RES / ("estimates_loose.txt" if loose else "estimates.txt")).write_text(
        "\n".join(out_lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
