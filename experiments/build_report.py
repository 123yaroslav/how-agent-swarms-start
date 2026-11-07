"""Build report/numbers.json and the figures from the confirmatory results. Safe on partial data.

    uv run python experiments/build_report.py

Inputs: results/confirmatory/c1c3.json (single-agent cells per model), results/confirmatory/c4*.json
and results/predictions/*.json (fleets and their frozen predictions), when present.
Outputs: report/numbers.json, report/figures/fig1_gate.{pdf,png}, fig2_fleets.{pdf,png},
fig3_freeride.{pdf,png}. Every number the report text uses is a key in numbers.json.
"""

from __future__ import annotations

import glob
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "results" / "confirmatory"
FIG = ROOT / "report" / "figures"
STYLE = Path.home() / ".claude/skills/ml-paper-writing/scripts/paper.mplstyle"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
INK, MUTED = "#2b2b29", "#6b6b66"
SHORT = {"deepseek-v4-flash": "DeepSeek V4 Flash", "gemma-4-31b-it": "Gemma 4 31B",
         "gemma-4-26b-a4b-it": "Gemma 4 26B-A4B", "qwen3.8-27b": "Qwen 3.8 27B",
         "glm-5.2": "GLM 5.2", "qwen3.5-397b": "Qwen 3.5 397B"}
CUES = ["cue_shared_fact", "cue_peers_have", "cue_team_culture", "cue_multiagent",
        "collab_line", "persist"]


def pct(x: float | None, d: int = 0) -> str:
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def cell(c: dict, model: str, arm: str, key: str) -> dict | None:
    v = c.get(model, {}).get(arm, {}).get(key)
    return v if v and v.get("n") else None


def pooled(c: dict, arm: str, key: str) -> tuple[int, int]:
    k = n = 0
    for m in c:
        v = cell(c, m, arm, key)
        if v:
            k += v["k"]
            n += v["n"]
    return k, n


def fig_gate(c: dict, models: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 2.7))
    xs = range(len(models))
    w = 0.34
    for i, m in enumerate(models):
        for dx, arm, col, lab in [(-w / 2, "plain-rule", BLUE, "Plain prompt"),
                                  (w / 2, "coordinate_ok-rule", ORANGE, "One permission sentence")]:
            v = cell(c, m, arm, "note_addressed")
            if not v:
                continue
            ax.bar(i + dx, v["p"], w * 0.92, color=col, label=lab if i == 0 else None,
                   linewidth=0)
            ax.errorbar(i + dx, v["p"], yerr=[[v["p"] - v["lo"]], [v["hi"] - v["p"]]],
                        color=INK, lw=1, capsize=2)
        for arm in CUES:
            v = cell(c, m, arm + "-rule", "note_addressed")
            if v:
                ax.plot(i - w / 2, v["p"], "o", ms=3.5, color=GREY, zorder=3,
                        label="Six deployment phrasings" if (i == 0 and arm == CUES[0]) else None)
    ax.set_xticks(list(xs), [SHORT.get(m, m) for m in models], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Blocked agents writing\nto other jobs", fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig1_gate.{ext}", dpi=300)
    plt.close(fig)


def fig_freeride(c: dict, models: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 2.3))
    w = 0.34
    for i, m in enumerate(models):
        for dx, arm, col, lab in [(-w / 2, "plain-rule", BLUE, "With the house rule"),
                                  (w / 2, "plain-norule", AQUA, "Without the rule")]:
            v = cell(c, m, arm, "use_get")
            if not v:
                continue
            ax.bar(i + dx, v["p"], w * 0.92, color=col, label=lab if i == 0 else None)
            ax.errorbar(i + dx, v["p"], yerr=[[v["p"] - v["lo"]], [v["hi"] - v["p"]]],
                        color=INK, lw=1, capsize=2)
    ax.set_xticks(range(len(models)), [SHORT.get(m, m) for m in models], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Agents using other\njobs' artifacts", fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig3_freeride.{ext}", dpi=300)
    plt.close(fig)


def fig_fleets(fleets: list[dict]) -> None:
    if not fleets:
        return
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for f in fleets:
        r = max(f.get("R_pred", 0.0), 1e-3)
        board = bool(f.get("board"))
        ax.scatter(r, f.get("fleet", 0), s=36, marker="o",
                   facecolor=ORANGE if board else "white", edgecolor=ORANGE if board else BLUE,
                   linewidth=1.5, zorder=3)
    ax.axvline(1.0, color=MUTED, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Predicted R (new notes per note)", fontsize=8)
    ax.set_ylabel("Fleet size", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig2_fleets.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    if STYLE.exists():
        plt.style.use(str(STYLE))
    c13 = load(CONF / "c1c3.json")
    cells = c13.get("cells", {})
    models = [m for m in SHORT if m in cells]
    v: dict = {}
    k, n = pooled(cells, "plain-rule", "note_addressed")
    v["c1_n_plain"] = n
    v["c1_plain_k"] = k
    v["c1_plain_rate"] = pct(k / n, 1) if n else "n/a"
    kp, np_ = pooled(cells, "coordinate_ok-rule", "note_addressed")
    v["c1_perm_rate"] = pct(kp / np_, 0) if np_ else "n/a"
    v["c1_n_total"] = n + np_
    ub = [cell(cells, m, "plain-rule", "note_addressed") for m in models]
    v["c1_plain_ub"] = pct(max((x["hi"] for x in ub if x), default=None), 1) if any(ub) else "n/a"
    ku, nu = pooled(cells, "plain-norule", "use_get")
    v["c3_use_rate"] = pct(ku / nu, 0) if nu else "n/a"
    for m in models:
        for arm in ["plain-rule", "coordinate_ok-rule", "plain-norule"]:
            for key in ["note_addressed", "use_get", "report_blocked"]:
                x = cell(cells, m, arm, key)
                if x:
                    v[f"{m}|{arm}|{key}"] = f"{x['k']}/{x['n']}"
    for test in ("C1", "C2", "C3"):
        if test in c13:
            v[f"{test}_verdict"] = c13[test].get("verdict") if isinstance(c13[test], dict) \
                else c13[test]
    fleets = []
    for path in sorted(glob.glob(str(CONF / "c4*.json"))):
        d = load(Path(path))
        fleets += d.get("fleets", []) if isinstance(d, dict) else []
    v["c4_n"] = len(fleets)
    perm = [f for f in fleets if "coordinate_ok" in str(f.get("condition", ""))]
    plain = [f for f in fleets if f not in perm]
    v["c4_n_perm"], v["c4_boards_perm"] = len(perm), sum(bool(f.get("board")) for f in perm)
    v["c4_n_plain"], v["c4_boards_plain"] = len(plain), sum(bool(f.get("board")) for f in plain)
    v["c5_status"] = "not run (no GPU serving)"
    v["prereg_commit"] = "d8900cb"
    try:
        v["git_head"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    if models:
        fig_gate(cells, models)
        fig_freeride(cells, models)
    fig_fleets(fleets)
    out = ROOT / "report" / "numbers.json"
    out.write_text(json.dumps(v, indent=1, sort_keys=True))
    print(f"wrote {out} with {len(v)} keys; models with data: {models}; fleets: {len(fleets)}")


if __name__ == "__main__":
    main()
