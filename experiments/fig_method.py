"""Figure 0 of the report: the setup and the links of the first step, drawn at print size.

Colours follow Figure 1. Blue is the house rule, green is no rule, orange is permission. Each
action is coloured by the condition that brings it out. The figure holds no numbers.

    uv run python experiments/fig_method.py
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, PALE = "#1a1a1a", "#55534e", "#6b6b66", "#f3f2ee"

for f in Path.home().joinpath("Library/Fonts").glob("OldStandard-*.ttf"):
    font_manager.fontManager.addfont(str(f))
mpl.rcParams.update({"font.family": "Old Standard TT", "font.size": 8.5, "pdf.fonttype": 42})


def main() -> None:
    fig = plt.figure(figsize=(6.5, 2.35))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 130), ax.set_ylim(0, 47)
    ax.axis("off")

    def box(x, y, w, h, text, fc="white", ec=INK2, tc=INK, size=8.5, bold=False, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=0.8, ls=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=tc,
                fontweight="bold" if bold else "normal")

    def arrow(p0, p1, color=INK2, lw=1.1, ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, color=color,
                                     lw=lw, ls=ls, shrinkA=0, shrinkB=0,
                                     connectionstyle=f"arc3,rad={rad}"))

    def label(x, y, head, sub, color, ha="left"):
        ax.text(x, y + 1.5, head, ha=ha, va="bottom", fontsize=8.5, color=color, fontweight="bold")
        ax.text(x, y + 1.2, sub, ha=ha, va="top", fontsize=7.4, color=MUTED, style="italic")

    # the blocked agent and its honest exit
    box(2, 21, 20, 9, "blocked agent", fc=INK, ec=INK, tc="white", bold=True)
    ax.text(12, 19.2, "its job lacks a dataset", ha="center", va="top", fontsize=7.4, color=MUTED)
    ax.text(12, 16.4, "that only job B makes", ha="center", va="top", fontsize=7.4, color=MUTED)
    box(2, 1.5, 20, 9, "", ec=BLUE)
    ax.text(12, 7.6, "reports the block", ha="center", va="center", fontsize=8.5, color=BLUE,
            fontweight="bold")
    ax.text(12, 4.0, "with the house rule", ha="center", va="center", fontsize=7.4,
            color=MUTED, style="italic")
    arrow((12, 13.6), (12, 11.0), color=BLUE)

    # the shared store
    ax.add_patch(FancyBboxPatch((44, 3), 40, 41, boxstyle="round,pad=0,rounding_size=2",
                                fc=PALE, ec="none"))
    ax.text(46, 41.2, "shared artifact store", fontsize=8.5, fontweight="bold", color=INK2,
            va="center")
    ax.text(46, 37.6, "a log the agents cannot see records", fontsize=7.2, color=MUTED, va="center")
    ax.text(46, 35.0, "every read and write", fontsize=7.2, color=MUTED, va="center")
    box(49, 25, 30, 6.5, "job B's output")
    box(49, 12, 30, 6.5, "notes for other jobs")
    ax.text(64, 7.5, "outputs of many other jobs", ha="center", fontsize=7.2, color=MUTED,
            va="center")

    # the blocked agent's two other choices
    arrow((22.6, 28.2), (48.4, 28.2), color=AQUA)
    ax.text(24, 32.9, "takes it", va="bottom", fontsize=8.5, color=AQUA, fontweight="bold")
    ax.text(24, 29.5, "if no rule forbids it", va="bottom", fontsize=7.4, color=MUTED,
            style="italic")
    arrow((22.6, 23.0), (48.4, 15.2), color=ORANGE)
    ax.text(24, 14.6, "asks for it", va="top", fontsize=8.5, color=ORANGE, fontweight="bold")
    ax.text(24, 11.4, "only if a sentence", va="top", fontsize=7.4, color=MUTED, style="italic")
    ax.text(24, 8.6, "permits it", va="top", fontsize=7.4, color=MUTED, style="italic")

    # the other agent and the chain that a board needs
    box(106, 21, 21, 9, "another agent", fc=INK, ec=INK, tc="white", bold=True)
    arrow((79.6, 16.6), (105.4, 23.4), color=INK2)
    ax.text(90.5, 22.6, "sees, opens", ha="center", va="bottom", fontsize=8.5, color=INK2,
            fontweight="bold", rotation=14)
    arrow((105.4, 20.4), (79.6, 13.4), color=ORANGE, ls=(0, (3, 2)), rad=-0.15)
    ax.text(97, 9.2, "answers", ha="center", va="center", fontsize=8.5, color=ORANGE,
            fontweight="bold")
    ax.text(97, 6.2, "almost never without permission", ha="center", va="center", fontsize=7.4,
            color=MUTED, style="italic")
    ax.text(116.5, 38.5, "A board grows only", ha="center", fontsize=7.6, color=INK2, va="center")
    ax.text(116.5, 35.6, "if notes trigger notes.", ha="center", fontsize=7.6, color=INK2,
            va="center")

    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig0_method.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {FIG / 'fig0_method.pdf'}")


if __name__ == "__main__":
    main()
