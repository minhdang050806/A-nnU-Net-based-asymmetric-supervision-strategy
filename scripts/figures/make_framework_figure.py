#!/usr/bin/env python3
"""SUPERSEDED: matplotlib draft of the OS50 + AURA framework diagram.

The published Figure 2 (``paper/figures/framework_os50_aura.{pdf,png}``) is
hand-authored in diagrams.net, not produced by this script. The two diagrams
show the same pipeline but differ in layout and typography.

This script is kept for provenance and writes to a distinct filename so it can
never overwrite the published figure. Running it is not part of reproducing the
paper.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "paper/figures/framework_os50_aura_matplotlib_draft.pdf"
BLUE = "#4477AA"
GREEN = "#228833"
ORANGE = "#EE7733"
PURPLE = "#AA3377"
GRAY = "#667080"


def box(ax, x, y, w, h, text, color, size=8.8, fill="white"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.025",
        facecolor=fill, edgecolor=color, linewidth=1.45, zorder=2,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, color="#20242A", zorder=3)


def arrow(ax, x1, y1, x2, y2, color=GRAY, dashed=False, width=1.4):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
        linewidth=width, linestyle="--" if dashed else "-", color=color,
        shrinkA=3, shrinkB=3, zorder=1,
    ))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="Output PDF path; a PNG is written alongside it")
    OUT = ap.parse_args().out

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.2, 4.25))
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.1, 2.35), 9.35, 1.85, boxstyle="round,pad=0.03",
                                facecolor="#F3F7FB", edgecolor="#D1DCE8", linewidth=1.0))
    ax.add_patch(FancyBboxPatch((0.1, 0.25), 9.35, 1.85, boxstyle="round,pad=0.03",
                                facecolor="#F8F3F7", edgecolor="#E3D2DF", linewidth=1.0))
    ax.add_patch(FancyBboxPatch((9.7, 0.25), 3.15, 3.95, boxstyle="round,pad=0.03",
                                facecolor="#FBF7F2", edgecolor="#E6D8C7", linewidth=1.0))
    ax.text(0.3, 3.93, "OS50 branch", fontsize=10.5, fontweight="bold", color=BLUE)
    ax.text(0.3, 1.83, "AURA branch", fontsize=10.5, fontweight="bold", color=PURPLE)
    ax.text(9.9, 3.93, "Inference fusion", fontsize=10.5, fontweight="bold", color="#303740")

    box(ax, 0.35, 2.70, 1.35, 0.90, "ULF MRI\n+ HF anchor", GREEN, 9.0)
    box(ax, 2.15, 2.62, 2.05, 1.05, "3D nnU-Net\nDice + TopK10\nOS=0.50, LS=0.05", BLUE, 8.7)
    box(ax, 4.75, 2.78, 1.55, 0.72, "OS50 model", BLUE, 9.2)
    box(ax, 7.05, 2.78, 1.45, 0.72, "$p^{OS50}$", BLUE, 10.0)
    arrow(ax, 1.70, 3.15, 2.15, 3.15, BLUE)
    arrow(ax, 4.20, 3.15, 4.75, 3.15, BLUE)
    arrow(ax, 6.30, 3.15, 7.05, 3.15, BLUE)

    box(ax, 0.35, 0.60, 1.35, 0.90, "ULF MRI\n+ HF/LF masks", ORANGE, 9.0)
    box(ax, 2.15, 0.45, 2.05, 1.20,
        "Reliability gate $g_v$\ndisagreement $\\times$ boundary\n$\\times$ entropy $\\times$ class weight\n$\\times$ training ramp",
        PURPLE, 8.0)
    box(ax, 4.75, 0.62, 1.55, 0.86, "Soft target\n$(1-g)q^{HF}+gq^{LF}$", PURPLE, 8.7)
    box(ax, 6.75, 0.48, 1.85, 1.13, "AURA nnU-Net\nsoft Dice + TopK10\n+ boundary Dice", PURPLE, 8.3)
    box(ax, 8.85, 0.70, 0.45, 0.68, "$p^{AURA}$", PURPLE, 8.7)
    arrow(ax, 1.70, 1.05, 2.15, 1.05, PURPLE)
    arrow(ax, 4.20, 1.05, 4.75, 1.05, PURPLE)
    arrow(ax, 6.30, 1.05, 6.75, 1.05, PURPLE)
    arrow(ax, 8.60, 1.05, 8.85, 1.05, PURPLE)

    arrow(ax, 5.52, 2.78, 7.10, 1.61, GRAY, dashed=True, width=1.1)
    ax.text(6.25, 2.08, "checkpoint initialization", fontsize=7.2,
            color=GRAY, ha="center", rotation=-18)

    box(ax, 10.05, 2.55, 1.20, 0.82, "Soft fusion\n0.6 / 0.4", ORANGE, 9.1)
    box(ax, 10.05, 1.20, 1.20, 0.72, "argmax", ORANGE, 9.1)
    box(ax, 11.65, 1.18, 0.90, 0.76, "Per-class\nLCC", GREEN, 8.7)
    arrow(ax, 8.50, 3.15, 10.05, 2.98, BLUE)
    arrow(ax, 9.30, 1.05, 10.05, 2.70, PURPLE)
    arrow(ax, 10.65, 2.55, 10.65, 1.92, ORANGE)
    arrow(ax, 11.25, 1.56, 11.65, 1.56, GREEN)

    ax.text(6.5, 0.02,
            "HF is the scored anchor; LF supervision is conditional and bounded.",
            ha="center", fontsize=9.0, color="#303740", fontweight="bold")
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
