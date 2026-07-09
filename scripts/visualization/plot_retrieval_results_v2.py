"""
Retrieval benchmark plot v2 — Nature-style, percentile-only (R@5%, R@10%).

Style conventions
-----------------
- Double-column width 183 mm (~7.2 in).
- Helvetica/Arial sans-serif; 7 pt ticks/legend, 8 pt axis labels.
- 0.5 pt spines and ticks, outward short ticks.
- No suptitle (caption carries it); bold panel letters "a", "b".
- Vector PDF primary.

Numbers carried over from scripts/plot_retrieval_results.py.

Output
------
figures/retrieval/retrieval_benchmark_v2.{png,pdf}
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


METHODS = ["SIGMMA", "OmiCLIP-ft", "PLIP-ft", "CLIP-ft", "BLEEP-ft"]
COLORS = ["#D55E00", "#0072B2", "#CC79A7", "#56B4E9", "#E69F00"]

METRICS = ["R@5%", "R@10%"]

DATA = {
    "he_to_st": {
        "title": r"H&E $\rightarrow$ ST",
        "values": np.array([
            [0.868, 0.903],   # SIGMMA
            [0.636, 0.825],   # OmiCLIP-ft
            [0.669, 0.871],   # PLIP-ft
            [0.593, 0.812],   # CLIP-ft
            [0.504, 0.722],   # BLEEP-ft
        ]),
    },
    "st_to_he": {
        "title": r"ST $\rightarrow$ H&E",
        "values": np.array([
            [0.851, 0.883],
            [0.605, 0.789],
            [0.664, 0.851],
            [0.584, 0.779],
            [0.419, 0.627],
        ]),
    },
}


def set_nature_rc():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def grouped_bar(ax, panel):
    n_methods = len(METHODS)
    n_metrics = len(METRICS)
    x = np.arange(n_metrics)
    bar_w = 0.82 / n_methods

    for i, (method, color) in enumerate(zip(METHODS, COLORS)):
        offsets = x - 0.41 + bar_w * (i + 0.5)
        ax.bar(offsets, panel["values"][i], bar_w, label=method, color=color,
               edgecolor="black", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS)
    ax.set_title(panel["title"], pad=4)
    ax.set_ylabel("Recall")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.tick_params(axis="x", length=0)  # category axis: drop tick marks


def main():
    set_nature_rc()

    # Double-column = 183 mm ≈ 7.2 in; compact height for Nature ratio.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    grouped_bar(axes[0], DATA["he_to_st"])
    grouped_bar(axes[1], DATA["st_to_he"])

    # Bold panel letters, upper-left of each axes.
    for ax, letter in zip(axes, "ab"):
        ax.text(-0.18, 1.05, letter, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="bottom", ha="left")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(METHODS),
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1.0))

    out_dir = Path(__file__).resolve().parents[1] / "figures" / "retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = out_dir / f"retrieval_benchmark_v2.{ext}"
        fig.savefig(out, dpi=600 if ext == "png" else None, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
