"""
Bar plot of in-house Sanger Xenium retrieval benchmark
(SIGMMA vs four fine-tuned baselines).

Source numbers
--------------
- SIGMMA: retrieval-SIGMMA.18481 log (ckpt in-house_sigmma_18328 step 500,
  per-scale LoRA). Replaces the earlier 18041 numbers (ckpt 16314 step 1500).
- BLEEP/CLIP/PLIP fine-tuned: retrieval-baselines-ft.18322 log
- OmiCLIP fine-tuned: retrieval-omiclip-ft.18327 log

Layout
------
2 x 2 grid:
 - row 1: H&E -> ST direction (image_recall in code).
 - row 2: ST -> H&E direction (spatial_recall in code).
 - col 1: percentile recall (Recall@5%/10%/15%).
 - col 2: strict recall (Recall@1/5/10), log y.

Output
------
figures/retrieval_benchmark.{png,pdf}
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


METHODS = ["SIGMMA", "OmiCLIP-ft", "PLIP-ft", "CLIP-ft", "BLEEP-ft"]
# SIGMMA distinct color, baselines on a blue->orange progression.
COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]

DATA = {
    "he_to_st_percent": {
        "title": r"H&E $\rightarrow$ ST  (percentile)",
        "metrics": ["R@5%", "R@10%", "R@15%"],
        "values": np.array([
            [0.868, 0.903, 0.921],   # SIGMMA
            [0.636, 0.825, 0.918],   # OmiCLIP-ft
            [0.669, 0.871, 0.950],   # PLIP-ft
            [0.593, 0.812, 0.918],   # CLIP-ft
            [0.504, 0.722, 0.845],   # BLEEP-ft
        ]),
        "log": False,
        "ylim": (0.0, 1.0),
    },
    "he_to_st_strict": {
        "title": r"H&E $\rightarrow$ ST  (strict)",
        "metrics": ["R@1", "R@5", "R@10"],
        "values": np.array([
            [0.0411, 0.121, 0.178],
            [0.00065, 0.00328, 0.00657],
            [0.00088, 0.00430, 0.00807],
            [0.00075, 0.00297, 0.00579],
            [0.00039, 0.00203, 0.00428],
        ]),
        "log": True,
        "ylim": (1e-4, 1.0),
    },
    "st_to_he_percent": {
        "title": r"ST $\rightarrow$ H&E  (percentile)",
        "metrics": ["R@5%", "R@10%", "R@15%"],
        "values": np.array([
            [0.851, 0.883, 0.902],
            [0.605, 0.789, 0.900],
            [0.664, 0.851, 0.944],
            [0.584, 0.779, 0.894],
            [0.419, 0.627, 0.766],
        ]),
        "log": False,
        "ylim": (0.0, 1.0),
    },
    "st_to_he_strict": {
        "title": r"ST $\rightarrow$ H&E  (strict)",
        "metrics": ["R@1", "R@5", "R@10"],
        "values": np.array([
            [0.0350, 0.108, 0.161],
            [0.00054, 0.00320, 0.00623],
            [0.00091, 0.00432, 0.00849],
            [0.00037, 0.00261, 0.00518],
            [0.00034, 0.00161, 0.00302],
        ]),
        "log": True,
        "ylim": (1e-4, 1.0),
    },
}


def grouped_bar(ax, panel):
    n_methods = len(METHODS)
    n_metrics = len(panel["metrics"])
    x = np.arange(n_metrics)
    bar_w = 0.85 / n_methods

    for i, (method, color) in enumerate(zip(METHODS, COLORS)):
        offsets = x - 0.425 + bar_w * (i + 0.5)
        ax.bar(offsets, panel["values"][i], bar_w, label=method, color=color,
               edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(panel["metrics"])
    ax.set_title(panel["title"])
    ax.set_ylabel("Recall")
    if panel["log"]:
        ax.set_yscale("log")
    ax.set_ylim(panel["ylim"])
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("he_to_st_percent", axes[0, 0]),
        ("he_to_st_strict",  axes[0, 1]),
        ("st_to_he_percent", axes[1, 0]),
        ("st_to_he_strict",  axes[1, 1]),
    ]
    for key, ax in panels:
        grouped_bar(ax, DATA[key])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(METHODS),
        bbox_to_anchor=(0.5, -0.02), frameon=False,
    )
    fig.suptitle(
        "Cross-modal retrieval on in-house Sanger Xenium "
        "(SIGMMA vs fine-tuned baselines)",
        fontsize=12, y=1.0,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))

    out_dir = Path(__file__).resolve().parents[1] / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = out_dir / f"retrieval_benchmark.{ext}"
        fig.savefig(out, dpi=220, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
