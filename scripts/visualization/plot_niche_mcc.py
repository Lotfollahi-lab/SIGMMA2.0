"""Figure (Nature Methods style): MCC for niche label transfer.

1×2 panel: niches_terra (a) | niche_category (b). Same conventions as
plot_niche_category_mcc_wf1.py — box + strip plot, paired Wilcoxon brackets
with asterisks, project method palette.
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from pathlib import Path

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
CSV = ROOT / "outputs/cell_niche/extra_metrics_per_section.csv"
OUT_PNG = ROOT / "figures/cell_niche/niche_mcc.png"
OUT_PDF = ROOT / "figures/cell_niche/niche_mcc.pdf"

mpl.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":         7,
    "axes.labelsize":    7,
    "axes.titlesize":    7,
    "axes.linewidth":    0.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size":  2.5,
    "ytick.major.size":  2.5,
    "xtick.labelsize":   7,
    "ytick.labelsize":   6,
    "legend.fontsize":   6,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "svg.fonttype":      "none",
})

MODEL_ORDER = ["UNI2-h", "BLEEP", "CLIP", "PLIP", "OmiCLIP", "SIGMMA α=0.1"]
COLORS = {
    "UNI2-h":       "#888888",
    "BLEEP":        "#882255",
    "CLIP":         "#44AA99",
    "PLIP":         "#DDCC77",
    "OmiCLIP":      "#999933",
    "SIGMMA α=0.1": "#d62728",
}
SECTION_ORDER = [
    "BK46-SKI-27-FO-2-s3", "BK39-SKI-27-FO-1-S3",
    "BK23-SKI-27-FO-1-S6-C2", "BK30-SKI-27-FO-4-S3",
    "BK50-SKI-27-FO-1-s2", "BK46-SKI-27-FO-2-s2",
]
LABEL_TITLES = {
    "niches_terra":   "niches_terra (19-class)",
    "niche_category": "niche_category (8-class)",
}


def sig_label(p):
    if np.isnan(p): return "NS"
    if p <= 0.001:  return "***"
    if p <= 0.01:   return "**"
    if p <= 0.05:   return "*"
    return "NS"


def paired(s, r):
    diff = np.asarray(s) - np.asarray(r)
    w = int((diff > 0).sum()); l = int((diff < 0).sum())
    try:
        _, p2 = wilcoxon(s, r, alternative="two-sided", zero_method="wilcox")
    except Exception:
        p2 = np.nan
    return diff.mean(), w, l, p2


def plot_panel(ax, pivot, ylabel, title, panel_letter):
    xpos = np.arange(len(MODEL_ORDER))
    data = [pivot[m].values for m in MODEL_ORDER]

    bp = ax.boxplot(
        data, positions=xpos, widths=0.55, patch_artist=True,
        showcaps=True, showfliers=False,
        boxprops=dict(linewidth=0.5),
        medianprops=dict(color="black", linewidth=1.0),
        whiskerprops=dict(linewidth=0.5, color="black"),
        capprops=dict(linewidth=0.5, color="black"),
    )
    for patch, m in zip(bp["boxes"], MODEL_ORDER):
        patch.set_facecolor(COLORS[m])
        patch.set_edgecolor("black" if m == "SIGMMA α=0.1" else "#333333")
        patch.set_linewidth(1.0 if m == "SIGMMA α=0.1" else 0.5)
        patch.set_alpha(0.85)

    ax.set_xticks(xpos)
    ax.set_xticklabels([m.replace(" α=0.1", "") for m in MODEL_ORDER],
                       rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=7, pad=14)
    ax.tick_params(axis="x", pad=1)

    sigmma_idx = MODEL_ORDER.index("SIGMMA α=0.1")
    sv = pivot["SIGMMA α=0.1"].values
    pvals = [None if r == "SIGMMA α=0.1" else paired(sv, pivot[r].values)[3]
             for r in MODEL_ORDER]

    ymax = float(max(np.max(v) for v in data))
    yrange = ymax - float(min(np.min(v) for v in data))
    bracket_base = ymax + 0.012 * yrange
    bracket_step = 0.045 * yrange

    sig_slot = 0
    for i, p in enumerate(pvals):
        if p is None: continue
        sig = sig_label(p)
        if sig == "NS": continue
        y_b = bracket_base + bracket_step * sig_slot
        sig_slot += 1
        ax.plot([xpos[i], xpos[i], xpos[sigmma_idx], xpos[sigmma_idx]],
                [y_b - 0.008 * yrange, y_b, y_b, y_b - 0.008 * yrange],
                color="black", linewidth=0.5, clip_on=False)
        ax.text((xpos[i] + xpos[sigmma_idx]) / 2, y_b + 0.002 * yrange,
                sig, ha="center", va="bottom",
                fontsize=6.5, fontweight="bold", color="black", clip_on=False)

    ymin = float(min(np.min(v) for v in data)) - 0.015 * yrange
    ymax_lim = bracket_base + bracket_step * max(sig_slot, 1) + 0.005 * yrange
    ax.set_ylim(ymin, ymax_lim)

    ax.text(-0.11, 1.01, panel_letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")


def main():
    df = pd.read_csv(CSV)
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.9))

    for ci, label in enumerate(("niches_terra", "niche_category")):
        pivot = (df[df["label"] == label]
                 .pivot_table(index="sample", columns="model", values="mcc")
                 .reindex(SECTION_ORDER)[MODEL_ORDER])
        plot_panel(axes[ci], pivot,
                   "Matthews correlation coefficient",
                   LABEL_TITLES[label],
                   ["a", "b"][ci])

    fig.text(
        0.5, 0.005,
        "Multi-class MCC (Gorodkin $R_K$). Box: median, Q1–Q3, 1.5 × IQR whiskers across "
        "n = 6 sections.  "
        "Paired Wilcoxon vs SIGMMA:  * P ≤ 0.05,  ** P ≤ 0.01,  *** P ≤ 0.001.",
        ha="center", fontsize=5.5, color="black",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1.0])
    plt.subplots_adjust(wspace=0.25)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    plt.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Wrote: {OUT_PNG}")
    print(f"Wrote: {OUT_PDF}")


if __name__ == "__main__":
    main()
