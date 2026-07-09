"""Figure: niche_category MCC + weighted F1, eva_critique-style.

Matches figures/eva_critique/global_metrics_bars.{pdf,png} conventions:
  * Bar plot with mean ± SD error bars
  * UNI2-h dashed horizontal reference line
  * Horizontal y-axis grid lines (#cccccc)
  * Left+bottom spines only, font Helvetica 8 pt
  * Bracketed significance annotation (paired t-test p-value)
  * Project method palette (UNI2-h grey + 4 baselines + SIGMMA red)

Statistical test: **paired two-sided t-test** across n=6 sections (instead of
Wilcoxon signed-rank, which has a 0.031/0.062 floor at n=6 — see project memo
[[niche-mcc-2026-05-20]] follow-up 2026-05-20).
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from pathlib import Path

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
OUT_PNG = ROOT / "figures/cell_niche/niche_category_mcc_wf1.png"
OUT_PDF = ROOT / "figures/cell_niche/niche_category_mcc_wf1.pdf"

# Project method palette (eva_critique convention)
CONDITIONS = [
    ("UNI2-h",       "UNI2-h",     "#888888"),
    ("BLEEP",        "BLEEP-ft",   "#882255"),
    ("CLIP",         "CLIP-ft",    "#44AA99"),
    ("PLIP",         "PLIP-ft",    "#DDCC77"),
    ("OmiCLIP",      "OmiCLIP-ft", "#999933"),
    ("SIGMMA α=0.1", "SIGMMA",     "#d62728"),
]
MODEL_KEYS = [c[0] for c in CONDITIONS]
SECTION_ORDER = [
    "BK46-SKI-27-FO-2-s3", "BK39-SKI-27-FO-1-S3",
    "BK23-SKI-27-FO-1-S6-C2", "BK30-SKI-27-FO-4-S3",
    "BK50-SKI-27-FO-1-s2", "BK46-SKI-27-FO-2-s2",
]


def sig_marker(p):
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def paired_t(a, b):
    """Two-sided paired t-test on arrays a, b (uses magnitudes, not only ranks)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2:
        return float("nan")
    return float(ttest_rel(a, b, alternative="two-sided").pvalue)


def plot_panel(ax, pivot, ylabel, panel_letter, ymin):
    data = [pivot[k].values for k in MODEL_KEYS]
    medians = np.array([np.median(v) for v in data])
    colors = [c[2] for c in CONDITIONS]
    labels = [c[1] for c in CONDITIONS]
    x = np.arange(len(CONDITIONS))
    sigmma_idx = len(CONDITIONS) - 1

    bp = ax.boxplot(
        data, positions=x, widths=0.55, patch_artist=True,
        showcaps=True, showfliers=False,
        boxprops=dict(linewidth=0.6),
        medianprops=dict(color="black", linewidth=1.0),
        whiskerprops=dict(linewidth=0.6, color="black"),
        capprops=dict(linewidth=0.6, color="black"),
    )
    for patch, (key, _, c) in zip(bp["boxes"], CONDITIONS):
        patch.set_facecolor(c)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2 if key == "SIGMMA α=0.1" else 0.6)
        patch.set_alpha(0.9)

    # UNI2-h dashed reference line at its median
    ax.axhline(medians[0], color="#888888", linestyle="--",
               linewidth=0.6, alpha=0.7, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="-", linewidth=0.3,
            color="#cccccc", alpha=0.7, zorder=-1)
    ax.set_axisbelow(True)

    # Dynamic ylim: cover whiskers fully + reserve headroom for n_baseline brackets.
    whisker_top = max(np.max(v) for v in data)
    whisker_bot = min(np.min(v) for v in data)
    n_baseline = len(MODEL_KEYS) - 1
    data_range = whisker_top - whisker_bot
    bracket_step = data_range * 0.045
    headroom = bracket_step * (n_baseline + 0.5)
    ymax = whisker_top + headroom
    ymin_eff = ymin
    ax.set_ylim(ymin_eff, ymax)
    yrange = ymax - ymin_eff

    # Paired t-test: SIGMMA vs every baseline; stack brackets (compact).
    sigmma_vals = pivot["SIGMMA α=0.1"].values
    tick_h = yrange * 0.005
    base_y = whisker_top + bracket_step * 0.3
    for slot, ref_key in enumerate(MODEL_KEYS[:-1]):
        p = paired_t(sigmma_vals, pivot[ref_key].values)
        mk = sig_marker(p)
        y_b = base_y + bracket_step * slot
        x_l, x_r = sorted([MODEL_KEYS.index(ref_key), sigmma_idx])
        ax.plot([x_l, x_l, x_r, x_r],
                [y_b, y_b + tick_h, y_b + tick_h, y_b],
                color="black", linewidth=0.4, clip_on=False)
        ax.text((x_l + x_r) / 2, y_b + tick_h - yrange * 0.002,
                mk, ha="center", va="bottom",
                fontsize=5.5, fontweight="bold", color="black",
                clip_on=False)

    ax.text(-0.18, 1.02, panel_letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    mcc_df = pd.read_csv(ROOT / "outputs/cell_niche/extra_metrics_per_section.csv")
    mcc_pivot = (mcc_df[mcc_df["label"] == "niche_category"]
                 .pivot_table(index="sample", columns="model", values="mcc")
                 .reindex(SECTION_ORDER))

    wf1_df = pd.read_csv(ROOT / "outputs/cell_niche/weighted_f1_per_section.csv")
    wf1_pivot = (wf1_df[wf1_df["label"] == "niche_category"]
                 .pivot_table(index="sample", columns="model", values="weighted_f1")
                 .reindex(SECTION_ORDER))

    fig, axes = plt.subplots(1, 2, figsize=(5.4, 4.4),
                             gridspec_kw={"wspace": 0.40})
    plot_panel(axes[0], mcc_pivot,
               "Matthews correlation coefficient", "a",
               ymin=0.55)
    plot_panel(axes[1], wf1_pivot,
               "Weighted F1 score", "b",
               ymin=0.70)

    fig.suptitle(
        r"H&E $\rightarrow$ Niche prediction",
        fontsize=10, fontweight="bold", y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(bottom=0.27)
    fig.text(
        0.5, 0.06,
        "Box: median, Q1–Q3, 1.5 × IQR whiskers (n = 6 sections); "
        "dashed line = UNI2-h median.",
        ha="center", fontsize=6.5, color="black",
    )
    fig.text(
        0.5, 0.02,
        "Paired t-test vs SIGMMA:  * P ≤ 0.05,  ** P ≤ 0.01,  *** P ≤ 0.001.",
        ha="center", fontsize=6.5, color="black",
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600)
    fig.savefig(OUT_PDF)
    print(f"Wrote: {OUT_PNG}")
    print(f"Wrote: {OUT_PDF}")


if __name__ == "__main__":
    main()
