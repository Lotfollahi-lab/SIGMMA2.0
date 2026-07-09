"""Figure: ST → H&E retrieval, per-tissue breakdown (in-house cohorts).

2 rows (R@1%, R@5%) × N tissue cols. Tissue grouping comes from the
authoritative sample_metadata.csv via sigmma.utils.tissue_map (not split('-')),
so beacon+CTCL merge to "Skin" and external HEST-1k public sections are dropped.
All in-house panel groups with >= MIN_N test sections are shown.
Uses spatial direction (ST → H&E) — the reverse of the primary downstream
direction. Mirror of plot_retrieval_per_tissue.py (which is H&E → ST); the
pooled bidirectional figure is plot_retrieval_r1pct_mrr.py (2x2).

Same eva_critique-style bar plot conventions.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sigmma.utils.tissue_map import tissue_of, GROUP_OF, GROUP_LABEL, PUBLIC_TT

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
DATA_JSON = ROOT / "outputs/retrieval/normalized_metrics_step2000.json"
OUT_PNG = ROOT / "figures/retrieval/retrieval_per_tissue_r5pct_st2he.png"
OUT_PDF = ROOT / "figures/retrieval/retrieval_per_tissue_r5pct_st2he.pdf"

CONDITIONS = [
    ("bleep",   "BLEEP-ft",   "#882255"),
    ("clip",    "CLIP-ft",    "#44AA99"),
    ("plip",    "PLIP-ft",    "#DDCC77"),
    ("omiclip", "OmiCLIP-ft", "#999933"),
    ("a01",     "SIGMMA",     "#d62728"),
]
COND_KEYS = [c[0] for c in CONDITIONS]

DIRECTION = "spatial"  # ST → H&E
MIN_N = 2


def group_sections(all_secs):
    """In-house sections grouped by display group; external public dropped."""
    by_group = {}
    for s in all_secs:
        t = tissue_of(s)
        if t in PUBLIC_TT:          # external HEST-1k -> genuine OOD
            continue
        grp = GROUP_OF.get(t)
        if grp is None:
            continue
        by_group.setdefault(grp, []).append(s)
    groups = [(g, secs) for g, secs in by_group.items() if len(secs) >= MIN_N]
    groups.sort(key=lambda x: -len(x[1]))
    return groups


def plot_panel(ax, vals_per_cond, ylabel, n_sec, ymin=0.0, show_ylabel=True):
    means = np.array([np.mean(v) for v in vals_per_cond])
    stds = np.array([np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in vals_per_cond])
    colors = [c[2] for c in CONDITIONS]
    x = np.arange(len(CONDITIONS))
    sigmma_idx = len(CONDITIONS) - 1

    err_lo = np.minimum(stds, means - ymin)
    yerr = np.vstack([err_lo, stds])

    bars = ax.bar(
        x, means, yerr=yerr, color=colors,
        edgecolor="black", linewidth=0.5,
        error_kw=dict(elinewidth=0.6, capsize=1.8, ecolor="black"),
        width=0.72,
    )
    bars[sigmma_idx].set_linewidth(1.0)

    non_sigmma = COND_KEYS[:-1]
    best_idx = max(range(len(non_sigmma)), key=lambda i: means[i])
    ax.axhline(means[best_idx], color="#888888", linestyle="--",
               linewidth=0.5, alpha=0.7, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in CONDITIONS],
                       rotation=40, ha="right", fontsize=6)
    if show_ylabel:
        ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="-", linewidth=0.25,
            color="#cccccc", alpha=0.7, zorder=-1)
    ax.set_axisbelow(True)

    top = (means + stds).max()
    rng = top - ymin
    ax.set_ylim(ymin, top + rng * 0.08 if rng > 0 else top + 0.05)


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 9,
        "axes.labelsize": 7,
        "axes.linewidth": 0.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "legend.fontsize": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    with open(DATA_JSON) as f:
        d = json.load(f)
    all_secs = list(d["a01"][DIRECTION]["per_section"].keys())

    # Group sections by display group (authoritative metadata); public dropped.
    groups = group_sections(all_secs)
    print(f"[{DIRECTION}] In-house test sections per shown group (>= {MIN_N}):")
    for g, secs in groups:
        print(f"  {GROUP_LABEL[g]:14s} n={len(secs)}")

    metrics = [("recall_at_1pct", "Recall @ 1%"),
               ("recall_at_5pct", "Recall @ 5%")]

    ncol = len(groups)
    fig, axes = plt.subplots(2, ncol, figsize=(2.0 * ncol + 0.5, 5.0),
                             gridspec_kw={"wspace": 0.30, "hspace": 0.70},
                             squeeze=False)

    for row, (mkey, mlbl) in enumerate(metrics):
        for col, (grp, secs) in enumerate(groups):
            ax = axes[row, col]
            vals_per_cond = [
                np.array([d[k][DIRECTION]["per_section"][s][mkey] for s in secs])
                for k in COND_KEYS
            ]
            plot_panel(ax, vals_per_cond, mlbl, len(secs),
                       ymin=0.0, show_ylabel=(col == 0))
            if row == 0:
                ax.set_title(f"{GROUP_LABEL[grp]}\n(n={len(secs)})",
                             fontsize=8, pad=4)

    # Panel labels
    letters = "abcdefghijkl"
    for i, ax in enumerate(axes.ravel()):
        ax.text(-0.20, 1.04, letters[i], transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom", ha="left")

    fig.suptitle(
        r"ST $\rightarrow$ H&E retrieval, per-tissue "
        r"(all in-house panels)",
        fontsize=10, fontweight="bold", y=0.985,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(bottom=0.10, top=0.90, hspace=0.85, wspace=0.30)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600)
    fig.savefig(OUT_PDF)
    print(f"Wrote: {OUT_PNG}")
    print(f"Wrote: {OUT_PDF}")


if __name__ == "__main__":
    main()
