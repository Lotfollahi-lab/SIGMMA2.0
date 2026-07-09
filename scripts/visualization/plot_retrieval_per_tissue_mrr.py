"""Figure: H&E ↔ ST retrieval, per-tissue breakdown (in-house cohorts).

2 rows (R@1%, MRR) × N tissue cols. Tissue grouping comes from the
authoritative sample_metadata.csv via sigmma.utils.tissue_map (not split('-')),
so beacon+CTCL merge to "Skin" and external HEST-1k public sections are dropped.
All in-house panel groups with >= MIN_N test sections are shown.
Uses image direction (H&E → ST) — the primary downstream direction.

Same eva_critique-style bar plot conventions as plot_retrieval_r1pct_mrr.py.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_rel

from sigmma.utils.tissue_map import tissue_of, GROUP_OF, GROUP_LABEL, PUBLIC_TT

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
DATA_JSON = ROOT / "outputs/retrieval/normalized_metrics_step2000.json"
OUT_PNG = ROOT / "figures/retrieval/retrieval_per_tissue_mrr.png"
OUT_PDF = ROOT / "figures/retrieval/retrieval_per_tissue_mrr.pdf"


def sig_marker(p):
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def paired_t(a, b):
    """Two-sided paired t-test on per-section arrays a, b (matched sections)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2:
        return float("nan")
    return float(ttest_rel(a, b, alternative="two-sided").pvalue)

CONDITIONS = [
    ("bleep",   "BLEEP-ft",   "#882255"),
    ("clip",    "CLIP-ft",    "#44AA99"),
    ("plip",    "PLIP-ft",    "#DDCC77"),
    ("omiclip", "OmiCLIP-ft", "#999933"),
    ("a01",     "SIGMMA",     "#d62728"),
]
COND_KEYS = [c[0] for c in CONDITIONS]

DIRECTION = "image"  # H&E → ST
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


def plot_panel(ax, vals_per_cond, ylabel, n_sec, ymin=0.0, show_ylabel=True, tag=""):
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

    # Paired t-test: SIGMMA vs each baseline (matched sections within tissue);
    # stack one compact bracket per baseline above the tallest error bar.
    top = (means + stds).max()
    rng = (top - ymin) if (top - ymin) > 0 else 0.05
    n_baseline = len(COND_KEYS) - 1
    bracket_step = rng * 0.085
    ymax = top + bracket_step * (n_baseline + 0.8)
    ax.set_ylim(ymin, ymax)
    yrange = ymax - ymin

    sigmma_vals = vals_per_cond[-1]
    tick_h = yrange * 0.006
    base_y = top + bracket_step * 0.45
    for slot in range(n_baseline):
        p = paired_t(sigmma_vals, vals_per_cond[slot])
        mk = sig_marker(p)
        print(f"  [{tag}] SIGMMA vs {COND_KEYS[slot]:<8s} "
              f"n={len(sigmma_vals)} paired-t p={p:.2e} {mk}")
        y_b = base_y + bracket_step * slot
        x_l, x_r = slot, sigmma_idx
        ax.plot([x_l, x_l, x_r, x_r],
                [y_b, y_b + tick_h, y_b + tick_h, y_b],
                color="black", linewidth=0.35, clip_on=False)
        ax.text((x_l + x_r) / 2, y_b + tick_h - yrange * 0.002,
                mk, ha="center", va="bottom",
                fontsize=5.0, fontweight="bold", color="black",
                clip_on=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_json", default=str(DATA_JSON))
    ap.add_argument("--out_dir", default=None,
                    help="If set, write retrieval_per_tissue_mrr.{png,pdf} here.")
    args = ap.parse_args()
    data_json = Path(args.data_json)
    if args.out_dir:
        out_png = Path(args.out_dir) / "retrieval_per_tissue_mrr.png"
        out_pdf = Path(args.out_dir) / "retrieval_per_tissue_mrr.pdf"
    else:
        out_png, out_pdf = OUT_PNG, OUT_PDF

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

    with open(data_json) as f:
        d = json.load(f)
    all_secs = list(d["a01"][DIRECTION]["per_section"].keys())

    # Group sections by display group (authoritative metadata); public dropped.
    groups = group_sections(all_secs)
    print(f"In-house test sections per shown group (>= {MIN_N}):")
    for g, secs in groups:
        print(f"  {GROUP_LABEL[g]:14s} n={len(secs)}")

    metrics = [("recall_at_1pct", "Recall @ 1%"),
               ("mrr",            "Mean Reciprocal Rank")]

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
                       ymin=0.0, show_ylabel=(col == 0),
                       tag=f"{grp} {mkey}")
            if row == 0:
                ax.set_title(f"{GROUP_LABEL[grp]}\n(n={len(secs)})",
                             fontsize=8, pad=4)

    # Panel labels
    import string
    letters = string.ascii_lowercase  # supports up to 26 subplots
    for i, ax in enumerate(axes.ravel()):
        ax.text(-0.20, 1.04, letters[i], transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom", ha="left")

    fig.suptitle(
        r"H&E $\rightarrow$ ST retrieval, per-tissue "
        r"(all in-house panels)",
        fontsize=10, fontweight="bold", y=0.985,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(bottom=0.10, top=0.90, hspace=0.85, wspace=0.30)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_pdf}")


if __name__ == "__main__":
    main()
