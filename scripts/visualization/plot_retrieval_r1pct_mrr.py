"""Figure: H&E ↔ ST cross-modal retrieval — R@1% + MRR, both directions.

2×2 layout:
  a, b: H&E → ST  (image queries against spatial references)
  c, d: ST → H&E  (spatial queries against image references)

Matches eva_critique-style bar plot conventions used in
plot_niche_category_mcc_wf1.py:
  * Bar plot with mean ± SD error bars (asymmetric, clipped at 0)
  * Dashed line = best non-SIGMMA baseline mean
  * Horizontal y-axis grid, left+bottom spines only
  * Helvetica 8 pt, project method palette
  * Significance brackets: paired t-test (SIGMMA vs each baseline), per-section
    matched, two-sided; */**/*** at p<0.05/0.01/0.001, else ns

Filtered to in-house sections (all Sanger panels, within-panel trained);
external HEST-1k public sections (IDC/PAAD/LUAD/SKCM) are excluded as genuine
OOD. In-house membership comes from the authoritative sample_metadata.csv via
sigmma.utils.tissue_map (not split('-')).
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_rel

from sigmma.utils.tissue_map import tissue_of, GROUP_OF, PUBLIC_TT


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


ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
DATA_JSON = ROOT / "outputs/retrieval/normalized_metrics_step2000.json"
OUT_PNG = ROOT / "figures/retrieval/retrieval_r1pct_mrr.png"
OUT_PDF = ROOT / "figures/retrieval/retrieval_r1pct_mrr.pdf"

CONDITIONS = [
    ("bleep",   "BLEEP-ft",   "#882255"),
    ("clip",    "CLIP-ft",    "#44AA99"),
    ("plip",    "PLIP-ft",    "#DDCC77"),
    ("omiclip", "OmiCLIP-ft", "#999933"),
    ("a01",     "SIGMMA",     "#d62728"),
]
COND_KEYS = [c[0] for c in CONDITIONS]


def is_in_house(s: str) -> bool:
    """In-house Sanger panel section: known tissue group, not external public."""
    t = tissue_of(s)
    return t not in PUBLIC_TT and GROUP_OF.get(t) is not None


def plot_panel(ax, pivot, ylabel, panel_letter, ymin, log_scale=False):
    """Plot bar panel; if log_scale=True, plot mean ± SD of log10(values)."""
    if log_scale:
        # Compute per-section log10 first, then aggregate.
        log_pivot = pivot.apply(lambda col: np.log10(np.clip(col, 1e-6, None)))
        data = [log_pivot[k].values for k in COND_KEYS]
    else:
        data = [pivot[k].values for k in COND_KEYS]

    means = np.array([np.mean(v) for v in data])
    stds = np.array([np.std(v, ddof=1) for v in data])
    colors = [c[2] for c in CONDITIONS]
    labels = [c[1] for c in CONDITIONS]
    x = np.arange(len(CONDITIONS))
    sigmma_idx = len(CONDITIONS) - 1

    if log_scale:
        # Bars drawn from ymin upward (so they don't extend "downward" from 0).
        bar_heights = means - ymin
        bottoms = np.full_like(means, ymin)
        # Symmetric error bars centered on means (in log space).
        yerr = np.vstack([stds, stds])
        bars = ax.bar(
            x, bar_heights, bottom=bottoms,
            yerr=yerr, color=colors,
            edgecolor="black", linewidth=0.6,
            error_kw=dict(elinewidth=0.7, capsize=2.2, ecolor="black"),
            width=0.72,
        )
    else:
        err_lo = np.minimum(stds, means - ymin)
        yerr = np.vstack([err_lo, stds])
        bars = ax.bar(
            x, means, yerr=yerr, color=colors,
            edgecolor="black", linewidth=0.6,
            error_kw=dict(elinewidth=0.7, capsize=2.2, ecolor="black"),
            width=0.72,
        )
    bars[sigmma_idx].set_linewidth(1.2)

    non_sigmma = COND_KEYS[:-1]
    best_idx = max(range(len(non_sigmma)), key=lambda i: means[i])
    ax.axhline(means[best_idx], color="#888888", linestyle="--",
               linewidth=0.6, alpha=0.7, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="-", linewidth=0.3,
            color="#cccccc", alpha=0.7, zorder=-1)
    ax.set_axisbelow(True)

    # Paired t-test: SIGMMA vs every baseline (matched sections). Reserve
    # headroom above the tallest error bar for one stacked bracket per baseline.
    top_of_bars = (means + stds).max()
    raw_range = top_of_bars - ymin
    n_baseline = len(COND_KEYS) - 1
    bracket_step = raw_range * 0.075
    ymax = top_of_bars + bracket_step * (n_baseline + 0.8)
    ax.set_ylim(ymin, ymax)
    yrange = ymax - ymin

    sigmma_key = COND_KEYS[-1]
    sigmma_vals = pivot[sigmma_key].values
    tick_h = yrange * 0.006
    base_y = top_of_bars + bracket_step * 0.45
    for slot, ref_key in enumerate(COND_KEYS[:-1]):
        p = paired_t(sigmma_vals, pivot[ref_key].values)
        mk = sig_marker(p)
        print(f"  [{panel_letter}] {ylabel:>22s} | SIGMMA vs {ref_key:<8s} "
              f"n={len(sigmma_vals)} paired-t p={p:.2e} {mk}")
        y_b = base_y + bracket_step * slot
        x_l, x_r = sorted([COND_KEYS.index(ref_key), sigmma_idx])
        ax.plot([x_l, x_l, x_r, x_r],
                [y_b, y_b + tick_h, y_b + tick_h, y_b],
                color="black", linewidth=0.4, clip_on=False)
        ax.text((x_l + x_r) / 2, y_b + tick_h - yrange * 0.002,
                mk, ha="center", va="bottom",
                fontsize=5.5, fontweight="bold", color="black",
                clip_on=False)

    ax.text(-0.20, 1.02, panel_letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_json", default=str(DATA_JSON),
                    help="normalized_metrics json (per-section MRR + recall)")
    ap.add_argument("--out_dir", default=None,
                    help="directory for retrieval_r1pct_mrr.{png,pdf}; "
                         "defaults to figures/retrieval/")
    args = ap.parse_args()

    data_json = Path(args.data_json)
    if args.out_dir:
        out_png = Path(args.out_dir) / "retrieval_r1pct_mrr.png"
        out_pdf = Path(args.out_dir) / "retrieval_r1pct_mrr.pdf"
    else:
        out_png, out_pdf = OUT_PNG, OUT_PDF

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

    with open(data_json) as f:
        d = json.load(f)
    image_secs = list(d["a01"]["image"]["per_section"].keys())
    spatial_secs = list(d["a01"]["spatial"]["per_section"].keys())
    image_in = [s for s in image_secs if is_in_house(s)]
    spatial_in = [s for s in spatial_secs if is_in_house(s)]
    print(f"image: {len(image_in)} / {len(image_secs)} sections in-house")
    print(f"spatial: {len(spatial_in)} / {len(spatial_secs)} sections in-house")

    def build_pivot(direction, sections, metric_key):
        table = {}
        for k in COND_KEYS:
            table[k] = np.array(
                [d[k][direction]["per_section"][s][metric_key] for s in sections]
            )
        return pd.DataFrame(table, index=sections)

    img_r1pct = build_pivot("image",   image_in,   "recall_at_1pct")
    img_mrr   = build_pivot("image",   image_in,   "mrr")
    spa_r1pct = build_pivot("spatial", spatial_in, "recall_at_1pct")
    spa_mrr   = build_pivot("spatial", spatial_in, "mrr")

    fig, axes = plt.subplots(2, 2, figsize=(5.4, 7.0),
                             gridspec_kw={"wspace": 0.42, "hspace": 0.55})
    plot_panel(axes[0, 0], img_r1pct, "Recall @ 1%",          "a", ymin=0.0)
    plot_panel(axes[0, 1], img_mrr,   "Mean Reciprocal Rank", "b", ymin=0.0)
    plot_panel(axes[1, 0], spa_r1pct, "Recall @ 1%",          "c", ymin=0.0)
    plot_panel(axes[1, 1], spa_mrr,   "Mean Reciprocal Rank", "d", ymin=0.0)

    fig.suptitle(
        r"H&E $\leftrightarrow$ ST cross-modal retrieval",
        fontsize=11, fontweight="bold", y=0.985,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    plt.subplots_adjust(bottom=0.06, top=0.91, hspace=0.55, wspace=0.42)

    # Row sub-titles positioned above each row's axes with clear gap.
    top_ax = axes[0, 0]
    bot_ax = axes[1, 0]
    fig.text(0.5, top_ax.get_position().y1 + 0.030, r"H&E $\rightarrow$ ST",
             ha="center", fontsize=9, fontstyle="italic", color="#444444")
    fig.text(0.5, bot_ax.get_position().y1 + 0.030, r"ST $\rightarrow$ H&E",
             ha="center", fontsize=9, fontstyle="italic", color="#444444")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_pdf}")


if __name__ == "__main__":
    main()
