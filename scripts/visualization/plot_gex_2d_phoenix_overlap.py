"""Phoenix-overlap 2D-GEX figure: zero-shot Phoenix vs SIGMMA / regressor /
contrastive-FT methods on the 16 genes shared by Phoenix's panel and SVG-100.

Two scale-invariant panels (Phoenix's units differ from log1p-count targets, so
absolute MSE / Moran's I are excluded — see aggregate_gex_2d_phoenix_overlap.py):
  (a) Per-gene Pearson  — distribution over the 16 overlap genes, paired *by gene*.
  (b) Per-slide SSIM    — per-slide median over the 16 genes, paired *by slide*.

Significance: paired t-test vs SIGMMA, marker above each box. eva_critique style
(Helvetica 8pt, 40° xticks, dashed UNI2-h reference line, horizontal grid).

Reads:  outputs/gex_2d/svg_panel_2d22ba31_step1000_test/_aggregate/phoenix_overlap_distributions.json
Writes: figures/gex_2d/phoenix_overlap.{png,pdf}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
DIST_JSON = (ROOT / "outputs/gex_2d/svg_panel_2d22ba31_step1000_test/_aggregate"
             / "phoenix_overlap_distributions.json")
OUT_DIR = ROOT / "figures/gex_2d"

# (label, color). Left→right: regressors, Phoenix (zero-shot), UNI2-h, contrastive FT, SIGMMA.
CONDITIONS = [
    ("ST-Net",              "#332288"),
    ("HisToGene",           "#88CCEE"),
    ("DeepPT",              "#4477AA"),
    ("Phoenix (zero-shot)", "#E69F00"),
    ("UNI2-h no FT",        "#888888"),
    ("BLEEP FT",            "#882255"),
    ("CLIP FT",             "#44AA99"),
    ("PLIP FT",             "#DDCC77"),
    ("OmiCLIP FT",          "#999933"),
    ("SIGMMA",              "#d62728"),
]
UNI_LABEL = "UNI2-h no FT"
SIGMMA_LABEL = "SIGMMA"


def sig_marker(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def pearson_arr(dists, label):
    return np.asarray(dists.get(label, {}).get("pearson_per_gene", []), dtype=float)


def ssim_arr(dists, label):
    return dists.get(label, {}).get("ssim_per_slide", {})  # dict {slide: val}


def paired_p_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Paired by gene (both arrays are the 16 overlap genes, same order)."""
    n = min(len(a), len(b))
    if n < 2:
        return float("nan")
    return float(ttest_rel(a[:n], b[:n]).pvalue)


def paired_p_slide(a: dict, b: dict) -> float:
    """Paired by slide on the shared slide ids."""
    keys = sorted(set(a) & set(b))
    if len(keys) < 2:
        return float("nan")
    return float(ttest_rel([a[k] for k in keys], [b[k] for k in keys]).pvalue)


def plot_panel(ax, data, colors, labels, ylabel, panel_letter, ref_idx, sig_idx, pvals):
    x = np.arange(len(data))
    bp = ax.boxplot(
        data, positions=x, widths=0.6, patch_artist=True,
        showcaps=True, showfliers=False, whis=1.5,
        boxprops=dict(linewidth=0.6),
        medianprops=dict(color="black", linewidth=1.0),
        whiskerprops=dict(linewidth=0.6, color="black"),
        capprops=dict(linewidth=0.6, color="black"),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i])
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2 if i == sig_idx else 0.6)
        patch.set_alpha(0.9)

    ax.axhline(float(np.median(data[ref_idx])), color="#888888", linestyle="--",
               linewidth=0.6, alpha=0.7, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="-", linewidth=0.3, color="#cccccc", alpha=0.7, zorder=-1)
    ax.set_axisbelow(True)

    tops = [np.percentile(d, 75) + 1.5 * (np.percentile(d, 75) - np.percentile(d, 25))
            for d in data]
    ymax = max(tops)
    for i, p in enumerate(pvals):
        if i == sig_idx or not np.isfinite(p):
            continue
        ax.text(x[i], min(tops[i] + 0.01 * ymax, ymax * 1.02), sig_marker(p),
                ha="center", va="bottom", fontsize=6)
    ax.text(-0.12, 1.04, panel_letter, transform=ax.transAxes, fontsize=10, fontweight="bold")


def main():
    blob = json.load(open(DIST_JSON))
    dists = blob["distributions"]
    n_genes = len(blob["overlap_genes"])
    present = [(l, c) for l, c in CONDITIONS if l in dists]
    labels = [l for l, _ in present]
    colors = [c for _, c in present]
    ref_idx = labels.index(UNI_LABEL)
    sig_idx = labels.index(SIGMMA_LABEL)

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
    })
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))

    # Panel a — Pearson (paired by gene)
    p_data = [pearson_arr(dists, l) for l in labels]
    sig_p = pearson_arr(dists, SIGMMA_LABEL)
    p_pvals = [paired_p_pearson(d, sig_p) for d in p_data]
    plot_panel(axes[0], p_data, colors, labels,
               r"Per-gene Pearson  $\uparrow$", "a", ref_idx, sig_idx, p_pvals)
    axes[0].set_title(f"paired by gene (n={n_genes})", fontsize=7, color="#555555")

    # Panel b — SSIM (paired by slide)
    s_dicts = [ssim_arr(dists, l) for l in labels]
    s_data = [np.asarray(list(d.values()), dtype=float) for d in s_dicts]
    sig_s = ssim_arr(dists, SIGMMA_LABEL)
    s_pvals = [paired_p_slide(d, sig_s) for d in s_dicts]
    n_slides = len(sig_s)
    plot_panel(axes[1], s_data, colors, labels,
               r"Per-slide SSIM  $\uparrow$", "b", ref_idx, sig_idx, s_pvals)
    axes[1].set_title(f"paired by slide (n={n_slides})", fontsize=7, color="#555555")

    fig.suptitle(f"H&E → ST  ·  zero-shot Phoenix vs SIGMMA on the "
                 f"Phoenix∩SVG-100 overlap ({n_genes} genes)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"phoenix_overlap.{ext}", dpi=300, bbox_inches="tight")
    print(f"[figure] wrote {OUT_DIR/'phoenix_overlap.png'} (and .pdf)")
    # stars summary
    for l, pp, sp in zip(labels, p_pvals, s_pvals):
        print(f"  {l:22s} SIGMMA-vs:  Pearson {sig_marker(pp):3s} (p={pp:.3g})  "
              f"SSIM {sig_marker(sp):3s} (p={sp:.3g})")


if __name__ == "__main__":
    main()
