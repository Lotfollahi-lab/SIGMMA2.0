"""Histogram of GT cell counts per FOV (MoNuSAC test, n=85).

Highlights low-cell-count FOVs (n_cells ≤ 2) as outliers — these patches
are statistically fragile (single instance error → 50% PQ drop) and dominated
by the EVA 224×224 resize protocol artifact (small native FOVs upscaled).

Output: figures/eva_critique/n_cells_histogram.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
OUT = ROOT / "figures" / "eva_critique"

rows = json.load(open("/tmp/fov_strat.json"))
n_cells = np.array([r["n_cells"] for r in rows])

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(8.0, 3.6))
gs = GridSpec(4, 2, figure=fig, hspace=0.20, wspace=0.30, width_ratios=[1, 1])
# Left column: pooled (spans all 4 rows) — becomes panel a
ax_pooled = fig.add_subplot(gs[:, 0])
# Right column: 4 stacked tissue panels — becomes panel b
ax_tissues = [fig.add_subplot(gs[i, 1]) for i in range(4)]
axes = [ax_tissues, ax_pooled]

n_outlier = int((n_cells <= 2).sum())
n_empty = int((n_cells == 0).sum())
n_lowish = int(((n_cells > 2) & (n_cells <= 10)).sum())

# Panel a: 4 stacked tissue histograms + KDE (one row per tissue)
organ_color = {"Lung": "#1f77b4", "Kidney": "#2ca02c", "Prostate": "#ff7f0e", "Breast": "#d62728"}
organs = [r["organ"] for r in rows]
log_bins_a = np.geomspace(1, n_cells.max() + 1, 25)
xs_log_a = np.linspace(0, np.log10(n_cells.max()) + 0.2, 400)
tissues_ordered = ["Lung", "Kidney", "Prostate", "Breast"]
for i, organ in enumerate(tissues_ordered):
    ax = axes[0][i]
    mask = np.array([o == organ for o in organs])
    n_o = n_cells[mask]
    n_o_safe = np.where(n_o == 0, 1.0, n_o)
    # Histogram (count axis)
    ax.hist(n_o_safe, bins=log_bins_a, color=organ_color[organ], alpha=0.5,
            edgecolor="black", linewidth=0.4)
    # KDE on twinx
    ax_k = ax.twinx()
    if mask.sum() >= 3:
        kde_o = gaussian_kde(np.log10(n_o_safe), bw_method=0.40)
        y = kde_o(xs_log_a)
        ax_k.fill_between(10**xs_log_a, y, color=organ_color[organ], alpha=0.25, zorder=1)
    ax_k.set_yticks([])
    ax_k.spines["top"].set_visible(False)
    ax_k.spines["right"].set_visible(False)
    # Outlier cutoff
    ax.axvline(2.5, color="#d62728", linestyle="--", linewidth=0.8, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(1.0, n_cells.max() * 1.2)
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Tissue label inside top-left
    ax.text(0.03, 0.92, f"{organ}  (n={int(mask.sum())})", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color=organ_color[organ], fontweight="bold")
    if i < 3:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    else:
        ax.set_xlabel("GT cell count per FOV (log scale)")
    if i == 0:
        ax.set_title("b", fontsize=10, fontweight="bold", loc="left")

# Panel b: log-scale full histogram + KDE in log space (x starts at 1)
ax = axes[1]
ax_pooled = ax  # alias
log_bins = np.geomspace(1, n_cells.max()+1, 25)
n_for_log = np.where(n_cells == 0, 1.0, n_cells)  # treat empties as 1 (no n=0 in this dataset anyway)
n_main = n_for_log[n_cells > 2]
n_out  = n_for_log[n_cells <= 2]
ax.hist(n_main, bins=log_bins, color="#888888", edgecolor="black", linewidth=0.5,
        alpha=0.75, label=f"n_cells > 2 (n={len(n_main)})")
ax.hist(n_out,  bins=log_bins, color="#d62728", edgecolor="black", linewidth=0.5,
        alpha=0.85, label=f"n_cells ≤ 2 (n={len(n_out)}, outlier)")
ax.set_xscale("log")
ax.set_xlim(1.0, n_cells.max() * 1.2)
# KDE in log space — kde over log10(n_cells) then transform back for plotting
log_n = np.log10(n_for_log)
ax2 = ax.twinx()
kde_log = gaussian_kde(log_n, bw_method=0.30)
xs_log = np.linspace(0, log_n.max() + 0.2, 400)
ax2.fill_between(10**xs_log, kde_log(xs_log), color="#555555", alpha=0.25, zorder=1,
                 label="KDE on log10(n)")
ax2.set_ylabel("density (KDE)", color="#555555", fontsize=7)
ax2.tick_params(axis="y", labelsize=6.5, colors="#555555")
ax2.spines["top"].set_visible(False)
ax2.set_ylim(0, ax2.get_ylim()[1] * 1.2)
ax.axvline(2.5, color="#d62728", linestyle="--", linewidth=1.0)
ax.set_xlabel("GT cell count per FOV (log scale)")
ax.set_ylabel("# FOVs")
ax.set_title("a", fontsize=10, fontweight="bold", loc="left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=6.5, frameon=False,
          loc="upper left", bbox_to_anchor=(0.12, 1.0))

fig.suptitle("FOV GT cell count distribution — MoNuSAC test (n=85)", fontsize=9, y=1.02)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
for ext in ["pdf", "png"]:
    out = OUT / f"n_cells_histogram.{ext}"
    fig.savefig(out, bbox_inches="tight", dpi=300 if ext == "png" else None)
    print(f"saved {out}")
print(f"\n0 cells: {n_empty}; ≤2 cells: {n_outlier}; 3-10: {n_lowish}; total: {len(n_cells)}")
