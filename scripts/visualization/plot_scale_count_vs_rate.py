"""Tiny 2-panel summary plot: absolute counts/acceptance-rate climb with scale,
but cross-type rate stays flat.

Panel A: BK24-SKI-28-FO-1-s3 (n=64 FOVs)
  - bars: per-FOV mean accepted-edge count
  - line (secondary y): acceptance rate (= accepted / proposed)

Panel B: BK23-SKI-27-FO-1-S6-C2 (n=256 FOVs)
  - bars: total accepted-edge count
  - line (secondary y): cross-type rate
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SCALES = ["micro", "meso", "macro"]
PALETTE = ["#d62728", "#2ca02c", "#1f77b4"]

# BK24 numbers from logs/job_logs/diag_edges.25614.out
bk24_accepted_mean = [18.6, 26.3, 23.4]
bk24_accepted_std  = [26.6, 35.6, 32.6]
bk24_acceptance_rate = [0.052, 0.141, 0.198]

# BK23 numbers from logs/job_logs/plot_edge_cells.25631.out
bk23_total_accepted = [1190, 1955, 1883]
bk23_cross_type_rate = [0.725, 0.723, 0.713]

out_dir = Path("figures/diagnostics/edge_candidate_cells")
out_dir.mkdir(parents=True, exist_ok=True)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)

# --- Panel A ---
x = np.arange(3)
bars = axA.bar(x, bk24_accepted_mean, yerr=bk24_accepted_std, color=PALETTE,
               capsize=4, alpha=0.85, edgecolor="black", linewidth=0.5)
for i, b in enumerate(bars):
    axA.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
             f"{bk24_accepted_mean[i]:.1f}", ha="center", va="bottom", fontsize=9)
axA.set_xticks(x); axA.set_xticklabels(SCALES)
axA.set_ylabel("per-FOV mean accepted edges  (+/- std)")
axA.set_ylim(0, max(bk24_accepted_mean) + max(bk24_accepted_std) + 5)
axA.set_title("A. Volume climbs with scale\n(BK24, n=64 FOVs)")

axA2 = axA.twinx()
axA2.plot(x, bk24_acceptance_rate, "o-", color="black", lw=2, markersize=7)
for i, r in enumerate(bk24_acceptance_rate):
    axA2.text(i, r + 0.01, f"{r:.1%}", ha="center", va="bottom",
              fontsize=10, fontweight="bold")
axA2.set_ylabel("acceptance rate  (= accepted / proposed)")
axA2.set_ylim(0, 0.30)
axA2.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))

# --- Panel B ---
bars = axB.bar(x, bk23_total_accepted, color=PALETTE, alpha=0.85,
               edgecolor="black", linewidth=0.5)
for i, b in enumerate(bars):
    axB.text(b.get_x() + b.get_width() / 2, b.get_height() + 30,
             f"{bk23_total_accepted[i]}", ha="center", va="bottom", fontsize=9)
axB.set_xticks(x); axB.set_xticklabels(SCALES)
axB.set_ylabel("total accepted edges")
axB.set_ylim(0, max(bk23_total_accepted) * 1.15)
axB.set_title("B. Cross-type rate stays flat\n(BK23, n=256 FOVs)")

axB2 = axB.twinx()
axB2.plot(x, bk23_cross_type_rate, "o-", color="black", lw=2, markersize=7)
for i, r in enumerate(bk23_cross_type_rate):
    axB2.text(i, r + 0.005, f"{r:.1%}", ha="center", va="bottom",
              fontsize=10, fontweight="bold")
axB2.set_ylabel("cross-type rate  (fraction of edges to different cell type)")
axB2.set_ylim(0, 1.0)
axB2.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
axB2.axhline(0.725, ls="--", color="grey", lw=0.7, alpha=0.6, label="micro level")

fig.suptitle("What changes with scale: volume & acceptance rate, NOT cross-type ratio",
             fontsize=12, fontweight="bold")

for ext in ("pdf", "png"):
    p = out_dir / f"figure_scale_count_vs_rate.{ext}"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print(f"[save] {p}")
