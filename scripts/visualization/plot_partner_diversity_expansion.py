"""Visualize partner-type diversity expansion across scales.

Two-panel figure from edges_*.csv:

Panel A — Slope plot:
  one line per top-N source cell type, y = effective number of partner types
  (= exp(Shannon entropy of dst_group distribution)) at micro/meso/macro.
  Lines that slope upward = source type connects to a richer set of partner
  types at coarser scales.

Panel B — Stacked composition bar for K top source types:
  for each source type, 3 stacked bars (micro/meso/macro) where each bar is
  segmented by the partner cell types (top-N partners explicitly, rest as
  "other"). Wider color spread at meso/macro = more diversity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def shannon_entropy(s: pd.Series) -> float:
    p = s.value_counts(normalize=True).values
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges_csv", required=True)
    ap.add_argument("--cells_csv", required=True)
    ap.add_argument("--slide", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_n_sources", type=int, default=15,
                    help="slope plot: top-N source groups by total edge count")
    ap.add_argument("--top_k_for_stack", type=int, default=6,
                    help="composition plot: top-K source groups")
    ap.add_argument("--top_partners_explicit", type=int, default=8,
                    help="composition plot: show top-N partner types explicitly, rest as 'other'")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    edges = pd.read_csv(args.edges_csv)
    n_layers = 3
    scale_names = {0: "micro", 1: "meso", 2: "macro"}

    # Rank source groups by total edge count (volume), pick top-N for slope
    src_counts = edges.groupby("src_group").size().sort_values(ascending=False)
    top_src = src_counts.head(args.top_n_sources).index.tolist()
    print(f"[top {args.top_n_sources}] {top_src}")

    # Compute per (src, scale) effective diversity = exp(H_dst)
    rows = []
    for g in top_src:
        for L in range(n_layers):
            sub = edges[(edges["src_group"] == g) & (edges["layer"] == L)]
            H = shannon_entropy(sub["dst_group"]) if len(sub) > 0 else np.nan
            n_distinct = int(sub["dst_group"].nunique())
            eff = float(2 ** H) if not np.isnan(H) else np.nan
            rows.append({"src_group": g, "layer": L, "scale": scale_names[L],
                         "n_edges": len(sub),
                         "n_distinct_dst": n_distinct,
                         "shannon_H": H,
                         "effective_n_partners": eff})
    div = pd.DataFrame(rows)
    div_pivot = div.pivot(index="src_group", columns="scale", values="effective_n_partners")[["micro", "meso", "macro"]]
    div_pivot = div_pivot.loc[top_src]
    print(div_pivot.round(2))

    # ---------- FIGURE ----------
    fig = plt.figure(figsize=(16, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.4])

    # Panel A: slope plot
    axA = fig.add_subplot(gs[0, 0])
    x = np.arange(3)
    cmap = plt.get_cmap("tab20")
    # Sort lines by macro-micro delta to choose highlight ones
    div_pivot["delta"] = div_pivot["macro"] - div_pivot["micro"]
    sorted_src = div_pivot.sort_values("delta", ascending=False).index.tolist()
    # Always-highlight: top 3 grow, top 2 shrink
    highlight = set(sorted_src[:3]) | set(sorted_src[-2:])

    for i, g in enumerate(top_src):
        vals = div_pivot.loc[g, ["micro", "meso", "macro"]].values
        if g in highlight:
            axA.plot(x, vals, "-", marker="o", lw=2.2, ms=7,
                     color=cmap(i % 20), label=g, zorder=3)
            axA.text(2.05, vals[-1], g, fontsize=8.5, va="center", color=cmap(i % 20))
        else:
            axA.plot(x, vals, "-", marker="o", lw=1.0, ms=4,
                     color="lightgrey", alpha=0.6, zorder=1)
    axA.set_xticks(x); axA.set_xticklabels(["micro", "meso", "macro"])
    axA.set_ylabel("effective # partner cell types  (= 2^H)")
    axA.set_xlabel("scale")
    axA.set_title(f"A. Partner-type diversity per source cell type\n(top {args.top_n_sources} by edge count; highlighted: 3 biggest growth, 2 biggest drop)")
    axA.grid(alpha=0.3)

    # Panel B: stacked composition bars
    axB = fig.add_subplot(gs[0, 1])
    top_k_src = src_counts.head(args.top_k_for_stack).index.tolist()
    # Global partner-type palette: take top-N partners overall
    global_partners = edges["dst_group"].value_counts().head(args.top_partners_explicit).index.tolist()
    palette = plt.get_cmap("tab20").colors[: len(global_partners)]
    partner_color = {p: palette[i] for i, p in enumerate(global_partners)}
    partner_color["other"] = "lightgrey"

    bar_w = 0.25
    group_x = np.arange(len(top_k_src))
    for gi, g in enumerate(top_k_src):
        for L, scale in enumerate(["micro", "meso", "macro"]):
            sub = edges[(edges["src_group"] == g) & (edges["layer"] == L)]
            counts = sub["dst_group"].value_counts(normalize=True)  # fractions
            # split into explicit + other
            bottom = 0.0
            for p in global_partners:
                frac = float(counts.get(p, 0.0))
                if frac > 0:
                    axB.bar(group_x[gi] + (L - 1) * bar_w, frac, bottom=bottom,
                            width=bar_w, color=partner_color[p],
                            edgecolor="white", lw=0.3)
                    bottom += frac
            other_frac = 1.0 - bottom
            if other_frac > 1e-6:
                axB.bar(group_x[gi] + (L - 1) * bar_w, other_frac, bottom=bottom,
                        width=bar_w, color=partner_color["other"],
                        edgecolor="white", lw=0.3)
            # scale label under each sub-bar
            axB.text(group_x[gi] + (L - 1) * bar_w, -0.03, scale,
                     ha="center", va="top", fontsize=7, rotation=90)

    axB.set_xticks(group_x); axB.set_xticklabels(top_k_src, fontsize=9, rotation=15, ha="right")
    axB.set_ylabel("share of new edges by partner cell type")
    axB.set_ylim(-0.08, 1.02)
    axB.set_title(f"B. Partner-type composition by scale\n(top {args.top_k_for_stack} source types; explicit colors for top {args.top_partners_explicit} global partners)")
    # legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=partner_color[p]) for p in global_partners] + \
              [plt.Rectangle((0, 0), 1, 1, color=partner_color["other"])]
    labels = global_partners + ["other"]
    axB.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.0, 1.0),
               fontsize=8, frameon=False, title="partner")

    fig.suptitle(f"SIGMMA — partner cell-type diversity widens at coarser scales  ({args.slide})",
                 fontweight="bold")

    for ext in ("pdf", "png"):
        p = out / f"figure_partner_diversity_{args.slide}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"[save] {p}")

    # Print a quick numerical summary
    print("\n[delta in effective # partners (macro - micro), top sources sorted]:")
    print(div_pivot[["micro", "meso", "macro", "delta"]].round(2).sort_values("delta", ascending=False).to_string())


if __name__ == "__main__":
    main()
