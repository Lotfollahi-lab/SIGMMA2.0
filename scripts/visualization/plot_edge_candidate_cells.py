"""Make multi-panel figure from edge-candidate-cells extractor output.

Inputs: edges_{slide}.csv + cells_{slide}.csv from
scripts/diagnose_edge_candidate_cells.py.

Panels:
  A — per-cell-type new-edge frequency per scale (heatmap, top N groups)
  B — cross-type rate per cell type per scale (sorted bars)
  C — orig_degree (mask=-1 incident edges) vs new_total_degree scatter,
      colored by top groups + Spearman corr annotated
  D — cross-type vs same-type edge counts per scale (stacked bars, total)

Also writes a derived CSV with per-(group, scale) rates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _palette_scales():
    # Project standard: micro #d62728, meso #2ca02c, macro #1f77b4
    return {0: "#d62728", 1: "#2ca02c", 2: "#1f77b4"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges_csv", required=True)
    ap.add_argument("--cells_csv", required=True)
    ap.add_argument("--slide", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_n_groups", type=int, default=20,
                    help="rank by total cells in CSV and keep top-N for plotting")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    edges = pd.read_csv(args.edges_csv)
    cells = pd.read_csv(args.cells_csv)
    print(f"[load] edges={len(edges)}  cells={len(cells)}  "
          f"groups={cells['group'].nunique()}")

    num_layers = 3
    scale_names = {0: "micro", 1: "meso", 2: "macro"}
    palette = _palette_scales()

    # Rank top-N groups by cell count
    group_n = cells.groupby("group").size().sort_values(ascending=False)
    top_groups = group_n.head(args.top_n_groups).index.tolist()
    print(f"[top {args.top_n_groups}]\n{group_n.head(args.top_n_groups)}\n")

    # --- Per-(group, layer) derived stats ---
    rows = []
    for g in top_groups:
        sub_c = cells[cells["group"] == g]
        for L in range(num_layers):
            new_in = float(sub_c[f"new_in_l{L}"].mean())
            new_out = float(sub_c[f"new_out_l{L}"].mean())
            # symmetric proposal => in==out approximately; use mean
            new_deg = 0.5 * (new_in + new_out)
            sub_e = edges[(edges["src_group"] == g) & (edges["layer"] == L)]
            n_e = len(sub_e)
            cross = float(sub_e["cross_type"].mean()) if n_e > 0 else np.nan
            rows.append({
                "group": g, "layer": L, "scale": scale_names[L],
                "n_cells": int(len(sub_c)),
                "new_degree_per_cell": new_deg,
                "n_edges_as_src": n_e,
                "cross_type_rate": cross,
            })
    derived = pd.DataFrame(rows)
    derived.to_csv(out / f"derived_{args.slide}.csv", index=False)

    # --- Figure ---
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # Panel A: heatmap of new-degree-per-cell across (group, layer)
    axA = fig.add_subplot(gs[0, 0:2])
    mat = derived.pivot(index="group", columns="layer", values="new_degree_per_cell").loc[top_groups]
    im = axA.imshow(mat.values, aspect="auto", cmap="viridis")
    axA.set_yticks(range(len(top_groups)))
    axA.set_yticklabels(top_groups, fontsize=9)
    axA.set_xticks(range(num_layers))
    axA.set_xticklabels([scale_names[i] for i in range(num_layers)])
    axA.set_xlabel("scale")
    axA.set_title("A. New-edge degree per cell, by cell type × scale")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            axA.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center",
                     fontsize=7, color="w" if mat.values[i, j] < mat.values.max() * 0.6 else "k")
    plt.colorbar(im, ax=axA, label="new edges / cell (mean)")

    # Panel B: cross-type rate per group per scale, horizontal bars per layer
    axB = fig.add_subplot(gs[0, 2])
    width = 0.27
    y = np.arange(len(top_groups))
    for L in range(num_layers):
        vals = derived[derived["layer"] == L].set_index("group").loc[top_groups, "cross_type_rate"].values
        axB.barh(y + (L - 1) * width, vals, height=width, color=palette[L], label=scale_names[L])
    axB.set_yticks(y)
    axB.set_yticklabels(top_groups, fontsize=8)
    axB.invert_yaxis()
    axB.set_xlabel("fraction of new edges going to a DIFFERENT cell type")
    axB.set_xlim(0, 1.05)
    axB.axvline(0.5, ls="--", c="grey", lw=0.6)
    axB.set_title("B. Cross-type rate (as source)")
    axB.legend(loc="lower right", frameon=False, fontsize=8)

    # Panel C: orig_degree vs new_total_degree scatter
    axC = fig.add_subplot(gs[1, 0])
    new_total = 0.5 * (cells["new_in_total"] + cells["new_out_total"])
    rho_all, p_all = spearmanr(cells["orig_degree"], new_total)
    # Color subset of top groups; rest grey
    colors = mpl.colormaps["tab20"].resampled(len(top_groups))
    for i, g in enumerate(top_groups):
        sub = cells[cells["group"] == g]
        nt = 0.5 * (sub["new_in_total"] + sub["new_out_total"])
        # jitter
        jx = sub["orig_degree"] + np.random.uniform(-0.15, 0.15, len(sub))
        jy = nt + np.random.uniform(-0.15, 0.15, len(sub))
        axC.scatter(jx, jy, s=10, alpha=0.55, color=colors(i), label=g if i < 10 else None)
    rest = cells[~cells["group"].isin(top_groups)]
    if len(rest):
        nt_r = 0.5 * (rest["new_in_total"] + rest["new_out_total"])
        axC.scatter(rest["orig_degree"], nt_r, s=4, alpha=0.2, c="lightgrey", label="other")
    axC.set_xlabel("orig degree (intra-grid mask==-1 neighbors)")
    axC.set_ylabel("new edges (mean of in/out, total over 3 scales)")
    axC.set_title(f"C. Orig vs new degree per cell  Spearman ρ={rho_all:.3f}  p={p_all:.1e}")
    axC.legend(loc="upper left", fontsize=7, ncols=2, frameon=False)

    # Panel D: per-layer cross vs same totals (stacked)
    axD = fig.add_subplot(gs[1, 1])
    bar_x = np.arange(num_layers)
    same_counts = np.array([((edges["layer"] == L) & (~edges["cross_type"])).sum() for L in range(num_layers)])
    cross_counts = np.array([((edges["layer"] == L) & (edges["cross_type"])).sum() for L in range(num_layers)])
    axD.bar(bar_x, same_counts, label="same-type", color="#777777")
    axD.bar(bar_x, cross_counts, bottom=same_counts, label="cross-type", color="#d62728")
    for i, L in enumerate(range(num_layers)):
        tot = same_counts[i] + cross_counts[i]
        rate = cross_counts[i] / max(tot, 1)
        axD.text(i, tot, f"{rate:.0%} cross\n(n={tot})", ha="center", va="bottom", fontsize=9)
    axD.set_xticks(bar_x)
    axD.set_xticklabels([scale_names[L] for L in range(num_layers)])
    axD.set_ylabel("count of accepted edges")
    axD.set_title("D. Cross-type vs same-type per scale")
    axD.legend(frameon=False)

    # Panel E (right of D): per-group new-degree breakdown (stacked across scales)
    axE = fig.add_subplot(gs[1, 2])
    stack_bottom = np.zeros(len(top_groups))
    for L in range(num_layers):
        vals = derived[derived["layer"] == L].set_index("group").loc[top_groups, "new_degree_per_cell"].values
        axE.barh(y, vals, left=stack_bottom, color=palette[L], label=scale_names[L])
        stack_bottom += vals
    axE.set_yticks(y)
    axE.set_yticklabels(top_groups, fontsize=8)
    axE.invert_yaxis()
    axE.set_xlabel("total new-edge degree per cell (sum across scales)")
    axE.set_title("E. New-edge degree per cell, stacked by scale")
    axE.legend(loc="lower right", frameon=False, fontsize=8)

    fig.suptitle(f"SIGMMA new edges — per cell type analysis  ({args.slide})",
                 fontsize=14, fontweight="bold")
    pdf_path = out / f"figure_edge_candidate_cells_{args.slide}.pdf"
    png_path = out / f"figure_edge_candidate_cells_{args.slide}.png"
    fig.savefig(pdf_path, dpi=200, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"[save] {pdf_path}\n[save] {png_path}")

    # Also a print of headline numbers
    print(f"\n[headline]")
    print(f"  total accepted edges = {len(edges)}")
    print(f"  per-layer cross-type rate (global):")
    for L in range(num_layers):
        sub = edges[edges["layer"] == L]
        if len(sub):
            print(f"    {scale_names[L]}: {sub['cross_type'].mean():.3f}  (n={len(sub)})")
    print(f"  Spearman(orig_degree, new_total_degree) over all cells: ρ={rho_all:.3f}  p={p_all:.1e}")

    # Top cross-type "bridging" cell types
    bridging = (derived.groupby("group")["cross_type_rate"]
                .mean().sort_values(ascending=False).head(8))
    print(f"\n[top bridging groups (avg cross-type rate across scales)]\n{bridging}")
    same_type = (derived.groupby("group")["cross_type_rate"]
                 .mean().sort_values().head(8))
    print(f"\n[top within-type groups]\n{same_type}")


if __name__ == "__main__":
    main()
