"""Spatial overlay: where are new-edge cells, and where do they reach to.

For a chosen pair of contrasting cell types (one "bridging", one "structural"),
plot on the slide's xy plane:
  - all annotated cells in light grey
  - cells of the focus type in solid color
  - new-edge partners (any cell on the other end of an accepted edge whose
    SOURCE was the focus type), colored by partner cell type (top-K)
  - line segments connecting focus -> partner (alpha-weighted by accepted layer)

Two side-by-side panels: bridging type | structural type. Plus a small
overlay legend with destination-type counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _scale_colors():
    return {0: "#d62728", 1: "#2ca02c", 2: "#1f77b4"}


def _line_alpha_by_layer():
    return {0: 0.55, 1: 0.45, 2: 0.35}


def panel(ax, edges, cells, focus_group, top_k_partners=6):
    """Draw single-type spatial overlay panel."""
    # all cells light grey
    ax.scatter(cells["x_centroid"], cells["y_centroid"], s=1.0, c="#dddddd", alpha=0.6, linewidths=0)

    # focus cells (anywhere on the slide that's annotated, not just observed)
    foc_cells = cells[cells["group"] == focus_group]
    ax.scatter(foc_cells["x_centroid"], foc_cells["y_centroid"], s=8,
               c="#222222", alpha=0.9, linewidths=0, label=f"{focus_group} (n={len(foc_cells)})")

    # edges where this group is the source
    sub_e = edges[edges["src_group"] == focus_group].copy()
    n_edges = len(sub_e)
    # destination types: top-K, rest grouped as "other"
    dst_counts = sub_e["dst_group"].value_counts()
    top_dst = dst_counts.head(top_k_partners).index.tolist()
    palette = plt.get_cmap("tab10").colors[: len(top_dst)]
    dst_color = {g: palette[i] for i, g in enumerate(top_dst)}

    scale_palette = _scale_colors()
    line_alpha = _line_alpha_by_layer()

    # Build xy lookup by obs_idx
    coord_map = cells.set_index("obs_idx")[["x_centroid", "y_centroid"]].to_dict("index")

    # Draw lines + partner dots
    for _, e in sub_e.iterrows():
        s_obs = int(e["src_obs"])
        d_obs = int(e["dst_obs"])
        if s_obs not in coord_map or d_obs not in coord_map:
            continue
        s_xy = coord_map[s_obs]
        d_xy = coord_map[d_obs]
        L = int(e["layer"])
        c = scale_palette[L]
        ax.plot([s_xy["x_centroid"], d_xy["x_centroid"]],
                [s_xy["y_centroid"], d_xy["y_centroid"]],
                c=c, alpha=line_alpha[L], lw=0.4, zorder=1)

    # Partner dots colored by group
    for g, c in dst_color.items():
        dst_obs = sub_e[sub_e["dst_group"] == g]["dst_obs"].unique()
        dst_cells = cells[cells["obs_idx"].isin(dst_obs)]
        if len(dst_cells):
            ax.scatter(dst_cells["x_centroid"], dst_cells["y_centroid"],
                       s=6, c=[c], alpha=0.85, linewidths=0,
                       label=f"{g}  n={dst_counts[g]}")
    # "other" partners
    other_obs = sub_e[~sub_e["dst_group"].isin(top_dst)]["dst_obs"].unique()
    other_cells = cells[cells["obs_idx"].isin(other_obs)]
    if len(other_cells):
        ax.scatter(other_cells["x_centroid"], other_cells["y_centroid"],
                   s=4, c="grey", alpha=0.6, linewidths=0,
                   label=f"other  n={dst_counts.iloc[top_k_partners:].sum()}")

    ax.set_aspect("equal")
    ax.set_title(f"{focus_group}   edges={n_edges}   cross-type rate={sub_e['cross_type'].mean():.0%}")
    ax.legend(loc="upper right", fontsize=7, frameon=False, markerscale=1.5)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges_csv", required=True)
    ap.add_argument("--cells_csv", required=True)
    ap.add_argument("--slide", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bridging_group", default="F3: FRC-like")
    ap.add_argument("--structural_group", default="Sweat gland")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    edges = pd.read_csv(args.edges_csv)
    cells = pd.read_csv(args.cells_csv)
    print(f"[load] edges={len(edges)}  cells={len(cells)}")

    # Need cell coordinates for ALL cells in the slide, not just observed.
    # The cells_csv only has those that appeared in a forwarded FOV. For
    # background scatter we'd ideally show every cell, but cells_csv is what
    # we have. Background still works for context if observed coverage is
    # spatially representative.

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    panel(axes[0], edges, cells, args.bridging_group)
    panel(axes[1], edges, cells, args.structural_group)

    # legend caption with scale color key
    sc = _scale_colors()
    fig.text(0.5, -0.02,
             f"line color = scale (micro {sc[0]}, meso {sc[1]}, macro {sc[2]})",
             ha="center", fontsize=10)
    fig.suptitle(f"SIGMMA new edges spatial overlay — bridging vs structural  ({args.slide})",
                 fontweight="bold")

    pdf = out / f"figure_spatial_overlay_{args.slide}.pdf"
    png = out / f"figure_spatial_overlay_{args.slide}.png"
    fig.savefig(pdf, dpi=200, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"[save] {pdf}\n[save] {png}")


if __name__ == "__main__":
    main()
