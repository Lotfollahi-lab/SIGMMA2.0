"""Focused figure: bridging vs structural cell-type dichotomy.

Panel A — Cross-type rate ranking
  Horizontal bar chart, top-N cell types sorted descending by cross-type rate.
  3 "bridging" exemplars highlighted in RED, 3 "structural" exemplars in BLUE,
  others in grey. Reference lines at 0.5 (random) and slide-mean.

Panel B — Partner-type composition for the 6 focal groups
  3 bridging on top + 3 structural on bottom; each as horizontal stacked bar
  where the SAME-type slice is drawn solid in the group's category color and
  cross-type partners are colored from a tab20 palette. Slice widths = fraction
  of accepted edges going to each partner type.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


BRIDGING_COLOR = "#d62728"
STRUCTURAL_COLOR = "#1f77b4"
OTHER_COLOR = "#bcbcbc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges_csv", required=True)
    ap.add_argument("--cells_csv", required=True)
    ap.add_argument("--slide", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_n_groups", type=int, default=20)
    ap.add_argument("--bridging", nargs="+",
                    default=["F3: FRC-like", "Mac", "F2/3: Perivascular"])
    ap.add_argument("--structural", nargs="+",
                    default=["Sweat gland", "KC_HF: Basal", "Smooth muscle"])
    ap.add_argument("--top_partners_explicit", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    edges = pd.read_csv(args.edges_csv)
    cells = pd.read_csv(args.cells_csv)

    # ----- compute per-group cross-type rate (avg over 3 scales) -----
    rate = (edges.groupby(["src_group"])["cross_type"]
            .mean().rename("cross_type_rate"))
    n_edges = edges.groupby("src_group").size().rename("n_edges")
    n_cells = cells.groupby("group").size().rename("n_cells")
    df = pd.concat([rate, n_edges, n_cells], axis=1).fillna(0)
    df = df[df["n_cells"] > 0].sort_values("n_cells", ascending=False)
    top_df = df.head(args.top_n_groups).sort_values("cross_type_rate", ascending=False)

    slide_mean = edges["cross_type"].mean()
    print(f"[slide-mean cross-type rate] {slide_mean:.3f}")
    print(top_df.round(3))

    bridging = list(args.bridging)
    structural = list(args.structural)
    focal_groups = bridging + structural

    # ----- FIGURE -----
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.5], height_ratios=[1, 1])

    # ----- Panel A: sorted bar -----
    axA = fig.add_subplot(gs[:, 0])
    y = np.arange(len(top_df))
    color = [
        BRIDGING_COLOR if g in bridging else
        STRUCTURAL_COLOR if g in structural else
        OTHER_COLOR
        for g in top_df.index
    ]
    bars = axA.barh(y, top_df["cross_type_rate"].values, color=color,
                    edgecolor="black", linewidth=0.4, alpha=0.95)
    for i, (g, r) in enumerate(zip(top_df.index, top_df["cross_type_rate"].values)):
        axA.text(r + 0.01, i, f"{r:.0%}", va="center", fontsize=8.5,
                 color="black", fontweight="bold" if g in focal_groups else "normal")

    axA.set_yticks(y)
    yticklabels = []
    for g in top_df.index:
        if g in focal_groups:
            mark = "  ←" if g in bridging else "  ←"
            yticklabels.append(g + mark)
        else:
            yticklabels.append(g)
    axA.set_yticklabels(yticklabels, fontsize=9)
    # bold focal labels
    for tick, g in zip(axA.get_yticklabels(), top_df.index):
        if g in focal_groups:
            tick.set_fontweight("bold")
            tick.set_color(BRIDGING_COLOR if g in bridging else STRUCTURAL_COLOR)
    axA.invert_yaxis()
    axA.set_xlim(0, 1.12)
    axA.axvline(0.5, ls="--", c="black", lw=0.8, alpha=0.6, label="random (50%)")
    axA.axvline(slide_mean, ls="-", c="darkgreen", lw=1.2, alpha=0.6,
                label=f"slide mean ({slide_mean:.0%})")
    axA.set_xlabel("cross-type rate  (fraction of new edges going to a different cell type)")
    axA.set_title("A. Cross-type rate ranking of top 20 cell types\nred = bridging exemplars   blue = structural exemplars")
    axA.legend(loc="lower right", fontsize=9, frameon=False)

    # ----- Panel B: stacked partner composition for focal groups -----
    # build a partner palette: top-K global partners get explicit colors;
    # rest collapsed as "other (cross-type)"
    K = args.top_partners_explicit
    global_partners = edges["dst_group"].value_counts().head(K).index.tolist()
    palette = list(plt.get_cmap("tab20").colors[:K])
    partner_color = {p: palette[i] for i, p in enumerate(global_partners)}

    # We split each focal group's edges into (same-type) + (top-K partners) + (other cross-type)
    def compose(g, color_solid):
        sub = edges[edges["src_group"] == g]
        n = len(sub)
        if n == 0:
            return [], [], 0
        same_frac = float((sub["dst_group"] == g).mean())
        # cross-type composition over top-K explicit + rest
        cross = sub[sub["dst_group"] != g]
        comp = cross["dst_group"].value_counts() / n
        segs = [(g + " (same-type)", same_frac, color_solid)]
        for p in global_partners:
            if p == g:
                continue
            v = float(comp.get(p, 0.0))
            if v > 0:
                segs.append((p, v, partner_color[p]))
        explicit_frac = sum(v for _, v, _ in segs)
        rest_frac = max(0.0, 1.0 - explicit_frac)
        if rest_frac > 1e-6:
            segs.append(("other cross-type", rest_frac, "#dddddd"))
        return segs, comp.index.tolist(), n

    axB_top = fig.add_subplot(gs[0, 1])
    axB_bot = fig.add_subplot(gs[1, 1], sharex=axB_top)

    def draw_panel(ax, groups, color_solid, title):
        for i, g in enumerate(groups):
            segs, _, n = compose(g, color_solid)
            left = 0.0
            for label, frac, c in segs:
                ax.barh(i, frac, left=left, color=c, edgecolor="white", lw=0.5)
                # annotate solid same-type slice
                if "(same-type)" in label:
                    ax.text(left + frac / 2, i, f"{frac:.0%} same",
                            ha="center", va="center", fontsize=8,
                            color="white", fontweight="bold")
                left += frac
            # group label + n
            ax.text(-0.02, i, f"{g}\nn={n}", ha="right", va="center", fontsize=9.5,
                    fontweight="bold", color=color_solid)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels([""] * len(groups))
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_title(title, fontsize=11, color=color_solid, fontweight="bold")
        for spine in ["right", "top"]:
            ax.spines[spine].set_visible(False)

    draw_panel(axB_top, bridging, BRIDGING_COLOR,
               "B-top. Bridging: edges spread across MANY partner types")
    draw_panel(axB_bot, structural, STRUCTURAL_COLOR,
               "B-bottom. Structural: edges concentrate on SAME or related types")
    axB_bot.set_xlabel("fraction of new edges by partner cell type")

    # Legend for panel B (partner color key)
    handles = []
    for p in global_partners:
        handles.append(mpatches.Patch(color=partner_color[p], label=p))
    handles.append(mpatches.Patch(color="#dddddd", label="other cross-type"))
    handles.append(mpatches.Patch(color=BRIDGING_COLOR, label="(focal bridging) same-type"))
    handles.append(mpatches.Patch(color=STRUCTURAL_COLOR, label="(focal structural) same-type"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.72, -0.06),
               ncols=4, fontsize=8, frameon=False, title="partner cell type")

    fig.suptitle(f"SIGMMA new edges — bridging vs structural cell types  ({args.slide})",
                 fontsize=13, fontweight="bold")

    for ext in ("pdf", "png"):
        p = out / f"figure_bridging_vs_structural_{args.slide}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"[save] {p}")


if __name__ == "__main__":
    main()
