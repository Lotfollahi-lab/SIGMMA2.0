"""Multi-slide replication of the scale orthogonality result.

Aggregates per-slide cosine summaries from
  outputs/diagnostics/scale_correlation/<JOBID>/scale_correlation_summary_<slide>.csv
and produces a single Nature-style bar figure: bar = mean across slides for each
correlation category, error bar = s.d. across slides, dots = individual slides.

Each category's value is itself a per-slide mean from a large pair sample
(within-scale 5×10⁵, cross-scale 2×10⁵, same-cell 1.7×10⁴), so the std bar here
captures *between-slide* reproducibility.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# (jobid, sample_name, short_tissue) ordered for plotting.
RUNS = [
    ("18670", "BK39-SKI-27-FO-1-S3",       "skin"),
    ("21176", "BK39-SKI-27-FO-3-S3",       "skin"),
    ("21177", "BK46-SKI-27-FO-2-s3",       "skin"),
    ("21178", "BK25-SKI-27-FO-5-S13-B1",   "skin"),
    ("21179", "BK30-SKI-28-FO-1-s4",       "skin"),
    ("21180", "CE4-SKI-27-FO-4-S22-A2",    "embryo"),
    ("21181", "DF15-LIV-0-FFPE-1-S1-i",    "liver"),
]

# Bar order + labels.
CATEGORIES = [
    ("crosscell_samescale_scale0", "within\nmicro",    "within"),
    ("crosscell_samescale_scale1", "within\nmeso",     "within"),
    ("crosscell_samescale_scale2", "within\nmacro",    "within"),
    ("samecell_crossscale_0vs1",   "same-cell\nmicro–meso",  "samecell"),
    ("samecell_crossscale_0vs2",   "same-cell\nmicro–macro", "samecell"),
    ("samecell_crossscale_1vs2",   "same-cell\nmeso–macro",  "samecell"),
    ("crosscell_crossscale_0vs1",  "cross-cell\nmicro–meso",  "crosscell"),
    ("crosscell_crossscale_0vs2",  "cross-cell\nmicro–macro", "crosscell"),
    ("crosscell_crossscale_1vs2",  "cross-cell\nmeso–macro",  "crosscell"),
]

GROUP_COLOR = {
    "within":    "#1f77b4",  # blue — within-scale baseline
    "samecell":  "#d62728",  # red — same-cell cross-scale (the punchline)
    "crosscell": "#888888",  # grey — random-pair cross-scale baseline
}

TISSUE_MARKER = {"skin": "o", "embryo": "^", "liver": "s"}

NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 5.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("outputs/diagnostics/scale_correlation"))
    ap.add_argument("--out", type=Path,
                    default=Path("figures/diagnostics/scale_orthogonality_multislide.png"))
    args = ap.parse_args()

    # ---- collect per-slide means ----
    rows = []
    for jid, slide, tissue in RUNS:
        csv = args.root / jid / f"scale_correlation_summary_{slide}.csv"
        if not csv.exists():
            print(f"[skip] {jid} {slide} (missing summary)")
            continue
        df = pd.read_csv(csv)
        for cat_key, _, _ in CATEGORIES:
            sub = df[df["name"] == cat_key]
            if sub.empty:
                continue
            rows.append({
                "slide":   slide,
                "tissue":  tissue,
                "category": cat_key,
                "mean":    float(sub["mean"].iloc[0]),
            })
    long = pd.DataFrame(rows)
    print(long.pivot(index="slide", columns="category", values="mean").round(3))

    # ---- aggregate across slides ----
    agg_means, agg_stds, agg_points = [], [], []
    for cat_key, _, _ in CATEGORIES:
        vals = long.loc[long["category"] == cat_key, "mean"].to_numpy()
        agg_means.append(vals.mean())
        agg_stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        agg_points.append(vals)

    # also save aggregated csv next to the figure
    agg_df = pd.DataFrame({
        "category": [c for c, _, _ in CATEGORIES],
        "mean_across_slides": agg_means,
        "std_across_slides":  agg_stds,
        "n_slides": [len(p) for p in agg_points],
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(args.out.with_suffix(".csv"), index=False)
    print(f"\nsaved aggregate csv -> {args.out.with_suffix('.csv')}")
    print(agg_df.to_string(index=False))

    with mpl.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(6.6, 2.6))
        x = np.arange(len(CATEGORIES))
        width = 0.7

        colors = [GROUP_COLOR[g] for _, _, g in CATEGORIES]
        ax.bar(x, agg_means, width, yerr=agg_stds, capsize=2.5,
               color=colors, edgecolor="black", linewidth=0.5,
               error_kw=dict(ecolor="black", lw=0.7))

        # per-slide dots (jittered).
        rng = np.random.RandomState(7)
        for i, (cat_key, _, _) in enumerate(CATEGORIES):
            sub = long[long["category"] == cat_key]
            jit = rng.uniform(-0.13, 0.13, size=len(sub))
            for (_, r), j in zip(sub.iterrows(), jit):
                ax.scatter(x[i] + j, r["mean"],
                           marker=TISSUE_MARKER.get(r["tissue"], "o"),
                           s=12, color="black", alpha=0.7,
                           linewidths=0, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab, _ in CATEGORIES])
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_ylabel("Mean cosine similarity")
        ax.set_title("Scale orthogonality replication across 7 slides", pad=6)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

        # category-group legend
        group_handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g],
                                       label={"within": "within-scale (different cells)",
                                              "samecell": "same-cell, cross-scale",
                                              "crosscell": "cross-cell, cross-scale"}[g])
                         for g in ["within", "samecell", "crosscell"]]
        # tissue-marker legend
        tissue_handles = [plt.Line2D([0], [0], marker=m, linestyle="",
                                     markerfacecolor="black", markeredgecolor="none",
                                     markersize=4, label=t)
                          for t, m in TISSUE_MARKER.items()]
        leg1 = ax.legend(handles=group_handles, loc="upper right",
                         frameon=False, ncol=1, handlelength=1.2,
                         labelspacing=0.25, borderpad=0.2)
        ax.add_artist(leg1)
        ax.legend(handles=tissue_handles, loc="center right",
                  bbox_to_anchor=(1.0, 0.55),
                  frameon=False, ncol=3, handletextpad=0.3,
                  columnspacing=0.7, labelspacing=0.15)

        fig.text(0.5, -0.04,
                 "n = 7 slides (5 skin biopsies, 1 embryo, 1 liver); "
                 "each dot = per-slide mean over its pair sample; bar = mean ± s.d. across slides",
                 ha="center", fontsize=5.5, color="0.3")
        fig.tight_layout()

        fig.savefig(args.out, dpi=600, bbox_inches="tight")
        fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
        print(f"\nsaved {args.out} and {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
