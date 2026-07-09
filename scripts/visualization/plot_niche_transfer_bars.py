"""Bar plot of cell-niche transfer (mean +/- std across held-out slides).

Nature-journal-style compact figure: sans-serif, ~7pt labels, 2-column width.
Bars: UNI (CLS-token baseline) + SIGMMA micro/meso/macro.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EMBEDDINGS = ["z_uni_macro_cls", "z_sigmma_micro", "z_sigmma_meso", "z_sigmma_macro"]
EMB_LABELS = {
    "z_uni_macro_cls": "UNI",
    "z_sigmma_micro":  "micro",
    "z_sigmma_meso":   "meso",
    "z_sigmma_macro":  "macro",
}
# Palette matched to scripts/eva_critique/plot_instance_evidence.py
EMB_COLORS = {
    "z_uni_macro_cls": "#888888",
    "z_sigmma_micro":  "#d62728",
    "z_sigmma_meso":   "#2ca02c",
    "z_sigmma_macro":  "#1f77b4",
}

METRICS = [("balanced_accuracy", "Balanced accuracy")]
LABELS = [("niche_category", "niche_category (8)"),
          ("niches_terra",  "niches_terra (25)")]

# ---- Nature-style rcParams (single-column ~89mm, double-column ~183mm) ----
NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path,
                        default=Path("outputs/cell_niche/21052/probe_metrics_per_slide.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("figures/cell_niche/21052/bars_niche_transfer.png"))
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["split"].isin(args.splits) & df["embedding"].isin(EMBEDDINGS)].copy()
    n_slides = df.groupby("embedding")["sample"].nunique().max()

    with mpl.rc_context(NATURE_RC):
        # Compact 1x2 grid (single metric across two label sets).
        fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.2), sharey=True)
        x = np.arange(len(EMBEDDINGS))
        width = 0.66
        panel_letters = ["a", "b"]

        for col_idx, (label_key, label_title) in enumerate(LABELS):
            ax = axes[col_idx]
            metric_key, metric_title = METRICS[0]
            sub = df[df["label"] == label_key]

            means, stds, points = [], [], []
            for emb in EMBEDDINGS:
                vals = sub.loc[sub["embedding"] == emb, metric_key].to_numpy()
                means.append(vals.mean())
                stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
                points.append(vals)

            ax.bar(x, means, width,
                   yerr=stds, capsize=2,
                   color=[EMB_COLORS[e] for e in EMBEDDINGS],
                   edgecolor="black", linewidth=0.4,
                   error_kw=dict(ecolor="black", lw=0.6))

            for i, vals in enumerate(points):
                jitter = np.random.RandomState(i + 7).uniform(-0.10, 0.10, size=len(vals))
                ax.scatter(np.full_like(vals, x[i]) + jitter, vals,
                           s=4, color="black", alpha=0.6, linewidth=0, zorder=3)

            ax.set_xticks(x)
            ax.set_xticklabels([EMB_LABELS[e] for e in EMBEDDINGS])
            ax.set_ylim(0, 0.8)
            ax.set_yticks(np.arange(0, 0.81, 0.2))
            if col_idx == 0:
                ax.set_ylabel(metric_title)
            ax.set_title(label_title, pad=4)
            ax.tick_params(pad=2)

            ax.text(-0.22, 1.08, panel_letters[col_idx], transform=ax.transAxes,
                    fontsize=8, fontweight="bold", va="top", ha="left")

        fig.text(0.5, -0.01,
                 f"n = {n_slides} held-out slides (val+test); error bars = s.d.",
                 ha="center", fontsize=6, color="0.3")
        fig.tight_layout(w_pad=1.0, h_pad=0.8)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=600, bbox_inches="tight")
        pdf_out = args.out.with_suffix(".pdf")
        fig.savefig(pdf_out, bbox_inches="tight")
        print(f"saved {args.out} and {pdf_out}  (n_slides={n_slides})")


if __name__ == "__main__":
    main()
