"""Nature-style summary figure for scale orthogonality evidence.

Reads cached diagnostics:
  outputs/diagnostics/scale_correlation/18670/  (per-pair cosine arrays)
  outputs/diagnostics/scale_collapse/18671/     (KNN overlap, PR summaries)

Panels:
  a) Cosine-similarity densities — within-scale baseline vs same-cell cross-scale.
  b) K-NN overlap (top-10) per scale pair, with chance line.
  c) Participation ratio per scale.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCALES = ["micro", "meso", "macro"]
PAIRS = [("0vs1", "micro–meso"),
         ("0vs2", "micro–macro"),
         ("1vs2", "meso–macro")]

# Palette: matches scripts/eva_critique/plot_instance_evidence.py.
SCALE_COLOR = {"micro": "#d62728", "meso": "#2ca02c", "macro": "#1f77b4"}
PAIR_COLOR  = {"0vs1": "#7f7f7f", "0vs2": "#9467bd", "1vs2": "#17becf"}
CROSS_COLOR = "#888888"

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


def kde(values: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
    """Simple Gaussian KDE on `grid` using random sub-sample for speed."""
    n_sub = min(20000, len(values))
    rng = np.random.RandomState(0)
    idx = rng.choice(len(values), size=n_sub, replace=False)
    x = values[idx]
    d = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * d * d).sum(axis=1) / (n_sub * bw * np.sqrt(2 * np.pi))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr_dir", type=Path,
                    default=Path("outputs/diagnostics/scale_correlation/18670"))
    ap.add_argument("--collapse_dir", type=Path,
                    default=Path("outputs/diagnostics/scale_collapse/18671"))
    ap.add_argument("--out", type=Path,
                    default=Path("figures/diagnostics/scale_orthogonality.png"))
    args = ap.parse_args()

    # ---- load distributions ----
    within = {s: np.load(args.corr_dir / f"crosscell_samescale_scale{i}.npy")
              for i, s in enumerate(SCALES)}
    samecell_cross = {p: np.load(args.corr_dir / f"samecell_crossscale_{p}.npy")
                      for p, _ in PAIRS}
    crosscell_cross_01 = np.load(args.corr_dir / "crosscell_crossscale_0vs1.npy")

    # ---- load collapse summary (T2 PR, T4 KNN) ----
    coll_csv = next(args.collapse_dir.glob("scale_collapse_summary_*.csv"))
    raw = coll_csv.read_text().splitlines()
    sections = {}
    cur = None
    for line in raw:
        if line.startswith("#"):
            cur = line.lstrip("# ").strip()
            sections[cur] = []
        elif cur and line.strip():
            sections[cur].append(line)
    pr_rows = pd.read_csv(pd.io.common.StringIO("\n".join(
        sections["T2 participation ratio per scale"])))
    knn_rows = pd.read_csv(pd.io.common.StringIO("\n".join(
        sections["T4 K-NN overlap across scales"])))

    with mpl.rc_context(NATURE_RC):
        fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.1))

        # ---- (a) Cosine densities ----
        ax = axes[0]
        grid = np.linspace(0.0, 1.0, 400)
        # within-scale (3 lines, scale-coloured, solid)
        for s in SCALES:
            d = kde(within[s], grid, bw=0.02)
            ax.plot(grid, d, color=SCALE_COLOR[s], lw=1.0,
                    label=f"within {s}")
        # same-cell cross-scale (3 lines, dashed grey-ish)
        for (p, lab) in PAIRS:
            d = kde(samecell_cross[p], grid, bw=0.006)
            ax.plot(grid, d, color=PAIR_COLOR[p], lw=0.9, ls="--",
                    label=f"cross {lab}")
        # cross-cell cross-scale (one line, dotted, baseline)
        d = kde(crosscell_cross_01, grid, bw=0.006)
        ax.plot(grid, d, color="black", lw=0.7, ls=":",
                label="cross-cell baseline")
        ax.set_xlim(0.15, 1.0)
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Density")
        ax.set_title("Within- vs cross-scale", pad=4)
        ax.legend(loc="upper right", frameon=False, handlelength=1.4,
                  borderpad=0.2, labelspacing=0.25)
        ax.text(-0.22, 1.08, "a", transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="left")

        # ---- (b) K-NN overlap ----
        ax = axes[1]
        labels = [lab for _, lab in PAIRS]
        overlaps = []
        chances = []
        for code, _ in PAIRS:
            i_, j_ = code.split("vs")
            row = knn_rows[(knn_rows["i"] == int(i_)) &
                           (knn_rows["j"] == int(j_))].iloc[0]
            overlaps.append(float(row["mean_overlap"]))
            chances.append(float(row["chance_overlap"]))
        x = np.arange(len(labels))
        bars = ax.bar(x, overlaps, 0.66,
                      color=[PAIR_COLOR[c] for c, _ in PAIRS],
                      edgecolor="black", linewidth=0.4)
        ax.axhline(np.mean(chances), color="black", lw=0.6, ls=":")
        ax.text(len(labels) - 0.5, np.mean(chances) + 0.005,
                f"chance = {np.mean(chances):.3f}",
                fontsize=5.5, color="black", ha="right", va="bottom")
        for b, v in zip(bars, overlaps):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                    f"{v*100:.1f}%", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylim(0, 0.25)
        ax.set_ylabel("Top-10 NN overlap")
        ax.set_title("K-NN manifold agreement", pad=4)
        ax.text(-0.22, 1.08, "b", transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="left")

        # ---- (c) Participation ratio ----
        ax = axes[2]
        pr_rows = pr_rows.sort_values("scale")
        pr = pr_rows["PR_raw"].to_numpy()
        x = np.arange(len(SCALES))
        bars = ax.bar(x, pr, 0.66,
                      color=[SCALE_COLOR[s] for s in SCALES],
                      edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, pr):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.12,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(SCALES)
        ax.set_ylim(0, max(pr) * 1.25)
        ax.set_ylabel("Participation ratio")
        ax.set_title("Effective dimension (D = 512)", pad=4)
        ax.text(-0.22, 1.08, "c", transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="left")

        fig.text(0.5, -0.04,
                 "Slide BK39-SKI-27-FO-1-S3; within-scale n=5×10⁵ pairs, "
                 "cross-scale n=1.7×10⁴ cells",
                 ha="center", fontsize=5.5, color="0.3")
        fig.tight_layout(w_pad=1.1)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=600, bbox_inches="tight")
        fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
        print(f"saved {args.out} and {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
