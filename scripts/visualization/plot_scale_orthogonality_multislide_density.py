"""Multi-slide cosine-similarity density: combines the single-slide density
panel from scripts/plot_scale_orthogonality.py with replication across 7 slides.

For each correlation category, computes a Gaussian KDE per slide on a common
cosine grid, then plots the mean KDE as a solid/dashed line with a shaded
±1 s.d. band across slides. Demonstrates simultaneously that
  (i) within-scale vs cross-scale cosine distributions are distinct, and
  (ii) the shapes replicate across skin/embryo/liver slides.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

RUNS = [
    ("18670", "BK39-SKI-27-FO-1-S3",       "skin"),
    ("21176", "BK39-SKI-27-FO-3-S3",       "skin"),
    ("21177", "BK46-SKI-27-FO-2-s3",       "skin"),
    ("21178", "BK25-SKI-27-FO-5-S13-B1",   "skin"),
    ("21179", "BK30-SKI-28-FO-1-s4",       "skin"),
    ("21180", "CE4-SKI-27-FO-4-S22-A2",    "embryo"),
    ("21181", "DF15-LIV-0-FFPE-1-S1-i",    "liver"),
]

# Categories to plot. (file_stem, display label, color, linestyle, KDE bandwidth)
SCALE_COLOR = {"micro": "#d62728", "meso": "#2ca02c", "macro": "#1f77b4"}
PAIR_COLOR  = {"0vs1": "#7f7f7f", "0vs2": "#9467bd", "1vs2": "#17becf"}

CATEGORIES = [
    ("crosscell_samescale_scale0", "within micro",          SCALE_COLOR["micro"], "-",  0.02),
    ("crosscell_samescale_scale1", "within meso",           SCALE_COLOR["meso"],  "-",  0.02),
    ("crosscell_samescale_scale2", "within macro",          SCALE_COLOR["macro"], "-",  0.02),
    ("samecell_crossscale_0vs1",   "same-cell micro–meso",  PAIR_COLOR["0vs1"],  "--", 0.006),
    ("samecell_crossscale_0vs2",   "same-cell micro–macro", PAIR_COLOR["0vs2"],  "--", 0.006),
    ("samecell_crossscale_1vs2",   "same-cell meso–macro",  PAIR_COLOR["1vs2"],  "--", 0.006),
    ("crosscell_crossscale_0vs1",  "cross-cell baseline",   "black",             ":",  0.006),
]

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


def kde(values: np.ndarray, grid: np.ndarray, bw: float,
        n_sub: int = 20000, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    if len(values) > n_sub:
        values = values[rng.choice(len(values), size=n_sub, replace=False)]
    d = (grid[:, None] - values[None, :]) / bw
    return np.exp(-0.5 * d * d).sum(axis=1) / (len(values) * bw * np.sqrt(2 * np.pi))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("outputs/diagnostics/scale_correlation"))
    ap.add_argument("--out", type=Path,
                    default=Path("figures/diagnostics/scale_orthogonality_multislide_density.png"))
    args = ap.parse_args()

    grid = np.linspace(0.0, 1.0, 600)

    # densities[cat_stem] -> np.ndarray of shape (n_slides, len(grid))
    densities: dict[str, list[np.ndarray]] = {c: [] for c, *_ in CATEGORIES}
    for jid, slide, tissue in RUNS:
        for cat_stem, _, _, _, bw in CATEGORIES:
            f = args.root / jid / f"{cat_stem}.npy"
            if not f.exists():
                print(f"[skip] {jid} {slide} missing {cat_stem}")
                continue
            arr = np.load(f)
            densities[cat_stem].append(kde(arr, grid, bw=bw))

    densities = {k: np.stack(v, axis=0) for k, v in densities.items() if v}

    with mpl.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(5.4, 2.6))

        for cat_stem, label, color, ls, _ in CATEGORIES:
            if cat_stem not in densities:
                continue
            arr = densities[cat_stem]  # (n_slides, len(grid))
            mean = arr.mean(axis=0)
            std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
            ax.fill_between(grid, mean - std, mean + std,
                            color=color, alpha=0.18, linewidth=0)
            ax.plot(grid, mean, color=color, lw=1.0, linestyle=ls, label=label)

        ax.set_xlim(0.18, 1.0)
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Density")
        ax.set_title("Within- vs cross-scale cosine — replicated across 7 slides", pad=4)

        leg = ax.legend(loc="upper right", frameon=False,
                        handlelength=1.6, labelspacing=0.25, borderpad=0.2)

        fig.text(0.5, -0.04,
                 "n = 7 slides (5 skin, 1 embryo, 1 liver); "
                 "line = mean Gaussian-KDE across slides, band = ±1 s.d.",
                 ha="center", fontsize=5.5, color="0.3")
        fig.tight_layout()

        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=600, bbox_inches="tight")
        fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
        print(f"saved {args.out} and {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
