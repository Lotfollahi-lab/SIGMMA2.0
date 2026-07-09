"""Combined 3-panel scale-orthogonality figure (multi-slide).

  (a) Within- vs cross-scale cosine density (mean ± s.d. KDE bands).
  (b) K-NN top-10 manifold overlap per scale pair (mean ± s.d. bars + dots).
  (c) Participation ratio per scale (mean ± s.d. bars + dots).

All panels aggregate across 7 slides (5 skin biopsies + 1 embryo + 1 liver).
Nature-style compact layout, eva_critique palette.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = [
    # (jobid_correlation, jobid_collapse, sample, tissue)
    ("18670", "18671", "BK39-SKI-27-FO-1-S3",       "skin"),
    ("21176", "21318", "BK39-SKI-27-FO-3-S3",       "skin"),
    ("21177", "21319", "BK46-SKI-27-FO-2-s3",       "skin"),
    ("21178", "21320", "BK25-SKI-27-FO-5-S13-B1",   "skin"),
    ("21179", "21321", "BK30-SKI-28-FO-1-s4",       "skin"),
    ("21180", "21322", "CE4-SKI-27-FO-4-S22-A2",    "embryo"),
    ("21181", "21323", "DF15-LIV-0-FFPE-1-S1-i",    "liver"),
]

SCALES = ["micro", "meso", "macro"]
PAIRS = [("0", "1", "micro–meso"),
         ("0", "2", "micro–macro"),
         ("1", "2", "meso–macro")]

SCALE_COLOR = {"micro": "#d62728", "meso": "#2ca02c", "macro": "#1f77b4"}
PAIR_COLOR  = {"0vs1": "#7f7f7f", "0vs2": "#9467bd", "1vs2": "#17becf",
               "0-1":  "#7f7f7f", "0-2":  "#9467bd", "1-2":  "#17becf"}
TISSUE_MARKER = {"skin": "o", "embryo": "^", "liver": "s"}

DENSITY_CATS = [
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


def _parse_sections(csv_path: Path) -> dict[str, pd.DataFrame]:
    sections, cur = {}, None
    for line in csv_path.read_text().splitlines():
        if line.startswith("#"):
            cur = line.lstrip("# ").strip()
            sections[cur] = []
        elif cur and line.strip():
            sections[cur].append(line)
    return {k: pd.read_csv(io.StringIO("\n".join(v))) for k, v in sections.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr_root", type=Path,
                    default=Path("outputs/diagnostics/scale_correlation"))
    ap.add_argument("--collapse_root", type=Path,
                    default=Path("outputs/diagnostics/scale_collapse"))
    ap.add_argument("--out", type=Path,
                    default=Path("figures/diagnostics/scale_orthogonality_combined.png"))
    args = ap.parse_args()

    # ---- panel (a): densities across slides ----
    grid = np.linspace(0.0, 1.0, 600)
    densities: dict[str, list[np.ndarray]] = {c: [] for c, *_ in DENSITY_CATS}
    for jid_corr, _, slide, _ in RUNS:
        for cat_stem, _, _, _, bw in DENSITY_CATS:
            f = args.corr_root / jid_corr / f"{cat_stem}.npy"
            if not f.exists():
                continue
            densities[cat_stem].append(kde(np.load(f), grid, bw=bw))
    densities = {k: np.stack(v, axis=0) for k, v in densities.items() if v}

    # ---- panel (b)/(c): per-slide collapse summaries ----
    pr_rows, knn_rows = [], []
    chance_overlap = None
    for _, jid_coll, slide, tissue in RUNS:
        csv = args.collapse_root / jid_coll / f"scale_collapse_summary_{slide}.csv"
        if not csv.exists():
            continue
        sections = _parse_sections(csv)
        pr_key = next((k for k in sections if "participation ratio" in k.lower()), None)
        knn_key = next((k for k in sections if "K-NN" in k or "knn" in k.lower()), None)
        if pr_key:
            for _, r in sections[pr_key].iterrows():
                pr_rows.append({"slide": slide, "tissue": tissue,
                                "scale_name": SCALES[int(r["scale"])],
                                "PR_raw": float(r["PR_raw"])})
        if knn_key:
            for _, r in sections[knn_key].iterrows():
                pair = f"{int(r['i'])}-{int(r['j'])}"
                knn_rows.append({"slide": slide, "tissue": tissue,
                                 "pair": pair,
                                 "overlap": float(r["mean_overlap"])})
                if chance_overlap is None:
                    chance_overlap = float(r["chance_overlap"])
    pr_df = pd.DataFrame(pr_rows)
    knn_df = pd.DataFrame(knn_rows)

    with mpl.rc_context(NATURE_RC):
        fig = plt.figure(figsize=(7.2, 2.9))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.7, 1.0, 1.0], wspace=0.45)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[0, 2])

        # ---- (a) density ----
        for cat_stem, label, color, ls, _ in DENSITY_CATS:
            if cat_stem not in densities:
                continue
            arr = densities[cat_stem]
            mean = arr.mean(axis=0)
            std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
            ax_a.fill_between(grid, mean - std, mean + std,
                              color=color, alpha=0.18, linewidth=0)
            ax_a.plot(grid, mean, color=color, lw=0.9, linestyle=ls, label=label)
        ax_a.set_xlim(0.18, 1.0)
        ax_a.set_xlabel("Cosine similarity")
        ax_a.set_ylabel("Density")
        ax_a.set_title("Within- vs cross-scale", pad=4)
        ax_a.legend(loc="upper right", frameon=False,
                    handlelength=1.5, labelspacing=0.22, borderpad=0.2)
        ax_a.text(-0.18, 1.06, "a", transform=ax_a.transAxes,
                  fontsize=8, fontweight="bold", va="top", ha="left")

        # ---- (b) KNN overlap ----
        x = np.arange(len(PAIRS))
        labels_pair = [lab for _, _, lab in PAIRS]
        keys = [f"{a}-{b}" for a, b, _ in PAIRS]
        knn_agg = knn_df.groupby("pair", sort=False)["overlap"].agg(["mean", "std"])
        knn_agg = knn_agg.reindex(keys)
        means_b = knn_agg["mean"].to_numpy()
        stds_b  = knn_agg["std"].to_numpy()
        ax_b.bar(x, means_b, 0.66, yerr=stds_b, capsize=2.5,
                 color=[PAIR_COLOR[k] for k in keys],
                 edgecolor="black", linewidth=0.5,
                 error_kw=dict(ecolor="black", lw=0.7))
        if chance_overlap is not None:
            ax_b.axhline(chance_overlap, color="black", lw=0.5, ls=":")
            ax_b.text(len(x) - 0.5, chance_overlap + 0.004,
                      f"chance = {chance_overlap:.3f}",
                      fontsize=5.5, color="black", ha="right", va="bottom")
        for i, (m, s) in enumerate(zip(means_b, stds_b)):
            ax_b.text(x[i], m + s + 0.008, f"{m*100:.1f}%",
                      ha="center", va="bottom", fontsize=5.5)
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(labels_pair, rotation=15)
        ax_b.set_ylabel("Top-10 NN overlap")
        ax_b.set_title("K-NN manifold agreement", pad=4)
        ax_b.set_ylim(0, max(0.25, (means_b + stds_b).max() * 1.28))
        ax_b.text(-0.30, 1.06, "b", transform=ax_b.transAxes,
                  fontsize=8, fontweight="bold", va="top", ha="left")

        # ---- (c) participation ratio ----
        x = np.arange(len(SCALES))
        pr_agg = pr_df.groupby("scale_name", sort=False)["PR_raw"].agg(["mean", "std"])
        pr_agg = pr_agg.reindex(SCALES)
        means_c = pr_agg["mean"].to_numpy()
        stds_c  = pr_agg["std"].to_numpy()
        ax_c.bar(x, means_c, 0.66, yerr=stds_c, capsize=2.5,
                 color=[SCALE_COLOR[s] for s in SCALES],
                 edgecolor="black", linewidth=0.5,
                 error_kw=dict(ecolor="black", lw=0.7))
        for i, (m, s) in enumerate(zip(means_c, stds_c)):
            ax_c.text(x[i], m + s + 0.25, f"{m:.1f}",
                      ha="center", va="bottom", fontsize=5.5)
        ax_c.set_xticks(x); ax_c.set_xticklabels(SCALES)
        ax_c.set_ylabel("Participation ratio")
        ax_c.set_title("Effective dimension", pad=4)
        ax_c.set_ylim(0, max(means_c + stds_c) * 1.32)
        ax_c.text(-0.30, 1.06, "c", transform=ax_c.transAxes,
                  fontsize=8, fontweight="bold", va="top", ha="left")

        n = len(pr_df["slide"].unique())
        fig.subplots_adjust(left=0.07, right=0.98, top=0.88,
                            bottom=0.30, wspace=0.45)
        fig.text(0.5, 0.05,
                 f"n = {n} slides (5 skin, 1 embryo, 1 liver); "
                 "(a) line = mean KDE, band = ±1 s.d.; (b,c) bar = mean ± s.d.",
                 ha="center", fontsize=5.5, color="0.3")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=600, bbox_inches="tight")
        fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
        print(f"saved {args.out} and {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
