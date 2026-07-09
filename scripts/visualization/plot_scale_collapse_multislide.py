"""Multi-slide replication of scale-collapse panels (KNN overlap + participation
ratio). Reads per-slide `scale_collapse_summary_<slide>.csv` (section-marker
format) and aggregates T2 (PR per scale) and T4 (KNN overlap per pair).

Mirrors plot_scale_orthogonality_multislide.py styling.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# (jobid, sample_name, tissue) — populated after job submission completes.
RUNS = [
    ("18671", "BK39-SKI-27-FO-1-S3",       "skin"),
    ("21318", "BK39-SKI-27-FO-3-S3",       "skin"),
    ("21319", "BK46-SKI-27-FO-2-s3",       "skin"),
    ("21320", "BK25-SKI-27-FO-5-S13-B1",   "skin"),
    ("21321", "BK30-SKI-28-FO-1-s4",       "skin"),
    ("21322", "CE4-SKI-27-FO-4-S22-A2",    "embryo"),
    ("21323", "DF15-LIV-0-FFPE-1-S1-i",    "liver"),
]

SCALES = ["micro", "meso", "macro"]
PAIRS = [("0", "1", "micro–meso"),
         ("0", "2", "micro–macro"),
         ("1", "2", "meso–macro")]

SCALE_COLOR = {"micro": "#d62728", "meso": "#2ca02c", "macro": "#1f77b4"}
PAIR_COLOR  = {"0-1": "#7f7f7f", "0-2": "#9467bd", "1-2": "#17becf"}
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


def _parse_sections(csv_path: Path) -> dict[str, pd.DataFrame]:
    """Parse a CSV with `# <title>` section markers into a dict of DataFrames."""
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
    ap.add_argument("--root", type=Path,
                    default=Path("outputs/diagnostics/scale_collapse"))
    ap.add_argument("--out", type=Path,
                    default=Path("figures/diagnostics/scale_collapse_multislide.png"))
    args = ap.parse_args()

    pr_rows, knn_rows = [], []
    chance_overlap = None
    for jid, slide, tissue in RUNS:
        csv = args.root / jid / f"scale_collapse_summary_{slide}.csv"
        if not csv.exists():
            print(f"[skip] {jid} {slide} (missing)")
            continue
        sections = _parse_sections(csv)
        pr_key = next((k for k in sections if "participation ratio" in k.lower()), None)
        knn_key = next((k for k in sections if "K-NN" in k or "knn" in k.lower()), None)
        if pr_key:
            for _, r in sections[pr_key].iterrows():
                pr_rows.append({
                    "slide": slide, "tissue": tissue,
                    "scale_idx": int(r["scale"]),
                    "scale_name": SCALES[int(r["scale"])],
                    "PR_raw": float(r["PR_raw"]),
                })
        if knn_key:
            for _, r in sections[knn_key].iterrows():
                key = f"{int(r['i'])}-{int(r['j'])}"
                knn_rows.append({
                    "slide": slide, "tissue": tissue,
                    "pair": key,
                    "overlap": float(r["mean_overlap"]),
                })
                if chance_overlap is None:
                    chance_overlap = float(r["chance_overlap"])
    pr_df  = pd.DataFrame(pr_rows)
    knn_df = pd.DataFrame(knn_rows)

    print("\n[T2 participation ratio per slide × scale]")
    print(pr_df.pivot(index="slide", columns="scale_name", values="PR_raw").round(2))
    print("\n[T4 KNN top-10 overlap per slide × pair]")
    print(knn_df.pivot(index="slide", columns="pair", values="overlap").round(3))

    # Aggregate
    pr_agg = pr_df.groupby("scale_name", sort=False)["PR_raw"].agg(["mean", "std", "count"])
    pr_agg = pr_agg.reindex(SCALES)
    knn_agg = knn_df.groupby("pair", sort=False)["overlap"].agg(["mean", "std", "count"])
    knn_agg = knn_agg.reindex([f"{a}-{b}" for a, b, _ in PAIRS])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    agg_csv = args.out.with_suffix(".csv")
    pd.concat([
        pr_agg.assign(metric="participation_ratio").reset_index().rename(columns={"scale_name": "key"}),
        knn_agg.assign(metric="knn_overlap").reset_index().rename(columns={"pair": "key"}),
    ]).to_csv(agg_csv, index=False)
    print(f"\nsaved aggregate -> {agg_csv}")

    with mpl.rc_context(NATURE_RC):
        fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.4))

        # ---- (a) K-NN overlap ----
        ax = axes[0]
        x = np.arange(len(PAIRS))
        labels = [lab for _, _, lab in PAIRS]
        keys = [f"{a}-{b}" for a, b, _ in PAIRS]
        means = knn_agg["mean"].to_numpy()
        stds  = knn_agg["std"].to_numpy()
        ax.bar(x, means, 0.66, yerr=stds, capsize=2.5,
               color=[PAIR_COLOR[k] for k in keys],
               edgecolor="black", linewidth=0.5,
               error_kw=dict(ecolor="black", lw=0.7))
        rng = np.random.RandomState(7)
        for i, k in enumerate(keys):
            sub = knn_df[knn_df["pair"] == k]
            jit = rng.uniform(-0.12, 0.12, size=len(sub))
            for (_, r), j in zip(sub.iterrows(), jit):
                ax.scatter(x[i] + j, r["overlap"],
                           marker=TISSUE_MARKER.get(r["tissue"], "o"),
                           s=12, color="black", alpha=0.7,
                           linewidths=0, zorder=3)
        if chance_overlap is not None:
            ax.axhline(chance_overlap, color="black", lw=0.6, ls=":")
            ax.text(len(x) - 0.5, chance_overlap + 0.004,
                    f"chance = {chance_overlap:.3f}",
                    fontsize=5.5, color="black", ha="right", va="bottom")
        for i, v in enumerate(means):
            ax.text(x[i], v + 0.005, f"{v*100:.1f}%",
                    ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
        ax.set_ylabel("Top-10 NN overlap")
        ax.set_title("K-NN manifold agreement", pad=4)
        ax.set_ylim(0, max(0.25, (means + stds).max() * 1.2))
        ax.text(-0.22, 1.08, "a", transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="left")

        # ---- (b) Participation ratio ----
        ax = axes[1]
        x = np.arange(len(SCALES))
        means = pr_agg["mean"].to_numpy()
        stds  = pr_agg["std"].to_numpy()
        ax.bar(x, means, 0.66, yerr=stds, capsize=2.5,
               color=[SCALE_COLOR[s] for s in SCALES],
               edgecolor="black", linewidth=0.5,
               error_kw=dict(ecolor="black", lw=0.7))
        rng = np.random.RandomState(11)
        for i, s in enumerate(SCALES):
            sub = pr_df[pr_df["scale_name"] == s]
            jit = rng.uniform(-0.12, 0.12, size=len(sub))
            for (_, r), j in zip(sub.iterrows(), jit):
                ax.scatter(x[i] + j, r["PR_raw"],
                           marker=TISSUE_MARKER.get(r["tissue"], "o"),
                           s=12, color="black", alpha=0.7,
                           linewidths=0, zorder=3)
        for i, v in enumerate(means):
            ax.text(x[i], v + 0.15, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x); ax.set_xticklabels(SCALES)
        ax.set_ylabel("Participation ratio")
        ax.set_title("Effective dimension (D = 512)", pad=4)
        ax.set_ylim(0, max(means + stds) * 1.25)
        ax.text(-0.22, 1.08, "b", transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="left")

        # tissue-marker legend (shared)
        tissue_handles = [plt.Line2D([0], [0], marker=m, linestyle="",
                                     markerfacecolor="black", markeredgecolor="none",
                                     markersize=4, label=t)
                          for t, m in TISSUE_MARKER.items()]
        fig.legend(handles=tissue_handles, loc="upper right",
                   bbox_to_anchor=(0.98, 1.02),
                   frameon=False, ncol=3, handletextpad=0.3,
                   columnspacing=0.7, labelspacing=0.15)

        n = len(pr_df["slide"].unique())
        fig.text(0.5, -0.04,
                 f"n = {n} slides (5 skin biopsies, 1 embryo, 1 liver); "
                 "each dot = per-slide value; bar = mean ± s.d. across slides",
                 ha="center", fontsize=5.5, color="0.3")
        fig.tight_layout(w_pad=1.1)

        fig.savefig(args.out, dpi=600, bbox_inches="tight")
        fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
        print(f"\nsaved {args.out} and {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
