"""Tumour-vs-Stroma compositional barplot for the RCC kidney cohort.

Reads per-cell ``level_2_cell_type`` ground truth from a curated ``labels.h5ad``
(``obs/batch`` × ``obs/level_2_cell_type``) and, for each evaluable section in
``cell_niche_evaluable_slides_rcc.csv``, draws a stacked proportion bar of:

    Tumor  = {Tumour epithelium}
    Stroma = {Fibroblast, Blood vasculature, Lymphatic vasculature,
              Perivascular cell}

Immune cells (B/T lymphocyte, Macrophage, Dendritic cell) and normal Nephron
epithelium are treated as "Other" and EXCLUDED from the ratio: each bar is
normalised to Stroma+Tumor so its split is the literal tumour-stroma ratio.
The "Other" fraction (of all labelled cells) is annotated per bar for honesty,
and the full per-class counts are written to a CSV next to the figure.

Usage:
  python scripts/plot_tumor_stroma_composition.py \
      --labels_h5ad <labels.h5ad> \
      --manifest configs/cell_niche_evaluable_slides_rcc.csv \
      --out_dir figures/clinical_outcome
"""
from __future__ import annotations
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TUMOR = ["Tumour epithelium"]
STROMA = ["Fibroblast", "Blood vasculature", "Lymphatic vasculature",
          "Perivascular cell"]

C_TUMOR = "#ca7695"   # muted pink-red
C_STROMA = "#a9c6d3"  # muted light blue


def _read_categorical(obs: h5py.Group, col: str) -> np.ndarray:
    """Decode an AnnData categorical obs column to a (N,) array of str."""
    g = obs[col]
    cats = [c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]]
    codes = g["codes"][:]
    cats_arr = np.array(cats + [""], dtype=object)  # -1 code -> ""
    return cats_arr[codes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_h5ad", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_col", default="batch")
    ap.add_argument("--celltype_col", default="level_2_cell_type")
    ap.add_argument("--order", default="tumor_frac",
                    choices=["tumor_frac", "donor"],
                    help="Bar ordering: global tumour-fraction sort, or grouped "
                         "by donor (CV1/CV2/...) with gaps + donor labels.")
    ap.add_argument("--figsize", default=None,
                    help="Override figure size as 'W,H' in inches (e.g. '10,7'). "
                         "Default scales width with the number of bars.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(args.manifest)
    # One bar per evaluable section, matched to labels' batch via the manifest
    # `norm` column (case-insensitive). De-dup in case two sample_names share a
    # norm; preserve manifest order.
    sections = man[["norm", "split"]].drop_duplicates("norm")
    norm_to_split = dict(zip(sections["norm"], sections["split"]))
    wanted = {str(n).upper(): n for n in sections["norm"]}

    print(f"[load] reading {args.labels_h5ad}")
    with h5py.File(args.labels_h5ad, "r") as f:
        obs = f["obs"]
        batch = _read_categorical(obs, args.batch_col)
        ctype = _read_categorical(obs, args.celltype_col)

    df = pd.DataFrame({"batch": batch, "ctype": ctype})
    df["batch_u"] = df["batch"].str.upper()
    df = df[df["batch_u"].isin(wanted.keys())]
    df["section"] = df["batch_u"].map(wanted)

    counts = (
        df.groupby(["section", "ctype"]).size().unstack(fill_value=0)
    )
    # Guard: classes referenced below must exist as columns.
    for c in TUMOR + STROMA:
        if c not in counts.columns:
            counts[c] = 0

    counts["n_total"] = counts.sum(axis=1)  # all labelled cells, before adding helper cols
    counts["n_tumor"] = counts[TUMOR].sum(axis=1)
    counts["n_stroma"] = counts[STROMA].sum(axis=1)
    counts["n_other"] = counts["n_total"] - counts["n_tumor"] - counts["n_stroma"]
    counts["n_ts"] = counts["n_tumor"] + counts["n_stroma"]
    counts["tumor_frac"] = counts["n_tumor"] / counts["n_ts"].replace(0, np.nan)
    counts["stroma_frac"] = counts["n_stroma"] / counts["n_ts"].replace(0, np.nan)
    counts["other_pct_of_all"] = 100.0 * counts["n_other"] / counts["n_total"]
    counts["split"] = [norm_to_split.get(s, "") for s in counts.index]
    # Donor = leading token of the section id (e.g. "CV9-KID-0-FT-1" -> "CV9").
    counts["donor"] = [str(s).split("-")[0] for s in counts.index]

    if args.order == "donor":
        # Group by donor; within each donor, most tumour-rich first.
        counts = counts.sort_values(["donor", "tumor_frac"],
                                    ascending=[True, False])
    else:
        # Global descending tumour fraction (most tumour-rich on the left).
        counts = counts.sort_values("tumor_frac", ascending=False)

    # x positions: insert a gap whenever the donor changes (only matters when
    # grouped by donor; harmless otherwise since donors are interleaved).
    donors = counts["donor"].to_numpy()
    xpos, cur = [], 0.0
    for i, d in enumerate(donors):
        if i > 0 and args.order == "donor" and d != donors[i - 1]:
            cur += 1.0  # gap between donor groups
        xpos.append(cur)
        cur += 1.0
    xpos = np.array(xpos)

    # ---- CSV ----
    csv_cols = (TUMOR + STROMA +
                ["n_tumor", "n_stroma", "n_other", "n_total", "n_ts",
                 "tumor_frac", "stroma_frac", "other_pct_of_all", "split"])
    csv_path = out_dir / "tumor_stroma_composition.csv"
    counts[csv_cols].to_csv(csv_path)
    print(f"[csv] wrote {csv_path}")

    # ---- figure ----
    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif",
                         "svg.fonttype": "none"})
    labels = list(counts.index)
    x = xpos
    tf = counts["tumor_frac"].fillna(0).to_numpy()
    sf = counts["stroma_frac"].fillna(0).to_numpy()

    if args.figsize:
        figsize = tuple(float(v) for v in args.figsize.split(","))
    else:
        figsize = (max(6, 0.5 * x.max() + 2.5), 4.4)
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, sf, width=0.8, color=C_STROMA, label="Stroma", edgecolor="white", linewidth=0.3)
    ax.bar(x, tf, width=0.8, bottom=sf, color=C_TUMOR, label="Tumour epithelium",
           edgecolor="white", linewidth=0.3)

    # Annotate "other" (excluded) fraction above each bar.
    for xi, op in zip(x, counts["other_pct_of_all"].to_numpy()):
        ax.text(xi, 1.01, f"oth {op:.0f}%", ha="center", va="bottom",
                fontsize=5.5, color="0.4", rotation=90)

    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Proportion (of Stroma + Tumour cells)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=6.5)
    ax.set_xlim(x.min() - 0.7, x.max() + 0.7)
    ax.axhline(0.5, color="0.6", lw=0.7, ls="--", zorder=0)

    # When grouped by donor, draw a centred donor label + separators below.
    if args.order == "donor":
        donors = counts["donor"].to_numpy()
        for d in pd.unique(donors):
            xs = x[donors == d]
            ax.text(xs.mean(), -0.30, d, ha="center", va="top",
                    fontsize=8, fontweight="bold",
                    transform=ax.get_xaxis_transform())
        # vertical separators at each donor boundary (midpoint of the gap)
        for i in range(1, len(donors)):
            if donors[i] != donors[i - 1]:
                ax.axvline((x[i] + x[i - 1]) / 2, color="0.85", lw=0.8, zorder=0)

    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=7,
              bbox_to_anchor=(1.0, 1.16))
    suffix = " by donor" if args.order == "donor" else ""
    ax.set_title("RCC kidney — tumour-stroma composition per section" + suffix +
                 f" (n={len(labels)}, GT level_2_cell_type)", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    stem = "tumor_stroma_composition" + ("_by_donor" if args.order == "donor" else "")
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"[fig] wrote {p}")
    plt.close(fig)

    # quick console table
    show = counts[["n_tumor", "n_stroma", "n_other", "tumor_frac",
                   "other_pct_of_all", "split"]].copy()
    show["tumor_frac"] = show["tumor_frac"].map("{:.3f}".format)
    show["other_pct_of_all"] = show["other_pct_of_all"].map("{:.1f}".format)
    print(show.to_string())


if __name__ == "__main__":
    main()
