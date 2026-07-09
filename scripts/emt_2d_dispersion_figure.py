"""Clean 2-panel contrast: EMT *proportion* (fails) vs EMT *dispersion* (tracks risk).

Reads the merged CSV from emt_2d_metadata_correlation.py and draws a glanceable
donor-ordered strip plot so the story reads at a glance:
  - LEFT  : EMT-active proportion (hybrid+mesenchymal %)  -> highest in the LOWEST-risk
            donor (CV1), i.e. anti-correlated / confounded by the per-section baseline.
  - RIGHT : EMT-axis spread (std of the projection onto the E-anchor->M-anchor axis),
            a translation-invariant dispersion metric -> rises with clinical risk.

x = donor ordered by Leibovich (low->high); each donor a column of its sections
(strip), a thick donor-mean bar, and a dashed trend line through the donor means.
The single M1 (metastatic) donor CV2 is drawn with star markers.

CAVEAT printed on the figure: 4 donors / 1 M1 / 3-of-4 grade-4 -> DESCRIPTIVE, not powered.
"""
import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

DCOL = {"CV1": "#4477AA", "CV2": "#EE6677", "CV6": "#228833", "CV9": "#AA3377"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="figures/cross_tissue_retrieval/emt_2d/"
                    "emt_2d_metadata_merged_union.csv")
    ap.add_argument("--tag", default="union")
    ap.add_argument("--spread", default="emt_axis_std",
                    help="dispersion column for the right panel "
                         "(emt_axis_std / total_var / std_E)")
    ap.add_argument("--outdir", default="figures/cross_tissue_retrieval/emt_2d")
    args = ap.parse_args()

    recs = []
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            r["leibovich"] = int(r["leibovich"]); r["M"] = int(r["M"])
            r["grade"] = int(r["grade"])
            r["proportion"] = float(r["hybrid_pct"]) + float(r["mesenchymal_pct"])
            r["spread"] = float(r[args.spread])
            recs.append(r)
    donors = sorted({r["donor"] for r in recs},
                    key=lambda d: (next(x["leibovich"] for x in recs if x["donor"] == d), d))
    meta = {d: next(x for x in recs if x["donor"] == d) for d in donors}

    SPLAB = {"emt_axis_std": "EMT-axis spread  (std along E→M axis)",
             "total_var": "total variance  (var$_E$ + var$_M$)",
             "std_E": "epithelial-score spread  (std E)"}

    fig, axs = plt.subplots(1, 2, figsize=(12, 5.6))
    panels = [("proportion", "EMT-active proportion\n(hybrid + mesenchymal, % of tumour-epi)",
               "the OLD metric — fails", False),
              ("spread", SPLAB.get(args.spread, args.spread),
               "the NEW metric — translation-invariant", True)]

    for ax, (key, ylab, sub, is_new) in zip(axs, panels):
        means = []
        for i, d in enumerate(donors):
            ys = [r[key] for r in recs if r["donor"] == d]
            xj = i + (np.arange(len(ys)) - (len(ys) - 1) / 2) * 0.12  # spread sections
            mk = "*" if meta[d]["M"] else "o"
            sz = 230 if meta[d]["M"] else 80
            ax.scatter(xj, ys, marker=mk, s=sz, c=DCOL[d], edgecolor="k",
                       linewidth=0.6, alpha=0.9, zorder=3)
            m = float(np.mean(ys)); means.append(m)
            ax.plot([i - 0.28, i + 0.28], [m, m], color=DCOL[d], lw=3, zorder=2)
        # trend through donor means (risk increases left->right)
        ax.plot(range(len(donors)), means, ls="--", color="#555555", lw=1.4, zorder=1)
        # donor-level Spearman vs Leibovich
        rho, _ = spearmanr([meta[d]["leibovich"] for d in donors], means)
        ax.set_xticks(range(len(donors)))
        ax.set_xticklabels([f"{d}\n{'M1' if meta[d]['M'] else 'M0'} · G{meta[d]['grade']}"
                            f" · Leib{meta[d]['leibovich']}" for d in donors], fontsize=8.5)
        ax.set_ylabel(ylab, fontsize=10)
        col = "#C44" if is_new else "#888"
        ax.set_title(f"{sub}\ndonor ρ vs Leibovich = {rho:+.2f}",
                     fontsize=11, fontweight="bold", color=col)
        ax.grid(axis="y", ls=":", alpha=0.5)
        ax.margins(x=0.08)
    axs[0].set_xlabel("donor  (left → right = increasing clinical risk)", fontsize=9)
    axs[1].set_xlabel("donor  (left → right = increasing clinical risk)", fontsize=9)

    handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor=DCOL[d],
                      markeredgecolor="k", markersize=8, label=d) for d in donors]
    handles += [Line2D([0], [0], marker="*", linestyle="", markerfacecolor="#888",
                       markeredgecolor="k", markersize=14, label="★ = M1 (metastatic)"),
                Line2D([0], [0], color="#555", ls="--", label="trend (donor means)")]
    fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=6,
               framealpha=0.95, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("RCC tumour-epithelium: EMT proportion vs dispersion across clinical risk",
                 fontsize=13, fontweight="bold")
    foot = ("Proportion is confounded by the per-section control-gene baseline (highest in the "
            "LOWEST-risk donor CV1); the variance-based dispersion is translation-invariant and "
            "rises with risk.  CAVEAT: 4 donors / 1 M1 / 3-of-4 grade-4 → DESCRIPTIVE, not a powered test.")
    fig.text(0.5, -0.085, foot, ha="center", va="top", fontsize=8, color="#444444", wrap=True)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    outdir = Path(args.outdir)
    stem = outdir / f"emt_2d_dispersion_contrast_{args.tag}"
    fig.savefig(f"{stem}.png", dpi=160, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    print(f"[saved] {stem}.png / .pdf")


if __name__ == "__main__":
    main()
