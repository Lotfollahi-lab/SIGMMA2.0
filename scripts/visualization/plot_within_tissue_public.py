"""Figure: within-tissue retrieval for PUBLIC HEST-1k (step2000, OOD).
Per direction: 2 rows (R@1%, MRR) x 4 tissue cols (IDC/PAAD/LUAD/SKCM).
Method-palette bars + dashed random-chance line (chance matters here: OOD).
No error bars / significance — values are query-pooled per tissue.
Reads outputs/retrieval/within_tissue_public_step2000.csv.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
CSV = ROOT / "outputs/retrieval/within_tissue_public_step2000.csv"
OUTDIR = ROOT / "figures/retrieval/all_gene_panel"
CONDITIONS = [("BLEEP", "#882255"), ("CLIP", "#44AA99"), ("PLIP", "#DDCC77"),
              ("OmiCLIP", "#999933"), ("SIGMMA", "#d62728")]
LABELS = {"BLEEP": "BLEEP-ft", "CLIP": "CLIP-ft", "PLIP": "PLIP-ft",
          "OmiCLIP": "OmiCLIP-ft", "SIGMMA": "SIGMMA"}
TISSUES = ["IDC", "PAAD", "LUAD", "SKCM"]
METRICS = [("r1pct", "Recall @ 1%", 0.01), ("mrr", "Mean Reciprocal Rank", None)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["H&E->ST", "ST->H&E"], default="H&E->ST")
    args = ap.parse_args()
    suffix = "he2st" if args.direction == "H&E->ST" else "st2he"
    title_dir = (r"H&E $\rightarrow$ ST" if suffix == "he2st"
                 else r"ST $\rightarrow$ H&E")

    df = pd.read_csv(CSV)
    df = df[df["direction"] == args.direction]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 9, "axes.labelsize": 7,
        "axes.linewidth": 0.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    ncol = len(TISSUES)
    fig, axes = plt.subplots(2, ncol, figsize=(2.0 * ncol + 0.5, 4.6),
                             gridspec_kw={"wspace": 0.32, "hspace": 0.55},
                             squeeze=False)
    x = np.arange(len(CONDITIONS))
    for row, (mkey, mlbl, fixed_chance) in enumerate(METRICS):
        for col, T in enumerate(TISSUES):
            ax = axes[row, col]
            sub = df[df["tissue"] == T].set_index("method")
            vals = [sub.loc[m, mkey] for m, _ in CONDITIONS]
            colors = [c for _, c in CONDITIONS]
            bars = ax.bar(x, vals, color=colors, edgecolor="black",
                          linewidth=0.5, width=0.72)
            bars[-1].set_linewidth(1.0)  # SIGMMA outline
            # chance line
            ch = fixed_chance if fixed_chance is not None else float(sub["chance_mrr"].iloc[0])
            ax.axhline(ch, color="#444444", linestyle=":", linewidth=0.8,
                       alpha=0.9, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels([LABELS[m] for m, _ in CONDITIONS],
                               rotation=40, ha="right", fontsize=6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", linestyle="-", linewidth=0.25, color="#cccccc",
                    alpha=0.7, zorder=-1)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel(mlbl)
            if row == 0:
                n_ref = int(sub["n_ref"].iloc[0])
                ax.set_title(f"{T}\n(gallery n={n_ref:,})", fontsize=8, pad=3)
            # annotate chance once (top-right panel of each row)
            if col == ncol - 1:
                ax.text(0.97, ch, "chance", transform=ax.get_yaxis_transform(),
                        ha="right", va="bottom", fontsize=5, fontstyle="italic",
                        color="#444444")
    letters = "abcdefgh"
    for i, ax in enumerate(axes.ravel()):
        ax.text(-0.22, 1.04, letters[i], transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom", ha="left")

    fig.suptitle(title_dir + " within-tissue retrieval, public HEST-1k (OOD)",
                 fontsize=10, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(bottom=0.12, top=0.88, hspace=0.55, wspace=0.32)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = OUTDIR / f"within_tissue_public_{suffix}.{ext}"
        fig.savefig(p, dpi=600 if ext == "png" else None)
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
