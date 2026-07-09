"""TEMPORARY figure — HEST-1k cross-modal retrieval on the NEW spatial h5
(regenerated 2026-06-08 22:xx). SIGMMA fullpanel a01 step2000.

Marked _tmp: quick look, not a manuscript artifact. Bars = R@1%/5%/10% per
public tissue, image->ST and ST->image panels, with dashed random-chance lines
(R@k% chance = k%). Reads recall_metrics.json straight from outputs/.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
BASE = ROOT / "outputs/retrieval/hest_all"
TISSUES = ["SKCM", "LUAD", "PAAD", "IDC"]
METRICS = [("recall@1percent", "R@1%", 0.01),
           ("recall@5percent", "R@5%", 0.05),
           ("recall@10percent", "R@10%", 0.10)]
MCOLORS = ["#d62728", "#ff9896", "#fdd0a2"]  # 1% / 5% / 10%
DIRS = [("image", r"H&E $\rightarrow$ ST (image query)"),
        ("spatial", r"ST $\rightarrow$ H&E (spatial query)")]


def load():
    out = {}
    for t in TISSUES:
        f = BASE / f"hest_{t}_retrieval_sigmma_fullpanel_a01_step2000" / "recall_metrics.json"
        out[t] = json.load(open(f))
    return out


def main():
    d = load()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 9, "axes.labelsize": 7,
        "axes.linewidth": 0.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2),
                             gridspec_kw={"wspace": 0.18}, squeeze=False)
    x = np.arange(len(TISSUES))
    w = 0.26
    for col, (dkey, dtitle) in enumerate(DIRS):
        ax = axes[0, col]
        for i, (mkey, mlbl, chance) in enumerate(METRICS):
            vals = [d[t][f"{dkey}_{mkey}"] for t in TISSUES]
            ax.bar(x + (i - 1) * w, vals, width=w, color=MCOLORS[i],
                   edgecolor="black", linewidth=0.5,
                   label=mlbl, zorder=3)
            # per-metric chance line
            ax.axhline(chance, color="black", linestyle="--",
                       linewidth=0.7, zorder=2, alpha=0.6)
        ax.text(len(TISSUES) - 0.5, 0.10, "chance @ 1/5/10%", fontsize=5,
                va="bottom", ha="right", color="black", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(TISSUES)
        ax.set_title(dtitle, fontsize=8)
        if col == 0:
            ax.set_ylabel("Recall")
        ax.set_ylim(0, 0.80)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="-", linewidth=0.25, color="#cccccc",
                alpha=0.7, zorder=0)
        ax.set_axisbelow(True)
        if col == 1:
            ax.legend(frameon=False, fontsize=6, ncol=1, loc="upper right")

    fig.suptitle(
        "HEST-1k cross-modal retrieval — NEW spatial h5 (regen 2026-06-08)  ·  "
        "SIGMMA fullpanel a01 step2000  [TMP]",
        fontsize=8.5, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = ROOT / "figures/retrieval"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = out / f"hest1k_retrieval_newh5ad_tmp.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
