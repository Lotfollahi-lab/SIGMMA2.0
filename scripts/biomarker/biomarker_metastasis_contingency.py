#!/usr/bin/env python
"""Contingency table of SIGMMA-predicted biomarker group (median split, high vs
low) against distant metastasis (AJCC M: M0 vs M1) in TCGA-KIRC.

This is an ASSOCIATION (contingency) table, NOT a confusion matrix / trained
classifier: rows = predicted biomarker group, cols = observed M stage. We report
the 2x2 counts (+ row-normalised percentages), a Fisher exact test p-value and
odds ratio, and the per-group metastasis rate. PAX8-high is expected (from the
survival result) to be metastasis-depleted.

Run in the project .venv (scipy + matplotlib).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, chi2_contingency
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--gene", default="PAX8")
    ap.add_argument("--agg", default="mean", choices=["mean", "q90"])
    ap.add_argument("--cohort", default="TCGA-KIRC")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fig_dir", required=True)
    args = ap.parse_args()

    pcol = f"{args.gene}_{args.agg}"
    pred = pd.read_csv(args.pred_csv)
    clin = pd.read_csv(args.clinical, sep="\t")[["case_submitter_id", "ajcc_m"]]
    df = pred.merge(clin, on="case_submitter_id", how="inner")

    # metastasis label: keep definite M0 / M1 only (drop MX / NaN)
    df["M"] = df["ajcc_m"].astype(str).str.strip().str.upper()
    df = df[df["M"].isin(["M0", "M1"])].copy()

    vals = pd.to_numeric(df[pcol], errors="coerce")
    med = np.nanmedian(vals)
    df["group"] = np.where(vals >= med, f"{args.gene}-high", f"{args.gene}-low")

    # 2x2 contingency: rows = predicted group (high, low), cols = M0, M1
    rows = [f"{args.gene}-high", f"{args.gene}-low"]
    cols = ["M0", "M1"]
    ct = pd.crosstab(df["group"], df["M"]).reindex(index=rows, columns=cols).fillna(0).astype(int)

    a, b = ct.loc[rows[0], "M0"], ct.loc[rows[0], "M1"]   # high: M0, M1
    c, d = ct.loc[rows[1], "M0"], ct.loc[rows[1], "M1"]   # low : M0, M1
    # OR for metastasis in low vs high (ref=high): (low M1 odds)/(high M1 odds)
    orr, p_fisher = fisher_exact([[d, c], [b, a]])
    chi2, p_chi2, _, _ = chi2_contingency(ct.values)
    met_rate_high = b / (a + b) if (a + b) else np.nan
    met_rate_low = d / (c + d) if (c + d) else np.nan

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    figdir = Path(args.fig_dir); figdir.mkdir(parents=True, exist_ok=True)
    ct.to_csv(out / f"{pcol}_metastasis_contingency.csv")
    summary = {
        "gene": args.gene, "agg": args.agg, "median_cutoff": float(med),
        "n": int(ct.values.sum()),
        "met_rate_high": float(met_rate_high), "met_rate_low": float(met_rate_low),
        "odds_ratio_low_vs_high": float(orr),
        "fisher_p": float(p_fisher), "chi2_p": float(p_chi2),
    }
    pd.DataFrame([summary]).to_csv(out / f"{pcol}_metastasis_stats.csv", index=False)
    print(summary)
    print(ct)

    # heatmap
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(ct.values, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["M0\n(no metastasis)", "M1\n(metastasis)"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(rows)
    ax.set_xlabel("observed AJCC M stage"); ax.set_ylabel("predicted group")
    tot = ct.values.sum()
    for i in range(2):
        rowsum = ct.values[i].sum()
        for j in range(2):
            n = ct.values[i, j]
            rowpct = 100 * n / rowsum if rowsum else 0
            ax.text(j, i, f"{n}\n({rowpct:.0f}% of row)", ha="center", va="center",
                    fontsize=9, color="white" if n > ct.values.max() / 2 else "black")
    ax.set_title(f"{args.cohort}: predicted {args.gene} ({args.agg}) vs metastasis (contingency table)\n"
                 f"metastasis rate  high={met_rate_high*100:.1f}%  "
                 f"low={met_rate_low*100:.1f}%\n"
                 f"OR(low vs high)={orr:.2f}, Fisher p={p_fisher:.2g} (n={tot})",
                 fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label="n patients")
    fig.tight_layout()
    fig.savefig(figdir / f"{pcol}_metastasis_contingency.png", dpi=250, bbox_inches="tight")
    fig.savefig(figdir / f"{pcol}_metastasis_contingency.pdf", bbox_inches="tight")
    print(f"[out] {figdir / (pcol + '_metastasis_contingency.png')}")


if __name__ == "__main__":
    main()
