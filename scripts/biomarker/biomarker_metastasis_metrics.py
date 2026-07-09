#!/usr/bin/env python
"""Imbalance-aware analysis of predicted-PAX8 vs distant metastasis (AJCC M0/M1)
in TCGA-KIRC. Metastasis is rare (~12%), so raw 2x2 counts are M0-dominated;
here we report metrics that are robust to (or explicit about) that imbalance.

Outputs:
  * per-group metastasis RATE with Wilson 95% CI + Fisher p + odds/risk ratio
    (rate plot, not count heatmap -> not visually dominated by the big M0 cell).
  * threshold-free ranking: ROC-AUC and PR-AUC (average precision) of the
    continuous metastasis score = -predicted PAX8 (low PAX8 -> metastasis-prone),
    with bootstrap 95% CIs; PR baseline = prevalence.
  * median-threshold classifier view (PAX8-low = predicted metastasis-prone):
    sensitivity, specificity, balanced accuracy, PPV, NPV.

Run in the project .venv (scikit-learn + statsmodels + matplotlib).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from statsmodels.stats.proportion import proportion_confint
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {"high": "#d62728", "low": "#888888"}


def boot_ci(y, score, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        vals.append(fn(y[b], score[b]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--gene", default="PAX8")
    ap.add_argument("--agg", default="mean")
    ap.add_argument("--cohort", default="TCGA-KIRC")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fig_dir", required=True)
    args = ap.parse_args()

    pcol = f"{args.gene}_{args.agg}"
    pred = pd.read_csv(args.pred_csv)
    clin = pd.read_csv(args.clinical, sep="\t")[["case_submitter_id", "ajcc_m"]]
    df = pred.merge(clin, on="case_submitter_id", how="inner")
    df["M"] = df["ajcc_m"].astype(str).str.strip().str.upper()
    df = df[df["M"].isin(["M0", "M1"])].copy()
    vals = pd.to_numeric(df[pcol], errors="coerce").to_numpy()
    y = (df["M"] == "M1").astype(int).to_numpy()               # 1 = metastasis
    score = -vals                                              # low PAX8 -> metastasis-prone
    prev = y.mean()
    med = np.nanmedian(vals)
    grp_low = vals < med                                       # predicted metastasis-prone

    # ranking metrics (threshold-free, prevalence-robust for AUROC)
    auroc = roc_auc_score(y, score)
    ap_pr = average_precision_score(y, score)
    auroc_ci = boot_ci(y, score, roc_auc_score)
    ap_ci = boot_ci(y, score, average_precision_score)

    # per-group metastasis rate + Wilson CI
    def rate_ci(mask):
        k, ntot = int(y[mask].sum()), int(mask.sum())
        lo, hi = proportion_confint(k, ntot, method="wilson")
        return k, ntot, k / ntot, lo, hi
    hi_k, hi_n, hi_r, hi_lo, hi_hi = rate_ci(~grp_low)         # PAX8-high
    lo_k, lo_n, lo_r, lo_lo, lo_hi = rate_ci(grp_low)          # PAX8-low
    # 2x2 for Fisher / OR / RR
    a, b = hi_n - hi_k, hi_k       # high: M0, M1
    c, d = lo_n - lo_k, lo_k       # low : M0, M1
    orr, p_fisher = fisher_exact([[d, c], [b, a]])
    rr = (lo_r / hi_r) if hi_r > 0 else np.nan

    # median-threshold classifier view (positive = metastasis; predict low->pos)
    tp, fn = d, (y[~grp_low].sum())               # low&M1 ; high&M1
    fp, tn = c, a                                  # low&M0 ; high&M0
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    bal_acc = np.nanmean([sens, spec])
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    figdir = Path(args.fig_dir); figdir.mkdir(parents=True, exist_ok=True)
    summ = {"gene": args.gene, "agg": args.agg, "n": len(y), "prevalence_M1": float(prev),
            "auroc": float(auroc), "auroc_lo": auroc_ci[0], "auroc_hi": auroc_ci[1],
            "pr_auc": float(ap_pr), "pr_auc_lo": ap_ci[0], "pr_auc_hi": ap_ci[1],
            "rate_high": hi_r, "rate_low": lo_r, "odds_ratio": float(orr),
            "risk_ratio": float(rr), "fisher_p": float(p_fisher),
            "sensitivity": float(sens), "specificity": float(spec),
            "balanced_acc": float(bal_acc), "ppv": float(ppv), "npv": float(npv)}
    pd.DataFrame([summ]).to_csv(out / f"{pcol}_metastasis_metrics.csv", index=False)
    for k, v in summ.items():
        print(f"  {k}: {v}")

    # figure: (1) rate bar w/ CI, (2) ROC  (PR-AUC kept in the CSV, not plotted)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    ax = axes[0]
    xs = [0, 1]
    rates = [hi_r, lo_r]
    errs = [[hi_r - hi_lo, lo_r - lo_lo], [hi_hi - hi_r, lo_hi - lo_r]]
    ax.bar(xs, rates, color=[PALETTE["high"], PALETTE["low"]], width=0.6)
    ax.errorbar(xs, rates, yerr=errs, fmt="none", ecolor="black", capsize=4, lw=1)
    ax.axhline(prev, ls="--", color="0.5", lw=1, label=f"cohort prevalence {prev*100:.1f}%")
    for x, r, n in zip(xs, rates, [hi_n, lo_n]):
        ax.text(x, r + 0.01, f"{r*100:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels([f"{args.gene}-high", f"{args.gene}-low"])
    ax.set_ylabel("metastasis (M1) rate"); ax.set_ylim(0, max(rates) * 1.5 + 0.05)
    ax.set_title(f"metastasis rate by group\nRR={rr:.2f}, OR={orr:.2f}, Fisher p={p_fisher:.2g}", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper left")

    fpr, tpr, _ = roc_curve(y, score)
    axes[1].plot(fpr, tpr, color="#d62728", lw=1.6)
    axes[1].plot([0, 1], [0, 1], ls="--", color="0.6", lw=1)
    axes[1].set_xlabel("1 - specificity"); axes[1].set_ylabel("sensitivity")
    axes[1].set_title(f"ROC: -{args.gene} -> M1\nAUC={auroc:.3f} "
                      f"[{auroc_ci[0]:.3f},{auroc_ci[1]:.3f}]", fontsize=8)
    axes[1].set_aspect("equal")

    fig.suptitle(f"{args.cohort}: predicted {args.gene} ({args.agg}) vs metastasis "
                 f"(M1 prevalence {prev*100:.1f}%, n={len(y)})", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(figdir / f"{pcol}_metastasis_metrics.png", dpi=250, bbox_inches="tight")
    fig.savefig(figdir / f"{pcol}_metastasis_metrics.pdf", bbox_inches="tight")
    print(f"[out] {figdir / (pcol + '_metastasis_metrics.png')}")


if __name__ == "__main__":
    main()
