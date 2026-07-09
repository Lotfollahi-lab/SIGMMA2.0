#!/usr/bin/env python
"""Negative control for the H&E->SIGMMA PAX8 probe: apply the frozen kidney-RCC
PAX8 probe to PAX8-NEGATIVE tissues (TCGA colorectal COAD/READ) and compare the
predicted-PAX8 distribution to PAX8-POSITIVE kidney (KIRC).

PAX8 is a renal/Mullerian/thyroid lineage marker; colon and rectum are reliably
PAX8-negative by IHC. So a probe that captures real PAX8 biology (not a kidney
artefact) should assign SYSTEMATICALLY LOWER predicted PAX8 to COAD/READ than to
KIRC. This also shows that KIRC 'PAX8-low' is a RELATIVE label (within-cohort
below median), not absence: a genuinely PAX8-negative tissue anchors the true low
end.

Per slide we predict per-nucleus PAX8 (frozen bundle on z_sigmma_concat) and take
the slide mean. Groups: KIRC-high / KIRC-low (split at KIRC slide-median), COAD,
READ. Reports Mann-Whitney KIRC-low vs COAD/READ (is even the low kidney group
above colorectal?) and plots per-slide distributions.

Run in the sigmma poetry venv (h5py + sklearn + joblib + matplotlib).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"KIRC-high": "#d62728", "KIRC-low": "#f4a582",
          "COAD": "#4393c3", "READ": "#2166ac"}


def slide_means(feat_dir, bundle, gene):
    keys, mu, sd = bundle["keys"], bundle["mu"], bundle["sd"]
    model = bundle["models"][gene]
    out = []
    for h5 in sorted(Path(feat_dir).glob("*.h5")):
        with h5py.File(h5, "r") as f:
            try:
                X = np.concatenate([f[k][:] for k in keys], axis=1).astype(np.float32)
            except KeyError:
                continue
        if X.shape[1] != bundle["n_features"]:
            continue
        pred = model.predict((X - mu) / sd)
        out.append({"sample": h5.stem, "pax_mean": float(pred.mean())})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--gene", default="PAX8")
    ap.add_argument("--kirc_dir", required=True)
    ap.add_argument("--coad_dir", required=True)
    ap.add_argument("--read_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fig_dir", required=True)
    args = ap.parse_args()

    bundle = joblib.load(args.bundle)
    gene = args.gene
    kirc = slide_means(args.kirc_dir, bundle, gene)
    coad = slide_means(args.coad_dir, bundle, gene)
    read = slide_means(args.read_dir, bundle, gene)
    print(f"[n slides] KIRC={len(kirc)} COAD={len(coad)} READ={len(read)}", flush=True)

    kmed = kirc["pax_mean"].median()
    groups = {
        "KIRC-high": kirc.loc[kirc["pax_mean"] >= kmed, "pax_mean"].to_numpy(),
        "KIRC-low": kirc.loc[kirc["pax_mean"] < kmed, "pax_mean"].to_numpy(),
        "COAD": coad["pax_mean"].to_numpy(),
        "READ": read["pax_mean"].to_numpy(),
    }
    order = ["KIRC-high", "KIRC-low", "COAD", "READ"]

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    figdir = Path(args.fig_dir); figdir.mkdir(parents=True, exist_ok=True)

    rows = [{"group": g, "n": len(v), "median": float(np.median(v)),
             "mean": float(np.mean(v)), "q25": float(np.percentile(v, 25)),
             "q75": float(np.percentile(v, 75))} for g, v in groups.items()]
    summ = pd.DataFrame(rows)
    print(summ.to_string(index=False))
    # is even KIRC-low above colorectal?
    tests = []
    for neg in ("COAD", "READ"):
        for ref in ("KIRC-high", "KIRC-low"):
            u, p = mannwhitneyu(groups[ref], groups[neg], alternative="greater")
            tests.append({"ref": ref, "neg": neg, "MWU_p_ref_gt_neg": float(p)})
            print(f"  {ref} > {neg}?  MWU p={p:.2e}")
    summ.to_csv(out / f"{gene}_negative_control_summary.csv", index=False)
    pd.DataFrame(tests).to_csv(out / f"{gene}_negative_control_tests.csv", index=False)

    # violin + jitter
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    data = [groups[g] for g in order]
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(COLORS[order[i]]); pc.set_alpha(0.55)
    parts["cmedians"].set_color("black")
    rng = np.random.default_rng(0)
    for i, g in enumerate(order):
        v = groups[g]
        ax.scatter(np.full(len(v), i + 1) + rng.uniform(-0.08, 0.08, len(v)), v,
                   s=5, color=COLORS[g], alpha=0.35, edgecolors="none")
    ax.axhline(kmed, ls="--", color="0.4", lw=1,
               label=f"KIRC median cutoff = {kmed:.2f}")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([f"{g}\n(n={len(groups[g])})" for g in order])
    ax.set_ylabel(f"predicted {gene} per slide (log1p-CP10k)")
    ax.set_title(f"Negative control: frozen kidney PAX8 probe on PAX8-negative "
                 f"tissues\nPAX8+ kidney (KIRC) vs PAX8- colorectal (COAD/READ)",
                 fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figdir / f"{gene}_negative_control.png", dpi=250, bbox_inches="tight")
    fig.savefig(figdir / f"{gene}_negative_control.pdf", bbox_inches="tight")
    print(f"[out] {figdir / (gene + '_negative_control.png')}")


if __name__ == "__main__":
    main()
