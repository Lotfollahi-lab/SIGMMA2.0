"""Report-ready per-tissue retrieval significance: section-level paired tests.

Unit = SECTION (calibrated for clustered tile data with few clusters; see
feedback-retrieval-significance-unit). For each in-house tissue and metric,
compares SIGMMA vs each FT baseline AND vs the best baseline (highest mean) with:
  * paired t-test (ttest_rel)
  * Wilcoxon signed-rank (exact; the distribution-free companion)
n = #sections. Writes a tidy CSV for the paper + prints a summary.

Wilcoxon floor caveat: two-sided min p = 2 / 2^n, so n=4 -> 0.125, n=5 -> 0.0625,
n=6 -> 0.031, n=8 -> 0.0078. Wilcoxon cannot reach p<0.05 for n<6 even at 100%
consistency; paired-t can. Report both; lean on paired-t for n<6, cross-check
direction with Wilcoxon.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

JSON = "outputs/retrieval/normalized_metrics_step2000.json"
META = ("/nfs/users/nfs_d/dj16/lustre_dabin/data/sanger_in_house_xenium/"
        "sample_metadata.csv")
OUT = "outputs/retrieval/per_tissue_significance_step2000.csv"
DIR = "image"
BASELINES = ["bleep", "clip", "plip", "omiclip"]
BLABEL = {"bleep": "BLEEP", "clip": "CLIP", "plip": "PLIP", "omiclip": "OmiCLIP"}
PUBLIC = {"IDC", "PAAD", "LUAD", "SKCM"}
METRICS = [("recall_at_1pct", "R@1%"), ("mrr", "MRR"), ("recall_at_1", "R@1")]
ORDER = ["beacon", "CTCL", "pediatric_ovary", "renal_cell_carcinoma",
         "pancreas", "liver", "lung", "whole_embryo", "glioblastoma",
         "adrenocortical_carcinoma"]
DISP = {"beacon": "Skin-beacon", "CTCL": "Skin-CTCL", "pediatric_ovary": "Ovary",
        "renal_cell_carcinoma": "Kidney-RCC", "pancreas": "Pancreas",
        "liver": "Liver", "lung": "Lung", "whole_embryo": "WholeEmbryo",
        "glioblastoma": "Glioblastoma", "adrenocortical_carcinoma": "Adrenocortical"}


def stars(p):
    if not np.isfinite(p):
        return "n/a"
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def main():
    d = json.load(open(JSON))
    meta = pd.read_csv(META)
    tt = dict(zip(meta["sample_id"], meta["tissue_type"]))
    secs = list(d["a01"][DIR]["per_section"].keys())
    by_t = {}
    for s in secs:
        by_t.setdefault(tt.get(s), []).append(s)

    def vals(sl, met, k):
        return np.array([d[k][DIR]["per_section"][s][met] for s in sl])

    rows = []
    for tissue in ORDER:
        if tissue not in by_t:
            continue
        sl = by_t[tissue]
        n = len(sl)
        for met, mlbl in METRICS:
            sig = vals(sl, met, "a01")
            means = {b: vals(sl, met, b).mean() for b in BASELINES}
            best_b = max(means, key=means.get)
            for b in BASELINES:
                base = vals(sl, met, b)
                diff = sig - base
                pt = ttest_rel(sig, base).pvalue if n >= 2 else np.nan
                # Wilcoxon: needs >=1 nonzero diff; guard all-zero / n<...
                try:
                    pw = wilcoxon(sig, base, zero_method="wilcox",
                                  alternative="two-sided", mode="exact").pvalue
                except Exception:
                    pw = np.nan
                rows.append(dict(
                    tissue=DISP[tissue], n_sec=n, panel="OOD" if tissue in PUBLIC else "in",
                    metric=mlbl, baseline=BLABEL[b], is_best=(b == best_b),
                    sigmma_mean=round(float(sig.mean()), 4),
                    base_mean=round(float(base.mean()), 4),
                    mean_diff=round(float(diff.mean()), 4),
                    paired_t_p=pt, paired_t_sig=stars(pt),
                    wilcoxon_p=pw, wilcoxon_sig=stars(pw),
                ))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT}  ({len(df)} rows)\n")

    # summary: vs BEST baseline only, per tissue x metric
    print("=== SIGMMA vs BEST baseline (section-level) ===")
    print(f"{'tissue':14}{'n':>3} {'metric':>6} {'best base':>9} "
          f"{'Δ':>8} {'paired-t':>10} {'sig':>4} {'Wilcoxon':>10} {'sig':>4}")
    best = df[df.is_best].copy()
    mo = {"R@1%": 0, "MRR": 1, "R@1": 2}
    best["mk"] = best["metric"].map(mo)
    for tissue in [DISP[t] for t in ORDER if t in by_t]:
        sub = best[best.tissue == tissue].sort_values("mk")
        for _, r in sub.iterrows():
            print(f"{r.tissue:14}{r.n_sec:>3} {r.metric:>6} {r.baseline:>9} "
                  f"{r.mean_diff:>+8.4f} {r.paired_t_p:>10.2e} {r.paired_t_sig:>4} "
                  f"{r.wilcoxon_p:>10.2e} {r.wilcoxon_sig:>4}")


if __name__ == "__main__":
    main()
