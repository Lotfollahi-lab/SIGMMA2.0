"""Per-tissue R@1% significance via a section-random-intercept mixed-effects
logistic model on tile-level hit@1% — the principled "use all tiles but account
for within-section clustering" alternative to (a) the naive section-level paired
t-test and (b) the pseudoreplicating tile-level binomial z.

Key fact: per-section (n_hits, n_tiles) are SUFFICIENT STATISTICS for the
tile-level Bernoulli model with a section random intercept, so we reconstruct
the tile-level 0/1 vector exactly from the per-section R@1% and n_queries that
already live in normalized_metrics_step2000.json (no score-matrix re-run).

Per in-house tissue, one model over all 5 methods (SIGMMA = reference):
    hit ~ bleep + clip + plip + omiclip + (1 | section)
A negative baseline coefficient => that baseline has lower hit log-odds than
SIGMMA (SIGMMA better). Reports the Wald/posterior 2-sided p per baseline.

Two fits per tissue:
  * MIXED  : BinomialBayesMixedGLM (variational Bayes) — true section random
             intercept; p from normal approx z = fe_mean / fe_sd.
  * GEE    : binomial GEE, groups=section, exchangeable — cluster-robust Wald p
             (marginal cross-check; needs enough clusters to trust).
"""
from __future__ import annotations
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

warnings.filterwarnings("ignore")

JSON = "outputs/retrieval/normalized_metrics_step2000.json"
META = ("/nfs/users/nfs_d/dj16/lustre_dabin/data/sanger_in_house_xenium/"
        "sample_metadata.csv")
DIR = "image"
MET = "recall_at_1pct"
BASELINES = ["bleep", "clip", "plip", "omiclip"]
METHODS = BASELINES + ["a01"]
PUBLIC = {"IDC", "PAAD", "LUAD", "SKCM"}
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


def build_tiles(d, sections):
    """Reconstruct tile-level long df (hit, method dummies, section) from the
    per-section hit counts. method coded with SIGMMA as reference."""
    rows = []
    for s in sections:
        nq = int(d["a01"][DIR]["per_section"][s]["n_queries"])
        for m in METHODS:
            rate = d[m][DIR]["per_section"][s][MET]
            hits = int(round(rate * nq))
            hits = min(max(hits, 0), nq)
            y = np.zeros(nq, dtype=np.int8)
            y[:hits] = 1
            sub = pd.DataFrame({"hit": y, "section": s})
            for b in BASELINES:
                sub[b] = 1 if m == b else 0
            rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def fit_mixed(df):
    """Section random-intercept binomial GLMM via variational Bayes.
    Returns {baseline: (coef, p)} for the baseline-vs-SIGMMA fixed effects."""
    fml = "hit ~ " + " + ".join(BASELINES)
    vc = {"section": "0 + C(section)"}
    m = BinomialBayesMixedGLM.from_formula(fml, vc, df)
    r = m.fit_vb(verbose=False)
    names = list(r.model.exog_names)
    out = {}
    for b in BASELINES:
        i = names.index(b)
        mean, sd = r.fe_mean[i], r.fe_sd[i]
        z = mean / sd if sd > 0 else np.nan
        out[b] = (float(mean), float(2 * (1 - norm.cdf(abs(z)))))
    return out


def fit_gee(df):
    fml = "hit ~ " + " + ".join(BASELINES)
    m = smf.gee(fml, groups="section", data=df,
                family=Binomial(), cov_struct=Exchangeable())
    r = m.fit()
    return {b: (float(r.params[b]), float(r.pvalues[b])) for b in BASELINES}


def main():
    d = json.load(open(JSON))
    meta = pd.read_csv(META)
    tt = dict(zip(meta["sample_id"], meta["tissue_type"]))
    secs = list(d["a01"][DIR]["per_section"].keys())
    by_t = {}
    for s in secs:
        by_t.setdefault(tt.get(s, "?"), []).append(s)

    print(f"Section-random-intercept mixed logistic on tile-level hit@1% "
          f"(threshold rank<=int(N*0.01); N={d['a01'][DIR]['n_references']})\n"
          f"Coef = baseline-vs-SIGMMA log-odds (negative => SIGMMA better). "
          f"p = 2-sided.\n")
    hdr = (f"{'tissue':14}{'n_sec':>6}{'n_tile':>8}  " +
           "  ".join(f"{b.upper():>16}" for b in BASELINES))
    for tissue in ORDER:
        if tissue not in by_t:
            continue
        sl = by_t[tissue]
        df = build_tiles(d, sl)
        try:
            mixed = fit_mixed(df)
        except Exception as e:
            mixed = {b: (np.nan, np.nan) for b in BASELINES}
            print(f"  [mixed fail {tissue}: {e}]")
        try:
            gee = fit_gee(df)
        except Exception:
            gee = {b: (np.nan, np.nan) for b in BASELINES}
        # best baseline = highest mean R@1% on this tissue
        means = {b: np.mean([d[b][DIR]["per_section"][s][MET] for s in sl])
                 for b in BASELINES}
        best_b = max(means, key=means.get)
        flag = " (n<4: underpowered)" if len(sl) < 4 else ""
        ntile = int(sum(d["a01"][DIR]["per_section"][s]["n_queries"] for s in sl))
        print(f"\n### {DISP[tissue]}  n_sec={len(sl)}  n_tile={ntile:,}{flag}")
        print(f"{'baseline':10}{'mixed coef':>12}{'mixed p':>12}{'':3}"
              f"{'GEE coef':>12}{'GEE p':>12}{'':3}{'best?':>6}")
        for b in BASELINES:
            mc, mp = mixed[b]
            gc, gp = gee[b]
            mark = "<<<" if b == best_b else ""
            print(f"{b:10}{mc:12.3f}{mp:12.2e} {stars(mp):>3}"
                  f"{gc:12.3f}{gp:12.2e} {stars(gp):>3}{mark:>9}")


if __name__ == "__main__":
    main()
