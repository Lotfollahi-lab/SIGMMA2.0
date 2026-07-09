#!/usr/bin/env python
"""Screen which CLINICAL-variable high/low split separates survival in TCGA-KIRC.

Uses only the GDC clinical TSV (no SIGMMA / molecular score). For each candidate
stratification we form a binary high-risk vs low-risk grouping, then report:
  * Kaplan-Meier log-rank test (statsmodels survdiff)
  * univariable Cox PH (PHReg): hazard ratio of high vs low + Wald p

Endpoints (TCGA-CDR convention in this TSV): OS / DSS / PFI / DFI, each with an
event column <EP> (0/1) and time column <EP>_time_days.

We also fit one multivariable Cox (stage ordinal + age + gender) per endpoint to
show which variables stay independently prognostic.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.duration.survfunc import SurvfuncRight, survdiff
from statsmodels.duration.hazard_regression import PHReg

PAL = {"high": "#d62728", "low": "#4477AA"}


def stage_ord(s):
    s = str(s).strip().upper()
    if s.startswith("STAGE IV"):
        return 4
    if s.startswith("STAGE III"):
        return 3
    if s.startswith("STAGE II"):
        return 2
    if s.startswith("STAGE I"):
        return 1
    return np.nan


def t_ord(s):
    s = str(s).strip().upper()
    if s.startswith("T4"):
        return 4
    if s.startswith("T3"):
        return 3
    if s.startswith("T2"):
        return 2
    if s.startswith("T1"):
        return 1
    return np.nan


def build_groups(df):
    """Return list of (name, description, pandas Series of {'high','low',NaN})."""
    groups = []

    so = df["ajcc_stage"].map(stage_ord)
    g = pd.Series(np.where(so >= 3, "high", np.where(so <= 2, "low", None)),
                  index=df.index)
    groups.append(("stage_III_IV_vs_I_II", "Stage III/IV vs I/II", g))

    to = df["ajcc_t"].map(t_ord)
    g = pd.Series(np.where(to >= 3, "high", np.where(to <= 2, "low", None)),
                  index=df.index)
    groups.append(("T_T3T4_vs_T1T2", "T3/T4 vs T1/T2", g))

    n = df["ajcc_n"].astype(str).str.strip().str.upper()
    g = pd.Series(np.where(n.str.startswith("N1") | n.str.startswith("N2"), "high",
                           np.where(n == "N0", "low", None)), index=df.index)
    groups.append(("N_pos_vs_N0", "Node+ (N1/N2) vs N0", g))

    m = df["ajcc_m"].astype(str).str.strip().str.upper()
    g = pd.Series(np.where(m == "M1", "high", np.where(m == "M0", "low", None)),
                  index=df.index)
    groups.append(("M_M1_vs_M0", "Metastasis M1 vs M0", g))

    age_yr = df["age_at_diagnosis_days"] / 365.25
    med = age_yr.median()
    g = pd.Series(np.where(age_yr > med, "high", np.where(age_yr <= med, "low", None)),
                  index=df.index)
    g[age_yr.isna()] = None
    groups.append(("age_above_median", f"Age > median ({med:.0f}y)", g))

    sex = df["gender"].astype(str).str.strip().str.lower()
    g = pd.Series(np.where(sex == "male", "high", np.where(sex == "female", "low", None)),
                  index=df.index)
    groups.append(("male_vs_female", "Male vs Female", g))

    return groups


def fit_split(time, event, grp):
    """grp: array of 0(low)/1(high). Returns dict of stats."""
    m = (~np.isnan(time)) & (~np.isnan(event)) & (~np.isnan(grp))
    time, event, grp = time[m], event[m], grp[m]
    out = {"n_low": int((grp == 0).sum()), "n_high": int((grp == 1).sum()),
           "ev_low": int(event[grp == 0].sum()), "ev_high": int(event[grp == 1].sum())}
    if out["n_low"] < 5 or out["n_high"] < 5 or out["ev_low"] + out["ev_high"] < 5:
        out.update(logrank_p=np.nan, hr=np.nan, hr_lo=np.nan, hr_hi=np.nan, cox_p=np.nan)
        return out
    chisq, p = survdiff(time, event, grp)
    out["logrank_p"] = float(p)
    try:
        mod = PHReg(time, grp.reshape(-1, 1), status=event)
        res = mod.fit()
        beta = res.params[0]
        se = res.bse[0]
        out["hr"] = float(np.exp(beta))
        out["hr_lo"] = float(np.exp(beta - 1.96 * se))
        out["hr_hi"] = float(np.exp(beta + 1.96 * se))
        out["cox_p"] = float(res.pvalues[0])
    except Exception as e:
        out.update(hr=np.nan, hr_lo=np.nan, hr_hi=np.nan, cox_p=np.nan)
        print(f"  Cox failed: {e}")
    return out


def km_plot(time, event, grp, title, path):
    m = (~np.isnan(time)) & (~np.isnan(event)) & (~np.isnan(grp))
    time, event, grp = time[m], event[m], grp[m]
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    for lvl, lab in [(1, "high"), (0, "low")]:
        sel = grp == lvl
        if sel.sum() == 0:
            continue
        sf = SurvfuncRight(time[sel], event[sel])
        sf.plot(ax=ax)
        # recolor last two lines (step + censor) to palette
        ax.lines[-1].set_color(PAL[lab]); ax.lines[-1].set_label(None)
        ax.lines[-2].set_color(PAL[lab])
        ax.lines[-2].set_label(f"{lab} (n={int(sel.sum())}, ev={int(event[sel].sum())})")
    ax.set_xlabel("days"); ax.set_ylabel("survival prob."); ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7); ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical_tsv", required=True)
    ap.add_argument("--out_dir", default="outputs/clinical_outcome")
    ap.add_argument("--fig_dir", default="figures/clinical_outcome")
    ap.add_argument("--endpoints", nargs="+", default=["OS", "DSS", "PFI", "DFI"])
    ap.add_argument("--cohort", default="KIRC")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.clinical_tsv, sep="\t")
    for c in ["age_at_diagnosis_days"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"Loaded {len(df)} cases from {args.clinical_tsv}")

    groups = build_groups(df)
    rows = []
    for ep in args.endpoints:
        if ep not in df.columns or f"{ep}_time_days" not in df.columns:
            print(f"  endpoint {ep} missing, skip"); continue
        time = pd.to_numeric(df[f"{ep}_time_days"], errors="coerce").to_numpy(float)
        event = pd.to_numeric(df[ep], errors="coerce").to_numpy(float)
        for name, desc, g in groups:
            grp = g.map({"low": 0.0, "high": 1.0}).to_numpy(float)
            st = fit_split(time, event, grp)
            st.update(endpoint=ep, variable=name, description=desc)
            rows.append(st)
            tag = f"{st['logrank_p']:.2e}" if not np.isnan(st['logrank_p']) else "na"
            print(f"[{ep}] {desc:28s} logrank_p={tag} "
                  f"HR={st['hr']:.2f} cox_p={st['cox_p']:.2e}"
                  if not np.isnan(st['hr']) else f"[{ep}] {desc}: insufficient")
            if not np.isnan(st["logrank_p"]) and st["logrank_p"] < 0.05 and ep in ("OS", "DSS"):
                km_plot(time, event, grp, f"KIRC {ep}: {desc}\nlog-rank p={st['logrank_p']:.1e}, HR={st['hr']:.2f}",
                        fig_dir / f"tcga_kirc_{ep.lower()}_{name}.png")

    res = pd.DataFrame(rows)[["endpoint", "variable", "description", "n_low", "n_high",
                              "ev_low", "ev_high", "logrank_p", "hr", "hr_lo", "hr_hi", "cox_p"]]
    res = res.sort_values(["endpoint", "logrank_p"])
    csv_path = out_dir / f"tcga_{args.cohort.lower()}_clinical_risk_screen.csv"
    res.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    # --- multivariable Cox (stage ordinal + age + male) per endpoint ---
    df["_stage"] = df["ajcc_stage"].map(stage_ord)
    df["_age"] = df["age_at_diagnosis_days"] / 365.25
    df["_male"] = (df["gender"].astype(str).str.lower() == "male").astype(float)
    mv_rows = []
    for ep in args.endpoints:
        if ep not in df.columns:
            continue
        sub = df[[f"{ep}_time_days", ep, "_stage", "_age", "_male"]].apply(
            pd.to_numeric, errors="coerce").dropna()
        if len(sub) < 30 or sub[ep].sum() < 10:
            continue
        X = sub[["_stage", "_age", "_male"]].to_numpy(float)
        try:
            res_mv = PHReg(sub[f"{ep}_time_days"].to_numpy(float), X,
                           status=sub[ep].to_numpy(float)).fit()
            for nm, b, se, p in zip(["stage", "age", "male"], res_mv.params,
                                    res_mv.bse, res_mv.pvalues):
                mv_rows.append({"endpoint": ep, "term": nm, "HR": float(np.exp(b)),
                                "HR_lo": float(np.exp(b - 1.96 * se)),
                                "HR_hi": float(np.exp(b + 1.96 * se)),
                                "p": float(p), "n": len(sub), "events": int(sub[ep].sum())})
        except Exception as e:
            print(f"  MV Cox {ep} failed: {e}")
    mv = pd.DataFrame(mv_rows)
    mv_path = out_dir / f"tcga_{args.cohort.lower()}_clinical_multivariable_cox.csv"
    mv.to_csv(mv_path, index=False)
    print(f"Wrote {mv_path}")
    print("\n=== univariable screen (sorted by log-rank p) ===")
    with pd.option_context("display.width", 160, "display.max_rows", 100):
        print(res.to_string(index=False))
    print("\n=== multivariable Cox ===")
    print(mv.to_string(index=False))


if __name__ == "__main__":
    main()
