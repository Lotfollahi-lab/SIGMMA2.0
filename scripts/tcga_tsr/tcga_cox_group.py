#!/usr/bin/env python
"""Cox PH for the DICHOTOMIZED TSR group (TSR-high vs TSR-low), reference=low.

Uses lifelines CoxPHFitter (Efron ties = field standard; statsmodels PHReg
defaulted to Breslow). Reports per model the group_high HR + 95% CI + p, the
concordance index, and a Schoenfeld proportional-hazards test p-value for each
term so the PH assumption is actually checked (not just assumed).

Three nested models per split (reads the already-written per-case csv):
  M1  group_high                      (univariate; "TSR low vs high only")
  M2  group_high + age                (age effect removed)
  M3  group_high + age + stage        (age + stage removed)
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

STAGE_MAP = {"STAGE I": 1, "STAGE IA": 1, "STAGE IB": 1,
             "STAGE II": 2, "STAGE IIA": 2, "STAGE IIB": 2,
             "STAGE III": 3, "STAGE IIIA": 3, "STAGE IIIB": 3, "STAGE IIIC": 3,
             "STAGE IV": 4, "STAGE IVA": 4, "STAGE IVB": 4}


def fit(df: pd.DataFrame, covs: list[str]):
    """Fit CoxPHFitter (Efron) on duration 'T', event 'E', covariates `covs`.
    Returns (per-cov dict of (HR, lo, hi, p), concordance, per-cov PH-test p)."""
    sub = df[["T", "E"] + covs].dropna().copy()
    cph = CoxPHFitter()
    cph.fit(sub, duration_col="T", event_col="E")  # Efron ties by default
    s = cph.summary  # index = covariate
    rows = {}
    for c in covs:
        rows[c] = (s.loc[c, "exp(coef)"],
                   s.loc[c, "exp(coef) lower 95%"],
                   s.loc[c, "exp(coef) upper 95%"],
                   s.loc[c, "p"])
    # Schoenfeld PH test per covariate.
    try:
        ph = proportional_hazard_test(cph, sub, time_transform="rank")
        ph_p = {c: float(ph.summary.loc[c, "p"]) for c in covs}
    except Exception as e:
        ph_p = {c: float("nan") for c in covs}
        ph_p["_err"] = str(e)
    return rows, float(cph.concordance_index_), ph_p


def run_one(csv: Path, label: str, fh):
    case = pd.read_csv(csv)
    t_col = "time_days" if "time_days" in case.columns else "OS_time_days"
    e_col = "event" if "event" in case.columns else "OS_event"
    df = pd.DataFrame()
    df["T"] = pd.to_numeric(case[t_col], errors="coerce") / 30.44   # months
    df["E"] = pd.to_numeric(case[e_col], errors="coerce").astype(int)
    grp = case["group"].astype(str)
    df["group_high"] = grp.str.contains("high").astype(float)  # ref = low
    age = pd.to_numeric(case.get("age_days"), errors="coerce") / 365.25
    df["age_z"] = (age - age.mean()) / age.std()
    df["stage_ord"] = (case.get("stage").astype(str).str.strip().str.upper()
                       .map(STAGE_MAP))
    n_hi = int(df["group_high"].sum()); n_lo = int((1 - df["group_high"]).sum())

    hdr = (f"\n=== {label}  (n={len(df)}, events={int(df['E'].sum())}, "
           f"TSR-high={n_hi} / TSR-low={n_lo})  [lifelines CoxPHFitter, Efron] ===")
    print(hdr); fh.write(hdr + "\n")

    models = {
        "M1 group_high (univariate)": ["group_high"],
        "M2 group_high + age": ["group_high", "age_z"],
        "M3 group_high + age + stage": ["group_high", "age_z", "stage_ord"],
    }
    for mname, covs in models.items():
        try:
            rows, c_idx, ph_p = fit(df, covs)
            line = f"[{mname}] C-index={c_idx:.3f}"
            for c in covs:
                hr, lo, hi, pv = rows[c]
                line += (f"  {c}: HR={hr:.3f} [{lo:.3f},{hi:.3f}] p={pv:.4g}"
                         f" (PH_p={ph_p.get(c, float('nan')):.3g})")
        except Exception as e:
            line = f"[{mname}] FAILED: {e}"
        print(line); fh.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/tcga_kirc_tsr")
    ap.add_argument("--stem", default="tcga_kirc_tsr")
    ap.add_argument("--out", default="outputs/tcga_kirc_tsr/cox_group_hr.txt")
    args = ap.parse_args()
    base = Path(args.base)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for split in ("fixed05", "median"):
            csv = base / split / f"{args.stem}_cases.csv"
            if csv.exists():
                run_one(csv, f"{split} split", fh)
            else:
                print(f"[skip] missing {csv}")
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
