#!/usr/bin/env python
"""Stage 5: TSR high/low survival analysis on TCGA-READ (statsmodels).

Merge per-slide stroma_fraction with the GDC OS table, collapse to case
(patient) level, split high/low by median, then:
  * Kaplan-Meier curves per group + log-rank test (survdiff)
  * Cox PH (PHReg): stroma_fraction (continuous), optionally + age.

Survival convention: classic TSR studies call high stromal fraction
("stroma-high") the poor-prognosis group; we report the empirical direction.
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

PALETTE = {"TSR-high (stroma-high)": "#d62728", "TSR-low (stroma-low)": "#4477AA"}

# AJCC stage -> ordinal 1-4 for Cox adjustment (KIRC = the dominant confounder).
STAGE_MAP = {"STAGE I": 1, "STAGE IA": 1, "STAGE IB": 1,
             "STAGE II": 2, "STAGE IIA": 2, "STAGE IIB": 2,
             "STAGE III": 3, "STAGE IIIA": 3, "STAGE IIIB": 3, "STAGE IIIC": 3,
             "STAGE IV": 4, "STAGE IVA": 4, "STAGE IVB": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsr_csv", required=True)
    ap.add_argument("--survival_tsv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fig_dir", default="figures/clinical_outcome")
    ap.add_argument("--cohort", default="READ",
                    help="TCGA project short name; sets plot title + output "
                         "filenames (tcga_<cohort_lower>_tsr_*).")
    ap.add_argument("--endpoint", default="OS",
                    choices=["OS", "DSS", "DFI", "PFI"],
                    help="survival endpoint (TCGA-CDR). Reads the <EP> event "
                         "column and <EP>_time_days from --survival_tsv. "
                         "OS=overall, DSS=disease-specific (non-cancer deaths "
                         "censored), DFI=disease-free interval, PFI=progression-"
                         "free interval. Default OS.")
    ap.add_argument("--min_cells", type=int, default=2000,
                    help="drop slides with fewer scored cells (unreliable TSR).")
    ap.add_argument("--min_valid_fields", type=int, default=0,
                    help="drop slides whose protocol TSR rested on fewer than "
                         "this many valid hotspot fields (n_valid_fields col); "
                         "0 = off. With the MAX-hotspot rule a single-field TSR "
                         "has no hotspot selection and is noisy.")
    ap.add_argument("--min_os_days", type=int, default=0,
                    help="drop CENSORED (event=0) cases with OS_time_days below "
                         "this (uninformative early loss-to-follow-up). DEATHS "
                         "are always kept regardless. 0 = off.")
    ap.add_argument("--stroma_cutoff", type=float, default=None,
                    help="fixed stroma_fraction threshold for stroma-high "
                         "(>= cutoff). PMC7887673 uses 0.5. Default: cohort "
                         "median. NOTE: stroma_fraction here is a CELL-COUNT "
                         "fraction, not the protocol's area %; a fixed 0.5 is "
                         "not unit-matched to the area-based 50% cutoff.")
    args = ap.parse_args()
    cohort = args.cohort.upper()
    stem = f"tcga_{cohort.lower()}_{args.endpoint.lower()}_tsr"

    tsr = pd.read_csv(args.tsr_csv)
    tsr = tsr[tsr["n_cells"] >= args.min_cells].copy()
    if args.min_valid_fields > 0 and "n_valid_fields" in tsr.columns:
        n0v = len(tsr)
        tsr = tsr[tsr["n_valid_fields"] >= args.min_valid_fields].copy()
        print(f"[min_valid_fields={args.min_valid_fields}] {n0v} -> {len(tsr)} slides")
    surv = pd.read_csv(args.survival_tsv, sep="\t")
    # tsr sample_name is the .svs stem; survival keys on file_name "<stem>.svs".
    surv["stem"] = surv["file_name"].str.replace(r"\.svs$", "", regex=True)

    m = tsr.merge(surv, left_on="sample_name", right_on="stem", how="inner")
    print(f"[merge] tsr={len(tsr)} surv={len(surv)} -> matched slides {len(m)}")

    # Collapse to patient (case) level. Protocol (van Pelt/Mesker): score the
    # SECTION WITH THE HIGHEST STROMA -> take the MAX stroma_fraction across a
    # case's slides (not the mean). Combined with the within-slide MAX hotspot
    # in tcga_protocol_tsr.py this is a double-MAX, so the distribution
    # saturates high and a fixed-0.5 split becomes imbalanced (lower power);
    # report both fixed-0.5 (clinical) and median splits.
    ep = args.endpoint.upper()
    ev_col, t_col = ep, f"{ep}_time_days"
    if ev_col not in m.columns or t_col not in m.columns:
        raise SystemExit(f"[endpoint] {ev_col}/{t_col} not in merged columns: "
                         f"{list(m.columns)}")
    m[t_col] = pd.to_numeric(m[t_col], errors="coerce")
    m[ev_col] = pd.to_numeric(m[ev_col], errors="coerce")
    case = (m.groupby("case_submitter_id")
              .agg(stroma_fraction=("stroma_fraction", "max"),
                   n_cells=("n_cells", "sum"),
                   event=(ev_col, "first"),
                   time_days=(t_col, "first"),
                   age_days=("age_at_diagnosis_days", "first"),
                   stage=("ajcc_stage", "first"))
              .reset_index())
    case["endpoint"] = ep
    n0 = len(case)
    case = case[(case["time_days"] > 0) & case["event"].isin([0, 1])
                & case["stroma_fraction"].notna()].copy()
    if args.min_os_days > 0:
        drop = (case["event"] == 0) & (case["time_days"] < args.min_os_days)
        print(f"[min_os_days={args.min_os_days}] dropping {int(drop.sum())} "
              f"early-censored {ep} cases (events kept)")
        case = case[~drop].copy()
    print(f"[case] {n0} cases -> {len(case)} with valid {ep} & TSR; "
          f"events={int(case['event'].sum())}")

    med = case["stroma_fraction"].median()
    cut = args.stroma_cutoff if args.stroma_cutoff is not None else med
    print(f"[split] stroma-high if stroma_fraction >= {cut:.4f} "
          f"({'fixed' if args.stroma_cutoff is not None else 'median'}; "
          f"cohort median={med:.4f})")
    case["group"] = np.where(case["stroma_fraction"] >= cut,
                             "TSR-high (stroma-high)", "TSR-low (stroma-low)")
    time = case["time_days"].to_numpy() / 30.44      # months
    event = case["event"].to_numpy().astype(int)
    grp = case["group"].to_numpy()

    # Log-rank.
    chisq, p_lr = survdiff(time, event, grp)
    print(f"[logrank] chisq={chisq:.3f} p={p_lr:.4g}")

    # Cox PH (continuous stroma_fraction; + standardized age if available).
    cox_txt = []
    try:
        exog = case[["stroma_fraction"]].copy()
        age = pd.to_numeric(case["age_days"], errors="coerce")
        if age.notna().sum() >= len(case) * 0.8:
            exog["age_yr_z"] = ((age / 365.25) - (age / 365.25).mean()) / (age / 365.25).std()
        st = case["stage"].astype(str).str.strip().str.upper().map(STAGE_MAP)
        if st.notna().sum() >= len(case) * 0.8:
            exog["stage_ord"] = st       # AJCC I-IV; dominant KIRC confounder
        exog = exog.fillna(exog.mean())
        mod = PHReg(time, exog.to_numpy(), status=event).fit()
        for name, coef, pv, hr in zip(exog.columns, mod.params, mod.pvalues,
                                      np.exp(mod.params)):
            cox_txt.append(f"{name}: HR={hr:.3f} (coef={coef:.3f}, p={pv:.4g})")
        print("[cox]", " | ".join(cox_txt))
    except Exception as e:
        cox_txt.append(f"cox failed: {e}")
        print("[cox] failed:", e)

    # KM plot.
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    for g in ["TSR-low (stroma-low)", "TSR-high (stroma-high)"]:
        sel = grp == g
        n = int(sel.sum()); ev = int(event[sel].sum())
        sf = SurvfuncRight(time[sel], event[sel])
        ax.step(sf.surv_times, sf.surv_prob, where="post",
                color=PALETTE[g], lw=1.8, label=f"{g}  (n={n}, d={ev})")
    ax.set_xlabel(f"{ep} (months)", fontsize=9)
    ax.set_ylabel(f"{ep} probability", fontsize=9)
    ax.set_ylim(0, 1.02)
    split_lbl = "fixed-0.5" if args.stroma_cutoff is not None else "median"
    ax.set_title(f"TCGA-{cohort}  H&E→SIGMMA cell-type TSR  ({ep})\n"
                 f"log-rank p={p_lr:.3g}  ({split_lbl} split, n={len(case)})",
                 fontsize=8)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"{stem}_km.{ext}", dpi=200)
    print(f"[fig] {fig_dir}/{stem}_km.png")

    case.to_csv(out_dir / f"{stem}_cases.csv", index=False)
    with open(out_dir / f"{stem}_stats.txt", "w") as f:
        f.write(f"endpoint={ep} n_cases={len(case)} events={int(event.sum())} "
                f"median_stroma_frac={med:.4f} split_cutoff={cut:.4f}\n")
        f.write(f"logrank chisq={chisq:.4f} p={p_lr:.6g}\n")
        f.write("\n".join(cox_txt) + "\n")
        for g in np.unique(grp):
            sel = grp == g
            f.write(f"{g}: n={int(sel.sum())} events={int(event[sel].sum())} "
                    f"median_OS_mo={np.median(time[sel]):.1f}\n")
    print(f"[out] {out_dir}/{stem}_cases.csv + _stats.txt")


if __name__ == "__main__":
    main()
