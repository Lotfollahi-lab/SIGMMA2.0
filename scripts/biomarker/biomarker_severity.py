#!/usr/bin/env python
"""Associate SIGMMA-predicted CA9/PAX8 biomarker levels (per patient) with
TCGA-KIRC severity: survival (OS, DSS), AJCC stage, and tumour grade.

Input:
  --pred_csv    per-patient predictions (predict_tcga_gene.py): case_submitter_id
                + <gene>_mean / <gene>_q90 columns.
  --clinical    TCGA-KIRC_clinical.tsv (case_submitter_id, OS, OS_time_days,
                DSS, DSS_time_days, age_at_diagnosis_days, ajcc_stage, ...).
  --grade_csv   fetch_kirc_grade.py output (case_submitter_id, grade_ord).

For each (gene, aggregation) predictor:
  * Ordinal association: Spearman(predicted level, stage_ord) and (·, grade_ord).
  * Median split -> biomarker-high vs -low group.
  * Survival (OS and DSS): KM curves + log-rank p; Cox HR for group_high in three
    nested models M1 (univariate), M2 (+age), M3 (+age+stage)  [lifelines, Efron].
  * Stage/grade by group: Mann-Whitney U on the ordinal severity.

Outputs a summary CSV + a text report + KM figures. Run in the project .venv
(lifelines + statsmodels).
"""
from __future__ import annotations
import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import mannwhitneyu, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Nature-Medicine-style KM defaults: clean sans-serif, small fonts.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.8, "svg.fonttype": "none",
})

STAGE_MAP = {"STAGE I": 1, "STAGE IA": 1, "STAGE IB": 1,
             "STAGE II": 2, "STAGE IIA": 2, "STAGE IIB": 2,
             "STAGE III": 3, "STAGE IIIA": 3, "STAGE IIIB": 3, "STAGE IIIC": 3,
             "STAGE IV": 4, "STAGE IVA": 4, "STAGE IVB": 4}
PALETTE = {"high": "#d62728", "low": "#888888"}  # SIGMMA red / grey (memory)


def cox_group(df, endpoint, out_lines):
    """Cox HR for group_high across M1/M2/M3 on the given endpoint (OS/DSS)."""
    d = pd.DataFrame()
    d["T"] = pd.to_numeric(df[f"{endpoint}_time_days"], errors="coerce") / 30.44
    d["E"] = pd.to_numeric(df[endpoint], errors="coerce")
    d["group_high"] = df["group_high"].astype(float)
    age = pd.to_numeric(df["age_at_diagnosis_days"], errors="coerce") / 365.25
    d["age_z"] = (age - age.mean()) / age.std()
    d["stage_ord"] = df["stage_ord"]
    d = d[d["T"] > 0]
    models = {"M1": ["group_high"], "M2": ["group_high", "age_z"],
              "M3": ["group_high", "age_z", "stage_ord"]}
    res = {}
    for mname, covs in models.items():
        sub = d[["T", "E"] + covs].dropna()
        if sub["E"].sum() < 5 or sub["group_high"].nunique() < 2:
            res[mname] = None
            out_lines.append(f"      [{endpoint} {mname}] insufficient events")
            continue
        try:
            cph = CoxPHFitter().fit(sub, duration_col="T", event_col="E")
            s = cph.summary.loc["group_high"]
            res[mname] = (s["exp(coef)"], s["exp(coef) lower 95%"],
                          s["exp(coef) upper 95%"], s["p"])
            out_lines.append(
                f"      [{endpoint} {mname}] HR_high={s['exp(coef)']:.3f} "
                f"[{s['exp(coef) lower 95%']:.3f},{s['exp(coef) upper 95%']:.3f}] "
                f"p={s['p']:.4g}  C={cph.concordance_index_:.3f}")
        except Exception as e:  # noqa: BLE001
            res[mname] = None
            out_lines.append(f"      [{endpoint} {mname}] FAILED: {e}")
    return d, res


def km_plot(d, title, path, ylabel="Survival probability", logrank_p=None,
            hr_m1=None, hr_m3=None):
    """Nature-Medicine-style KM: censor ticks, %-axis, HR/p box, number-at-risk
    table below the axis, despined."""
    fig, ax = plt.subplots(figsize=(4.2, 4.4))
    fitters = []
    for grp, val in (("high", 1.0), ("low", 0.0)):  # fixed order: high first
        s = d[d["group_high"] == val].dropna(subset=["T", "E"])
        if len(s) == 0:
            continue
        kmf = KaplanMeierFitter(label=f"{grp} (n={len(s)})")
        kmf.fit(s["T"], s["E"])
        kmf.plot_survival_function(
            ax=ax, ci_show=False, color=PALETTE[grp], linewidth=1.6,
            show_censors=True,
            censor_styles={"marker": "|", "ms": 4, "mew": 0.8})
        fitters.append(kmf)

    # survival as percent
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_ylabel(f"{ylabel} (%)")
    ax.set_xlabel("Time (months)")
    ax.set_xlim(left=0)
    ax.set_title(title, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, handlelength=1.4)

    # in-plot stats: log-rank p + Cox HR (high vs low), univariate + adjusted.
    txt = []
    if logrank_p is not None:
        txt.append(f"log-rank $P$ = {logrank_p:.2g}")
    if hr_m1 is not None:
        hr, lo, hi, pv = hr_m1
        txt.append(f"HR = {hr:.2f} ({lo:.2f}-{hi:.2f}), $P$ = {pv:.2g}")
    if hr_m3 is not None:
        hr, lo, hi, pv = hr_m3
        txt.append(f"HR$_{{adj}}$ = {hr:.2f} ({lo:.2f}-{hi:.2f}), $P$ = {pv:.2g}")
    if txt:
        ax.text(0.035, 0.06, "\n".join(txt), transform=ax.transAxes,
                fontsize=7, va="bottom", ha="left")

    # number-at-risk table below the x-axis, coloured per group to match the
    # curves (red=high, grey=low). Built manually because lifelines'
    # add_at_risk_counts stacks both groups into one black tick label.
    xmax = ax.get_xlim()[1]
    ticks = [t for t in ax.get_xticks() if 0 <= t <= xmax]
    trans = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
    y_head, dy = -0.24, 0.075
    ax.text(-0.13, y_head, "No. at risk", transform=ax.transAxes,
            fontsize=7.5, fontweight="bold", ha="left", va="top")
    for i, (grp, val) in enumerate((("high", 1.0), ("low", 0.0))):
        color = PALETTE[grp]
        T = d[d["group_high"] == val].dropna(subset=["T", "E"])["T"].to_numpy()
        y = y_head - (i + 1) * dy
        ax.text(-0.13, y, grp, transform=ax.transAxes, color=color,
                fontsize=7.5, ha="left", va="top")
        for t in ticks:
            ax.text(t, y, str(int((T >= t).sum())), transform=trans,
                    color=color, fontsize=7, ha="center", va="top")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(str(path).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--grade_csv", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fig_dir", default="figures/clinical_outcome/biomarker")
    ap.add_argument("--cohort", default="TCGA-KIRC",
                    help="cohort name shown in figure titles")
    ap.add_argument("--min_days", type=float, default=30.0)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    figdir = Path(args.fig_dir); figdir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(args.pred_csv)
    clin = pd.read_csv(args.clinical, sep="\t")
    df = pred.merge(clin, on="case_submitter_id", how="inner")
    df["stage_ord"] = (df["ajcc_stage"].astype(str).str.strip().str.upper()
                       .map(STAGE_MAP))
    if args.grade_csv and Path(args.grade_csv).exists():
        grade = pd.read_csv(args.grade_csv)[["case_submitter_id", "grade_ord"]]
        df = df.merge(grade, on="case_submitter_id", how="left")
    else:
        df["grade_ord"] = np.nan
    # drop very-short follow-up (ambiguous events)
    for ep in ("OS", "DSS"):
        tcol = f"{ep}_time_days"
        if tcol in df:
            df.loc[pd.to_numeric(df[tcol], errors="coerce") < args.min_days, ep] = np.nan
    print(f"[merge] {len(pred)} pred x {len(clin)} clinical -> {len(df)} matched patients")

    predictors = [c for c in pred.columns if c.endswith(("_mean", "_q90"))]
    summary = []
    lines = []
    for pcol in predictors:
        gene = pcol.rsplit("_", 1)[0]
        vals = pd.to_numeric(df[pcol], errors="coerce")
        med = np.nanmedian(vals)
        df["group_high"] = (vals >= med).astype(float)
        n_hi = int(df["group_high"].sum()); n_lo = int((1 - df["group_high"]).sum())
        lines.append(f"\n===== {pcol}  (median={med:.3f}, high={n_hi}/low={n_lo}) =====")

        rho_stage, p_stage = spearmanr(vals, df["stage_ord"], nan_policy="omit")
        rho_grade, p_grade = spearmanr(vals, df["grade_ord"], nan_policy="omit")
        lines.append(f"  Spearman vs stage: rho={rho_stage:+.3f} p={p_stage:.4g} | "
                     f"vs grade: rho={rho_grade:+.3f} p={p_grade:.4g}")

        # ordinal severity by group (Mann-Whitney)
        for sev in ("stage_ord", "grade_ord"):
            hi = df.loc[df["group_high"] == 1, sev].dropna()
            lo = df.loc[df["group_high"] == 0, sev].dropna()
            if len(hi) > 3 and len(lo) > 3:
                u, pu = mannwhitneyu(hi, lo, alternative="two-sided")
                lines.append(f"  {sev}: high median={hi.median():.1f} vs "
                             f"low median={lo.median():.1f}  MWU p={pu:.4g}")

        # survival OS + DSS
        rec = {"predictor": pcol, "gene": gene, "n_high": n_hi, "n_low": n_lo,
               "rho_stage": rho_stage, "p_stage": p_stage,
               "rho_grade": rho_grade, "p_grade": p_grade}
        for ep in ("OS", "DSS"):
            lines.append(f"    -- {ep} --")
            d, res = cox_group(df, ep, lines)
            s = d.dropna(subset=["T", "E"])
            if s["group_high"].nunique() == 2:
                hi = s[s["group_high"] == 1]; lo = s[s["group_high"] == 0]
                lr = logrank_test(hi["T"], lo["T"], hi["E"], lo["E"])
                lines.append(f"      log-rank p={lr.p_value:.4g}")
                rec[f"{ep}_logrank_p"] = lr.p_value
                agg = "hotspot" if pcol.endswith("_q90") else "mean"
                ylab = {"OS": "Overall survival",
                        "DSS": "Disease-specific survival"}.get(ep, ep)
                title = (f"{args.cohort}  |  predicted {gene} ({agg}), "
                         f"high vs low")
                km_plot(s, title, figdir / f"km_{pcol}_{ep}.png", ylabel=ylab,
                        logrank_p=lr.p_value, hr_m1=res.get("M1"),
                        hr_m3=res.get("M3"))
            for m in ("M1", "M2", "M3"):
                if res.get(m):
                    hr, lo95, hi95, pv = res[m]
                    rec[f"{ep}_{m}_HR"] = hr
                    rec[f"{ep}_{m}_p"] = pv
        summary.append(rec)

    sdf = pd.DataFrame(summary)
    sdf.to_csv(out / "biomarker_severity_summary.csv", index=False)
    report = out / "biomarker_severity_report.txt"
    report.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[out] {out / 'biomarker_severity_summary.csv'}")
    print(f"[out] {report}")
    print(f"[figs] {figdir}")


if __name__ == "__main__":
    main()
