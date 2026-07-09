"""Relate per-section EMT-state composition (tumour-epi quadrants) to RCC clinical
metadata (M.stage / grade / Leibovich) for the labeled donors (CV1/CV2/CV6/CV9).

Quadrant fractions are parsed from the emt_2d_score.py run logs (the
`[tumour-epi quadrants]` block); metadata from the RCC metadata.csv. Donor =
section name prefix (CVx). Saves a merged CSV + a 2-panel figure.

CAVEAT (do not overclaim): only 4 donors are labeled, only CV2 is M1, and three of
the four are grade-4 — so this is descriptive, not a powered test. The hybrid score
also depends on the per-section control-gene baseline, a technical confound.
"""
import argparse, glob, re, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

QCOL = {"epithelial": "#1f77b4", "hybrid": "#FFB300",
        "mesenchymal": "#2ca02c", "low-signal": "#cccccc"}
QORDER = ["epithelial", "hybrid", "mesenchymal", "low-signal"]
QLAB = {"epithelial": "epithelial (no EMT)", "hybrid": "hybrid (partial EMT)",
        "mesenchymal": "mesenchymal (full EMT)", "low-signal": "low-signal (E−/M−)"}
META_CSV = ("/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/"
            "sanger_renal_cell_carcinoma/metadata.csv")


# intra-tumour EMT-heterogeneity (dispersion) metrics emitted by emt_2d_score.py
# under the "[tumour-epi dispersion]" block. Translation-invariant spreads (all but
# quad_entropy) are robust to the per-section control-gene baseline confound that
# distorts the quadrant proportions.
DISP_KEYS = ["std_E", "std_M", "total_var", "gen_var",
             "emt_axis_std", "emt_axis_iqr", "quad_entropy"]


def parse_logs(log_glob):
    rows = {}
    for f in glob.glob(log_glob):
        t = Path(f).read_text()
        s = re.search(r"section=(\S+)", t)
        comp = re.search(r"tumour-epi ([\d,]+)\s+nephron-epi", t)
        if not s:
            continue
        q = {}
        for k in QORDER:
            m = re.search(rf"^\s+{re.escape(k)}\s+([\d,]+)\s+\(([\d.]+)%\)", t, re.M)
            if m:
                q[k] = float(m.group(2))
        if len(q) == 4:
            ntum = int(comp.group(1).replace(",", "")) if comp else 0
            row = {"n_tum": ntum, **{f"{k}_pct": q[k] for k in QORDER}}
            # dispersion metrics (optional; older logs predate this block)
            for dk in DISP_KEYS:
                dm = re.search(rf"^\s+{re.escape(dk)}\s+([-\d.]+)", t, re.M)
                if dm:
                    row[dk] = float(dm.group(1))
            rows[s.group(1)] = row
    return rows


def load_meta():
    meta = {}
    with open(META_CSV) as f:
        for r in csv.DictReader(f):
            if r["Patient"].strip():
                meta[r["Patient"].strip()] = {
                    "M": int(r["M.stage"]), "T": r["T.stage"].strip(),
                    "grade": int(r["grade"]), "leibovich": int(r["Leibovich"]),
                    "sex": r["sex"].strip(), "age": int(r["age"])}
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_glob", default="logs/job_logs/emt_2d.77557.*.out")
    ap.add_argument("--outdir", default="figures/cross_tissue_retrieval/emt_2d")
    ap.add_argument("--tag", default="union",
                    help="output filename suffix / gene-set label (e.g. union, emt_program)")
    args = ap.parse_args()

    quad = parse_logs(args.log_glob)
    meta = load_meta()
    recs = []
    for sec, q in sorted(quad.items()):
        donor = sec.split("-")[0]
        if donor not in meta:
            continue
        recs.append({"section": sec, "donor": donor, **q, **meta[donor]})
    if not recs:
        raise SystemExit("no sections matched donors in metadata")
    print(f"[data] {len(recs)} sections across donors {sorted({r['donor'] for r in recs})}")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    have_disp = all(dk in recs[0] for dk in DISP_KEYS)
    # merged CSV
    cols = ["section", "donor", "n_tum"] + [f"{k}_pct" for k in QORDER] + \
           (DISP_KEYS if have_disp else []) + \
           ["M", "T", "grade", "leibovich", "sex", "age"]
    merged_csv = outdir / f"emt_2d_metadata_merged_{args.tag}.csv"
    with open(merged_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in recs:
            w.writerow({c: r[c] for c in cols})
    print(f"[saved] {merged_csv}")

    # order donors by Leibovich (low->high), then group sections
    donors = sorted({r["donor"] for r in recs},
                    key=lambda d: (meta[d]["leibovich"], d))
    DCOL = {"CV1": "#4477AA", "CV2": "#EE6677", "CV6": "#228833", "CV9": "#AA3377"}

    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0], wspace=0.22)

    # ---- Panel A: per-section stacked EMT-state composition, grouped by donor ----
    axA = fig.add_subplot(gs[0, 0])
    x = 0.0; xticks = []; xticklabels = []; group_centers = {}
    for d in donors:
        secs = sorted([r for r in recs if r["donor"] == d], key=lambda r: r["section"])
        start = x
        for r in secs:
            bottom = 0.0
            for k in QORDER:
                h = r[f"{k}_pct"]
                axA.bar(x, h, bottom=bottom, width=0.8, color=QCOL[k],
                        edgecolor="white", linewidth=0.3)
                bottom += h
            xticks.append(x)
            xticklabels.append(r["section"].replace(f"{d}-KID-0-", "").replace("-", ""))
            x += 1.0
        group_centers[d] = (start + x - 1.0) / 2.0
        x += 0.8  # gap between donor groups
    axA.set_xticks(xticks); axA.set_xticklabels(xticklabels, rotation=60, ha="right", fontsize=6)
    axA.set_ylabel("tumour-epithelium cells (%)", fontsize=10)
    axA.set_ylim(0, 100)
    axA.set_title("EMT-state composition per section, grouped by donor "
                  "(donors ordered by Leibovich score)", fontsize=10)
    # donor metadata header band
    for d in donors:
        m = meta[d]
        mtxt = "M1" if m["M"] else "M0"
        col = "#CC3311" if m["M"] else "#444444"
        axA.text(group_centers[d], 104,
                 f"{d}\n{mtxt} · G{m['grade']} · Leib{m['leibovich']}",
                 ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=col)
    axA.legend(handles=[Line2D([0], [0], marker="s", linestyle="", markersize=8,
                               markerfacecolor=QCOL[k], markeredgecolor="white", label=QLAB[k])
                        for k in QORDER], fontsize=7, loc="lower right", framealpha=0.95)
    axA.set_ylim(0, 112)

    # ---- Panel B: EMT-active fraction (hybrid + mesenchymal) vs Leibovich ----
    axB = fig.add_subplot(gs[0, 1])
    for r in recs:
        emt_active = r["hybrid_pct"] + r["mesenchymal_pct"]
        m = meta[r["donor"]]
        jit = (hash(r["section"]) % 7 - 3) * 0.06  # deterministic x-jitter by donor
        mk = "*" if m["M"] else "o"
        sz = 230 if m["M"] else 80
        axB.scatter(m["leibovich"] + jit, emt_active, marker=mk, s=sz,
                    c=DCOL.get(r["donor"], "#777777"), edgecolor="k", linewidth=0.5,
                    alpha=0.9, zorder=3)
    # donor means
    for d in donors:
        rs = [r for r in recs if r["donor"] == d]
        ym = np.mean([r["hybrid_pct"] + r["mesenchymal_pct"] for r in rs])
        axB.scatter(meta[d]["leibovich"], ym, marker="_", s=900,
                    c=DCOL.get(d, "#777777"), linewidth=2.5, zorder=2)
    axB.set_xlabel("Leibovich score (higher = worse prognosis)", fontsize=10)
    axB.set_ylabel("EMT-active tumour-epi (%)\nhybrid + mesenchymal", fontsize=10)
    axB.set_title("EMT-active fraction vs clinical risk", fontsize=10)
    axB.grid(axis="y", ls=":", alpha=0.5)
    handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor=DCOL[d],
                      markeredgecolor="k", markersize=8,
                      label=f"{d} ({'M1' if meta[d]['M'] else 'M0'}, G{meta[d]['grade']})")
               for d in donors]
    handles.append(Line2D([0], [0], marker="*", linestyle="", markerfacecolor="#888888",
                          markeredgecolor="k", markersize=14, label="★ = M1 (metastatic)"))
    handles.append(Line2D([0], [0], marker="_", linestyle="", markeredgecolor="#888888",
                          markersize=14, markeredgewidth=2.5, label="— = donor mean"))
    axB.legend(handles=handles, fontsize=7, loc="best", framealpha=0.95)

    fig.suptitle(f"RCC EMT-state composition vs clinical metastasis / grade / Leibovich  "
                 f"[{args.tag} gene set]  (n={len(recs)} sections, {len(donors)} labeled donors)",
                 fontsize=12, fontweight="bold")
    foot = ("Quadrants from the tumour-epi-restricted 2D E/M score (control-gene baseline, cohort-pooled anchors). "
            "CAVEAT: only CV2 is M1 and 3/4 donors are grade-4 → descriptive, not a powered test; "
            "hybrid fraction also depends on the per-section control-gene baseline (technical confound).")
    fig.text(0.5, 0.005, foot, ha="center", va="bottom", fontsize=8, color="#444444")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    stem = outdir / f"emt_2d_metadata_correlation_{args.tag}"
    fig.savefig(f"{stem}.png", dpi=160, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    print(f"[saved] {stem}.png / .pdf")

    # ================= dispersion (intra-tumour EMT heterogeneity) =================
    # Pivot from "EMT proportion" (above) to "how SPREAD the tumour-epi cloud is in
    # (E,M) space". The variance-based metrics are translation-invariant, so unlike
    # the proportion they are NOT distorted by the per-section baseline shift.
    if not have_disp:
        print("[disp] no dispersion metrics in logs (re-run emt_2d_score.py); "
              "skipping dispersion figure")
        return
    from scipy.stats import spearmanr
    DLAB = {"std_E": "std E", "std_M": "std M",
            "total_var": "total variance (var_E+var_M)",
            "gen_var": "generalised spread √|cov|",
            "emt_axis_std": "EMT-axis spread (std)",
            "emt_axis_iqr": "EMT-axis spread (IQR)",
            "quad_entropy": "state entropy (bits)"}
    show = ["emt_axis_std", "total_var", "gen_var", "std_E", "std_M", "quad_entropy"]
    # donor-mean table for the donor-level rank correlation (severity is donor-level)
    donor_leib = {d: meta[d]["leibovich"] for d in donors}
    print("[disp] Spearman ρ vs Leibovich (higher Leibovich = worse prognosis):")
    fig2, axs = plt.subplots(2, 3, figsize=(15, 9))
    for ax, dk in zip(axs.ravel(), show):
        # section-level points
        xs = [meta[r["donor"]]["leibovich"] for r in recs]
        ys = [r[dk] for r in recs]
        rho_s, p_s = spearmanr(xs, ys)
        # donor-mean points
        dmean = {d: np.mean([r[dk] for r in recs if r["donor"] == d]) for d in donors}
        xs_d = [donor_leib[d] for d in donors]
        ys_d = [dmean[d] for d in donors]
        rho_d, p_d = spearmanr(xs_d, ys_d)
        for r in recs:
            m = meta[r["donor"]]
            jit = (hash(r["section"]) % 7 - 3) * 0.06
            axB_mk = "*" if m["M"] else "o"
            sz = 230 if m["M"] else 70
            ax.scatter(m["leibovich"] + jit, r[dk], marker=axB_mk, s=sz,
                       c=DCOL.get(r["donor"], "#777777"), edgecolor="k",
                       linewidth=0.5, alpha=0.9, zorder=3)
        for d in donors:
            ax.scatter(donor_leib[d], dmean[d], marker="_", s=900,
                       c=DCOL.get(d, "#777777"), linewidth=2.5, zorder=2)
        ax.set_title(f"{DLAB[dk]}\nsection ρ={rho_s:+.2f} (p={p_s:.2f})  ·  "
                     f"donor ρ={rho_d:+.2f}", fontsize=9)
        ax.set_xlabel("Leibovich score", fontsize=9)
        ax.grid(axis="y", ls=":", alpha=0.5)
        print(f"   {dk:<13} section ρ={rho_s:+.3f} (p={p_s:.3f})  donor ρ={rho_d:+.3f}")
    handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor=DCOL[d],
                      markeredgecolor="k", markersize=8,
                      label=f"{d} ({'M1' if meta[d]['M'] else 'M0'}, G{meta[d]['grade']}, Leib{meta[d]['leibovich']})")
               for d in donors]
    handles.append(Line2D([0], [0], marker="*", linestyle="", markerfacecolor="#888888",
                          markeredgecolor="k", markersize=14, label="★ = M1 (metastatic)"))
    fig2.legend(handles=handles, fontsize=8, loc="lower center", ncol=len(donors) + 1,
                framealpha=0.95, bbox_to_anchor=(0.5, -0.02))
    fig2.suptitle("Intra-tumour EMT heterogeneity (dispersion) vs clinical risk  "
                  f"[{args.tag}]  (n={len(recs)} sections, {len(donors)} donors — "
                  "DESCRIPTIVE, n=4 donors / 1 M1, NOT powered)",
                  fontsize=12, fontweight="bold")
    foot2 = ("Dispersion = spread of tumour-epi cells in (E,M) control-gene-baseline space. "
             "Variance/spread metrics are translation-invariant → robust to the per-section "
             "baseline confound that distorts the EMT proportion. EMT-axis spread = std of the "
             "projection onto the cohort-pooled E-anchor→M-anchor direction. ρ = Spearman; "
             "donor ρ over 4 donor means (no p — underpowered).")
    fig2.text(0.5, 0.005, foot2, ha="center", va="bottom", fontsize=8, color="#444444")
    fig2.tight_layout(rect=[0, 0.06, 1, 0.95])
    stem2 = outdir / f"emt_2d_dispersion_correlation_{args.tag}"
    fig2.savefig(f"{stem2}.png", dpi=160, bbox_inches="tight")
    fig2.savefig(f"{stem2}.pdf", bbox_inches="tight")
    print(f"[saved] {stem2}.png / .pdf")


if __name__ == "__main__":
    main()
