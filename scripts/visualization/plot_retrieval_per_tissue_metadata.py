"""Figure: H&E -> ST retrieval, per-tissue breakdown (metadata-driven).

Revisit of plot_retrieval_per_tissue_mrr.py that fixes two flaws of the old
split('-')[1] tissue parser:
  * Tissue labels now come from sample_metadata.csv (tissue_type), so public
    HEST-1k sections and odd IDs no longer fall into a '?' bucket.
  * SIGMMA + baselines are trained with a WITHIN-PANEL contrastive loss over
    ALL Sanger panels, so every in-house targeted panel (5K, hOvary, hPancreas,
    hLung, hBrain, ...) is a valid retrieval cohort -- not just the 5K panel.
    The only genuine OOD held-out are the external HEST-1k PUBLIC sections
    (IDC/PAAD/LUAD/SKCM), which have no Sanger panel entry; we exclude those.

Skin cohorts (beacon + CTCL) are MERGED into a single "Skin" group.
Only tissues with >= MIN_N in-house test sections are shown (default 4):
currently Skin / Ovary / Kidney (RCC) / Pancreas / Liver.

2 rows (R@1%, MRR) x N tissue cols. Image direction (H&E -> ST).
Same eva_critique bar-plot conventions + paired-t significance brackets.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_rel

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
DATA_JSON = ROOT / "outputs/retrieval/normalized_metrics_step2000.json"
META_CSV = Path("/nfs/users/nfs_d/dj16/lustre_dabin/data/sanger_in_house_xenium/"
                "sample_metadata.csv")
PANEL_CSV = Path("/lustre/scratch126/cellgen/lotfollahi/dj16/data/"
                 "sanger_in_house_xenium/gene_panel_info/section_to_gene_panel.csv")
FIVE_K = "Xenium Human 5K Pan Tissue & Pathways Panel"

CONDITIONS = [
    ("bleep",   "BLEEP-ft",   "#882255"),
    ("clip",    "CLIP-ft",    "#44AA99"),
    ("plip",    "PLIP-ft",    "#DDCC77"),
    ("omiclip", "OmiCLIP-ft", "#999933"),
    ("a01",     "SIGMMA",     "#d62728"),
]
COND_KEYS = [c[0] for c in CONDITIONS]

DIRECTION = "image"  # H&E -> ST
MIN_N = 2

# tissue_type -> display group. beacon + CTCL collapse to a single "skin".
GROUP_OF = {
    "beacon": "skin",
    "CTCL": "skin",
    "pediatric_ovary": "pediatric_ovary",
    "renal_cell_carcinoma": "renal_cell_carcinoma",
    "pancreas": "pancreas",
    "liver": "liver",
    "lung": "lung",
    "whole_embryo": "whole_embryo",
    "adrenocortical_carcinoma": "adrenocortical_carcinoma",
    "glioblastoma": "glioblastoma",
}
GROUP_LABEL = {
    "skin": "Skin",
    "pediatric_ovary": "Ovary",
    "renal_cell_carcinoma": "Kidney (RCC)",
    "pancreas": "Pancreas",
    "liver": "Liver",
    "lung": "Lung",
    "whole_embryo": "Whole embryo",
    "adrenocortical_carcinoma": "Adrenocortical",
    "glioblastoma": "Glioblastoma",
}
# External HEST-1k public tissue_types: no Sanger panel -> genuine OOD, excluded.
PUBLIC_TT = {"IDC", "PAAD", "LUAD", "SKCM"}


def sig_marker(p):
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def paired_t(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2:
        return float("nan")
    return float(ttest_rel(a, b, alternative="two-sided").pvalue)


def plot_panel(ax, vals_per_cond, ylabel, ymin=0.0, show_ylabel=True, tag=""):
    means = np.array([np.mean(v) for v in vals_per_cond])
    stds = np.array([np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in vals_per_cond])
    colors = [c[2] for c in CONDITIONS]
    x = np.arange(len(CONDITIONS))
    sigmma_idx = len(CONDITIONS) - 1

    err_lo = np.minimum(stds, means - ymin)
    yerr = np.vstack([err_lo, stds])
    bars = ax.bar(
        x, means, yerr=yerr, color=colors,
        edgecolor="black", linewidth=0.5,
        error_kw=dict(elinewidth=0.6, capsize=1.8, ecolor="black"),
        width=0.72,
    )
    bars[sigmma_idx].set_linewidth(1.0)

    best_idx = max(range(len(COND_KEYS) - 1), key=lambda i: means[i])
    ax.axhline(means[best_idx], color="#888888", linestyle="--",
               linewidth=0.5, alpha=0.7, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in CONDITIONS],
                       rotation=40, ha="right", fontsize=6)
    if show_ylabel:
        ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="-", linewidth=0.25,
            color="#cccccc", alpha=0.7, zorder=-1)
    ax.set_axisbelow(True)

    # paired-t brackets, stacked. ns / n/a are NOT drawn; only significant
    # comparisons get a bracket, stacked compactly.
    top = (means + stds).max()
    rng = (top - ymin) if (top - ymin) > 0 else 0.05
    n_baseline = len(COND_KEYS) - 1
    bracket_step = rng * 0.085
    sigmma_vals = vals_per_cond[-1]
    sig_slots = []
    for slot in range(n_baseline):
        p = paired_t(sigmma_vals, vals_per_cond[slot])
        mk = sig_marker(p)
        print(f"  [{tag}] SIGMMA vs {COND_KEYS[slot]:<8s} "
              f"n={len(sigmma_vals)} paired-t p={p:.2e} {mk}")
        if mk not in ("ns", "n/a"):
            sig_slots.append((slot, mk))
    ymax = top + bracket_step * (len(sig_slots) + 0.8)
    ax.set_ylim(ymin, ymax)
    yrange = ymax - ymin
    tick_h = yrange * 0.006
    base_y = top + bracket_step * 0.45
    for stack, (slot, mk) in enumerate(sig_slots):
        y_b = base_y + bracket_step * stack
        x_l, x_r = slot, sigmma_idx
        ax.plot([x_l, x_l, x_r, x_r],
                [y_b, y_b + tick_h, y_b + tick_h, y_b],
                color="black", linewidth=0.35, clip_on=False)
        ax.text((x_l + x_r) / 2, y_b + tick_h - yrange * 0.002,
                mk, ha="center", va="bottom",
                fontsize=5.0, fontweight="bold", color="black", clip_on=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_json", default=str(DATA_JSON))
    ap.add_argument("--out_dir", default=str(ROOT / "figures/retrieval/all_gene_panel"))
    ap.add_argument("--min_n", type=int, default=MIN_N)
    ap.add_argument("--direction", choices=["image", "spatial"], default="image",
                    help="image = H&E->ST, spatial = ST->H&E")
    args = ap.parse_args()
    direction = args.direction
    dir_title = (r"H&E $\rightarrow$ ST" if direction == "image"
                 else r"ST $\rightarrow$ H&E")
    dir_suffix = "_he2st" if direction == "image" else "_st2he"

    meta = pd.read_csv(META_CSV)
    panel = pd.read_csv(PANEL_CSV)
    tt = dict(zip(meta["sample_id"], meta["tissue_type"]))
    pn = dict(zip(panel["sample_name"], panel["panel_name"]))

    with open(args.data_json) as f:
        d = json.load(f)
    all_secs = list(d["a01"][direction]["per_section"].keys())

    # in-house test sections grouped by display group. Within-panel contrastive
    # training makes every Sanger panel valid; exclude only external HEST-1k
    # public sections (no panel-map entry / known public tissue_type).
    by_group = {}
    for s in all_secs:
        t = tt.get(s)
        if t in PUBLIC_TT or pn.get(s) is None:   # external HEST-1k -> OOD
            continue
        grp = GROUP_OF.get(t)
        if grp is None:
            continue
        by_group.setdefault(grp, []).append(s)

    groups = [(g, secs) for g, secs in by_group.items() if len(secs) >= args.min_n]
    groups.sort(key=lambda x: -len(x[1]))
    print("In-house test sections per shown group (>= "
          f"{args.min_n}):")
    for g, secs in groups:
        print(f"  {GROUP_LABEL[g]:14s} n={len(secs)}")
    dropped = {g: len(s) for g, s in by_group.items() if len(s) < args.min_n}
    if dropped:
        print(f"  (dropped n<{args.min_n}: "
              + ", ".join(f"{GROUP_LABEL[g]}={n}" for g, n in dropped.items()) + ")")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 9, "axes.labelsize": 7,
        "axes.linewidth": 0.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2, "legend.fontsize": 6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    metrics = [("recall_at_1pct", "Recall @ 1%"),
               ("mrr",            "Mean Reciprocal Rank")]
    ncol = len(groups)
    fig, axes = plt.subplots(2, ncol, figsize=(2.0 * ncol + 0.5, 5.0),
                             gridspec_kw={"wspace": 0.30, "hspace": 0.70},
                             squeeze=False)

    for row, (mkey, mlbl) in enumerate(metrics):
        for col, (grp, secs) in enumerate(groups):
            ax = axes[row, col]
            vals_per_cond = [
                np.array([d[k][direction]["per_section"][s][mkey] for s in secs])
                for k in COND_KEYS
            ]
            plot_panel(ax, vals_per_cond, mlbl, ymin=0.0,
                       show_ylabel=(col == 0), tag=f"{grp} {mkey}")
            if row == 0:
                ax.set_title(f"{GROUP_LABEL[grp]}\n(n={len(secs)})",
                             fontsize=8, pad=4)

    letters = "abcdefghijkl"
    for i, ax in enumerate(axes.ravel()):
        ax.text(-0.20, 1.06, letters[i], transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom", ha="left")

    fig.suptitle(
        dir_title + " retrieval, per-tissue "
        r"(all in-house panels, within-panel trained)",
        fontsize=10, fontweight="bold", y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(bottom=0.10, top=0.90, hspace=0.85, wspace=0.30)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"retrieval_per_tissue_allpanel{dir_suffix}.png"
    out_pdf = out_dir / f"retrieval_per_tissue_allpanel{dir_suffix}.pdf"
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_pdf}")


if __name__ == "__main__":
    main()
