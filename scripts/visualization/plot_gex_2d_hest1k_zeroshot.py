"""Figure: HEST-1k public zero-shot GEX-2D — SIGMMA (36852) vs baselines.

External-OOD validation: the in-house svg0529 main-panel probe heads are applied
WITHOUT retraining to HEST-1k public test sections (TENX*/NCBI*), evaluated on the
genes shared between the in-house main panel and each HEST section's measured
var.index (strict zero-shot, no expression-based leakage).

SECTION-LEVEL boxplot, identical unit to the in-house test figure
(``plot_gex_2d_all_gene_panel_test.py``): the two evaluable HEST panels are POOLED
into one HEST-1k cohort so the box has enough sections to be meaningful and a
section-paired t-test is possible.
  * 80b4f72e3da8  SKCM  (15 genes)  TENX115, TENX117
  * 2bbc576266c5  IDC   (12 genes)  NCBI783, NCBI785, TENX95, TENX99
  -> n = 6 sections.  PAAD/LUAD 11bbd (7 genes) excluded.

Box = per-section value of each metric (each section's own shared-gene summary),
paired across methods by section. SIGMMA shown at headline repr = ``uni`` (the
SIGMMA-LoRA-finetuned UNI features), matching the in-house figure.

Metrics (project GEX-2D convention, all section-level):
  * median per-gene Pearson         — higher better
  * MSE per slide, standardized     — lower better  (headline signal)
  * SSIM per slide                  — higher better

Paired t-test (n = 6 sections) SIGMMA vs each baseline; n is small so treat
significance as suggestive. Read from
``outputs/gex_2d/public_zeroshot_strict_step2000/<panel>/``.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
RES = ROOT / "outputs/gex_2d/public_zeroshot_strict_step2000"
OUT_DIR = ROOT / "figures/gex_2d/hest1k_zeroshot"

# HEST-1k panels pooled into one cohort (>= MIN_GENES shared genes each).
#   11bbd69265ed PAAD/LUAD = 7 shared  -> excluded
MIN_GENES = 10
PANELS = [
    ("80b4f72e3da8", "SKCM"),
    ("2bbc576266c5", "IDC"),
]

# (label, color, json-stem). Regressors at far LEFT (distinct method class),
# then UNI2-h backbone + 4 contrastive FT, then SIGMMA. Colors match the in-house
# gex_2d figures: regressors = Paul Tol blues, rest = project method palette.
CONDS = [
    ("ST-Net",    "#332288", "stnet_bin"),
    ("HisToGene", "#88CCEE", "histogene_bin"),
    ("DeepPT",    "#4477AA", "deeppt_bin"),
    ("UNI2-h",    "#888888", "uni2h_no_ft"),
    ("BLEEP",     "#882255", "bleep_uni_step2000"),
    ("CLIP",      "#44AA99", "clip_uni_step2000"),
    ("PLIP",      "#DDCC77", "plip_uni_step2000"),
    ("OmiCLIP",   "#999933", "omiclip_uni_step2000"),
    ("SIGMMA",    "#d62728", "sigmma_a01_step2000__uni"),
]
CLUSTER_LABELS = {"UNI2-h", "BLEEP", "CLIP", "PLIP", "OmiCLIP"}
REGRESSOR_LABELS = {"ST-Net", "HisToGene", "DeepPT"}
UNI_LABEL = "UNI2-h"
SIGMMA_LABEL = "SIGMMA"

METRICS = [
    ("median_per_gene_pearson",            r"Per-gene Pearson  $\uparrow$",     "a"),
    ("median_mse_per_slide_standardized",  r"MSE per slide (std) $\downarrow$", "b"),
    ("median_ssim_per_slide",              r"SSIM per slide  $\uparrow$",       "c"),
]


def load(panel: str, stem: str) -> dict:
    p = RES / panel / f"{stem}.json"
    return json.load(open(p)) if p.exists() else {}


def section_values(stem: str, metric: str) -> dict[str, float]:
    """Pool per-section values for `metric` across both panels -> {section: val}."""
    out = {}
    for h, _ in PANELS:
        blk = load(h, stem).get("per_slide", {}).get(metric)
        if isinstance(blk, dict) and isinstance(blk.get("values"), dict):
            for sec, v in blk["values"].items():
                if v is not None and np.isfinite(v):
                    out[sec] = float(v)
    return out


def collect(metric: str) -> tuple[list[str], np.ndarray]:
    """Return (section_order, arr [n_methods, n_sections]) — paired by section."""
    per_method = [section_values(stem, metric) for _, _, stem in CONDS]
    sections = sorted(set.intersection(*[set(d) for d in per_method]))
    arr = np.array([[d[s] for s in sections] for d in per_method])
    return sections, arr


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


def whisker_top(v):
    q1, q3 = np.percentile(v, [25, 75]); iqr = q3 - q1
    return float(v[v <= q3 + 1.5 * iqr].max())


def whisker_bot(v):
    q1, q3 = np.percentile(v, [25, 75]); iqr = q3 - q1
    return float(v[v >= q1 - 1.5 * iqr].min())


def plot_panel(ax, arr, ylabel, panel_letter, lower_is_better):
    labels = [c[0] for c in CONDS]
    colors = [c[1] for c in CONDS]
    n_methods = arr.shape[0]
    x = np.arange(n_methods)
    uni_idx = labels.index(UNI_LABEL)
    sigmma_idx = labels.index(SIGMMA_LABEL)
    cluster_idx = sorted([i for i, l in enumerate(labels) if l in CLUSTER_LABELS],
                         reverse=True)
    regressor_idx = sorted([i for i, l in enumerate(labels) if l in REGRESSOR_LABELS],
                           reverse=True)

    data = [arr[i] for i in range(n_methods)]
    bp = ax.boxplot(
        data, positions=x, widths=0.6, patch_artist=True,
        showcaps=True, showfliers=False, whis=1.5,
        boxprops=dict(linewidth=0.6),
        medianprops=dict(color="black", linewidth=1.0),
        whiskerprops=dict(linewidth=0.6, color="black"),
        capprops=dict(linewidth=0.6, color="black"),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i])
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2 if i == sigmma_idx else 0.6)
        patch.set_alpha(0.9)
    # individual sections as dots (deterministic jitter, no RNG)
    for xi in range(n_methods):
        dv = arr[xi]
        jit = np.linspace(-0.14, 0.14, len(dv)) if len(dv) > 1 else np.zeros(len(dv))
        ax.scatter(xi + jit, dv, s=7, color="black", zorder=3, linewidths=0, alpha=0.3)

    ax.axhline(float(np.median(arr[uni_idx])), color="#888888", linestyle="--",
               linewidth=0.6, alpha=0.7, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="-", linewidth=0.3, color="#cccccc",
            alpha=0.7, zorder=-1)
    ax.set_axisbelow(True)

    all_top = max(whisker_top(arr[i]) for i in range(n_methods))
    all_bot = min(whisker_bot(arr[i]) for i in range(n_methods))
    cluster_top = max(whisker_top(arr[i]) for i in cluster_idx + [sigmma_idx])

    ref_range = all_top - (0.0 if lower_is_better else all_bot)
    bracket_step = ref_range * 0.05
    tick_h = ref_range * 0.006

    sig_vals = arr[sigmma_idx]

    # only draw SIGNIFICANT brackets (skip ns / n-a); stack without gaps
    def sig_only(idx_list):
        return [bi for bi in idx_list
                if sig_marker(paired_t(sig_vals, arr[bi])) not in ("ns", "n/a")]

    cluster_sig = sig_only(cluster_idx)
    regressor_sig = sig_only(regressor_idx)

    t1_base = cluster_top + bracket_step * 0.8
    t1_top = t1_base + bracket_step * max(len(cluster_sig) - 1, 0) + tick_h
    t2_base = max(all_top, t1_top) + bracket_step * 0.8
    t2_top = t2_base + bracket_step * max(len(regressor_sig) - 1, 0) + tick_h

    ymin = 0.0 if lower_is_better else all_bot - ref_range * 0.05
    ymax = max(all_top, t2_top) + ref_range * 0.06
    ax.set_ylim(ymin, ymax)

    def draw_brackets(idx_list, base):
        for slot, bi in enumerate(idx_list):
            mk = sig_marker(paired_t(sig_vals, arr[bi]))
            y_b = base + bracket_step * slot
            x_l, x_r = sorted([bi, sigmma_idx])
            ax.plot([x_l, x_l, x_r, x_r],
                    [y_b, y_b + tick_h, y_b + tick_h, y_b],
                    color="black", linewidth=0.4, clip_on=False)
            ax.text((x_l + x_r) / 2, y_b + tick_h - ref_range * 0.002,
                    mk, ha="center", va="bottom",
                    fontsize=5.5, fontweight="bold", color="black", clip_on=False)

    draw_brackets(cluster_sig, t1_base)
    draw_brackets(regressor_sig, t2_base)

    ax.text(-0.18, 1.02, panel_letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "axes.linewidth": 0.6, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(8.6, 4.6),
                             gridspec_kw={"wspace": 0.42})
    n_sections = None
    for ax, (metric, ylabel, letter) in zip(axes, METRICS):
        sections, arr = collect(metric)
        n_sections = len(sections)
        plot_panel(ax, arr, ylabel, letter, lower_is_better="mse" in metric.lower())

    fig.suptitle(r"H&E $\rightarrow$ Gene expression prediction : HEST-1k public zero-shot",
                 fontsize=10, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(bottom=0.30)
    fig.text(
        0.5, 0.085,
        f"Box: median + Q1-Q3, 1.5 x IQR whiskers; dots = sections "
        f"(n = {n_sections}: SKCM + IDC pooled); dashed = UNI2-h median. "
        "ST-Net / HisToGene / DeepPT = K=14 dense regressors.",
        ha="center", fontsize=6.5, color="black",
    )
    fig.text(
        0.5, 0.04,
        f"Paired t-test (n = {n_sections} sections) SIGMMA vs every baseline:  "
        "* P ≤ 0.05,  ** P ≤ 0.01,  *** P ≤ 0.001  "
        "(upper brackets = regressors, lower = backbone/contrastive).",
        ha="center", fontsize=6.5, color="black",
    )

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"hest1k_zeroshot_box.{ext}",
                    dpi=600 if ext == "png" else None)
    print(f"wrote {OUT_DIR}/hest1k_zeroshot_box.png / .pdf")

    print(f"\n=== HEST-1k zero-shot section-level median [Q1, Q3] (n={n_sections}) ===")
    for metric, ylabel, _ in METRICS:
        sections, arr = collect(metric)
        labels = [c[0] for c in CONDS]
        sig_vals = arr[labels.index(SIGMMA_LABEL)]
        print(f"\n  {metric}")
        for i, (label, _, _) in enumerate(CONDS):
            q1, q3 = np.percentile(arr[i], [25, 75])
            mk = "" if label == SIGMMA_LABEL else sig_marker(paired_t(sig_vals, arr[i]))
            print(f"    {label:<10} {np.median(arr[i]):.3f} [{q1:.3f}, {q3:.3f}]   {mk}")


if __name__ == "__main__":
    main()
