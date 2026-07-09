"""Report 1/3: GEX-2D benchmark on ALL test sections pooled (in-house + HEST-1k).

Single pooled cohort = 33 in-house svg0529 test sections + 6 HEST-1k public
zero-shot sections (SKCM 2 + IDC 4) = n=39. Section-level boxplot, paired t-test
SIGMMA vs each baseline (ns brackets omitted).

CAVEAT: in-house per-section metrics are over the FULL svg0529 gene panel; HEST
per-section metrics are over each public section's 12-15 SHARED genes (strict
zero-shot). The pooled view mixes these regimes — see the per-tissue reports
(plot_gex_2d_all_gene_panel_test_per_tissue.py / plot_gex_2d_hest1k_per_tissue.py)
for the disaggregated, like-for-like picture.

SIGMMA at headline repr=uni, step2000. Tissue/section provenance is irrelevant
here (pooled) but section keys are taken straight from each result's per_slide.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
PROBE = ROOT / "outputs/gex_2d/svg0529_step2000_test"
REG = ROOT / "outputs/gex_2d/svg0529_panel_35a7bd78801c_test"
HEST = ROOT / "outputs/gex_2d/public_zeroshot_strict_step2000"
HEST_PANELS = ["80b4f72e3da8", "2bbc576266c5"]  # SKCM, IDC (>=10 shared genes)
OUT_PNG = ROOT / "figures/gex_2d/all_gene_panel/test_boxplot_all.png"
OUT_PDF = ROOT / "figures/gex_2d/all_gene_panel/test_boxplot_all.pdf"

# (label, color, in-house result json, HEST result stem). Regressors far-left.
CONDITIONS = [
    ("ST-Net",     "#332288", REG / "stnet_svg0529_cache/stnet_bin.json",          "stnet_bin"),
    ("HisToGene",  "#88CCEE", REG / "histogene_svg0529_cache/histogene_bin.json",  "histogene_bin"),
    ("DeepPT",     "#4477AA", REG / "deeppt_svg0529_cache/deeppt_bin.json",        "deeppt_bin"),
    ("UNI2-h",     "#888888", PROBE / "uni2h_no_ft/results_linear_sweep/uni.json", "uni2h_no_ft"),
    ("BLEEP-ft",   "#882255", PROBE / "bleep_uni_step2000/results_linear_sweep/uni.json",   "bleep_uni_step2000"),
    ("CLIP-ft",    "#44AA99", PROBE / "clip_uni_step2000/results_linear_sweep/uni.json",    "clip_uni_step2000"),
    ("PLIP-ft",    "#DDCC77", PROBE / "plip_uni_step2000/results_linear_sweep/uni.json",    "plip_uni_step2000"),
    ("OmiCLIP-ft", "#999933", PROBE / "omiclip_uni_step2000/results_linear_sweep/uni.json", "omiclip_uni_step2000"),
    ("SIGMMA",     "#d62728", PROBE / "sigmma_a01_step2000/results/uni.json",       "sigmma_a01_step2000__uni"),
]
CLUSTER_LABELS = {"UNI2-h", "BLEEP-ft", "CLIP-ft", "PLIP-ft", "OmiCLIP-ft"}
REGRESSOR_LABELS = {"ST-Net", "HisToGene", "DeepPT"}
UNI_LABEL = "UNI2-h"
SIGMMA_LABEL = "SIGMMA"

METRICS = [
    ("median_per_gene_pearson",            r"Per-gene Pearson  $\uparrow$",     "a"),
    ("median_mse_per_slide_standardized",  r"MSE per slide (std) $\downarrow$", "b"),
    ("median_ssim_per_slide",              r"SSIM per slide  $\uparrow$",       "c"),
]


def section_values(inhouse_path: Path, hest_stem: str, metric: str) -> dict[str, float]:
    out = {}
    blk = json.load(open(inhouse_path))["per_slide"].get(metric, {})
    for s, v in (blk.get("values") or {}).items():
        if v is not None and np.isfinite(v):
            out[s] = float(v)
    for h in HEST_PANELS:
        d = json.load(open(HEST / h / f"{hest_stem}.json"))
        for s, v in (d["per_slide"].get(metric, {}).get("values") or {}).items():
            if v is not None and np.isfinite(v):
                out[s] = float(v)
    return out


def collect(metric: str) -> tuple[list[str], np.ndarray]:
    per_method = [section_values(p, hs, metric) for _, _, p, hs in CONDITIONS]
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
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(ttest_rel(a, b).pvalue) if len(a) >= 2 else float("nan")


def whisker_top(v):
    q1, q3 = np.percentile(v, [25, 75]); iqr = q3 - q1
    return float(v[v <= q3 + 1.5 * iqr].max())


def whisker_bot(v):
    q1, q3 = np.percentile(v, [25, 75]); iqr = q3 - q1
    return float(v[v >= q1 - 1.5 * iqr].min())


def plot_panel(ax, arr, ylabel, letter, lower_is_better):
    labels = [c[0] for c in CONDITIONS]
    colors = [c[1] for c in CONDITIONS]
    n = arr.shape[0]
    x = np.arange(n)
    uni_idx = labels.index(UNI_LABEL)
    sig_idx = labels.index(SIGMMA_LABEL)
    cluster_idx = sorted([i for i, l in enumerate(labels) if l in CLUSTER_LABELS], reverse=True)
    regressor_idx = sorted([i for i, l in enumerate(labels) if l in REGRESSOR_LABELS], reverse=True)

    bp = ax.boxplot([arr[i] for i in range(n)], positions=x, widths=0.6,
                    patch_artist=True, showcaps=True, showfliers=False, whis=1.5,
                    boxprops=dict(linewidth=0.6), medianprops=dict(color="black", linewidth=1.0),
                    whiskerprops=dict(linewidth=0.6, color="black"),
                    capprops=dict(linewidth=0.6, color="black"))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i]); patch.set_edgecolor("black")
        patch.set_linewidth(1.2 if i == sig_idx else 0.6); patch.set_alpha(0.9)
    for i in range(n):
        jit = np.linspace(-0.13, 0.13, len(arr[i]))
        ax.scatter(i + jit, arr[i], s=4, color="black", alpha=0.22, zorder=4, linewidth=0)

    ax.axhline(float(np.median(arr[uni_idx])), color="#888888", linestyle="--",
               linewidth=0.6, alpha=0.7, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="-", linewidth=0.3, color="#cccccc", alpha=0.7, zorder=-1)
    ax.set_axisbelow(True)

    all_top = max(whisker_top(arr[i]) for i in range(n))
    all_bot = min(whisker_bot(arr[i]) for i in range(n))
    cluster_top = max(whisker_top(arr[i]) for i in cluster_idx + [sig_idx])
    ref_range = all_top - (0.0 if lower_is_better else all_bot)
    bstep = ref_range * 0.05
    tick_h = ref_range * 0.006
    sig_vals = arr[sig_idx]

    def sig_only(idx_list):
        return [bi for bi in idx_list
                if sig_marker(paired_t(sig_vals, arr[bi])) not in ("ns", "n/a")]
    cluster_sig = sig_only(cluster_idx)
    regressor_sig = sig_only(regressor_idx)

    t1_base = cluster_top + bstep * 0.8
    t1_top = t1_base + bstep * max(len(cluster_sig) - 1, 0) + tick_h
    t2_base = max(all_top, t1_top) + bstep * 0.8
    t2_top = t2_base + bstep * max(len(regressor_sig) - 1, 0) + tick_h
    ax.set_ylim(0.0 if lower_is_better else all_bot - ref_range * 0.05,
                max(all_top, t2_top) + ref_range * 0.06)

    def draw(idx_list, base):
        for slot, bi in enumerate(idx_list):
            mk = sig_marker(paired_t(sig_vals, arr[bi]))
            y_b = base + bstep * slot
            x_l, x_r = sorted([bi, sig_idx])
            ax.plot([x_l, x_l, x_r, x_r], [y_b, y_b + tick_h, y_b + tick_h, y_b],
                    color="black", linewidth=0.4, clip_on=False)
            ax.text((x_l + x_r) / 2, y_b + tick_h - ref_range * 0.002, mk,
                    ha="center", va="bottom", fontsize=5.5, fontweight="bold", clip_on=False)
    draw(cluster_sig, t1_base)
    draw(regressor_sig, t2_base)
    ax.text(-0.18, 1.02, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left")


def main():
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "axes.linewidth": 0.6, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 4.6), gridspec_kw={"wspace": 0.42})
    n_sections = None
    for ax, (metric, ylabel, letter) in zip(axes, METRICS):
        sections, arr = collect(metric)
        n_sections = len(sections)
        plot_panel(ax, arr, ylabel, letter, lower_is_better="mse" in metric.lower())

    fig.suptitle(r"H&E $\rightarrow$ Gene expression prediction : all test sections (pooled)",
                 fontsize=10, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(bottom=0.30)
    fig.text(0.5, 0.085,
             f"Box: median + Q1-Q3, 1.5 x IQR whiskers; dots = sections "
             f"(n = {n_sections}: 33 in-house full-panel + 6 HEST-1k public shared-gene zero-shot); "
             "dashed = UNI2-h median.", ha="center", fontsize=6.5)
    fig.text(0.5, 0.04,
             f"Paired t-test (n = {n_sections}) SIGMMA vs every baseline:  "
             "* P ≤ 0.05,  ** P ≤ 0.01,  *** P ≤ 0.001  (ns omitted; "
             "upper = regressors, lower = backbone/contrastive).", ha="center", fontsize=6.5)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600); fig.savefig(OUT_PDF)
    print(f"Wrote: {OUT_PNG}\nWrote: {OUT_PDF}")

    print(f"\nPooled median [Q1, Q3] (n={n_sections}):")
    for metric, _, _ in METRICS:
        sections, arr = collect(metric)
        labels = [c[0] for c in CONDITIONS]
        sv = arr[labels.index(SIGMMA_LABEL)]
        print(f"\n  {metric}")
        for i, (lab, _, _, _) in enumerate(CONDITIONS):
            q1, q3 = np.percentile(arr[i], [25, 75])
            mk = "" if lab == SIGMMA_LABEL else sig_marker(paired_t(sv, arr[i]))
            print(f"    {lab:<11} {np.median(arr[i]):.3f} [{q1:.3f}, {q3:.3f}]  {mk}")


if __name__ == "__main__":
    main()
