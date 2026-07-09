"""Figure: GEX-2D TEST-cohort benchmark on panel-aware SVG-100 (panel_2d22ba31).

Three metrics, all section-level (median + IQR across 14 test sections):
  * Per-gene Pearson (median_per_gene_pearson)   — higher better
  * MSE per slide, standardized (MSE_sl_std)      — lower better  (headline signal)
  * SSIM per slide (median_ssim_per_slide)        — higher better

eva_critique style (per [[project_benchmark_figure_style]]):
  - Boxplot (median, Q1-Q3, 1.5xIQR whiskers)
  - Regressor baselines (ST-Net, HisToGene; K=14 dense bin) at far LEFT,
    distinct hues — a different method class from the contrastive FTs.
  - Project method palette (UNI2-h grey + 4 contrastive FT + SIGMMA red)
  - UNI2-h dashed reference line at its median
  - Helvetica 8 pt, 40 deg xticks
  - Paired t-test SIGMMA vs each contrastive/backbone baseline. Brackets vs
    ST-Net/HisToGene OMITTED (gap overwhelming; placing them would clutter and
    collide with their tall boxes).

SIGMMA shown at alpha=0.1, repr=uni (uni repr identical across alpha in this panel).
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from pathlib import Path

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
RES_ROOT = ROOT / "outputs/gex_2d/svg_panel_2d22ba31_step1000_test"
OUT_PNG = ROOT / "figures/gex_2d/svg_panel_2d22ba31_test_boxplot.png"
OUT_PDF = ROOT / "figures/gex_2d/svg_panel_2d22ba31_test_boxplot.pdf"

# (label, color, result_path). Regressor baselines first (far left).
CONDITIONS = [
    ("ST-Net",     "#332288", "stnet_svg_cache/stnet_bin.json"),
    ("HisToGene",  "#88CCEE", "histogene_svg_cache/histogene_bin.json"),
    ("DeepPT",     "#4477AA", "deeppt_svg_cache/deeppt_bin.json"),
    ("UNI2-h",     "#888888", "uni2h_no_ft/results_linear_sweep/uni.json"),
    ("BLEEP-ft",   "#882255", "bleep_uni_step1000/results_linear_sweep/uni.json"),
    ("CLIP-ft",    "#44AA99", "clip_uni_step1000/results_linear_sweep/uni.json"),
    ("PLIP-ft",    "#DDCC77", "plip_uni_step1000/results_linear_sweep/uni.json"),
    ("OmiCLIP-ft", "#999933", "omiclip_uni_step1000/results_linear_sweep/uni.json"),
    ("SIGMMA",     "#d62728", "sigmma_a01_step1000/results/uni.json"),
]
# Two-tier significance brackets vs SIGMMA:
#   cluster (lower tier) = backbone + contrastive FTs, anchored just above their boxes
#   regressors (upper tier) = ST-Net / HisToGene, anchored above ALL boxes (their
#   tall boxes would be crossed by a mid-height bracket spanning the full panel).
CLUSTER_LABELS = {"UNI2-h", "BLEEP-ft", "CLIP-ft", "PLIP-ft", "OmiCLIP-ft"}
REGRESSOR_LABELS = {"ST-Net", "HisToGene", "DeepPT"}
UNI_LABEL = "UNI2-h"
SIGMMA_LABEL = "SIGMMA"

# (metric_key, ylabel)  — lower_is_better inferred from "mse" in key.
METRICS = [
    ("median_per_gene_pearson",            r"Per-gene Pearson  $\uparrow$",     "a"),
    ("median_mse_per_slide_standardized",  r"MSE per slide (std) $\downarrow$", "b"),
    ("median_ssim_per_slide",              r"SSIM per slide  $\uparrow$",       "c"),
]


def load_per_slide(path: Path, metric: str) -> dict[str, float]:
    return dict(json.load(open(path))["per_slide"][metric]["values"])


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


def collect(metric: str) -> tuple[list[str], np.ndarray]:
    """Return (section_order, 2-D array [n_methods, n_sections]) — paired."""
    per_method = [load_per_slide(RES_ROOT / sub, metric) for _, _, sub in CONDITIONS]
    sections = sorted(set.intersection(*[set(d) for d in per_method]))
    arr = np.array([[d[s] for s in sections] for d in per_method])
    return sections, arr


def whisker_top(v):
    q1, q3 = np.percentile(v, [25, 75]); iqr = q3 - q1
    return float(v[v <= q3 + 1.5 * iqr].max())


def whisker_bot(v):
    q1, q3 = np.percentile(v, [25, 75]); iqr = q3 - q1
    return float(v[v >= q1 - 1.5 * iqr].min())


def plot_panel(ax, arr, ylabel, panel_letter, lower_is_better):
    labels = [c[0] for c in CONDITIONS]
    colors = [c[1] for c in CONDITIONS]
    n_methods = arr.shape[0]
    x = np.arange(n_methods)
    uni_idx = labels.index(UNI_LABEL)
    sigmma_idx = labels.index(SIGMMA_LABEL)
    # Stack brackets nearest-to-SIGMMA first (shortest span at bottom of each tier).
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

    # UNI2-h reference line at its median
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

    # Extents. all_* spans every box (incl. tall regressors); cluster_top is the
    # top of the SIGMMA-vs-contrastive group, used to anchor the lower bracket tier
    # just above those boxes (well below the tall regressor boxes on the left).
    all_top = max(whisker_top(arr[i]) for i in range(n_methods))
    all_bot = min(whisker_bot(arr[i]) for i in range(n_methods))
    cluster_top = max(whisker_top(arr[i]) for i in cluster_idx + [sigmma_idx])

    ref_range = all_top - (0.0 if lower_is_better else all_bot)
    bracket_step = ref_range * 0.05
    tick_h = ref_range * 0.006

    # Lower tier: SIGMMA vs cluster, anchored above the cluster boxes.
    t1_base = cluster_top + bracket_step * 0.8
    t1_top = t1_base + bracket_step * (len(cluster_idx) - 1) + tick_h
    # Upper tier: SIGMMA vs regressors, anchored above ALL boxes and the lower tier.
    t2_base = max(all_top, t1_top) + bracket_step * 0.8
    t2_top = t2_base + bracket_step * (max(len(regressor_idx) - 1, 0)) + tick_h

    ymin = 0.0 if lower_is_better else all_bot - ref_range * 0.05
    ymax = max(all_top, t2_top) + ref_range * 0.06
    ax.set_ylim(ymin, ymax)

    sig_vals = arr[sigmma_idx]

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

    draw_brackets(cluster_idx, t1_base)
    draw_brackets(regressor_idx, t2_base)

    ax.text(-0.18, 1.02, panel_letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(8.6, 4.6),
                             gridspec_kw={"wspace": 0.42})
    n_sections = None
    for ax, (metric, ylabel, letter) in zip(axes, METRICS):
        sections, arr = collect(metric)
        n_sections = len(sections)
        plot_panel(ax, arr, ylabel, letter, lower_is_better="mse" in metric.lower())

    fig.suptitle(
        r"H&E $\rightarrow$ Gene expression prediction",
        fontsize=10, fontweight="bold", y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(bottom=0.30)
    fig.text(
        0.5, 0.085,
        f"Box: median + Q1-Q3, 1.5 x IQR whiskers (n = {n_sections} test sections); "
        "dashed line = UNI2-h median. ST-Net / HisToGene / DeepPT = K=14 dense regressors.",
        ha="center", fontsize=6.5, color="black",
    )
    fig.text(
        0.5, 0.04,
        "Paired t-test (n = 14) SIGMMA vs every baseline:  "
        "* P ≤ 0.05,  ** P ≤ 0.01,  *** P ≤ 0.001  "
        "(upper brackets = regressors, lower = backbone/contrastive).",
        ha="center", fontsize=6.5, color="black",
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600)
    fig.savefig(OUT_PDF)
    print(f"Wrote: {OUT_PNG}")
    print(f"Wrote: {OUT_PDF}")

    print(f"\nPer-metric median [Q1, Q3] (n={n_sections} test sections):")
    for metric, ylabel, _ in METRICS:
        sections, arr = collect(metric)
        print(f"\n  {metric}")
        labels = [c[0] for c in CONDITIONS]
        sig_idx = labels.index(SIGMMA_LABEL)
        sig_vals = arr[sig_idx]
        for i, (label, _, _) in enumerate(CONDITIONS):
            med = float(np.median(arr[i]))
            q1, q3 = np.percentile(arr[i], [25, 75])
            mk = "" if label == SIGMMA_LABEL else sig_marker(paired_t(sig_vals, arr[i]))
            print(f"    {label:<12} {med:.4f} [{q1:.4f}, {q3:.4f}]   {mk}")


if __name__ == "__main__":
    main()
