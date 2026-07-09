"""Figure: GEX-2D TEST-cohort benchmark stratified by tissue.

5 tissues × 3 metrics grid. Each cell shows a 6-method boxplot for one tissue
× one metric, with n=2-5 sections per tissue. Per-tissue stats are dropped
(n too small for reliable paired tests); the figure is a qualitative
consistency check on the pooled view.

Metrics match the pooled test figure (Pearson / MSE_slide_std / SSIM_slide).
Pooled view: figures/gex_2d/svg_panel_2d22ba31_test_boxplot.{png,pdf}
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sigmma.utils.tissue_map import tissue_of, GROUP_OF, PUBLIC_TT

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
RES_ROOT = ROOT / "outputs/gex_2d/svg_panel_2d22ba31_step1000_test"
OUT_PNG = ROOT / "figures/gex_2d/svg_panel_2d22ba31_test_boxplot_per_tissue.png"
OUT_PDF = ROOT / "figures/gex_2d/svg_panel_2d22ba31_test_boxplot_per_tissue.pdf"

# Regressor baselines (K=14 dense bin) first (far left), distinct hues.
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
UNI_LABEL = "UNI2-h"

# (metric_key, ylabel, panel_letter) — match pooled test figure.
METRICS = [
    ("median_per_gene_pearson",            r"Per-gene Pearson  $\uparrow$",     "a"),
    ("median_mse_per_slide_standardized",  r"MSE per slide (std) $\downarrow$", "b"),
    ("median_ssim_per_slide",              r"SSIM per slide  $\uparrow$",       "c"),
]

# Tissue groups to facet by (row order), keyed by display group (GROUP_OF).
# In-panel tissues only — glioblastoma (BRA) dropped (5K-panel under-represented,
# mostly on the separate GBM panel); external HEST-1k public excluded by the
# group filter (skin = beacon + CTCL merged).
TISSUES = [
    ("renal_cell_carcinoma", "Kidney"),
    ("liver", "Liver"),
    ("skin", "Skin"),
    ("whole_embryo", "Whole embryo"),
]


def group_of(section_name: str) -> str | None:
    """Authoritative display group; None for public/unknown sections."""
    t = tissue_of(section_name)
    if t in PUBLIC_TT:
        return None
    return GROUP_OF.get(t)


def load_per_slide(path: Path, metric: str) -> dict[str, float]:
    return dict(json.load(open(path))["per_slide"][metric]["values"])


def collect(metric: str) -> tuple[list[str], np.ndarray]:
    per_method = [load_per_slide(RES_ROOT / sub, metric) for _, _, sub in CONDITIONS]
    sections = sorted(set.intersection(*[set(d) for d in per_method]))
    arr = np.array([[d[s] for s in sections] for d in per_method])
    return sections, arr


def plot_panel(ax, arr_tissue: np.ndarray, ylabel: str, panel_letter: str | None,
               tissue_name: str | None, show_xlabels: bool, show_ylabel: bool,
               metric_key: str):
    n_methods, n_sec = arr_tissue.shape
    x = np.arange(n_methods)
    colors = [c[1] for c in CONDITIONS]
    labels = [c[0] for c in CONDITIONS]
    sigmma_idx = n_methods - 1

    data = [arr_tissue[i] for i in range(n_methods)]
    bp = ax.boxplot(
        data, positions=x, widths=0.55, patch_artist=True,
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

    # Overlay per-section dots so n is visible
    for i, vals in enumerate(data):
        jitter = np.random.default_rng(42 + i).uniform(-0.10, 0.10, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   s=4, color="black", alpha=0.5, zorder=4, linewidth=0)

    # UNI2-h reference at its median (UNI2-h is no longer index 0 — regressors lead)
    uni_idx = [c[0] for c in CONDITIONS].index(UNI_LABEL)
    uni_med = float(np.median(arr_tissue[uni_idx]))
    ax.axhline(uni_med, color="#888888", linestyle="--",
               linewidth=0.6, alpha=0.7, zorder=0)

    ax.set_xticks(x)
    if show_xlabels:
        ax.set_xticklabels(labels, rotation=40, ha="right")
    else:
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="-", linewidth=0.3, color="#cccccc",
            alpha=0.7, zorder=-1)
    ax.set_axisbelow(True)

    if tissue_name is not None:
        ax.text(-0.30, 0.5, tissue_name + f"\n(n={n_sec})",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, fontweight="bold", rotation=90)


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

    # Pre-collect metric data once; index into per-tissue subsets
    metric_data = {}
    for metric, _, _ in METRICS:
        sections, arr = collect(metric)
        metric_data[metric] = (sections, arr)

    n_rows, n_cols = len(TISSUES), len(METRICS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.2, 2.0 * n_rows + 1.5),
                             gridspec_kw={"wspace": 0.35, "hspace": 0.30})

    for r, (tcode, tname) in enumerate(TISSUES):
        for c, (metric, ylabel, _) in enumerate(METRICS):
            ax = axes[r, c]
            sections, arr = metric_data[metric]
            mask = np.array([group_of(s) == tcode for s in sections])
            arr_t = arr[:, mask]
            plot_panel(
                ax, arr_t,
                ylabel=ylabel if r == n_rows - 1 else "",
                panel_letter=None,
                tissue_name=tname if c == 0 else None,
                show_xlabels=(r == n_rows - 1),
                show_ylabel=True,
                metric_key=metric,
            )
            # Column-shared metric title above top row only
            if r == 0:
                ax.set_title(ylabel, fontsize=8, pad=4)
            # Hide the in-cell ylabel (we use shared column titles + row labels)
            ax.set_ylabel("")

    fig.suptitle(
        r"H&E $\rightarrow$ Gene expression prediction  (per tissue, test cohort)",
        fontsize=10, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0.02, 0.04, 1.0, 0.97])
    fig.text(
        0.5, 0.015,
        "Box: median + Q1–Q3, 1.5 × IQR whiskers; black dots = per-section "
        "values. n shown per tissue. Dashed line = UNI2-h median.",
        ha="center", fontsize=6.5, color="black",
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600)
    fig.savefig(OUT_PDF)
    print(f"Wrote: {OUT_PNG}")
    print(f"Wrote: {OUT_PDF}")

    # Console: per-tissue median ± section counts
    print("\nPer-tissue median by method (n_sec in parentheses):")
    for tcode, tname in TISSUES:
        print(f"\n  {tname} ({tcode})")
        for metric, _, _ in METRICS:
            sections, arr = metric_data[metric]
            mask = np.array([group_of(s) == tcode for s in sections])
            arr_t = arr[:, mask]
            meds = [float(np.median(arr_t[i])) for i in range(arr_t.shape[0])]
            line = f"    {metric:<38} (n={arr_t.shape[1]}) " + \
                   "  ".join(f"{lbl}={m:.3f}" for (lbl, _, _), m in zip(CONDITIONS, meds))
            print(line)


if __name__ == "__main__":
    main()
