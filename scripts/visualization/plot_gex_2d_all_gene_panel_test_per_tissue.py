"""Report 2/3: GEX-2D TEST benchmark stratified by IN-HOUSE tissue (all-gene svg0529).

7 tissues × 3 metrics grid. Each cell = a 9-method boxplot for one tissue × one
metric, n=2-8 sections per tissue. Per-tissue stats dropped (n too small for
reliable paired tests); qualitative consistency check on the pooled view.

Tissue assignment is AUTHORITATIVE from sample_metadata.csv (sample_id ->
tissue_type), NOT section-name parsing — so beacon and CTCL (both "SKI" by name)
are kept SEPARATE. Glioblastoma (n=1) excluded. HEST-1k public sections are not
in the in-house result set and have their own per-tissue report
(plot_gex_2d_hest1k_per_tissue.py).

Metrics match the pooled figures (Pearson / MSE_slide_std / SSIM_slide).
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
SAMPLE_META = Path("/lustre/scratch126/cellgen/lotfollahi/dj16/data/"
                   "sanger_in_house_xenium/sample_metadata.csv")
_TISSUE_OF = pd.read_csv(SAMPLE_META).set_index("sample_id")["tissue_type"].to_dict()
PROBE_ROOT = ROOT / "outputs/gex_2d/svg0529_step2000_test"
REGRESS_ROOT = ROOT / "outputs/gex_2d/svg0529_panel_35a7bd78801c_test"
OUT_PNG = ROOT / "figures/gex_2d/all_gene_panel/test_boxplot_per_tissue.png"
OUT_PDF = ROOT / "figures/gex_2d/all_gene_panel/test_boxplot_per_tissue.pdf"

# Regressor baselines (K=14 dense bin) first (far left), distinct hues.
CONDITIONS = [
    ("ST-Net",     "#332288", REGRESS_ROOT / "stnet_svg0529_cache/stnet_bin.json"),
    ("HisToGene",  "#88CCEE", REGRESS_ROOT / "histogene_svg0529_cache/histogene_bin.json"),
    ("DeepPT",     "#4477AA", REGRESS_ROOT / "deeppt_svg0529_cache/deeppt_bin.json"),
    ("UNI2-h",     "#888888", PROBE_ROOT / "uni2h_no_ft/results_linear_sweep/uni.json"),
    ("BLEEP-ft",   "#882255", PROBE_ROOT / "bleep_uni_step2000/results_linear_sweep/uni.json"),
    ("CLIP-ft",    "#44AA99", PROBE_ROOT / "clip_uni_step2000/results_linear_sweep/uni.json"),
    ("PLIP-ft",    "#DDCC77", PROBE_ROOT / "plip_uni_step2000/results_linear_sweep/uni.json"),
    ("OmiCLIP-ft", "#999933", PROBE_ROOT / "omiclip_uni_step2000/results_linear_sweep/uni.json"),
    ("SIGMMA",     "#d62728", PROBE_ROOT / "sigmma_a01_step2000/results/uni.json"),
]
UNI_LABEL = "UNI2-h"

METRICS = [
    ("median_per_gene_pearson",            r"Per-gene Pearson  $\uparrow$",     "a"),
    ("median_mse_per_slide_standardized",  r"MSE per slide (std) $\downarrow$", "b"),
    ("median_ssim_per_slide",              r"SSIM per slide  $\uparrow$",       "c"),
]

# In-house tissue_type values (sample_metadata.csv) with n>=2 test sections.
# beacon and CTCL kept SEPARATE; glioblastoma (n=1) excluded.
TISSUES = [
    ("beacon",                   "Beacon (skin)"),
    ("CTCL",                     "CTCL (skin)"),
    ("renal_cell_carcinoma",     "Kidney (RCC)"),
    ("pancreas",                 "Pancreas"),
    ("liver",                    "Liver"),
    ("adrenocortical_carcinoma", "ACC"),
    ("whole_embryo",             "Whole embryo"),
]


def group_of(section_name: str) -> str | None:
    """Authoritative tissue_type from sample_metadata.csv; None if unmapped."""
    return _TISSUE_OF.get(section_name)


def load_per_slide(path: Path, metric: str) -> dict[str, float]:
    return dict(json.load(open(path))["per_slide"][metric]["values"])


def collect(metric: str) -> tuple[list[str], np.ndarray]:
    per_method = [load_per_slide(path, metric) for _, _, path in CONDITIONS]
    sections = sorted(set.intersection(*[set(d) for d in per_method]))
    arr = np.array([[d[s] for s in sections] for d in per_method])
    return sections, arr


def plot_panel(ax, arr_tissue: np.ndarray, ylabel: str, tissue_name: str | None,
               show_xlabels: bool):
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

    for i, vals in enumerate(data):
        jitter = np.random.default_rng(42 + i).uniform(-0.10, 0.10, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   s=4, color="black", alpha=0.28, zorder=4, linewidth=0)

    uni_idx = labels.index(UNI_LABEL)
    ax.axhline(float(np.median(arr_tissue[uni_idx])), color="#888888",
               linestyle="--", linewidth=0.6, alpha=0.7, zorder=0)

    ax.set_xticks(x)
    if show_xlabels:
        ax.set_xticklabels(labels, rotation=40, ha="right")
    else:
        ax.set_xticklabels([])
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

    metric_data = {}
    for metric, _, _ in METRICS:
        sections, arr = collect(metric)
        metric_data[metric] = (sections, arr)

    n_rows, n_cols = len(TISSUES), len(METRICS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.2, 1.55 * n_rows + 1.3),
                             gridspec_kw={"wspace": 0.35, "hspace": 0.35})

    for r, (tcode, tname) in enumerate(TISSUES):
        for c, (metric, ylabel, _) in enumerate(METRICS):
            ax = axes[r, c]
            sections, arr = metric_data[metric]
            mask = np.array([group_of(s) == tcode for s in sections])
            arr_t = arr[:, mask]
            plot_panel(
                ax, arr_t,
                ylabel=ylabel,
                tissue_name=tname if c == 0 else None,
                show_xlabels=(r == n_rows - 1),
            )
            if r == 0:
                ax.set_title(ylabel, fontsize=8, pad=4)

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
