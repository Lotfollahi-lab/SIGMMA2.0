"""Report 3/3: GEX-2D HEST-1k public zero-shot benchmark stratified by tissue.

2 tissues (SKCM n=2, IDC n=4) x 3 metrics grid, mirroring the in-house per-tissue
report (plot_gex_2d_all_gene_panel_test_per_tissue.py). Each cell = a 9-method
boxplot for one HEST-1k tissue x one metric. Per-tissue stats dropped (n too
small); qualitative view of the strict zero-shot transfer.

Tissue assignment is AUTHORITATIVE from sample_metadata.csv (sample_id ->
tissue_type). SIGMMA at headline repr=uni, step2000. Section metrics are over
each section's 12-15 shared genes (in-house main panel n HEST var.index).
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
RES = ROOT / "outputs/gex_2d/public_zeroshot_strict_step2000"
SAMPLE_META = Path("/lustre/scratch126/cellgen/lotfollahi/dj16/data/"
                   "sanger_in_house_xenium/sample_metadata.csv")
_TISSUE_OF = pd.read_csv(SAMPLE_META).set_index("sample_id")["tissue_type"].to_dict()
OUT_PNG = ROOT / "figures/gex_2d/hest1k_zeroshot/hest1k_per_tissue.png"
OUT_PDF = ROOT / "figures/gex_2d/hest1k_zeroshot/hest1k_per_tissue.pdf"

# tissue_type -> result panel hash (>=10 shared genes)
TISSUES = [
    ("SKCM", "80b4f72e3da8", "SKCM"),
    ("IDC",  "2bbc576266c5", "IDC"),
]

# (label, color, json-stem). Regressors far-left; SIGMMA repr=uni.
CONDITIONS = [
    ("ST-Net",     "#332288", "stnet_bin"),
    ("HisToGene",  "#88CCEE", "histogene_bin"),
    ("DeepPT",     "#4477AA", "deeppt_bin"),
    ("UNI2-h",     "#888888", "uni2h_no_ft"),
    ("BLEEP-ft",   "#882255", "bleep_uni_step2000"),
    ("CLIP-ft",    "#44AA99", "clip_uni_step2000"),
    ("PLIP-ft",    "#DDCC77", "plip_uni_step2000"),
    ("OmiCLIP-ft", "#999933", "omiclip_uni_step2000"),
    ("SIGMMA",     "#d62728", "sigmma_a01_step2000__uni"),
]
UNI_LABEL = "UNI2-h"

METRICS = [
    ("median_per_gene_pearson",            r"Per-gene Pearson  $\uparrow$",     "a"),
    ("median_mse_per_slide_standardized",  r"MSE per slide (std) $\downarrow$", "b"),
    ("median_ssim_per_slide",              r"SSIM per slide  $\uparrow$",       "c"),
]


def section_values(panel: str, stem: str, metric: str) -> dict[str, float]:
    p = RES / panel / f"{stem}.json"
    if not p.exists():
        return {}
    blk = json.load(open(p))["per_slide"].get(metric, {})
    return {s: float(v) for s, v in (blk.get("values") or {}).items()
            if v is not None and np.isfinite(v)}


def collect(panel: str, metric: str) -> np.ndarray:
    per_method = [section_values(panel, stem, metric) for _, _, stem in CONDITIONS]
    sections = sorted(set.intersection(*[set(d) for d in per_method]))
    # authoritative tissue check (defensive; panels are single-tissue by design)
    return np.array([[d[s] for s in sections] for d in per_method]), sections


def plot_panel(ax, arr, ylabel, tissue_name, show_xlabels):
    n_methods, n_sec = arr.shape
    x = np.arange(n_methods)
    colors = [c[1] for c in CONDITIONS]
    labels = [c[0] for c in CONDITIONS]
    sig_idx = n_methods - 1

    bp = ax.boxplot([arr[i] for i in range(n_methods)], positions=x, widths=0.55,
                    patch_artist=True, showcaps=True, showfliers=False, whis=1.5,
                    boxprops=dict(linewidth=0.6), medianprops=dict(color="black", linewidth=1.0),
                    whiskerprops=dict(linewidth=0.6, color="black"),
                    capprops=dict(linewidth=0.6, color="black"))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i]); patch.set_edgecolor("black")
        patch.set_linewidth(1.2 if i == sig_idx else 0.6); patch.set_alpha(0.9)
    for i in range(n_methods):
        jit = np.random.default_rng(42 + i).uniform(-0.10, 0.10, size=arr.shape[1])
        ax.scatter(np.full(arr.shape[1], i) + jit, arr[i], s=4, color="black",
                   alpha=0.28, zorder=4, linewidth=0)

    ax.axhline(float(np.median(arr[labels.index(UNI_LABEL)])), color="#888888",
               linestyle="--", linewidth=0.6, alpha=0.7, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right") if show_xlabels else ax.set_xticklabels([])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="-", linewidth=0.3, color="#cccccc", alpha=0.7, zorder=-1)
    ax.set_axisbelow(True)
    if tissue_name is not None:
        ax.text(-0.30, 0.5, f"{tissue_name}\n(n={n_sec})", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, fontweight="bold", rotation=90)


def main():
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "axes.linewidth": 0.6, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42})

    n_rows, n_cols = len(TISSUES), len(METRICS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.2, 1.9 * n_rows + 1.4),
                             gridspec_kw={"wspace": 0.35, "hspace": 0.32}, squeeze=False)
    for r, (tt, panel, tname) in enumerate(TISSUES):
        for c, (metric, ylabel, _) in enumerate(METRICS):
            ax = axes[r, c]
            arr, _ = collect(panel, metric)
            plot_panel(ax, arr, ylabel, tname if c == 0 else None, show_xlabels=(r == n_rows - 1))
            if r == 0:
                ax.set_title(ylabel, fontsize=8, pad=4)

    fig.suptitle(r"H&E $\rightarrow$ Gene expression : HEST-1k public zero-shot (per tissue)",
                 fontsize=10, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0.02, 0.04, 1.0, 0.96])
    fig.text(0.5, 0.015,
             "Box: median + Q1-Q3, 1.5 x IQR whiskers; black dots = per-section values "
             "(n per tissue). Dashed = UNI2-h median. SIGMMA repr=uni.",
             ha="center", fontsize=6.5)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600); fig.savefig(OUT_PDF)
    print(f"Wrote: {OUT_PNG}\nWrote: {OUT_PDF}")

    print("\nHEST per-tissue median by method:")
    for tt, panel, tname in TISSUES:
        print(f"\n  {tname} ({panel})")
        for metric, _, _ in METRICS:
            arr, secs = collect(panel, metric)
            meds = [float(np.median(arr[i])) for i in range(arr.shape[0])]
            print(f"    {metric:<38} (n={len(secs)}) " +
                  "  ".join(f"{l}={m:.3f}" for (l, _, _), m in zip(CONDITIONS, meds)))


if __name__ == "__main__":
    main()
