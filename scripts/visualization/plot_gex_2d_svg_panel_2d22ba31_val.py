"""Figure: GEX-2D val-cohort benchmark on panel-aware SVG-100 (panel_2d22ba31).

Three metrics, all section-level (median + IQR across 15 val sections):
  * R² pooled (per_bin_r2_pooled)             — higher better
  * MSE per slide, standardized (MSE_sl_std)  — lower better
  * SSIM per slide (median_ssim_per_slide)    — higher better

eva_critique style (per [[project_benchmark_figure_style]]):
  - Boxplot (median, Q1–Q3, 1.5×IQR whiskers)
  - Project method palette (UNI2-h grey + 4 contrastive FT + SIGMMA red)
  - UNI2-h dashed reference line at its median
  - Helvetica 8 pt, 40° xticks
  - Paired t-test SIGMMA vs each baseline
  - "H&E → Gene expression prediction" suptitle

SIGMMA shown at α=0.1, repr=uni (matches main comparison; uni repr identical
across α=0/0.1/0.2/0.4 in this panel).
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from pathlib import Path

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
RES_ROOT = ROOT / "outputs/gex_2d/svg_panel_2d22ba31_step1000"
OUT_PNG = ROOT / "figures/gex_2d/svg_panel_2d22ba31_val_boxplot.png"
OUT_PDF = ROOT / "figures/gex_2d/svg_panel_2d22ba31_val_boxplot.pdf"

# (label, color, result_path)
CONDITIONS = [
    ("UNI2-h",     "#888888", "uni2h_no_ft/results_linear_sweep/uni.json"),
    ("BLEEP-ft",   "#882255", "bleep_uni_step1000/results_linear_sweep/uni.json"),
    ("CLIP-ft",    "#44AA99", "clip_uni_step1000/results_linear_sweep/uni.json"),
    ("PLIP-ft",    "#DDCC77", "plip_uni_step1000/results_linear_sweep/uni.json"),
    ("OmiCLIP-ft", "#999933", "omiclip_uni_step1000/results_linear_sweep/uni.json"),
    ("SIGMMA",     "#d62728", "sigmma_a01_step1000/results/uni.json"),
]

# (metric_key, ylabel, panel_letter, ylim_override)
# ylim_override = (ymin, ymax) tightens the y-axis; None = auto from data range.
# - MSE: anchored at 0 to keep ratio scale honest (auto range exaggerated SIGMMA gap).
# - SSIM: focused window (whiskers extend ~0.20-0.81 but boxes sit ~0.39-0.53;
#   auto range drowns the +0.017 median signal).
METRICS = [
    ("per_bin_r2_pooled",                  r"$R^2$ pooled  $\uparrow$",          "a", None),
    ("median_mse_per_slide_standardized",  r"MSE per slide (std) $\downarrow$",  "b", (0.0, 0.20)),
    ("median_ssim_per_slide",              r"SSIM per slide  $\uparrow$",        "c", (0.17, 0.70)),
]


def load_per_slide(path: Path, metric: str) -> dict[str, float]:
    d = json.load(open(path))
    return dict(d["per_slide"][metric]["values"])


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
    per_method = []
    for label, _, sub in CONDITIONS:
        v = load_per_slide(RES_ROOT / sub, metric)
        per_method.append(v)
    sections = sorted(set.intersection(*[set(d) for d in per_method]))
    arr = np.array([[d[s] for s in sections] for d in per_method])
    return sections, arr


def plot_panel(ax, arr: np.ndarray, ylabel: str, panel_letter: str, lower_is_better: bool,
               ylim_override=None):
    n_methods, n_sec = arr.shape
    medians = np.median(arr, axis=1)
    x = np.arange(n_methods)
    colors = [c[1] for c in CONDITIONS]
    labels = [c[0] for c in CONDITIONS]
    sigmma_idx = n_methods - 1

    # 1.5×IQR whiskers everywhere (matplotlib default) for consistency across
    # panels. ylim_override must accommodate the full whisker extent.
    whis = 1.5

    data = [arr[i] for i in range(n_methods)]
    bp = ax.boxplot(
        data, positions=x, widths=0.55, patch_artist=True,
        showcaps=True, showfliers=False, whis=whis,
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
    ax.axhline(medians[0], color="#888888", linestyle="--",
               linewidth=0.6, alpha=0.7, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="-", linewidth=0.3, color="#cccccc",
            alpha=0.7, zorder=-1)
    ax.set_axisbelow(True)

    # ylim: focus window if override provided, else full whisker extent + headroom.
    # When zoomed, brackets sit inside the override window — bracket_step scales
    # to the focused y-range so they remain compact and legible.
    n_baseline = n_methods - 1
    if ylim_override is not None:
        ymin, ymax = ylim_override
        yrange = ymax - ymin
        bracket_step = yrange * 0.040
        # Place brackets near top of focus window so they don't sit on boxes.
        base_y_anchor = ymax - bracket_step * (n_baseline + 0.6)
    else:
        whisker_top = float(arr.max())
        whisker_bot = float(arr.min())
        data_range = whisker_top - whisker_bot
        bracket_step = data_range * 0.055
        headroom = bracket_step * (n_baseline + 0.8)
        ymax = whisker_top + headroom
        # Allow negative ymin if the data goes negative (e.g. R² < 0 on hard
        # sections). Hard-clamping at 0 would clip whiskers.
        ymin = whisker_bot - data_range * 0.05
        yrange = ymax - ymin
        base_y_anchor = whisker_top + bracket_step * 0.35
    ax.set_ylim(ymin, ymax)

    # Stacked paired t-test brackets, SIGMMA vs each baseline
    sigmma_vals = arr[sigmma_idx]
    tick_h = yrange * 0.006
    base_y = base_y_anchor
    for slot in range(n_methods - 1):
        p = paired_t(sigmma_vals, arr[slot])
        mk = sig_marker(p)
        y_b = base_y + bracket_step * slot
        x_l, x_r = sorted([slot, sigmma_idx])
        ax.plot([x_l, x_l, x_r, x_r],
                [y_b, y_b + tick_h, y_b + tick_h, y_b],
                color="black", linewidth=0.4, clip_on=False)
        ax.text((x_l + x_r) / 2, y_b + tick_h - yrange * 0.002,
                mk, ha="center", va="bottom",
                fontsize=5.5, fontweight="bold", color="black",
                clip_on=False)

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

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 4.4),
                             gridspec_kw={"wspace": 0.50})
    for ax, (metric, ylabel, letter, ylim_override) in zip(axes, METRICS):
        sections, arr = collect(metric)
        lower_is_better = "mse" in metric.lower()
        plot_panel(ax, arr, ylabel, letter, lower_is_better,
                   ylim_override=ylim_override)

    fig.suptitle(
        r"H&E $\rightarrow$ Gene expression prediction",
        fontsize=10, fontweight="bold", y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(bottom=0.28)
    fig.text(
        0.5, 0.07,
        "Box: median + Q1–Q3, 1.5 × IQR whiskers (n = 15 val sections); "
        "dashed line = UNI2-h median.",
        ha="center", fontsize=6.5, color="black",
    )
    fig.text(
        0.5, 0.03,
        "Paired t-test vs SIGMMA:  * P ≤ 0.05,  ** P ≤ 0.01,  *** P ≤ 0.001.",
        ha="center", fontsize=6.5, color="black",
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600)
    fig.savefig(OUT_PDF)
    print(f"Wrote: {OUT_PNG}")
    print(f"Wrote: {OUT_PDF}")

    # Console summary
    print("\nPer-metric median [Q1, Q3] (n=15 val sections):")
    for metric, ylabel, _, _ in METRICS:
        sections, arr = collect(metric)
        print(f"\n  {metric}")
        sigmma_vals = arr[-1]
        for i, (label, _, _) in enumerate(CONDITIONS):
            med = float(np.median(arr[i]))
            q1, q3 = np.percentile(arr[i], [25, 75])
            p = paired_t(sigmma_vals, arr[i]) if i != len(CONDITIONS) - 1 else float("nan")
            mk = sig_marker(p) if i != len(CONDITIONS) - 1 else ""
            print(f"    {label:<12} {med:.4f} [{q1:.4f}, {q3:.4f}]   {mk}")


if __name__ == "__main__":
    main()
