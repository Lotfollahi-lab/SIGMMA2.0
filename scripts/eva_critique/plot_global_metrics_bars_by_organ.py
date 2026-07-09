"""EVA-MoNuSAC PQ bars stratified by organ (Lung, Kidney, Prostate, Breast).

Mirrors `plot_global_metrics_bars.py` style but generates one panel per organ.
SIGMMA α=0.1 vs 4 baseline-FT + UNI2-h backbone, FOV-paired Wilcoxon per organ.

Layout: 1 row × 4 columns (one per organ).
Output: figures/eva_critique/global_metrics_bars_by_organ.{pdf,png}
"""
from __future__ import annotations

import csv
import glob
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
OUT = ROOT / "figures" / "eva_critique"

# TCGA participant → organ (MoNuSAC 4-class)
ORGAN = {
    "TCGA-50": "Lung", "TCGA-55": "Lung", "TCGA-49": "Lung", "TCGA-73": "Lung", "TCGA-78": "Lung",
    "TCGA-DW": "Kidney", "TCGA-2Z": "Kidney", "TCGA-MH": "Kidney", "TCGA-A8": "Kidney",
    "TCGA-IZ": "Kidney",
    "TCGA-A2": "Breast", "TCGA-F9": "Breast",
    "TCGA-G9": "Prostate", "TCGA-VP": "Prostate", "TCGA-YL": "Prostate",
    "TCGA-MP": "Prostate", "TCGA-X4": "Prostate",
}
ORGANS = ["Lung", "Kidney", "Prostate", "Breast"]

def organ_of(fov: str) -> str:
    pid = "-".join(fov.split("-")[:2])
    return ORGAN.get(pid, "Unknown")

CONDITIONS = [
    ("uni2h",       "UNI2-h",       "#888888"),
    ("bleep",       "BLEEP-ft",     "#882255"),
    ("clip",        "CLIP-ft",      "#44AA99"),
    ("plip",        "PLIP-ft",      "#DDCC77"),
    ("omiclip",     "OmiCLIP-ft",   "#999933"),
    ("sigmma_a01",  "SIGMMA",       "#d62728"),
]
SIGMMA_IDX = len(CONDITIONS) - 1

PATTERNS = {
    "uni2h":       f"{ROOT}/outputs/eva_critique_uni2h/seed*/instance_metrics_global.csv",
    "bleep":       f"{ROOT}/outputs/eva_critique_baselines_uni2h_step1000/bleep_uni_ft/seed*/instance_metrics_global.csv",
    "clip":        f"{ROOT}/outputs/eva_critique_baselines_uni2h_step1000/clip_uni_ft/seed*/instance_metrics_global.csv",
    "plip":        f"{ROOT}/outputs/eva_critique_baselines_uni2h_step1000/plip_uni_ft/seed*/instance_metrics_global.csv",
    "omiclip":     f"{ROOT}/outputs/eva_critique_baselines_uni2h_step1000/omiclip_uni_ft/seed*/instance_metrics_global.csv",
    "sigmma_a01":  f"{ROOT}/outputs/eva_critique_gex_sweep_step1000/a01/seed*/instance_metrics_global.csv",
}

# Pre-specified outlier filter (see plot_global_metrics_bars.py).
OUTLIER_FOVS = {
    "TCGA-2Z-A9JG-01Z-00-DX1_2",
    "TCGA-MH-A55W-01Z-00-DX1_2",
    "TCGA-MH-A55W-01Z-00-DX1_4",
    "TCGA-A2-A04X-01Z-00-DX1_003",
    "TCGA-F9-A8NY-01Z-00-DX1_4",
}


def per_fov_seed_avg(pat: str, metric: str) -> dict[str, float]:
    by_fov = defaultdict(list)
    for f in sorted(glob.glob(pat)):
        with open(f) as fp:
            for r in csv.DictReader(fp):
                if r[metric] not in ("", "nan") and r["image_id"] not in OUTLIER_FOVS:
                    by_fov[r["image_id"]].append(float(r[metric]))
    return {fov: statistics.mean(vs) for fov, vs in by_fov.items()}


def per_seed_means(pat: str, metric: str) -> list[float]:
    out = []
    for f in sorted(glob.glob(pat)):
        with open(f) as fp:
            vals = [float(r[metric]) for r in csv.DictReader(fp)
                    if r[metric] not in ("", "nan") and r["image_id"] not in OUTLIER_FOVS]
        out.append(statistics.mean(vals))
    return out


def per_seed_means_organ(pat: str, metric: str, organ: str) -> list[float]:
    """Mean PQ per seed restricted to FOVs of one organ (outliers excluded)."""
    out = []
    for f in sorted(glob.glob(pat)):
        with open(f) as fp:
            vals = [float(r[metric]) for r in csv.DictReader(fp)
                    if r[metric] not in ("", "nan")
                    and organ_of(r["image_id"]) == organ
                    and r["image_id"] not in OUTLIER_FOVS]
        if vals:
            out.append(statistics.mean(vals))
    return out


def sig_marker(p):
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


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
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    # Per-FOV (seed-averaged) data for Wilcoxon
    fov_pq = {key: per_fov_seed_avg(PATTERNS[key], "PQ") for key, _, _ in CONDITIONS}

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.6), gridspec_kw={"wspace": 0.45})
    x = np.arange(len(CONDITIONS))
    panel_letters = "abcdefghij"

    for panel_i, organ in enumerate(ORGANS):
        ax = axes[panel_i]
        ax.text(-0.22, 1.06, panel_letters[panel_i], transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="left")

        # Build seed-level means restricted to this organ
        means, stds, colors, labels = [], [], [], []
        for key, lbl, color in CONDITIONS:
            vals = per_seed_means_organ(PATTERNS[key], "PQ", organ)
            means.append(np.mean(vals))
            stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            colors.append(color)
            labels.append(lbl)

        # Y axis: dynamic per organ to show effect; use a common-ish range
        all_means = np.array(means)
        all_stds = np.array(stds)
        y_lo = max(0.0, (all_means - all_stds).min() - 0.05)
        y_hi = (all_means + all_stds).max() + 0.10
        ylim = (y_lo, y_hi)

        bars = ax.bar(x, means, yerr=stds, color=colors,
                      edgecolor="black", linewidth=0.6,
                      error_kw=dict(elinewidth=0.7, capsize=2.2, ecolor="black"),
                      width=0.72)
        bars[SIGMMA_IDX].set_edgecolor("black")
        bars[SIGMMA_IDX].set_linewidth(1.2)
        ax.axhline(means[0], color="#888888", linestyle="--", linewidth=0.6, alpha=0.7, zorder=0)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.set_ylim(ylim)
        ax.set_ylabel("PQ (Global)" if panel_i == 0 else "")
        ax.set_title(f"{organ}  (n_FOV={sum(1 for f in fov_pq['sigmma_a01'] if organ_of(f)==organ)})", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="-", linewidth=0.3, color="#cccccc", alpha=0.7, zorder=-1)
        ax.set_axisbelow(True)

        # FOV-paired Wilcoxon: SIGMMA vs each baseline (organ-restricted FOVs)
        organ_fovs = sorted(f for f in fov_pq["sigmma_a01"] if organ_of(f) == organ)
        sig_v = np.array([fov_pq["sigmma_a01"][f] for f in organ_fovs])
        print(f"\n  --- {organ} (n={len(organ_fovs)} FOVs) ---")
        per_baseline_p = []
        for j in range(SIGMMA_IDX):
            key = CONDITIONS[j][0]
            lbl = CONDITIONS[j][1]
            b_v = np.array([fov_pq[key][f] for f in organ_fovs])
            try:
                stat, p = wilcoxon(sig_v, b_v, zero_method="wilcox", alternative="two-sided")
                n_nz = int(np.sum((sig_v - b_v) != 0))
            except Exception:
                stat, p, n_nz = float("nan"), float("nan"), 0
            delta = means[SIGMMA_IDX] - means[j]
            marker = sig_marker(p)
            per_baseline_p.append((j, p, marker, delta))
            print(f"    SIGMMA vs {lbl:12s}  Δ={delta:+.4f}  p={p:.3g}  [{marker}]  n={n_nz}")

        # Stacked brackets for significant SIGMMA-wins
        y_top = ylim[1]
        y_bot = ylim[0]
        span = y_top - y_bot
        significant = [(j, p, m, d) for j, p, m, d in per_baseline_p
                       if m not in ("ns", "n/a") and d > 0]
        significant.sort(key=lambda t: abs(SIGMMA_IDX - t[0]))
        base_y = max(means[i] + stds[i] for i in range(len(CONDITIONS))) + span * 0.025
        stack_h = span * 0.045
        tick_h = span * 0.008
        for stack_i, (j, p, marker, delta) in enumerate(significant):
            bracket_y = base_y + stack_i * stack_h
            x_l, x_r = sorted([j, SIGMMA_IDX])
            ax.plot([x_l, x_l, x_r, x_r],
                    [bracket_y, bracket_y + tick_h, bracket_y + tick_h, bracket_y],
                    color="black", linewidth=0.5, clip_on=False)
            ax.text((x_l + x_r) / 2, bracket_y + tick_h - span * 0.001,
                    marker, ha="center", va="bottom",
                    fontsize=6.5, color="black", fontweight="bold")

    fig.suptitle(
        "Global PQ stratified by organ — SIGMMA α=0.1 vs UNI2-h + 4 contrastive-FT baselines\n"
        "Mean ± SD across 5 seeds; */**/*** = FOV-paired Wilcoxon p<0.05/0.01/0.001",
        fontsize=9, y=1.02,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        out = OUT / f"global_metrics_bars_by_organ.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=300 if ext == "png" else None)
        print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
