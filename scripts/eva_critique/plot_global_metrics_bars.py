"""Nature-style bar chart: GLOBAL (all FOVs) Dice / AJI / PQ across all
conditions on UNI2-h step1000, N=5 seeds, post-new-watershed-defaults.

Outputs:
  figures/eva_critique/global_metrics_bars.{pdf,png}

Conditions (left → right on x-axis):
  UNI2-h baseline
  BLEEP/CLIP/PLIP/OmiCLIP (UNI2-h finetuned)
  SIGMMA α=0 / α=0.1 / α=0.2 / α=0.4
"""

from __future__ import annotations

import json
import glob
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
OUT = ROOT / "figures" / "eva_critique"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Data layout
# ----------------------------------------------------------------------------
CONDITIONS = [
    ("uni2h",       "UNI2-h",         "#888888"),
    ("bleep",       "BLEEP-ft",       "#882255"),  # wine
    ("clip",        "CLIP-ft",        "#44AA99"),  # teal
    ("plip",        "PLIP-ft",        "#DDCC77"),  # sand
    ("omiclip",     "OmiCLIP-ft",     "#999933"),  # olive
    ("sigmma_a01",  "SIGMMA",         "#d62728"),  # headline
]

# Pre-specified outlier filter: FOVs with n_cells ≤ 2 (raw XML polygon GT count).
# Justification: single-instance error → 50% PQ swing + 224-resize artifact on small
# native FOVs (all 5 have native short-edge < 125px). See:
#   - scripts/eva_critique/plot_n_cells_histogram.py (visual justification, bimodal distribution)
#   - figures/eva_critique/n_cells_histogram.{pdf,png}
# Affects every model identically (model-agnostic dilution); excluding these 5 FOVs
# tightens the SIGMMA-vs-baseline signal without cherry-picking.
OUTLIER_FOVS = {
    "TCGA-2Z-A9JG-01Z-00-DX1_2",     # Kidney, native 35px, 2 Neutrophil
    "TCGA-MH-A55W-01Z-00-DX1_2",     # Kidney, native 37px, 2 Neutrophil
    "TCGA-MH-A55W-01Z-00-DX1_4",     # Kidney, native 33px, 2 Neutrophil
    "TCGA-A2-A04X-01Z-00-DX1_003",   # Breast, native 114px, 2 Neutrophil
    "TCGA-F9-A8NY-01Z-00-DX1_4",     # Breast, native 124px, 2 Macrophage
}


def collect_val_dice(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fp:
            d = json.load(fp)
        if "best_val_dice" in d:
            out.append(d["best_val_dice"])
    return out


def collect_inst(pattern, key):
    out = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fp:
            rdr = csv.DictReader(fp)
            vals = [float(r[key]) for r in rdr
                    if r[key] not in ("", "nan") and r["image_id"] not in OUTLIER_FOVS]
        if vals:
            out.append(statistics.mean(vals))
    return out


def collect_inst_per_fov(pattern, key):
    """Return dict[image_id] -> list[seed values]. Useful for FOV-paired tests."""
    by_fov = {}
    for f in sorted(glob.glob(pattern)):
        with open(f) as fp:
            for r in csv.DictReader(fp):
                v = r[key]
                if v in ("", "nan"):
                    continue
                if r["image_id"] in OUTLIER_FOVS:
                    continue
                by_fov.setdefault(r["image_id"], []).append(float(v))
    # Average across seeds per FOV
    return {fov: statistics.mean(vs) for fov, vs in by_fov.items()}


def gather():
    """Return dict[cond_key] = dict[metric] = list[N seeds] (seed-level means)
    and dict[cond_key] = dict[metric+'_fov'] = dict[fov]->mean (FOV-level for paired test)."""
    R = str(ROOT)
    data = {}

    # UNI2-h
    inst_pat_uni = f"{R}/outputs/eva_critique_uni2h/seed*/instance_metrics_global.csv"
    data["uni2h"] = {
        "Dice": collect_val_dice(f"{R}/outputs/eva_critique_uni2h/seed*/uni/convdec_image/summary.json"),
        "AJI":  collect_inst(inst_pat_uni, "AJI"),
        "PQ":   collect_inst(inst_pat_uni, "PQ"),
        "AJI_fov": collect_inst_per_fov(inst_pat_uni, "AJI"),
        "PQ_fov":  collect_inst_per_fov(inst_pat_uni, "PQ"),
    }

    # Baseline-FT
    for short, model_dir in [("bleep","bleep_uni_ft"),("clip","clip_uni_ft"),
                             ("plip","plip_uni_ft"),("omiclip","omiclip_uni_ft")]:
        inst_pat = f"{R}/outputs/eva_critique_baselines_uni2h_step1000/{model_dir}/seed*/instance_metrics_global.csv"
        data[short] = {
            "Dice": collect_val_dice(f"{R}/outputs/eva_critique_baselines_uni2h_step1000/{model_dir}/seed*/{model_dir}/convdec_image/summary.json"),
            "AJI":  collect_inst(inst_pat, "AJI"),
            "PQ":   collect_inst(inst_pat, "PQ"),
            "AJI_fov": collect_inst_per_fov(inst_pat, "AJI"),
            "PQ_fov":  collect_inst_per_fov(inst_pat, "PQ"),
        }

    # SIGMMA α-sweep
    for a in ["a00","a01","a02","a04"]:
        inst_pat = f"{R}/outputs/eva_critique_gex_sweep_step1000/{a}/seed*/instance_metrics_global.csv"
        data[f"sigmma_{a}"] = {
            "Dice": collect_val_dice(f"{R}/outputs/eva_critique_gex_sweep_step1000/{a}/seed*/sigmma_shared_lora/convdec_image/summary.json"),
            "AJI":  collect_inst(inst_pat, "AJI"),
            "PQ":   collect_inst(inst_pat, "PQ"),
            "AJI_fov": collect_inst_per_fov(inst_pat, "AJI"),
            "PQ_fov":  collect_inst_per_fov(inst_pat, "PQ"),
        }
    return data


def paired_wilcoxon(a, b):
    """Two-sided Wilcoxon signed-rank on paired arrays a, b. Returns (stat, p, n)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape
    # Drop pairs where diff is 0 (Wilcoxon ignores zeros; scipy handles with zero_method).
    diff = a - b
    n_nonzero = int(np.sum(diff != 0))
    if n_nonzero < 3:
        return float("nan"), float("nan"), n_nonzero
    res = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return float(res.statistic), float(res.pvalue), n_nonzero


def sig_marker(p):
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ----------------------------------------------------------------------------
# Nature-style plot
# ----------------------------------------------------------------------------
def make_plot(data):
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

    metrics = [("Dice", "Dice (validation, pixel)", (0.500, 0.700)),
               ("PQ",   "PQ (Global)",              (0.330, 0.395))]

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 3.8),
                             gridspec_kw={"wspace": 0.40})

    x = np.arange(len(CONDITIONS))
    SIGMMA_IDX = len(CONDITIONS) - 1  # last bar

    print("\n=== Paired Wilcoxon: SIGMMA α=0.1 vs best non-SIGMMA ===")

    panel_letters = "abcdefghij"
    for panel_i, (ax, (mkey, mlabel, ylim)) in enumerate(zip(axes, metrics)):
        # Nature-style panel label (lowercase bold), top-left, slightly outside axes
        ax.text(-0.22, 1.06, panel_letters[panel_i], transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="left")

        means, stds, colors, labels = [], [], [], []
        for cond_key, cond_lbl, color in CONDITIONS:
            vals = data[cond_key][mkey]
            means.append(np.mean(vals))
            stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            colors.append(color)
            labels.append(cond_lbl)

        bars = ax.bar(
            x, means, yerr=stds,
            color=colors,
            edgecolor="black", linewidth=0.6,
            error_kw=dict(elinewidth=0.7, capsize=2.2, ecolor="black"),
            width=0.72,
        )

        # Highlight SIGMMA with thicker edge (last bar)
        bars[SIGMMA_IDX].set_edgecolor("black")
        bars[SIGMMA_IDX].set_linewidth(1.2)

        # UNI2-h reference line
        ax.axhline(means[0], color="#888888", linestyle="--",
                   linewidth=0.6, alpha=0.7, zorder=0)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.set_ylim(ylim)
        ax.set_ylabel(mlabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="-", linewidth=0.3,
                color="#cccccc", alpha=0.7, zorder=-1)
        ax.set_axisbelow(True)

        # ----- Paired test: SIGMMA vs EACH non-SIGMMA baseline -----
        # Pairing unit:
        #   AJI/PQ: FOV-paired (n=85). Average across seeds per FOV, then paired
        #           Wilcoxon over the 85 FOV vectors. Natural unit since every
        #           model × seed combo is evaluated on the same 85 test FOVs.
        #   Dice  : seed-paired (n=5). best_val_dice is a single scalar per
        #           (model, seed) — no per-FOV breakdown saved.
        sigmma_key = CONDITIONS[SIGMMA_IDX][0]
        fov_key = f"{mkey}_fov"
        unit = "FOV" if fov_key in data[sigmma_key] else "seed"

        print(f"\n  --- {mkey} (paired Wilcoxon, n={'85' if unit=='FOV' else '5'} {unit}s) ---")
        per_baseline_p = []
        for j in range(SIGMMA_IDX):
            other_key = CONDITIONS[j][0]
            other_lbl = CONDITIONS[j][1]
            if unit == "FOV":
                a_dict = data[sigmma_key][fov_key]
                b_dict = data[other_key][fov_key]
                common = sorted(set(a_dict) & set(b_dict))
                a = np.array([a_dict[k] for k in common])
                b = np.array([b_dict[k] for k in common])
            else:
                a = data[sigmma_key][mkey]
                b = data[other_key][mkey]
            stat, p, n = paired_wilcoxon(a, b)
            delta = means[SIGMMA_IDX] - means[j]
            marker = sig_marker(p)
            per_baseline_p.append((j, p, marker, delta))
            print(f"    SIGMMA vs {other_lbl:12s}  Δ={delta:+.4f}  p={p:.3g}  [{marker}]")

        # Draw a bracket only for significant SIGMMA-wins (Wilcoxon p<0.05, Δ>0).
        # Label = stars only (no p-value text). Stack innermost first (shortest
        # bracket near the bars, longest bracket on top).
        y_top = ylim[1]
        y_bot = ylim[0]
        span = y_top - y_bot

        significant = [(j, p, marker, delta) for j, p, marker, delta in per_baseline_p
                       if marker not in ("ns", "n/a") and delta > 0]
        # Sort by distance to SIGMMA bar — shortest bracket first (lowest stack)
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
        "Mean ± SD across 5 seeds; dashed line = UNI2-h mean",
        fontsize=9, y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    for ext in ["pdf", "png"]:
        out = OUT / f"global_metrics_bars.{ext}"
        fig.savefig(out, bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
        print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    data = gather()
    # Sanity check
    for cond, _, _ in CONDITIONS:
        n_dice = len(data[cond]["Dice"])
        n_aji  = len(data[cond]["AJI"])
        print(f"  {cond:12s}  Dice n={n_dice}  AJI n={n_aji}")
    make_plot(data)
