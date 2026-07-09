"""Nature-style bar chart: GLOBAL (all FOVs) Dice / AJI / PQ across all
conditions on the STEP2000 full-rerun ckpts (svg0529 corrected pipeline), N=5 seeds.

Same script/methodology as plot_global_metrics_bars.py — only the input
checkpoint paths (outputs/eva_step2000/<cond>/seed*/) and the figure output
directory differ.

Outputs:
  figures/eva_critique/all_gene_panel/global_metrics_bars.{pdf,png}

Conditions (left → right on x-axis):
  UNI2-h baseline
  BLEEP/CLIP/PLIP/OmiCLIP (UNI2-h finetuned)
  SIGMMA α=0.1
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
OUT = ROOT / "figures" / "eva_critique" / "all_gene_panel"
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

# Outlier filter DISABLED per user decision (2026-06-03): use ALL 85 test FOVs.
# (Previously excluded 5 n_cells ≤ 2 FOVs; kept here documented but empty so the
#  full test set is reported.)
OUTLIER_FOVS = set()


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


def collect_test_dice(npz_pattern):
    """Per-FOV foreground Dice on the TEST set, from test_artifacts.npz.

    Method A: pred = (prob > 0.5) vs gt (>0 = any cell). Returns
      (per_seed_means, per_fov_seedavg)
    where per_seed_means = list[N seeds] (each = mean Dice over FOVs, for the bar
    height/SD) and per_fov_seedavg = dict[image_id] -> mean-over-seeds (for the
    FOV-paired Wilcoxon). Outlier FOVs excluded — matches collect_inst/per_fov.
    """
    per_fov = {}
    per_seed_means = []
    for f in sorted(glob.glob(npz_pattern)):
        d = np.load(f, allow_pickle=True)
        ids, prob, gt = d["image_id"], d["prob"].astype(np.float32), d["gt"]
        seed_vals = []
        for i, iid in enumerate(ids):
            iid = str(iid)
            if iid in OUTLIER_FOVS:
                continue
            p = prob[i] > 0.5
            g = gt[i] > 0
            denom = p.sum() + g.sum()
            dice = 1.0 if denom == 0 else 2.0 * np.logical_and(p, g).sum() / denom
            per_fov.setdefault(iid, []).append(float(dice))
            seed_vals.append(float(dice))
        if seed_vals:
            per_seed_means.append(statistics.mean(seed_vals))
    fov_seedavg = {fov: statistics.mean(vs) for fov, vs in per_fov.items()}
    return per_seed_means, fov_seedavg


def gather():
    """Return dict[cond_key] = dict[metric] = list[N seeds] (seed-level means)
    and dict[cond_key] = dict[metric+'_fov'] = dict[fov]->mean (FOV-level for paired test)."""
    R = str(ROOT)
    data = {}

    # (cond_key, output_dir, art_subdir) — art_subdir holds convdec_image/test_artifacts.npz
    LAYOUT = [
        ("uni2h",      "uni2h",   "uni"),
        ("bleep",      "bleep",   "bleep_uni_ft"),
        ("clip",       "clip",    "clip_uni_ft"),
        ("plip",       "plip",    "plip_uni_ft"),
        ("omiclip",    "omiclip", "omiclip_uni_ft"),
        ("sigmma_a01", "a01",     "sigmma_shared_lora"),
    ]
    for key, outdir, art in LAYOUT:
        inst_pat = f"{R}/outputs/eva_step2000/{outdir}/seed*/instance_metrics_global.csv"
        # Dice: per-FOV foreground Dice on TEST (Method A), NOT validation best_val_dice.
        npz_pat = f"{R}/outputs/eva_step2000/{outdir}/seed*/{art}/convdec_image/test_artifacts.npz"
        dice_seed, dice_fov = collect_test_dice(npz_pat)
        data[key] = {
            "Dice":     dice_seed,                          # seed-level means → bar height/SD
            "Dice_fov": dice_fov,                           # FOV seed-avg → FOV-paired test
            "AJI":      collect_inst(inst_pat, "AJI"),
            "PQ":       collect_inst(inst_pat, "PQ"),
            "AJI_fov":  collect_inst_per_fov(inst_pat, "AJI"),
            "PQ_fov":   collect_inst_per_fov(inst_pat, "PQ"),
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

    metrics = [("Dice", "Dice (test, per-FOV)", (0.620, 0.700)),
               ("PQ",   "PQ (Global)",          (0.330, 0.395))]

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
        # Method A for ALL metrics (Dice/AJI/PQ): average across seeds per FOV,
        # then paired Wilcoxon over the per-FOV vectors (outliers excluded → n=80).
        # Dice is per-FOV foreground Dice on the TEST set (not validation), so it
        # shares the exact same pairing unit as AJI/PQ.
        sigmma_key = CONDITIONS[SIGMMA_IDX][0]
        fov_key = f"{mkey}_fov"
        unit = "FOV" if fov_key in data[sigmma_key] else "seed"
        n_fov = len(data[sigmma_key].get(fov_key, [])) if unit == "FOV" else 5

        print(f"\n  --- {mkey} (paired Wilcoxon, n={n_fov} {unit}s) ---")
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
