"""Headline EVA figure (2026-06-06 polygon GT), boxplot version.

Three panels — Dice / AJI / PQ. Distribution unit = per-FOV value, averaged
across the 5 seeds (n=85 TEST FOVs), which is also the pairing unit for the
significance test.

Sources:
  AJI / PQ  -> outputs/eva_polygon_per_fov.csv  (polygon-instance GT)
  Dice      -> per-FOV foreground Dice from test_artifacts.npz (GT-def-independent)

Lineup (left -> right), per user decision 2026-06-06:
  UNI2-h (frozen)            -- ERA1 (swapped into the frozen-baseline row)
  BLEEP/CLIP/PLIP/OmiCLIP-ft -- ERA2
  SIGMMA                     -- ERA2 step2000  (s500 excluded)

Significance brackets = FOV-paired Wilcoxon (SIGMMA vs each condition), drawn
only when p<0.05 and SIGMMA's median is higher.

Outputs:
  figures/eva_critique/all_gene_panel/polygon_metrics_box.{pdf,png}
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
POLY_CSV = ROOT / "outputs" / "eva_polygon_per_fov.csv"
OUT = ROOT / "figures" / "eva_critique" / "all_gene_panel"
OUT.mkdir(parents=True, exist_ok=True)

# (era, method-in-poly-csv, npz-root-for-Dice, display label, colour)
CONDITIONS = [
    ("ERA1", "UNI2-h(frozen)", "outputs/eva_critique_uni2h",      "UNI2-h",     "#888888"),
    ("ERA2", "BLEEP-FT",       "outputs/eva_step2000/bleep",      "BLEEP-ft",   "#882255"),
    ("ERA2", "CLIP-FT",        "outputs/eva_step2000/clip",       "CLIP-ft",    "#44AA99"),
    ("ERA2", "PLIP-FT",        "outputs/eva_step2000/plip",       "PLIP-ft",    "#DDCC77"),
    ("ERA2", "OmiCLIP-FT",     "outputs/eva_step2000/omiclip",    "OmiCLIP-ft", "#999933"),
    ("ERA2", "SIGMMA s2000",   "outputs/eva_step2000/a01",        "SIGMMA",     "#d62728"),
]


def load_poly_fov():
    """dict[(era,method)] -> {'AJI': {iid: seedavg}, 'PQ': {iid: seedavg}}."""
    acc = defaultdict(lambda: defaultdict(lambda: [[], []]))  # key -> iid -> [aji[],pq[]]
    for r in csv.DictReader(open(POLY_CSV)):
        a = acc[(r["era"], r["method"])][r["image_id"]]
        a[0].append(float(r["AJI_poly"]))
        a[1].append(float(r["PQ_poly"]))
    out = {}
    for key, fovs in acc.items():
        out[key] = {
            "AJI": {iid: statistics.mean(v[0]) for iid, v in fovs.items()},
            "PQ": {iid: statistics.mean(v[1]) for iid, v in fovs.items()},
        }
    return out


def load_dice_fov(npz_root):
    """dict[iid] -> seed-avg foreground Dice on the TEST set."""
    per_fov = defaultdict(list)
    for f in sorted(glob.glob(str(ROOT / npz_root / "seed*" / "**" / "test_artifacts.npz"),
                              recursive=True)):
        d = np.load(f, allow_pickle=True)
        ids, prob, gt = d["image_id"], d["prob"].astype(np.float32), d["gt"]
        for i, iid in enumerate(ids):
            p = prob[i] > 0.5
            g = gt[i] > 0
            den = p.sum() + g.sum()
            dice = 1.0 if den == 0 else 2.0 * np.logical_and(p, g).sum() / den
            per_fov[str(iid)].append(float(dice))
    return {iid: statistics.mean(v) for iid, v in per_fov.items()}


def sig_marker(p):
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def make_plot(metric_data):
    """metric_data[mkey] = list (over CONDITIONS) of dict[iid]->value."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "axes.linewidth": 0.6, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    metrics = [("Dice", "Dice (test, per-FOV)"),
               ("AJI",  "AJI"),
               ("PQ",   "PQ")]
    labels = [c[3] for c in CONDITIONS]
    colors = [c[4] for c in CONDITIONS]
    x = np.arange(1, len(CONDITIONS) + 1)
    SIG = len(CONDITIONS) - 1  # SIGMMA index (0-based)

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.9),
                             gridspec_kw={"wspace": 0.42})
    panel_letters = "abc"

    for panel_i, (ax, (mkey, mlabel)) in enumerate(zip(axes, metrics)):
        ax.text(-0.26, 1.06, panel_letters[panel_i], transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="left")

        fov_dicts = metric_data[mkey]
        # Common FOV order across all conditions for a fair paired comparison.
        common = sorted(set.intersection(*[set(d) for d in fov_dicts]))
        series = [np.array([d[k] for k in common]) for d in fov_dicts]

        bp = ax.boxplot(series, positions=x, widths=0.62, patch_artist=True,
                        showfliers=True,
                        flierprops=dict(marker="o", markersize=2.0,
                                        markerfacecolor="black", markeredgecolor="none",
                                        alpha=0.4),
                        medianprops=dict(color="black", linewidth=1.0),
                        whiskerprops=dict(color="black", linewidth=0.6),
                        capprops=dict(color="black", linewidth=0.6),
                        boxprops=dict(linewidth=0.6))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
        bp["boxes"][SIG].set_linewidth(1.3)

        # Reference line at UNI2-h median.
        ax.axhline(np.median(series[0]), color="#888888", linestyle="--",
                   linewidth=0.6, alpha=0.7, zorder=0)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.set_ylabel(mlabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="-", linewidth=0.3,
                color="#cccccc", alpha=0.7, zorder=-1)
        ax.set_axisbelow(True)

        # ---- FOV-paired Wilcoxon: SIGMMA vs each other condition ----
        a = series[SIG]
        print(f"\n  --- {mkey} (FOV-paired Wilcoxon, n={len(common)}) ---")
        sig_brackets = []
        for j in range(SIG):
            b = series[j]
            stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            d_med = float(np.median(a) - np.median(b))
            marker = sig_marker(p)
            print(f"    SIGMMA vs {labels[j]:11s}  Δmed={d_med:+.4f}  p={p:.3g}  [{marker}]")
            if marker not in ("ns", "n/a") and d_med > 0:
                sig_brackets.append((j, marker))

        # ---- dynamic y-limits + clean nested brackets (shortest lowest) ----
        data_top = max(s.max() for s in series)
        data_bot = min(s.min() for s in series)
        span = data_top - data_bot
        # Sort so the WIDEST bracket sits on top, shortest just above the boxes.
        sig_brackets.sort(key=lambda t: abs(SIG - t[0]))
        n_b = len(sig_brackets)
        gap = span * 0.06
        base_y = data_top + span * 0.04
        tick_h = span * 0.012
        for stack_i, (j, marker) in enumerate(sig_brackets):
            by = base_y + stack_i * gap
            x_l, x_r = sorted([x[j], x[SIG]])
            ax.plot([x_l, x_l, x_r, x_r],
                    [by, by + tick_h, by + tick_h, by],
                    color="black", linewidth=0.5, clip_on=False)
            ax.text((x_l + x_r) / 2, by + tick_h, marker, ha="center",
                    va="bottom", fontsize=6.5, fontweight="bold")
        y_top = base_y + max(n_b, 1) * gap
        ax.set_ylim(data_bot - span * 0.05, y_top + span * 0.02)

    fig.suptitle("H&E → cell segmentation   ·   per-FOV, seed-averaged (n=85)",
                 fontsize=9, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ["pdf", "png"]:
        out = OUT / f"polygon_metrics_box.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=300 if ext == "png" else None)
        print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    poly = load_poly_fov()
    metric_data = {"Dice": [], "AJI": [], "PQ": []}
    for era, meth, npz_root, lbl, _ in CONDITIONS:
        metric_data["AJI"].append(poly[(era, meth)]["AJI"])
        metric_data["PQ"].append(poly[(era, meth)]["PQ"])
        metric_data["Dice"].append(load_dice_fov(npz_root))
        d = metric_data
        print(f"  {era} {meth:16s}  Dice {np.median(list(d['Dice'][-1].values())):.4f}  "
              f"AJI {np.median(list(d['AJI'][-1].values())):.4f}  "
              f"PQ {np.median(list(d['PQ'][-1].values())):.4f}  "
              f"(nFOV Dice={len(d['Dice'][-1])}, AJI={len(d['AJI'][-1])})")
    make_plot(metric_data)
