"""Headline EVA figure (2026-06-06 polygon GT), mean±SD bar version.

Three panels — Dice / AJI / PQ. Bar height = mean over 5 seeds (each seed = mean
over 85 TEST FOVs); error bar = SD over seeds (same convention as
plot_global_metrics_bars_step2000.py). Significance = FOV-paired Wilcoxon
(seed-avg per FOV, n=85), SIGMMA vs each condition.

Sources:
  AJI / PQ  -> outputs/eva_polygon_per_fov.csv  (polygon-instance GT)
  Dice      -> per-FOV foreground Dice from test_artifacts.npz (GT-def-independent;
               (prob>0.5) vs (gt>0) semantic foreground — NOT polygon-instance based)

Lineup (left -> right), per user decision 2026-06-06:
  UNI2-h (frozen)            -- ERA1 (swapped into the frozen-baseline row)
  BLEEP/CLIP/PLIP/OmiCLIP-ft -- ERA2
  SIGMMA                     -- ERA2 step2000  (s500 excluded)

Outputs:
  figures/eva_critique/all_gene_panel/polygon_metrics_bars.{pdf,png}
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
# colours = global_metrics_bars scheme.
CONDITIONS = [
    ("ERA1", "UNI2-h(frozen)", "outputs/eva_critique_uni2h",      "UNI2-h",     "#888888"),
    ("ERA2", "BLEEP-FT",       "outputs/eva_step2000/bleep",      "BLEEP-ft",   "#882255"),
    ("ERA2", "CLIP-FT",        "outputs/eva_step2000/clip",       "CLIP-ft",    "#44AA99"),
    ("ERA2", "PLIP-FT",        "outputs/eva_step2000/plip",       "PLIP-ft",    "#DDCC77"),
    ("ERA2", "OmiCLIP-FT",     "outputs/eva_step2000/omiclip",    "OmiCLIP-ft", "#999933"),
    ("ERA2", "SIGMMA s2000",   "outputs/eva_step2000/a01",        "SIGMMA",     "#d62728"),
]


def load_poly():
    """dict[(era,method)] -> {'AJI'/'PQ': {'seed':[per-seed means], 'fov':{iid:seedavg}}}."""
    # key -> seed -> iid -> (aji, pq)
    raw = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader(open(POLY_CSV)):
        raw[(r["era"], r["method"])][r["seed"]][r["image_id"]] = (
            float(r["AJI_poly"]), float(r["PQ_poly"]))
    out = {}
    for key, seeds in raw.items():
        d = {"AJI": {"seed": [], "fov": defaultdict(list)},
             "PQ":  {"seed": [], "fov": defaultdict(list)}}
        for _seed, fovs in seeds.items():
            d["AJI"]["seed"].append(statistics.mean(v[0] for v in fovs.values()))
            d["PQ"]["seed"].append(statistics.mean(v[1] for v in fovs.values()))
            for iid, (a, p) in fovs.items():
                d["AJI"]["fov"][iid].append(a)
                d["PQ"]["fov"][iid].append(p)
        for m in ("AJI", "PQ"):
            d[m]["fov"] = {iid: statistics.mean(v) for iid, v in d[m]["fov"].items()}
        out[key] = d
    return out


def load_dice(npz_root):
    """{'seed':[per-seed means], 'fov':{iid:seedavg}} foreground Dice on TEST."""
    per_fov = defaultdict(list)
    per_seed = []
    for f in sorted(glob.glob(str(ROOT / npz_root / "seed*" / "**" / "test_artifacts.npz"),
                              recursive=True)):
        d = np.load(f, allow_pickle=True)
        ids, prob, gt = d["image_id"], d["prob"].astype(np.float32), d["gt"]
        vals = []
        for i, iid in enumerate(ids):
            p = prob[i] > 0.5
            g = gt[i] > 0
            den = p.sum() + g.sum()
            dice = 1.0 if den == 0 else 2.0 * np.logical_and(p, g).sum() / den
            per_fov[str(iid)].append(float(dice))
            vals.append(float(dice))
        if vals:
            per_seed.append(statistics.mean(vals))
    return {"seed": per_seed, "fov": {iid: statistics.mean(v) for iid, v in per_fov.items()}}


def sig_marker(p):
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def make_plot(data):
    """data[(era,method)][mkey] = {'seed':[...], 'fov':{iid:val}}."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "axes.linewidth": 0.6, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    metrics = [("Dice", "Dice (test, per-FOV)"), ("AJI", "AJI"), ("PQ", "PQ")]
    # AJI/PQ share a unified SPAN (0.20) so an equal absolute difference
    # occupies an equal pixel height. Dice is non-significant on every
    # comparison, so it is shown on a full zero-baseline axis (0–1) where the
    # ~0.01 spread reads flat — i.e. no visual difference where the test finds none.
    FIXED_YLIM = {"Dice": (0.4, 0.8), "AJI": (0.35, 0.55), "PQ": (0.30, 0.50)}
    labels = [c[3] for c in CONDITIONS]
    colors = [c[4] for c in CONDITIONS]
    x = np.arange(len(CONDITIONS))
    SIG = len(CONDITIONS) - 1
    sig_key = (CONDITIONS[SIG][0], CONDITIONS[SIG][1])

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.9),
                             gridspec_kw={"wspace": 0.42})
    panel_letters = "abc"

    for panel_i, (ax, (mkey, mlabel)) in enumerate(zip(axes, metrics)):
        ax.text(-0.26, 1.06, panel_letters[panel_i], transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="left")

        means, stds = [], []
        for era, meth, *_ in CONDITIONS:
            vals = data[(era, meth)][mkey]["seed"]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        means, stds = np.array(means), np.array(stds)

        bars = ax.bar(x, means, yerr=stds, color=colors,
                      edgecolor="black", linewidth=0.6,
                      error_kw=dict(elinewidth=0.7, capsize=2.2, ecolor="black"),
                      width=0.72)
        bars[SIG].set_linewidth(1.2)
        ax.axhline(means[0], color="#888888", linestyle="--",
                   linewidth=0.6, alpha=0.7, zorder=0)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.set_ylabel(mlabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="-", linewidth=0.3,
                color="#cccccc", alpha=0.7, zorder=-1)
        ax.set_axisbelow(True)

        # FOV-paired Wilcoxon: SIGMMA vs each other condition.
        a_dict = data[sig_key][mkey]["fov"]
        print(f"\n  --- {mkey} (FOV-paired Wilcoxon, n=85) ---")
        sig_brackets = []
        for j in range(SIG):
            b_dict = data[(CONDITIONS[j][0], CONDITIONS[j][1])][mkey]["fov"]
            common = sorted(set(a_dict) & set(b_dict))
            a = np.array([a_dict[k] for k in common])
            b = np.array([b_dict[k] for k in common])
            stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            delta = means[SIG] - means[j]
            marker = sig_marker(p)
            print(f"    SIGMMA vs {labels[j]:11s}  Δ={delta:+.4f}  p={p:.3g}  [{marker}]")
            if marker not in ("ns", "n/a") and delta > 0:
                sig_brackets.append((j, marker))

        # ---- fixed, unified-span y-axis + brackets in ABSOLUTE units ----
        ax.set_ylim(*FIXED_YLIM[mkey])
        top_data = float(np.max(means + stds))
        sig_brackets.sort(key=lambda t: abs(SIG - t[0]))  # shortest lowest
        gap, tick_h = 0.0115, 0.003  # absolute (same scale across panels)
        base_y = top_data + 0.008
        for stack_i, (j, marker) in enumerate(sig_brackets):
            by = base_y + stack_i * gap
            x_l, x_r = sorted([x[j], x[SIG]])
            ax.plot([x_l, x_l, x_r, x_r],
                    [by, by + tick_h, by + tick_h, by],
                    color="black", linewidth=0.5, clip_on=False)
            ax.text((x_l + x_r) / 2, by + tick_h, marker, ha="center",
                    va="bottom", fontsize=6.5, fontweight="bold")

    fig.suptitle("H&E → cell segmentation   ·   mean ± SD across 5 seeds; dashed line = UNI2-h",
                 fontsize=9, y=1.0)

    caption = (
        "Bars: mean ± s.d. across 5 random seeds (each seed = mean over 85 MoNuSAC test FOVs). "
        "Significance: two-sided Wilcoxon signed-rank test on seed-averaged per-FOV values "
        "(FOV-paired, n = 85), SIGMMA vs each condition; only significant SIGMMA gains are bracketed "
        "(*p<0.05, **p<0.01, ***p<0.001).\n"
        "Dice = foreground overlap (prob>0.5 vs any annotated cell); AJI and PQ use polygon-instance GT "
        "(one annotated polygon = one cell). UNI2-h is the frozen backbone (ERA1); SIGMMA at step 2000."
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.text(0.5, -0.04, caption, ha="center", va="top", fontsize=6.2,
             color="#333333", linespacing=1.4)
    for ext in ["pdf", "png"]:
        out = OUT / f"polygon_metrics_bars.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=300 if ext == "png" else None)
        print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    poly = load_poly()
    data = {}
    for era, meth, npz_root, lbl, _ in CONDITIONS:
        key = (era, meth)
        data[key] = {"AJI": poly[key]["AJI"], "PQ": poly[key]["PQ"],
                     "Dice": load_dice(npz_root)}
        d = data[key]
        print(f"  {era} {meth:16s}  Dice {np.mean(d['Dice']['seed']):.4f}  "
              f"AJI {np.mean(d['AJI']['seed']):.4f}  PQ {np.mean(d['PQ']['seed']):.4f}")
    make_plot(data)
