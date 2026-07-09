"""Per-tile (per-FOV) AJI & PQ scatter: SIGMMA a01 vs a baseline (default UNI2-h).

Method A values: each tile = mean over the 5 seeds. One dot per test FOV.
Points above the y=x diagonal → SIGMMA wins that tile. Colored by organ.
No outlier filter (all 85 FOVs), step2000 ckpts.

Output: figures/eva_critique/all_gene_panel/per_tile_scatter_vs_<baseline>.{pdf,png}
"""
from __future__ import annotations
import csv, glob, statistics as st, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
OUT = ROOT / "figures" / "eva_critique" / "all_gene_panel"
OUT.mkdir(parents=True, exist_ok=True)

BASELINE = sys.argv[1] if len(sys.argv) > 1 else "uni2h"   # uni2h|bleep|clip|plip|omiclip
BASE_LBL = {"uni2h": "UNI2-h", "bleep": "BLEEP-ft", "clip": "CLIP-ft",
            "plip": "PLIP-ft", "omiclip": "OmiCLIP-ft"}[BASELINE]

ORGAN = {
    "TCGA-50": "Lung", "TCGA-55": "Lung", "TCGA-49": "Lung", "TCGA-73": "Lung", "TCGA-78": "Lung",
    "TCGA-DW": "Kidney", "TCGA-2Z": "Kidney", "TCGA-MH": "Kidney", "TCGA-A8": "Kidney", "TCGA-IZ": "Kidney",
    "TCGA-A2": "Breast", "TCGA-F9": "Breast",
    "TCGA-G9": "Prostate", "TCGA-VP": "Prostate", "TCGA-YL": "Prostate",
    "TCGA-MP": "Prostate", "TCGA-X4": "Prostate",
}
OCOLOR = {"Lung": "#1f77b4", "Kidney": "#2ca02c", "Prostate": "#9467bd", "Breast": "#d62728"}
def organ_of(fov): return ORGAN.get("-".join(fov.split("-")[:2]), "Unknown")

def per_fov(cond, metric):
    by = {}
    for f in sorted(glob.glob(f"{ROOT}/outputs/eva_step2000/{cond}/seed*/instance_metrics_global.csv")):
        for r in csv.DictReader(open(f)):
            if r[metric] in ("", "nan"):
                continue
            by.setdefault(r["image_id"], []).append(float(r[metric]))
    return {k: st.mean(v) for k, v in by.items()}

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), gridspec_kw={"wspace": 0.32})
for ax, metric in zip(axes, ["AJI", "PQ"]):
    sig = per_fov("a01", metric)
    base = per_fov(BASELINE, metric)
    fovs = sorted(set(sig) & set(base))
    xs = np.array([base[f] for f in fovs])      # baseline
    ys = np.array([sig[f] for f in fovs])        # SIGMMA
    for org in ["Lung", "Kidney", "Prostate", "Breast"]:
        idx = [i for i, f in enumerate(fovs) if organ_of(f) == org]
        if idx:
            ax.scatter(xs[idx], ys[idx], s=18, alpha=0.75, c=OCOLOR[org],
                       edgecolors="none", label=org)
    lo = min(xs.min(), ys.min()) - 0.02
    hi = max(xs.max(), ys.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "--", color="#888888", lw=0.8, zorder=0)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    wins = int((ys > xs).sum()); n = len(fovs)
    ax.set_title(f"{metric}   SIGMMA wins {wins}/{n} tiles  (Δ̄={np.mean(ys-xs):+.3f})", fontsize=8)
    ax.set_xlabel(f"{BASE_LBL}  (per-tile {metric})")
    ax.set_ylabel(f"SIGMMA a01  (per-tile {metric})")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if metric == "AJI":
        ax.legend(fontsize=6.5, frameon=False, loc="upper left", handletextpad=0.2)

fig.suptitle(f"H&E → cell segmentation: per-tile SIGMMA vs {BASE_LBL} "
             f"(step2000, seed-avg, n=85)", fontsize=9, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.94])
for ext in ["pdf", "png"]:
    p = OUT / f"per_tile_scatter_vs_{BASELINE}.{ext}"
    fig.savefig(p, bbox_inches="tight", dpi=300 if ext == "png" else None)
    print(f"saved {p}")
