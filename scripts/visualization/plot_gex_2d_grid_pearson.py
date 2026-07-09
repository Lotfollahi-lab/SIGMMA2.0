"""Line plot: median per-gene Pearson vs ST bin granularity for H&E→GEX regressor
baselines (ST-Net, HisToGene, DeepPT).

x-axis is ordered coarse→fine (grid 1, 2, 7, 14) so the curve falls left→right:
finer physical bins (128 µm FOV → 9 µm) are harder to predict from H&E. Each tick
is annotated with both the grid count (K per FOV side) and the bin size in µm.

Reads the val-set JSONs written by ``predict_baseline_gex_2d.py`` (one dir per
grid, ``baselines_step950[_g{tg}]``). Pearson only — see resolution_curve for R²/MSE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

# H&E FOV physical size: fov_size_px * um_per_px = 256 * 0.5 = 128 µm.
FOV_UM = 128.0

# Distinct hues for the three regressor baselines (not in the locked method
# palette, which covers UNI/BLEEP/CLIP/PLIP/OmiCLIP/SIGMMA only).
STYLE = {
    "ST-Net":    {"color": "#1f77b4", "marker": "o"},
    "HisToGene": {"color": "#ff7f0e", "marker": "s"},
    "DeepPT":    {"color": "#2ca02c", "marker": "^"},
}

# (baseline_key, display_name)
BASELINES = [("stnet", "ST-Net"), ("histogene", "HisToGene"), ("deeppt", "DeepPT")]

# x-axis order: coarse → fine. grid 1 = FOV-level (fov.json), others = bin.json.
GRIDS = [1, 2, 7, 14]


def grid_dir(base: str, tg: int) -> Path:
    if tg == 1:
        return Path(base) / "baselines_step950"
    if tg == 14:
        return Path(base) / "baselines_step950_g14"
    return Path(base) / f"baselines_step950_g{tg}"


def load_pearson(base: str, baseline: str, tg: int) -> float | None:
    d = grid_dir(base, tg)
    fname = f"{baseline}_fov.json" if tg == 1 else f"{baseline}_bin.json"
    p = d / fname
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("median_per_gene_pearson")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_base", default="outputs/gex_2d/results")
    ap.add_argument("--out", default="figures/gex_2d/grid_pearson_baselines")
    args = ap.parse_args()

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
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    x = list(range(len(GRIDS)))  # evenly-spaced categorical positions
    fig, ax = plt.subplots(figsize=(4.2, 3.2))

    for key, name in BASELINES:
        ys, xs = [], []
        for xi, tg in zip(x, GRIDS):
            v = load_pearson(args.results_base, key, tg)
            if v is not None:
                xs.append(xi)
                ys.append(v)
        st = STYLE[name]
        ax.plot(xs, ys, color=st["color"], marker=st["marker"], markersize=5,
                linewidth=1.8, label=name)

    # Tick labels: grid count + physical bin size (uniform 2 lines).
    labels = []
    for tg in GRIDS:
        gtxt = "1 (FOV)" if tg == 1 else f"{tg}"
        um = f"{FOV_UM / tg:.0f} µm" if tg == 1 else f"{FOV_UM / tg:.1f} µm"
        labels.append(f"{gtxt}\n{um}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylabel("Median per-gene Pearson")
    ax.set_title("H&E → ST grid: prediction vs bin granularity")
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.set_ylim(0.3, 0.95)
    ax.legend(loc="upper right", frameon=False)

    # Direction cues at the two bottom corners (below the tick labels).
    ax.annotate("← coarser bins", xy=(0.0, -0.26), xycoords="axes fraction",
                ha="left", va="top", fontsize=7, color="#555555")
    ax.annotate("finer bins →", xy=(1.0, -0.26), xycoords="axes fraction",
                ha="right", va="top", fontsize=7, color="#555555")
    ax.set_xlabel("ST grid (bins per FOV side) / bin size", labelpad=24)

    fig.subplots_adjust(bottom=0.30)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out.with_suffix('.png')} and .pdf", flush=True)


if __name__ == "__main__":
    main()
