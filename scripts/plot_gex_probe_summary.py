"""Plot per-gene Pearson comparison across input_reprs for 2D GEX probing.

Reads outputs/gex_2d/results/*.json and writes a 3-panel figure to
figures/gex_2d/per_gene_pearson_summary.{png,pdf}.

Panel A : violin + strip of per-gene Pearson across 5 input_reprs.
Panel B : paired delta (concat − uni) per gene, sorted; highlights ties / wins.
Panel C : summary table — median, IQR, per-bin R², Moran's I diff.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPRS = ["uni", "micro", "meso", "macro", "concat"]
COLORS = {
    "uni":    "#7f7f7f",
    "micro":  "#1f77b4",
    "meso":   "#2ca02c",
    "macro":  "#ff7f0e",
    "concat": "#d62728",
}

RESULTS_DIR = Path("outputs/gex_2d/results")
OUT_DIR = Path("figures/gex_2d")


def load_all() -> dict:
    out = {}
    for r in REPRS:
        with open(RESULTS_DIR / f"{r}.json") as f:
            out[r] = json.load(f)
    return out


def main():
    data = load_all()
    genes = list(data["uni"]["per_gene_pearson"].keys())
    pearson = {r: np.array([data[r]["per_gene_pearson"][g] for g in genes]) for r in REPRS}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])

    # ----- Panel A : violin per repr -----
    axA = fig.add_subplot(gs[0, 0])
    parts = axA.violinplot(
        [pearson[r] for r in REPRS],
        positions=range(len(REPRS)),
        showmedians=True,
        widths=0.8,
    )
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(COLORS[REPRS[i]])
        body.set_alpha(0.55)
        body.set_edgecolor("black")
    for k in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[k].set_color("black")
    # strip overlay
    rng = np.random.default_rng(0)
    for i, r in enumerate(REPRS):
        x = np.full_like(pearson[r], i, dtype=float) + rng.uniform(-0.08, 0.08, len(pearson[r]))
        axA.scatter(x, pearson[r], s=12, color=COLORS[r], edgecolor="black", linewidth=0.3, alpha=0.85, zorder=3)
    axA.set_xticks(range(len(REPRS)))
    axA.set_xticklabels(REPRS)
    axA.set_ylabel("per-gene Pearson r")
    axA.set_title("A. Per-gene Pearson across 50 HVGs (one slide, 54 val FOVs)")
    axA.axhline(0.10, ls="--", color="grey", lw=0.8, label="smoke pass = 0.10")
    axA.legend(loc="lower right", fontsize=8)
    axA.grid(axis="y", alpha=0.3)

    # ----- Panel B : paired delta concat − uni -----
    axB = fig.add_subplot(gs[0, 1])
    delta = pearson["concat"] - pearson["uni"]
    order = np.argsort(delta)
    genes_sorted = [genes[i] for i in order]
    bar_colors = ["#d62728" if d > 0 else "#1f77b4" for d in delta[order]]
    axB.barh(np.arange(len(genes_sorted)), delta[order], color=bar_colors, edgecolor="black", linewidth=0.3)
    axB.set_yticks(np.arange(len(genes_sorted)))
    axB.set_yticklabels(genes_sorted, fontsize=6)
    axB.axvline(0, color="black", lw=0.6)
    axB.set_xlabel("Δ Pearson  (concat − uni)")
    n_win = int((delta > 0).sum())
    n_loss = int((delta < 0).sum())
    axB.set_title(f"B. Per-gene Δ — wins {n_win}/{len(genes)}  median Δ={np.median(delta):+.4f}")
    axB.grid(axis="x", alpha=0.3)

    # ----- Panel C : summary table -----
    axC = fig.add_subplot(gs[1, :])
    axC.axis("off")
    header = ["input_repr", "median Pearson", "IQR (Q25, Q75)", "per-bin R² pooled", "median |ΔMoran's I|"]
    rows = []
    for r in REPRS:
        d = data[r]
        rows.append([
            r,
            f"{d['median_per_gene_pearson']:.4f}",
            f"({d['iqr_per_gene_pearson'][0]:.3f}, {d['iqr_per_gene_pearson'][1]:.3f})",
            f"{d['per_bin_r2_pooled']:.4f}",
            f"{d['median_morans_i_diff']:.3f}",
        ])
    table = axC.table(
        cellText=rows, colLabels=header, loc="center", cellLoc="center", colLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    # bold header row
    for j in range(len(header)):
        table[0, j].set_text_props(weight="bold")
    # color rows by repr
    for i, r in enumerate(REPRS, start=1):
        for j in range(len(header)):
            table[i, j].set_facecolor(COLORS[r])
            table[i, j].set_alpha(0.18)
    # delta annotation footer
    axC.text(
        0.5, -0.05,
        f"Reference: uni median = {data['uni']['median_per_gene_pearson']:.4f}   "
        f"Δ(concat − uni) = {data['concat']['median_per_gene_pearson'] - data['uni']['median_per_gene_pearson']:+.4f}   "
        f"(weak-signal threshold ~ +0.05 per memory)",
        ha="center", va="top", transform=axC.transAxes, fontsize=10, color="#444"
    )
    axC.set_title("C. Summary metrics", y=0.95)

    fig.suptitle(
        "2D GEX probing — single-slide smoke (BK24-SKI-27-FO-01-S3, 54 val FOVs)",
        fontsize=13, weight="bold"
    )

    png = OUT_DIR / "per_gene_pearson_summary.png"
    pdf = OUT_DIR / "per_gene_pearson_summary.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"saved {png}")
    print(f"saved {pdf}")


if __name__ == "__main__":
    main()
