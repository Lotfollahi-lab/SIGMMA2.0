"""Bar plot of 2D GEX prediction (per-gene Pearson) for UNI + SIGMMA scales.

Note: as of 2026-05-12 this experiment is single-slide only (smoke); spread is
across the n_genes evaluated genes on that slide, not across slides.
Cross-slide rerun is pending (see memory: project_gex_2d_followup_todo).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

METHODS = ["uni", "micro", "meso", "macro"]
METHOD_LABEL = {"uni": "UNI", "micro": "micro", "meso": "meso", "macro": "macro"}
METHOD_COLOR = {
    "uni":   "#888888",
    "micro": "#d62728",
    "meso":  "#2ca02c",
    "macro": "#1f77b4",
}

NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=Path,
                    default=Path("outputs/gex_2d/disp_g14/results"))
    ap.add_argument("--out", type=Path,
                    default=Path("figures/gex_2d/disp_g14/bars_per_gene_pearson.png"))
    args = ap.parse_args()

    payload = {}
    for m in METHODS:
        d = json.loads((args.results_dir / f"{m}.json").read_text())
        payload[m] = {
            "per_gene": np.array(list(d["per_gene_pearson"].values()), dtype=float),
            "bin_r2":   float(d.get("per_bin_r2_pooled", float("nan"))),
            "slide":    d.get("smoke_slide", "?"),
        }
    n_genes = len(payload["uni"]["per_gene"])
    slide = payload["uni"]["slide"]

    with mpl.rc_context(NATURE_RC):
        fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.2))
        x = np.arange(len(METHODS))
        width = 0.66

        # ---- (a) per-gene Pearson (mean ± SD across genes) ----
        ax = axes[0]
        means = [payload[m]["per_gene"].mean() for m in METHODS]
        stds  = [payload[m]["per_gene"].std(ddof=1) for m in METHODS]
        colors = [METHOD_COLOR[m] for m in METHODS]
        ax.bar(x, means, width, yerr=stds, capsize=2,
               color=colors, edgecolor="black", linewidth=0.4,
               error_kw=dict(ecolor="black", lw=0.6))
        rng = np.random.RandomState(7)
        for i, m in enumerate(METHODS):
            vals = payload[m]["per_gene"]
            jit = rng.uniform(-0.10, 0.10, size=len(vals))
            ax.scatter(np.full_like(vals, x[i]) + jit, vals,
                       s=2.2, color="black", alpha=0.45, linewidth=0, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABEL[m] for m in METHODS])
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_ylabel("Per-gene Pearson r")
        ax.set_title("Per-gene", pad=4)
        ax.text(-0.22, 1.08, "a", transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="left")
        ax.tick_params(pad=2)

        # ---- (b) Pooled per-bin R² (single value per method) ----
        ax = axes[1]
        r2s = [payload[m]["bin_r2"] for m in METHODS]
        ax.bar(x, r2s, width,
               color=colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABEL[m] for m in METHODS])
        ax.set_ylim(0, 0.6)
        ax.set_yticks(np.arange(0, 0.61, 0.2))
        ax.set_ylabel("Pooled per-bin R²")
        ax.set_title("Spatial bin", pad=4)
        ax.text(-0.22, 1.08, "b", transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="left")
        ax.tick_params(pad=2)
        for i, v in enumerate(r2s):
            ax.text(x[i], v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=6)

        fig.text(0.5, -0.02,
                 f"single smoke slide ({slide}); error bars = s.d. across n = {n_genes} genes",
                 ha="center", fontsize=6, color="0.3")
        fig.tight_layout(w_pad=1.0)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=600, bbox_inches="tight")
        fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
        print(f"saved {args.out} and {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
