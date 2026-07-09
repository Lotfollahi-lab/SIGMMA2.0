"""Supplementary: H&E <-> ST retrieval on UNSEEN HEST-1k public panels (OOD).

4 rows x 4 tissue cols, both retrieval directions:
  row 0: H&E -> ST  Recall @ 1%   (dashed random line = 0.01)
  row 1: H&E -> ST  MRR           (no random line; pooled-gallery chance ~ 0)
  row 2: ST -> H&E  Recall @ 1%   (dashed random line = 0.01)
  row 3: ST -> H&E  MRR           (no random line)

The point: every method hovers around chance on the held-out public panels —
no usable retrieval signal in either direction. Contrast with the in-house
per-tissue figure.

Unit = TILE (query FOV), not section. Two error conventions:
  * Recall rows: per-tile hit RATE, so we pool all of a tissue's tiles (weight
    per-section rate by n_queries) and show the tile-level Wilson CI of the
    binary hit indicator — NOT a section-to-section SD.
  * MRR rows: MRR is continuous (no binomial), so we plot the n_queries-weighted
    pooled mean with a section-level SD error bar. Public panels have only
    n=2-4 sections, so that SD is indicative, not a calibrated CI.
Method palette per project convention.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
DATA_JSON = ROOT / "outputs/retrieval/normalized_metrics_step2000.json"
META_CSV = Path("/nfs/users/nfs_d/dj16/lustre_dabin/data/sanger_in_house_xenium/"
                "sample_metadata.csv")

CONDITIONS = [
    ("bleep",   "BLEEP-ft",   "#882255"),
    ("clip",    "CLIP-ft",    "#44AA99"),
    ("plip",    "PLIP-ft",    "#DDCC77"),
    ("omiclip", "OmiCLIP-ft", "#999933"),
    ("a01",     "SIGMMA",     "#d62728"),
]
COND_KEYS = [c[0] for c in CONDITIONS]
PUBLIC = ["IDC", "PAAD", "SKCM", "LUAD"]
LABEL = {"IDC": "IDC", "PAAD": "PAAD", "SKCM": "SKCM", "LUAD": "LUAD"}

# (direction, metric_key, row_label, chance, kind). kind selects the error model:
#   "rate" -> binomial Wilson CI; "mrr" -> section-level SD.
# chance=None omits the random reference line (MRR has no clean panel-indep chance).
ROWS = [
    ("image",   "recall_at_1pct", "H&E$\\rightarrow$ST\nRecall @ 1%", 0.01, "rate"),
    ("image",   "mrr",            "H&E$\\rightarrow$ST\nMRR",          None, "mrr"),
    ("spatial", "recall_at_1pct", "ST$\\rightarrow$H&E\nRecall @ 1%", 0.01, "rate"),
    ("spatial", "mrr",            "ST$\\rightarrow$H&E\nMRR",          None, "mrr"),
]


def wilson_ci(p, n, z=1.96):
    """Wilson score interval for a binomial proportion. Returns (lo, hi)."""
    if n == 0:
        return p, p
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def tile_pooled(per_section, sections, mkey):
    """Tile-level pooled rate + Wilson CI: weight per-section R@k% by n_queries."""
    n = np.array([per_section[s]["n_queries"] for s in sections], dtype=float)
    r = np.array([per_section[s][mkey] for s in sections], dtype=float)
    N = n.sum()
    p = (r * n).sum() / N if N > 0 else 0.0
    lo, hi = wilson_ci(p, int(N))
    return p, max(0.0, lo), hi, int(N)


def section_pooled_mean(per_section, sections, mkey):
    """n_queries-weighted pooled mean + section-level SD (for continuous MRR).

    MRR is not a binomial proportion, so Wilson does not apply. We weight each
    section's mean by its tile count, and use the (unweighted) section-to-section
    SD as an indicative error bar — public panels have only n=2-4 sections.
    """
    n = np.array([per_section[s]["n_queries"] for s in sections], dtype=float)
    v = np.array([per_section[s][mkey] for s in sections], dtype=float)
    N = n.sum()
    mean = (v * n).sum() / N if N > 0 else 0.0
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return mean, max(0.0, mean - sd), mean + sd, int(N)


def pooled(per_section, sections, mkey, kind):
    return (tile_pooled if kind == "rate" else section_pooled_mean)(
        per_section, sections, mkey)


def plot_panel(ax, pts, ylabel, chance, ymax, show_ylabel):
    # pts: list of (p, lo, hi) per condition
    p = np.array([t[0] for t in pts])
    lo = np.array([t[1] for t in pts])
    hi = np.array([t[2] for t in pts])
    colors = [c[2] for c in CONDITIONS]
    x = np.arange(len(CONDITIONS))
    # Wilson interval can sit slightly off the raw proportion at p~0; clip to >=0.
    yerr = np.vstack([np.clip(p - lo, 0, None), np.clip(hi - p, 0, None)])
    ax.bar(x, p, yerr=yerr, color=colors,
           edgecolor="black", linewidth=0.5,
           error_kw=dict(elinewidth=0.6, capsize=1.8, ecolor="black"), width=0.72)

    # random reference line, label at far left (omitted when chance is None,
    # i.e. for MRR where pooled-gallery chance ~ 0 and not panel-independent)
    if chance is not None:
        ax.axhline(chance, color="black", linestyle="--", linewidth=0.8, zorder=5)
        ax.text(-0.45, chance, "random", color="black",
                fontsize=5.5, va="bottom", ha="left")

    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in CONDITIONS], rotation=40, ha="right", fontsize=6)
    if show_ylabel:
        ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="-", linewidth=0.25, color="#cccccc", alpha=0.7, zorder=-1)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ymax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_json", default=str(DATA_JSON))
    ap.add_argument("--out_dir", default=str(ROOT / "figures/retrieval/all_gene_panel"))
    args = ap.parse_args()

    meta = pd.read_csv(META_CSV)
    tt = dict(zip(meta["sample_id"], meta["tissue_type"]))
    with open(args.data_json) as f:
        d = json.load(f)

    def sections_by_tissue(direction):
        secs = list(d["a01"][direction]["per_section"].keys())
        return {t: [s for s in secs if tt.get(s) == t] for t in PUBLIC}

    by_t_img = sections_by_tissue("image")
    for t in PUBLIC:
        print(f"{t}: {len(by_t_img[t])} sections")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 9, "axes.labelsize": 7,
        "axes.linewidth": 0.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(len(ROWS), len(PUBLIC), figsize=(8.0, 9.2),
                             gridspec_kw={"wspace": 0.30, "hspace": 0.85},
                             squeeze=False)
    for row, (direction, mkey, mlbl, chance, kind) in enumerate(ROWS):
        by_t = sections_by_tissue(direction)
        # shared y-max per row for comparability
        rowmax = 0.0
        cache = {}
        for t in PUBLIC:
            pts = []
            for k in COND_KEYS:
                psm = d[k][direction]["per_section"]
                p, lo, hi, N = pooled(psm, by_t[t], mkey, kind)
                pts.append((p, lo, hi))
                print(f"[{direction} {mkey}] {t:5s} {k:8s}: "
                      f"p={p:.4f} CI=[{lo:.4f},{hi:.4f}] N_tiles={N}")
            cache[t] = pts
            rowmax = max(rowmax, max(hi for _, _, hi in pts))
        ymax = max(rowmax * 1.12, (chance * 1.6) if chance is not None else 0.0)
        for col, t in enumerate(PUBLIC):
            ax = axes[row, col]
            plot_panel(ax, cache[t], mlbl, chance, ymax, show_ylabel=(col == 0))
            if row == 0:
                ntiles = int(sum(d["a01"][direction]["per_section"][s]["n_queries"]
                                 for s in by_t[t]))
                ax.set_title(f"{LABEL[t]}\n({len(by_t[t])} sec, {ntiles:,} tiles)",
                             fontsize=7.5, pad=4)

    import string
    letters = string.ascii_lowercase  # up to 26 subplots
    for i, ax in enumerate(axes.ravel()):
        ax.text(-0.22, 1.06, letters[i], transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom", ha="left")

    fig.suptitle(
        r"H&E $\leftrightarrow$ ST retrieval on unseen HEST-1k public panels (OOD)",
        fontsize=10, fontweight="bold", y=0.992,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(bottom=0.06, top=0.93, hspace=0.95, wspace=0.32)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "retrieval_public_ood_chance.png", dpi=600)
    fig.savefig(out / "retrieval_public_ood_chance.pdf")
    print(f"Wrote: {out/'retrieval_public_ood_chance.png'}")
    print(f"Wrote: {out/'retrieval_public_ood_chance.pdf'}")


if __name__ == "__main__":
    main()
