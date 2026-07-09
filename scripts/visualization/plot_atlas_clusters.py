"""Step 4 of the cross-tissue atlas pipeline.

Read cluster_assignments.parquet (Step 2) + cluster_metrics.csv (Step 3),
render four Nature-style panels:

    a) UMAP of image_emb_meso coloured by cluster_id.
    b) Stacked bar of tissue composition per cluster (8 tissues).
    c) Stacked bar of dominant niche fraction per cluster
       (curated tissues only; one stack per tissue per cluster).
    d) H_tissue_norm vs H_niche_norm scatter; bubble = n_patches; top-10
       by score annotated.

Output: figures/cross_tissue_retrieval/atlas_overview.pdf and
        figures/cross_tissue_retrieval/atlas_priority_scatter.pdf.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

from scripts.atlas_aggregate_niche_composition import (
    CACHE_DIR as NICHE_CACHE_DIR,
    LABELLED_TISSUES,
)
from scripts.cross_tissue_extract_patches import TISSUES

EMB_DIR = Path("outputs/cross_tissue_retrieval/embeddings")
ATLAS_DIR = Path("outputs/cross_tissue_retrieval/atlas")
FIG_DIR = Path("figures/cross_tissue_retrieval")
UMAP_SAMPLE = 30_000
RANDOM_STATE = 42

# Tissue palette: cool→warm to make tissue identity readable.
TISSUE_PALETTE = {
    "skin": "#1f77b4",
    "glioblastoma": "#9467bd",
    "liver": "#8c564b",
    "lung": "#17becf",
    "pancreas": "#bcbd22",
    "pediatric_ovary": "#e377c2",
    "renal_cell_carcinoma": "#ff7f0e",
    "whole_embryo": "#2ca02c",
}


def _nature_rc():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _load_scale_embedding(scale: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Stack the requested scale's H&E embedding across all tissues."""
    chunks, rows = [], []
    for tissue, _ in TISSUES:
        path = EMB_DIR / f"{tissue}.h5"
        if not path.exists():
            continue
        with h5py.File(path, "r") as f:
            if scale == "concat":
                emb = np.concatenate(
                    [f["image_emb_micro"][:], f["image_emb_meso"][:], f["image_emb_macro"][:]],
                    axis=1,
                )
            else:
                emb = f[f"image_emb_{scale}"][:]
            section = np.array([s.decode() for s in f["section_id"][:]])
            ys, xs = f["y_start"][:], f["x_start"][:]
        chunks.append(emb)
        rows.append(pd.DataFrame({
            "tissue": tissue, "section_id": section,
            "y_start": ys, "x_start": xs,
        }))
    return np.concatenate(chunks, axis=0), pd.concat(rows, ignore_index=True)


def _panel_umap(ax, X: np.ndarray, cluster_id: np.ndarray, k: int, scale: str):
    rng = np.random.default_rng(RANDOM_STATE)
    if X.shape[0] > UMAP_SAMPLE:
        idx = rng.choice(X.shape[0], size=UMAP_SAMPLE, replace=False)
        Xs = X[idx]
        cs = cluster_id[idx]
    else:
        Xs = X
        cs = cluster_id
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                        metric="cosine", random_state=RANDOM_STATE,
                        low_memory=True)
    Z = reducer.fit_transform(Xs)
    cmap = plt.get_cmap("tab20", k)
    ax.scatter(Z[:, 0], Z[:, 1], c=cs, cmap=cmap, s=1.5, alpha=0.6, linewidths=0)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"UMAP ({scale}, n={Xs.shape[0]:,} of {X.shape[0]:,})")


def _panel_tissue_stack(ax, assignments: pd.DataFrame, k: int):
    cluster_ids = np.arange(k)
    tissue_names = [t for t, _ in TISSUES]
    counts = np.zeros((k, len(tissue_names)), dtype=np.int64)
    for ci in cluster_ids:
        sub = assignments[assignments["cluster_id"] == ci]
        for ti, t in enumerate(tissue_names):
            counts[ci, ti] = int((sub["tissue"] == t).sum())
    fracs = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    bottom = np.zeros(k)
    for ti, t in enumerate(tissue_names):
        ax.bar(cluster_ids, fracs[:, ti], bottom=bottom,
               color=TISSUE_PALETTE[t], width=0.85, label=t, linewidth=0)
        bottom += fracs[:, ti]
    ax.set_xticks(cluster_ids)
    ax.set_xticklabels([str(i) for i in cluster_ids], rotation=90)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Tissue fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Tissue composition per cluster")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1)


def _panel_dominant_niche_bar(ax, metrics: pd.DataFrame):
    m = metrics.copy().sort_values("score", ascending=False, na_position="last")
    m = m.head(20)
    x = np.arange(len(m))
    ax.bar(x, m["dominant_niche_frac"].fillna(0),
           color="#888888", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    labels = [
        f"{cid}\n{tn[:20]}"
        for cid, tn in zip(m["cluster_id"], m["dominant_niche"].astype(str))
    ]
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("Dominant niche fraction (top 20 by score)")
    ax.set_title("Per-cluster dominant niche (curated tissues only)")
    ax.set_ylim(0, 1)


def _panel_priority_scatter(ax, metrics: pd.DataFrame, top_n: int = 10):
    m = metrics.dropna(subset=["H_niche_norm"]).copy()
    if m.empty:
        ax.text(0.5, 0.5, "no cluster meets min labelled coverage",
                ha="center", va="center", transform=ax.transAxes)
        return
    sizes = np.sqrt(m["n_patches"].to_numpy()) * 1.2
    sc = ax.scatter(
        m["H_tissue_norm"], m["H_niche_norm"], s=sizes,
        c=m["score"], cmap="viridis", edgecolor="black", linewidths=0.3, alpha=0.85,
    )
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("score = H_tissue − H_niche")
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4)

    top = m.sort_values("score", ascending=False).head(top_n)
    for _, r in top.iterrows():
        ax.annotate(f"{int(r.cluster_id)}",
                    xy=(r.H_tissue_norm, r.H_niche_norm),
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=7)
    ax.set_xlabel(r"$H_{\mathrm{tissue}}$ (normalised)")
    ax.set_ylabel(r"$H_{\mathrm{niche}}$ (normalised)")
    ax.set_xlim(-0.02, 1.05); ax.set_ylim(-0.02, 1.05)
    ax.set_title("Cluster priority scatter — top-10 annotated")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scale", default="meso",
                   choices=["micro", "meso", "macro", "concat"],
                   help="Picks the per-scale assignments / metrics / UMAP input.")
    p.add_argument("--assignments", default=None)
    p.add_argument("--metrics", default=None)
    p.add_argument("--fig_dir", default=str(FIG_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _nature_rc()
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    assignments_path = args.assignments or str(
        ATLAS_DIR / f"cluster_assignments_{args.scale}.parquet"
    )
    metrics_path = args.metrics or str(
        ATLAS_DIR / f"cluster_metrics_{args.scale}.csv"
    )
    assignments = pd.read_parquet(assignments_path)
    metrics = pd.read_csv(metrics_path)
    k = int(assignments["K"].iloc[0])
    print(f"scale = {args.scale}, K = {k}, n_patches = {len(assignments)}")

    # Overview figure: 2x2.
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    X, _ = _load_scale_embedding(args.scale)
    _panel_umap(axes[0, 0], X, assignments["cluster_id"].to_numpy(), k, args.scale)
    _panel_tissue_stack(axes[0, 1], assignments, k)
    _panel_dominant_niche_bar(axes[1, 0], metrics)
    _panel_priority_scatter(axes[1, 1], metrics)
    for ax, letter in zip(axes.flat, "abcd"):
        ax.text(-0.18, 1.05, letter, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="bottom", ha="left")
    fig.tight_layout()
    out_overview = fig_dir / f"atlas_overview_{args.scale}.pdf"
    fig.savefig(out_overview, dpi=600, bbox_inches="tight")
    fig.savefig(out_overview.with_suffix(".png"), dpi=600, bbox_inches="tight")
    print(f"wrote {out_overview}")

    # Standalone priority scatter (larger).
    fig2, ax2 = plt.subplots(figsize=(5.0, 4.5))
    _panel_priority_scatter(ax2, metrics)
    fig2.tight_layout()
    out_scatter = fig_dir / f"atlas_priority_scatter_{args.scale}.pdf"
    fig2.savefig(out_scatter, dpi=600, bbox_inches="tight")
    fig2.savefig(out_scatter.with_suffix(".png"), dpi=600, bbox_inches="tight")
    print(f"wrote {out_scatter}")


if __name__ == "__main__":
    main()
