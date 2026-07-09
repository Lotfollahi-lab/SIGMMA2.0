"""UMAP of SIGMMA's learned CELL/ST (spatial) tile embeddings, coloured by
(a) biological tissue type and (b) source dataset / cohort.

Reads the per-cohort HDF5 caches written by scripts/cross_tissue_extract_patches.py
(keys st_emb_{micro,meso,macro}, section_id) for the CURRENT authoritative SIGMMA
checkpoint (dense attn-pool a01, step2000, job 36852, full test-split coverage).

Labels are resolved per tile by joining section_id -> sample_metadata.csv's
`tissue_type` column (which also covers the HEST-1k publics), then mapping that
study/cohort label to a cleaned biological organ ("tissue type") and to a
per-cohort source ("dataset type", e.g. beacon / sanger-liver / HEST-PAAD).

Figure (per scale): 1 row x 2 cols
  col 1 = ST embedding coloured by tissue type
  col 2 = ST embedding coloured by dataset type
With --scale all, scales are stacked as rows (micro / meso / macro).
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap

RNG = np.random.RandomState(0)
EMB_DIR = (
    "outputs/cross_tissue_retrieval/sigmma_dense_attnpool_gex_svg_a01/"
    "36852/embeddings_step2000"
)
META_CSV = (
    "/lustre/scratch126/cellgen/lotfollahi/dj16/data/"
    "sanger_in_house_xenium/sample_metadata.csv"
)
MAX_TOTAL = 40000  # subsample cap for UMAP

# CSV `tissue_type` -> (cleaned biological tissue, per-cohort dataset label)
LABEL_MAP = {
    "beacon": ("skin", "beacon"),
    "CTCL": ("skin", "CTCL"),
    "liver": ("liver", "sanger-liver"),
    "lung": ("lung", "sanger-lung"),
    "pancreas": ("pancreas", "sanger-pancreas"),
    "fetal_pancreas": ("pancreas", "sanger-fetal-pancreas"),
    "glioblastoma": ("brain", "sanger-glioblastoma"),
    "pediatric_ovary": ("ovary", "sanger-ovary"),
    "renal_cell_carcinoma": ("kidney", "sanger-RCC"),
    "whole_embryo": ("embryo", "sanger-embryo"),
    "adrenocortical_carcinoma": ("adrenal", "sanger-adrenal"),
    "IDC": ("breast", "HEST-IDC"),
    "LUAD": ("lung", "HEST-LUAD"),
    "PAAD": ("pancreas", "HEST-PAAD"),
    "SKCM": ("skin", "HEST-SKCM"),
}


def load_section_to_csvlabel():
    meta = pd.read_csv(META_CSV)
    return dict(zip(meta["sample_id"].astype(str), meta["tissue_type"].astype(str)))


def load_st(scale, sec2csv):
    """Return (st_emb, tissue[], dataset[]) over all cohort files for one scale."""
    files = sorted(glob.glob(f"{EMB_DIR}/*.h5"))
    st, tissue, dataset = [], [], []
    for f in files:
        cohort = os.path.splitext(os.path.basename(f))[0]
        with h5py.File(f) as h:
            sk = f"st_emb_{scale}"
            if sk not in h:
                continue
            emb = h[sk][:]
            sids = [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in h["section_id"][:]
            ]
        n_unmapped = 0
        for sid in sids:
            csv_lab = sec2csv.get(sid)
            if csv_lab is None or csv_lab not in LABEL_MAP:
                tissue.append("unknown")
                dataset.append(cohort)
                n_unmapped += 1
            else:
                t, d = LABEL_MAP[csv_lab]
                tissue.append(t)
                dataset.append(d)
        st.append(emb)
        flag = f"  ({n_unmapped} unmapped)" if n_unmapped else ""
        print(f"{cohort:24s} {emb.shape[0]:6d} tiles{flag}")
    return (
        np.concatenate(st),
        np.array(tissue),
        np.array(dataset),
    )


def umap2d(X, seed=1):
    Xs = StandardScaler().fit_transform(X)
    return umap.UMAP(
        n_neighbors=30, min_dist=0.3, random_state=seed
    ).fit_transform(Xs)


def scatter(ax, Y, labels, title, legend_fs=7, ncol=2):
    cats = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("tab20")
    colors = {c: cmap(i % 20) for i, c in enumerate(cats)}
    for c in cats:
        m = labels == c
        ax.scatter(
            Y[m, 0], Y[m, 1], s=3, color=colors[c], alpha=0.5, lw=0,
            label=f"{c} ({int(m.sum())})", rasterized=True,
        )
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP-1", fontsize=8)
    ax.set_ylabel("UMAP-2", fontsize=8)
    ax.legend(markerscale=4, frameon=False, fontsize=legend_fs, ncol=ncol, loc="best")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scale", default="all",
        choices=["micro", "meso", "macro", "all"],
    )
    ap.add_argument("--fig_dir", default="figures/diagnostics")
    args = ap.parse_args()

    scales = ["micro", "meso", "macro"] if args.scale == "all" else [args.scale]
    sec2csv = load_section_to_csvlabel()

    fig, axes = plt.subplots(
        len(scales), 2, figsize=(16, 6.5 * len(scales)), squeeze=False,
    )
    for r, scale in enumerate(scales):
        print(f"\n=== scale: {scale} ===")
        st, tissue, dataset = load_st(scale, sec2csv)
        n = len(tissue)
        if n > MAX_TOTAL:
            idx = RNG.choice(n, MAX_TOTAL, replace=False)
            st, tissue, dataset = st[idx], tissue[idx], dataset[idx]
            print(f"subsampled {n} -> {len(tissue)}")
        Y = umap2d(st, seed=1)
        scatter(axes[r, 0], Y, tissue, f"ST tile embedding ({scale}) — by tissue type")
        scatter(axes[r, 1], Y, dataset, f"ST tile embedding ({scale}) — by dataset type")
        print("tissues:", {t: int((tissue == t).sum()) for t in sorted(set(tissue))})
        print("datasets:", {d: int((dataset == d).sum()) for d in sorted(set(dataset))})

    fig.suptitle(
        "SIGMMA ST tile embeddings (step2000): by tissue type and dataset type",
        fontsize=13, y=1.0,
    )
    fig.tight_layout()
    os.makedirs(args.fig_dir, exist_ok=True)
    tag = args.scale
    for ext in ("png", "pdf"):
        out = f"{args.fig_dir}/st_emb_umap_tissue_dataset_{tag}.{ext}"
        fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"\nsaved {args.fig_dir}/st_emb_umap_tissue_dataset_{tag}.png")


if __name__ == "__main__":
    main()
