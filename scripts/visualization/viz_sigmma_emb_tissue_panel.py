"""UMAP of SIGMMA's learned IMAGE and CELL/ST (spatial) embeddings, coloured by
tissue and by gene panel.

Reads the per-tissue HDF5 caches written by scripts/cross_tissue_extract_patches.py
(keys image_emb_{micro,meso,macro}, st_emb_{...}, section_id) for the CURRENT
SIGMMA (dense attn-pool a01 step2000), spanning in-house (5K/variant panels) and
HEST-1k publics (small 10x panels).

Figure: 2 rows x 2 cols
  row 1 = image embedding,  row 2 = cell/ST (spatial) embedding
  col 1 = coloured by tissue, col 2 = coloured by gene panel
Panel label per section = panel_hash(var.index) (same hashing as sigmma.caches),
so in-house 5K sections collapse to one panel and each HEST dataset is its own.
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd
import h5py
import scanpy as sc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap

from sigmma.caches import panel_hash

RNG = np.random.RandomState(0)
EMB_DIR = "outputs/cross_tissue_retrieval/embeddings_step2000"
MAX_TOTAL = 40000  # subsample cap for UMAP
DATA_ROOT = "/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium"
PUBLIC_ROOT = "/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_xenium"

MANIFESTS = {
    "skin": f"{DATA_ROOT}/sanger_beacon_MANIFEST_test.csv",
    "glioblastoma": f"{DATA_ROOT}/sanger_glioblastoma_MANIFEST_test.csv",
    "liver": f"{DATA_ROOT}/sanger_liver_MANIFEST_test.csv",
    "lung": f"{DATA_ROOT}/sanger_lung_MANIFEST_test.csv",
    "pancreas": f"{DATA_ROOT}/sanger_pancreas_MANIFEST_test.csv",
    "pediatric_ovary": f"{DATA_ROOT}/sanger_pediatric_ovary_MANIFEST_test.csv",
    "renal_cell_carcinoma": f"{DATA_ROOT}/sanger_renal_cell_carcinoma_MANIFEST_test.csv",
    "whole_embryo": f"{DATA_ROOT}/sanger_whole_embryo_MANIFEST_test.csv",
    "hest_PAAD": f"{PUBLIC_ROOT}/MANIFEST_hest_1k_PAAD_test.csv",
    "hest_IDC": f"{PUBLIC_ROOT}/MANIFEST_hest_1k_IDC_test.csv",
    "hest_LUAD": f"{PUBLIC_ROOT}/MANIFEST_hest_1k_LUAD_test.csv",
    "hest_SKCM": f"{PUBLIC_ROOT}/MANIFEST_hest_1k_SKCM_test.csv",
}


def build_section_to_h5():
    """sample_name -> spatial_preprocessed path, across all manifests."""
    m = {}
    for path in MANIFESTS.values():
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            m[str(r["sample_name"])] = r["spatial_preprocessed"]
    return m


def panel_for_sections(section_ids, sec2h5, cache={}):
    """Map each unique section -> short panel_hash (read var.index once per section)."""
    out = {}
    for sid in set(section_ids):
        if sid in cache:
            out[sid] = cache[sid]
            continue
        h5 = sec2h5.get(sid)
        if h5 is None or not os.path.exists(h5):
            out[sid] = "unknown"
            continue
        a = sc.read_h5ad(h5)
        genes = [str(g) for g in a.var.index]
        ph = panel_hash(genes).replace("sha256:", "")[:8]
        out[sid] = f"{ph} ({len(genes)}g)"
        cache[sid] = out[sid]
    return out


def load_all(scale):
    files = sorted(glob.glob(f"{EMB_DIR}/*.h5"))
    img, st, tissue, sec = [], [], [], []
    for f in files:
        tname = os.path.splitext(os.path.basename(f))[0]
        with h5py.File(f) as h:
            ik, sk = f"image_emb_{scale}", f"st_emb_{scale}"
            if ik not in h:
                continue
            img.append(h[ik][:])
            st.append(h[sk][:])
            sids = [s.decode() if isinstance(s, bytes) else str(s) for s in h["section_id"][:]]
            sec += sids
            tissue += [tname] * len(sids)
            print(f"{tname:24s} {h[ik].shape[0]:6d} patches")
    return (np.concatenate(img), np.concatenate(st), np.array(tissue), np.array(sec))


def umap2d(X, seed=0):
    Xs = StandardScaler().fit_transform(X)
    return umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=seed).fit_transform(Xs)


def scatter(ax, Y, labels, title, palette=None, legend_fs=6, ncol=1):
    cats = list(dict.fromkeys(labels.tolist()))
    cmap = plt.get_cmap("tab20")
    colors = palette or {c: cmap(i % 20) for i, c in enumerate(cats)}
    for c in cats:
        m = labels == c
        ax.scatter(Y[m, 0], Y[m, 1], s=3, color=colors[c], alpha=0.5, lw=0,
                   label=c, rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP-1", fontsize=8); ax.set_ylabel("UMAP-2", fontsize=8)
    ax.legend(markerscale=4, frameon=False, fontsize=legend_fs, ncol=ncol, loc="best")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="macro", choices=["micro", "meso", "macro"])
    args = ap.parse_args()

    img, st, tissue, sec = load_all(args.scale)
    n = len(tissue)
    if n > MAX_TOTAL:
        idx = RNG.choice(n, MAX_TOTAL, replace=False)
        img, st, tissue, sec = img[idx], st[idx], tissue[idx], sec[idx]
        print(f"subsampled {n} -> {len(tissue)}")

    sec2h5 = build_section_to_h5()
    sec2panel = panel_for_sections(sec, sec2h5)
    panel = np.array([sec2panel[s] for s in sec])

    # tissue colours: in-house tab20, HEST highlighted reds/oranges
    tcats = list(dict.fromkeys(tissue.tolist()))
    cmap = plt.get_cmap("tab20")
    tpal = {t: cmap(i % 20) for i, t in enumerate(tcats)}

    Yi = umap2d(img, seed=0)
    Ys = umap2d(st, seed=1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    scatter(axes[0, 0], Yi, tissue, f"Image embedding ({args.scale}) — by tissue", tpal, 7, 2)
    scatter(axes[0, 1], Yi, panel, f"Image embedding ({args.scale}) — by gene panel", None, 6, 1)
    scatter(axes[1, 0], Ys, tissue, f"Cell/ST embedding ({args.scale}) — by tissue", tpal, 7, 2)
    scatter(axes[1, 1], Ys, panel, f"Cell/ST embedding ({args.scale}) — by gene panel", None, 6, 1)

    fig.suptitle(
        "SIGMMA learned embeddings (step2000): image vs cell/ST, by tissue and gene panel",
        fontsize=13, y=1.0,
    )
    fig.tight_layout()
    os.makedirs("figures/diagnostics", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/diagnostics/sigmma_emb_tissue_panel_{args.scale}.{ext}",
                    dpi=170, bbox_inches="tight")
    print(f"saved figures/diagnostics/sigmma_emb_tissue_panel_{args.scale}.png")
    # quick panel/tissue counts
    print("tissues:", {t: int((tissue == t).sum()) for t in tcats})
    print("panels:", {p: int((panel == p).sum()) for p in dict.fromkeys(panel.tolist())})


if __name__ == "__main__":
    main()
