"""Visualize the NEMO-embedding distribution shift between the in-house Xenium 5K
panel and the HEST-1k public small panels.

This is the mechanistic picture behind poor HEST-1k zero-shot: the SIGMMA ST graph
encoder consumes NEMO cell embeddings (obsm['nemo_latent'], 384-d) as node features.
NEMO embeddings are computed per-section from that section's gene panel, so the small
HEST panels (278-469 genes) land in a different region of the 384-d space than the
in-house 5K panel (~4289 genes) the encoder was trained on -> out-of-distribution input.

Panels:
  A) global UMAP coloured by source (in-house vs HEST)         -> overall gap
  B) same UMAP coloured by tissue                              -> tissue confound shown
  C) skin-only UMAP (in-house SKI vs HEST SKCM)                -> tissue fixed, isolates panel/batch
Plus 384-d kNN same-source purity (0.5 = mixed, 1.0 = separated).
"""
import os
import numpy as np
import scanpy as sc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import umap

RNG = np.random.RandomState(0)
N_PER_SECTION = 3000
BASE = "/lustre/scratch126/cellgen/lotfollahi/dj16/data"

# (path, source, tissue)
SECTIONS = [
    # ----- in-house, Xenium 5K panel -----
    (f"{BASE}/sanger_in_house_xenium/sanger_CTCL/20241115__150035__SGP206_run1/AX6-SKI-0-FFPE-1-S10/spatial_data_processed.h5", "in-house", "skin"),
    (f"{BASE}/sanger_in_house_xenium/sanger_CTCL/20241115__150035__SGP206_run1/AX12-SKI-0-FFPE-1-S11/spatial_data_processed.h5", "in-house", "skin"),
    (f"{BASE}/sanger_in_house_xenium/sanger_renal_cell_carcinoma/20250319__163455__SGP196_Run5r/CV7-KID-0-FT-5-s5/spatial_data_processed.h5", "in-house", "kidney"),
    (f"{BASE}/sanger_in_house_xenium/sanger_renal_cell_carcinoma/20250319__163455__SGP196_Run5r/CV2-KID-0-FO-1-s16/spatial_data_processed.h5", "in-house", "kidney"),
    (f"{BASE}/sanger_in_house_xenium/sanger_liver/20250626__122115__SGP251_run1/DF31-LIV-0-FFPE-1-S1-i/spatial_data_processed.h5", "in-house", "liver"),
    (f"{BASE}/sanger_in_house_xenium/sanger_whole_embryo/20241003__123612__SGP193_WEM_RUN3/SOB26-WEM-FFPE-1-S93/spatial_data_processed.h5", "in-house", "whole_embryo"),
    # ----- HEST-1k public, small 10x panels -----
    (f"{BASE}/public_xenium/hest_1k_PAAD/TENX116/spatial_data_processed.h5", "HEST-1k", "PAAD"),
    (f"{BASE}/public_xenium/hest_1k_PAAD/TENX140/spatial_data_processed.h5", "HEST-1k", "PAAD"),
    (f"{BASE}/public_xenium/hest_1k_IDC/NCBI785/spatial_data_processed.h5", "HEST-1k", "IDC(breast)"),
    (f"{BASE}/public_xenium/hest_1k_LUAD/TENX118/spatial_data_processed.h5", "HEST-1k", "LUAD"),
    (f"{BASE}/public_xenium/hest_1k_SKCM/TENX115/spatial_data_processed.h5", "HEST-1k", "SKCM(skin)"),
    (f"{BASE}/public_xenium/hest_1k_SKCM/TENX117/spatial_data_processed.h5", "HEST-1k", "SKCM(skin)"),
]


def load():
    embs, sources, tissues, secids, ngenes = [], [], [], [], []
    for si, (path, src, tis) in enumerate(SECTIONS):
        if not os.path.exists(path):
            print("MISSING", path)
            continue
        a = sc.read_h5ad(path)
        X = np.asarray(a.obsm["nemo_latent"], dtype=np.float32)
        n = X.shape[0]
        idx = RNG.choice(n, size=min(N_PER_SECTION, n), replace=False)
        embs.append(X[idx])
        sources += [src] * len(idx)
        tissues += [tis] * len(idx)
        secids += [si] * len(idx)
        ngenes.append((tis, a.shape[1]))
        print(f"{src:8s} {tis:14s} genes={a.shape[1]:5d}  cells={n:7d} -> {len(idx)}")
    return (np.concatenate(embs), np.array(sources), np.array(tissues),
            np.array(secids), ngenes)


def knn_purity(X, labels, k=30):
    """Fraction of each cell's k nearest neighbours (in 384-d) sharing its label.
    Balanced baseline = max class proportion if separated, ~class-prior if mixed."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, ind = nn.kneighbors(X)
    ind = ind[:, 1:]
    same = (labels[ind] == labels[:, None]).mean()
    return same


def run_umap(X, seed=0):
    Xs = StandardScaler().fit_transform(X)
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=seed, metric="euclidean")
    return reducer.fit_transform(Xs), Xs


def pair_purity(emb, secids, ia, ib):
    """Standardise the two sections together, then kNN same-section purity."""
    m = np.isin(secids, [ia, ib])
    Xp = StandardScaler().fit_transform(emb[m])
    return knn_purity(Xp, (secids[m] == ia).astype(int))


def main():
    emb, src, tis, sec, ngenes = load()
    Y, Xs = run_umap(emb)

    pur_global = knn_purity(Xs, src)

    # skin-only subset: in-house skin vs HEST SKCM
    skin_mask = np.isin(tis, ["skin", "SKCM(skin)"])
    emb_s = emb[skin_mask]
    src_s = src[skin_mask]
    Ys, Xss = run_umap(emb_s, seed=1)
    pur_skin = knn_purity(Xss, src_s)
    # mixing baseline = expected purity if perfectly mixed = sum(p_c^2)
    p = np.array([(src_s == c).mean() for c in np.unique(src_s)])
    base_skin = (p ** 2).sum()

    # CONTROL: per-section batch baseline (same source/tissue/panel, two sections)
    # SECTIONS idx 0,1 = in-house skin; 10,11 = HEST SKCM
    ctrl = {
        "in-house SKI\nsec1 vs sec2\n(same panel)": pair_purity(emb, sec, 0, 1),
        "HEST SKCM\nsec1 vs sec2\n(same panel)": pair_purity(emb, sec, 10, 11),
        "in-house SKI\nvs HEST SKCM\n(cross-panel)": pur_skin,
    }

    src_colors = {"in-house": "#888888", "HEST-1k": "#d62728"}
    tis_list = list(dict.fromkeys(tis.tolist()))
    cmap = plt.get_cmap("tab10")
    tis_colors = {t: cmap(i % 10) for i, t in enumerate(tis_list)}

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))

    # A: source
    ax = axes[0]
    for s, c in src_colors.items():
        m = src == s
        ax.scatter(Y[m, 0], Y[m, 1], s=3, c=c, alpha=0.45, lw=0, label=s, rasterized=True)
    ax.set_title(f"A. NEMO space by source\n384-d kNN same-source purity = {pur_global:.2f}", fontsize=10)
    ax.legend(markerscale=4, frameon=False, fontsize=9, loc="best")

    # B: tissue
    ax = axes[1]
    for t in tis_list:
        m = tis == t
        ax.scatter(Y[m, 0], Y[m, 1], s=3, c=[tis_colors[t]], alpha=0.45, lw=0, label=t, rasterized=True)
    ax.set_title("B. same UMAP by tissue\n(tissue confound is real)", fontsize=10)
    ax.legend(markerscale=4, frameon=False, fontsize=7, ncol=2, loc="best")

    # C: skin only
    ax = axes[2]
    for s, c in src_colors.items():
        m = src_s == s
        lab = {"in-house": "in-house SKI", "HEST-1k": "HEST SKCM"}[s]
        ax.scatter(Ys[m, 0], Ys[m, 1], s=4, c=c, alpha=0.5, lw=0, label=lab, rasterized=True)
    ax.set_title(
        f"C. skin only: tissue fixed\nkNN purity = {pur_skin:.2f} (mixed baseline {base_skin:.2f})",
        fontsize=10,
    )
    ax.legend(markerscale=4, frameon=False, fontsize=8, loc="best")

    for ax in axes[:3]:
        ax.set_xlabel("UMAP-1", fontsize=9)
        ax.set_ylabel("UMAP-2", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # D: control bar chart - per-section batch vs cross-panel separation
    ax = axes[3]
    names = list(ctrl.keys())
    vals = [ctrl[n] for n in names]
    bar_c = ["#888888", "#d62728", "#d62728"]
    ax.bar(range(len(names)), vals, color=bar_c, alpha=0.85)
    ax.axhline(0.5, ls="--", c="k", lw=1)
    ax.text(0.02, 0.52, "0.50 = fully mixed", transform=ax.get_yaxis_transform(), fontsize=8)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylim(0.45, 1.05)
    ax.set_ylabel("kNN same-section purity (384-d)", fontsize=9)
    ax.set_title("D. control: batch alone already separates\n(NEMO is not harmonised across sections)", fontsize=10)

    fig.suptitle(
        "NEMO cell embeddings (SIGMMA ST-encoder input) are section/panel-specific -> every HEST-1k section is OOD",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    os.makedirs("figures/diagnostics", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/diagnostics/nemo_panel_shift.{ext}", dpi=180, bbox_inches="tight")
    print("\nsaved figures/diagnostics/nemo_panel_shift.{png,pdf}")
    print(f"global kNN same-source purity = {pur_global:.3f}")
    print(f"skin-only kNN purity = {pur_skin:.3f} (mixed baseline {base_skin:.3f})")
    print("panel sizes:", ngenes)


if __name__ == "__main__":
    main()
