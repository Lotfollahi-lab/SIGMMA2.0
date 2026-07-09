"""Global NEMO-embedding OOD check across the ENTIRE sanger_MANIFEST.

Question: looking at the NEMO cell embeddings (obsm['nemo_latent'], 384-d) of *all*
sections in the manifest, are the HEST-1k public sections (TENX*/NCBI*) sitting off on
their own, or are they mixed into the in-house Xenium-5K manifold?

This is the full-manifest generalisation of `viz_nemo_panel_shift.py` (which only used a
hand-picked 12-section subset). We cannot UMAP all 76M cells, so we subsample a fixed
N_PER_SECTION cells per section (equal weight) and read only obsm/nemo_latent via h5py.

Outputs:
  figures/diagnostics/nemo_manifest_ood.{png,pdf}
  outputs/diagnostics/nemo_manifest_ood/per_section_isolation.csv

Metrics (computed in PCA-50 space for speed):
  - global kNN same-source purity (in-house vs HEST-1k)
  - per-section "fraction of kNN that share the cell's source" = isolation score
    mixed baseline for a HEST cell = global HEST fraction (~#HEST sec / #sec)
"""
import os
import argparse
import h5py
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import umap

BASE = "/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium"
MANIFEST = f"{BASE}/sanger_MANIFEST.csv"
METADATA = f"{BASE}/sample_metadata.csv"
OUT_FIG = "figures/diagnostics"
OUT_CSV = "outputs/diagnostics/nemo_manifest_ood"


def is_public(sid: str) -> bool:
    return sid.startswith("TENX") or sid.startswith("NCBI")


def load(n_per_section: int, seed: int):
    rng = np.random.RandomState(seed)
    man = pd.read_csv(MANIFEST)
    meta = pd.read_csv(METADATA).set_index("sample_id")["tissue_type"].to_dict()

    embs, sources, tissues, secids, sec_names = [], [], [], [], []
    missing, empty = [], []
    for si, row in man.iterrows():
        sid = row["sample_name"]
        path = row["spatial_preprocessed"]
        if not isinstance(path, str) or not os.path.exists(path):
            missing.append(sid)
            continue
        try:
            with h5py.File(path, "r") as f:
                if "obsm/nemo_latent" not in f:
                    empty.append(sid)
                    continue
                d = f["obsm/nemo_latent"]
                n = d.shape[0]
                if n == 0:
                    empty.append(sid)
                    continue
                k = min(n_per_section, n)
                idx = np.sort(rng.choice(n, size=k, replace=False))
                X = d[idx, :].astype(np.float32)
        except Exception as e:  # noqa: BLE001
            print(f"  ! FAIL {sid}: {e}")
            missing.append(sid)
            continue
        src = "HEST-1k" if is_public(sid) else "in-house"
        tis = meta.get(sid, "HEST-1k" if src == "HEST-1k" else "unknown")
        embs.append(X)
        sources += [src] * k
        tissues += [tis] * k
        secids += [si] * k
        sec_names.append(sid)
        if si % 50 == 0:
            print(f"  [{si:3d}/{len(man)}] {sid:30s} {src:8s} {tis:22s} n={n:7d}->{k}")

    print(f"\nloaded {len(sec_names)} sections | missing={len(missing)} empty={len(empty)}")
    if missing:
        print("  missing (first 10):", missing[:10])
    return (
        np.concatenate(embs),
        np.array(sources),
        np.array(tissues),
        np.array(secids),
        man,
    )


def knn_same_label(X, labels, k=30):
    """For each point: fraction of its k NN (excl. self) that share its label.
    Returns the per-point array."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, ind = nn.kneighbors(X)
    ind = ind[:, 1:]
    return (labels[ind] == labels[:, None]).mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_section", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pca", type=int, default=50)
    ap.add_argument("--knn", type=int, default=30)
    args = ap.parse_args()

    emb, src, tis, sec, man = load(args.n_per_section, args.seed)
    n_total = emb.shape[0]
    n_hest = int((src == "HEST-1k").sum())
    hest_frac = n_hest / n_total
    print(f"\ntotal points = {n_total} | HEST-1k = {n_hest} ({hest_frac:.3%})")

    # standardise -> PCA-50 (metric space) -> UMAP (display)
    Xs = StandardScaler().fit_transform(emb)
    Xp = PCA(n_components=args.pca, random_state=0).fit_transform(Xs)
    print(f"PCA-{args.pca} done. running UMAP on {n_total} points ...")
    Y = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0,
                  metric="euclidean").fit_transform(Xp)

    # ---- per-cell metric in PCA space ----
    # CAVEAT: NEMO is section-specific, so EVERY section clusters -> per-cell
    # same-source purity is inflated by within-section clustering (true of
    # in-house too). The decisive OOD test is at the SECTION-CENTROID level below.
    print("computing kNN same-source purity ...")
    same_src = knn_same_label(Xp, src, k=args.knn)        # per-point
    pur_global = same_src.mean()

    # ---- section-centroid level (the clean OOD test) ----
    # Each section -> its mean position in PCA-50 space. Then for every section,
    # distance to its NEAREST IN-HOUSE section centroid. If HEST sections sit
    # farther from the in-house manifold than in-house sections do from each
    # other, HEST is genuinely off on its own (not just "every section blobs").
    sec_ids = np.unique(sec)
    cents, c_src, c_tis, c_name = [], [], [], []
    for si in sec_ids:
        m = sec == si
        cents.append(Xp[m].mean(0))
        c_src.append(src[m][0])
        c_tis.append(tis[m][0])
        c_name.append(man.iloc[si]["sample_name"])
    cents = np.vstack(cents)
    c_src = np.array(c_src)
    c_tis = np.array(c_tis)
    inh_idx = np.where(c_src == "in-house")[0]
    inh_cent = cents[inh_idx]
    nn_inh = NearestNeighbors(n_neighbors=2).fit(inh_cent)
    dist_to_inh = []
    for i in range(len(cents)):
        d, j = nn_inh.kneighbors(cents[i:i + 1], n_neighbors=2)
        # for an in-house section its own centroid is the nearest -> take 2nd
        dist_to_inh.append(d[0, 1] if c_src[i] == "in-house" else d[0, 0])
    dist_to_inh = np.array(dist_to_inh)

    rows = []
    for k, si in enumerate(sec_ids):
        m = sec == si
        rows.append({
            "section": man.iloc[si]["sample_name"],
            "source": c_src[k],
            "tissue": c_tis[k],
            "n_sampled": int(m.sum()),
            "same_source_purity": float(same_src[m].mean()),
            "dist_to_nearest_inhouse_centroid": float(dist_to_inh[k]),
        })
    iso = pd.DataFrame(rows)
    os.makedirs(OUT_CSV, exist_ok=True)
    iso.sort_values("dist_to_nearest_inhouse_centroid", ascending=False).to_csv(
        f"{OUT_CSV}/per_section_isolation.csv", index=False)

    hest_iso = iso[iso.source == "HEST-1k"]["same_source_purity"]
    inh_iso = iso[iso.source == "in-house"]["same_source_purity"]
    hest_d = dist_to_inh[c_src == "HEST-1k"]
    inh_d = dist_to_inh[c_src == "in-house"]
    inh_d99 = np.percentile(inh_d, 99)
    n_hest_outlier = int((hest_d > inh_d99).sum())
    print(f"\n=== per-cell same-source purity (section-clustering inflated) ===")
    print(f"HEST mean={hest_iso.mean():.3f}  in-house mean={inh_iso.mean():.3f}  "
          f"global={pur_global:.3f} (mixed baseline {hest_frac:.3f})")
    print(f"\n=== SECTION-CENTROID dist to nearest in-house section (the OOD test) ===")
    print(f"in-house: median={np.median(inh_d):.2f}  p99={inh_d99:.2f}  max={inh_d.max():.2f}")
    print(f"HEST-1k : median={np.median(hest_d):.2f}  min={hest_d.min():.2f}  max={hest_d.max():.2f}")
    print(f"HEST sections beyond in-house p99 radius: {n_hest_outlier}/{len(hest_d)}")

    # ---- figure ----
    src_colors = {"in-house": "#888888", "HEST-1k": "#d62728"}
    tis_list = list(dict.fromkeys(tis.tolist()))
    cmap = plt.get_cmap("tab20")
    tis_colors = {t: cmap(i % 20) for i, t in enumerate(tis_list)}

    fig, axes = plt.subplots(1, 4, figsize=(24, 5.6))

    # A: by source (draw in-house first, HEST on top)
    ax = axes[0]
    for s in ["in-house", "HEST-1k"]:
        m = src == s
        ax.scatter(Y[m, 0], Y[m, 1], s=2, c=src_colors[s], alpha=0.4, lw=0,
                   label=f"{s} (n_sec={iso[iso.source==s].shape[0]})", rasterized=True)
    ax.set_title(f"A. Whole-manifest NEMO space by source\n"
                 f"global kNN same-source purity = {pur_global:.2f} "
                 f"(mixed≈{(src=='HEST-1k').mean()**2+(src=='in-house').mean()**2:.2f})",
                 fontsize=10)
    ax.legend(markerscale=6, frameon=False, fontsize=9, loc="best")

    # B: by tissue
    ax = axes[1]
    for t in tis_list:
        m = tis == t
        ax.scatter(Y[m, 0], Y[m, 1], s=2, c=[tis_colors[t]], alpha=0.4, lw=0,
                   label=t, rasterized=True)
    ax.set_title("B. same UMAP by tissue", fontsize=10)
    ax.legend(markerscale=6, frameon=False, fontsize=6, ncol=2, loc="best")

    for ax in axes[:2]:
        ax.set_xlabel("UMAP-1", fontsize=9)
        ax.set_ylabel("UMAP-2", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # C: section-CENTROID 2D (PCA-2 of the 451 centroids) by source
    C2 = PCA(n_components=2, random_state=0).fit_transform(cents)
    ax = axes[2]
    for s in ["in-house", "HEST-1k"]:
        m = c_src == s
        ax.scatter(C2[m, 0], C2[m, 1], s=28,
                   c=src_colors[s], alpha=0.75, lw=0.3, edgecolor="white",
                   label=f"{s} (n={m.sum()})")
    ax.set_xlabel("PCA-1 (section centroids)", fontsize=9)
    ax.set_ylabel("PCA-2 (section centroids)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("C. one point per section (centroid)\nare HEST centroids at the edge?",
                 fontsize=10)
    ax.legend(markerscale=1.5, frameon=False, fontsize=9, loc="best")

    # D: distance from each section centroid to nearest IN-HOUSE centroid
    ax = axes[3]
    bins = np.linspace(0, max(inh_d.max(), hest_d.max()) * 1.02, 35)
    ax.hist(inh_d, bins=bins, color="#888888", alpha=0.65, density=True,
            label=f"in-house→in-house (n={len(inh_d)})")
    ax.hist(hest_d, bins=bins, color="#d62728", alpha=0.65, density=True,
            label=f"HEST-1k→in-house (n={len(hest_d)})")
    ax.axvline(inh_d99, color="k", ls="--", lw=1.2,
               label=f"in-house p99 = {inh_d99:.1f}")
    ax.set_xlabel("centroid distance to nearest in-house section\n(PCA-50 Euclidean)",
                  fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    ax.set_title("D. OOD test: HEST sits FARTHER\n"
                 f"{n_hest_outlier}/{len(hest_d)} HEST sections beyond in-house p99",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle(
        f"NEMO embeddings across the FULL manifest "
        f"({len(iso)} sections, {args.n_per_section}/section subsample, {n_total:,} cells) "
        f"— are HEST-1k public sections OOD?",
        fontsize=12, y=1.02)
    fig.tight_layout()
    os.makedirs(OUT_FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_FIG}/nemo_manifest_ood.{ext}", dpi=170, bbox_inches="tight")
    print(f"\nsaved {OUT_FIG}/nemo_manifest_ood.{{png,pdf}}")
    print(f"saved {OUT_CSV}/per_section_isolation.csv")
    print(f"\nGLOBAL kNN same-source purity = {pur_global:.3f}")


if __name__ == "__main__":
    main()
