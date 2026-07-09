#!/usr/bin/env python
"""Score an arbitrary gene signature and overlay high-scoring cells on H&E.

Used to flag candidate regions (e.g. proliferation as a tumour-associated, but
NOT tumour-specific, signal — regeneration/inflammation also proliferate). There
is no pathology annotation for these sections, so output is suggestive only.
"""
import argparse, csv
from pathlib import Path
import numpy as np
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIGNATURES = {
    # full cell-cycle / proliferation module (all 12 present in the 5K panel)
    "proliferation": ["MKI67", "TOP2A", "PCNA", "CCNB1", "CCNB2", "CDK1",
                       "AURKA", "BIRC5", "CENPF", "UBE2C", "MCM2", "PLK1"],
    "cholangio": ["SOX9", "EPCAM", "MUC1", "KRT19", "KRT7"],
    "progenitor": ["PROM1", "THY1", "NCAM1", "EPCAM"],
}
MANIFESTS = [
    "/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_MANIFEST_train.csv",
    "/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_MANIFEST_val.csv",
]


def resolve_paths(section):
    for mf in MANIFESTS:
        if not Path(mf).exists():
            continue
        for row in csv.DictReader(open(mf)):
            if row["sample_name"] == section:
                return row["spatial_preprocessed"], row.get("hematoxylin_eosin_image", "")
    raise SystemExit(f"section {section} not found")


def load_he(he_zarr, ds):
    import zarr
    z = zarr.open(he_zarr, mode="r")
    return np.transpose(np.asarray(z[:, ::ds, ::ds]), (1, 2, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="DF29-LIV-0-FFPE-1-S1-i")
    ap.add_argument("--signature", default="proliferation", choices=list(SIGNATURES))
    ap.add_argument("--outdir", default="figures/cross_tissue_retrieval/emt")
    ap.add_argument("--point_size", type=float, default=1.5)
    ap.add_argument("--he_downsample", type=int, default=8)
    ap.add_argument("--top_pct", type=float, default=95.0,
                    help="cells above this score percentile are highlighted")
    args = ap.parse_args()

    genes = SIGNATURES[args.signature]
    h5, he_zarr = resolve_paths(args.section)
    a = ad.read_h5ad(h5)
    if "gene_name" in a.var.columns:
        a.var_names = [str(g) for g in a.var["gene_name"].values]
    a.var_names = [str(g) for g in a.var_names]
    a.var_names_make_unique()
    present = [g for g in genes if g in a.var_names]
    print(f"[{args.signature}] {len(present)}/{len(genes)} present: {present}")
    sc.tl.score_genes(a, present, score_name="score", ctrl_size=50)
    s = a.obs["score"].values
    thr = np.percentile(s, args.top_pct)
    hi = s > thr
    print(f"[score] top{100-args.top_pct:.0f}% threshold={thr:.3f}  high cells={hi.sum():,} ({hi.mean():.1%})")

    xy = a.obsm["spatial"]; x, y = xy[:, 0], xy[:, 1]
    order = np.random.default_rng(0).permutation(a.n_obs)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    he = load_he(he_zarr, args.he_downsample) if (he_zarr and Path(he_zarr).exists()) else None
    ds = args.he_downsample

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    # panel 1: H&E
    if he is not None:
        H, W = he.shape[:2]; ext = [0, W*ds, H*ds, 0]
        axes[0].imshow(he, extent=ext, interpolation="nearest")
        axes[0].set_xlim(0, W*ds); axes[0].set_ylim(H*ds, 0)
    axes[0].set_title("H&E", fontsize=11)

    # panel 2: continuous score (all cells)
    vmax = np.percentile(s, 99.5)
    sm = axes[1].scatter(x[order], y[order], c=s[order], s=args.point_size,
                         cmap="inferno", vmin=0, vmax=vmax, linewidths=0, rasterized=True)
    axes[1].set_title(f"{args.signature} score ({len(present)} genes)", fontsize=11)
    plt.colorbar(sm, ax=axes[1], fraction=0.04, pad=0.02).ax.tick_params(labelsize=7)

    # panel 3: high-scoring cells on H&E
    if he is not None:
        axes[2].imshow(he, extent=ext, interpolation="nearest")
        axes[2].set_xlim(0, W*ds); axes[2].set_ylim(H*ds, 0)
    axes[2].scatter(x[hi], y[hi], s=args.point_size+1, c="#00E5FF", linewidths=0,
                    rasterized=True, label=f"top {100-args.top_pct:.0f}% (n={hi.sum():,})")
    axes[2].set_title(f"high {args.signature} on H&E", fontsize=11)
    axes[2].legend(loc="upper right", fontsize=8, markerscale=3, framealpha=0.9)

    for ax in axes:
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        if he is None: ax.invert_yaxis()
        for sp in ax.spines.values(): sp.set_visible(False)

    fig.suptitle(f"{args.signature} - {args.section}  ({a.n_obs:,} cells)", fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    stem = outdir / f"{args.signature}_he_{args.section}"
    for ext_ in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext_}", dpi=180, bbox_inches="tight")
    print(f"[saved] {stem}.png / .pdf")


if __name__ == "__main__":
    main()
