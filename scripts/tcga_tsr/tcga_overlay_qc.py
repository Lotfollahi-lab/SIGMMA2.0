#!/usr/bin/env python
"""Visual QC of per-cell tumour/stroma/other predictions on TCGA COAD H&E.

The RCC-trained 3-class probe yields a *protective* (literature-opposite) TSR
sign on COAD across both count and area estimators, and the count->area fix
ruled out the unit-mismatch hypothesis (see project_tcga_tsr_external_cohort).
The remaining prime suspect is probe LABEL TRANSFER: does the probe actually
call epithelial tumour nests "tumour" and collagen/fibroblast stroma "stroma"
on colorectal H&E, or has the renal-trained boundary swapped them?

This renders, per slide, the densest intratumoural tile (same definition as
tcga_tile_tsr / tcga_voronoi_tsr: a 1000 um tile with >= min_tumour predicted
tumour cells) as an H&E background. To match the REPORTED survival estimator,
the overlay colours the nearest-nucleus (Voronoi) AREA territory exactly as
tcga_voronoi_tsr (tissue-foreground only via white_thr, capped at max_dist um,
"other" territory excluded from the ratio), and each panel's stroma_frac is the
AREA fraction str_area/(tum_area+str_area) -- NOT a cell count. Per-nucleus dots
(tumour=red, stroma=blue, other=grey) are drawn on top so the class labelling
itself stays verifiable. A pathologist eye on these panels decides whether the
sign flip is a labelling artefact.

Inputs match the voronoi/predict jobscripts:
  --features_dir  cache/.../sigmma_36852_step2000   (per-slide <sample>.h5, cell_id)
  --per_cell_dir  outputs/tcga_coad_tsr/per_cell_pred (<sample>.npz, key "pred")
  --manifest      configs/tcga_coad_manifest.csv     (sample_name,data_h5,he image)
"""
from __future__ import annotations
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import zarr
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CLASS_COLOR = {"tumour": "#d62728", "stroma": "#1f77b4", "other": "#999999"}
CLASS_CODE = {"tumour": 0, "stroma": 1, "other": 2}
# RGBA fill for Voronoi territory (tumour/stroma solid-ish, other faint)
TERR_RGB = {0: (0.84, 0.15, 0.16), 1: (0.12, 0.47, 0.71), 2: (0.60, 0.60, 0.60)}
TERR_ALPHA = {0: 0.45, 1: 0.45, 2: 0.12}


def _tissue_foreground(rgb: np.ndarray, white_thr: int) -> np.ndarray:
    """rgb: (h, w, 3) uint8 -> (h, w) bool. Tissue = min channel < white_thr.

    Identical to tcga_voronoi_tsr._tissue_foreground so the QC territory matches
    the reported area estimator (drops near-white glass/whitespace/mucin).
    """
    return rgb.min(axis=2) < white_thr


def _territory(he, tree, code, px0, py0, win_px, *, area_px, max_dist_px, white_thr):
    """Voronoi area territory for a crop window, matching tcga_voronoi_tsr.

    Returns (cls_map (ny,nx) int8 in {-1,0,1,2}, ny, nx) where -1 = background /
    beyond max_dist, or None if the window has no coarse cells.
    """
    _, H, W = he.shape
    px1, py1 = min(px0 + win_px, W), min(py0 + win_px, H)
    block = np.transpose(he[:, py0:py1, px0:px1], (1, 2, 0))   # (h, w, 3) uint8
    h, w = block.shape[:2]
    ny, nx = h // area_px, w // area_px
    if ny == 0 or nx == 0:
        return None
    crop = block[:ny * area_px, :nx * area_px]
    coarse = crop.reshape(ny, area_px, nx, area_px, 3).mean(axis=(1, 3))
    tissue = _tissue_foreground(coarse.astype(np.uint8), white_thr)   # (ny, nx)
    cls_map = np.full((ny, nx), -1, np.int8)
    if tissue.any():
        gy = py0 + (np.arange(ny) + 0.5) * area_px
        gx = px0 + (np.arange(nx) + 0.5) * area_px
        GX, GY = np.meshgrid(gx, gy)
        pts = np.column_stack([GX[tissue], GY[tissue]])
        d, idx = tree.query(pts, k=1)
        c = code[idx].astype(np.int8)
        c[d > max_dist_px] = -1
        cls_map[tissue] = c
    return cls_map, ny, nx


def _recover_coords(feat_h5: Path, data_h5: Path):
    """Return (x, y) for scored cells in feature-h5 order, + a matched mask.

    Copied from tcga_voronoi_tsr so the QC reads the EXACT same cell ordering
    the survival pipeline scored.
    """
    with h5py.File(feat_h5, "r") as f:
        cids = [c.decode() for c in f["cell_id"][:]]
    with h5py.File(data_h5, "r") as f:
        dcid = [c.decode() for c in f["cell_id"][:]]
        dx = f["x_pixel"][:]; dy = f["y_pixel"][:]
    row = {c: i for i, c in enumerate(dcid)}
    matched = np.array([c in row for c in cids])
    ridx = np.array([row[c] for c in cids if c in row])
    x = dx[ridx].astype(np.int64); y = dy[ridx].astype(np.int64)
    return x, y, matched


def _densest_tumour_tile(x, y, pred, tile_px, min_tumour, rank=0):
    """Return (px0, py0) of the rank-th tumour tile by total scored-cell count."""
    tx = (x // tile_px).astype(np.int64)
    ty = (y // tile_px).astype(np.int64)
    tile = tx * 100003 + ty
    is_tum = pred == "tumour"
    s = pd.DataFrame({"tile": tile, "is_tum": is_tum})
    tum_per = s.groupby("tile")["is_tum"].sum()
    tot_per = s.groupby("tile")["is_tum"].size()
    tumour_tiles = tum_per.index[tum_per >= min_tumour]
    if len(tumour_tiles) == 0:
        return None
    order = tot_per[tumour_tiles].sort_values(ascending=False)
    if rank >= len(order):
        return None
    t = order.index[rank]
    txi, tyi = divmod(int(t), 100003)
    return txi * tile_px, tyi * tile_px


def _draw(ax, he, x, y, pred, tree, code, px0, py0, win_px, title, dot_size, *,
          area_px, max_dist_px, white_thr):
    """Crop a win_px square at (px0,py0): H&E + Voronoi area overlay + nuclei.

    The stroma_frac in the title is the AREA fraction over this window, computed
    the same way as tcga_voronoi_tsr (tissue-foreground, max_dist cap, "other"
    territory excluded), so it is directly comparable to the survival estimator.
    """
    _, H, W = he.shape
    px1, py1 = min(px0 + win_px, W), min(py0 + win_px, H)
    block = np.transpose(he[:, py0:py1, px0:px1], (1, 2, 0))   # (h,w,3) uint8
    ax.imshow(block, interpolation="nearest", zorder=1)

    # Voronoi area territory overlay (RGBA at coarse resolution)
    tum_a = str_a = 0
    terr = _territory(he, tree, code, px0, py0, win_px,
                      area_px=area_px, max_dist_px=max_dist_px, white_thr=white_thr)
    if terr is not None:
        cls_map, ny, nx = terr
        rgba = np.zeros((ny, nx, 4), float)
        for c, (r, g, b) in TERR_RGB.items():
            m = cls_map == c
            rgba[m, 0], rgba[m, 1], rgba[m, 2], rgba[m, 3] = r, g, b, TERR_ALPHA[c]
        ax.imshow(rgba, extent=(0, nx * area_px, ny * area_px, 0),
                  interpolation="nearest", zorder=2)
        tum_a = int((cls_map == 0).sum())
        str_a = int((cls_map == 1).sum())

    # per-nucleus dots on top (label verification)
    sel = (x >= px0) & (x < px1) & (y >= py0) & (y < py1)
    for cls, col in CLASS_COLOR.items():
        m = sel & (pred == cls)
        if m.any():
            ax.scatter(x[m] - px0, y[m] - py0, s=dot_size, c=col,
                       edgecolors="white", linewidths=0.2, alpha=0.9, zorder=3)
    n = int(sel.sum())
    nt = int((sel & (pred == "tumour")).sum())
    ns = int((sel & (pred == "stroma")).sum())
    frac = str_a / (tum_a + str_a) if (tum_a + str_a) else float("nan")
    ax.set_title(f"{title}\nN={n} (tum={nt} str={ns})  "
                 f"area_frac={frac:.2f}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def qc_slide(sample, feat_h5, data_h5, he_zarr, pred, out_png, *,
             tile_px, zoom_px, slide_frac, dot_full, dot_zoom,
             area_px, max_dist_px, white_thr):
    x, y, matched = _recover_coords(Path(feat_h5), Path(data_h5))
    if matched.sum() == 0:
        print(f"[skip] {sample}: no cell_id matched"); return False
    if matched.sum() < len(pred):
        pred = pred[matched]
    he = zarr.open(he_zarr, mode="r")            # (3,H,W) uint8 @ 0.5 um/px
    # KDTree + class codes over ALL scored nuclei (matches tcga_voronoi_tsr)
    tree = cKDTree(np.column_stack([x, y]))
    code = np.where(pred == "tumour", 0,
                    np.where(pred == "stroma", 1, 2)).astype(np.int8)

    tile = _densest_tumour_tile(x, y, pred, tile_px, min_tumour=1, rank=0)
    if tile is None:
        print(f"[skip] {sample}: no tumour tile"); return False
    tpx0, tpy0 = tile
    # zoom = window centred on the DENSEST scored-cell cluster inside the tile
    # (the geometric centre often lands on acellular gland lumen / whitespace).
    insel = (x >= tpx0) & (x < tpx0 + tile_px) & (y >= tpy0) & (y < tpy0 + tile_px)
    if insel.sum() >= 5:
        # coarse 2D histogram of cell positions, pick the hottest zoom-sized cell
        bins = max(1, tile_px // zoom_px)
        hx = ((x[insel] - tpx0) * bins // tile_px).clip(0, bins - 1)
        hy = ((y[insel] - tpy0) * bins // tile_px).clip(0, bins - 1)
        hist = np.zeros((bins, bins), int)
        np.add.at(hist, (hy, hx), 1)
        by, bx = np.unravel_index(int(hist.argmax()), hist.shape)
        zpx0 = tpx0 + int(bx * tile_px / bins)
        zpy0 = tpy0 + int(by * tile_px / bins)
    else:
        zpx0 = tpx0 + (tile_px - zoom_px) // 2
        zpy0 = tpy0 + (tile_px - zoom_px) // 2
    zpx0 = min(zpx0, tpx0 + tile_px - zoom_px)
    zpy0 = min(zpy0, tpy0 + tile_px - zoom_px)

    # slide-wide class balance (probe routing) printed for context
    vc = pd.Series(pred).value_counts()
    print(f"      {sample.split('.')[0]} slide-wide: "
          + " ".join(f"{k}={int(vc.get(k,0))}" for k in CLASS_COLOR))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.7))
    _draw(axes[0], he, x, y, pred, tree, code, tpx0, tpy0, tile_px,
          f"intratumoural tile ({tile_px*0.5:.0f}um)", dot_full,
          area_px=area_px, max_dist_px=max_dist_px, white_thr=white_thr)
    # mark the zoom box on the overview at its actual location
    axes[0].add_patch(plt.Rectangle((zpx0 - tpx0, zpy0 - tpy0), zoom_px, zoom_px,
                                    fill=False, ec="yellow", lw=1.4))
    _draw(axes[1], he, x, y, pred, tree, code, zpx0, zpy0, zoom_px,
          f"zoom ({zoom_px*0.5:.0f}um, native res)", dot_zoom,
          area_px=area_px, max_dist_px=max_dist_px, white_thr=white_thr)

    handles = [Line2D([0], [0], marker="o", ls="", mfc=c, mec="none", label=k)
               for k, c in CLASS_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(sample, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {sample} -> {out_png}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--per_cell_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--samples", nargs="+", required=True,
                    help="sample_name(s) to QC")
    ap.add_argument("--um_per_px", type=float, default=0.5)
    ap.add_argument("--tile_um", type=float, default=1000.0)
    ap.add_argument("--zoom_um", type=float, default=250.0)
    ap.add_argument("--slide_frac", type=float, default=0.0)  # reserved
    ap.add_argument("--dot_full", type=float, default=3.0)
    ap.add_argument("--dot_zoom", type=float, default=22.0)
    # Voronoi area-territory params -- keep in sync with tcga_voronoi_tsr defaults
    ap.add_argument("--area_um", type=float, default=8.0,
                    help="coarse grid pixel edge in um for area territory.")
    ap.add_argument("--max_dist_um", type=float, default=50.0,
                    help="max nucleus->pixel distance to assign territory.")
    ap.add_argument("--white_thr", type=int, default=220,
                    help="min-channel below this = tissue (not white bg).")
    args = ap.parse_args()

    tile_px = int(args.tile_um / args.um_per_px)
    zoom_px = int(args.zoom_um / args.um_per_px)
    area_px = max(1, int(args.area_um / args.um_per_px))
    max_dist_px = args.max_dist_um / args.um_per_px

    man = pd.read_csv(args.manifest)
    he_of = {str(r["sample_name"]).strip(): str(r["hematoxylin_eosin_image"]).strip()
             for _, r in man.iterrows()}
    data_of = {str(r["sample_name"]).strip(): str(r["data_h5"]).strip()
               for _, r in man.iterrows()}
    feat_dir = Path(args.features_dir); pcd = Path(args.per_cell_dir)

    for sample in args.samples:
        feat_h5 = feat_dir / f"{sample}.h5"
        npz = pcd / f"{sample}.npz"
        data_h5 = data_of.get(sample); he_zarr = he_of.get(sample)
        miss = [n for n, p in [("feat", feat_h5), ("npz", npz)]
                if not Path(p).exists()]
        if miss or not data_h5 or not Path(data_h5).exists() \
                or not he_zarr or not Path(he_zarr).exists():
            print(f"[skip] {sample}: missing {miss or 'data/he'}"); continue
        pred = np.load(npz, allow_pickle=True)["pred"].astype(str)
        out_png = Path(args.out_dir) / f"{sample.split('.')[0]}_qc.png"
        qc_slide(sample, feat_h5, data_h5, he_zarr, pred, out_png,
                 tile_px=tile_px, zoom_px=zoom_px, slide_frac=args.slide_frac,
                 dot_full=args.dot_full, dot_zoom=args.dot_zoom,
                 area_px=area_px, max_dist_px=max_dist_px, white_thr=args.white_thr)


if __name__ == "__main__":
    main()
