#!/usr/bin/env python
"""Area-based intratumoural TSR via nearest-nucleus (Voronoi) tessellation.

The cell-COUNT ratio (tcga_predict_tsr.py / tcga_tile_tsr.py) systematically
under-weights stroma: stroma is collagen-rich ECM with sparse, small fibroblast
nuclei, whereas tumour epithelium has large, densely packed nuclei. The clinical
van Pelt / Mesker TSR (Virchows Arch 2018; validation PMC7887673) is an AREA
estimate of stromal fraction in a 10x field, scored stroma-high (>50%) vs
stroma-low (<=50%), with stroma-high = WORSE prognosis. Counting nuclei flips
the sign because area != count.

This script converts the per-cell point predictions to an AREA ratio:

  1. recover each scored cell's (x, y) in he.zarr space (0.5 um/px),
  2. bin cells into square tiles; a tile is "intratumoural" if it holds
     >= --min_tumour predicted tumour cells (the 10x-field / 4-sides analog),
  3. inside each tumour tile, lay a coarse grid (--area_um, default 8 um/px),
     keep only TISSUE-foreground coarse pixels (H&E not white -> excludes
     whitespace, fat, and acellular mucin/necrosis ~ the protocol exclusions),
  4. assign each kept pixel to its NEAREST nucleus's class (KDTree), but only if
     that nucleus is within --max_dist um (caps runaway territory into gaps),
  5. stroma_fraction = stroma_area / (tumour_area + stroma_area); "other" cells
     (normal epithelium, immune) claim territory but are EXCLUDED from the ratio,
     matching the count-based denominator and the clinical tumour-vs-stroma TSR.

Sparse stromal nuclei thus receive large ECM territories, restoring the area
semantics. Output schema matches tcga_predict_tsr.py (sample_name, n_cells,
n_tumour, n_stroma, stroma_fraction) so it feeds tcga_survival.py directly
(use --stroma_cutoff 0.5 there for the clinical 50% threshold).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import zarr
from scipy.spatial import cKDTree


def _recover_coords(feat_h5: Path, data_h5: Path):
    """Return (x, y) for scored cells in feature-h5 order, + a matched mask."""
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


def _tissue_foreground(rgb: np.ndarray, white_thr: int) -> np.ndarray:
    """rgb: (h, w, 3) uint8. Tissue = not background-white. Returns (h, w) bool.

    H&E background (glass / whitespace / fat lumen) is bright and low-saturation;
    acellular mucin is pale too. Keep a pixel as tissue if it is not near-white,
    i.e. min channel below white_thr (catches eosin pink + haematoxylin blue).
    """
    return rgb.min(axis=2) < white_thr


def voronoi_tsr_for_slide(feat_h5, data_h5, he_zarr, pred, *,
                          tile_px, area_px, max_dist_px, white_thr, min_tumour):
    x, y, matched = _recover_coords(Path(feat_h5), Path(data_h5))
    if matched.sum() == 0:
        return None
    if matched.sum() < len(pred):
        pred = pred[matched]

    # --- intratumoural tiles (same definition as tcga_tile_tsr.py) ---
    tx = (x // tile_px).astype(np.int64)
    ty = (y // tile_px).astype(np.int64)
    tile = tx * 100003 + ty
    is_tum = pred == "tumour"
    tum_per_tile = pd.Series(is_tum).groupby(tile).sum()
    tumour_tiles = set(tum_per_tile.index[tum_per_tile >= min_tumour])
    if not tumour_tiles:
        return dict(n_tumour=0, n_stroma=0, stroma_fraction=np.nan,
                    n_tumour_tiles=0, tumour_area_px=0, stroma_area_px=0)

    he = zarr.open(he_zarr, mode="r")            # (3, H, W) uint8 @ 0.5 um/px
    _, H, W = he.shape

    # class code per nucleus: 0 tumour, 1 stroma, 2 other
    code = np.where(pred == "tumour", 0, np.where(pred == "stroma", 1, 2)).astype(np.int8)
    tree = cKDTree(np.column_stack([x, y]))

    tum_area = str_area = 0
    n_tum = int(is_tum[np.isin(tile, list(tumour_tiles))].sum())
    n_str = int((pred[np.isin(tile, list(tumour_tiles))] == "stroma").sum())

    for t in tumour_tiles:
        txi, tyi = divmod(t, 100003)
        # mod arithmetic above used tx*100003+ty with ty possibly negative? coords>=0 so ok
        px0, py0 = txi * tile_px, tyi * tile_px
        px1, py1 = min(px0 + tile_px, W), min(py0 + tile_px, H)
        if px1 <= px0 or py1 <= py0:
            continue
        # read native H&E for the tile, block-reduce to coarse grid
        block = he[:, py0:py1, px0:px1]                      # (3, h, w)
        block = np.transpose(block, (1, 2, 0))               # (h, w, 3)
        h, w = block.shape[:2]
        ny, nx = h // area_px, w // area_px
        if ny == 0 or nx == 0:
            continue
        crop = block[:ny * area_px, :nx * area_px]
        coarse = crop.reshape(ny, area_px, nx, area_px, 3).mean(axis=(1, 3))
        tissue = _tissue_foreground(coarse.astype(np.uint8), white_thr)   # (ny, nx)
        if not tissue.any():
            continue
        # coarse pixel centres in he.zarr px coords
        gy = py0 + (np.arange(ny) + 0.5) * area_px
        gx = px0 + (np.arange(nx) + 0.5) * area_px
        GX, GY = np.meshgrid(gx, gy)
        pts = np.column_stack([GX[tissue], GY[tissue]])
        d, idx = tree.query(pts, k=1)
        keep = d <= max_dist_px
        cls = code[idx[keep]]
        tum_area += int((cls == 0).sum())
        str_area += int((cls == 1).sum())                    # other (==2) dropped

    denom = tum_area + str_area
    frac = str_area / denom if denom > 0 else np.nan
    return dict(n_tumour=n_tum, n_stroma=n_str, stroma_fraction=frac,
                n_tumour_tiles=len(tumour_tiles),
                tumour_area_px=tum_area, stroma_area_px=str_area)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--per_cell_dir", required=True)
    ap.add_argument("--manifest", required=True,
                    help="CSV with sample_name, data_h5, hematoxylin_eosin_image.")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--tile_um", type=float, default=1000.0)
    ap.add_argument("--um_per_px", type=float, default=0.5)
    ap.add_argument("--area_um", type=float, default=8.0,
                    help="coarse grid pixel edge in um for area accounting.")
    ap.add_argument("--max_dist_um", type=float, default=50.0,
                    help="max nucleus->pixel distance to assign territory.")
    ap.add_argument("--white_thr", type=int, default=220,
                    help="min-channel below this = tissue (not white bg).")
    ap.add_argument("--min_tumour", type=int, default=1)
    args = ap.parse_args()

    tile_px = int(args.tile_um / args.um_per_px)
    area_px = max(1, int(args.area_um / args.um_per_px))
    max_dist_px = args.max_dist_um / args.um_per_px

    man = pd.read_csv(args.manifest)
    he_of = {str(r["sample_name"]).strip(): str(r["hematoxylin_eosin_image"]).strip()
             for _, r in man.iterrows()}
    data_of = {str(r["sample_name"]).strip(): str(r["data_h5"]).strip()
               for _, r in man.iterrows()}
    feat_dir = Path(args.features_dir); pcd = Path(args.per_cell_dir)
    print(f"[voronoi] tile={tile_px}px area={area_px}px max_dist={max_dist_px:.0f}px "
          f"white_thr={args.white_thr} min_tumour={args.min_tumour}")

    rows = []
    for npz in sorted(pcd.glob("*.npz")):
        sample = npz.stem
        feat_h5 = feat_dir / f"{sample}.h5"
        data_h5 = data_of.get(sample); he_zarr = he_of.get(sample)
        if not feat_h5.exists() or not data_h5 or not Path(data_h5).exists() \
                or not he_zarr or not Path(he_zarr).exists():
            print(f"[skip] {sample}: missing feature/data/he", flush=True)
            continue
        pred = np.load(npz, allow_pickle=True)["pred"].astype(str)
        try:
            res = voronoi_tsr_for_slide(
                feat_h5, data_h5, he_zarr, pred,
                tile_px=tile_px, area_px=area_px, max_dist_px=max_dist_px,
                white_thr=args.white_thr, min_tumour=args.min_tumour)
        except Exception as e:
            print(f"[err] {sample}: {e}", flush=True)
            continue
        if res is None:
            print(f"[skip] {sample}: no cell_id matched data_h5", flush=True)
            continue
        n_cells = res["n_tumour"] + res["n_stroma"]
        row = {"sample_name": sample, "n_cells": n_cells, **res}
        rows.append(row)
        print(f"  {sample}: tum_tiles={res['n_tumour_tiles']} "
              f"area(tum={res['tumour_area_px']} str={res['stroma_area_px']}) "
              f"stroma_frac={res['stroma_fraction']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\n[voronoi-tsr] {len(df)} slides -> {args.out_csv}")
    if len(df):
        print(df[["n_cells", "n_tumour_tiles", "stroma_fraction"]].describe())


if __name__ == "__main__":
    main()
