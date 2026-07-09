#!/usr/bin/env python
"""Intratumoural tile-restricted TSR from per-cell predictions.

Classic TSR is measured *inside the tumour* (a high-power field where tumour
borders the region), not slide-wide -- a whole-slide average is diluted by
normal mucosa / muscularis. We approximate the intratumoural restriction
without a separate tissue classifier:

  1. recover each scored cell's (x, y) from the source data_h5 (the feature
     cell_id is "<sample>_n<idx>", an index into that slide's cell array),
  2. bin cells into square tiles (default 1000 um ~ a 10x field),
  3. call a tile "tumour tile" if it holds >= --min_tumour predicted tumour
     cells (this is the intratumoural region),
  4. within tumour tiles only, stroma_fraction = N_stroma / (N_tumour +
     N_stroma). "other" cells (normal epithelium etc.) are excluded from the
     ratio, as in tcga_predict_tsr.py.

The ratio is still cell-count based *within* a tile, but the field is now
localised to tumour, which is the part of the protocol that matters most.
Coordinates live in the he.zarr space (resolution_um_per_px, default 0.5),
so --tile_um is converted to pixels with --um_per_px.

Output schema matches tcga_predict_tsr.py (sample_name, n_cells, n_tumour,
n_stroma, stroma_fraction) so it feeds tcga_survival.py directly, plus tile
diagnostics.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def _recover_coords(feat_h5: Path, data_h5: Path):
    """Return (x, y) for the scored cells, aligned to feature-h5 order.

    The "_n<idx>" suffix is a cell_id from the FULL segmentation, not a row
    index into the 30k-capped data_h5 (whose ids are sparse), so we join on the
    cell_id string. Returns NaN-free arrays + a mask of cells that matched.
    """
    with h5py.File(feat_h5, "r") as f:
        cids = [c.decode() for c in f["cell_id"][:]]
    with h5py.File(data_h5, "r") as f:           # only the small coord arrays
        dcid = [c.decode() for c in f["cell_id"][:]]
        dx = f["x_pixel"][:]; dy = f["y_pixel"][:]
    row = {c: i for i, c in enumerate(dcid)}
    matched = np.array([c in row for c in cids])
    ridx = np.array([row[c] for c in cids if c in row])
    x = dx[ridx].astype(np.int64); y = dy[ridx].astype(np.int64)
    return x, y, matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True,
                    help="dir of <sample>.h5 SIGMMA features (for cell_id).")
    ap.add_argument("--per_cell_dir", required=True,
                    help="dir of <sample>.npz with a 'pred' array (cell order "
                         "matches features_dir).")
    ap.add_argument("--manifest", required=True,
                    help="CSV with sample_name + data_h5 (source coords).")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--tile_um", type=float, default=1000.0,
                    help="square tile edge in microns (default 1000 ~ 10x field).")
    ap.add_argument("--um_per_px", type=float, default=0.5,
                    help="coord resolution (he.zarr resolution_um_per_px).")
    ap.add_argument("--min_tumour", type=int, default=1,
                    help="min predicted tumour cells for a tile to count as "
                         "intratumoural.")
    args = ap.parse_args()

    tile_px = args.tile_um / args.um_per_px
    man = pd.read_csv(args.manifest)
    data_of = {str(r["sample_name"]).strip(): str(r["data_h5"]).strip()
               for _, r in man.iterrows()}
    feat_dir = Path(args.features_dir); pcd = Path(args.per_cell_dir)
    print(f"[tile] {args.tile_um:.0f}um = {tile_px:.0f}px  min_tumour={args.min_tumour}")

    rows = []
    for npz in sorted(pcd.glob("*.npz")):
        sample = npz.stem
        feat_h5 = feat_dir / f"{sample}.h5"
        data_h5 = data_of.get(sample)
        if not feat_h5.exists() or not data_h5 or not Path(data_h5).exists():
            print(f"[skip] {sample}: missing feature/data h5")
            continue
        pred = np.load(npz, allow_pickle=True)["pred"].astype(str)
        x, y, matched = _recover_coords(feat_h5, Path(data_h5))
        if matched.sum() == 0:
            print(f"[skip] {sample}: no cell_id matched data_h5")
            continue
        if matched.sum() < len(pred):
            pred = pred[matched]                  # drop cells absent from coords

        tx = (x // int(tile_px)).astype(np.int64)
        ty = (y // int(tile_px)).astype(np.int64)
        tile = tx * 100003 + ty                       # cheap 2D hash
        is_tum = pred == "tumour"
        # tumour cells per tile -> tumour tiles
        tum_per_tile = pd.Series(is_tum).groupby(tile).sum()
        tumour_tiles = set(tum_per_tile.index[tum_per_tile >= args.min_tumour])
        in_tum = np.array([t in tumour_tiles for t in tile])

        sub = pred[in_tum]
        n_tum = int((sub == "tumour").sum())
        n_str = int((sub == "stroma").sum())
        denom = n_tum + n_str
        frac = n_str / denom if denom > 0 else np.nan
        rows.append({
            "sample_name": sample,
            "n_cells": denom,                          # tum+str in tumour tiles
            "n_tumour": n_tum, "n_stroma": n_str,
            "stroma_fraction": frac,
            "n_cells_total": int(len(pred)),
            "n_tiles": int(pd.Series(tile).nunique()),
            "n_tumour_tiles": int(len(tumour_tiles)),
            "n_cells_in_tumour_tiles": int(in_tum.sum()),
        })
        print(f"  {sample}: tum_tiles={len(tumour_tiles)} "
              f"intratum_cells={int(in_tum.sum())} tum={n_tum} str={n_str} "
              f"stroma_frac={frac:.3f}", flush=True)

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\n[tile-tsr] {len(df)} slides -> {args.out_csv}")
    if len(df):
        print(df[["n_cells", "n_tumour_tiles", "stroma_fraction"]].describe())


if __name__ == "__main__":
    main()
