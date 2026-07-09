#!/usr/bin/env python
"""Diagnose where cells are lost across the TSR pipeline for given slides.

Reports per slide: nuclei segmented (data_h5), cells with SIGMMA features
(feat_h5 = the PREDICTED cells), predictions in npz, matched (feat cell_id found
in data_h5), and the implied cell density (cells / mm^2) over the H&E tissue
area + within one 1.86mm field. Pinpoints whether sparse prediction dots are a
SEGMENTATION (few nuclei) or a FEATURE-EXTRACTION/matching loss.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcga_voronoi_tsr import _recover_coords, _tissue_foreground


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--per_cell_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--um_per_px", type=float, default=0.5)
    ap.add_argument("--field_um", type=float, default=1860.0)
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    data_of = {str(r["sample_name"]).strip(): str(r["data_h5"]).strip()
               for _, r in man.iterrows()}
    he_of = {str(r["sample_name"]).strip(): str(r["hematoxylin_eosin_image"]).strip()
             for _, r in man.iterrows()}
    feat_dir = Path(args.features_dir); pcd = Path(args.per_cell_dir)
    tile_px = int(args.field_um / args.um_per_px)

    for s in args.samples:
        feat_h5 = feat_dir / f"{s}.h5"; npz = pcd / f"{s}.npz"
        data_h5 = data_of.get(s); he_zarr = he_of.get(s)
        print(f"\n===== {s.split('.')[0]} =====")
        if not (feat_h5.exists() and data_h5 and Path(data_h5).exists()):
            print("  missing feat/data h5"); continue
        with h5py.File(data_h5, "r") as f:
            n_nuc = len(f["cell_id"]); dx = f["x_pixel"][:]; dy = f["y_pixel"][:]
            data_keys = list(f.keys())
        with h5py.File(feat_h5, "r") as f:
            n_feat = len(f["cell_id"]); feat_keys = list(f.keys())
        n_pred = len(np.load(npz, allow_pickle=True)["pred"]) if npz.exists() else -1
        _, _, matched = _recover_coords(feat_h5, data_h5)

        # tissue area from H&E (coarse) for density
        tissue_mm2 = np.nan; dens_tissue = np.nan
        if he_zarr and Path(he_zarr).exists():
            he = zarr.open(he_zarr, mode="r"); _, H, W = he.shape
            s2 = max(1, int(max(H, W) / 1500))
            thumb = np.transpose(he[:, ::s2, ::s2], (1, 2, 0)).astype(np.uint8)
            tiss = _tissue_foreground(thumb, 220)
            px_um = args.um_per_px * s2
            tissue_mm2 = tiss.sum() * (px_um / 1000) ** 2
            dens_tissue = n_nuc / tissue_mm2 if tissue_mm2 else np.nan
            slide_mm = (W * args.um_per_px / 1000, H * args.um_per_px / 1000)
        else:
            slide_mm = (np.nan, np.nan)

        # densest 1.86mm field nuclei count (data nuclei)
        tx = (dx // tile_px).astype(np.int64); ty = (dy // tile_px).astype(np.int64)
        vc = pd.Series(list(zip(tx, ty))).value_counts()
        max_field = int(vc.iloc[0]) if len(vc) else 0

        print(f"  nuclei segmented (data_h5)   : {n_nuc}")
        print(f"  cells w/ SIGMMA feat (feat_h5): {n_feat}  ({100*n_feat/max(n_nuc,1):.1f}% of nuclei)")
        print(f"  predictions (npz)            : {n_pred}")
        print(f"  matched feat->data           : {int(matched.sum())} / {len(matched)}")
        print(f"  slide ~{slide_mm[0]:.1f} x {slide_mm[1]:.1f} mm; tissue {tissue_mm2:.1f} mm^2")
        print(f"  density (nuclei/mm^2 tissue) : {dens_tissue:.0f}")
        print(f"  densest 1.86mm field nuclei  : {max_field}  (field=3.46mm^2)")
        print(f"  data keys: {data_keys}")
        print(f"  feat keys: {feat_keys}")


if __name__ == "__main__":
    main()
