#!/usr/bin/env python
"""TCGA H&E -> regular TILE GRID -> per-tile ``data.h5`` (no nuclei, no labels).

Option-A (tile-level tumour/stroma map) replacement for the per-nucleus
``tcga_segment_and_build_data.py``. Instead of segmenting nuclei (which only
covers a random ~3-14% of tissue and is far too sparse for an AREA ratio), we
lay a UNIFORM grid over the tissue foreground and treat each grid point as a
"cell": x/y = tile centre, mask_<scale> = a FULL square so the unmodified
``encode_with_mask_pool`` averages every patch token of the crop -> a tile-level
SIGMMA embedding (plus the mask-free CLS). The downstream feature/predict stages
are byte-identical to the per-cell path, so a tile head trained on the SAME grid
embeddings of the RCC slides transfers directly.

Why a grid fixes the sparsity problem:
  * 100% tissue coverage (every foreground tile gets a point);
  * every tile is the same area -> TSR becomes a clean tile-COUNT ratio with no
    Voronoi / density-gate bias.

Output schema matches ``process_slide_data`` so ``process_slide_features`` can
consume it unchanged:
  attrs: sample_name, normalized_sample, split, scale_sizes, label_names=[]
  datasets: cell_id (S), x_pixel, y_pixel (int32),
            bbox_{micro,meso,macro} (N,2 int32),
            mask_{micro,meso,macro}  (N,S,S bool, all True)

Coordinates are zarr pixels (0.5 um/px). CPU only (no model, no cellpose) -> run
in ``.venv`` (zarr + h5py + numpy).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import h5py
import numpy as np
import zarr

SCALE_NAMES = ("micro", "meso", "macro")


def tissue_foreground_mask(he_zarr, stride: int = 32, white_thresh: int = 220):
    """Coarse (strided) boolean tissue mask + the stride used, like the seg path."""
    thumb = np.asarray(he_zarr[:, ::stride, ::stride]).mean(axis=0)   # (h, w)
    return thumb < white_thresh                                       # True = tissue


def grid_points(he_zarr, tile_px: int, *, thumb_stride: int = 32,
                white_thresh: int = 220, min_frac: float = 0.5):
    """Tile-centre (x, y) pixel coords for every grid cell whose tissue fraction
    (from the strided thumbnail) is >= min_frac. Grid step == tile_px."""
    _, H, W = he_zarr.shape
    tissue = tissue_foreground_mask(he_zarr, thumb_stride, white_thresh)
    th, tw = tissue.shape
    step = max(1, tile_px // thumb_stride)                            # thumb px / tile
    xs, ys = [], []
    for ty in range(0, H - tile_px + 1, tile_px):
        for tx in range(0, W - tile_px + 1, tile_px):
            sy, sx = ty // thumb_stride, tx // thumb_stride
            block = tissue[sy:sy + step, sx:sx + step]
            if block.size and block.mean() >= min_frac:
                xs.append(tx + tile_px // 2)
                ys.append(ty + tile_px // 2)
    return np.asarray(xs, np.int32), np.asarray(ys, np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", required=True, help="hematoxylin_eosin .he.zarr path")
    ap.add_argument("--sample", required=True, help="sample name (slide id)")
    ap.add_argument("--out_h5", required=True)
    ap.add_argument("--scale_sizes", type=int, nargs=3, default=[64, 128, 256])
    ap.add_argument("--tile_um", type=float, default=32.0,
                    help="grid step (tile side) in microns. Map resolution; "
                         "keep small so tumour/stroma boundaries are resolved.")
    ap.add_argument("--um_per_px", type=float, default=0.5)
    ap.add_argument("--white_thr", type=int, default=220)
    ap.add_argument("--thumb_stride", type=int, default=32)
    ap.add_argument("--min_frac", type=float, default=0.5,
                    help="min tissue fraction of a tile to keep it.")
    ap.add_argument("--max_tiles", type=int, default=0,
                    help="cap (seeded random subsample) for a quick pilot; 0=all.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tile_px = int(round(args.tile_um / args.um_per_px))
    he = zarr.open(args.zarr, mode="r")
    _, H, W = he.shape
    print(f"[{args.sample}] zarr {W}x{H}  tile={tile_px}px ({args.tile_um}um)",
          flush=True)

    x_pixel, y_pixel = grid_points(
        he, tile_px, thumb_stride=args.thumb_stride,
        white_thresh=args.white_thr, min_frac=args.min_frac)
    n_tot = len(x_pixel)
    if n_tot == 0:
        raise SystemExit(f"[{args.sample}] no tissue tiles")
    if args.max_tiles and n_tot > args.max_tiles:
        idx = np.sort(np.random.default_rng(args.seed)
                      .choice(n_tot, args.max_tiles, replace=False))
        x_pixel, y_pixel = x_pixel[idx], y_pixel[idx]
    N = len(x_pixel)
    print(f"[{args.sample}] tissue tiles: {n_tot}  keeping {N}", flush=True)

    cell_ids = np.array([f"{args.sample}_t{i}" for i in range(N)], dtype="S")
    bboxes = {s: np.empty((N, 2), np.int32) for s in SCALE_NAMES}
    masks = {}
    for sname, ssize in zip(SCALE_NAMES, args.scale_sizes):
        bx = np.clip(x_pixel - ssize // 2, 0, W - ssize)
        by = np.clip(y_pixel - ssize // 2, 0, H - ssize)
        bboxes[sname][:, 0] = bx
        bboxes[sname][:, 1] = by
        # FULL-square mask -> encode_with_mask_pool averages all patch tokens
        # of the crop => a tile-level (whole-patch) pooled embedding.
        masks[sname] = np.ones((N, ssize, ssize), dtype=bool)

    out = Path(args.out_h5); out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.attrs["sample_name"] = args.sample
        f.attrs["normalized_sample"] = args.sample
        f.attrs["split"] = "test"
        f.attrs["scale_sizes"] = list(args.scale_sizes)
        f.attrs["label_names"] = np.array([], dtype="S")
        f.attrs["tile_um"] = float(args.tile_um)
        f.create_dataset("cell_id", data=cell_ids)
        f.create_dataset("x_pixel", data=x_pixel)
        f.create_dataset("y_pixel", data=y_pixel)
        for s in SCALE_NAMES:
            f.create_dataset(f"bbox_{s}", data=bboxes[s])
            f.create_dataset(f"mask_{s}", data=masks[s], compression="gzip",
                             compression_opts=4, chunks=True)
    print(f"[{args.sample}] wrote {out} ({N} tiles)", flush=True)


if __name__ == "__main__":
    main()
