#!/usr/bin/env python
"""TCGA H&E → nucleus segmentation → per-cell ``data.h5`` (no labels).

Produces the SAME HDF5 schema that ``process_slide_data`` writes, so the
unmodified ``process_slide_features`` (SIGMMA stage) can consume it:

  attrs: sample_name, normalized_sample, split, scale_sizes, label_names=[]
  datasets: cell_id (S), x_pixel, y_pixel (int32),
            bbox_{micro,meso,macro} (N,2 int32),
            mask_{micro,meso,macro}  (N,S,S bool)

Pipeline per slide:
  1. Tissue foreground from a strided thumbnail (drop white/background tiles).
  2. Random foreground 512px tiles -> Cellpose ``nuclei`` on the hematoxylin
     (rgb2hed H) channel -> per-nucleus polygon + centroid in global pixels.
     Nuclei touching a tile border are dropped (avoids double-count / clipping).
  3. Subsample to ``--max_cells`` (seeded). TSR is a ratio, so a few×10^4
     nuclei sampled uniformly over tissue estimate it well.
  4. Center-crop bbox (clamped) + rasterized nucleus mask per scale.

Coordinates are already in zarr pixels (0.5 um/px), so polygons feed
``rasterize_polygon`` directly (no um->px scaling, unlike the Xenium path).

Run in the ``tcga_seg`` venv (cellpose + skimage + cv2 + zarr + h5py).
GPU strongly recommended.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import zarr
from skimage.color import rgb2hed
from skimage.draw import polygon as _skim_polygon

SCALE_NAMES = ("micro", "meso", "macro")


def rasterize_polygon(vertices_px, crop_origin_xy, crop_size_hw):
    """Inlined copy of sigmma.application.cell_niche_transfer.crops.rasterize_polygon
    (the seg venv has no torchvision, so we cannot import the sigmma package).
    vertices_px: (N,2) x,y in image px; origin (x0,y0); size (h,w)."""
    x0, y0 = crop_origin_xy
    h, w = crop_size_hw
    rr, cc = _skim_polygon(vertices_px[:, 1] - y0, vertices_px[:, 0] - x0, shape=(h, w))
    mask = np.zeros((h, w), dtype=bool)
    mask[rr, cc] = True
    return mask


def tissue_foreground_tiles(he_zarr, tile: int, stride: int = 32,
                            white_thresh: int = 220, min_frac: float = 0.10):
    """Return list of (x0, y0) tile origins overlapping tissue.

    Builds a coarse grayscale thumbnail by strided reads, marks non-white
    pixels as tissue, then keeps tiles with >= ``min_frac`` tissue.
    """
    _, H, W = he_zarr.shape
    thumb = np.asarray(he_zarr[:, ::stride, ::stride])           # (3, h, w)
    gray = thumb.mean(axis=0)
    tissue = gray < white_thresh                                  # True = tissue
    th, tw = tissue.shape
    tiles = []
    step = tile // stride                                         # thumb px per tile
    for ty in range(0, H, tile):
        for tx in range(0, W, tile):
            if tx + tile > W or ty + tile > H:
                continue
            sy, sx = ty // stride, tx // stride
            block = tissue[sy:sy + step, sx:sx + step]
            if block.size and block.mean() >= min_frac:
                tiles.append((tx, ty))
    return tiles


def segment_tile(model, he_zarr, x0: int, y0: int, tile: int, diameter: float):
    """Cellpose nucleus segmentation on one tile. Returns list of
    (centroid_xy_global, polygon_xy_global) for non-border-touching nuclei."""
    crop = np.asarray(he_zarr[:, y0:y0 + tile, x0:x0 + tile])     # (3, T, T) uint8
    rgb = np.transpose(crop, (1, 2, 0))
    # Hematoxylin channel highlights nuclei; normalize to [0,1].
    h = rgb2hed(rgb / 255.0)[..., 0]
    lo, hi = np.percentile(h, (1, 99))
    if hi <= lo:
        return []
    h = np.clip((h - lo) / (hi - lo), 0, 1).astype(np.float32)

    labels, _, _, _ = model.eval(h, diameter=diameter, channels=[0, 0],
                                 flow_threshold=0.4, cellprob_threshold=0.0)
    out = []
    ids = np.unique(labels)
    ids = ids[ids != 0]
    T = tile
    for lid in ids:
        m = (labels == lid).astype(np.uint8)
        ys, xs = np.nonzero(m)
        if xs.size < 8:
            continue
        # Drop nuclei touching the tile border (split across tiles / clipped).
        if xs.min() == 0 or ys.min() == 0 or xs.max() == T - 1 or ys.max() == T - 1:
            continue
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 6:
            continue
        poly = cnt[:, 0, :].astype(np.float32)                    # (V,2) x,y local
        poly[:, 0] += x0
        poly[:, 1] += y0
        cx, cy = xs.mean() + x0, ys.mean() + y0
        out.append(((cx, cy), poly))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", required=True, help="hematoxylin_eosin .he.zarr path")
    ap.add_argument("--sample", required=True, help="sample name (slide id)")
    ap.add_argument("--out_h5", required=True)
    ap.add_argument("--scale_sizes", type=int, nargs=3, default=[64, 128, 256])
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--diameter", type=float, default=15.0,
                    help="nucleus diameter in px @0.5um/px (~7.5um).")
    ap.add_argument("--max_cells", type=int, default=30000)
    ap.add_argument("--max_tiles", type=int, default=600,
                    help="cap segmented tiles (after which we stop & subsample).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    from cellpose import models
    gpu = not args.cpu
    model = models.Cellpose(gpu=gpu, model_type="nuclei")  # v3: eval -> 4-tuple

    he = zarr.open(args.zarr, mode="r")
    _, H, W = he.shape
    print(f"[{args.sample}] zarr {W}x{H}  tile={args.tile} diam={args.diameter}",
          flush=True)

    tiles = tissue_foreground_tiles(he, args.tile)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(tiles)
    print(f"[{args.sample}] foreground tiles: {len(tiles)} (cap {args.max_tiles})",
          flush=True)

    centroids, polys = [], []
    for i, (x0, y0) in enumerate(tiles[: args.max_tiles]):
        res = segment_tile(model, he, x0, y0, args.tile, args.diameter)
        for c, p in res:
            centroids.append(c); polys.append(p)
        if (i + 1) % 25 == 0 or i + 1 == min(len(tiles), args.max_tiles):
            print(f"  tile {i+1}: cumulative nuclei {len(centroids)}", flush=True)
        if len(centroids) >= int(args.max_cells * 1.15):
            print(f"  reached nucleus target at tile {i+1}", flush=True)
            break

    n_tot = len(centroids)
    if n_tot == 0:
        raise SystemExit(f"[{args.sample}] no nuclei segmented")
    idx = np.arange(n_tot)
    if n_tot > args.max_cells:
        idx = rng.choice(idx, size=args.max_cells, replace=False)
    idx = np.sort(idx)
    N = len(idx)
    print(f"[{args.sample}] segmented {n_tot}, keeping {N}", flush=True)

    cell_ids = np.array([f"{args.sample}_n{int(k)}" for k in idx], dtype="S")
    x_pixel = np.empty(N, np.int32); y_pixel = np.empty(N, np.int32)
    bboxes = {s: np.empty((N, 2), np.int32) for s in SCALE_NAMES}
    masks = {s: np.empty((N, sz, sz), bool)
             for s, sz in zip(SCALE_NAMES, args.scale_sizes)}

    for j, k in enumerate(idx):
        cx, cy = centroids[k]
        x_pixel[j] = int(cx); y_pixel[j] = int(cy)
        verts = polys[k]
        for sname, ssize in zip(SCALE_NAMES, args.scale_sizes):
            bx = int(round(cx - ssize / 2)); by = int(round(cy - ssize / 2))
            bx = max(0, min(bx, W - ssize)); by = max(0, min(by, H - ssize))
            bboxes[sname][j] = (bx, by)
            masks[sname][j] = rasterize_polygon(verts, (bx, by), (ssize, ssize))

    out = Path(args.out_h5); out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.attrs["sample_name"] = args.sample
        f.attrs["normalized_sample"] = args.sample
        f.attrs["split"] = "test"
        f.attrs["scale_sizes"] = list(args.scale_sizes)
        f.attrs["label_names"] = np.array([], dtype="S")
        f.create_dataset("cell_id", data=cell_ids)
        f.create_dataset("x_pixel", data=x_pixel)
        f.create_dataset("y_pixel", data=y_pixel)
        for s in SCALE_NAMES:
            f.create_dataset(f"bbox_{s}", data=bboxes[s])
            f.create_dataset(f"mask_{s}", data=masks[s], compression="gzip",
                             compression_opts=4, chunks=True)
    print(f"[{args.sample}] wrote {out} ({N} nuclei)", flush=True)


if __name__ == "__main__":
    main()
