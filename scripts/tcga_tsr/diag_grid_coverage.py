#!/usr/bin/env python
"""Diagnose grid-build tissue coverage on a TCGA slide.

The Option-A grid build (tcga_build_grid_data.py) only lays a tile where a
32-stride thumbnail block is >= ``min_frac`` tissue, and "tissue" = thumbnail
mean intensity < ``white_thr`` (default 220). For clear-cell RCC the pale/clear
cytoplasm sits close to white, so genuine tumour can be read as background and
get NO tile -> bare tissue regions in the overlay.

This script quantifies, for the SAME thumbnail the build uses, how much tissue
the (white_thr, min_frac) gate keeps vs a generous reference threshold, sweeps a
few thresholds, and renders an overlay: H&E + current grid points + the
"tissue-at-reference-but-no-tile" pixels highlighted. Read-only (CPU/.venv).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import h5py
import numpy as np
import zarr


def thumb_mean(he, stride):
    return np.asarray(he[:, ::stride, ::stride]).mean(axis=0)   # (h, w), 0..255


def coverage(tissue, tile_px, thumb_stride, min_frac):
    """Fraction of foreground thumbnail blocks that would receive a tile."""
    th, tw = tissue.shape
    step = max(1, tile_px // thumb_stride)
    kept = np.zeros_like(tissue, dtype=bool)
    n_blocks = n_kept = 0
    for sy in range(0, th - step + 1, step):
        for sx in range(0, tw - step + 1, step):
            block = tissue[sy:sy + step, sx:sx + step]
            if not block.size:
                continue
            n_blocks += 1
            if block.mean() >= min_frac:
                n_kept += 1
                kept[sy:sy + step, sx:sx + step] = True
    return kept, n_kept, n_blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--grid_h5", default=None, help="cached grid64_data h5 (current tile x/y)")
    ap.add_argument("--grid_h5_b", default=None,
                    help="second grid h5 (e.g. rebuilt at a higher white_thr); if "
                         "set, the right panel shows ITS tiles instead of the "
                         "dropped-tissue thumbnail blocks.")
    ap.add_argument("--label_a", default="current (white_thr=220)")
    ap.add_argument("--label_b", default="rebuilt (full coverage)")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--fig_dir", required=True)
    ap.add_argument("--tile_um", type=float, default=64.0)
    ap.add_argument("--um_per_px", type=float, default=0.5)
    ap.add_argument("--thumb_stride", type=int, default=32)
    ap.add_argument("--min_frac", type=float, default=0.5)
    ap.add_argument("--ref_white_thr", type=int, default=240,
                    help="generous tissue threshold for the 'true tissue' reference")
    args = ap.parse_args()

    he = zarr.open(args.zarr, mode="r")
    _, H, W = he.shape
    tile_px = int(round(args.tile_um / args.um_per_px))
    tm = thumb_mean(he, args.thumb_stride)

    print(f"[{args.sample}] slide {W}x{H}px tile_px={tile_px} "
          f"thumb {tm.shape} stride={args.thumb_stride}")
    print(f"intensity pct: p1={np.percentile(tm,1):.0f} p50={np.percentile(tm,50):.0f} "
          f"p90={np.percentile(tm,90):.0f} p99={np.percentile(tm,99):.0f}")

    # reference 'true tissue' at a generous threshold
    ref_tissue = tm < args.ref_white_thr
    _, ref_kept, n_blocks = coverage(ref_tissue, tile_px, args.thumb_stride, args.min_frac)

    print("\nthr  min_frac  kept_blocks  %of_ref_tissue_blocks")
    sweep = {}
    for thr in (220, 230, 235, 240):
        for mf in (args.min_frac, 0.3):
            tis = tm < thr
            kept_mask, n_kept, _ = coverage(tis, tile_px, args.thumb_stride, mf)
            frac = 100.0 * n_kept / max(1, ref_kept)
            print(f"{thr:3d}   {mf:4.2f}     {n_kept:7d}      {frac:6.1f}%")
            sweep[(thr, mf)] = (kept_mask, n_kept)

    # ---- overlay: H&E thumb + current grid points + dropped-tissue pixels ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = max(1, int(max(H, W) / 1600))
    rgb = np.transpose(he[:, ::s, ::s], (1, 2, 0))
    cur_thr_mask, _ = sweep[(220, args.min_frac)]
    ref_thr_mask, _ = sweep[(args.ref_white_thr, args.min_frac)]
    dropped = ref_thr_mask & ~cur_thr_mask    # tissue at 240 but dropped at 220

    def _load_xy(p):
        with h5py.File(p, "r") as f:
            return f["x_pixel"][:], f["y_pixel"][:]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5 * H / W))
    axes[0].imshow(rgb)
    na = 0
    if args.grid_h5 and Path(args.grid_h5).exists():
        gx, gy = _load_xy(args.grid_h5); na = len(gx)
        axes[0].scatter(gx / s, gy / s, s=1.5, c="#2ca02c", alpha=0.5,
                        edgecolors="none", label=f"grid tiles (n={na})")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].set_title(f"{args.label_a}, min_frac={args.min_frac}  [n={na}]", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(rgb)
    if args.grid_h5_b and Path(args.grid_h5_b).exists():
        # right panel = the REBUILT full-coverage grid's actual tiles
        bx, by = _load_xy(args.grid_h5_b); nb = len(bx)
        axes[1].scatter(bx / s, by / s, s=1.5, c="#1f77b4", alpha=0.5,
                        edgecolors="none", label=f"grid tiles (n={nb})")
        delta = nb - na
        axes[1].set_title(f"{args.label_b}  [n={nb}, +{delta} tiles]", fontsize=9)
    else:
        # fallback: dropped-tissue thumbnail blocks
        yy, xx = np.where(dropped)
        dx = (xx * args.thumb_stride + args.thumb_stride / 2) / s
        dy = (yy * args.thumb_stride + args.thumb_stride / 2) / s
        axes[1].scatter(dx, dy, s=2, c="#d62728", alpha=0.5, edgecolors="none",
                        label=f"tissue<{args.ref_white_thr} dropped at thr=220")
        axes[1].set_title("DROPPED tissue (red) = pale regions lost by white_thr=220",
                          fontsize=9)
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].axis("off")

    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)
    out = Path(args.fig_dir) / f"{args.sample.split('.')[0]}_coverage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig] {out}")
    drop_frac = 100.0 * dropped.sum() / max(1, ref_thr_mask.sum())
    print(f"[summary] {drop_frac:.1f}% of true tissue (thr<{args.ref_white_thr}) "
          f"is DROPPED by the white_thr=220 gate")


if __name__ == "__main__":
    main()
