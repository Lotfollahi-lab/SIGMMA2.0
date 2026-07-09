"""Downsample a Visium full-res H&E .tiff to a target µm/px and save it tiled+pyramidal.

The Sanger in-house Visium .tiff files are single-strip JPEG-compressed WSIs
(~0.21 µm/px, one strip), so any read decodes the whole ~4 GB image into memory.
This script pays that cost once on a compute node, resizes to the requested
resolution, and writes a tiled, pyramidal BigTIFF so downstream notebooks can
crop regions lazily (tiff.imread(..., aszarr=True)[y0:y1, x0:x1]).

Usage:
    python scripts/downsample_visium_tiff.py \
        --section_id AT15-BRA-4-FO-F1-S31 \
        --target_um_px 0.5
"""
import argparse
import io
from pathlib import Path

import cv2
import numpy as np
import scanpy
import tifffile as tiff
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # these WSIs exceed PIL's decompression-bomb guard


def fullres_um_px(h5ad_path: Path, section_id: str) -> float:
    """µm per pixel of the full-res tiff, from the Visium scalefactors (55 µm spot)."""
    adata = scanpy.read_h5ad(h5ad_path)
    sf = adata.uns["spatial"][section_id]["scalefactors"]
    return 55.0 / sf["spot_diameter_fullres"]


def read_scaled(tiff_path: Path, factor: float) -> np.ndarray:
    """Low-memory read of the single-strip JPEG WSI.

    tifffile.imread would decode the whole ~4 GB full-res strip. Instead we hand
    the raw JPEG stream to libjpeg via PIL.draft, which decodes at the nearest
    1/2^k scale >= the target size, then resize to the exact target. Peak memory
    is roughly the draft-scale buffer (~2 GB at 1/2) instead of 4 GB.
    Falls back to tifffile.imread for any layout we can't fast-path.
    """
    with tiff.TiffFile(tiff_path) as t:
        p = t.pages[0]
        H, W = p.shape[:2]
        single_jpeg = (len(p.dataoffsets) == 1 and int(p.compression) == 7
                       and 347 not in p.tags)  # 347 = JPEGTables (needs prepending)
        off, n = int(p.dataoffsets[0]), int(p.databytecounts[0])

    if not single_jpeg:
        return tiff.imread(tiff_path)

    with open(tiff_path, "rb") as fh:
        fh.seek(off)
        buf = fh.read(n)
    im = Image.open(io.BytesIO(buf))
    im.draft("RGB", (round(W * factor), round(H * factor)))  # libjpeg 1/2^k decode
    return np.asarray(im.convert("RGB"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path,
                    default=Path("/lustre/scratch126/cellgen/lotfollahi/dj16/data/"
                                 "sanger_in_house_visium/sanger_glioblastoma/"))
    ap.add_argument("--section_id", required=True)
    ap.add_argument("--target_um_px", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=None,
                    help="output tiff (default: {repo}/{section_id}_{target}umpx.tiff)")
    args = ap.parse_args()

    tiff_path = args.repo / f"{args.section_id}.tiff"
    h5ad_path = args.repo / f"{args.section_id}.h5ad"
    out_path = args.out or args.repo / f"{args.section_id}_{args.target_um_px}umpx.tiff"

    src_um_px = fullres_um_px(h5ad_path, args.section_id)
    factor = src_um_px / args.target_um_px  # <1 => downsample
    print(f"[{args.section_id}] full-res = {src_um_px:.4f} µm/px  "
          f"-> target = {args.target_um_px} µm/px  (factor {factor:.4f})", flush=True)

    print("reading tiff via low-memory draft decode...", flush=True)
    img = read_scaled(tiff_path, factor)  # (H', W', 3) uint8 at ~1/2^k of full res
    H, W = img.shape[:2]  # NOTE: draft-scale dims, not full-res

    with tiff.TiffFile(tiff_path) as t:
        fullH, fullW = t.pages[0].shape[:2]
    new_w, new_h = round(fullW * factor), round(fullH * factor)  # exact target size
    print(f"draft {W}x{H} -> resize {new_w}x{new_h}", flush=True)

    # INTER_AREA = area-averaging, the correct anti-aliased kernel for shrinking.
    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    del img

    # resolution tag stores px/cm so readers report the physical scale.
    px_per_cm = 1e4 / args.target_um_px
    n_levels = 3  # extra /2 pyramid levels below full for cheap thumbnails
    print(f"writing tiled+pyramidal BigTIFF -> {out_path}", flush=True)
    opts = dict(photometric="rgb", tile=(512, 512), compression="jpeg")
    with tiff.TiffWriter(out_path, bigtiff=True) as tw:
        tw.write(
            small,
            subifds=n_levels,
            resolution=(px_per_cm, px_per_cm),
            resolutionunit="CENTIMETER",
            metadata={"um_per_px": args.target_um_px, "source": str(tiff_path)},
            **opts,
        )
        # sub-resolutions: each halves the previous level (area-averaged).
        lvl = small
        for _ in range(n_levels):
            lvl = cv2.resize(lvl, (lvl.shape[1] // 2, lvl.shape[0] // 2),
                             interpolation=cv2.INTER_AREA)
            tw.write(lvl, subfiletype=1, **opts)
    print(f"done: {out_path}  ({new_w}x{new_h}x3, {args.target_um_px} µm/px)", flush=True)


if __name__ == "__main__":
    main()
