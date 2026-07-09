#!/usr/bin/env python
"""Render a single-section H&E overview for one in-house ccRCC section and one TCGA-KIRC slide.

H&E images are zarr v3 arrays of shape (3, H, W) uint8 @ 0.5 um/px. Slides are large, so a
high-resolution thumbnail is built by reading a strided subset of 256-px inner chunks
(one chunk -> one thumbnail pixel), capping I/O regardless of slide size.

Outputs (figures/viz):
  inhouse_ccRCC_<section>_HE.png/.pdf
  tcga_kirc_<barcode>_HE.png/.pdf
"""
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from PIL import Image

INHOUSE_HE = (
    "/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/"
    "sanger_renal_cell_carcinoma/20241105__155751__SGP196_KID_RUN_1_RE-RUN/"
    "CV1-KID-0-FO-1/hematoxylin_eosin.zarr"
)
TCGA_HE = (
    "/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_TCGA/kirc_slides/"
    "000fe7e2-d9a9-413d-9b71-aebda9a615bf/"
    "TCGA-CJ-4868-01Z-00-DX1.95E8398F-7709-42F0-9F2B-2D185B4429AE.he.zarr"
)
OUT_DIR = "figures/viz"
CHUNK = 256


def make_thumb(path, T=400, target_w=1400):
    """Read a strided subset of inner chunks; downsample each chunk to a sub x sub block
    (rather than a single pixel) so the assembled thumbnail reaches ~target_w px wide."""
    z = zarr.open(str(path), mode="r")
    C, H, W = z.shape
    ncy = math.ceil(H / CHUNK)
    ncx = math.ceil(W / CHUNK)
    stepy = max(1, math.ceil(ncy / T))
    stepx = max(1, math.ceil(ncx / T))
    ys = list(range(0, ncy, stepy))
    xs = list(range(0, ncx, stepx))
    sub = int(np.clip(round(target_w / max(1, len(xs))), 1, CHUNK))
    out = np.full((len(ys) * sub, len(xs) * sub, 3), 255, dtype=np.uint8)
    for i, cy in enumerate(ys):
        y0, y1 = cy * CHUNK, min(cy * CHUNK + CHUNK, H)
        for j, cx in enumerate(xs):
            x0, x1 = cx * CHUNK, min(cx * CHUNK + CHUNK, W)
            blk = z[:, y0:y1, x0:x1]  # (3, h, w)
            hwc = np.transpose(blk, (1, 2, 0))  # (h, w, 3)
            small = np.asarray(Image.fromarray(hwc).resize((sub, sub), Image.BILINEAR))
            out[i * sub:(i + 1) * sub, j * sub:(j + 1) * sub] = small
    return out, (H, W)


def render(path, title, out_stem):
    print(f"[{title}] reading {path}")
    thumb, (H, W) = make_thumb(path)
    um_per_px = 0.5
    print(f"  full slide {W}x{H}px (~{W*um_per_px/1000:.1f}x{H*um_per_px/1000:.1f} mm); "
          f"thumb {thumb.shape[1]}x{thumb.shape[0]}")
    fig_w = 8.0
    fig_h = fig_w * thumb.shape[0] / thumb.shape[1]
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(thumb, interpolation="nearest")
    ax.axis("off")
    ax.set_title(f"{title}\n{W}x{H}px @ {um_per_px} um/px "
                 f"(~{W*um_per_px/1000:.1f}x{H*um_per_px/1000:.1f} mm)", fontsize=9)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    png = os.path.join(OUT_DIR, out_stem + ".png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {png}")


def main():
    render(INHOUSE_HE, "In-house ccRCC  CV1-KID-0-FO-1 (Xenium)",
           "inhouse_ccRCC_CV1-KID-0-FO-1_HE")
    render(TCGA_HE, "TCGA-KIRC  TCGA-CJ-4868-01Z (diagnostic WSI)",
           "tcga_kirc_TCGA-CJ-4868-01Z_HE")


if __name__ == "__main__":
    main()
