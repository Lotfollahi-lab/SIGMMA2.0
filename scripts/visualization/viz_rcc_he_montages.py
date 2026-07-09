#!/usr/bin/env python
"""Render H&E overview montages for the in-house ccRCC cohort and the TCGA-KIRC cohort.

Each H&E image is a zarr v3 array of shape (3, H, W) uint8 @ 0.5 um/px. Whole slides are
large (TCGA up to ~1.5 TB total), so thumbnails are built by reading a *strided subset of
256-px inner chunks* (one chunk -> one thumbnail pixel) rather than the full array. This caps
I/O at ~T*T chunk reads per slide regardless of slide size.

Outputs:
  figures/clinical_outcome/he_overview/inhouse_ccRCC_HE_montage.png
  figures/clinical_outcome/he_overview/tcga_kirc_HE_montage.png
"""
import argparse
import glob
import math
import os
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

INHOUSE_ROOT = (
    "/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/"
    "sanger_renal_cell_carcinoma"
)
TCGA_ROOT = "/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_TCGA/kirc_slides"
META_CSV = os.path.join(INHOUSE_ROOT, "metadata.csv")
OUT_DIR = "figures/clinical_outcome/he_overview"

CHUNK = 256  # inner chunk size in px


def make_thumb(path, T=72):
    """Read a strided subset of inner chunks; average each to a single RGB pixel."""
    try:
        z = zarr.open(str(path), mode="r")
        C, H, W = z.shape
        ncy = math.ceil(H / CHUNK)
        ncx = math.ceil(W / CHUNK)
        stepy = max(1, math.ceil(ncy / T))
        stepx = max(1, math.ceil(ncx / T))
        ys = list(range(0, ncy, stepy))
        xs = list(range(0, ncx, stepx))
        out = np.full((len(ys), len(xs), 3), 255, dtype=np.uint8)
        for i, cy in enumerate(ys):
            y0 = cy * CHUNK
            y1 = min(y0 + CHUNK, H)
            for j, cx in enumerate(xs):
                x0 = cx * CHUNK
                x1 = min(x0 + CHUNK, W)
                blk = z[:, y0:y1, x0:x1]  # (3, h, w)
                out[i, j] = blk.reshape(3, -1).mean(axis=1).astype(np.uint8)
        return out, (H, W)
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def _worker(arg):
    label, path = arg
    thumb, info = make_thumb(path)
    return label, thumb, info


def build_montage(items, title, out_path, ncols, title_fs=5.0, suptitle_fs=14):
    """items: list of (label, path). Renders a grid montage."""
    with Pool(processes=min(16, len(items))) as pool:
        results = pool.map(_worker, items)

    ok = [(lab, th) for (lab, th, info) in results if th is not None]
    bad = [(lab, info) for (lab, th, info) in results if th is None]
    for lab, err in bad:
        print(f"  [WARN] failed {lab}: {err}")
    print(f"  {len(ok)}/{len(items)} thumbnails OK for '{title}'")

    n = len(ok)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 1.0, nrows * 1.05), squeeze=False
    )
    for k in range(nrows * ncols):
        ax = axes[k // ncols][k % ncols]
        ax.axis("off")
        if k < n:
            lab, th = ok[k]
            ax.imshow(th, interpolation="nearest")
            ax.set_title(lab, fontsize=title_fs, pad=1.0)
    fig.suptitle(f"{title}  (n={n})", fontsize=suptitle_fs, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")
    return ok


def inhouse_items():
    paths = sorted(glob.glob(os.path.join(INHOUSE_ROOT, "*", "*", "hematoxylin_eosin.zarr")))
    # grade lookup by patient (CVx)
    grade = {}
    if os.path.exists(META_CSV):
        meta = pd.read_csv(META_CSV)
        for _, r in meta.iterrows():
            if pd.notna(r.get("Patient")):
                grade[str(r["Patient"]).strip()] = r.get("grade")
    items = []
    for p in paths:
        sec = Path(p).parts[-2]  # e.g. CV1-KID-0-FO-1
        pat = sec.split("-")[0]
        g = grade.get(pat)
        lab = f"{sec}\nG{int(g)}" if pd.notna(g) else sec
        items.append((lab, p))
    # sort by patient number then section
    items.sort(key=lambda t: (int(t[0].split("-")[0][2:]) if t[0][2:3].isdigit() else 99, t[0]))
    return items


def tcga_items():
    paths = sorted(glob.glob(os.path.join(TCGA_ROOT, "*", "*.he.zarr")))
    items = []
    for p in paths:
        name = Path(p).name  # TCGA-XX-XXXX-...he.zarr
        parts = name.split(".")[0].split("-")
        barcode = "-".join(parts[:3]) if len(parts) >= 3 else name
        items.append((barcode, p))
    items.sort(key=lambda t: t[0])
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["inhouse", "tcga", "both"], default="both")
    args = ap.parse_args()

    if args.cohort in ("inhouse", "both"):
        items = inhouse_items()
        print(f"In-house ccRCC: {len(items)} sections")
        build_montage(
            items,
            "In-house ccRCC H&E (Xenium sections)",
            os.path.join(OUT_DIR, "inhouse_ccRCC_HE_montage.png"),
            ncols=8,
            title_fs=5.0,
            suptitle_fs=13,
        )

    if args.cohort in ("tcga", "both"):
        items = tcga_items()
        print(f"TCGA-KIRC: {len(items)} slides")
        build_montage(
            items,
            "TCGA-KIRC H&E (diagnostic WSIs)",
            os.path.join(OUT_DIR, "tcga_kirc_HE_montage.png"),
            ncols=24,
            title_fs=2.6,
            suptitle_fs=16,
        )


if __name__ == "__main__":
    main()
