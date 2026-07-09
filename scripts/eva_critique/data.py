"""MoNuSAC test loader + per-pixel region label builder (R1-R4).

R1 cell-dense   : GT nuclei mask, dilated by 8 px
R2 lipid        : inside tissue AND grayscale > 220 AND not R1
R3 OOT          : outside tissue (Otsu on saturation channel)
R4 other        : inside tissue AND not R1 AND not R2

Tissue mask: Otsu on the HSV S channel (high S = tissue, low S = whitespace).
Grayscale channel: standard ITU-R BT.601 luma on 8-bit RGB.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation, disk
from torchvision.transforms import v2 as tv2

from eva.vision import datasets as eva_datasets


_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_AMBIGUOUS_IDX = 5  # MoNuSAC class 5 == "Ambiguous"


_eval_tx = tv2.Compose(
    [
        tv2.Resize(size=224, antialias=True),
        tv2.CenterCrop(size=224),
        tv2.ToDtype(torch.float32, scale=True),
        tv2.Normalize(mean=list(_MEAN), std=list(_STD)),
    ]
)


@dataclass
class Tile:
    image_id: str            # base filename (no extension)
    image_norm: torch.Tensor # (3, 224, 224) normalised
    image_rgb: np.ndarray    # (224, 224, 3) uint8 — for figures + region builder
    nuclei_gt: np.ndarray    # (224, 224) bool
    sem_mask: np.ndarray     # (224, 224) uint8 — 0 bg, 1-4 nuclei subtype, 5 ambig
    R1: np.ndarray           # (224, 224) bool
    R2: np.ndarray
    R3: np.ndarray
    R4: np.ndarray
    tissue: np.ndarray       # (224, 224) bool


def _resize_image_uint8(img_chw: np.ndarray) -> np.ndarray:
    """Match the eval transform's geometry on the raw uint8 RGB image.

    `img_chw` is (3, H, W) uint8 from eva's loader. Apply Resize(short=224) +
    CenterCrop(224) using torchvision so the region masks land on the same
    pixels as the model input.
    """
    t = torch.from_numpy(img_chw)  # uint8 (3,H,W)
    t = tv2.Resize(size=224, antialias=True)(t)
    t = tv2.CenterCrop(size=224)(t)
    return t.permute(1, 2, 0).numpy()  # (224,224,3) uint8


def _resize_mask_nearest(mask_hw: np.ndarray) -> np.ndarray:
    """Nearest-neighbour resize/crop to (224,224) for label preservation."""
    m = torch.from_numpy(mask_hw).unsqueeze(0)  # (1,H,W)
    m = tv2.Resize(size=224, antialias=False, interpolation=tv2.InterpolationMode.NEAREST)(m)
    m = tv2.CenterCrop(size=224)(m)
    return m.squeeze(0).numpy()


def build_regions(
    rgb: np.ndarray, nuclei_gt: np.ndarray, dilate_px: int = 8, lipid_lum: int = 220
):
    """Return (R1, R2, R3, R4, tissue) bool masks — all (H,W)."""
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)

    # tissue: Otsu on saturation; high S => tissue.
    hsv = rgb2hsv(rgb)
    s = (hsv[..., 1] * 255).astype(np.uint8)
    try:
        thr = int(threshold_otsu(s))
    except Exception:
        thr = 30  # degenerate (all-background) tile fallback
    tissue = s > thr

    R1 = binary_dilation(nuclei_gt, footprint=disk(dilate_px))
    gray = (
        0.299 * rgb[..., 0].astype(np.float32)
        + 0.587 * rgb[..., 1].astype(np.float32)
        + 0.114 * rgb[..., 2].astype(np.float32)
    )
    bright = gray > lipid_lum

    R3 = ~tissue
    R2 = tissue & bright & ~R1
    R4 = tissue & ~R1 & ~R2
    return R1.astype(bool), R2.astype(bool), R3.astype(bool), R4.astype(bool), tissue.astype(bool)


def iter_monusac_test(data_root: str, split: str = "test") -> Iterator[Tile]:
    """Yield Tile objects from the EVA MoNuSAC dataset (default: test split).

    Returns the model-ready normalised tensor + the resized-to-224 uint8 image
    + the binary nuclei GT (ambiguous excluded) + R1/R2/R3/R4/tissue masks.
    """
    ds = eva_datasets.MoNuSAC(root=data_root, split=split, download=False)
    for i in range(len(ds)):
        # eva returns Image, Mask via load_data/load_target.
        img_t = ds.load_data(i)        # tv_tensors.Image (3, H, W) uint8
        msk_t = ds.load_target(i)      # tv_tensors.Mask  (H, W)    int64
        img = np.asarray(img_t)
        msk = np.asarray(msk_t).astype(np.uint8)

        rgb_224 = _resize_image_uint8(img)
        msk_224 = _resize_mask_nearest(msk)
        nuclei_gt = (msk_224 >= 1) & (msk_224 != _AMBIGUOUS_IDX)

        # Re-run the model transform on the resized uint8 image so the geometry
        # matches the regions exactly.
        img_norm = _eval_tx(torch.from_numpy(rgb_224).permute(2, 0, 1))

        R1, R2, R3, R4, tissue = build_regions(rgb_224, nuclei_gt)
        image_id = ds.filename(i).rsplit("/", 1)[-1].replace(".tif", "")
        yield Tile(
            image_id=image_id,
            image_norm=img_norm,
            image_rgb=rgb_224,
            nuclei_gt=nuclei_gt.astype(bool),
            sem_mask=msk_224,
            R1=R1,
            R2=R2,
            R3=R3,
            R4=R4,
            tissue=tissue,
        )
