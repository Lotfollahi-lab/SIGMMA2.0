"""
Cell-centered crops + polygon rasterization for the niche label-transfer task.

Coordinate convention
---------------------
- Xenium polygons (cell_boundaries.parquet) live in micrometres.
- The H&E zarr is at 0.5 µm/pixel, so ``vertex_px = vertex_um * 2``.
- The H&E zarr layout is ``(3, H, W)`` uint8 (channel-first).
"""
from __future__ import annotations
from typing import Tuple

import numpy as np
from skimage.draw import polygon as _skim_polygon


def rasterize_polygon(
    vertices_px: np.ndarray,
    crop_origin_xy: Tuple[int, int],
    crop_size_hw: Tuple[int, int],
) -> np.ndarray:
    """Rasterize a polygon into a binary mask aligned to a crop window.

    Args:
        vertices_px: (N, 2) polygon vertices in image-pixel coordinates (x, y).
        crop_origin_xy: (x0, y0) top-left corner of the crop window in image px.
        crop_size_hw: (h, w) crop size.

    Returns:
        bool array of shape (h, w); True inside the polygon.
    """
    x0, y0 = crop_origin_xy
    h, w = crop_size_hw
    x = vertices_px[:, 0] - x0
    y = vertices_px[:, 1] - y0
    rr, cc = _skim_polygon(y, x, shape=(h, w))
    mask = np.zeros((h, w), dtype=bool)
    mask[rr, cc] = True
    return mask


def extract_centered_crop(
    he_chw: np.ndarray | "zarr.core.array.Array",
    centroid_xy: Tuple[float, float],
    size: int,
    pad_value: int = 0,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Extract a ``size × size`` RGB crop centered on ``centroid_xy``.

    Pads with ``pad_value`` (default 0) when the window leaves the slide.
    Returns the crop as ``(3, size, size)`` uint8 plus the top-left ``(x0, y0)``
    in slide-pixel coordinates so the caller can rasterize masks aligned to it.
    """
    cx, cy = int(round(centroid_xy[0])), int(round(centroid_xy[1]))
    half = size // 2
    x0, y0 = cx - half, cy - half

    _, H, W = he_chw.shape
    pad_x0 = max(0, -x0)
    pad_y0 = max(0, -y0)
    pad_x1 = max(0, x0 + size - W)
    pad_y1 = max(0, y0 + size - H)
    clip_x0 = max(0, x0)
    clip_y0 = max(0, y0)
    clip_x1 = min(W, x0 + size)
    clip_y1 = min(H, y0 + size)

    # Lazy zarr slicing: returns a numpy array.
    crop = np.asarray(he_chw[:, clip_y0:clip_y1, clip_x0:clip_x1])
    if pad_x0 or pad_y0 or pad_x1 or pad_y1:
        crop = np.pad(
            crop,
            ((0, 0), (pad_y0, pad_y1), (pad_x0, pad_x1)),
            mode="constant",
            constant_values=pad_value,
        )
    return crop, (x0, y0)
