"""Polygon-derived GT instance labels for MoNuSAC tiles.

EVA's MoNuSAC loader collapses per-cell polygon annotations into a multi-class
semantic mask, then we reduce that to a binary nuclei mask. Connected-
component labelling on that binary mask fuses any cells whose pixels touch,
silently merging ~70% of cells in dense tiles. This module reads the original
ImageScope XML annotations (one Region per cell) and rasterizes each polygon
with a unique instance ID, so downstream metrics see true cell-instance
boundaries.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from skimage.draw import polygon as sk_polygon


_TEST_SUBDIR = "MoNuSAC Testing Data and Annotations"


def _slide_id(image_id: str) -> str:
    """Strip the trailing `_NNN` tile index from an image_id."""
    return image_id.rsplit("_", 1)[0]


def _xml_path(data_root: Path, image_id: str) -> Path:
    return data_root / _TEST_SUBDIR / _slide_id(image_id) / f"{image_id}.xml"


def _tif_path(data_root: Path, image_id: str) -> Path:
    return data_root / _TEST_SUBDIR / _slide_id(image_id) / f"{image_id}.tif"


def load_polygon_instance_label(
    image_id: str,
    data_root: Path | str,
    target_size: int = 224,
    exclude_classes: Sequence[str] = ("Ambiguous",),
) -> np.ndarray:
    """Build a (target_size, target_size) int32 instance label map from
    MoNuSAC XML polygons. 0 = background; each cell gets a unique positive ID.

    Resizes from the TIF's native resolution to `target_size` via nearest-
    neighbour, matching how `_resize_image_uint8` in `data.py` resizes the
    H&E image (so coordinates align with the model input + cached `gt_bin`).
    """
    data_root = Path(data_root)
    xml_p = _xml_path(data_root, image_id)
    tif_p = _tif_path(data_root, image_id)
    if not xml_p.exists() or not tif_p.exists():
        raise FileNotFoundError(
            f"polygon GT files missing for {image_id}: "
            f"xml={xml_p.exists()}, tif={tif_p.exists()}"
        )

    with Image.open(tif_p) as img:
        W, H = img.size  # PIL: (width, height)

    inst = np.zeros((H, W), dtype=np.int32)
    cur_id = 0
    tree = ET.parse(xml_p)
    for ann in tree.getroot().findall("Annotation"):
        attr = ann.find("Attributes/Attribute")
        cls = attr.get("Name") if attr is not None else ""
        if cls in exclude_classes:
            continue
        for reg in ann.findall("Regions/Region"):
            verts = reg.find("Vertices")
            if verts is None:
                continue
            xs, ys = [], []
            for v in verts.findall("Vertex"):
                xs.append(float(v.get("X")))
                ys.append(float(v.get("Y")))
            if len(xs) < 3:
                continue
            rr, cc = sk_polygon(np.asarray(ys), np.asarray(xs), shape=(H, W))
            if rr.size == 0:
                continue
            cur_id += 1
            inst[rr, cc] = cur_id

    # Match EVA's geometry exactly: torchvision Resize(short=target_size) +
    # CenterCrop(target_size). PIL.Image.resize uses (width, height).
    short = min(H, W)
    if short != target_size:
        scale = target_size / short
        new_W = int(round(W * scale))
        new_H = int(round(H * scale))
        pil = Image.fromarray(inst, mode="I").resize(
            (new_W, new_H), Image.NEAREST)
        inst = np.asarray(pil, dtype=np.int32)
        H, W = new_H, new_W

    # Center crop to target_size × target_size.
    top = (H - target_size) // 2
    left = (W - target_size) // 2
    return inst[top:top + target_size, left:left + target_size]
