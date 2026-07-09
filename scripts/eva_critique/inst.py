"""Instance segmentation pipeline: prob -> threshold -> distance transform +
watershed -> labelled instances.

Used by the object-level component (#3). Choice of seed minima distance and
small-object filter follow Kumar (2020) MoNuSAC challenge defaults.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed


def prob_to_instances(
    prob: np.ndarray,
    threshold: float,
    min_size: int = 32,
    seed_min_distance: int = 7,
) -> np.ndarray:
    """Threshold probability map and split into instances by watershed.

    Args:
        prob: (H, W) float in [0, 1] (foreground / nuclei probability).
        threshold: foreground threshold.
        min_size: drop predicted instances with fewer pixels than this.
        seed_min_distance: minimum distance between watershed seeds, in pixels.
    Returns:
        int32 instance label map; background = 0.
    """
    fg = prob >= threshold
    if not fg.any():
        return np.zeros_like(prob, dtype=np.int32)
    fg = remove_small_objects(fg, min_size=min_size)
    if not fg.any():
        return np.zeros_like(prob, dtype=np.int32)

    dist = ndi.distance_transform_edt(fg)
    coords = peak_local_max(
        dist,
        min_distance=seed_min_distance,
        labels=fg.astype(np.int32),
    )
    if coords.size == 0:
        # Fall back to one seed per connected component.
        markers, _ = ndi.label(fg)
        return watershed(-dist, markers=markers, mask=fg).astype(np.int32)

    seed_mask = np.zeros_like(fg, dtype=bool)
    seed_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(seed_mask)
    return watershed(-dist, markers=markers, mask=fg).astype(np.int32)
