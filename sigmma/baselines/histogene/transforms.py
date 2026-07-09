"""HisToGene preprocessing.

The original training uses flattened RGB patches with values in [0, 1]
(no ImageNet normalisation — patches are passed directly to a Linear
embedding). We preserve that.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def to_model_input(tile_uint8, patch_size: int = 112, device=None) -> torch.Tensor:
    """``(B, 3, H, W)`` uint8 → ``(B, 1, 3·P·P)`` float32 in [0, 1].

    Returns a single-spot sequence per FOV. Multi-spot eval batches K²
    bin-centered crops by stacking on the N dimension (handled in eval).
    """
    if hasattr(tile_uint8, "numpy"):
        t = tile_uint8.float()
    else:
        t = torch.as_tensor(tile_uint8, dtype=torch.float32)
    if t.ndim == 3:
        t = t.unsqueeze(0)
    t = t / 255.0
    if t.shape[-1] != patch_size or t.shape[-2] != patch_size:
        t = F.interpolate(t, size=(patch_size, patch_size),
                          mode="bilinear", align_corners=False)
    B, C, H, W = t.shape
    patches = t.reshape(B, 1, C * H * W)
    if device is not None:
        patches = patches.to(device, non_blocking=True)
    return patches


def make_center_default(batch_size: int, n_pos_default: int = 32,
                        device=None) -> torch.Tensor:
    """Single-spot centre coordinate placeholder (mid-grid). Used at training
    when each FOV is treated as one spot.
    """
    c = torch.full((batch_size, 1, 2), n_pos_default, dtype=torch.long)
    if device is not None:
        c = c.to(device, non_blocking=True)
    return c
