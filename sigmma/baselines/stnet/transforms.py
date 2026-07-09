"""ImageNet preprocessing for ST-Net."""

from __future__ import annotations

import torch
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_model_input(tile_uint8, target_size: int = 224, device=None) -> torch.Tensor:
    """``(B, 3, H, W)`` uint8 or ``(3, H, W)`` uint8 → normalised float32 tensor.

    Handles both numpy.ndarray and torch.Tensor input. Resize is bilinear
    so multi-scale ROI assembly stays consistent.
    """
    if hasattr(tile_uint8, "numpy"):
        t = tile_uint8.float()
    else:
        t = torch.as_tensor(tile_uint8, dtype=torch.float32)
    if t.ndim == 3:
        t = t.unsqueeze(0)
    t = t / 255.0
    if t.shape[-1] != target_size or t.shape[-2] != target_size:
        t = F.interpolate(t, size=(target_size, target_size),
                          mode="bilinear", align_corners=False)
    mean = IMAGENET_MEAN.to(t.device)
    std = IMAGENET_STD.to(t.device)
    t = (t - mean) / std
    if device is not None:
        t = t.to(device, non_blocking=True)
    return t
