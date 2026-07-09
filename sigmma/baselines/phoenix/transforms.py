"""Phoenix preprocessing.

Phoenix uses its own H&E normalisation constants (NOT ImageNet) and a bicubic
resize to 224, matching the ``v2.Compose`` pipeline in the upstream README.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Phoenix-specific H&E statistics (peng-lab/phoenix README).
PHOENIX_MEAN = torch.tensor([0.707223, 0.578729, 0.703617]).view(1, 3, 1, 1)
PHOENIX_STD = torch.tensor([0.211883, 0.230117, 0.177517]).view(1, 3, 1, 1)


def to_model_input(tile_uint8, target_size: int = 224, device=None) -> torch.Tensor:
    """``(B, 3, H, W)`` or ``(3, H, W)`` uint8 → normalised float32 tensor.

    Bicubic resize to ``target_size`` then Phoenix mean/std normalisation.
    Accepts numpy.ndarray or torch.Tensor.
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
                          mode="bicubic", align_corners=False).clamp_(0.0, 1.0)
    mean = PHOENIX_MEAN.to(t.device)
    std = PHOENIX_STD.to(t.device)
    t = (t - mean) / std
    if device is not None:
        t = t.to(device, non_blocking=True)
    return t
