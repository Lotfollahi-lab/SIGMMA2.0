"""DeepPT preprocessing — same ImageNet normalisation as ST-Net."""

from __future__ import annotations

from ..stnet.transforms import IMAGENET_MEAN, IMAGENET_STD, to_model_input

__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "to_model_input"]
