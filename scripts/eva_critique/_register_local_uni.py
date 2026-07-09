"""Register a local-file UNI backbone in EVA's registry.

Importing this module (before invoking `python -m eva fit`) adds
`local/uni` to `backbone_registry`, which loads the UNI ckpt from a
local path instead of attempting `hf-hub:MahmoodLab/uni` (gated).

Local file source is the same `pytorch_model.bin` that
`vit_large_patch16_224.dinov2.uni_mass100k/download.sh` originally
fetched from `huggingface.co/MahmoodLab/UNI/resolve/main/pytorch_model.bin`
— i.e. bit-identical to what EVA would download.
"""
from typing import Tuple

import timm
import torch
from eva.vision.models import wrappers
from eva.vision.models.networks.backbones.registry import backbone_registry
from torch import nn


LOCAL_UNI_CKPT = (
    "/nfs/team361/dj16/projects/img_spatial_alignment/assets/ckpts/"
    "vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin"
)

LOCAL_UNI2H_CKPT = (
    "/nfs/team361/dj16/projects/SIGMMA/assets/ckpts/uni_h/pytorch_model.bin"
)


@backbone_registry.register("local/uni")
def local_uni(
    dynamic_img_size: bool = True,
    out_indices: int | Tuple[int, ...] | None = None,
) -> nn.Module:
    """UNI ViT-L/16 loaded from local pytorch_model.bin (avoids HF gating)."""
    return wrappers.TimmModel(
        model_name="vit_large_patch16_224",
        pretrained=False,
        checkpoint_path=LOCAL_UNI_CKPT,
        out_indices=out_indices,
        model_kwargs={
            "init_values": 1e-5,
            "dynamic_img_size": dynamic_img_size,
            "img_size": 224,
            "patch_size": 16,
        },
    )


@backbone_registry.register("local/uni2_h")
def local_uni2_h(
    dynamic_img_size: bool = True,
    out_indices: int | Tuple[int, ...] | None = None,
) -> nn.Module:
    """UNI2-h (ViT-H, 1536-d, patch_size=14, reg_tokens=8, SwiGLU MLP).

    Mirrors EVA upstream `mahmood_uni2_h` (https://github.com/kaiko-ai/eva/blob/
    main/src/eva/vision/models/networks/backbones/pathology/mahmood.py) but loads
    weights from a local pytorch_model.bin instead of `hf-hub:MahmoodLab/UNI2-h`.
    """
    return wrappers.TimmModel(
        model_name="vit_large_patch14_224",  # placeholder; kwargs override arch
        pretrained=False,
        checkpoint_path=LOCAL_UNI2H_CKPT,
        out_indices=out_indices,
        model_kwargs={
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": dynamic_img_size,
        },
    )
