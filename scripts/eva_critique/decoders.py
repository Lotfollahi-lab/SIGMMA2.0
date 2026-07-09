"""Decoder heads for the EVA-critique sweep.

All decoders consume frozen ViT features (channels-first, 14x14 spatial) and
produce (B, num_classes, 224, 224) logits. The encoder forward pass and feature
extraction are handled by the caller; decoders never touch the encoder.

  D0 linear         : 1x1 conv on the final-block feature map, then bilinear up.
  D1 mlp            : 1x1 -> GELU -> 1x1, then bilinear up.
  D2 light_unet     : 4 conv-upsample stages (no skips), 14 -> 28 -> 56 -> 112 -> 224.
  D3 unet_skip      : 4 conv-upsample stages with skip-feature inputs from
                      ViT layers {6, 12, 18, 24} (all 14x14, projected and
                      injected at progressively higher resolutions).
  EVA convdec_image : Faithful re-implementation of EVA upstream's
                      ConvDecoderWithImage (the head used in the canonical
                      MoNuSAC online config). Feature map is upsampled +
                      Conv-BN-ReLU(D->64), bilinear-upscaled to image size,
                      concatenated with the input RGB image, two more
                      Conv-BN-ReLU stages (67->32->32), then a 1x1 classifier.
                      Decoder needs the input image — train_decoder.py passes
                      it when `needs_image` is set.
  EVA convdec_ms    : Faithful re-implementation of EVA upstream's
                      ConvDecoderMS — the DINOv2 "+ms" head EVA actually
                      uses for the MoNuSAC leaderboard row (configs/.../
                      monusac.yaml). Upsample x2 + Conv3x3(D->64), Upsample
                      x2 + Conv3x3(64->C); final bilinear upscale to image
                      size. No BatchNorm, no ReLU, no skips, no image-concat.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch),
        nn.GELU(),
    )


class LinearDecoder(nn.Module):
    """D0 — EVA default linear head."""

    needs_pyramid = False

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.head = nn.Conv2d(in_features, num_classes, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B, D, 14, 14)
        x = self.head(feat)
        return F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)


class MLPDecoder(nn.Module):
    """D1 — 2-layer MLP applied per-patch."""

    needs_pyramid = False

    def __init__(self, in_features: int, num_classes: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_features, hidden, 1, bias=False),
            nn.GroupNorm(min(32, hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, num_classes, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.net(feat)
        return F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)


class LightUNetDecoder(nn.Module):
    """D2 — light UNet (no skips). 14 -> 28 -> 56 -> 112 -> 224."""

    needs_pyramid = False

    def __init__(self, in_features: int, num_classes: int, base_ch: int = 256):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch // 2, base_ch // 4, base_ch // 8
        self.proj = nn.Conv2d(in_features, c1, 1)
        self.up1 = _conv_block(c1, c2)
        self.up2 = _conv_block(c2, c3)
        self.up3 = _conv_block(c3, c4)
        self.up4 = _conv_block(c4, c4)
        self.head = nn.Conv2d(c4, num_classes, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.proj(feat)                                                   # (B, c1, 14, 14)
        x = self.up1(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))  # 28
        x = self.up2(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))  # 56
        x = self.up3(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))  # 112
        x = self.up4(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))  # 224
        return self.head(x)


class UNetSkipDecoder(nn.Module):
    """D3 — UNet with skip features from ViT layers {6, 12, 18, 24}.

    All ViT taps share the same 14x14 spatial size; we inject one tap into
    each of the 4 upsampling stages by upsampling the tap to that stage's
    target resolution and concatenating with the upsampled bottleneck stream.
    Channel projections keep the parameter count comparable to D2.
    """

    needs_pyramid = True
    pyramid_layers: Sequence[int] = (6, 12, 18, 24)

    def __init__(self, in_features: int, num_classes: int, base_ch: int = 256):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch // 2, base_ch // 4, base_ch // 8
        # Skip projections: one per ViT tap (low -> high resolution stage).
        self.skip_proj = nn.ModuleList(
            [nn.Conv2d(in_features, c, 1) for c in (c1, c2, c3, c4)]
        )
        self.proj = nn.Conv2d(in_features, c1, 1)
        # After each upsample we concat skip(c_k) with stream(c_k) -> 2*c_k channels.
        self.up1 = _conv_block(c1 + c1, c2)
        self.up2 = _conv_block(c2 + c2, c3)
        self.up3 = _conv_block(c3 + c3, c4)
        self.up4 = _conv_block(c4 + c4, c4)
        self.head = nn.Conv2d(c4, num_classes, 1)

    def forward(self, pyramid: List[torch.Tensor]) -> torch.Tensor:
        # pyramid is in increasing-depth order (layer 6 -> 24).
        # Bottleneck stream starts from the deepest tap.
        f6, f12, f18, f24 = pyramid
        skips = [self.skip_proj[0](f6), self.skip_proj[1](f12),
                 self.skip_proj[2](f18), self.skip_proj[3](f24)]

        x = self.proj(f24)  # (B, c1, 14, 14)
        for stage, (block, skip, target_size) in enumerate(zip(
            (self.up1, self.up2, self.up3, self.up4),
            skips,
            (28, 56, 112, 224),
        )):
            x = F.interpolate(x, size=(target_size, target_size),
                              mode="bilinear", align_corners=False)
            s = F.interpolate(skip, size=(target_size, target_size),
                              mode="bilinear", align_corners=False)
            x = block(torch.cat([x, s], dim=1))
        return self.head(x)


class ConvDecoderWithImage(nn.Module):
    """Faithful re-implementation of EVA upstream's ConvDecoderWithImage.

    Mirrors `eva.vision.models.networks.decoders.segmentation.semantic.with_image`
    so the segmentation row in our manuscript uses the same head EVA's MoNuSAC
    leaderboard reports. Layer choice, channel widths, BN/ReLU, and the
    image-concat at full resolution all match upstream.
    """

    needs_pyramid = False
    needs_image = True
    _default_hidden_dims = (64, 32, 32)

    def __init__(self, in_features: int, num_classes: int, greyscale: bool = False):
        super().__init__()
        h0, h1, h2 = self._default_hidden_dims
        self.greyscale = greyscale
        # Stage 1 mirrors `super().forward` of EVA's class:
        # Upsample x2 + Conv-BN-ReLU(in_features -> h0), then bilinear upscale to image_size.
        self.stage1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_features, h0, kernel_size=3, padding=1),
            nn.BatchNorm2d(h0),
            nn.ReLU(inplace=True),
        )
        extra = 1 if greyscale else 3
        # Stage 2: image_block — two Conv-BN-ReLU at full resolution after image concat.
        self.image_block = nn.Sequential(
            nn.Conv2d(h0 + extra, h1, kernel_size=3, padding=1),
            nn.BatchNorm2d(h1),
            nn.ReLU(inplace=True),
            nn.Conv2d(h1, h2, kernel_size=3, padding=1),
            nn.BatchNorm2d(h2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(h2, num_classes, kernel_size=1)

    def forward(self, feat: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        # feat: (B, D, 14, 14); image: (B, 3, 224, 224) (normalized RGB).
        x = self.stage1(feat)
        x = F.interpolate(x, size=image.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, image], dim=1)
        x = self.image_block(x)
        return self.classifier(x)


class ConvDecoderMS(nn.Module):
    """Faithful re-implementation of EVA upstream's ConvDecoderMS.

    Mirrors `eva.vision.models.networks.decoders.segmentation.semantic.common
    .ConvDecoderMS` exactly — the DINOv2 "+ms" head used in EVA's MoNuSAC
    leaderboard config (`configs/vision/pathology/offline/segmentation/
    monusac.yaml`).

    Architecture:
        Upsample(x2) -> Conv2d(D, 64, 3x3, pad=1)
        Upsample(x2) -> Conv2d(64, C, 3x3, pad=1)
        bilinear interpolate to image_size

    No BatchNorm, no activation between conv layers, no skips, no image
    concat. Final 4x upscale via `F.interpolate(..., 'bilinear')` matches
    `Decoder2D._upscale` (note: upstream uses default `align_corners`,
    which is False in torch.nn.functional.interpolate's bilinear mode).

    `image_size` may be passed to forward() to support variable input
    resolutions — matches EVA's per-forward `DecoderInputs.image_size`.
    Defaults to the value supplied at construction (224).
    """

    needs_pyramid = False
    # Tells _SegHead to pass the input image's spatial size at forward time
    # (mirrors EVA's DecoderInputs.image_size plumbing).
    needs_image_size = True

    def __init__(self, in_features: int, num_classes: int,
                 image_size: int = 224):
        super().__init__()
        self.default_image_size = (image_size, image_size)
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_features, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1),
        )

    def forward(self, feat: torch.Tensor,
                image_size: tuple[int, int] | None = None) -> torch.Tensor:
        # feat: (B, D, h, w) -> (B, 64, 2h, 2w) -> (B, C, 4h, 4w)
        logits = self.layers(feat)
        # Final bilinear upscale to image_size; matches Decoder2D._upscale.
        target = image_size if image_size is not None else self.default_image_size
        return F.interpolate(logits, size=target, mode="bilinear")


def build_decoder(name: str, in_features: int, num_classes: int) -> nn.Module:
    name = name.lower()
    if name == "linear":
        return LinearDecoder(in_features, num_classes)
    if name == "mlp":
        return MLPDecoder(in_features, num_classes)
    if name in ("light_unet", "unet"):
        return LightUNetDecoder(in_features, num_classes)
    if name in ("unet_skip", "unet_skips"):
        return UNetSkipDecoder(in_features, num_classes)
    if name in ("convdec_image", "eva_default"):
        return ConvDecoderWithImage(in_features, num_classes)
    if name in ("convdec_ms", "eva_ms"):
        return ConvDecoderMS(in_features, num_classes)
    raise ValueError(f"Unknown decoder name: {name}")


DECODER_NAMES = ["linear", "mlp", "light_unet", "unet_skip",
                 "convdec_image", "convdec_ms"]
