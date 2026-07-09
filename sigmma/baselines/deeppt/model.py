"""DeepPT (Hoang et al., Nat Cancer 2023/2024) — minimal reimplementation.

Architecture: ImageNet-pretrained ResNet-50 image encoder → autoencoder
(2048 → 512 → 256 bottleneck → 512 → 2048) → MLP regressor → G genes.

Native loss: MSE on z-scored gene expression (mean/std fit on training set).
Original training pipeline (Zenodo 11125591) is cancer-type-specific; we
fine-tune the regressor on SIGMMA's Xenium training manifest.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torchvision


class _AutoEncoder(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden_dim: int = 512,
                 bottleneck_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


class DeepPT(nn.Module):
    def __init__(self, n_genes: int, pretrained: bool = True,
                 ae_hidden_dim: int = 512, ae_bottleneck_dim: int = 256,
                 head_hidden_dim: int = 512):
        super().__init__()
        backbone = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            if pretrained else None
        )
        in_features = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.ae = _AutoEncoder(in_dim=in_features, hidden_dim=ae_hidden_dim,
                               bottleneck_dim=ae_bottleneck_dim)
        self.head = nn.Sequential(
            nn.Linear(ae_bottleneck_dim, head_hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(head_hidden_dim, n_genes),
        )

    def forward(self, x: torch.Tensor, return_recon: bool = False):
        feats = self.backbone(x)
        z, recon = self.ae(feats)
        pred = self.head(z)
        if return_recon:
            return pred, feats, recon
        return pred


def build_model(n_genes: int, pretrained_ckpt: Optional[str] = None,
                pretrained_imagenet: bool = True) -> DeepPT:
    model = DeepPT(n_genes=n_genes, pretrained=pretrained_imagenet)
    if pretrained_ckpt is not None:
        state = torch.load(pretrained_ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        own = model.state_dict()
        transferable = {k: v for k, v in state.items()
                        if k in own and own[k].shape == v.shape and not k.startswith("head.")}
        model.load_state_dict(transferable, strict=False)
        print(f"[deeppt] loaded {len(transferable)} tensors from {pretrained_ckpt}")
    return model
