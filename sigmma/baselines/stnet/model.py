"""ST-Net (He et al., Nat Biomed Eng 2020) — minimal reimplementation.

Architecture: ImageNet-pretrained DenseNet-121 → flatten → Linear(1024, G).
Bias of the final layer is initialised to the per-gene mean of the training
log1p target so the model starts at the unconditional baseline (mirrors the
original ``stnet.utils.nn.set_out_features`` initialisation).

Reference: https://github.com/bryanhe/ST-Net (``stnet/cmd/run_spatial.py``).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torchvision


class STNet(nn.Module):
    def __init__(self, n_genes: int, pretrained: bool = True):
        super().__init__()
        backbone = torchvision.models.densenet121(
            weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
            if pretrained else None
        )
        num_features = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(num_features, n_genes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) — typically 224×224 ImageNet-normalised.
        features = self.backbone(x)
        return self.head(features)

    def set_output_bias(self, mean_expression: torch.Tensor) -> None:
        """Initialise the regression head's bias to the mean log1p expression
        across the training set. Matches the original ST-Net training script.
        """
        assert mean_expression.numel() == self.head.bias.numel(), (
            f"mean_expression has {mean_expression.numel()} entries, head expects "
            f"{self.head.bias.numel()}"
        )
        with torch.no_grad():
            self.head.bias.copy_(mean_expression.to(self.head.bias.device))


def build_model(n_genes: int, pretrained_ckpt: Optional[str] = None,
                pretrained_imagenet: bool = True) -> STNet:
    model = STNet(n_genes=n_genes, pretrained=pretrained_imagenet)
    if pretrained_ckpt is not None:
        state = torch.load(pretrained_ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # The author's checkpoints predict a different gene-set, so we only
        # transfer the backbone weights and let the head be fine-tuned.
        backbone_state = {k.replace("backbone.", "", 1): v for k, v in state.items()
                          if k.startswith("backbone.")}
        if backbone_state:
            missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
            if unexpected:
                print(f"[stnet] backbone load: {len(unexpected)} unexpected keys")
    return model
