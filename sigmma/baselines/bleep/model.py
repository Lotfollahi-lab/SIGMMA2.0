"""BLEEP retrieval baseline (Xie et al. 2023).

Fine-tuned ResNet50 (or UNI) image encoder + MLP/identity spot encoder,
trained with the original BLEEP soft-target cross-entropy contrastive loss.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from sigmma.baselines._retrieval_common import _normalize, _UNIImageBackbone
from sigmma.model.losses import bleep_cross_entropy


class _BLEEPSpotEncoder(nn.Module):
    """MLP spot encoder used by BLEEP (Xie et al. 2023, Sec. 3.1)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _BLEEPProjectionHead(nn.Module):
    """Projection head with residual connection, mirroring BLEEP's official
    implementation (https://github.com/bowang-lab/BLEEP/blob/main/modules.py).
    """

    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        return self.layer_norm(x)


class BLEEP(nn.Module):
    """Fine-tunable BLEEP baseline.

    Image encoder defaults to ImageNet-pretrained ResNet50 (manuscript
    Suppl. D.3: no public BLEEP checkpoint, ResNet50 backbone is used). The
    spot encoder maps the pre-aggregated tile-level cell embedding (or gene
    expression vector) to the shared latent space. Training uses the BLEEP
    soft-target cross-entropy contrastive loss.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        spot_input_dim: int,
        image_embedding_dim: int = 2048,
        projection_dim: int = 256,
        spot_hidden_dim: int = 512,
        temperature: float = 1.0,
        dropout: float = 0.1,
        image_size: int = 224,
        pretrained_image_encoder: bool = True,
        image_backbone: str = "default",
        uni_ckpt_path: Optional[str] = None,
    ):
        super().__init__()
        self.image_backbone = image_backbone
        if image_backbone == "uni":
            if uni_ckpt_path is None:
                raise ValueError("image_backbone='uni' requires uni_ckpt_path.")
            self.image_encoder = _UNIImageBackbone(uni_ckpt_path, image_size=image_size)
            # UNI emits 1024-d features; override the BLEEP default 2048.
            image_embedding_dim = self.image_encoder.out_features
        elif image_backbone == "default":
            weights = (
                models.ResNet50_Weights.IMAGENET1K_V2 if pretrained_image_encoder else None
            )
            backbone = models.resnet50(weights=weights)
            backbone.fc = nn.Identity()
            self.image_encoder = backbone
        else:
            raise ValueError(f"unknown image_backbone={image_backbone!r}")

        self.image_projection = _BLEEPProjectionHead(
            embedding_dim=image_embedding_dim,
            projection_dim=projection_dim,
            dropout=dropout,
        )

        # Upstream-faithful: spot input goes DIRECTLY into ProjectionHead
        # (matches bowang-lab/BLEEP `models.py:CLIPModel.forward` where
        # `spot_features = batch["reduced_expression"]` with no encoder).
        # The earlier `_BLEEPSpotEncoder` MLP added ~1.6M params with no
        # image-side counterpart and was a deviation from upstream.
        self.spot_encoder = nn.Identity()
        self.spot_projection = _BLEEPProjectionHead(
            embedding_dim=spot_input_dim,
            projection_dim=projection_dim,
            dropout=dropout,
        )

        self.temperature = temperature
        self.image_size = image_size
        self.register_buffer(
            "mean_rgb", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std_rgb", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1)
        )

    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-1] != self.image_size:
            image = F.interpolate(
                image,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return (image - self.mean_rgb) / self.std_rgb

    def forward(self, image: torch.Tensor, spatial: torch.Tensor):
        if self.image_backbone == "uni":
            # _UNIImageBackbone handles ImageNet normalization internally and
            # returns a (B, 1024) feature vector directly.
            image_features = self.image_encoder(image)
        else:
            pixel_values = self._preprocess_image(image)
            image_features = self.image_encoder(pixel_values).flatten(1)
        spot_features = self.spot_encoder(spatial)

        image_embedding = self.image_projection(image_features)
        spot_embedding = self.spot_projection(spot_features)

        # BLEEP soft-target contrastive loss. NOTE: BLEEP's loss operates on the
        # UNnormalized projections, so those are the loss-space embeddings cached
        # by the gradient-accumulation path (the returned normalized embeddings
        # are for downstream/retrieval use only).
        loss = self.contrastive_loss_from_embeddings(image_embedding, spot_embedding)

        return {
            "loss": loss,
            "image_embedding": _normalize(image_embedding),
            "spatial_embedding": _normalize(spot_embedding),
            "image_embedding_for_loss": image_embedding,
            "spatial_embedding_for_loss": spot_embedding,
        }

    def contrastive_loss_from_embeddings(self, image_embedding, spot_embedding):
        """BLEEP soft-target contrastive loss over (possibly pooled) UNnormalized
        projections. Used both by forward and by the gradient-accumulation path
        to compute the loss over the full effective batch."""
        logits = (spot_embedding @ image_embedding.T) / self.temperature
        images_similarity = image_embedding @ image_embedding.T
        spots_similarity = spot_embedding @ spot_embedding.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / (2 * self.temperature), dim=-1
        )
        loss_si = bleep_cross_entropy(logits, targets, reduction="none")
        loss_is = bleep_cross_entropy(logits.T, targets.T, reduction="none")
        return ((loss_si + loss_is) / 2.0).mean()
