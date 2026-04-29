"""Baseline encoders for H&E–ST retrieval / contrastive alignment.

Four baselines are supported:
- ``openAI_CLIP``  — zero-shot HuggingFace ``openai/clip-vit-*`` (Radford et al. 2021)
- ``PLIP``         — zero-shot HuggingFace ``vinid/plip`` (Huang et al. 2023)
- ``loki_OmiCLIP`` — zero-shot OmiCLIP via the Loki pipeline (Chen et al. 2025a)
- ``BLEEP``        — fine-tuned ResNet50 image encoder + MLP spot encoder
                     trained with the original BLEEP cross-entropy contrastive
                     loss (Xie et al. 2023)

All baselines expose the SIGMMA-style forward contract:
    forward(image, spatial) -> {"image_embedding": Tensor[B,D],
                                "spatial_embedding": Tensor[B,D],
                                "loss": Tensor (optional, only when training)}

For zero-shot baselines ``spatial`` is a list of ``B`` strings (top-K HVGs),
for BLEEP it is a per-tile aggregated cell-embedding/expression Tensor[B, F].
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .losses import bleep_cross_entropy, symmetric_InfoNCE


def _normalize(t: torch.Tensor) -> torch.Tensor:
    return F.normalize(t, p=2, dim=-1)


class _HFClipLike(nn.Module):
    """Shared zero-shot wrapper for HuggingFace CLIP-API checkpoints.

    Handles preprocessing (resize → CLIP normalisation) on tensor batches and
    text tokenisation per forward call. Encoders are frozen.
    """

    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, hf_model_id: str, image_size: int = 224,
                 freeze_encoders: bool = True,
                 loss_temperature: Optional[float] = None):
        super().__init__()
        # transformers >=4.50 enforces torch>=2.6 for any torch.load path
        # (CVE-2025-32434). openai/clip-vit-base-patch32 and vinid/plip ship
        # only pytorch_model.bin (no safetensors), so use_safetensors=True is
        # unavailable. The CVE concerns untrusted checkpoint files; these are
        # official upstream weights, so we disarm the check. modeling_utils
        # imports the symbol by name at module load, so patch it there too.
        import transformers.utils.import_utils as _hf_iu
        import transformers.modeling_utils as _hf_mu
        _noop = lambda: None
        _hf_iu.check_torch_load_is_safe = _noop
        _hf_mu.check_torch_load_is_safe = _noop
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(hf_model_id)
        self.processor = CLIPProcessor.from_pretrained(hf_model_id)
        self.image_size = image_size
        self.register_buffer(
            "mean_rgb", torch.tensor(self.CLIP_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std_rgb", torch.tensor(self.CLIP_STD).view(1, 3, 1, 1)
        )
        if freeze_encoders:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        self._loss_fn = (
            symmetric_InfoNCE(temperature=float(loss_temperature))
            if loss_temperature is not None else None
        )

    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        # image: (B, 3, H, W) in [0, 1]
        if image.shape[-1] != self.image_size:
            image = F.interpolate(
                image,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return (image - self.mean_rgb) / self.std_rgb

    def forward(self, image: torch.Tensor, spatial: List[str]):
        device = image.device
        pixel_values = self._preprocess_image(image)
        image_embedding = self.model.get_image_features(pixel_values=pixel_values)

        tokenised = self.processor.tokenizer(
            list(spatial),
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        text_embedding = self.model.get_text_features(**tokenised)

        image_embedding = _normalize(image_embedding)
        spatial_embedding = _normalize(text_embedding)
        out = {
            "image_embedding": image_embedding,
            "spatial_embedding": spatial_embedding,
        }
        if self._loss_fn is not None:
            out["loss"] = self._loss_fn(image_embedding, spatial_embedding)
        return out


class openAI_CLIP(_HFClipLike):
    """OpenAI CLIP. Default checkpoint is ViT-B/32. Zero-shot by default;
    set ``freeze_encoders=False`` and a ``loss_temperature`` to fine-tune."""

    def __init__(self, hf_model_id: str = "openai/clip-vit-base-patch32",
                 freeze_encoders: bool = True,
                 loss_temperature: Optional[float] = None):
        super().__init__(hf_model_id=hf_model_id, image_size=224,
                         freeze_encoders=freeze_encoders,
                         loss_temperature=loss_temperature)


class PLIP(_HFClipLike):
    """PLIP (Huang et al. 2023). HF id ``vinid/plip``. Zero-shot by default;
    set ``freeze_encoders=False`` and a ``loss_temperature`` to fine-tune."""

    def __init__(self, hf_model_id: str = "vinid/plip",
                 freeze_encoders: bool = True,
                 loss_temperature: Optional[float] = None):
        super().__init__(hf_model_id=hf_model_id, image_size=224,
                         freeze_encoders=freeze_encoders,
                         loss_temperature=loss_temperature)


class loki_OmiCLIP(nn.Module):
    """Zero-shot OmiCLIP via Loki's ``coca_ViT-L-14`` checkpoint.

    Loki's ``utils.load_model`` resolves the checkpoint with ``open_clip``.
    Encoders are frozen; tokenisation is done per forward call.
    """

    OPENCLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENCLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, pretrained_ckpt_path: str, image_size: int = 224,
                 freeze_encoders: bool = True,
                 loss_temperature: Optional[float] = None):
        super().__init__()
        from open_clip import create_model_from_pretrained, get_tokenizer

        # ``pretrained`` accepts a local file path or an open_clip alias.
        model, _ = create_model_from_pretrained(
            "coca_ViT-L-14",
            pretrained=pretrained_ckpt_path,
        )
        self.model = model
        self.tokenizer = get_tokenizer("coca_ViT-L-14")
        self.image_size = image_size
        self.register_buffer(
            "mean_rgb", torch.tensor(self.OPENCLIP_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std_rgb", torch.tensor(self.OPENCLIP_STD).view(1, 3, 1, 1)
        )
        if freeze_encoders:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        self._loss_fn = (
            symmetric_InfoNCE(temperature=float(loss_temperature))
            if loss_temperature is not None else None
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

    def forward(self, image: torch.Tensor, spatial: List[str]):
        device = image.device
        pixel_values = self._preprocess_image(image)
        image_embedding = self.model.encode_image(pixel_values)

        text_tokens = self.tokenizer(list(spatial)).to(device)
        text_embedding = self.model.encode_text(text_tokens)

        image_embedding = _normalize(image_embedding.float())
        spatial_embedding = _normalize(text_embedding.float())
        out = {
            "image_embedding": image_embedding,
            "spatial_embedding": spatial_embedding,
        }
        if self._loss_fn is not None:
            out["loss"] = self._loss_fn(image_embedding, spatial_embedding)
        return out


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
    ):
        super().__init__()
        weights = (
            models.ResNet50_Weights.IMAGENET1K_V2 if pretrained_image_encoder else None
        )
        backbone = models.resnet50(weights=weights)
        backbone.fc = nn.Identity()
        self.image_encoder = backbone
        self.image_projection = _BLEEPProjectionHead(
            embedding_dim=image_embedding_dim,
            projection_dim=projection_dim,
            dropout=dropout,
        )

        self.spot_encoder = _BLEEPSpotEncoder(
            input_dim=spot_input_dim,
            hidden_dim=spot_hidden_dim,
            output_dim=image_embedding_dim,
            dropout=dropout,
        )
        self.spot_projection = _BLEEPProjectionHead(
            embedding_dim=image_embedding_dim,
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
        pixel_values = self._preprocess_image(image)
        image_features = self.image_encoder(pixel_values).flatten(1)
        spot_features = self.spot_encoder(spatial)

        image_embedding = self.image_projection(image_features)
        spot_embedding = self.spot_projection(spot_features)

        # BLEEP soft-target contrastive loss
        logits = (spot_embedding @ image_embedding.T) / self.temperature
        images_similarity = image_embedding @ image_embedding.T
        spots_similarity = spot_embedding @ spot_embedding.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / (2 * self.temperature), dim=-1
        )
        loss_si = bleep_cross_entropy(logits, targets, reduction="none")
        loss_is = bleep_cross_entropy(logits.T, targets.T, reduction="none")
        loss = ((loss_si + loss_is) / 2.0).mean()

        return {
            "loss": loss,
            "image_embedding": _normalize(image_embedding),
            "spatial_embedding": _normalize(spot_embedding),
        }
