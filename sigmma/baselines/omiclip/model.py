"""OmiCLIP retrieval baseline via the Loki pipeline (Chen et al. 2025a).

Zero-shot OmiCLIP using Loki's ``coca_ViT-L-14`` checkpoint, resolved through
``open_clip``. Encoders are frozen; tokenisation is done per forward call.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from sigmma.baselines._retrieval_common import _normalize, _UNIImageBackbone
from sigmma.model.losses import symmetric_InfoNCE


class loki_OmiCLIP(nn.Module):
    """Zero-shot OmiCLIP via Loki's ``coca_ViT-L-14`` checkpoint.

    Loki's ``utils.load_model`` resolves the checkpoint with ``open_clip``.
    Encoders are frozen; tokenisation is done per forward call.
    """

    OPENCLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENCLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, pretrained_ckpt_path: str, image_size: int = 224,
                 freeze_encoders: bool = True,
                 loss_temperature: Optional[float] = None,
                 image_backbone: str = "default",
                 uni_ckpt_path: Optional[str] = None):
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
        self.image_backbone = image_backbone
        self.register_buffer(
            "mean_rgb", torch.tensor(self.OPENCLIP_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std_rgb", torch.tensor(self.OPENCLIP_STD).view(1, 3, 1, 1)
        )

        if image_backbone == "uni":
            if uni_ckpt_path is None:
                raise ValueError("image_backbone='uni' requires uni_ckpt_path.")
            self._uni_image_encoder = _UNIImageBackbone(uni_ckpt_path, image_size=image_size)
            # CoCa ViT-L/14 text/contrastive embedding dim is 768.
            # Probe via a token forward to be safe.
            with torch.no_grad():
                dummy = self.tokenizer(["probe"])
                txt = self.model.encode_text(dummy)
                text_dim = int(txt.shape[-1])
            self.image_projection = nn.Linear(self._uni_image_encoder.out_features, text_dim)
        elif image_backbone != "default":
            raise ValueError(f"unknown image_backbone={image_backbone!r}")

        if freeze_encoders:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
            if image_backbone == "uni":
                for p in self._uni_image_encoder.parameters():
                    p.requires_grad = False
                for p in self.image_projection.parameters():
                    p.requires_grad = False
        elif image_backbone == "uni":
            # Freeze the discarded CoCa visual tower (and any cross-modal
            # captioning bits that depend on the visual features) when
            # fine-tuning the UNI variant.
            for p in self.model.visual.parameters():
                p.requires_grad = False

        self._loss_fn = (
            symmetric_InfoNCE(temperature=float(loss_temperature))
            if loss_temperature is not None else None
        )

    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-1] != self.image_size:
            image = torch.nn.functional.interpolate(
                image,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return (image - self.mean_rgb) / self.std_rgb

    def forward(self, image: torch.Tensor, spatial: List[str]):
        device = image.device
        if self.image_backbone == "uni":
            image_features = self._uni_image_encoder(image)
            image_embedding = self.image_projection(image_features)
        else:
            pixel_values = self._preprocess_image(image)
            image_embedding = self.model.encode_image(pixel_values)

        text_tokens = self.tokenizer(list(spatial)).to(device)
        text_embedding = self.model.encode_text(text_tokens)

        image_embedding = _normalize(image_embedding.float())
        spatial_embedding = _normalize(text_embedding.float())
        out = {
            "image_embedding": image_embedding,
            "spatial_embedding": spatial_embedding,
            # Loss-space embeddings (== normalized here; symmetric_InfoNCE
            # re-normalizes idempotently). Cached by the gradient-accumulation
            # path to pool negatives across micro-batches.
            "image_embedding_for_loss": image_embedding,
            "spatial_embedding_for_loss": spatial_embedding,
        }
        if self._loss_fn is not None:
            out["loss"] = self._loss_fn(image_embedding, spatial_embedding)
        return out

    def contrastive_loss_from_embeddings(self, image_embedding, spatial_embedding):
        """Contrastive loss over (possibly pooled) loss-space embeddings.

        Used by the gradient-accumulation path to compute symmetric InfoNCE over
        the full effective batch (all accumulated micro-batches concatenated)."""
        assert self._loss_fn is not None, "loss_temperature must be set to train."
        return self._loss_fn(image_embedding, spatial_embedding)
