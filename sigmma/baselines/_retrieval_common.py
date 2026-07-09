"""Shared components for the H&E–ST retrieval / contrastive baselines.

The retrieval baselines (``openai_clip``, ``plip``, ``omiclip``, ``bleep``)
each live in their own subpackage but share:

- ``_normalize``          — L2 normalisation helper.
- ``_UNIImageBackbone``   — UNI image encoder usable as a swap-in image tower.
- ``_HFClipLike``         — base class for the HuggingFace CLIP-API baselines
                            (``openai_clip`` and ``plip``).

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

from sigmma.model.losses import symmetric_InfoNCE
from sigmma.model.models_image_backbones import UNI


def _normalize(t: torch.Tensor) -> torch.Tensor:
    return F.normalize(t, p=2, dim=-1)


# ImageNet normalization used by UNI (timm ViT trained with ImageNet stats).
_UNI_MEAN = (0.485, 0.456, 0.406)
_UNI_STD = (0.229, 0.224, 0.225)


class _UNIImageBackbone(nn.Module):
    """UNI image encoder for use as a swap-in replacement in baseline classes.

    No LoRA injection: baseline UNI variants do full fine-tuning to match each
    baseline's native FT regime. Output dim is read from the underlying UNI
    wrapper (1024 for UNI v1, 1536 for UNI2-h).
    """

    def __init__(self, uni_ckpt_path: str, image_size: int = 224):
        super().__init__()
        self.uni = UNI(checkpoint_path=uni_ckpt_path)
        self.image_size = image_size
        self.register_buffer("mean_rgb", torch.tensor(_UNI_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std_rgb", torch.tensor(_UNI_STD).view(1, 3, 1, 1))
        self.out_features = int(self.uni.feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-1] != self.image_size:
            image = F.interpolate(
                image, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
        image = (image - self.mean_rgb) / self.std_rgb
        return self.uni.model(image)  # bypass UNI.forward's no-grad path


class _HFClipLike(nn.Module):
    """Shared zero-shot wrapper for HuggingFace CLIP-API checkpoints.

    Handles preprocessing (resize → CLIP normalisation) on tensor batches and
    text tokenisation per forward call. Encoders are frozen.
    """

    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, hf_model_id: str, image_size: int = 224,
                 freeze_encoders: bool = True,
                 loss_temperature: Optional[float] = None,
                 image_backbone: str = "default",
                 uni_ckpt_path: Optional[str] = None):
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
        self.image_backbone = image_backbone
        self.register_buffer(
            "mean_rgb", torch.tensor(self.CLIP_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std_rgb", torch.tensor(self.CLIP_STD).view(1, 3, 1, 1)
        )

        if image_backbone == "uni":
            if uni_ckpt_path is None:
                raise ValueError("image_backbone='uni' requires uni_ckpt_path.")
            self._uni_image_encoder = _UNIImageBackbone(uni_ckpt_path, image_size=image_size)
            proj_dim = int(self.model.config.projection_dim)
            self.image_projection = nn.Linear(self._uni_image_encoder.out_features, proj_dim)
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
            # When fine-tuning the UNI variant, the text side is still loaded
            # from the HF checkpoint but only the text tower is needed; freeze
            # the unused vision_model to avoid wasted optimizer state.
            for p in self.model.vision_model.parameters():
                p.requires_grad = False
            self.model.visual_projection.requires_grad_(False)

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
        if self.image_backbone == "uni":
            # UNI does its own ImageNet normalization inside _UNIImageBackbone.
            image_features = self._uni_image_encoder(image)
            image_embedding = self.image_projection(image_features)
        else:
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
            # Loss-space embeddings (== normalized here; symmetric_InfoNCE
            # re-normalizes idempotently). The gradient-accumulation path caches
            # these to pool negatives across micro-batches.
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
