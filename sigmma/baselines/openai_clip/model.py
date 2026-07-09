"""OpenAI CLIP retrieval baseline (Radford et al. 2021).

Zero-shot HuggingFace ``openai/clip-vit-*``. Shares the CLIP-API forward /
preprocessing logic with PLIP via ``_HFClipLike``.
"""

from typing import Optional

from sigmma.baselines._retrieval_common import _HFClipLike


class openAI_CLIP(_HFClipLike):
    """OpenAI CLIP. Default checkpoint is ViT-B/32. Zero-shot by default;
    set ``freeze_encoders=False`` and a ``loss_temperature`` to fine-tune."""

    def __init__(self, hf_model_id: str = "openai/clip-vit-base-patch32",
                 freeze_encoders: bool = True,
                 loss_temperature: Optional[float] = None,
                 image_backbone: str = "default",
                 uni_ckpt_path: Optional[str] = None):
        super().__init__(hf_model_id=hf_model_id, image_size=224,
                         freeze_encoders=freeze_encoders,
                         loss_temperature=loss_temperature,
                         image_backbone=image_backbone,
                         uni_ckpt_path=uni_ckpt_path)
