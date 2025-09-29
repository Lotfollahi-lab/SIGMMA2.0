import torch
import torch.nn as nn
import timm
from pathlib import Path
from PIL import Image
from torchvision.transforms import Resize

from .losses import symmetric_InfoNCE

import loki

class openAI_CLIP(nn.Module):
    def __init__(self, config, model, processor):
        super(openAI_CLIP, self).__init__()
        self.config = config
        self.model = model
        self.processor = processor

    def forward(self, x, y):
        """
        x: PIL image
        y: list of text descriptions
        """
        x = x.clamp(0, 1)  # Ensure image tensor is in the range [0, 1]
        inputs = self.processor(text=y, images=x, return_tensors="pt", padding=True, do_rescale=False, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        inputs['return_loss'] = True

        # Ensure inputs are on the same device as the model
        outputs = self.model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        model_outputs = {"loss": outputs.loss,
                         "image_embedding": image_embeds,
                         "spatial_embedding": text_embeds}
        return model_outputs

class PLIP(nn.Module):
    def __init__(self, config, model, processor):
        super(PLIP, self).__init__()
        self.config = config
        self.model = model
        self.processor = processor

    def forward(self, x, y):
        """
        x: PIL image
        y: list of text descriptions
        """
        x = x.clamp(0, 1)  # Ensure image tensor is in the range [0, 1]
        inputs = self.processor(text=y, images=x, return_tensors="pt", padding=True, do_rescale=False, truncation=True, max_length=77)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        inputs['return_loss'] = True

        # Ensure inputs are on the same device as the model
        outputs = self.model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        model_outputs = {"loss": outputs.loss,
                         "image_embedding": image_embeds,
                         "spatial_embedding": text_embeds}
        return model_outputs


class OmiCLIP(nn.Module):
    def __init__(self, config, model, preprocess, tokenizer):
        super(OmiCLIP, self).__init__()
        self.config = config
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer

        self.loss_fn = symmetric_InfoNCE(temperature=config.model.temperature)

    def _encode_images_flexible(self, model, preprocess, inputs, device=None, *,
                           mode="auto",  # "auto" | "paths" | "pil" | "tensor" | "embeddings"
                           normalize_outputs=False):
        """
        Flexible encoder:
        - file path (str or list[str])
        - PIL.Image (or list[PIL.Image])
        - image tensor (C,H,W or B,C,H,W)  -> pixel space
        - precomputed embeddings tensor (D or B,D) -> returns as-is (optionally normalized)
        """
        if device is None:
            device = next(model.parameters()).device

        if mode == "auto":
            if isinstance(inputs, (str, Path)) or (isinstance(inputs, (list, tuple)) and inputs and isinstance(inputs[0], (str, Path))):
                mode = "paths"
            elif isinstance(inputs, Image.Image) or (isinstance(inputs, (list, tuple)) and inputs and isinstance(inputs[0], Image.Image)):
                mode = "pil"
            elif torch.is_tensor(inputs):
                if inputs.dim() in (1, 2) and (inputs.dim()==1 or inputs.size(-1) >= 32) and inputs.dim()==2 and inputs.size(-2) < 64:
                    pass
                if inputs.dim() in (3, 4):  # (C,H,W) or (B,C,H,W) -> pixel
                    mode = "tensor"
                elif inputs.dim() in (1, 2):  # (D,) or (B,D) -> likely embedding
                    mode = "embeddings"
                else:
                    raise ValueError(f"Unsupported tensor shape for auto mode: {tuple(inputs.shape)}")
            elif isinstance(inputs, (list, tuple)) and inputs and torch.is_tensor(inputs[0]):
                # list of tensors
                t0 = inputs[0]
                if t0.dim() == 3:
                    mode = "tensor"
                elif t0.dim() == 1 or t0.dim() == 2:
                    mode = "embeddings"
                else:
                    raise ValueError(f"Unsupported tensor list shape: {t0.shape}")
            else:
                raise ValueError(f"Cannot auto-detect mode for type {type(inputs)}. Specify mode explicitly.")

        if mode == "embeddings":
            emb = inputs if torch.is_tensor(inputs) else torch.stack(inputs, dim=0)
            emb = emb.to(device)
            if normalize_outputs:
                emb = torch.nn.functional.normalize(emb, dim=-1)
            return emb

        batch = None
        if mode == "paths":
            paths = [inputs] if isinstance(inputs, (str, Path)) else list(inputs)
            images = [Image.open(p).convert("RGB") for p in paths]
            tensors = [preprocess(img) for img in images]
            batch = torch.stack(tensors, dim=0)
        elif mode == "pil":
            images = [inputs] if isinstance(inputs, Image.Image) else list(inputs)
            tensors = [preprocess(img.convert("RGB")) for img in images]
            batch = torch.stack(tensors, dim=0)
        elif mode == "tensor":
            # (C,H,W) or (B,C,H,W)
            if inputs.dim() == 3:
                batch = inputs.unsqueeze(0)
            else:
                batch = inputs
            transform = Resize((224, 224))
            batch = transform(batch)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        batch = batch.to(device, non_blocking=True)

        if hasattr(model, "encode_image"):
            emb = model.encode_image(batch)
        else:
            emb = model(batch)

        if normalize_outputs:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb

    def forward(self, image, text):
        # loki.utils.encode_images(model: Module, preprocess: callable, image_paths: List[str], device: str | device)→ Tensor
        device = next(self.model.parameters()).device
        image_embeddings = self._encode_images_flexible(self.model, self.preprocess, image, device=device)
        text_embeddings = loki.utils.encode_texts(self.model, self.tokenizer, text, device=device)

        loss = self.loss_fn(image_embeddings, text_embeddings)
        model_outputs = {"loss": loss,
                         "image_embedding": image_embeddings,
                         "spatial_embedding": text_embeddings}
        return model_outputs

class UNI(nn.Module):
    def __init__(self, checkpoint_path=None):
        super(UNI, self).__init__()
        uni_model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )

        if checkpoint_path:
            uni_model.load_state_dict(
                torch.load(
                    Path(checkpoint_path, "pytorch_model.bin"), map_location="cpu"
                ),
                strict=True,
            )
        self.model = uni_model

    def get_image_features(self, x):
        with torch.inference_mode():
            image_embedding = self.model(x)
        return image_embedding