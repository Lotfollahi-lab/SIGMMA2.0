import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union
import timm
from pathlib import Path
from PIL import Image
from torchvision.transforms import Resize

from .losses import symmetric_InfoNCE, bleep_cross_entropy
from .models_image_encoders import ResNet50

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
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

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

    @torch.no_grad()
    def get_image_features(self, x):
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.inference_mode():
            image_embedding = self.model.get_image_features(x)
        return image_embedding

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
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
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

    @torch.no_grad()
    def get_image_features(self, x):
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = x.clamp(0, 1)

        inputs = self.processor(images=x, return_tensors="pt", padding=True, do_rescale=False, truncation=True, max_length=77)

        pixel_values = inputs["pixel_values"].to(self.model.device)

        self.model.eval()
        with torch.inference_mode():
            image_embedding = self.model.get_image_features(pixel_values=pixel_values)

        return image_embedding


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

    def _encode_texts_flexible(
            self,
            model: torch.nn.Module,
            tokenizer: callable,
            texts: List[str],
            device: Union[str, torch.device]
        ) -> torch.Tensor:
        """
        Adapted from https://github.com/GuangyuWangLab2021/Loki/blob/main/src/loki/utils.py
        =================================
        Batch–encode a list of strings into L2‑normalized embeddings.
        Returns a tensor of shape (N, D).
        """
        # Tokenizer returns a dict of tensors
        text_inputs = tokenizer(texts)
        text_inputs = text_inputs.to(device)
        feats = model.encode_text(text_inputs).to(device)             # (N, D)
        return F.normalize(feats, p=2, dim=-1)

    def forward(self, image, text):
        # loki.utils.encode_images(model: Module, preprocess: callable, image_paths: List[str], device: str | device)→ Tensor
        device = next(self.model.parameters()).device
        image_embeddings = self._encode_images_flexible(self.model, self.preprocess, image, device)
        text_embeddings = self._encode_texts_flexible(self.model, self.tokenizer, text, device)

        loss = self.loss_fn(image_embeddings, text_embeddings)
        model_outputs = {"loss": loss,
                         "image_embedding": image_embeddings,
                         "spatial_embedding": text_embeddings}
        return model_outputs

    @torch.no_grad()
    def get_image_features(self, x):
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = x.clamp(0, 1)
        device = next(self.model.parameters()).device
        image_embedding = self._encode_images_flexible(self.model, self.preprocess, x, device)
        return image_embedding
    
class BleepProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=64,
        dropout=0.1,
    ):
        '''
        Adapted from https://github.com/bowang-lab/BLEEP/blob/main/modules.py
        Config parameters from https://github.com/bowang-lab/BLEEP/blob/main/config.py
        Accessed November 2025
        
        '''
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
    
class BLEEP(nn.Module):
    '''
    Based on https://github.com/bowang-lab/BLEEP/blob/main/models.py
    Accessed November 2025
    
    '''
    def __init__(
        self,
        temperature:float,
        spot_embedding:50, # number of Harmony batch corrected PCss
        image_embedding=2048, # for ResNet50
    ):
        super().__init__()
        self.image_encoder = ResNet50()
        
        # freeze parameters for ResNet50
        for n, p in self.image_encoder.named_parameters():
            if 'layer4' in n:
                p.requires_grad = True
            elif 'bn' in n:
                p.requires_grad = True  # let BNs adapt
            else:
                p.requires_grad = False
            
        self.image_projection = BleepProjectionHead(embedding_dim=image_embedding) # input dim, 2048 for resnet50
        self.spot_projection = BleepProjectionHead(embedding_dim=spot_embedding) # number of shared hvgs
        self.temperature = temperature

    def forward(self, image_features, spot_features):
        # ==== Embed ====
        image_features = self.image_encoder(image_features)  # [B, 2048]
        
        # ==== Project ====
        image_embeddings = self.image_projection(image_features)            # [B, D]
        spot_embeddings  = self.spot_projection(spot_features)              # [B, D]

        # ==== Normalize ====
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        spot_embeddings  = F.normalize(spot_embeddings, dim=-1)

        # ==== Logits ====
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature  # [B, B]

        # ==== Similarity matrices (true) ====
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity  = spot_embeddings  @ spot_embeddings.T

        # ==== Targets (soft or one-hot) ====
        with torch.no_grad():
            targets = F.softmax(((images_similarity + spots_similarity) / 2) / self.temperature, dim=-1)

        # ==== Debug diagnostics ====
        B = logits.size(0)
        eye = torch.eye(B, dtype=torch.bool, device=logits.device)
        off = ~eye

        diag_mean = logits.diag().mean().item()
        off_mean = logits[off].mean().item()
        pos_gap = diag_mean - off_mean
        top1 = (logits.argmax(1) == torch.arange(B, device=logits.device)).float().mean().item()

        print(f"\n[BLEEP Alignment] pos_gap={pos_gap:.4f} | diag={diag_mean:.4f} | off_diag={off_mean:.4f} | top1={top1*100:.1f}%")
        print(f"[BLEEP Img_Sim] off_diag={images_similarity[off].mean().item():.4f} | std={images_similarity[off].std().item():.4f}")
        print(f"[BLEEP Spot_Sim] off_diag={spots_similarity[off].mean().item():.4f} | std={spots_similarity[off].std().item():.4f}")
        print(f"[BLEEP Targets] diag_mean={targets.diag().mean().item():.6f} | off_mean={targets[off].mean().item():.6e}")
        
        with torch.no_grad():
            img_off = (image_embeddings @ image_embeddings.T)
            spot_off = (spot_embeddings  @ spot_embeddings.T)
            eye = torch.eye(B, device=img_off.device).bool()
            print(f"[Spread] img_off mean={img_off[~eye].mean():.3f} std={img_off[~eye].std():.3f} | "
                f"spot_off mean={spot_off[~eye].mean():.3f} std={spot_off[~eye].std():.3f} | "
                f"logit_scale={self.temperature}")

        # ==== Symmetric contrastive loss ====
        spots_loss  = bleep_cross_entropy(logits, targets, reduction='none')
        images_loss = bleep_cross_entropy(logits.T, targets.T, reduction='none')
        loss = (spots_loss + images_loss) / 2.0
        
        model_out = {
            "loss": loss.mean(),
            "image_embedding": image_embeddings,
            "spatial_embedding": spot_embeddings,
        }
        
        return model_out

    @torch.no_grad()
    def get_image_features(self, image_features):
        """Extract image embeddings for inference."""
        self.eval()
        image_embeddings = self.image_encoder(image_features)
        image_embeddings = self.image_projection(image_embeddings)
        # Normalize for consistent similarity computation
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        return image_embeddings


