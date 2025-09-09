import torch
import torch.nn as nn
import timm
from pathlib import Path

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
                         "text_embedding": text_embeds}
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
                         "text_embedding": text_embeds}
        return model_outputs


class OmiCLIP(nn.Module):
    def __init__(self, config, model, preprocess, tokenizer):
        super(OmiCLIP, self).__init__()
        self.config = config
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer

        self.loss_fn = symmetric_InfoNCE(temperature=config.model.temperature)

    def forward(self, image_path, text):
        # loki.utils.encode_images(model: Module, preprocess: callable, image_paths: List[str], device: str | device)→ Tensor
        image_embeddings = loki.utils.encode_images(self.model, self.preprocess, image_path, device=self.model.device)
        text_embeddings = loki.utils.encode_texts(self.model, self.tokenizer, text, device=self.model.device)

        loss = self.loss_fn(image_embeddings, text_embeddings)
        model_outputs = {"loss": loss,
                         "image_embedding": image_embeddings,
                         "text_embedding": text_embeddings}
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