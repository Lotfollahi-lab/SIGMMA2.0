import torch
import torch.nn as nn

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

        # Ensure inputs are on the same device as the model
        outputs = self.model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        ground_truth = torch.arange(len(x),dtype=torch.long,device=self.model.device)
        loss_img = nn.CrossEntropyLoss()(image_embeds @ text_embeds.T, ground_truth)
        loss_txt = nn.CrossEntropyLoss()(text_embeds @ image_embeds.T, ground_truth)

        loss = (loss_img + loss_txt) / 2.0
        return loss, image_embeds, text_embeds

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

        # Ensure inputs are on the same device as the model
        outputs = self.model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        ground_truth = torch.arange(len(x),dtype=torch.long,device=self.model.device)
        loss_img = nn.CrossEntropyLoss()(image_embeds @ text_embeds.T, ground_truth)
        loss_txt = nn.CrossEntropyLoss()(text_embeds @ image_embeds.T, ground_truth)

        loss = (loss_img + loss_txt) / 2.0
        return loss, image_embeds, text_embeds


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
        return loss, image_embeddings, text_embeddings