import numpy as np
import torch
import torch.nn as nn

from .losses import InfoNCE, symmetric_InfoNCE
from .layers import DINOHead

import loki

class base_contrastive(nn.Module):
    def __init__(self, config, image_encoder):
        super(base_contrastive, self).__init__()
        self.config = config

        self.image_encoder = image_encoder
        self.image_projection = DINOHead(
            in_dim = config.model.image_encoder_output_dim,
            out_dim = config.model.projection_dim,
            )

        self.spatial_projection = DINOHead(
            in_dim = config.model.spatial_encoder_output_dim,
            out_dim = config.model.projection_dim,
        )

        self.loss_fn = InfoNCE(temperature=config.model.infoNCE_temperature)

    def forward(self, x, y):
        """
        x: image tensor, shape (B, C, H, W)
        y: spatial tensor, shape (B, D) where D is the number of spatial features
        """
        # image embedding
        with torch.no_grad():
            image_embedding = self.image_encoder(x)
        image_embedding = self.image_projection(image_embedding)

        # spatial embedding
        # y[torch.isnan(y)] = 0.0
        spatial_embedding = self.spatial_projection(y)

        image_embedding = torch.nn.functional.normalize(image_embedding, dim=-1)
        spatial_embedding = torch.nn.functional.normalize(spatial_embedding, dim=-1)

        # constrastive loss
        loss = self.loss_fn(image_embedding, spatial_embedding)
        return loss, image_embedding, spatial_embedding


class cel_type_composition_predictor(nn.Module):
    def __init__(self, config, image_encoder, spatial_encoder):
        super(cel_type_composition_predictor, self).__init__()
        self.config = config

        self.image_encoder = image_encoder
        self.image_projection = nn.Linear(
            in_features=config.model.image_encoder_output_dim,
            out_features=config.model.projection_dim,
        )

        self.spatial_projection = nn.Linear(
            in_features=config.model.spatial_encoder_output_dim,
            out_features=config.model.projection_dim,
        )
        self.predictor = nn.Linear(
            config.model.encoder_dim, config.model.num_cell_types
        )

    def forward(self, x, y, z):
        """
        x: image tensor, shape (B, C, H, W)
        y: spatial tensor, shape (B, D) where D is the number of spatial features
        z: cell type composition, shape (B, L) with L being the number of cell types
        """
        # image embedding
        self.image_encoder.eval()
        image_embedding = self.image_encoder(x)
        image_embedding = self.image_projection(image_embedding)

        # spatial embedding
        spatial_embedding = self.spatial_projection(y)

        # cell type composition prediction
        composition_prediction1 = self.predictor(image_embedding)
        composition_prediction2 = self.predictor(spatial_embedding)

        loss1 = nn.CrossEntropyLoss()(composition_prediction1, z)
        loss2 = nn.CrossEntropyLoss()(composition_prediction2, z)
        loss = loss1 + loss2
        return (
            loss,
            composition_prediction1,
            composition_prediction2,
            image_embedding,
            spatial_embedding,
        )

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