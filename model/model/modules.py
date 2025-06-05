import torch
import torch.nn as nn

from .losses import InfoNCE
from .layers import DINOHead

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
        image_embedding = self.image_encoder(x)
        image_embedding = self.image_projection(image_embedding)

        # spatial embedding
        y[torch.isnan(y)] = 0.0
        spatial_embedding = self.spatial_projection(y)

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
