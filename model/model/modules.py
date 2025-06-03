import torch
import torch.nn as nn

from novae import Novae
from .losses import HMLC


class base_contrastive(nn.Module):
    def __init__(self, config, image_encoder, spatial_encoder):
        super(base_contrastive, self).__init__()
        self.config = config

        self.image_encoder = image_encoder
        self.image_projection = nn.Linear(in_features=config.image_encoder.output_dim, out_features=config.model.projection_dim)

        self.spatial_encoder = spatial_encoder
        self.spatial_projection = nn.Linear(in_features=config.spatial_encoder.output_dim, out_features=config.model.projection_dim)


    def forward(self, x, y):
        # image embedding
        image_embedding = self.image_encoder(x)
        image_embedding = self.image_projection(image_embedding)

        # spatial embedding
        self.spatial_encoder.spatial_neighbors(y, radius=self.config.model.spatial_radius)
        self.spatial_encoder.compute_representations(
            y, zero_shot=True
        )  # y should be adata
        spatial_embedding = y.obsm["novae_latent"]
        spatial_embedding = self.spatial_projection(spatial_embedding)

        # constrastive loss
        loss = self.contrastive_loss(image_embedding, spatial_embedding)
        return loss, image_embedding, spatial_embedding


class cel_type_composition_predictor(nn.Module):
    def __init__(self, config, image_encoder, spatial_encoder):
        super(cel_type_composition_predictor, self).__init__()
        self.config = config
        
        self.image_encoder = image_encoder
        self.image_projection = nn.Linear(in_features=config.image_encoder.output_dim, out_features=config.model.projection_dim)

        self.spatial_encoder = spatial_encoder
        
        self.spatial_projection = nn.Linear(in_features=config.spatial_encoder.output_dim, out_features=config.model.projection_dim)
        self.predictor = nn.Linear(
            config.model.encoder_dim, config.model.num_cell_types
        )

    def forward(self, x, y, z):
        # image embedding
        self.image_encoder.eval()
        image_embedding = self.image_encoder(x)
        image_embedding = self.image_projection(image_embedding)

        # spatial embedding
                # spatial embedding
        self.spatial_encoder.spatial_neighbors(y, radius=self.config.model.spatial_radius)
        self.spatial_encoder.compute_representations(
            y, zero_shot=True
        )  # y should be adata
        spatial_embedding = y.obsm["novae_latent"]
        spatial_embedding = self.spatial_projection(spatial_embedding)

        # cell type composition prediction
        composition_prediction1 = self.predictor(image_embedding)
        composition_prediction2 = self.predictor(spatial_embedding)

        loss1 = nn.CrossEntropyLoss()(composition_prediction1, z)
        loss2 = nn.CrossEntropyLoss()(composition_prediction2, z)
        loss = loss1 + loss2
        return loss, composition_prediction1, composition_prediction2, image_embedding, spatial_embedding
        



