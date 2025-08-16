import torch
import torch.nn as nn

from .losses import InfoNCE
from .layers import DINOHead

class single_img_spatial(nn.Module):
    def __init__(self, config, image_encoder):
        super(single_img_spatial, self).__init__()
        self.config = config

        self.image_encoder = image_encoder
        self.image_projection = DINOHead(
            in_dim = config.image_encoder_output_dim,
            out_dim = config.projection_dim,
            )

        self.spatial_projection = DINOHead(
            in_dim = config.spatial_encoder_output_dim,
            out_dim = config.projection_dim,
        )

        self.celltype_prediction = self._cell_type_composition_predictor(input_dim = config.projection_dim,
                                                                         output_dim = config.num_cell_types,
                                                                         dropout=config.dropout,
                                                                         activation=config.activation)
        self.loss_fn = InfoNCE(temperature=config.temperature)

    def _cell_type_composition_predictor(self,
                            input_dim: int,
                            output_dim: int,
                            dropout: float,
                            activation: str = 'relu'):

        if activation == 'relu':
            activation_layer = torch.nn.ReLU()
        elif activation == 'sigmoid':
            activation_layer = torch.nn.Sigmoid()
        elif activation == 'leaky_relu':
            activation_layer = torch.nn.LeakyReLU()

        # TODO: stage 1 (contrastive only)/ stage 2 (cross entropy only) split
        cell_type_composition_predictor = torch.nn.Sequential(
                                    torch.nn.Linear(input_dim,output_dim),torch.nn.BatchNorm1d(output_dim),
                                    activation_layer,
                                    torch.nn.Dropout(p=dropout))

        return cell_type_composition_predictor

    def forward(self, x1, x2, y):
        """
        x1: image features, shape (B, C, H, W)
        x2: spatial features, shape (B, D) where D is the number of spatial features
        y: cell type composition, shape (B, L) with L being the number of cell types
        """
        # image embedding
        with torch.no_grad():
            image_embedding = self.image_encoder(x1)
        image_embedding = self.image_projection(image_embedding)

        # spatial embedding
        spatial_embedding = self.spatial_projection(x2)

        image_embedding = torch.nn.functional.normalize(image_embedding, dim=-1)
        spatial_embedding = torch.nn.functional.normalize(spatial_embedding, dim=-1)

        # cell type prediction
        y_hat = self.celltype_prediction(image_embedding)

        # constrastive loss
        loss_contrastive = self.loss_fn(image_embedding, spatial_embedding)

        # cell type composition prediction
        loss_cross_entropy = nn.functional.cross_entropy(y_hat, y)

        # total loss
        loss = loss_contrastive + loss_cross_entropy 

        return loss, image_embedding, spatial_embedding