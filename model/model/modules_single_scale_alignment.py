import torch
import torch.nn as nn

from .losses import InfoNCE
from .layers import DINOHead
from .layers_graph import BaseGRAPHLayer, READOUTLayer


class single_img_spatial_graph(nn.Module):
    def __init__(self, config, image_encoder):
        super(single_img_spatial_graph, self).__init__()
        self.config = config
        assert config.image_encoder_output_dim == config.spatial_encoder_output_dim, "Image and spatial encoder output dimensions must match."

        self.image_encoder = image_encoder
        self.image_projection = DINOHead(
            in_dim = config.image_encoder_input_dim,
            out_dim = config.image_encoder_output_dim,
            )

        self.spatial_projection = torch.nn.ModuleList()
        self.spatial_projection.append(BaseGRAPHLayer(config.spatial_encoder_input_dim,
                                                      config.spatial_encoder_output_dim,
                                                      config.activation,
                                                      config.dropout))
        self.spatial_projection.append(READOUTLayer(config.spatial_encoder_output_dim,
                                                   config.spatial_encoder_output_dim,
                                                   config.aggregate,
                                                   config.activation,
                                                   config.dropout))

        self.celltype_prediction = self._cell_type_composition_predictor(input_dim = config.image_encoder_output_dim,
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
        image_embedding = self.image_encoder(x1)
        image_embedding = self.image_projection(image_embedding)

        # spatial embedding
        g, h_x2 = x2, x2.ndata['feat']
        for layer in self.spatial_projection:
            h_x2 = layer(g, h_x2)
        spatial_embedding = h_x2

        image_embedding_norm= torch.nn.functional.normalize(image_embedding, dim=-1)
        spatial_embedding_norm = torch.nn.functional.normalize(spatial_embedding, dim=-1)

        # cell type prediction
        y_hat = self.celltype_prediction(image_embedding)

        # constrastive loss
        loss_contrastive = self.loss_fn(image_embedding_norm, spatial_embedding_norm)

        # cell type composition prediction
        loss_cross_entropy = nn.functional.cross_entropy(y_hat, y)

        # total loss
        loss = loss_contrastive + loss_cross_entropy

        return loss, image_embedding_norm, spatial_embedding

class single_img_spatial(nn.Module):
    def __init__(self, config, image_encoder):
        super(single_img_spatial, self).__init__()
        self.config = config

        self.image_encoder = image_encoder
        self.image_projection = DINOHead(
            in_dim = config.image_encoder_input_dim,
            out_dim = config.image_encoder_output_dim,
            )

        self.spatial_projection = DINOHead(
            in_dim = config.spatial_encoder_input_dim,
            out_dim = config.spatial_encoder_output_dim,
        )

        self.celltype_prediction = self._cell_type_composition_predictor(input_dim = config.image_encoder_output_dim,
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
        image_embedding = self.image_encoder(x1)
        image_embedding = self.image_projection(image_embedding)

        # spatial embedding
        spatial_embedding = self.spatial_projection(x2)

        image_embedding_norm = torch.nn.functional.normalize(image_embedding, dim=-1)
        spatial_embedding_norm = torch.nn.functional.normalize(spatial_embedding, dim=-1)

        # cell type prediction
        y_hat = self.celltype_prediction(image_embedding)

        # constrastive loss
        loss_contrastive = self.loss_fn(image_embedding_norm, spatial_embedding_norm)

        # cell type composition prediction
        loss_cross_entropy = nn.functional.cross_entropy(y_hat, y)

        # total loss
        loss = loss_contrastive + loss_cross_entropy 

        return loss, image_embedding_norm, spatial_embedding_norm