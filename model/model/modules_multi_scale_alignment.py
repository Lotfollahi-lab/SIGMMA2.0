from re import I
import torch
import torch.nn as nn

from .layers_graph import GRAPHLayer, READOUTLayer
from .layers import CrossAttention

from .modules_resnet import ResNet, ResidualBlock

from .losses import InfoNCE

class hierarchical_img_cellgraph(nn.Module):
    def __init__(self, config):
        super(hierarchical_img_cellgraph, self).__init__()
        self.config = config

        block, layers = ResidualBlock, [4, 3, 3] 
        self.image_encoder = ResNet(block, layers)

        self.spatial_encoder = torch.nn.ModuleList()
        for _ in range(config.num_resolutions):
            gnn_layer = torch.nn.Sequential(
                GRAPHLayer(config.spatial_encoder_input_dim,
                           config.spatial_encoder_output_dim,
                           config.activation,
                           config.dropout
                           ),
                READOUTLayer(config.spatial_encoder_input_dim,
                             config.spatial_encoder_output_dim,
                             config.aggregate,
                             config.activation,
                             config.dropout)
            )
            self.spatial_encoder.append(gnn_layer)

        self.cross_attention = CrossAttention(config.spatial_encoder_output_dim, config.spatial_encoder_output_dim)
        
        self.predictors = []
        for _ in range(config.num_resolutions):
            y_per_layer = self._cell_type_composition_predictor(
            input_dim=config.image_encoder_output_dim,
            output_dim=config.num_cell_types,
            dropout=config.dropout,
            activation=config.activation
            )
            self.predictors.append(y_per_layer)

        self.contrastive_loss = InfoNCE(temperature=config.infoNCE_temperature)

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
     
        cell_type_composition_predictor = torch.nn.Sequential(
                                    torch.nn.Linear(input_dim,output_dim),torch.nn.BatchNorm1d(output_dim),
                                    activation_layer,
                                    torch.nn.Dropout(p=dropout))
        
        return cell_type_composition_predictor
 
    def forward(self, x1, x2, y):
        """
        x1: image features, shape (B, C, H, W)
        x2: spatial features, shape (B, g) where g is a dgl graph object
        y: target label (B, num_celltypes)
        """
        embedding_pairs = []

        # Imaage embeddings
        img_embeddings = self.image_encoder(x1)

        # Spatial embeddings
        g, h_x2 = x2, x2.ndata['feat']
        for i in range(self.image_encoder):

            # spatial encoder
            h_x2 = self.spatial_encoder[i](h_x2)

            breakpoint()
            loss_kl = graph_layer.kl_terms

            h_x1 = img_embeddings[i]
            embedding_pairs.append((h_x1, h_x2))

            loss_contrastive = self.contrastive_loss(h_x1, h_x2)

            # cross attention #TODO
            att = self.cross_attention(h_x1, h_x2)
            h_x1 = h_x1 + att

            # cell type composition predictor
            y_hat = self.predictors[i](h_x1)
            loss_mse = nn.functional.cross_entropy(y_hat, y)

        # loss: mean of KL divergence
        loss = loss_kl + loss_contrastive + loss_mse

        return loss, embedding_pairs
