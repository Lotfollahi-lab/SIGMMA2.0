import torch
import torch.nn as nn

from .layers_graph import GRAPHLayer, READOUTLayer
from .layers import CrossAttention

from .modules_resnet import ResNet, BasicBlock, BottleneckBlock

from .losses import InfoNCE

class hierarchical_img_cellgraph(nn.Module):
    def __init__(self, config):
        super(hierarchical_img_cellgraph, self).__init__()
        self.config = config

        block, layers = BasicBlock, [4, 3, 3] # TODO: take arbitary resnet config
        # block, layers = BottleneckBlock, [4, 3, 3]
        self.image_encoder = ResNet(block, layers)

        self.num_resolutions = config.num_resolutions
        self.spatial_encoder = torch.nn.ModuleList()
        for i in range(config.num_resolutions):
            if i == 0:
                self.spatial_encoder.append(
                    GRAPHLayer(config.spatial_encoder_input_dim,
                            config.spatial_encoder_output_dim,
                            config.activation,
                            config.dropout
                            ))
                self.spatial_encoder.append(
                    READOUTLayer(config.spatial_encoder_output_dim,
                                config.spatial_encoder_output_dim,
                                config.aggregate,
                                config.activation,
                                config.dropout)
                )
            else:
                self.spatial_encoder.append(
                    GRAPHLayer(config.spatial_encoder_output_dim,
                            config.spatial_encoder_output_dim,
                            config.activation,
                            config.dropout
                            ))
                self.spatial_encoder.append(
                    READOUTLayer(config.spatial_encoder_output_dim,
                                config.spatial_encoder_output_dim,
                                config.aggregate,
                                config.activation,
                                config.dropout)
                )

        self.cross_attention = CrossAttention(config.image_encoder_output_dim, config.spatial_encoder_output_dim)
        
        self.predictors = nn.ModuleList()
        for _ in range(config.num_resolutions):
            y_per_layer = self._cell_type_composition_predictor(
            input_dim=config.image_encoder_output_dim,
            output_dim=config.num_cell_types,
            dropout=config.dropout,
            activation=config.activation
            )
            self.predictors.append(y_per_layer)

        self.contrastive_loss = InfoNCE(temperature=config.temperature)

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
 
    def forward(self, x1, x2):
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
        self_loops = (g.edges()[0] == g.edges()[1])
        loss_kl = []
        for i in range(self.num_resolutions):
            # spatial encoder
            ## (1) graph layer
            h_x2 = self.spatial_encoder[i*2](g, h_x2)
            ## (2) graph readout layer
            h_x2_pooled = self.spatial_encoder[i*2+1](g, h_x2)

            loss_kl.extend(self.spatial_encoder[i*2].kl_terms)

            h_x1 = img_embeddings[i]
            embedding_pairs.append((h_x1, h_x2_pooled))

            loss_contrastive = self.contrastive_loss(h_x1, h_x2_pooled)

            # # cross attention
            # att = self.cross_attention(h_x1, h_x2_pooled)
            # h_x1 = h_x1 + att

            # # cell type composition predictor
            # y_hat = self.predictors[i](h_x1)
            # loss_ce = nn.functional.cross_entropy(y_hat, y)
        loss_kl = torch.stack(loss_kl).mean()

        # loss: mean of KL divergence
        loss = loss_kl + loss_contrastive

        # embedding_pairs = [(u1, v1), (u2, v2), ...] where u is image embedding and v is spatial embedding
        image_embeddings, spatial_embeddings = embedding_pairs[-1]

        model_outputs = {"loss": loss,
                         "loss_kl": loss_kl,
                         "loss_contrastive": loss_contrastive,
                         "image_embedding": image_embeddings,
                         "spatial_embedding": spatial_embeddings}
        return model_outputs
