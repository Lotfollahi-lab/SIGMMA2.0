import torch
import torch.nn as nn

from .layers_graph import GRAPHLayer, READOUTLayer
from .layers import CrossAttention

from .modules_resnet import ResNet, BasicBlock, BottleneckBlock

from .losses import InfoNCE

class hierarchical_img_spatial_graph(nn.Module):
    def __init__(self, config):
        super(hierarchical_img_spatial_graph, self).__init__()
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

        self.contrastive_loss = InfoNCE(temperature=config.temperature)

    def forward(self, x1, x2):
        """
        x1: image features, shape (B, C, H, W)
        x2: spatial features, shape (B, g) where g is a dgl graph object
        y: target label (B, num_celltypes)
        """
        embedding_pairs = []

        # Image embeddings
        x1_multiple_resolution = self.image_encoder(x1)
        # Spatial embeddings
        g, h_x2 = x2, x2.ndata['feat']
        loss_kl = []
        for i in range(self.num_resolutions):
            # spatial encoder
            ## (1) graph layer
            h_x2 = self.spatial_encoder[i*2](g, h_x2)
            ## (2) graph readout layer
            h_x2_pooled = self.spatial_encoder[i*2+1](g, h_x2)

            loss_kl.extend(self.spatial_encoder[i*2].kl_terms)

            h_x1 = x1_multiple_resolution[i]

            # normalize embeddings
            h_x1_norm = nn.functional.normalize(h_x1, dim=-1)
            h_x2_norm = nn.functional.normalize(h_x2_pooled, dim=-1)

            embedding_pairs.append((h_x1_norm, h_x2_norm))
            loss_contrastive = self.contrastive_loss(h_x1_norm, h_x2_norm)

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
