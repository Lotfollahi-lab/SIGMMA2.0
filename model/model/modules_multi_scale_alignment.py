import torch
import torch.nn as nn

from .layers_graph import GRAPHLayer, READOUTLayer
from .layers import CrossAttention

from .modules_resnet import ResNet, BasicBlock, BottleneckBlock

from .losses import InfoNCE

class hierarchical_img_spatial_graph(nn.Module):
    def __init__(self, config, image_encoder):
        super(hierarchical_img_spatial_graph, self).__init__()
        self.config = config
        assert config.image_encoder_output_dim == config.spatial_encoder_output_dim, "Image and spatial encoder output dimensions must match."
        self.image_encoder = image_encoder
        self.image_projection = nn.Linear(config.image_encoder_input_dim, config.image_encoder_output_dim)

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

        self.lambda_csc_img = getattr(config, "lambda_csc_img", 1.0)
        self.csc_mode = getattr(config, "csc_mode", "both")  # "both" or "alt"
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))


    def _csc_loss(self, z_list, mode="both", step: int = 0):
        """
        Cross-Scale Consistency Loss
        =============================================
        [Input]
        z_list [ [B,D] for each scale ]: list of embeddings at different scales
        mode (str): "both"-both-way optimization / "alt"-alternate optimization
        step (int): current training step, used for "alt" mode
        [Output]
        loss (torch.Tensor): scalar loss
        per_pair (dict): dictionary of loss values for each pair of scales
        """
        if len(z_list) <= 1:
            return z_list[0].new_tensor(0.0), {}

        pairs = list(itertools.combinations(range(len(z_list)), 2))
        losses = []
        per_pair = {}
        for (a, b) in pairs:
            za, zb = z_list[a], z_list[b]  # [B,D]
            if mode == "both":
                l1 = 1.0 - F.cosine_similarity(za, zb.detach(), dim=-1)
                l2 = 1.0 - F.cosine_similarity(zb, za.detach(), dim=-1)
                l = 0.5 * (l1 + l2)
            elif mode == "alt":
                if (step % 2) == 0:
                    l = 1.0 - F.cosine_similarity(za, zb.detach(), dim=-1)
                else:
                    l = 1.0 - F.cosine_similarity(zb, za.detach(), dim=-1)
            else:
                raise ValueError("csc_mode must be 'both' or 'alt'")
            losses.append(l.mean())
            per_pair[f"{a}-{b}"] = float(l.mean().detach())
        loss = torch.stack(losses).mean()
        return loss, per_pair

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
            g, h_x2 = self.spatial_encoder[i*2](g, h_x2, layer_id=i)
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
        loss_contrastive = torch.stack(loss_contrastive).mean()

        self.global_step += 1
        loss_csc_img, csc_img_terms = self._csc_loss(embedding_imgs,mode=self.csc_mode, step=self.global_step)

        loss = loss_kl + loss_contrastive + self.lambda_csc_img*loss_csc_img

        # embedding_pairs = [(u1, v1), (u2, v2), ...] where u is image embedding and v is spatial embedding
        image_embeddings, spatial_embeddings = embedding_pairs[-1]

        model_outputs = {"loss": loss,
                         "loss_kl": loss_kl,
                         "loss_contrastive": loss_contrastive,
                         "loss_csc_img": loss_csc_img,
                         "image_embedding": image_embeddings,
                         "spatial_embedding": spatial_embeddings}
        return model_outputs
