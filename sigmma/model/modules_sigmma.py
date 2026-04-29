import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize
from torch.utils.checkpoint import checkpoint

from .models_image_backbones import set_lora_adapter

from .layers_graph import GRAPHLayer, READOUTLayer

from .losses import symmetric_InfoNCE

class sigmma(nn.Module):
    def __init__(self, config, image_encoder):
        super(sigmma, self).__init__()
        self.config = config
        assert config.image_encoder_output_dim == config.spatial_encoder_output_dim, "Image and spatial encoder output dimensions must match."
        self.image_encoder = image_encoder
        if hasattr(self.image_encoder, "set_grad_checkpointing"):
            self.image_encoder.set_grad_checkpointing(True)
        self.image_projection_heads = torch.nn.ModuleList([nn.Linear(config.image_encoder_input_dim, config.image_encoder_output_dim) for _ in range(config.num_resolutions)])

        self.encoder_input_size = getattr(config, "encoder_input_size", 224)
        self.use_imagenet_norm = getattr(config, "use_imagenet_norm", False)
        if self.use_imagenet_norm:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
            self.register_buffer("mean_rgb", mean)
            self.register_buffer("std_rgb", std)

        self.num_resolutions = config.num_resolutions
        candidate_top_k = getattr(config, "candidate_top_k", 8)
        candidate_sim_threshold = getattr(config, "candidate_sim_threshold", 0.4)
        self.spatial_encoder = torch.nn.ModuleList()
        for i in range(config.num_resolutions):
            if i == 0:
                self.spatial_encoder.append(
                    GRAPHLayer(config.spatial_encoder_input_dim,
                            config.spatial_encoder_output_dim,
                            config.activation,
                            config.dropout,
                            threshold=config.edge_threshold,
                            temperature=config.gumbel_temperature,
                            candidate_top_k=candidate_top_k,
                            candidate_sim_threshold=candidate_sim_threshold,
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
                            config.dropout,
                            threshold=config.edge_threshold,
                            temperature=config.gumbel_temperature,
                            candidate_top_k=candidate_top_k,
                            candidate_sim_threshold=candidate_sim_threshold,
                            ))
                self.spatial_encoder.append(
                    READOUTLayer(config.spatial_encoder_output_dim,
                                config.spatial_encoder_output_dim,
                                config.aggregate,
                                config.activation,
                                config.dropout)
                )

        self.contrastive_loss = symmetric_InfoNCE(temperature=config.contrastive_temperature)

        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))


    def _encode_image_multiscale(self, x1: torch.Tensor):
        """
        Multi-scale image encoding
        =============================================
        [Input]
        x1 (tensor): Original image features, shape [B, C, H, W]
        [Output]
        embeddings (list of tensor [B, D]): list of image embeddings at different scales
        """
        assert x1.dim() == 4, "x1 must be [B, C, H, W]"
        B, C, H, W = x1.shape
        assert H == W, "x1 must be square"

        # Patchfying
        crops = []
        # From coarse to fine grid (2x2, 4x4, ...)
        for s in range(self.num_resolutions):
            grid = 2 ** s
            step = H // grid
            batch_crops = []
            for hi, wi in itertools.product(range(grid), range(grid)):
                top = hi * step
                left = wi * step
                patch = x1[:, :, top:top+step, left:left+step] # [B, C, step, step]
                batch_crops.append(patch)
            patches = torch.stack(batch_crops, dim=1)  # [B, num_patches, C, step, step]
            crops.append(patches)

        # Resize & Normalize
        resized = []
        for im in crops:
            B, num_patches, C, h, w = im.shape
            im = im.view(B * num_patches, C, h, w)
            if h != self.encoder_input_size:
                im = F.interpolate(im, size=(self.encoder_input_size, self.encoder_input_size),
                                   mode="bilinear", align_corners=False)
            if self.use_imagenet_norm:
                im = (im - self.mean_rgb) / self.std_rgb
            resized.append((im, num_patches))  # [B*num_patches, C, enc_in, enc_in]

        # Make embeddings
        adapter_names = getattr(self.image_encoder, "lora_adapter_names", None)
        embeddings = []
        for idx, (im, num_patches) in enumerate(resized[::-1]):  # from low to high resolution
            pre_hook = None
            if adapter_names is not None:
                name = adapter_names[idx]
                pre_hook = lambda _name=name: set_lora_adapter(self.image_encoder, _name)
            feat = encode_in_microbatches_ckpt(self.image_encoder, im, microbatch=64, amp_dtype=torch.bfloat16, pre_hook=pre_hook)  # [B, D]
            # Mean pooling over patches
            feat = feat.view(B, num_patches, -1).mean(dim=1)
            # L2 norm
            feat = F.normalize(feat, dim=-1)
            embeddings.append(feat)
        return embeddings

    def forward(self, x1, x2):
        """
        x1: image features, shape (B, C, H, W)
        x2: spatial features, shape (B, g) where g is a dgl graph object
        y: target label (B, num_celltypes)
        """
        embedding_imgs, embedding_spatial = [], []

        # Image embeddings
        x1_multiple_resolution = self._encode_image_multiscale(x1)
        x1_multiple_resolution = [self.image_projection_heads[i](feat) for i, feat in enumerate(x1_multiple_resolution)]

        # Spatial embeddings
        g, h_x2 = x2, x2.ndata['feat']
        loss_contrastive, loss_kl = [], []
        for i in range(self.num_resolutions):
            # spatial encoder
            ## (1) graph layer
            g, h_x2 = self.spatial_encoder[i*2](g, h_x2, layer_id=i)
            ## (2) graph readout layer
            h_x2_pooled = self.spatial_encoder[i*2+1](g, h_x2)

            loss_kl.extend(self.spatial_encoder[i*2].kl_terms)

            # corresponding image embeddings
            h_x1 = x1_multiple_resolution[i]

            # normalize embeddings
            h_x1_norm = nn.functional.normalize(h_x1, dim=-1)
            h_x2_norm = nn.functional.normalize(h_x2_pooled, dim=-1)

            embedding_imgs.append(h_x1_norm)
            embedding_spatial.append(h_x2_norm)

            loss_contrastive.append(self.contrastive_loss(h_x1_norm, h_x2_norm))

        loss_kl = torch.stack(loss_kl).mean()
        loss_contrastive = torch.stack(loss_contrastive).mean()

        self.global_step += 1

        loss = loss_contrastive

        model_outputs = {"loss": loss,
                         "image_embedding": embedding_imgs,
                         "spatial_embedding": embedding_spatial}
        return model_outputs

    def get_image_features(self, x1):
        with torch.no_grad():
            transform = Resize((224, 224))
            x1 = transform(x1)
            x1_multiple_resolution = self._encode_image_multiscale(x1)
            x1_multiple_resolution = [self.image_projection_heads[i](feat) for i, feat in enumerate(x1_multiple_resolution)]
            image_embedding = x1_multiple_resolution[-1]
            image_embedding_norm= torch.nn.functional.normalize(image_embedding, dim=-1)
        return image_embedding_norm

    def get_spatial_features(self, x2):
        with torch.no_grad():
            g, h_x2 = x2, x2.ndata['feat']
            for i in range(self.num_resolutions):
                # spatial encoder
                ## (1) graph layer
                g, h_x2 = self.spatial_encoder[i*2](g, h_x2, layer_id=i)
                ## (2) graph readout layer
                h_x2_pooled = self.spatial_encoder[i*2+1](g, h_x2)
            spatial_embedding = h_x2_pooled
            spatial_embedding_norm = torch.nn.functional.normalize(spatial_embedding, dim=-1)
        return spatial_embedding_norm

def encode_in_microbatches_ckpt(encoder, images, microbatch=16, amp_dtype=torch.bfloat16, pre_hook=None):
    # Outer activation checkpointing was redundant with timm's built-in
    # per-block grad checkpointing (set via set_grad_checkpointing in
    # sigmma.__init__). Only the inner one is kept; we still chunk the
    # forward into microbatches so peak activation memory stays bounded.
    outs = []
    N = images.size(0)
    i = 0
    while i < N:
        chunk = images[i:i+microbatch]
        if pre_hook is not None:
            pre_hook()
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
            feat = encoder(chunk)
        outs.append(feat)
        i += microbatch
    return torch.cat(outs, dim=0)