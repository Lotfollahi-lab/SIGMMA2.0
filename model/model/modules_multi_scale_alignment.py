import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize
from torch.utils.checkpoint import checkpoint

from .layers_graph import GRAPHLayer, READOUTLayer

from .losses import symmetric_InfoNCE

class hierarchical_img_spatial_graph(nn.Module):
    def __init__(self, config, image_encoder):
        super(hierarchical_img_spatial_graph, self).__init__()
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

        self.contrastive_loss = symmetric_InfoNCE(temperature=config.temperature)

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
        embeddings = []
        for im, num_patches in resized[::-1]:  # from low to high resolution
            feat = encode_in_microbatches_ckpt(self.image_encoder, im, microbatch=16, amp_dtype=torch.bfloat16)  # [B, D]
            # Mean pooling over patches
            feat = feat.view(B, num_patches, -1).mean(dim=1)
            # L2 norm
            feat = F.normalize(feat, dim=-1)
            embeddings.append(feat)
            torch.cuda.empty_cache()
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
        loss_csc_img, csc_img_terms = self._csc_loss(embedding_imgs,mode=self.csc_mode, step=self.global_step)

        loss = loss_kl + loss_contrastive + self.lambda_csc_img*loss_csc_img

        model_outputs = {"loss": loss,
                         "loss_kl": loss_kl,
                         "loss_contrastive": loss_contrastive,
                         "loss_csc_img": loss_csc_img*self.lambda_csc_img,
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

def encode_in_microbatches_ckpt(encoder, images, microbatch=16, amp_dtype=torch.bfloat16):
    outs = []
    N = images.size(0)
    i = 0
    while i < N:
        chunk = images[i:i+microbatch]
        def fn(x):
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                return encoder(x)
        feat = checkpoint(fn, chunk, use_reentrant=False)
        outs.append(feat)
        i += microbatch
    return torch.cat(outs, dim=0)