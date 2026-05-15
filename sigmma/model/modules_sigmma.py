import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize

from .models_image_backbones import set_lora_adapter

from .layers_graph import GRAPHLayer, READOUTLayer

class sigmma(nn.Module):
    def __init__(self, config, image_encoder):
        super(sigmma, self).__init__()
        self.config = config
        assert config.image_encoder_output_dim == config.spatial_encoder_output_dim, "Image and spatial encoder output dimensions must match."
        self.image_encoder = image_encoder
        if hasattr(self.image_encoder, "set_grad_checkpointing"):
            self.image_encoder.set_grad_checkpointing(True)

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

        # Scale-specific image-side heads on top of a single ViT forward.
        # ViT yields a 14x14 patch grid (for encoder_input_size=224); each scale
        # adaptively pools that grid to (2^k)x(2^k) regions (fine -> coarse),
        # then a learnable query + multihead cross-attention reduces regions to
        # one (B, D) vector, followed by a per-scale 2-layer MLP.
        D = config.image_encoder_input_dim
        D_out = config.image_encoder_output_dim
        D_hidden = int(getattr(config, "image_mlp_hidden_dim", 2 * D))
        num_heads = int(getattr(config, "image_attn_pool_heads", 8))
        self.scale_queries = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, D) * 0.02)
            for _ in range(config.num_resolutions)
        ])
        self.scale_pools = nn.ModuleList([
            nn.MultiheadAttention(D, num_heads=num_heads, batch_first=True)
            for _ in range(config.num_resolutions)
        ])
        self.scale_mlps = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(D),
                nn.Linear(D, D_hidden),
                nn.GELU(),
                nn.Linear(D_hidden, D_out),
            )
            for _ in range(config.num_resolutions)
        ])

        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))


    def _encode_image_multiscale(self, x1: torch.Tensor):
        """
        Multi-scale image encoding via a single ViT forward + per-scale heads.

        Pipeline
        --------
        x1 (B, C, H, W) -- bilinear --> (B, C, S, S), S = encoder_input_size
                       -- ViT -------> patch tokens (B, P^2, D), P = S // 16
                       -- per scale s -> adaptive_avg_pool2d to (g_s, g_s) regions
                                        g_s = 2^(num_resolutions - 1 - s)
                                        (fine s=0 -> g_max, coarse s=last -> 1)
                                        cross-attn: scale_queries[s] over regions
                                        scale_mlps[s]: D -> D_hidden -> D_out
        Returns
        -------
        list[Tensor] length num_resolutions, fine -> coarse, each (B, D_out).
        Caller is expected to L2-normalize.
        """
        assert x1.dim() == 4, "x1 must be [B, C, H, W]"
        B, _, H, W = x1.shape
        assert H == W, "x1 must be square"

        if H != self.encoder_input_size:
            x1 = F.interpolate(
                x1, size=(self.encoder_input_size, self.encoder_input_size),
                mode="bilinear", align_corners=False,
            )
        if self.use_imagenet_norm:
            x1 = (x1 - self.mean_rgb) / self.std_rgb

        # If multi-adapter LoRA is configured, pick the first adapter for the
        # single forward. Single-adapter and no-LoRA cases leave the model
        # untouched.
        adapter_names = getattr(self.image_encoder, "lora_adapter_names", None)
        if adapter_names:
            set_lora_adapter(self.image_encoder, adapter_names[0])

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=x1.is_cuda):
            tokens_all = self.image_encoder.model.forward_features(x1)  # (B, 1+P^2, D)
        patches = tokens_all[:, 1:, :].float()  # drop CLS, cast to fp32 for heads
        P = self.encoder_input_size // 16
        assert patches.shape[1] == P * P, (
            f"unexpected patch count {patches.shape[1]} for P={P}"
        )
        D = patches.shape[-1]
        patches_2d = patches.transpose(1, 2).contiguous().view(B, D, P, P)

        embeddings = []
        for s in range(self.num_resolutions):
            grid = 2 ** (self.num_resolutions - 1 - s)
            region_tokens = F.adaptive_avg_pool2d(patches_2d, grid)        # (B, D, g, g)
            region_tokens = region_tokens.flatten(2).transpose(1, 2)        # (B, g^2, D)
            q = self.scale_queries[s].expand(B, -1, -1)                     # (B, 1, D)
            pooled, _ = self.scale_pools[s](q, region_tokens, region_tokens)
            embeddings.append(self.scale_mlps[s](pooled.squeeze(1)))        # (B, D_out)
        return embeddings

    def forward(self, x1, x2):
        """
        x1: image features, shape (B, C, H, W)
        x2: spatial features, shape (B, g) where g is a dgl graph object
        y: target label (B, num_celltypes)
        """
        embedding_imgs, embedding_spatial = [], []

        # Image embeddings (already projected to D_out by scale_mlps inside).
        x1_multiple_resolution = self._encode_image_multiscale(x1)

        # Spatial embeddings
        g, h_x2 = x2, x2.ndata['feat']
        for i in range(self.num_resolutions):
            # spatial encoder
            ## (1) graph layer
            g, h_x2 = self.spatial_encoder[i*2](g, h_x2, layer_id=i)
            ## (2) graph readout layer
            h_x2_pooled = self.spatial_encoder[i*2+1](g, h_x2)

            # corresponding image embeddings
            h_x1 = x1_multiple_resolution[i]

            # normalize embeddings
            h_x1_norm = nn.functional.normalize(h_x1, dim=-1)
            h_x2_norm = nn.functional.normalize(h_x2_pooled, dim=-1)

            embedding_imgs.append(h_x1_norm)
            embedding_spatial.append(h_x2_norm)

        self.global_step += 1

        model_outputs = {"image_embedding": embedding_imgs,
                         "spatial_embedding": embedding_spatial}
        return model_outputs

    def get_image_features(self, x1):
        with torch.no_grad():
            transform = Resize((self.encoder_input_size, self.encoder_input_size))
            x1 = transform(x1)
            x1_multiple_resolution = self._encode_image_multiscale(x1)
            image_embedding = x1_multiple_resolution[-1]
            image_embedding_norm = torch.nn.functional.normalize(image_embedding, dim=-1)
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
