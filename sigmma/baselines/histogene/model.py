"""HisToGene (Pang et al. 2021) — vendored model.

Reference: https://github.com/maxpmx/HisToGene/blob/main/vis_model.py

The original processes a *sequence* of spots per slide (N flattened patches +
2D coordinates). For SIGMMA fine-tuning we operate at FOV granularity: each
FOV is fed as a single-spot sequence (N=1) with the FOV center as coordinate.
The same module supports both modes — pass a (B, N, 3·P·P) patch tensor and
(B, N, 2) integer coordinate tensor.

At eval time for Path B (14×14 bin sliding window) we feed N=K² patches per
FOV in one batch with the bin (x, y) indices as coordinates, exploiting the
positional encoding to give the model spatial context across the bins of one
FOV (this is the multi-spot variant flagged in the plan).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .transformer import ViT


class HisToGene(nn.Module):
    def __init__(self, patch_size: int = 112, n_layers: int = 4, n_genes: int = 1000,
                 dim: int = 1024, dropout: float = 0.1, n_pos: int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.n_pos = n_pos
        patch_dim = 3 * patch_size * patch_size
        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.x_embed = nn.Embedding(n_pos, dim)
        self.y_embed = nn.Embedding(n_pos, dim)
        self.vit = ViT(dim=dim, depth=n_layers, heads=16, mlp_dim=2 * dim,
                       dropout=dropout, emb_dropout=dropout)
        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes),
        )

    def forward(self, patches: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """
        patches: (B, N, 3*P*P) flattened RGB patches in [0, 1] (or normalised).
        centers: (B, N, 2) long-tensor x/y coordinates in [0, n_pos).
        Returns (B, N, G) gene-expression predictions.
        """
        if centers.dtype != torch.long:
            centers = centers.long()
        centers = centers.clamp(min=0, max=self.n_pos - 1)
        h = self.patch_embedding(patches)
        h = h + self.x_embed(centers[:, :, 0]) + self.y_embed(centers[:, :, 1])
        h = self.vit(h)
        return self.gene_head(h)


def build_model(n_genes: int, patch_size: int = 112,
                pretrained_ckpt: Optional[str] = None,
                n_layers: int = 4, dim: int = 1024, n_pos: int = 64) -> HisToGene:
    model = HisToGene(patch_size=patch_size, n_layers=n_layers, n_genes=n_genes,
                      dim=dim, n_pos=n_pos)
    if pretrained_ckpt is not None:
        state = torch.load(pretrained_ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Author's checkpoints predict a different gene set — only transfer
        # the shared backbone weights (everything except gene_head + patch_embedding
        # if patch_dim differs).
        own = model.state_dict()
        transferable = {k: v for k, v in state.items()
                        if k in own and own[k].shape == v.shape and not k.startswith("gene_head")}
        missing, unexpected = model.load_state_dict(transferable, strict=False)
        print(f"[histogene] loaded {len(transferable)} tensors from {pretrained_ckpt}; "
              f"missing={len(missing)} unexpected={len(unexpected)}")
    return model
