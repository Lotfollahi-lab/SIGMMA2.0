"""Load the pretrained Phoenix flow-matching model for zero-shot inference.

Phoenix is a pan-cancer latent flow-matching generative model (peng-lab/phoenix)
that predicts spatial gene expression from H&E. We use it **zero-shot**: load the
released Nest/multi/cell/20x weights and sample at each FOV bin centre.

We deliberately import from ``phoenix.models.flow_simple`` (pure-PyTorch RMSNorm /
SwiGLU / ``F.scaled_dot_product_attention``) rather than ``flow_llama3`` (which
needs apex / flash_attn / xformers). ``flow_simple`` loads the checkpoint
``strict=True`` — verified 763/763 tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import timm
import torch

# Default asset locations (see assets/phoenix/README.md).
ASSETS = Path("assets/phoenix")
DEFAULT_WEIGHTS = ASSETS / "flow_model.pth"
DEFAULT_GENES = ASSETS / "xenium_human_multi.npy"
DEFAULT_STATS = ASSETS / "stats_table.npz"

# Config of the released Nest checkpoint (peng-lab/phoenix README).
_VISION_NAME = "vit_giant_patch14_reg4_dinov2"
_FLOW_CFG = dict(
    d_genes=1, d_image=1536, d_model=512, d_cross=512,
    n_heads=8, n_layers=8, qkv_bias=False, ffn_bias=False,
    ffn_mult=4, attn_drop=0.0, proj_drop=0.0,
    n_classes=0, cls_drop=0.1, checkpoint=False,
)


@dataclass
class PhoenixModel:
    """Bundle of everything the adapter needs."""
    model: torch.nn.Module          # FlowTransformerModel (wraps DINOv2-giant)
    gene_list: list[str]            # 377 Phoenix output genes
    mean: np.ndarray                # (377,) de-normalisation mean
    std: np.ndarray                 # (377,) de-normalisation std


def build_model(
    weights_path: Optional[str] = None,
    genes_path: Optional[str] = None,
    stats_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> PhoenixModel:
    """Construct the vision encoder + flow transformer and load pretrained weights."""
    from phoenix.models.flow_simple import FlowTransformerConfig, FlowTransformerModel

    weights_path = str(weights_path or DEFAULT_WEIGHTS)
    genes_path = str(genes_path or DEFAULT_GENES)
    stats_path = str(stats_path or DEFAULT_STATS)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vision = timm.create_model(
        _VISION_NAME, pretrained=False, img_size=224, num_classes=0,
        global_pool="token", init_values=1e-5, dynamic_img_size=False,
    )
    model = FlowTransformerModel(FlowTransformerConfig(**_FLOW_CFG), vision_model=vision)

    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    gene_list = [str(g) for g in np.load(genes_path, allow_pickle=True)]
    stats = np.load(stats_path, allow_pickle=True)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    assert len(gene_list) == mean.shape[0] == std.shape[0], (
        f"gene/stats mismatch: {len(gene_list)}/{mean.shape}/{std.shape}"
    )
    return PhoenixModel(model=model, gene_list=gene_list, mean=mean, std=std)
