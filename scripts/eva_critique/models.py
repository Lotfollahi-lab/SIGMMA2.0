"""Encoder wrappers for UNI vs SIGMMA-finetuned UNI on MoNuSAC.

Both models are frozen. Each wrapper exposes:
  - forward_features_grid(x)   -> (B, D, 14, 14)  spatial feature map (last block, normed)
  - forward_features_pyramid(x, layers) -> List[(B, D, 14, 14)]  intermediate maps for skip-decoder
  - forward_cls_mean(x)        -> dict(cls=(B,D), mean=(B,D))    for linear-probe inputs
  - extract_cls_attention(x)   -> (B, H=16, 14, 14)  CLS->patch attention from last block

The SIGMMA backbone is loaded via the same `load_model` path used elsewhere in
the repo (config yaml + per-scale LoRA adapters). The active LoRA adapter is
fixed at construction time — `--scale` defaults to 0 (micro), which is the
scale most relevant to nucleus-level segmentation.

The HF identifier `AnonymousAuthor42/SIGMMA` is the anonymised release of the
same checkpoint pointed to by the config's `model.checkpoint_path` field.
"""

from __future__ import annotations

import inspect
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from ml_collections.config_dict import ConfigDict

from sigmma.utils import load_model
from sigmma.model.models_image_backbones import UNI, set_lora_adapter


# ============================================================
# Helpers
# ============================================================
def _strip_cls(tokens: torch.Tensor, num_prefix: int) -> torch.Tensor:
    """Drop the first `num_prefix` tokens (CLS + register tokens, if any)."""
    return tokens[:, num_prefix:, :]


def _grid_from_tokens(tokens: torch.Tensor, num_prefix: int) -> torch.Tensor:
    """(B, N, D) -> (B, D, sqrt(N-prefix), sqrt(N-prefix))."""
    patches = _strip_cls(tokens, num_prefix)
    b, n, d = patches.shape
    s = int(round(math.sqrt(n)))
    if s * s != n:
        raise RuntimeError(f"_grid_from_tokens: token count {n} is not a perfect square")
    return patches.transpose(1, 2).reshape(b, d, s, s).contiguous()


def _num_prefix_tokens(vit: nn.Module) -> int:
    # timm ViTs expose either `num_prefix_tokens` (newer) or set 1 implicitly (CLS).
    return int(getattr(vit, "num_prefix_tokens", 1))


# ============================================================
# Base ViT wrapper (no LoRA) — used by the UNI baseline
# ============================================================
class _ViTBackbone(nn.Module):
    """Wraps a timm ViT-L/16 trunk, frozen, with feature/attention extraction."""

    def __init__(self, vit: nn.Module, name: str):
        super().__init__()
        self.model = vit  # inner timm ViT
        self.name = name
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Cache structural info for attention reshape.
        last_block = self.model.blocks[-1]
        self.num_heads = int(last_block.attn.num_heads)
        # head_dim is usually exposed; fall back to dim // heads.
        dim = int(last_block.attn.qkv.in_features)
        self.head_dim = int(getattr(last_block.attn, "head_dim", dim // self.num_heads))
        self.scale = self.head_dim ** -0.5
        self.num_prefix = _num_prefix_tokens(self.model)

        # Detect intermediate-layers kwarg compatibility.
        sig = inspect.signature(self.model.get_intermediate_layers)
        self._inter_kwargs = dict(reshape=True, norm=True)
        if "return_prefix_tokens" in sig.parameters:
            self._inter_kwargs["return_prefix_tokens"] = False
        elif "return_class_token" in sig.parameters:
            self._inter_kwargs["return_class_token"] = False

        # Probe output dim once.
        with torch.no_grad():
            d = next(self.model.parameters()).device
            dummy = torch.randn(1, 3, 224, 224, device=d)
            feat = self.model.get_intermediate_layers(dummy, n=1, **self._inter_kwargs)
            self._feat_dim = int(feat[0].shape[1])

    @property
    def out_features(self) -> int:
        return self._feat_dim

    # ---- public API ----------------------------------------------------------
    def forward_features_grid(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,224,224) -> (B, D, 14, 14).

        Note: no `@torch.no_grad()` wrapper — gradients are governed by the
        usual `requires_grad` flags on the parameters. With the default frozen
        encoder the leaves have `requires_grad=False` so autograd skips them;
        when callers unfreeze blocks (experiment B), gradients flow through.
        """
        feats = self.model.get_intermediate_layers(x, n=1, **self._inter_kwargs)
        return feats[0]  # already (B, D, H, W) thanks to reshape=True

    @torch.no_grad()
    def forward_features_pyramid(
        self, x: torch.Tensor, layers: List[int]
    ) -> List[torch.Tensor]:
        """Tap the *output* of the requested 1-indexed ViT blocks.

        timm `get_intermediate_layers(n=List[int])` accepts an explicit list of
        block indices (0-indexed). We accept 1-indexed externally to match the
        spec ("layers {6,12,18,24}") and convert here.
        """
        idx0 = [int(i) - 1 for i in layers]
        feats = self.model.get_intermediate_layers(x, n=idx0, **self._inter_kwargs)
        return list(feats)

    @torch.no_grad()
    def forward_cls_mean(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Use forward_features (full token sequence) for clean CLS + mean access.
        tokens = self.model.forward_features(x)  # (B, N, D)
        if tokens.dim() == 4:
            # CNN-like backbone? Shouldn't happen for ViT but be safe.
            b, c, h, w = tokens.shape
            patches = tokens.flatten(2).transpose(1, 2)  # (B, HW, C)
            return {"cls": patches.mean(1), "mean": patches.mean(1)}
        cls = tokens[:, 0, :]
        mean = _strip_cls(tokens, self.num_prefix).mean(dim=1)
        return {"cls": cls, "mean": mean}

    @torch.no_grad()
    def extract_cls_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, num_heads, 14, 14) CLS->patch attention from the last block.

        Uses a forward hook on `model.blocks[-1].attn.qkv` to intercept the qkv
        output (B, N, 3*D), then computes attention manually so timm internals
        are not modified.
        """
        last_attn = self.model.blocks[-1].attn
        captured = {}

        def hook(_module, _inp, out):
            captured["qkv"] = out  # (B, N, 3*D)

        h = last_attn.qkv.register_forward_hook(hook)
        try:
            # Run a forward; we discard the output but the hook fires.
            _ = self.model.forward_features(x)
        finally:
            h.remove()

        qkv = captured["qkv"]
        b, n, three_d = qkv.shape
        d = three_d // 3
        # (B, N, 3, H, hd) -> (3, B, H, N, hd)
        qkv = qkv.reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        # CLS row, attending over patch tokens (drop prefix columns).
        cls_row = attn[:, :, 0, self.num_prefix :]  # (B, H, N - prefix)
        s = int(round(math.sqrt(cls_row.shape[-1])))
        return cls_row.reshape(b, self.num_heads, s, s).contiguous()


# ============================================================
# Public loaders
# ============================================================
def load_uni(config_path: str, device: torch.device) -> _ViTBackbone:
    """Pretrained UNI baseline (no LoRA)."""
    cfg = ConfigDict(yaml.safe_load(open(config_path, "r")))
    uni = UNI(checkpoint_path=cfg.model.pretrained_image_encoder_ckpt_path)
    return _ViTBackbone(uni.model.to(device).eval(), name="uni")


def load_sigmma(
    config_path: str,
    device: torch.device,
    scale_idx: int = 0,
) -> _ViTBackbone:
    """SIGMMA-fine-tuned UNI with the given LoRA adapter active (scale_idx).

    When the config sets ``lora.enabled: false`` (no-LoRA ablation), the
    SIGMMA UNI weights are shared across scales; ``scale_idx`` is ignored
    and the wrapper is named ``sigmma_no_lora``.
    """
    cfg = ConfigDict(yaml.safe_load(open(config_path, "r")))
    model = load_model(
        cfg,
        pretrained_image_encoder=cfg.model.pretrained_image_encoder,
        pretrained_ckpt_path=cfg.model.checkpoint_path,
        train=False,
    )
    inner = model.image_encoder.model.to(device).eval()

    lora_cfg = cfg.model.get("lora", None)
    lora_enabled = bool(lora_cfg.get("enabled", False)) if lora_cfg is not None else False
    per_scale = bool(lora_cfg.get("per_scale", False)) if lora_cfg is not None else False

    if lora_enabled and per_scale:
        # Multi-adapter case: select the named scale_N adapter.
        adapter = f"scale_{int(scale_idx)}"
        set_lora_adapter(inner, adapter)
        wrapper = _ViTBackbone(inner, name=f"sigmma_scale{scale_idx}")
        wrapper.scale_idx = int(scale_idx)
    elif lora_enabled:
        # Single-adapter case (per_scale=false): peft uses the default name
        # "default" — already active after load_model. No set_adapter needed.
        wrapper = _ViTBackbone(inner, name="sigmma_shared_lora")
        wrapper.scale_idx = -1
    else:
        wrapper = _ViTBackbone(inner, name="sigmma_no_lora")
        wrapper.scale_idx = -1
    return wrapper


def load_pair(
    config_path: str, device: torch.device, scale_idx: int = 0
) -> Dict[str, _ViTBackbone]:
    """Load both backbones for paired comparison."""
    return {
        "uni": load_uni(config_path, device),
        "sigmma": load_sigmma(config_path, device, scale_idx=scale_idx),
    }


# ============================================================
# SIGMMA per-scale region-grid encoder
# ============================================================
class _SIGMMAScaleGridBackbone(nn.Module):
    """Per-scale region patch tokens from SIGMMA's get_image_patch_tokens_per_scale.

    Taps the model AFTER the per-scale adaptive_avg_pool2d but BEFORE scale_pools
    / scale_mlps, so the wrapper sees the spatial layout the trained head queries
    over. Output (B, D, g, g) is bilinear-upsampled to (B, D, target_grid, target_grid)
    so a ConvDec_image decoder can consume it like a raw ViT patch grid.

    scale_idx -> name: 0=micro (g=4), 1=meso (g=2), 2=macro (g=1).
    At 0.5 μm/pixel and fov_size=256, micro tokens cover ~32 μm (cell-scale),
    meso ~64 μm (neighborhood), macro 128 μm (whole FOV).
    """

    SCALE_NAMES = ("micro", "meso", "macro")

    def __init__(self, sigmma_module: nn.Module, scale_idx: int, target_grid: int):
        super().__init__()
        self.sigmma = sigmma_module
        self.scale_idx = int(scale_idx)
        self.target_grid = int(target_grid)
        self.name = f"sigmma_scale_{self.SCALE_NAMES[self.scale_idx]}"
        # Probe feature dim with one dummy forward.
        with torch.no_grad():
            d = next(sigmma_module.parameters()).device
            dummy = torch.randn(1, 3, 224, 224, device=d)
            per_scale = sigmma_module.get_image_patch_tokens_per_scale(dummy)
            self._feat_dim = int(per_scale[self.scale_idx].shape[1])
        for p in self.parameters():
            p.requires_grad = False

    @property
    def out_features(self) -> int:
        return self._feat_dim

    @property
    def model(self) -> nn.Module:
        # Used by the freeze sanity check in train_decoder.py.
        return self.sigmma.image_encoder.model

    def forward_features_grid(self, x: torch.Tensor) -> torch.Tensor:
        per_scale = self.sigmma.get_image_patch_tokens_per_scale(x)
        feat = per_scale[self.scale_idx]  # (B, D, g, g)
        if feat.shape[-1] != self.target_grid:
            feat = F.interpolate(
                feat, size=(self.target_grid, self.target_grid),
                mode="bilinear", align_corners=False,
            )
        return feat

    def forward_features_pyramid(self, x, layers):
        raise NotImplementedError("SIGMMA scale-grid backbone has no pyramid layers")

    def forward_cls_mean(self, x):
        raise NotImplementedError

    def extract_cls_attention(self, x):
        raise NotImplementedError


def load_sigmma_scale_grid(
    config_path: str, device: torch.device, scale_idx: int, target_grid: int = 16,
) -> _SIGMMAScaleGridBackbone:
    """SIGMMA backbone that outputs the scale_idx-th per-scale region grid.

    target_grid defaults to 16 (UNI2-h patch grid). Bilinear-upsamples from
    g ∈ {4, 2, 1} for scale_idx ∈ {0, 1, 2} respectively.
    """
    cfg = ConfigDict(yaml.safe_load(open(config_path, "r")))
    model = load_model(
        cfg,
        pretrained_image_encoder=cfg.model.pretrained_image_encoder,
        pretrained_ckpt_path=cfg.model.checkpoint_path,
        train=False,
    ).to(device).eval()
    return _SIGMMAScaleGridBackbone(model, scale_idx=scale_idx, target_grid=target_grid)


# ============================================================
# Experiment A: SIGMMA multi-scale concat encoder
# ============================================================
class _MultiScaleSIGMMABackbone(nn.Module):
    """Concatenate per-scale SIGMMA UNI feature grids along the channel axis.

    Each SIGMMA LoRA adapter ("scale_0/1/2") gives its own (B, D, 14, 14)
    feature map. We concatenate them to (B, 3D, 14, 14). The combined map
    feeds into any decoder that accepts `in_features = 3 * D` (e.g. the
    EVA `convdec_image` head).

    Only `forward_features_grid` is implemented — pyramid skip / CLS attention
    are not meaningful across mixed-LoRA forwards, so they raise.
    """

    def __init__(self, scale_wrappers: List[_ViTBackbone]):
        super().__init__()
        assert len(scale_wrappers) == 3, "MultiScaleSIGMMA expects 3 scales"
        self.scales = nn.ModuleList(scale_wrappers)
        self.name = "sigmma_multiscale"
        self._feat_dim = sum(int(w.out_features) for w in scale_wrappers)
        for w in self.scales:
            for p in w.parameters():
                p.requires_grad = False

    @property
    def out_features(self) -> int:
        return self._feat_dim

    @property
    def model(self) -> nn.Module:
        # Some downstream code accesses encoder.model.parameters() for the
        # freeze sanity check; return the first scale's inner ViT as a stand-in.
        return self.scales[0].model

    @torch.no_grad()
    def forward_features_grid(self, x: torch.Tensor) -> torch.Tensor:
        from sigmma.model.models_image_backbones import set_lora_adapter
        feats = []
        # Each scale shares the same inner UNI ViT but with a different LoRA
        # adapter active; route per-scale by toggling the adapter and running
        # forward through that scale's wrapper.
        for i, w in enumerate(self.scales):
            adapter = getattr(w, "scale_idx", i)
            set_lora_adapter(w.model, f"scale_{int(adapter)}")
            feats.append(w.forward_features_grid(x))
        return torch.cat(feats, dim=1)

    def forward_features_pyramid(self, x, layers):
        raise NotImplementedError("MultiScaleSIGMMA does not support pyramid layers")

    def forward_cls_mean(self, x):
        raise NotImplementedError

    def extract_cls_attention(self, x):
        raise NotImplementedError


def load_sigmma_multiscale(
    config_path: str, device: torch.device,
) -> _MultiScaleSIGMMABackbone:
    """Three SIGMMA scales (micro/meso/macro) wrapped + concatenated."""
    scales = [load_sigmma(config_path, device, scale_idx=i) for i in range(3)]
    return _MultiScaleSIGMMABackbone(scales)


# ============================================================
# UNI-backbone baseline loaders (controlled-backbone EVA rows)
# ============================================================
_BASELINE_CLASS_BY_NAME = {
    "openAI_CLIP": ("sigmma.model.baselines", "openAI_CLIP"),
    "PLIP": ("sigmma.model.baselines", "PLIP"),
    "loki_OmiCLIP": ("sigmma.model.baselines", "loki_OmiCLIP"),
    "BLEEP": ("sigmma.model.baselines", "BLEEP"),
}


def _short_name(model_name: str) -> str:
    return {
        "openAI_CLIP": "clip",
        "PLIP": "plip",
        "loki_OmiCLIP": "omiclip",
        "BLEEP": "bleep",
    }[model_name]


def load_baseline_uni(
    baseline_config_path: str,
    device: torch.device,
) -> _ViTBackbone:
    """Load a UNI-backbone fine-tuned baseline as a frozen ViT encoder.

    Expects a config that already contains:
      - model.name  : one of BLEEP / openAI_CLIP / PLIP / loki_OmiCLIP
      - model.image_backbone = "uni"
      - model.uni_ckpt_path  : UNI base weights (for arch + init)
      - model.checkpoint_path: the fine-tuned state_dict to load on top

    The inner timm ViT (UNI's ``vit_large_patch16_224``) is extracted and
    wrapped in ``_ViTBackbone`` so it plugs into the existing decoder pipeline.
    """
    import importlib

    cfg = ConfigDict(yaml.safe_load(open(baseline_config_path, "r")))
    if cfg.model.get("image_backbone", "default") != "uni":
        raise ValueError(
            f"baseline_config_path {baseline_config_path} does not set "
            f"model.image_backbone=uni; got {cfg.model.get('image_backbone')!r}"
        )

    mod_name, cls_name = _BASELINE_CLASS_BY_NAME[cfg.model.name]
    BaselineCls = getattr(importlib.import_module(mod_name), cls_name)

    kwargs = dict(
        image_backbone="uni",
        uni_ckpt_path=str(cfg.model.uni_ckpt_path),
        freeze_encoders=False,
        # We don't need the loss head; pass a temperature only so the
        # ``loss`` key won't be added to forward output. We discard the model
        # after extracting the UNI encoder anyway.
        loss_temperature=None,
    )
    if cfg.model.name == "BLEEP":
        kwargs.pop("loss_temperature", None)
        kwargs.pop("freeze_encoders", None)  # BLEEP signature differs
        kwargs.update(dict(
            spot_input_dim=int(cfg.model.spot_input_dim),
            image_embedding_dim=int(cfg.model.get("image_embedding_dim", 1024)),
            projection_dim=int(cfg.model.get("projection_dim", 256)),
            spot_hidden_dim=int(cfg.model.get("spot_hidden_dim", 512)),
            temperature=float(cfg.model.get("contrastive_temperature", 1.0)),
            dropout=float(cfg.model.get("dropout", 0.1)),
            image_size=int(cfg.model.get("image_size", 224)),
            pretrained_image_encoder=False,
        ))
    elif cfg.model.name == "loki_OmiCLIP":
        kwargs.update(dict(
            pretrained_ckpt_path=str(cfg.model.pretrained_ckpt_path),
        ))
    else:  # CLIP / PLIP
        kwargs.update(dict(
            hf_model_id=cfg.model.get(
                "hf_model_id",
                "openai/clip-vit-base-patch32" if cfg.model.name == "openAI_CLIP"
                else "vinid/plip",
            ),
        ))

    model = BaselineCls(**kwargs)

    ckpt_path = cfg.model.get("checkpoint_path", None)
    if ckpt_path is None:
        raise ValueError(
            f"{baseline_config_path}: model.checkpoint_path required for EVA "
            "loading (point at the fine-tuned baseline-UNI checkpoint)."
        )
    state = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = state.get("model_state_dict", state)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_baseline_uni/{cfg.model.name}] missing: {len(missing)} keys "
              f"(first: {missing[:3]})")
    if unexpected:
        print(f"[load_baseline_uni/{cfg.model.name}] unexpected: {len(unexpected)} "
              f"keys (first: {unexpected[:3]})")

    # All four baseline classes expose the swapped image encoder at
    # `model._uni_image_encoder` (for CLIP/PLIP/OmiCLIP) or as the
    # `image_encoder` attribute (for BLEEP, which renames it). Both wrap a
    # `_UNIImageBackbone` whose inner timm ViT is `.uni.model`.
    if cfg.model.name == "BLEEP":
        uni_holder = model.image_encoder
    else:
        uni_holder = model._uni_image_encoder
    inner_vit = uni_holder.uni.model.to(device).eval()

    short = _short_name(cfg.model.name)
    wrapper = _ViTBackbone(inner_vit, name=f"{short}_uni_ft")
    return wrapper
