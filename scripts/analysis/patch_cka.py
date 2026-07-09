"""Test 2 — layer-wise patch-feature CKA between frozen UNI and each FT model.

For ViT blocks {6, 12, 18, 24} compute linear CKA between frozen UNI patch
features and each of (SIGMMA-LoRA-UNI@meso, BLEEP-UNI) on two domains:
  - MoNuSAC test tiles   (matches EVA-critique evaluation domain)
  - Sanger Xenium FOVs   (matches retrieval training domain)

Decision rules (see plans/floofy-enchanting-avalanche.md):
  CKA(SIGMMA, frozen UNI) at block 24 expected ≥ 0.95 on both domains
  CKA(BLEEP-UNI, frozen UNI) at block 24 expected ≤ 0.85 on at least one
  Gap = CKA_SIGMMA - CKA_BLEEP ≥ 0.10 at block 24 supports localization.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
import zarr
from ml_collections.config_dict import ConfigDict

# Reuse existing eva_critique loaders so we share the exact same model wiring
# the EVA-critique decoder benchmark uses.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eva_critique.models import load_uni, load_baseline_uni, load_sigmma
from eva_critique.data import iter_monusac_test
from smoke_lora_divergence import _sample_fovs
from sigmma.model.models_image_backbones import set_lora_adapter


LAYERS = [6, 12, 18, 24]


# ---------- linear CKA ----------------------------------------------------

def _center(x: torch.Tensor) -> torch.Tensor:
    """Center columns: subtract the per-feature mean across the N axis.

    x: (N, D) returns (N, D) with each column zero-mean.
    """
    return x - x.mean(dim=0, keepdim=True)


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA via the dual identity:
        ||X_c^T Y_c||_F^2 / (||X_c^T X_c||_F * ||Y_c^T Y_c||_F)
    X, Y: (N, D) float32, will be centered along N inside this function.
    """
    Xc = _center(X.float())
    Yc = _center(Y.float())
    xtY = Xc.t() @ Yc
    xtX = Xc.t() @ Xc
    ytY = Yc.t() @ Yc
    num = (xtY ** 2).sum()
    den = torch.sqrt(((xtX ** 2).sum()) * ((ytY ** 2).sum()) + 1e-12)
    return float(num / den)


# ---------- patch extraction ---------------------------------------------

@torch.no_grad()
def extract_patch_features(
    backbone, images: torch.Tensor, layers: List[int], batch_size: int = 8
) -> List[torch.Tensor]:
    """Run patches through ``backbone.forward_features_pyramid`` block-by-block.

    images: (N, 3, 224, 224) float in [0,1] (will be ImageNet-normalised by
            the timm preprocessor inside the ViT — backbones in this repo
            expect the *raw* tensor only at the eva_critique loaders'
            transform path; we pass already-normalised images here).
    Returns list of (N, D, H, W) tensors, one per requested layer.
    """
    device = next(backbone.model.parameters()).device
    out: List[List[torch.Tensor]] = [[] for _ in layers]
    for i in range(0, images.shape[0], batch_size):
        batch = images[i : i + batch_size].to(device, non_blocking=True)
        feats = backbone.forward_features_pyramid(batch, layers)
        for k, f in enumerate(feats):
            out[k].append(f.cpu())
    return [torch.cat(parts, dim=0) for parts in out]


def flatten_patches(feat: torch.Tensor) -> torch.Tensor:
    """(N, D, H, W) -> (N*H*W, D)."""
    N, D, H, W = feat.shape
    return feat.permute(0, 2, 3, 1).reshape(N * H * W, D)


# ---------- domain loaders -----------------------------------------------

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _normalise(x: torch.Tensor) -> torch.Tensor:
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def load_monusac_tiles(data_root: str, max_tiles: int = 64) -> torch.Tensor:
    """Load up to ``max_tiles`` MoNuSAC test tiles as (N, 3, 224, 224)
    ImageNet-normalised tensors. Reuses the existing test-tile iterator that
    already resizes to 224x224 and applies the eval transform.
    """
    tiles: List[torch.Tensor] = []
    monusac_root = os.path.join(data_root, "monusac")
    for tile in iter_monusac_test(monusac_root, split="test"):
        tiles.append(tile.image_norm)  # already normalised, (3, 224, 224)
        if len(tiles) >= max_tiles:
            break
    if not tiles:
        raise RuntimeError(f"No MoNuSAC tiles found at {monusac_root}")
    return torch.stack(tiles, dim=0)


def load_sanger_fovs(
    manifest_path: str,
    n_slides: int = 5,
    fovs_per_slide: int = 10,
    fov_size: int = 224,
    seed: int = 42,
) -> torch.Tensor:
    """Sample non-empty 224x224 FOVs from Sanger Xenium slides. Returns
    (N, 3, 224, 224) ImageNet-normalised tensors.
    """
    df = pd.read_csv(manifest_path)
    if "hematoxylin_eosin_image" not in df.columns:
        raise ValueError(
            f"Expected 'hematoxylin_eosin_image' column in {manifest_path}"
        )
    rng = np.random.default_rng(seed)
    slide_paths = df["hematoxylin_eosin_image"].dropna().tolist()
    # Use first ``n_slides`` after shuffling for stability.
    rng.shuffle(slide_paths)
    slide_paths = slide_paths[:n_slides]

    tiles: List[torch.Tensor] = []
    for sp in slide_paths:
        try:
            image_array = zarr.open(sp, mode="r")
        except Exception as e:
            print(f"[warn] could not open {sp}: {e}")
            continue
        fovs = _sample_fovs(image_array, fovs_per_slide, fov_size, seed=seed)
        for _, _, tile_np in fovs:
            t = torch.from_numpy(tile_np).float() / 255.0  # (C, fov, fov)
            tiles.append(t)
    if not tiles:
        raise RuntimeError("No Sanger FOVs collected")
    batch = torch.stack(tiles, dim=0)
    # ImageNet-normalise (same as eva_critique transforms).
    return _normalise(batch)


# ---------- main ---------------------------------------------------------

SCALE_NAMES = {0: "sigmma_micro", 1: "sigmma_meso", 2: "sigmma_macro"}


def encoders_to_compare(
    sigmma_config: str,
    bleep_uni_config: str,
    device: torch.device,
    sigmma_scales: List[int] = [0, 1, 2],
):
    """Return dict {name: _ViTBackbone}; frozen UNI first (the reference).

    SIGMMA scales share the same UNI backbone with different LoRA adapters.
    We load SIGMMA once and clone the wrapper per scale; ``set_lora_adapter``
    will be invoked per-scale right before each feature-extraction loop.
    """
    uni = load_uni(sigmma_config, device)
    bleep = load_baseline_uni(bleep_uni_config, device)
    encoders = {"uni_frozen": uni, "bleep_uni": bleep}

    if sigmma_scales:
        sigmma = load_sigmma(sigmma_config, device, scale_idx=sigmma_scales[0])
        encoders["_sigmma_shared"] = sigmma  # sentinel; we switch adapters
        encoders["_sigmma_scales"] = sigmma_scales
    return encoders


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sigmma_config",
        default="configs/in-house_retrieval_sigmma_meso.yaml",
        help="Used by load_uni AND load_sigmma — needs "
        "pretrained_image_encoder_ckpt_path + lora config",
    )
    ap.add_argument(
        "--bleep_uni_config",
        default="configs/in-house_retrieval_bleep_uni_finetuned.yaml",
        help="Points at the BLEEP-UNI 21523 ckpt",
    )
    ap.add_argument(
        "--monusac_root",
        default="data",
        help="Parent dir; expects <root>/monusac/ inside",
    )
    ap.add_argument(
        "--sanger_manifest",
        default="/lustre/scratch126/cellgen/lotfollahi/dj16/data/"
        "sanger_in_house_xenium/sanger_MANIFEST_val.csv",
    )
    ap.add_argument("--n_monusac", type=int, default=64)
    ap.add_argument("--n_sanger_slides", type=int, default=5)
    ap.add_argument("--n_fovs_per_slide", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument(
        "--out_dir", default="outputs/analysis/patch_cka"
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[load] encoders")
    encoders = encoders_to_compare(args.sigmma_config, args.bleep_uni_config, device)

    rows: List[dict] = []
    for domain_name, loader_fn in [
        (
            "monusac",
            lambda: load_monusac_tiles(args.monusac_root, max_tiles=args.n_monusac),
        ),
        (
            "sanger",
            lambda: load_sanger_fovs(
                args.sanger_manifest,
                n_slides=args.n_sanger_slides,
                fovs_per_slide=args.n_fovs_per_slide,
            ),
        ),
    ]:
        print(f"\n[load] domain={domain_name}")
        images = loader_fn()
        print(f"[ok] {domain_name}: {images.shape}")

        # Extract patches per encoder, per layer.
        feats_per_encoder = {}
        sigmma_shared = encoders.get("_sigmma_shared", None)
        sigmma_scales = encoders.get("_sigmma_scales", [])

        for name, backbone in encoders.items():
            if name.startswith("_sigmma_"):  # handled below
                continue
            print(f"  [extract] {name}")
            feats_per_encoder[name] = extract_patch_features(
                backbone, images, LAYERS, batch_size=args.batch_size
            )

        # SIGMMA: switch LoRA adapter per scale, then extract.
        for s in sigmma_scales:
            adapter_name = f"scale_{int(s)}"
            set_lora_adapter(sigmma_shared.model, adapter_name)
            tag = SCALE_NAMES[int(s)]
            print(f"  [extract] {tag} (adapter={adapter_name})")
            feats_per_encoder[tag] = extract_patch_features(
                sigmma_shared, images, LAYERS, batch_size=args.batch_size
            )

        # CKA(frozen UNI, *) per layer.
        ref_feats = feats_per_encoder["uni_frozen"]
        for li, layer in enumerate(LAYERS):
            X = flatten_patches(ref_feats[li])
            for enc_name, enc_feats in feats_per_encoder.items():
                if enc_name == "uni_frozen":
                    continue
                Y = flatten_patches(enc_feats[li])
                cka = linear_cka(X, Y)
                row = dict(
                    encoder=enc_name,
                    reference="uni_frozen",
                    domain=domain_name,
                    layer=layer,
                    n_images=int(images.shape[0]),
                    n_patches=int(X.shape[0]),
                    cka=cka,
                )
                print(f"  cka [{domain_name}|{enc_name}|L{layer}] = {cka:.4f}")
                rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, "cka_by_layer.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[done] wrote {out_csv}")
    print(df.pivot_table(index=["domain", "layer"], columns="encoder", values="cka").to_string())


if __name__ == "__main__":
    main()
